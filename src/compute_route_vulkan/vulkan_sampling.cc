#include "vulkan_program_internal.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <new>

#include "hyle/compute/sampling_ops.h"
#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {
namespace {
bool device_limits(const Device &device, std::uint64_t groups,
                     std::uint64_t scratch_bytes) noexcept {
  const auto &limits = device.properties.limits;
  return limits.maxComputeWorkGroupInvocations >= 128 &&
         limits.maxComputeWorkGroupSize[0] >= 128 &&
         limits.maxComputeSharedMemorySize >= 128 * 12 &&
         limits.maxPerStageDescriptorStorageBuffers >= 6 &&
         limits.maxDescriptorSetStorageBuffers >= 6 &&
         limits.maxComputeWorkGroupCount[0] != 0 &&
         groups <= std::uint64_t(limits.maxComputeWorkGroupCount[0]) *
                     limits.maxComputeWorkGroupCount[1] &&
         scratch_bytes <= limits.maxStorageBufferRange;
}
} // namespace

bool single_topk_operation(const ProgramState &program,
                             const PlanStep &step) noexcept {
  const auto *topk = step.operation.get_if<TopK>();
  if (!topk || topk->count != 1 || step.inputs.size() != 1 || step.outputs.size() != 2)
    return false;
  const auto &input = program.values[step.inputs[0]].description;
  if (!input.is_plain() || !input.is_dense() || input.rank() == 0 ||
      input.data_type() != DataType::kFloat32)
    return false;
  const auto axis = topk->axis < 0 ? topk->axis + static_cast<int>(input.rank()) : topk->axis;
  if (axis != static_cast<int>(input.rank()) - 1) return false;
  const auto width = input.dimensions().back();
  if (width <= 0 || width > INT32_MAX) return false;
  const auto count = input.element_count();
  const auto bytes = input.required_bytes();
  if (!count || !bytes || *bytes > program.runtime->device->properties.limits.maxStorageBufferRange)
    return false;
  const auto rows = *count / static_cast<std::size_t>(width);
  const auto chunks = (std::uint64_t(width) + 4095) / 4096;
  return device_limits(*program.runtime->device, rows * chunks, rows * chunks * 8);
}

Status lower_sampling(ProgramState &program, const PlanStep &step) noexcept try {
  const auto bits = step.operation.custom_as<sampling_ops::RandomBits>();
  const auto categorical = step.operation.custom_as<sampling_ops::Categorical>();
  const auto *topk = step.operation.get_if<TopK>();
  if (!bits && !categorical && !single_topk_operation(program, step))
    return Status::Unimplemented("Vulkan sampling numerical domain is not bound");
  const auto input = program.values[step.inputs[0]].description;
  const auto &device = *program.runtime->device;
  for (auto id : step.inputs)
    if (*program.values[id].description.required_bytes() > device.properties.limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan sampling input buffer range exceeded");
  for (auto id : step.outputs)
    if (*program.values[id].description.required_bytes() > device.properties.limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan sampling output buffer range exceeded");
  const auto row_count = topk ? *input.element_count() / input.dimensions().back()
                              : static_cast<std::uint64_t>(input.dimensions()[0]);
  const auto width = bits ? bits->word_count : static_cast<std::uint32_t>(input.dimensions().back());
  if (row_count == 0 || width == 0) return Status::Ok();
  if (row_count > INT32_MAX)
    return Status::Unimplemented("Vulkan sampling row index capacity exceeded");
  const auto rows = static_cast<std::uint32_t>(row_count);
  const auto chunks = bits ? 1u : (width + 4095u) / 4096u;
  const auto partials = std::uint64_t(rows) * chunks;
  const auto scratch_words = bits ? 0 : topk ? partials * 2 : partials * 4 + rows;
  const auto groups = bits ? (std::uint64_t(rows) * ((std::uint64_t(width) + 3) / 4) + 127) / 128
                           : partials;
  if (!device_limits(device, groups, scratch_words * 4))
    return Status::Unimplemented("Vulkan sampling dispatch/scratch exceeds native limits");
  while (program.aliases.size() < program.values.size()) {
    const auto id = static_cast<std::uint32_t>(program.aliases.size());
    program.aliases.push_back({id, 0});
  }
  const auto append = [&](TensorDesc description) {
    const auto id = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({description, {}});
    program.aliases.push_back({id, 0});
    return id;
  };
  auto scratch = step.inputs[0]; // Raw bits need no temporary value.
  if (scratch_words) {
    HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kUInt32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(scratch_words)}));
    scratch = append(description);
  }
  auto status = scratch;
  if (categorical) {
    if (!program.status_value) {
      HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kInt32, {}));
      program.status_value = append(description);
    }
    status = *program.status_value;
  }
  const auto type = input.data_type() == DataType::kFloat16 ? 1u
                    : input.data_type() == DataType::kBFloat16 ? 2u : 0u;
  const std::array<std::uint32_t, 6> values{step.inputs[0],
      categorical ? step.inputs[1] : step.inputs[0], status, step.outputs[0],
      scratch, step.outputs.size() == 2 ? step.outputs[1] : step.outputs[0]};
  const auto dispatch = [&](std::uint32_t mode, std::uint32_t work) -> Status {
    const std::array<SpecializationConstant, 7> constants{
        SpecializationConstant{0, mode}, {1, type}, {2, rows}, {3, width},
        {4, chunks}, {5, categorical && categorical->return_log_probability ? 1u : 0u},
        {6, topk && !topk->largest ? 0u : 1u}};
    HYLE_ASSIGN_OR_RETURN(auto pipeline,
        create_pipeline(program.runtime->device, {shaders::sampling(), 6, 12, constants}));
    Dispatch stage;
    stage.pipeline = std::move(pipeline);
    stage.values.assign(values.begin(), values.end());
    stage.push.resize(12);
    stage.tensor_byte_origins = true;
    stage.checked_status = bool(categorical);
    const auto x = (std::min)(work, device.properties.limits.maxComputeWorkGroupCount[0]);
    stage.groups = {x, (work + x - 1) / x, 1};
    program.dispatches.push_back(std::move(stage));
    return Status::Ok();
  };
  if (bits) return dispatch(0, static_cast<std::uint32_t>(groups));
  if (topk) {
    HYLE_RETURN_IF_ERROR(dispatch(1, static_cast<std::uint32_t>(partials)));
    return dispatch(2, rows);
  }
  HYLE_RETURN_IF_ERROR(dispatch(3, static_cast<std::uint32_t>(partials)));
  HYLE_RETURN_IF_ERROR(dispatch(4, rows));
  HYLE_RETURN_IF_ERROR(dispatch(5, static_cast<std::uint32_t>(partials)));
  return dispatch(6, rows);
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan sampling metadata allocation failed");
}
} // namespace hyle::compute::vulkan::detail
