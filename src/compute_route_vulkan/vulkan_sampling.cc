#include "vulkan_program_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstdint>
#include <cstring>
#include <new>

#include "hyle/compute/sampling_ops.h"
#include "vulkan_shaders.h"
#include "vulkan_tensor.h"

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

Status lower_portable_topk(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto selection = step.operation.get_if<TopK>();
    if (!selection || step.inputs.size() != 1 || step.outputs.size() != 2)
      return Status::InvalidArgument("Vulkan portable TopK schema");
    const auto input = program.values[step.inputs[0]].description;
    if (input.rank() != 2 || !input.is_plain() || !input.is_dense() ||
        input.data_type() != DataType::kFloat32 || selection->axis != -1)
      return Status::Unimplemented("Vulkan portable TopK requires canonical F32 rows");
    const auto rows64 = static_cast<std::uint64_t>(input.dimensions()[0]);
    const auto width64 = static_cast<std::uint64_t>(input.dimensions()[1]);
    if (!rows64 || !width64 || width64 > (1U << 30) || !selection->count ||
        selection->count > width64)
      return Status::Unimplemented("Vulkan portable TopK row capacity");
    const auto rows = static_cast<std::uint32_t>(rows64);
    const auto width = static_cast<std::uint32_t>(width64);
    const auto padded = std::bit_ceil(width);
    const auto &limits = program.runtime->device->properties.limits;
    if (limits.maxComputeWorkGroupInvocations < 128 ||
        limits.maxComputeWorkGroupSize[0] < 128 ||
        limits.maxPerStageDescriptorStorageBuffers < 4 ||
        limits.maxDescriptorSetStorageBuffers < 4 ||
        rows64 > limits.maxStorageBufferRange / 8 / padded ||
        rows64 > UINT32_MAX / 8 / padded)
      return Status::Unimplemented("Vulkan portable TopK scratch exceeds native limits");
    const auto append = [&](TensorDesc description) {
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({description, {}});
      program.aliases.push_back({id, 0});
      return id;
    };
    HYLE_ASSIGN_OR_RETURN(auto scratch, TensorDesc::dense(DataType::kInt32,
        std::array<std::int64_t,3>{rows, padded, 2}));
    const std::array temporary{
        append(program.values[step.outputs[0]].description),
        append(program.values[step.outputs[1]].description)};
    std::uint32_t source = append(scratch), destination = append(scratch);
    std::array<std::shared_ptr<Pipeline>, 3> pipelines;
    for (std::uint32_t mode = 0; mode < pipelines.size(); ++mode) {
      const std::array<SpecializationConstant, 6> constants{
          SpecializationConstant{0, mode}, {1, rows}, {2, width}, {3, padded},
          {4, static_cast<std::uint32_t>(selection->count)}, {5, selection->largest ? 1U : 0U}};
      HYLE_ASSIGN_OR_RETURN(pipelines[mode], create_pipeline(program.runtime->device,
          {shaders::portable_topk(), 4, 24, constants}));
    }
    const auto dispatch = [&](std::uint32_t mode, std::uint32_t input_value,
                              std::uint32_t output_value, std::uint32_t size = 0,
                              std::uint32_t distance = 0) -> Status {
      const auto elements = rows64 * (mode == 2 ? selection->count : padded);
      const auto groups = (elements + 127) / 128;
      const auto x = (std::min)(groups, std::uint64_t{limits.maxComputeWorkGroupCount[0]});
      if (!x || (groups + x - 1) / x > limits.maxComputeWorkGroupCount[1])
        return Status::Unimplemented("Vulkan portable TopK dispatch exceeds native limits");
      Dispatch stage;
      stage.pipeline = pipelines[mode];
      stage.values = {input_value, output_value, temporary[0], temporary[1]};
      const std::array<std::uint32_t,6> push{0,0,0,0,size,distance};
      stage.push.resize(sizeof(push));
      std::memcpy(stage.push.data(), push.data(), sizeof(push));
      stage.tensor_byte_origins = true;
      stage.tensor_byte_origin_count = 4;
      stage.groups = {static_cast<std::uint32_t>(x),
                      static_cast<std::uint32_t>((groups + x - 1) / x), 1};
      program.dispatches.push_back(std::move(stage));
      return Status::Ok();
    };
    HYLE_RETURN_IF_ERROR(dispatch(0, step.inputs[0], source));
    for (std::uint32_t size = 2; size <= padded; size *= 2)
      for (auto distance = size / 2; distance; distance /= 2) {
        HYLE_RETURN_IF_ERROR(dispatch(1, source, destination, size, distance));
        std::swap(source, destination);
      }
    HYLE_RETURN_IF_ERROR(dispatch(2, source, destination));
    // Publish through the existing typed copy, including any output aliases.
    // Private sort payloads and the original Work retire under the same owner.
    for (std::size_t port = 0; port < temporary.size(); ++port)
      HYLE_RETURN_IF_ERROR(lower_tensor(program,
          {Operation(Copy{}), {temporary[port]}, {step.outputs[port]}, {}, {1,1,1}, {}}));
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan portable TopK metadata");
  }
}

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
  // Unused raw-word bindings still need an aligned descriptor. Reuse the
  // output owner rather than restricting a valid unaligned identity View.
  auto scratch = step.outputs[0]; // Raw bits need no temporary value.
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
