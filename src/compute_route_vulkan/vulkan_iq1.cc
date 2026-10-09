#include "vulkan_iq1.h"
#include "vulkan_mlx.h"
#include "vulkan_tensor.h"

#include <limits>
#include <new>
#include <optional>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::vulkan::detail {
namespace iv = iree_vulkan;
namespace {
std::optional<iv::ProgramKind> kind(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (type == GgufQuantizedType::kIq1Xs) return iv::ProgramKind::Iq1XsGrouped;
  if (type == GgufQuantizedType::kIq1Xxs) return iv::ProgramKind::Iq1XxsGrouped;
  if (type == GgufQuantizedType::kIq1Xxxs) return iv::ProgramKind::Iq1XxxsGrouped;
  return std::nullopt;
}
std::uint32_t append(ProgramState &program, const TensorDesc &description) {
  const auto id = static_cast<std::uint32_t>(program.values.size());
  program.values.push_back({description, {}});
  program.aliases.push_back({id, 0});
  return id;
}
} // namespace

bool iq1_storage_description(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (!kind(description) || description.rank() != 3 || !type) return false;
  const auto shape = description.dimensions();
  const auto canonical = gguf_grouped_weight_description(*type,
      description.data_type(), shape[0], shape[1], shape[2]);
  return canonical && *canonical == description;
}

bool iq1_operation(const ProgramState &program, const PlanStep &step) noexcept {
  return step.operation.custom_as<model_ops::GroupedMatMul>() &&
         step.inputs.size() == 3 &&
         kind(program.values[step.inputs[0]].description).has_value();
}

Status lower_iq1(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto &weight = program.values[step.inputs[0]].description;
    const auto &activation = program.values[step.inputs[1]].description;
    const auto &output = program.values[step.outputs[0]].description;
    if (!iq1_storage_description(weight) ||
        activation.data_type() != DataType::kFloat32 || !activation.is_plain() ||
        output.data_type() != DataType::kFloat32)
      return Status::Unimplemented("Vulkan original IQ1 packed/F32 contract");
    iv::ProgramSpec spec;
    spec.kind = *kind(weight);
    spec.experts = weight.dimensions()[0];
    spec.columns = weight.dimensions()[1];
    spec.depth = weight.dimensions()[2];
    spec.rows = static_cast<std::int64_t>(*output.element_count()) / spec.columns;
    const auto &device = *program.runtime->device;
    const auto &limits = device.properties.limits;
    const auto &features = device.provider.native();
    const auto row_bytes = spec.depth / 256 * iv::iq1_block_bytes(spec.kind);
    if (spec.rows <= 0 || spec.depth <= 0 || spec.depth % 256 ||
        spec.rows > INT32_MAX / 4 / spec.depth ||
        spec.experts > INT32_MAX / spec.columns ||
        spec.experts * spec.columns > INT32_MAX / row_bytes ||
        spec.rows > INT32_MAX / 4 / 32 / spec.columns ||
        static_cast<std::uint64_t>(spec.rows * spec.columns) * 32 * 4 > limits.maxStorageBufferRange ||
        limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
        !features.shader_int16 || !features.shader_float16 ||
        !features.storage_buffer16 || !features.preserve_float32_specials)
      return Status::Unimplemented("Vulkan original IQ1 packed index, lane workspace or numeric features");
    HYLE_RETURN_IF_ERROR(ensure_iree_device(program));
    if (!program.status_value) {
      HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
      program.status_value = append(program, status);
    }
    const auto ids = program.values[step.inputs[2]].description;
    const auto safe = append(program, ids);
    HYLE_RETURN_IF_ERROR(lower_index_check(program, step.inputs[2], safe,
        static_cast<std::uint64_t>(spec.experts), true));
    const auto destination = append(program, output);
    Dispatch dispatch;
    dispatch.iree = spec;
    dispatch.values = {step.inputs[1], step.inputs[0], safe, destination};
    program.dispatches.push_back(std::move(dispatch));
    return lower_checked_copy(program, destination, step.outputs[0]);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan original IQ1 preparation metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
