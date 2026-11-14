#include "vulkan_packed_sequence.h"
#include "vulkan_program_internal.h"
#include "vulkan_shaders.h"
#include "hyle/compute/sequence_ops.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <new>

namespace hyle::compute::vulkan::detail {

Status lower_packed_conv(ProgramState &program, const PlanStep &step) noexcept try {
  if (!step.operation.custom_as<sequence_ops::PackedCausalConv1d>() ||
      step.inputs.size() != 4 || step.outputs.size() != 2)
    return Status::InvalidArgument("Vulkan packed convolution ports differ");
  std::array<TensorDesc, 4> inputs;
  std::array<TensorDesc, 2> outputs;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation, inputs, outputs));
  for (std::size_t i = 0; i < outputs.size(); ++i)
    if (outputs[i] != program.values[step.outputs[i]].description)
      return Status::InvalidArgument("Vulkan packed convolution output descriptions differ");
  const auto x = inputs[0].dimensions(), weights = inputs[1].dimensions();
  const auto history = inputs[2].dimensions();
  const auto &limits = program.runtime->device->properties.limits;
  constexpr std::uint32_t bindings = 7, push_bytes = bindings * 4;
  if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
      limits.maxPerStageDescriptorStorageBuffers < bindings ||
      limits.maxDescriptorSetStorageBuffers < bindings || limits.maxPushConstantsSize < push_bytes)
    return Status::Unimplemented("Vulkan packed convolution descriptor/workgroup limits");
  for (const auto &description : inputs)
    if (*description.element_count() > UINT32_MAX ||
        *description.required_bytes() > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan packed convolution input indexing/buffer limits");
  // The tap address traverses history || real rows, including empty ports.
  if (std::uint64_t(x[0]) + std::uint64_t(weights[1]) - 1 > UINT32_MAX)
    return Status::Unimplemented("Vulkan packed convolution tap indexing limit");
  const auto groups = (std::uint64_t(x[1]) + 63) / 64;
  if (groups > limits.maxComputeWorkGroupCount[0] ||
      std::uint64_t(history[0]) > limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan packed convolution dispatch limits");
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
    const auto id = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({status, {}});
    program.aliases.push_back({id, 0});
    program.status_value = id;
  }
  const auto type = inputs[0].data_type();
  const auto encoding = type == DataType::kFloat16 ? 1U : type == DataType::kBFloat16 ? 2U : 0U;
  const std::array<std::uint32_t, bindings> values{
      step.inputs[0], step.inputs[1], *program.status_value, step.inputs[2],
      step.inputs[3], step.outputs[0], step.outputs[1]};
  for (std::uint32_t phase = 0; phase < 2; ++phase) {
    const std::array<SpecializationConstant, 6> constants{{
        {0, phase}, {1, std::uint32_t(x[0])}, {2, std::uint32_t(x[1])},
        {3, std::uint32_t(weights[1])}, {4, std::uint32_t(history[0])}, {5, encoding}}};
    Dispatch dispatch;
    const SpirvDescription shader{encoding ? shaders::packed_conv_raw16() : shaders::packed_conv(),
        bindings, push_bytes, constants};
    if (encoding) {
      HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_halfword_pipeline(program.runtime->device, shader));
    } else {
      HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device, shader));
    }
    dispatch.values.assign(values.begin(), values.end());
    dispatch.push.resize(push_bytes);
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.groups = phase == 0 ? std::array<std::uint32_t, 3>{1, 1, 1}
        : std::array<std::uint32_t, 3>{std::uint32_t(groups), std::uint32_t(history[0]), 1};
    program.dispatches.push_back(std::move(dispatch));
  }
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan packed convolution preparation");
}

Status lower_packed_gdn(ProgramState &program, const PlanStep &step) noexcept try {
  const auto descriptor = step.operation.custom_as<sequence_ops::PackedGatedDeltaNet>();
  const auto prefixes = step.operation.custom_as<sequence_ops::PackedGatedDeltaNetPrefixes>();
  if ((!descriptor && !prefixes) || step.inputs.size() != 7 || step.outputs.size() != 2)
    return Status::InvalidArgument("Vulkan packed GDN ports differ");
  std::array<TensorDesc, 7> inputs;
  std::array<TensorDesc, 2> outputs;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation, inputs, outputs));
  for (std::size_t i = 0; i < outputs.size(); ++i)
    if (outputs[i] != program.values[step.outputs[i]].description)
      return Status::InvalidArgument("Vulkan packed GDN output descriptions differ");
  if (inputs[0].data_type() != DataType::kFloat32)
    return Status::Unimplemented("Vulkan packed GDN requires F32 Q/K/V");
  const auto q = inputs[0].dimensions(), v = inputs[2].dimensions();
  const auto state = inputs[5].dimensions();
  // The native recurrence has a bounded private state row. Every depth within
  // this capacity is supported, independently of model or head geometry.
  if (q[2] > 256)
    return Status::Unimplemented("Vulkan packed GDN private state depth exceeds 256");
  const auto &limits = program.runtime->device->properties.limits;
  constexpr std::uint32_t bindings = 10, push_bytes = bindings * 4;
  if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
      limits.maxPerStageDescriptorStorageBuffers < bindings ||
      limits.maxDescriptorSetStorageBuffers < bindings || limits.maxPushConstantsSize < push_bytes)
    return Status::Unimplemented("Vulkan packed GDN descriptor/workgroup limits");
  for (const auto &description : inputs) {
    const auto count = description.element_count().value();
    if (count > UINT32_MAX || *description.required_bytes() > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan packed GDN input indexing/buffer limits");
  }
  for (const auto &description : outputs)
    if (*description.element_count() > UINT32_MAX ||
        *description.required_bytes() > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan packed GDN output indexing/buffer limits");
  const auto groups = (std::uint64_t(v[1]) * std::uint64_t(v[2]) + 63) / 64;
  if (groups > limits.maxComputeWorkGroupCount[0] ||
      std::uint64_t(state[0]) > limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan packed GDN dispatch limits");
  const auto scale = (prefixes ? prefixes->output_scale : descriptor->output_scale)
      .value_or(1.0F / std::sqrt(float(q[2])));
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
    const auto id = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({status, {}});
    program.aliases.push_back({id, 0});
    program.status_value = id;
  }
  const std::array<std::uint32_t, bindings> values{
      step.inputs[0], step.inputs[1], *program.status_value, step.inputs[2],
      step.inputs[3], step.inputs[4], step.inputs[5], step.inputs[6],
      step.outputs[0], step.outputs[1]};
  for (std::uint32_t phase = 0; phase < 2; ++phase) {
    const std::array<SpecializationConstant, 11> constants{{
        {0, phase}, {1, std::uint32_t(q[0])}, {2, std::uint32_t(q[1])},
        {3, std::uint32_t(v[1])}, {4, std::uint32_t(q[2])}, {5, std::uint32_t(v[2])},
        {6, std::uint32_t(state[0])}, {7, std::uint32_t(inputs[3].dimensions()[2])},
        {8, prefixes ? prefixes->max_prefixes : descriptor->snapshot_count},
        {9, std::bit_cast<std::uint32_t>(scale)}, {10, prefixes ? 1U : 0U}}};
    Dispatch dispatch;
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {shaders::packed_gdn(), bindings, push_bytes, constants}));
    dispatch.values.assign(values.begin(), values.end());
    dispatch.push.resize(push_bytes);
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.groups = phase == 0 ? std::array<std::uint32_t, 3>{1, 1, 1}
        : std::array<std::uint32_t, 3>{std::uint32_t(groups), std::uint32_t(state[0]), 1};
    program.dispatches.push_back(std::move(dispatch));
  }
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan packed GDN preparation");
}
} // namespace hyle::compute::vulkan::detail
