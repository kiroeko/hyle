#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <new>
#include <utility>

#include "../compute_packed_shared/artifact_packed_linear.h"
#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {
namespace {
Status check_storage(const ProgramState &program, const TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(bytes));
  if (capacity > UINT32_MAX ||
      capacity > program.runtime->device->properties.limits.maxStorageBufferRange)
    return Status::Unimplemented("Vulkan packed actual storage exceeds device limits");
  return Status::Ok();
}

// Original five semantic roles A/Q/Z/SF/g_idx; the published decoder supplies
// Q for the unused A slot. Neither route allocates an intermediate decoded bank.
Status lower_gptq_dispatches(ProgramState &program,
    const std::array<std::uint32_t, 5> &inputs, std::uint32_t output,
    std::uint64_t rows, std::uint64_t columns, std::uint64_t depth,
    std::uint32_t group_size, DataType decode_type, DataType operand_type,
    bool decode) {
  const auto &limits = program.runtime->device->properties.limits;
  if (rows == 0 || rows > UINT32_MAX || columns > UINT32_MAX ||
      depth > UINT32_MAX - 31U || limits.maxComputeWorkGroupInvocations < 64 ||
      limits.maxComputeWorkGroupSize[0] < 64 || !limits.maxComputeWorkGroupCount[0] ||
      !limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan GPTQ indexing/dispatch domain");
  for (auto value : inputs)
    HYLE_RETURN_IF_ERROR(check_storage(program, program.values[value].description));
  HYLE_RETURN_IF_ERROR(check_storage(program, program.values[output].description));
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
    const auto value = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({status, {}});
    program.aliases.push_back({value, 0});
    program.status_value = value;
  }
  const auto add = [&](std::uint32_t phase, std::uint64_t work,
                       std::uint64_t lanes) -> Status {
    const std::array<SpecializationConstant, 9> constants{{
        {0, std::uint32_t(rows)}, {1, std::uint32_t(columns)},
        {2, std::uint32_t(depth)}, {3, group_size}, {4, 0}, {5, 1},
        {6, *tensor_storage_type(decode_type)},
        {7, *tensor_storage_type(operand_type)}, {8, phase}}};
    Dispatch dispatch;
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {shaders::gptq_v1(), 7, 24, constants}));
    dispatch.values = {inputs[0], inputs[1], *program.status_value,
        inputs[2], inputs[3], inputs[4], output};
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.push.resize(24); // Six original/status origins; output is aligned.
    const auto groups = (work + lanes - 1) / lanes;
    const auto x = (std::min)(groups, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
    const auto y = (groups + x - 1) / x;
    if (y > limits.maxComputeWorkGroupCount[1])
      return Status::Unimplemented("Vulkan GPTQ grid exceeds native limits");
    dispatch.groups = {std::uint32_t(x), std::uint32_t(y), 1};
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  };
  // Every mutable index is validated on every replay, before any Z/SF address
  // can depend on it. The original per-dispatch barrier orders the next phase.
  HYLE_RETURN_IF_ERROR(add(0, depth, 64));
  const auto elements = *program.values[output].description.element_count();
  return add(decode ? 1 : 2, (elements + 1) / 2, decode ? 64 : 1);
}
Status lower_ct_grouped(ProgramState &program, const PlanStep &step,
    const packed::internal::ArtifactPackedLinear &descriptor,
    std::span<const TensorDesc> inputs, const TensorDesc &output) {
  const auto &a = inputs[0];
  const auto &ids = inputs.back();
  const auto experts = descriptor.expert_count;
  const auto flat_columns = std::uint64_t(inputs[1].dimensions()[0]);
  const auto columns = flat_columns / experts;
  const auto depth = std::uint64_t(a.dimensions()[2]);
  const auto rows = *ids.element_count();
  const auto slots = std::uint64_t(ids.dimensions()[1]);
  const auto &limits = program.runtime->device->properties.limits;
  if (rows == 0 || rows > UINT32_MAX || experts > UINT32_MAX || columns > UINT32_MAX ||
      slots == 0 || slots > UINT32_MAX || depth > UINT32_MAX - 31U ||
      a.data_type() != DataType::kFloat32 || output.data_type() != DataType::kFloat32 ||
      limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
      !limits.maxComputeWorkGroupCount[0] || !limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan CT grouped indexing/dispatch domain");
  for (const auto &input : inputs) HYLE_RETURN_IF_ERROR(check_storage(program, input));
  HYLE_RETURN_IF_ERROR(check_storage(program, output));
  const auto stride = a.byte_strides();
  for (const auto value : stride)
    if (!std::in_range<std::uint32_t>(value))
      return Status::Unimplemented("Vulkan CT grouped activation stride exceeds U32");
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
    const auto value = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({status, {}});
    program.aliases.push_back({value, 0});
    program.status_value = value;
  }
  const bool asymmetric = descriptor.format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4;
  const auto scale = step.inputs[asymmetric ? 3 : 2];
  const auto add = [&](std::uint32_t phase, std::uint64_t work, std::uint64_t lanes) -> Status {
    const std::array<SpecializationConstant, 14> constants{{
        {0, std::uint32_t(rows)}, {1, std::uint32_t(columns)}, {2, std::uint32_t(depth)},
        {3, descriptor.group_size}, {4, asymmetric ? 1U : 0U},
        {5, *tensor_storage_type(program.values[scale].description.data_type())},
        {6, *tensor_storage_type(descriptor.decode_type)}, {7, *tensor_storage_type(descriptor.operand_type)},
        {8, phase}, {9, std::uint32_t(experts)}, {10, std::uint32_t(slots)},
        {11, std::uint32_t(stride[0])},
        {12, a.dimensions()[1] == 1 ? 0U : std::uint32_t(stride[1])},
        {13, std::uint32_t(stride[2])}}};
    Dispatch dispatch;
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {shaders::ct_packed_grouped(), 7, 24, constants}));
    dispatch.values = {step.inputs[0], step.inputs[1], *program.status_value,
        asymmetric ? step.inputs[2] : step.inputs[1], scale, step.inputs.back(), step.outputs[0]};
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.push.resize(24);
    const auto groups = (work + lanes - 1) / lanes;
    const auto x = (std::min)(groups, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
    const auto y = (groups + x - 1) / x;
    if (y > limits.maxComputeWorkGroupCount[1])
      return Status::Unimplemented("Vulkan CT grouped grid exceeds native limits");
    dispatch.groups = {std::uint32_t(x), std::uint32_t(y), 1};
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  };
  // Every mutable ID is checked before the first selected bank address. The
  // existing recorder inserts the visibility barrier and reports failed status.
  HYLE_RETURN_IF_ERROR(add(0, rows, 64));
  return add(1, (*output.element_count() + 1) / 2, 1);
}
} // namespace

Status lower_gptq_decode(ProgramState &program, const PlanStep &step) noexcept try {
  const auto descriptor = step.operation.custom_as<model_ops::ArtifactWeightDecode>();
  if (!descriptor || descriptor->format != model_ops::WeightArtifactFormat::kGptqV1Int4 ||
      step.inputs.size() != 4 || step.outputs.size() != 1)
    return Status::InvalidArgument("Vulkan GPTQ decoder ports differ");
  std::array<TensorDesc, 4> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  std::array<TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation, inputs, output));
  if (output[0] != program.values[step.outputs[0]].description)
    return Status::InvalidArgument("Vulkan GPTQ decoder output differs");
  const std::array source{step.inputs[0], step.inputs[0], step.inputs[1],
      step.inputs[2], step.inputs[3]};
  return lower_gptq_dispatches(program, source, step.outputs[0], 1,
      std::uint64_t(output[0].dimensions()[0]),
      std::uint64_t(output[0].dimensions()[1]),
      descriptor->group_size,
      output[0].data_type(), output[0].data_type(), true);
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan GPTQ decoder preparation metadata");
}

Status lower_ct_packed(ProgramState &program, const PlanStep &step) noexcept try {
  const auto descriptor = step.operation.custom_as<packed::internal::ArtifactPackedLinear>();
  if (!descriptor || step.outputs.size() != 1 ||
      step.inputs.size() != descriptor->operation_traits().inputs.minimum)
    return Status::InvalidArgument("Vulkan CT packed ports differ");
  std::array<TensorDesc, 5> inputs;
  for (std::size_t port = 0; port < step.inputs.size(); ++port)
    inputs[port] = program.values[step.inputs[port]].description;
  std::array<TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(descriptor->infer_output_descriptions(
      std::span(inputs).first(step.inputs.size()), output));
  if (output[0] != program.values[step.outputs[0]].description ||
      !inputs[0].is_plain() || (!descriptor->expert_count && !inputs[0].is_dense()) ||
      !output[0].is_dense())
    return Status::Unimplemented("Vulkan CT packed requires admitted original operands/output");
  if (descriptor->expert_count)
    return lower_ct_grouped(program, step, *descriptor,
        std::span(inputs).first(step.inputs.size()), output[0]);
  const auto &limits = program.runtime->device->properties.limits;
  const bool gptq = descriptor->format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  const auto depth = std::uint64_t(inputs[1].dimensions()[gptq ? 0 : 1]) * 8;
  const auto columns = std::uint64_t(inputs[1].dimensions()[gptq ? 1 : 0]);
  const auto rows = *inputs[0].element_count() / depth;
  if (gptq) {
    const std::array source{step.inputs[0], step.inputs[1], step.inputs[2],
        step.inputs[3], step.inputs[4]};
    return lower_gptq_dispatches(program, source, step.outputs[0], rows, columns,
        depth, descriptor->group_size, descriptor->decode_type, descriptor->operand_type, false);
  }
  if (rows == 0 || columns > UINT32_MAX || depth > UINT32_MAX - 31U || rows > UINT32_MAX ||
      limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
      !limits.maxComputeWorkGroupCount[0] || !limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan CT packed indexing/dispatch domain");
  // Only real byte carriers are bounded. There is no dense N*K allocation.
  const auto check = [&](const TensorDesc &description) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(bytes));
    if (capacity > UINT32_MAX || capacity > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan CT packed actual storage exceeds device limits");
    return Status::Ok();
  };
  for (const auto &input : std::span(inputs).first(step.inputs.size())) HYLE_RETURN_IF_ERROR(check(input));
  HYLE_RETURN_IF_ERROR(check(output[0]));
  const auto asymmetric = descriptor->format ==
      model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4;
  const auto scale = step.inputs.back();
  const std::array<SpecializationConstant, 8> constants{{
      {0, std::uint32_t(rows)}, {1, std::uint32_t(columns)}, {2, std::uint32_t(depth)},
      {3, descriptor->group_size}, {4, asymmetric ? 1U : 0U},
      {5, *tensor_storage_type(program.values[scale].description.data_type())},
      {6, *tensor_storage_type(descriptor->decode_type)},
      {7, *tensor_storage_type(descriptor->operand_type)}}};
  Dispatch dispatch;
  HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
      {shaders::ct_packed(), 5, 16, constants}));
  dispatch.values = {step.inputs[0], step.inputs[1],
      asymmetric ? step.inputs[2] : step.inputs[1], scale, step.outputs[0]};
  dispatch.tensor_byte_origins = true;
  dispatch.push.resize(16); // Four input origins; output is an aligned owned word span.
  const auto pairs = (*output[0].element_count() + 1) / 2;
  const auto x = (std::min)(pairs, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
  const auto y = (pairs + x - 1) / x;
  if (y > limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan CT packed grid exceeds native limits");
  dispatch.groups = {std::uint32_t(x), std::uint32_t(y), 1};
  program.dispatches.push_back(std::move(dispatch));
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan CT packed preparation metadata");
}
} // namespace hyle::compute::vulkan::detail
