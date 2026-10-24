#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <new>

#include "../compute_packed_shared/static_fp8_linear.h"
#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {

Status lower_static_fp8(ProgramState &program, const PlanStep &step) noexcept try {
  const auto descriptor = step.operation.custom_as<packed::internal::StaticFloat8Linear>();
  if (!descriptor || step.inputs.size() != 4 || step.outputs.size() != 1)
    return Status::InvalidArgument("Vulkan static FP8 linear ports differ");
  std::array<TensorDesc, 4> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port)
    inputs[port] = program.values[step.inputs[port]].description;
  std::array<TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(descriptor->infer_output_descriptions(inputs, output));
  if (output[0] != program.values[step.outputs[0]].description)
    return Status::InvalidArgument("Vulkan static FP8 linear output differs");
  const auto rows = std::uint64_t(inputs[0].dimensions()[0]);
  const auto depth = std::uint64_t(inputs[0].dimensions()[1]);
  const auto columns = std::uint64_t(inputs[1].dimensions()[0]);
  const auto &limits = program.runtime->device->properties.limits;
  if (rows > UINT32_MAX || columns > UINT32_MAX || depth > UINT32_MAX - 31U ||
      limits.maxComputeWorkGroupInvocations < 64 ||
      limits.maxComputeWorkGroupSize[0] < 64 || !limits.maxComputeWorkGroupCount[0] ||
      !limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan static FP8 linear indexing/dispatch domain");
  // Only actual typed carriers and output exist. No hypothetical dense F32 W
  // capacity is imposed on the byte-addressed packed consumer.
  const auto check = [&](const TensorDesc &description) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(bytes));
    if (capacity > UINT32_MAX || capacity > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan static FP8 actual storage exceeds device limits");
    return Status::Ok();
  };
  for (const auto &input : inputs) HYLE_RETURN_IF_ERROR(check(input));
  HYLE_RETURN_IF_ERROR(check(output[0]));
  const std::array<SpecializationConstant, 8> constants{{
      {0, std::uint32_t(rows)}, {1, std::uint32_t(columns)},
      {2, std::uint32_t(depth)}, {3, *tensor_storage_type(inputs[0].data_type())},
      {4, *tensor_storage_type(output[0].data_type())},
      {5, *inputs[2].element_count() > 1 ? 1u : 0u},
      {6, *inputs[3].element_count() > 1 ? 1u : 0u},
      {7, descriptor->scale_order == packed::internal::StaticFloat8ScaleOrder::kWeightThenActivation ? 1u : 0u}}};
  Dispatch dispatch;
  HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
      {shaders::static_fp8(), 5, 16, constants}));
  dispatch.values = {step.inputs[0], step.inputs[1], step.inputs[2], step.inputs[3], step.outputs[0]};
  dispatch.tensor_byte_origins = true;
  dispatch.push.resize(16); // Four original inputs; output owns aligned padded words.
  const auto pairs = (*output[0].element_count() + 1) / 2;
  const auto x = (std::min)(pairs, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
  const auto y = (pairs + x - 1) / x;
  if (y > limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan static FP8 linear grid exceeds device limits");
  dispatch.groups = {std::uint32_t(x), std::uint32_t(y), 1};
  program.dispatches.push_back(std::move(dispatch));
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan static FP8 linear preparation metadata");
}
} // namespace hyle::compute::vulkan::detail
