#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <new>

#include "hyle/compute/model_ops.h"
#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {
Status lower_blockwise_w4a8(ProgramState &program, const PlanStep &step) noexcept try {
  const bool grouped = step.operation.custom_as<model_ops::GroupedBlockwiseW4A8Linear>().has_value();
  if ((!grouped && !step.operation.custom_as<model_ops::BlockwiseW4A8Linear>()) ||
      step.inputs.size() != (grouped ? 4U : 3U) || step.outputs.size() != 1)
    return Status::InvalidArgument("Vulkan blockwise W4A8 ports differ");
  std::array<TensorDesc, 4> descriptions;
  const std::span inputs(descriptions.data(), step.inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  std::array<TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation, inputs, output));
  if (output[0] != program.values[step.outputs[0]].description)
    return Status::InvalidArgument("Vulkan blockwise W4A8 output differs from schema");
  const auto columns = std::uint64_t(inputs[1].dimensions()[grouped ? 1 : 0]);
  const auto depth = std::uint64_t(inputs[0].dimensions().back());
  const auto rows = *output[0].element_count() / columns;
  const auto slots = grouped ? std::uint64_t(inputs[3].dimensions()[1]) : 1;
  const auto experts = grouped ? std::uint64_t(inputs[1].dimensions()[0]) : 1;
  const auto token_stride = grouped && inputs[0].dimensions()[0] > 1
      ? std::uint64_t(inputs[0].byte_strides()[0]) : 0;
  const auto slot_stride = grouped && inputs[0].dimensions()[1] != 1
      ? std::uint64_t(inputs[0].byte_strides()[1]) : 0;
  const auto quant_slots = grouped && slot_stride != 0 ? slots : 1;
  const auto quant_rows = rows / slots * quant_slots;
  const auto &limits = program.runtime->device->properties.limits;
  if (rows > UINT32_MAX || columns > UINT32_MAX || depth > UINT32_MAX ||
      slots > UINT32_MAX || experts > UINT32_MAX || token_stride > UINT32_MAX ||
      slot_stride > UINT32_MAX || limits.maxComputeWorkGroupInvocations < 64 ||
      limits.maxComputeWorkGroupSize[0] < 64 || !limits.maxComputeWorkGroupCount[0] ||
      !limits.maxComputeWorkGroupCount[1])
    return Status::Unimplemented("Vulkan blockwise W4A8 indexing/dispatch domain");
  const auto check_storage = [&](const TensorDesc &description) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(bytes));
    if (capacity > UINT32_MAX || capacity > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan blockwise W4A8 actual storage exceeds device limits");
    return Status::Ok();
  };
  for (const auto &input : inputs) HYLE_RETURN_IF_ERROR(check_storage(input));
  HYLE_RETURN_IF_ERROR(check_storage(output[0]));
  // Original packed weights remain resident. Only the unique source rows'
  // I8 codes and one widened, already BF16-rounded scale per group are stored.
  HYLE_ASSIGN_OR_RETURN(auto codes_description, TensorDesc::dense(DataType::kUInt8,
      std::array{std::int64_t(quant_rows), std::int64_t(depth)}));
  HYLE_ASSIGN_OR_RETURN(auto scales_description, TensorDesc::dense(DataType::kFloat32,
      std::array{std::int64_t(quant_rows), std::int64_t(depth / 512)}));
  HYLE_RETURN_IF_ERROR(check_storage(codes_description));
  HYLE_RETURN_IF_ERROR(check_storage(scales_description));
  const auto append = [&](const TensorDesc &description) {
    const auto value = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({description, {}});
    program.aliases.push_back({value, 0});
    return value;
  };
  const auto codes = append(codes_description), scales = append(scales_description);
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kInt32, {}));
    program.status_value = append(description);
  }
  const auto add = [&](std::uint32_t phase, std::uint32_t activation,
                       std::uint32_t destination, std::uint64_t work) -> Status {
    const std::array<SpecializationConstant, 9> constants{{
        {0, phase}, {1, std::uint32_t(rows)}, {2, std::uint32_t(columns)},
        {3, std::uint32_t(depth)}, {4, std::uint32_t(slots)},
        {5, std::uint32_t(experts)}, {6, std::uint32_t(token_stride)},
        {7, std::uint32_t(slot_stride)}, {8, std::uint32_t(quant_slots)}}};
    const auto bindings = grouped ? 7U : 6U, push_bytes = (bindings - 1) * 4U;
    Dispatch dispatch;
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {grouped ? shaders::blockwise_w4a8_grouped() : shaders::blockwise_w4a8(),
         bindings, push_bytes, constants}));
    // The existing checked-dispatch ABI keeps its status at binding2.
    dispatch.values = {activation, step.inputs[1], *program.status_value,
        step.inputs[2], scales};
    if (grouped) dispatch.values.push_back(step.inputs[3]);
    dispatch.values.push_back(destination);
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.push.resize(push_bytes);
    const auto groups = (work + 63) / 64;
    const auto x = (std::min)(groups, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
    const auto y = (groups + x - 1) / x;
    if (y > limits.maxComputeWorkGroupCount[1])
      return Status::Unimplemented("Vulkan blockwise W4A8 grid exceeds native limits");
    dispatch.groups = {std::uint32_t(x), std::uint32_t(y), 1};
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  };
  // The original recorder orders dispatches and preserves checked Completion.
  // No selected scale or weight address is formed before all IDs pass.
  if (grouped) HYLE_RETURN_IF_ERROR(add(0, step.inputs[0], codes, rows));
  HYLE_RETURN_IF_ERROR(add(1, step.inputs[0], codes, quant_rows * (depth / 512)));
  return add(2, codes, step.outputs[0], (*output[0].element_count() + 1) / 2);
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan blockwise W4A8 preparation metadata");
}
} // namespace hyle::compute::vulkan::detail
