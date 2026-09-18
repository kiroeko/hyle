#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <limits>

#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {
std::optional<std::uint32_t> tensor_storage_type(DataType type) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return 0;
  case DataType::kFloat16:
    return 1;
  case DataType::kBFloat16:
    return 2;
  case DataType::kInt32:
    return 3;
  case DataType::kInt64:
    return 4;
  case DataType::kBool:
    return 5;
  default:
    return std::nullopt;
  }
}

Status lower_tensor(ProgramState &program, const PlanStep &step) {
  std::uint32_t operation = 0, mode = 0;
  std::uint64_t fill = 0;
  switch (step.operation.kind()) {
  case OperationKind::kCopy:
    operation = 0;
    break;
  case OperationKind::kFill:
    operation = 1;
    fill = step.operation.get_if<Fill>()->value.bits();
    break;
  case OperationKind::kCast:
    operation = 2;
    break;
  case OperationKind::kUnary:
    operation = 3;
    mode = static_cast<std::uint32_t>(step.operation.get_if<Unary>()->kind);
    break;
  case OperationKind::kBinary:
    operation = 4;
    mode = static_cast<std::uint32_t>(step.operation.get_if<Binary>()->kind);
    break;
  case OperationKind::kSelect:
    operation = 5;
    break;
  default:
    return Status::Unimplemented(
        "operation has no admitted Vulkan tensor lowering");
  }
  static_assert(static_cast<unsigned>(UnaryKind::kSoftplus) == 12);
  static_assert(static_cast<unsigned>(BinaryKind::kMaximum) == 5);
  const auto &output = program.values[step.output].description;
  const auto output_type = tensor_storage_type(output.data_type());
  if (!output_type || !output.is_dense())
    return Status::Unimplemented(
        "Vulkan tensor output requires admitted dense storage");
  if (operation >= 2 && operation <= 4) {
    if (*output_type > 2)
      return Status::Unimplemented(
          "Vulkan tensor arithmetic requires F32/F16/BF16");
    for (auto value : step.inputs) {
      const auto type =
          tensor_storage_type(program.values[value].description.data_type());
      if (!type || *type > 2)
        return Status::Unimplemented(
            "Vulkan tensor arithmetic requires F32/F16/BF16");
    }
  }
  const auto elements = *output.element_count();
  if (elements == 0)
    return Status::Ok();
  // Specialization freezes types, modes and address mapping into the native
  // pipeline. No layout metadata upload or interpretation is added to replay.
  std::array<SpecializationConstant, 42> specialization{};
  for (std::uint32_t index = 0; index < specialization.size(); ++index)
    specialization[index].id = index;
  specialization[0].value = operation;
  specialization[4].value = *output_type;
  specialization[5].value = mode;
  specialization[6].value = static_cast<std::uint32_t>(elements);
  specialization[7].value = static_cast<std::uint32_t>(fill);
  specialization[8].value = static_cast<std::uint32_t>(fill >> 32);
  // ID9 is reserved for later tensor recipes; all current dimensions occupy
  // IDs10..17, followed by three right-aligned byte-stride vectors.
  for (std::size_t axis = 0; axis < 8; ++axis) {
    const auto padding = 8 - output.rank();
    specialization[10 + axis].value =
        axis < padding
            ? 1u
            : static_cast<std::uint32_t>(output.dimensions()[axis - padding]);
  }
  Dispatch dispatch;
  dispatch.values.resize(4, step.output);
  for (std::size_t port = 0; port < step.inputs.size(); ++port) {
    const auto &input = program.values[step.inputs[port]].description;
    specialization[1 + port].value = *tensor_storage_type(input.data_type());
    dispatch.values[port] = step.inputs[port];
    for (std::size_t axis = 0; axis < input.rank(); ++axis) {
      const auto stride =
          input.dimensions()[axis] == 1 ? 0 : input.byte_strides()[axis];
      if (!std::in_range<std::uint32_t>(stride))
        return Status::Unimplemented("Vulkan tensor byte stride exceeds U32");
      specialization[18 + port * 8 + 8 - input.rank() + axis].value =
          static_cast<std::uint32_t>(stride);
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto pipeline, create_pipeline(program.runtime->device,
                                                       {shaders::tensor(), 4,
                                                        12, specialization}));
  dispatch.pipeline = std::move(pipeline);
  dispatch.tensor_byte_origins = true;
  dispatch.push.resize(12);
  const auto words = (*output.required_bytes() + 3) / 4;
  const auto groups = static_cast<std::uint32_t>((words + 127) / 128);
  const auto &limits = program.runtime->device->properties.limits;
  const auto x = (std::min)(groups, limits.maxComputeWorkGroupCount[0]);
  if (x == 0 || (groups + x - 1) / x > limits.maxComputeWorkGroupCount[1])
    return Status::OutOfRange("Vulkan tensor dispatch exceeds device limits");
  dispatch.groups = {x, (groups + x - 1) / x, 1};
  program.dispatches.push_back(std::move(dispatch));
  return Status::Ok();
}
} // namespace hyle::compute::vulkan::detail
