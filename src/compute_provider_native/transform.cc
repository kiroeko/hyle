#include "native_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"

namespace hyle::compute::native::internal {
namespace {

Result<Completion> submit_view(const NativePlan &plan,
                               std::span<const ConstTensorView> inputs,
                               std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("native view plan is invalid"));

  const mem::ConstBufferView input = inputs[0].storage();
  const mem::BufferView output = outputs[0].storage();
  if (output.size() == 0)
    return completed(Status::Ok());
  const auto expected_offset =
      math::try_add(input.offset(), plan.parameters[0]);
  if (!expected_offset || input.resource() != output.resource() ||
      input.native_handle() != output.native_handle() ||
      output.offset() != *expected_offset) {
    return completed(Status::InvalidArgument(
        "view output does not bind the inferred input storage range"));
  }
  return completed(Status::Ok());
}

Result<Completion> submit_copy(const NativePlan &,
                               std::span<const ConstTensorView> inputs,
                               std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("native copy plan is invalid"));

  const TensorDesc &input_desc = inputs[0].description();
  const TensorDesc &output_desc = outputs[0].description();
  const mem::ConstBufferView input = inputs[0].storage();
  const mem::BufferView output = outputs[0].storage();
  auto elements = output_desc.element_count();
  if (!elements)
    return completed(std::move(elements).error());
  if (*elements == 0)
    return completed(Status::Ok());

  // Dense materialization is exactly Mem's range-copy contract. Preserve that
  // optimized resource seam instead of remapping and reimplementing memcpy.
  if (input_desc.is_dense())
    return completed(output.copy_from(input));

  const mem::MemoryProperties input_properties = input.resource().properties();
  const mem::MemoryProperties output_properties =
      output.resource().properties();
  if (!input_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = input.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }

  auto input_result = input.map_host();
  if (!input_result)
    return completed(std::move(input_result).error());
  mem::HostMapping input_mapping = std::move(*input_result);
  auto output_result = output.map_host(mem::HostAccess::kWrite);
  if (!output_result) {
    input_mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }

  const std::byte *source = input_mapping.bytes().data();
  std::byte *destination = writable->data();
  const std::size_t element_size = data_type_size(input_desc.data_type());
  if (!aligned_for_storage(source, input_desc.required_alignment()) ||
      !aligned_for_storage(destination, output_desc.required_alignment())) {
    output_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU copy mapping does not satisfy tensor alignment"));
  }

  const auto dimensions = input_desc.dimensions();
  const auto strides = input_desc.byte_strides();
  for (std::size_t linear = 0; linear < *elements; ++linear) {
    std::size_t remaining = linear;
    std::size_t source_offset = 0;
    for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
      const std::size_t axis = dimensions.size() - 1 - reverse;
      const std::size_t extent = static_cast<std::size_t>(dimensions[axis]);
      const std::size_t coordinate = remaining % extent;
      remaining /= extent;
      source_offset += coordinate * static_cast<std::size_t>(strides[axis]);
    }
    std::memcpy(destination + linear * element_size, source + source_offset,
                element_size);
  }

  Status terminal = Status::Ok();
  if (!output_properties.has(mem::MemoryProperty::kHostDeviceCoherent))
    preserve_first_error(terminal, output_mapping.flush());
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return completed(std::move(terminal));
}

Result<Completion> submit_fill(const NativePlan &plan,
                               std::span<const ConstTensorView> inputs,
                               std::span<const TensorView> outputs) noexcept {
  if (!inputs.empty() || outputs.size() != 1)
    return completed(Status::Internal("native fill plan is invalid"));
  const mem::BufferView output = outputs[0].storage();
  const std::size_t elements = plan.parameters[0];
  const std::size_t element_size = plan.parameters[1];
  if (elements == 0)
    return completed(Status::Ok());

  auto result = output.map_host(mem::HostAccess::kWrite);
  if (!result)
    return completed(std::move(result).error());
  mem::HostMapping mapping = std::move(*result);
  auto writable = mapping.writable_bytes();
  if (!writable) {
    mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  if (!aligned_for_storage(writable->data(),
                           outputs[0].description().required_alignment())) {
    mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU fill mapping does not satisfy tensor alignment"));
  }

  switch (element_size) {
  case sizeof(std::uint8_t):
    std::fill_n(reinterpret_cast<std::uint8_t *>(writable->data()), elements,
                static_cast<std::uint8_t>(plan.scalar_bits));
    break;
  case sizeof(std::uint16_t):
    std::fill_n(reinterpret_cast<std::uint16_t *>(writable->data()), elements,
                static_cast<std::uint16_t>(plan.scalar_bits));
    break;
  case sizeof(std::uint32_t):
    std::fill_n(reinterpret_cast<std::uint32_t *>(writable->data()), elements,
                static_cast<std::uint32_t>(plan.scalar_bits));
    break;
  case sizeof(std::uint64_t):
    std::fill_n(reinterpret_cast<std::uint64_t *>(writable->data()), elements,
                plan.scalar_bits);
    break;
  default:
    mapping.close().IgnoreError();
    return completed(Status::Internal("native fill scalar width is invalid"));
  }

  Status terminal = Status::Ok();
  if (!output.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, mapping.flush());
  }
  preserve_first_error(terminal, mapping.close());
  return completed(std::move(terminal));
}

} // namespace

Status prepare_transform(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept {
  switch (operation.kind()) {
  case OperationKind::kView: {
    const View *view = operation.get_if<View>();
    std::size_t byte_offset = 0;
    auto output_bytes = outputs[0].required_bytes();
    if (!output_bytes)
      return std::move(output_bytes).error();
    if (view->kind() == ViewKind::kSlice && *output_bytes != 0) {
      std::int32_t axis = view->axis();
      if (axis < 0)
        axis += static_cast<std::int32_t>(inputs[0].rank());
      const auto start = math::try_integer_cast<std::size_t>(view->start());
      const auto stride = math::try_integer_cast<std::size_t>(
          inputs[0].byte_strides()[static_cast<std::size_t>(axis)]);
      if (!start || !stride)
        return Status::OutOfRange("view storage offset exceeds size_t");
      const auto offset = math::try_multiply(*start, *stride);
      if (!offset)
        return Status::OutOfRange("view storage offset overflows size_t");
      byte_offset = *offset;
    }
    plan.parameters[0] = byte_offset;
    plan.submit_route = &submit_view;
    return Status::Ok();
  }
  case OperationKind::kCopy:
    if (!inputs[0].is_plain() || !outputs[0].is_plain())
      return Status::Unimplemented("native Copy requires plain storage");
    plan.submit_route = &submit_copy;
    return Status::Ok();
  case OperationKind::kFill: {
    const Fill *fill = operation.get_if<Fill>();
    auto elements = outputs[0].element_count();
    if (!elements)
      return std::move(elements).error();
    plan.parameters[0] = *elements;
    plan.parameters[1] = data_type_size(fill->output_type);
    plan.scalar_bits = fill->value.bits();
    plan.submit_route = &submit_fill;
    return Status::Ok();
  }
  default:
    return Status::Unimplemented("operation is not a native transform");
  }
}

} // namespace hyle::compute::native::internal
