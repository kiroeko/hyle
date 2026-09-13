#include "native_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"

namespace hyle::compute::native::internal {
namespace {

Status require_plain_scalar(const TensorDesc &description) noexcept {
  if (!description.is_plain() || data_type_size(description.data_type()) == 0) {
    return Status::Unimplemented(
        "native indexing requires plain scalar tensor storage");
  }
  return Status::Ok();
}

Result<std::size_t> normalized_axis(std::int32_t requested,
                                    std::size_t rank) noexcept {
  std::int64_t axis = requested;
  if (axis < 0)
    axis += static_cast<std::int64_t>(rank);
  if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) {
    return std::unexpected(
        Status::OutOfRange("native indexing axis is out of range"));
  }
  return static_cast<std::size_t>(axis);
}

Status invalidate_for_read(mem::ConstBufferView storage) noexcept {
  if (storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    return Status::Ok();
  }
  return storage.invalidate();
}

bool same_storage(mem::ConstBufferView left,
                  mem::BufferView right) noexcept {
  return left.resource() == right.resource() &&
         left.native_handle() == right.native_handle() &&
         left.offset() == right.offset() && left.size() == right.size();
}

void coordinates(std::span<const std::int64_t> dimensions,
                 std::size_t linear,
                 std::array<std::size_t, kMaxTensorRank> &result) noexcept {
  result.fill(0);
  for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
    const std::size_t axis = dimensions.size() - 1 - reverse;
    const std::size_t extent = static_cast<std::size_t>(dimensions[axis]);
    result[axis] = linear % extent;
    linear /= extent;
  }
}

std::int64_t read_index(const std::byte *data,
                        const TensorDesc &description,
                        std::size_t linear) noexcept {
  const std::size_t offset = logical_byte_offset(description, linear);
  if (description.data_type() == DataType::kInt32) {
    std::int32_t value = 0;
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
  }
  std::int64_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

Status validate_indices(const NativePlan &plan, const std::byte *data,
                        const TensorDesc &description,
                        std::size_t axis_extent) noexcept {
  if (plan.index_bounds == IndexBoundsPolicy::kUnchecked)
    return Status::Ok();
  auto count = description.element_count();
  if (!count)
    return std::move(count).error();
  for (std::size_t linear = 0; linear < *count; ++linear) {
    const std::int64_t value = read_index(data, description, linear);
    if (value < 0 || static_cast<std::uint64_t>(value) >= axis_extent) {
      return Status::OutOfRange("native index is outside the selected axis");
    }
  }
  return Status::Ok();
}

std::size_t product(std::span<const std::int64_t> dimensions,
                    std::size_t first, std::size_t last) noexcept {
  std::size_t result = 1;
  for (std::size_t axis = first; axis < last; ++axis)
    result *= static_cast<std::size_t>(dimensions[axis]);
  return result;
}

std::size_t indexed_source_offset(
    const TensorDesc &source, const TensorDesc &indices, std::size_t axis,
    const std::array<std::size_t, kMaxTensorRank> &output_coordinates,
    std::int64_t index) noexcept {
  std::size_t offset = 0;
  const auto strides = source.byte_strides();
  for (std::size_t source_axis = 0; source_axis < source.rank();
       ++source_axis) {
    std::size_t coordinate = 0;
    if (source_axis < axis) {
      coordinate = output_coordinates[source_axis];
    } else if (source_axis == axis) {
      coordinate = static_cast<std::size_t>(index);
    } else {
      coordinate = output_coordinates[source_axis - 1 + indices.rank()];
    }
    offset += coordinate * static_cast<std::size_t>(strides[source_axis]);
  }
  return offset;
}

std::size_t index_offset_from_output(
    const TensorDesc &indices, std::size_t source_axis,
    const std::array<std::size_t, kMaxTensorRank> &output_coordinates) noexcept {
  std::size_t offset = 0;
  const auto strides = indices.byte_strides();
  for (std::size_t axis = 0; axis < indices.rank(); ++axis) {
    offset += output_coordinates[source_axis + axis] *
              static_cast<std::size_t>(strides[axis]);
  }
  return offset;
}

std::int64_t read_index_at_offset(const std::byte *data, DataType type,
                                  std::size_t offset) noexcept {
  if (type == DataType::kInt32) {
    std::int32_t value = 0;
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
  }
  std::int64_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

detail::EnqueueResult
submit_gather(const NativePlan &plan,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 2 || outputs.size() != 1)
    return completed(Status::Internal("native Gather plan is invalid"));

  const TensorDesc &source_desc = inputs[0].description();
  const TensorDesc &indices_desc = inputs[1].description();
  const TensorDesc &output_desc = outputs[0].description();
  auto output_elements = output_desc.element_count();
  if (!output_elements)
    return completed(std::move(output_elements).error());
  if (*output_elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView source_storage = inputs[0].storage();
  const mem::ConstBufferView indices_storage = inputs[1].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  Status status = invalidate_for_read(source_storage);
  if (!status.ok())
    return completed(std::move(status));
  status = invalidate_for_read(indices_storage);
  if (!status.ok())
    return completed(std::move(status));

  auto source_result = source_storage.map_host();
  if (!source_result)
    return completed(std::move(source_result).error());
  mem::HostMapping source_mapping = std::move(*source_result);
  auto indices_result = indices_storage.map_host();
  if (!indices_result) {
    source_mapping.close().IgnoreError();
    return completed(std::move(indices_result).error());
  }
  mem::HostMapping indices_mapping = std::move(*indices_result);
  const std::byte *source = source_mapping.bytes().data();
  const std::byte *indices = indices_mapping.bytes().data();
  if (!aligned_for_storage(source, source_desc.required_alignment()) ||
      !aligned_for_storage(indices, indices_desc.required_alignment())) {
    indices_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU Gather input mapping does not satisfy tensor alignment"));
  }

  const std::size_t axis_extent =
      static_cast<std::size_t>(source_desc.dimensions()[plan.axis]);
  if (axis_extent == 0) {
    indices_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    return completed(
        Status::OutOfRange("native Gather cannot index an empty axis"));
  }
  status = validate_indices(plan, indices, indices_desc, axis_extent);
  if (!status.ok()) {
    indices_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    return completed(std::move(status));
  }

  auto output_result = output_storage.map_host(mem::HostAccess::kWrite);
  if (!output_result) {
    indices_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    indices_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  std::byte *output = writable->data();
  if (!aligned_for_storage(output, output_desc.required_alignment())) {
    output_mapping.close().IgnoreError();
    indices_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU Gather output mapping does not satisfy tensor alignment"));
  }

  const std::size_t element_size = data_type_size(source_desc.data_type());
  if (source_desc.is_dense()) {
    const auto source_dimensions = source_desc.dimensions();
    const std::size_t prefix = product(source_dimensions, 0, plan.axis);
    const std::size_t suffix =
        product(source_dimensions, plan.axis + 1, source_desc.rank());
    const std::size_t index_count = indices_desc.element_count().value();
    const std::size_t block_bytes = suffix * element_size;
    for (std::size_t outer = 0; outer < prefix; ++outer) {
      for (std::size_t item = 0; item < index_count; ++item) {
        const std::size_t selected =
            static_cast<std::size_t>(read_index(indices, indices_desc, item));
        const std::size_t source_element =
            (outer * axis_extent + selected) * suffix;
        const std::size_t output_element =
            (outer * index_count + item) * suffix;
        std::memcpy(output + output_element * element_size,
                    source + source_element * element_size, block_bytes);
      }
    }
  } else {
    std::array<std::size_t, kMaxTensorRank> coordinate{};
    for (std::size_t linear = 0; linear < *output_elements; ++linear) {
      coordinates(output_desc.dimensions(), linear, coordinate);
      const std::size_t index_offset =
          index_offset_from_output(indices_desc, plan.axis, coordinate);
      const std::int64_t selected = read_index_at_offset(
          indices, indices_desc.data_type(), index_offset);
      const std::size_t source_offset = indexed_source_offset(
          source_desc, indices_desc, plan.axis, coordinate, selected);
      std::memcpy(output + linear * element_size, source + source_offset,
                  element_size);
    }
  }

  Status terminal = Status::Ok();
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, indices_mapping.close());
  preserve_first_error(terminal, source_mapping.close());
  return completed(std::move(terminal));
}

detail::EnqueueResult
submit_scatter(const NativePlan &plan,
               std::span<const ConstTensorView> inputs,
               std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 3 || outputs.size() != 1)
    return completed(Status::Internal("native Scatter plan is invalid"));

  const TensorDesc &source_desc = inputs[0].description();
  const TensorDesc &indices_desc = inputs[1].description();
  const TensorDesc &updates_desc = inputs[2].description();
  const TensorDesc &output_desc = outputs[0].description();
  auto output_elements = output_desc.element_count();
  auto update_elements = updates_desc.element_count();
  if (!output_elements)
    return completed(std::move(output_elements).error());
  if (!update_elements)
    return completed(std::move(update_elements).error());
  if (*output_elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView source_storage = inputs[0].storage();
  const mem::ConstBufferView indices_storage = inputs[1].storage();
  const mem::ConstBufferView updates_storage = inputs[2].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place = same_storage(source_storage, output_storage);

  Status status = invalidate_for_read(indices_storage);
  if (!status.ok())
    return completed(std::move(status));
  status = invalidate_for_read(updates_storage);
  if (!status.ok())
    return completed(std::move(status));
  // Exact input0/output alias still reads the old source value. A non-coherent
  // resource therefore needs the same explicit invalidate as an out-of-place
  // source before the read-write mapping is opened.
  status = invalidate_for_read(source_storage);
  if (!status.ok())
    return completed(std::move(status));

  auto indices_result = indices_storage.map_host();
  if (!indices_result)
    return completed(std::move(indices_result).error());
  mem::HostMapping indices_mapping = std::move(*indices_result);
  const std::byte *indices = indices_mapping.bytes().data();
  const std::size_t axis_extent =
      static_cast<std::size_t>(source_desc.dimensions()[plan.axis]);
  if (axis_extent == 0 && *update_elements != 0) {
    indices_mapping.close().IgnoreError();
    return completed(
        Status::OutOfRange("native Scatter cannot index an empty axis"));
  }
  status = validate_indices(plan, indices, indices_desc, axis_extent);
  if (!status.ok()) {
    indices_mapping.close().IgnoreError();
    return completed(std::move(status));
  }

  auto updates_result = updates_storage.map_host();
  if (!updates_result) {
    indices_mapping.close().IgnoreError();
    return completed(std::move(updates_result).error());
  }
  mem::HostMapping updates_mapping = std::move(*updates_result);
  const std::byte *updates = updates_mapping.bytes().data();

  mem::HostMapping source_mapping;
  const std::byte *source = nullptr;
  if (!in_place) {
    auto source_result = source_storage.map_host();
    if (!source_result) {
      updates_mapping.close().IgnoreError();
      indices_mapping.close().IgnoreError();
      return completed(std::move(source_result).error());
    }
    source_mapping = std::move(*source_result);
    source = source_mapping.bytes().data();
  }

  auto output_result = output_storage.map_host(
      in_place ? mem::HostAccess::kReadWrite : mem::HostAccess::kWrite);
  if (!output_result) {
    source_mapping.close().IgnoreError();
    updates_mapping.close().IgnoreError();
    indices_mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    updates_mapping.close().IgnoreError();
    indices_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  std::byte *output = writable->data();
  if (in_place)
    source = output;
  if (!aligned_for_storage(indices, indices_desc.required_alignment()) ||
      !aligned_for_storage(updates, updates_desc.required_alignment()) ||
      !aligned_for_storage(source, source_desc.required_alignment()) ||
      !aligned_for_storage(output, output_desc.required_alignment())) {
    output_mapping.close().IgnoreError();
    source_mapping.close().IgnoreError();
    updates_mapping.close().IgnoreError();
    indices_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU Scatter mapping does not satisfy tensor alignment"));
  }

  const std::size_t element_size = data_type_size(source_desc.data_type());
  if (!in_place) {
    if (source_desc.is_dense()) {
      std::memcpy(output, source, *output_elements * element_size);
    } else {
      for (std::size_t linear = 0; linear < *output_elements; ++linear) {
        std::memcpy(output + linear * element_size,
                    source + logical_byte_offset(source_desc, linear),
                    element_size);
      }
    }
  }

  std::array<std::size_t, kMaxTensorRank> coordinate{};
  for (std::size_t linear = 0; linear < *update_elements; ++linear) {
    coordinates(updates_desc.dimensions(), linear, coordinate);
    const std::size_t index_offset =
        index_offset_from_output(indices_desc, plan.axis, coordinate);
    const std::int64_t selected = read_index_at_offset(
        indices, indices_desc.data_type(), index_offset);
    const std::size_t output_offset = indexed_source_offset(
        output_desc, indices_desc, plan.axis, coordinate, selected);
    const std::size_t update_offset =
        logical_byte_offset(updates_desc, linear);
    if (plan.scatter_kind == ScatterKind::kAssign) {
      std::memcpy(output + output_offset, updates + update_offset,
                  element_size);
    } else {
      float destination = 0.0F;
      float addition = 0.0F;
      std::memcpy(&destination, output + output_offset, sizeof(destination));
      std::memcpy(&addition, updates + update_offset, sizeof(addition));
      destination += addition;
      std::memcpy(output + output_offset, &destination, sizeof(destination));
    }
  }

  Status terminal = Status::Ok();
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, source_mapping.close());
  preserve_first_error(terminal, updates_mapping.close());
  preserve_first_error(terminal, indices_mapping.close());
  return completed(std::move(terminal));
}

detail::EnqueueResult
submit_concat(const NativePlan &plan,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  if (inputs.empty() || outputs.size() != 1)
    return completed(Status::Internal("native Concat plan is invalid"));
  const TensorDesc &output_desc = outputs[0].description();
  auto output_elements = output_desc.element_count();
  if (!output_elements)
    return completed(std::move(output_elements).error());
  if (*output_elements == 0)
    return completed(Status::Ok());

  const mem::BufferView output_storage = outputs[0].storage();
  auto output_result = output_storage.map_host(mem::HostAccess::kWrite);
  if (!output_result)
    return completed(std::move(output_result).error());
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  std::byte *output = writable->data();
  if (!aligned_for_storage(output, output_desc.required_alignment())) {
    output_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU Concat output mapping does not satisfy tensor alignment"));
  }

  const std::size_t element_size = data_type_size(output_desc.data_type());
  const std::size_t output_axis =
      static_cast<std::size_t>(output_desc.dimensions()[plan.axis]);
  const std::size_t prefix = product(output_desc.dimensions(), 0, plan.axis);
  const std::size_t suffix = product(output_desc.dimensions(), plan.axis + 1,
                                     output_desc.rank());
  std::size_t axis_base = 0;
  Status terminal = Status::Ok();
  for (const ConstTensorView &input_view : inputs) {
    const TensorDesc &input_desc = input_view.description();
    const mem::ConstBufferView input_storage = input_view.storage();
    terminal = invalidate_for_read(input_storage);
    if (!terminal.ok())
      break;
    auto input_result = input_storage.map_host();
    if (!input_result) {
      terminal = std::move(input_result).error();
      break;
    }
    mem::HostMapping input_mapping = std::move(*input_result);
    const std::byte *input = input_mapping.bytes().data();
    if (!aligned_for_storage(input, input_desc.required_alignment())) {
      terminal = Status::FailedPrecondition(
          "CPU Concat input mapping does not satisfy tensor alignment");
    } else {
      const std::size_t input_axis =
          static_cast<std::size_t>(input_desc.dimensions()[plan.axis]);
      if (input_desc.is_dense()) {
        const std::size_t chunk_elements = input_axis * suffix;
        for (std::size_t outer = 0; outer < prefix; ++outer) {
          const std::size_t source_element = outer * chunk_elements;
          const std::size_t output_element =
              (outer * output_axis + axis_base) * suffix;
          std::memcpy(output + output_element * element_size,
                      input + source_element * element_size,
                      chunk_elements * element_size);
        }
      } else {
        auto input_elements = input_desc.element_count();
        if (!input_elements) {
          terminal = std::move(input_elements).error();
        } else {
          std::array<std::size_t, kMaxTensorRank> coordinate{};
          const auto output_strides = output_desc.byte_strides();
          for (std::size_t linear = 0; linear < *input_elements; ++linear) {
            coordinates(input_desc.dimensions(), linear, coordinate);
            std::size_t output_offset = 0;
            for (std::size_t axis = 0; axis < input_desc.rank(); ++axis) {
              const std::size_t value =
                  coordinate[axis] + (axis == plan.axis ? axis_base : 0);
              output_offset +=
                  value * static_cast<std::size_t>(output_strides[axis]);
            }
            std::memcpy(output + output_offset,
                        input + logical_byte_offset(input_desc, linear),
                        element_size);
          }
        }
      }
      axis_base += input_axis;
    }
    preserve_first_error(terminal, input_mapping.close());
    if (!terminal.ok())
      break;
  }

  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  return completed(std::move(terminal));
}

} // namespace

Status prepare_indexing(Operation operation,
                        std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs,
                        NativePlan &plan) noexcept {
  switch (operation.kind()) {
  case OperationKind::kGather: {
    if (inputs.size() != 2 || outputs.size() != 1)
      return Status::Internal("native Gather arity is invalid");
    Status status = require_plain_scalar(inputs[0]);
    if (!status.ok())
      return status;
    status = require_plain_scalar(inputs[1]);
    if (!status.ok())
      return status;
    status = require_plain_scalar(outputs[0]);
    if (!status.ok() || !outputs[0].is_dense())
      return status.ok()
                 ? Status::Unimplemented(
                       "native Gather requires a dense output tensor")
                 : status;
    const Gather *gather = operation.get_if<Gather>();
    auto axis = normalized_axis(gather->axis, inputs[0].rank());
    if (!axis)
      return std::move(axis).error();
    plan.axis = *axis;
    plan.index_bounds = gather->bounds;
    plan.submit_route = &submit_gather;
    return Status::Ok();
  }
  case OperationKind::kScatter: {
    if (inputs.size() != 3 || outputs.size() != 1)
      return Status::Internal("native Scatter arity is invalid");
    for (const TensorDesc &input : inputs) {
      Status status = require_plain_scalar(input);
      if (!status.ok())
        return status;
    }
    Status status = require_plain_scalar(outputs[0]);
    if (!status.ok() || !outputs[0].is_dense())
      return status.ok()
                 ? Status::Unimplemented(
                       "native Scatter requires a dense output tensor")
                 : status;
    const Scatter *scatter = operation.get_if<Scatter>();
    if (scatter->kind == ScatterKind::kAdd &&
        inputs[0].data_type() != DataType::kFloat32) {
      return Status::Unimplemented(
          "native Scatter add currently supports float32 tensors");
    }
    auto axis = normalized_axis(scatter->axis, inputs[0].rank());
    if (!axis)
      return std::move(axis).error();
    plan.axis = *axis;
    plan.index_bounds = scatter->bounds;
    plan.scatter_kind = scatter->kind;
    if (inputs[0] == outputs[0])
      plan.alias_input_mask = 1;
    plan.submit_route = &submit_scatter;
    return Status::Ok();
  }
  case OperationKind::kConcat: {
    if (inputs.empty() || outputs.size() != 1)
      return Status::Internal("native Concat arity is invalid");
    for (const TensorDesc &input : inputs) {
      Status status = require_plain_scalar(input);
      if (!status.ok())
        return status;
    }
    Status status = require_plain_scalar(outputs[0]);
    if (!status.ok() || !outputs[0].is_dense())
      return status.ok()
                 ? Status::Unimplemented(
                       "native Concat requires a dense output tensor")
                 : status;
    auto axis = normalized_axis(operation.get_if<Concat>()->axis,
                                inputs[0].rank());
    if (!axis)
      return std::move(axis).error();
    plan.axis = *axis;
    plan.submit_route = &submit_concat;
    return Status::Ok();
  }
  default:
    return Status::Unimplemented("operation is not native indexing");
  }
}

} // namespace hyle::compute::native::internal
