#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"

namespace hyle::compute::native::internal {
namespace {

Status invalidate_for_read(mem::ConstBufferView storage) noexcept {
  if (storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    return Status::Ok();
  }
  return storage.invalidate();
}

bool same_storage(mem::ConstBufferView left, mem::BufferView right) noexcept {
  return left.resource() == right.resource() &&
         left.native_handle() == right.native_handle() &&
         left.offset() == right.offset() && left.size() == right.size();
}

std::size_t product(std::span<const std::int64_t> dimensions, std::size_t first,
                    std::size_t last) noexcept {
  std::size_t result = 1;
  for (std::size_t axis = first; axis < last; ++axis)
    result *= static_cast<std::size_t>(dimensions[axis]);
  return result;
}

template <class Value>
float masked_softmax_value(const NativePlan &plan, const std::byte *input,
                           const TensorDesc &input_desc, const std::byte *mask,
                           const TensorDesc *mask_desc,
                           std::size_t logical) noexcept {
  float value = numeric::load<Value>(
      input + logical_byte_offset(input_desc, logical));
  value *= plan.scalar;
  const std::size_t mask_offset = logical_broadcast_byte_offset(
      *mask_desc, input_desc.dimensions(), logical);
  return mask[mask_offset] == std::byte{0}
             ? -std::numeric_limits<float>::infinity()
             : value;
}

template <class Value>
detail::EnqueueResult submit_softmax_impl(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs, mem::BufferView workspace) noexcept {
  if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1)
    return completed(Status::Internal("native Softmax plan is invalid"));
  const TensorDesc &input_desc = inputs[0].description();
  const TensorDesc &output_desc = outputs[0].description();
  const TensorDesc *mask_desc =
      inputs.size() == 2 ? &inputs[1].description() : nullptr;
  auto elements = output_desc.element_count();
  if (!elements)
    return completed(std::move(elements).error());
  if (*elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView input_storage = inputs[0].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place = same_storage(input_storage, output_storage);
  Status status = invalidate_for_read(input_storage);
  if (!status.ok())
    return completed(std::move(status));
  mem::ConstBufferView mask_storage;
  if (inputs.size() == 2) {
    mask_storage = inputs[1].storage();
    status = invalidate_for_read(mask_storage);
    if (!status.ok())
      return completed(std::move(status));
  }

  mem::HostMapping input_mapping;
  const std::byte *input = nullptr;
  if (!in_place) {
    auto input_result = input_storage.map_host();
    if (!input_result)
      return completed(std::move(input_result).error());
    input_mapping = std::move(*input_result);
    input = input_mapping.bytes().data();
  }
  mem::HostMapping mask_mapping;
  const std::byte *mask = nullptr;
  if (inputs.size() == 2) {
    auto mask_result = mask_storage.map_host();
    if (!mask_result) {
      input_mapping.close().IgnoreError();
      return completed(std::move(mask_result).error());
    }
    mask_mapping = std::move(*mask_result);
    mask = mask_mapping.bytes().data();
  }
  auto output_result = output_storage.map_host(
      in_place ? mem::HostAccess::kReadWrite : mem::HostAccess::kWrite);
  if (!output_result) {
    mask_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    mask_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  std::byte *output = writable->data();
  if (in_place)
    input = output;
  if (!aligned_for_storage(input, input_desc.required_alignment()) ||
      !aligned_for_storage(output, output_desc.required_alignment()) ||
      (mask_desc != nullptr &&
       !aligned_for_storage(mask, mask_desc->required_alignment()))) {
    output_mapping.close().IgnoreError();
    mask_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU Softmax mapping does not satisfy tensor alignment"));
  }

  const auto dimensions = input_desc.dimensions();
  const std::size_t axis_extent =
      static_cast<std::size_t>(dimensions[plan.axis]);
  const std::size_t outer = product(dimensions, 0, plan.axis);
  const std::size_t inner =
      product(dimensions, plan.axis + 1, dimensions.size());
  Status terminal = Status::Ok();

    auto scratch_result = workspace.map_host(mem::HostAccess::kWrite);
    if (!scratch_result) {
      output_mapping.close().IgnoreError();
      mask_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed(std::move(scratch_result).error());
    }
    mem::HostMapping scratch_mapping = std::move(*scratch_result);
    auto scratch_bytes = scratch_mapping.writable_bytes();
    if (!scratch_bytes ||
        !aligned_for_storage(scratch_bytes->data(), alignof(float))) {
      scratch_mapping.close().IgnoreError();
      output_mapping.close().IgnoreError();
      mask_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed(Status::FailedPrecondition(
          "CPU Softmax workspace mapping is invalid"));
    }
    auto *row_scratch = reinterpret_cast<float *>(scratch_bytes->data());
    for (std::size_t row = 0; row < outer; ++row) {
      for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
        if (input_desc.is_dense() && inner == 1 && mask_desc == nullptr &&
            plan.scalar == 1.0F) {
          numeric::load_tile<Value>(input, row * axis_extent, row_scratch,
                                    axis_extent);
        } else {
          for (std::size_t lane = 0; lane < axis_extent; ++lane) {
            const std::size_t logical =
                (row * axis_extent + lane) * inner + inner_index;
            row_scratch[lane] = masked_softmax_value<Value>(
                plan, input, input_desc, mask, mask_desc, logical);
          }
        }
        simd::softmax_f32(row_scratch, axis_extent);
        if (inner == 1) {
          numeric::store_tile<Value>(row_scratch, output, row * axis_extent,
                                     axis_extent);
        } else {
          for (std::size_t lane = 0; lane < axis_extent; ++lane) {
            const std::size_t logical =
                (row * axis_extent + lane) * inner + inner_index;
            numeric::store<Value>(output + logical * sizeof(Value),
                                  row_scratch[lane]);
          }
        }
      }
    }
    preserve_first_error(terminal, scratch_mapping.close());

  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, mask_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return completed(std::move(terminal));
}

} // namespace

Result<std::size_t> normalized_axis(std::int32_t requested,
                                  std::size_t rank) noexcept {
  std::int64_t axis = requested;
  if (axis < 0)
    axis += static_cast<std::int64_t>(rank);
  if (axis < 0 || axis >= static_cast<std::int64_t>(rank))
    return std::unexpected(
        Status::OutOfRange("native transformer axis is out of range"));
  return static_cast<std::size_t>(axis);
}

Status prepare_transformer(Operation operation,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           NativePlan &plan) noexcept {
  switch (operation.kind()) {
  case OperationKind::kSoftmax: {
    if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1)
      return Status::Internal("native Softmax arity is invalid");
    const DataType type = inputs[0].data_type();
    if ((type != DataType::kFloat32 && type != DataType::kFloat16 &&
         type != DataType::kBFloat16) ||
        !inputs[0].is_plain() || outputs[0].data_type() != type ||
        !outputs[0].is_plain() || !outputs[0].is_dense())
      return Status::Unimplemented("native Softmax requires plain "
                                   "float32/float16/bfloat16 and dense output");
    const Softmax *softmax = operation.get_if<Softmax>();
    if (softmax->kind != SoftmaxKind::kSoftmax || softmax->mask != MaskKind::kBoolean ||
        type != DataType::kFloat16)
      return Status::Unimplemented("native Softmax retains qualified F16 Bool softmax only");
    if (softmax->accumulation_type != DataType::kUnknown &&
        softmax->accumulation_type != DataType::kFloat32) {
      return Status::Unimplemented(
          "native Softmax currently supports float32 accumulation");
    }
    if (type != DataType::kFloat32 &&
        softmax->accumulation_type != DataType::kFloat32)
      return Status::Unimplemented(
          "native half Softmax requires float32 accumulation");
    if (softmax->mask != MaskKind::kNone) {
      if (!inputs[1].is_plain() || (softmax->mask == MaskKind::kAdditive &&
                                    inputs[1].data_type() != type)) {
        return Status::Unimplemented(
            "native Softmax requires plain Bool or matching floating masks");
      }
    }
    auto axis = normalized_axis(softmax->axis, inputs[0].rank());
    if (!axis)
      return std::move(axis).error();
    plan.axis = *axis;
    plan.scalar = softmax->scale;
    plan.mask_kind = softmax->mask;
    if (inputs[0] == outputs[0])
      plan.alias_input_mask = 1;
    {
      const auto extent =
          math::try_integer_cast<std::size_t>(inputs[0].dimensions()[*axis]);
      if (!extent)
        return Status::OutOfRange("native Softmax row exceeds size_t");
      HYLE_ASSIGN_OR_RETURN(const auto output_count,
                            outputs[0].element_count());
      const auto scratch =
          math::try_multiply(output_count == 0 ? 0 : *extent, sizeof(float));
      if (!scratch)
        return Status::OutOfRange("native Softmax row workspace overflows");
      plan.workspace = {.size = *scratch, .alignment = alignof(float)};
      plan.workspace_submit_route = &submit_softmax_impl<simd::float16_t>;
    }
    return Status::Ok();
  }
  default:
    return Status::Unimplemented("operation is not a native transformer op");
  }
}

} // namespace hyle::compute::native::internal
