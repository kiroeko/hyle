#include "native_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/ops.h"
#include "hyle/simd/search.h"

namespace hyle::compute::native::internal {
namespace {

float reduction_identity(ReductionKind kind) noexcept {
  switch (kind) {
  case ReductionKind::kSum:
  case ReductionKind::kMean:
    return 0.0F;
  case ReductionKind::kMinimum:
    return std::numeric_limits<float>::infinity();
  case ReductionKind::kMaximum:
    return -std::numeric_limits<float>::infinity();
  }
  return std::numeric_limits<float>::quiet_NaN();
}

void accumulate(ReductionKind kind, float value, float &result) noexcept {
  switch (kind) {
  case ReductionKind::kSum:
  case ReductionKind::kMean:
    result += value;
    return;
  case ReductionKind::kMinimum:
    if (!std::isnan(result))
      result = std::isnan(value) ? value : std::fmin(result, value);
    return;
  case ReductionKind::kMaximum:
    if (!std::isnan(result))
      result = std::isnan(value) ? value : std::fmax(result, value);
    return;
  }
}

void finish_mean(float *output, std::size_t count,
                 std::size_t reduced_elements) noexcept {
  if (reduced_elements == 0) {
    std::fill_n(output, count, std::numeric_limits<float>::quiet_NaN());
    return;
  }
  const float divisor = static_cast<float>(reduced_elements);
  for (std::size_t index = 0; index < count; ++index)
    output[index] /= divisor;
}

float reduce_dense(ReductionKind kind, const float *input,
                   std::size_t count) noexcept {
  switch (kind) {
  case ReductionKind::kSum:
  case ReductionKind::kMean:
    return simd::sum(input, count);
  case ReductionKind::kMinimum:
    return simd::min_value(input, count);
  case ReductionKind::kMaximum:
    return simd::max_value(input, count);
  }
  return std::numeric_limits<float>::quiet_NaN();
}

detail::EnqueueResult
submit_reduction(const NativePlan &plan,
                 std::span<const ConstTensorView> inputs,
                 std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("native reduction plan is invalid"));
  const std::size_t input_elements = plan.parameters[0];
  const std::size_t output_elements = plan.parameters[1];
  const std::size_t reduced_elements = plan.parameters[2];
  if (output_elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView input_storage = inputs[0].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  if (input_elements != 0 && !input_storage.resource().properties().has(
                                 mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = input_storage.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }

  mem::HostMapping input_mapping;
  const std::byte *input_data = nullptr;
  if (input_elements != 0) {
    auto mapping = input_storage.map_host();
    if (!mapping)
      return completed(std::move(mapping).error());
    input_mapping = std::move(*mapping);
    input_data = input_mapping.bytes().data();
  }
  auto output_result = output_storage.map_host(mem::HostAccess::kWrite);
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
  auto *output_data = reinterpret_cast<float *>(writable->data());
  if (!aligned_for_storage(output_data,
                           outputs[0].description().required_alignment()) ||
      (input_elements != 0 &&
       !aligned_for_storage(input_data,
                            inputs[0].description().required_alignment()))) {
    output_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU reduction mapping does not satisfy tensor alignment"));
  }

  std::fill_n(output_data, output_elements,
              reduction_identity(plan.reduction_kind));
  const auto *dense_input = reinterpret_cast<const float *>(input_data);
  if (plan.reduction_route == ReductionRoute::kAllDense) {
    output_data[0] =
        reduce_dense(plan.reduction_kind, dense_input, input_elements);
  } else if (plan.reduction_route == ReductionRoute::kLastAxisDense) {
    for (std::size_t row = 0; row < output_elements; ++row) {
      const float *row_input =
          dense_input == nullptr ? nullptr
                                 : dense_input + row * reduced_elements;
      output_data[row] =
          reduce_dense(plan.reduction_kind, row_input, reduced_elements);
    }
  } else {
    const TensorDesc &input_description = inputs[0].description();
    const auto dimensions = input_description.dimensions();
    const auto strides = input_description.byte_strides();
    for (std::size_t linear = 0; linear < input_elements; ++linear) {
      std::size_t remaining = linear;
      std::size_t input_offset = 0;
      std::size_t output_index = 0;
      std::size_t output_stride = 1;
      for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
        const std::size_t axis = dimensions.size() - 1 - reverse;
        const std::size_t extent = static_cast<std::size_t>(dimensions[axis]);
        const std::size_t coordinate = remaining % extent;
        remaining /= extent;
        input_offset += coordinate * static_cast<std::size_t>(strides[axis]);
        if ((plan.reduction_axes_mask & (std::uint8_t{1} << axis)) == 0) {
          output_index += coordinate * output_stride;
          output_stride *= extent;
        }
      }
      float value = 0.0F;
      std::memcpy(&value, input_data + input_offset, sizeof(value));
      accumulate(plan.reduction_kind, value, output_data[output_index]);
    }
  }

  if (plan.reduction_kind == ReductionKind::kMean)
    finish_mean(output_data, output_elements, reduced_elements);

  Status terminal = Status::Ok();
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return completed(std::move(terminal));
}

std::uint8_t reduction_axes_mask(const Reduce &reduce,
                                 std::size_t rank) noexcept {
  if (reduce.axes.is_all()) {
    return rank == kMaxTensorRank
               ? std::numeric_limits<std::uint8_t>::max()
               : static_cast<std::uint8_t>((std::uint16_t{1} << rank) - 1);
  }
  std::uint8_t result = 0;
  for (const std::int8_t requested : reduce.axes.values()) {
    const std::int32_t axis =
        requested < 0 ? static_cast<std::int32_t>(rank) + requested : requested;
    result = static_cast<std::uint8_t>(result | (std::uint8_t{1} << axis));
  }
  return result;
}

std::uint8_t all_axes_mask(std::size_t rank) noexcept {
  return rank == kMaxTensorRank
             ? std::numeric_limits<std::uint8_t>::max()
             : static_cast<std::uint8_t>((std::uint16_t{1} << rank) - 1);
}

} // namespace

Status prepare_reduction(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return Status::Internal("native Reduce arity is invalid");
  if (inputs[0].data_type() != DataType::kFloat32 ||
      outputs[0].data_type() != DataType::kFloat32 || !inputs[0].is_plain() ||
      !outputs[0].is_plain() || !outputs[0].is_dense()) {
    return Status::Unimplemented(
        "native Reduce currently requires plain float32 and dense output");
  }
  const Reduce *reduce = operation.get_if<Reduce>();
  if (reduce == nullptr)
    return Status::Internal("native Reduce descriptor is missing");
  if (reduce->accumulation_type != DataType::kUnknown &&
      reduce->accumulation_type != DataType::kFloat32) {
    return Status::Unimplemented(
        "native Reduce currently supports float32 accumulation");
  }

  auto input_elements = inputs[0].element_count();
  if (!input_elements)
    return std::move(input_elements).error();
  auto output_elements = outputs[0].element_count();
  if (!output_elements)
    return std::move(output_elements).error();
  const std::uint8_t axes = reduction_axes_mask(*reduce, inputs[0].rank());
  std::size_t reduced_elements = 1;
  for (std::size_t axis = 0; axis < inputs[0].rank(); ++axis) {
    if ((axes & (std::uint8_t{1} << axis)) == 0)
      continue;
    auto extent =
        math::try_integer_cast<std::size_t>(inputs[0].dimensions()[axis]);
    if (!extent)
      return Status::OutOfRange("reduction extent exceeds size_t");
    auto next = math::try_multiply(reduced_elements, *extent);
    if (!next)
      return Status::OutOfRange("reduction extent product overflows size_t");
    reduced_elements = *next;
  }

  ReductionRoute route = ReductionRoute::kGeneral;
  if (inputs[0].is_dense()) {
    const std::uint8_t all_axes = all_axes_mask(inputs[0].rank());
    if (axes == all_axes) {
      route = ReductionRoute::kAllDense;
    } else if (inputs[0].rank() != 0 &&
               axes == (std::uint8_t{1} << (inputs[0].rank() - 1))) {
      route = ReductionRoute::kLastAxisDense;
    }
  }

  plan.parameters = {*input_elements, *output_elements, reduced_elements};
  plan.reduction_kind = reduce->kind;
  plan.reduction_route = route;
  plan.reduction_axes_mask = axes;
  plan.submit_route = &submit_reduction;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
