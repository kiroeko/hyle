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
#include "hyle/simd/ops.h"
#include "hyle/simd/scan.h"
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

template <class Value>
void reduce_narrow(const NativePlan &plan, const TensorDesc &description,
                   const std::byte *input, Value *output) noexcept {
  const std::size_t count = plan.parameters[2];
  const auto dimensions = description.dimensions();
  const auto strides = description.byte_strides();
  std::array<float, 128> tile;
  for (std::size_t result = 0; result < plan.parameters[1]; ++result) {
    float accumulator = reduction_identity(plan.reduction_kind);
    if (plan.reduction_route != ReductionRoute::kGeneral) {
      for (std::size_t begin = 0; begin < count; begin += tile.size()) {
        const std::size_t size = (std::min)(tile.size(), count - begin);
        numeric::load_tile<Value>(input, result * count + begin, tile.data(),
                                  size);
        accumulate(plan.reduction_kind,
                   reduce_dense(plan.reduction_kind, tile.data(), size),
                   accumulator);
      }
    } else {
      // Decode the unreduced coordinates once per output. Every input is
      // visited once overall; no full-input scan for each output tile.
      std::size_t remaining = result;
      std::size_t base = 0;
      for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
        const std::size_t axis = dimensions.size() - 1 - reverse;
        if ((plan.reduction_axes_mask & (std::uint8_t{1} << axis)) != 0)
          continue;
        const auto extent = static_cast<std::size_t>(dimensions[axis]);
        base += (remaining % extent) * static_cast<std::size_t>(strides[axis]);
        remaining /= extent;
      }
      for (std::size_t reduced = 0; reduced < count; ++reduced) {
        remaining = reduced;
        std::size_t offset = base;
        for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
          const std::size_t axis = dimensions.size() - 1 - reverse;
          if ((plan.reduction_axes_mask & (std::uint8_t{1} << axis)) == 0)
            continue;
          const auto extent = static_cast<std::size_t>(dimensions[axis]);
          offset +=
              (remaining % extent) * static_cast<std::size_t>(strides[axis]);
          remaining /= extent;
        }
        accumulate(plan.reduction_kind, numeric::load<Value>(input + offset),
                   accumulator);
      }
    }
    if (plan.reduction_kind == ReductionKind::kMean)
      accumulator = count == 0 ? std::numeric_limits<float>::quiet_NaN()
                               : accumulator / static_cast<float>(count);
    output[result] = numeric::encode<Value>(accumulator);
  }
}

template <class Value>
detail::EnqueueResult
submit_reduction(const NativePlan &plan,
                 std::span<const ConstTensorView> inputs,
                 std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("native reduction plan is invalid"));
  const std::size_t input_elements = plan.parameters[0];
  const std::size_t output_elements = plan.parameters[1];
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
  auto *output_data = reinterpret_cast<Value *>(writable->data());
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

  if constexpr (!std::is_same_v<Value, float>) {
    reduce_narrow(plan, inputs[0].description(), input_data, output_data);
  } else {
    const std::size_t reduced_elements = plan.parameters[2];
    std::fill_n(output_data, output_elements,
                reduction_identity(plan.reduction_kind));
    const auto *dense_input = reinterpret_cast<const float *>(input_data);
    if (plan.reduction_route == ReductionRoute::kAllDense) {
      output_data[0] =
          reduce_dense(plan.reduction_kind, dense_input, input_elements);
    } else if (plan.reduction_route == ReductionRoute::kLastAxisDense) {
      for (std::size_t row = 0; row < output_elements; ++row) {
        const float *row_input = dense_input == nullptr
                                     ? nullptr
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
  }

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

template <class Value>
Value scan_value(const std::byte *input, std::size_t offset) noexcept {
  Value value;
  std::memcpy(&value, input + offset, sizeof(value));
  return value;
}

Status check_scan_prefixes(const NativePlan &plan, const TensorDesc &input_desc,
                           const std::byte *input) noexcept {
  const auto [outer, length, inner] = plan.parameters;
  const auto stride =
      static_cast<std::size_t>(input_desc.byte_strides()[plan.axis]);
  for (std::size_t group = 0; group < outer; ++group) {
    for (std::size_t column = 0; column < inner; ++column) {
      const std::size_t base =
          logical_byte_offset(input_desc, group * length * inner + column);
      std::int32_t carry = 0;
      for (std::size_t position = 0; position < length; ++position) {
        const auto next = math::try_add(
            carry, scan_value<std::int32_t>(input, base + position * stride));
        if (!next)
          return Status::OutOfRange("Scan integer prefix exceeds int32 range");
        carry = *next;
      }
    }
  }
  return Status::Ok();
}

template <class Value, bool Exclusive>
detail::EnqueueResult
submit_scan(const NativePlan &plan, std::span<const ConstTensorView> inputs,
            std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("native Scan plan arity is invalid"));
  const auto [outer, length, inner] = plan.parameters;
  if (outer == 0 || length == 0 || inner == 0)
    return completed(Status::Ok());
  const auto &description = inputs[0].description();
  const auto source = inputs[0].storage();
  const auto destination = outputs[0].storage();
  const bool in_place = source.resource() == destination.resource() &&
                        source.native_handle() == destination.native_handle() &&
                        source.offset() == destination.offset() &&
                        source.size() == destination.size();
  if (!source.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    auto status = source.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }
  mem::HostMapping input_mapping;
  mem::HostMapping output_mapping;
  const std::byte *input = nullptr;
  std::byte *output = nullptr;
  if (in_place) {
    auto mapping = destination.map_host(mem::HostAccess::kReadWrite);
    if (!mapping)
      return completed(std::move(mapping).error());
    output_mapping = std::move(*mapping);
    auto bytes = output_mapping.writable_bytes();
    if (!bytes) {
      output_mapping.close().IgnoreError();
      return completed(std::move(bytes).error());
    }
    input = output = bytes->data();
  } else {
    auto mapping = source.map_host();
    if (!mapping)
      return completed(std::move(mapping).error());
    input_mapping = std::move(*mapping);
    input = input_mapping.bytes().data();
  }
  Status terminal = Status::Ok();
  if constexpr (std::is_same_v<Value, std::int32_t>)
    terminal = check_scan_prefixes(plan, description, input);
  // Validation covers all rows before any output write, including exact alias.
  if (!terminal.ok()) {
    output_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(std::move(terminal));
  }
  if (!in_place) {
    auto mapping = destination.map_host(mem::HostAccess::kWrite);
    if (!mapping) {
      input_mapping.close().IgnoreError();
      return completed(std::move(mapping).error());
    }
    output_mapping = std::move(*mapping);
    auto bytes = output_mapping.writable_bytes();
    if (!bytes) {
      output_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed(std::move(bytes).error());
    }
    output = bytes->data();
  }
  if (!aligned_for_storage(input, description.required_alignment()) ||
      !aligned_for_storage(output, alignof(Value))) {
    output_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(
        Status::FailedPrecondition("CPU Scan mapping alignment is invalid"));
  }
  auto *values = reinterpret_cast<Value *>(output);
  const auto stride =
      static_cast<std::size_t>(description.byte_strides()[plan.axis]);
  for (std::size_t group = 0; group < outer; ++group) {
    for (std::size_t column = 0; column < inner; ++column) {
      const std::size_t logical = group * length * inner + column;
      const std::size_t base = logical_byte_offset(description, logical);
      if (description.is_dense() && inner == 1) {
        const auto *row = reinterpret_cast<const Value *>(input + base);
        if constexpr (Exclusive)
          simd::exclusive_scan(row, values + logical, length);
        else
          simd::inclusive_scan(row, values + logical, length);
      } else {
        Value carry = 0;
        for (std::size_t position = 0; position < length; ++position) {
          const auto value = scan_value<Value>(input, base + position * stride);
          const Value next = carry + value;
          values[logical + position * inner] = Exclusive ? carry : next;
          carry = next;
        }
      }
    }
  }
  if (!destination.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent))
    preserve_first_error(terminal, output_mapping.flush());
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return completed(std::move(terminal));
}

} // namespace

Status prepare_scan(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    NativePlan &plan) noexcept {
  const auto *scan = operation.get_if<Scan>();
  if (scan == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return Status::Internal("native Scan arity is invalid");
  const auto type = inputs[0].data_type();
  if ((type != DataType::kInt32 && type != DataType::kFloat32) ||
      !inputs[0].is_plain() || outputs[0].data_type() != type ||
      !outputs[0].is_plain() || !outputs[0].is_dense())
    return Status::Unimplemented(
        "native Scan requires plain int32/float32 and dense output");
  const auto rank = inputs[0].rank();
  const auto axis = scan->axis < 0
                        ? scan->axis + static_cast<std::int32_t>(rank)
                        : scan->axis;
  if (axis < 0 || static_cast<std::size_t>(axis) >= rank)
    return Status::InvalidArgument("Scan axis is out of range");
  plan.axis = static_cast<std::size_t>(axis);
  const auto dimensions = inputs[0].dimensions();
  std::size_t outer = 1, inner = 1;
  for (std::size_t index = 0; index < rank; ++index) {
    if (index == plan.axis)
      continue;
    auto &product = index < plan.axis ? outer : inner;
    const auto extent = math::try_integer_cast<std::size_t>(dimensions[index]);
    const auto next =
        extent ? math::try_multiply(product, *extent) : std::nullopt;
    if (!next)
      return Status::OutOfRange("Scan dimension product overflows size_t");
    product = *next;
  }
  const auto length =
      math::try_integer_cast<std::size_t>(dimensions[plan.axis]);
  if (!length)
    return Status::OutOfRange("Scan axis extent exceeds size_t");
  plan.parameters = {outer, *length, inner};
  plan.alias_input_mask = inputs[0].is_dense() ? 1 : 0;
  if (type == DataType::kInt32)
    plan.submit_route = scan->exclusive ? &submit_scan<std::int32_t, true>
                                        : &submit_scan<std::int32_t, false>;
  else
    plan.submit_route = scan->exclusive ? &submit_scan<float, true>
                                        : &submit_scan<float, false>;
  return Status::Ok();
}

Status prepare_reduction(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return Status::Internal("native Reduce arity is invalid");
  const DataType type = inputs[0].data_type();
  if ((type != DataType::kFloat32 && type != DataType::kFloat16 &&
       type != DataType::kBFloat16) ||
      outputs[0].data_type() != type || !inputs[0].is_plain() ||
      !outputs[0].is_plain() || !outputs[0].is_dense()) {
    return Status::Unimplemented("native Reduce requires plain "
                                 "float32/float16/bfloat16 and dense output");
  }
  const Reduce *reduce = operation.get_if<Reduce>();
  if (reduce == nullptr)
    return Status::Internal("native Reduce descriptor is missing");
  if (reduce->accumulation_type != DataType::kUnknown &&
      reduce->accumulation_type != DataType::kFloat32) {
    return Status::Unimplemented(
        "native Reduce currently supports float32 accumulation");
  }
  if (type != DataType::kFloat32 &&
      reduce->accumulation_type != DataType::kFloat32) {
    return Status::Unimplemented(
        "native half Reduce requires explicit float32 accumulation");
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
  plan.submit_route =
      type == DataType::kFloat16    ? &submit_reduction<simd::float16_t>
      : type == DataType::kBFloat16 ? &submit_reduction<simd::bfloat16_t>
                                    : &submit_reduction<float>;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
