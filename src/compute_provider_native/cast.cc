#include "native_plan.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/mem/resource.h"
#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"

namespace hyle::compute::native::internal {
namespace {

// The full I32 domain fits F32's finite range. Form its nearest-even IEEE
// significand directly so this Cast does not inherit the caller's FP mode.
float int32_to_float32(std::int32_t value) noexcept {
  const auto raw = static_cast<std::uint32_t>(value);
  const auto sign = raw & 0x80000000u;
  const auto magnitude = value < 0 ? 0u - raw : raw; // Defined for INT32_MIN.
  if (magnitude == 0) return 0.F;
  auto exponent = 31u - static_cast<unsigned>(std::countl_zero(magnitude));
  std::uint32_t significand;
  if (exponent <= 23) {
    significand = magnitude << (23 - exponent);
  } else {
    const auto shift = exponent - 23;
    significand = magnitude >> shift;
    const auto remainder = magnitude & ((1u << shift) - 1);
    const auto halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (significand & 1u)))
      ++significand;
    if (significand == (1u << 24)) { significand >>= 1; ++exponent; }
  }
  return std::bit_cast<float>(sign | ((exponent + 127) << 23) |
                              (significand & 0x7fffffu));
}

template <class Value> float decode(Value value) noexcept {
  if constexpr (std::is_same_v<Value, float>) {
    return value;
  } else if constexpr (std::is_same_v<Value, simd::float16_t>) {
    return simd::f16_to_f32(value);
  } else if constexpr (std::is_same_v<Value, simd::bfloat16_t>) {
    return simd::bf16_to_f32(value);
  } else if constexpr (std::is_same_v<Value, simd::float8_e4m3_t>) {
    return simd::f8e4m3_to_f32(value);
  } else {
    static_assert(std::is_same_v<Value, simd::float8_e5m2_t>);
    return simd::f8e5m2_to_f32(value);
  }
}

template <class Value> Value encode(float value) noexcept {
  if constexpr (std::is_same_v<Value, float>) {
    return value;
  } else if constexpr (std::is_same_v<Value, simd::float16_t>) {
    return simd::f32_to_f16(value);
  } else if constexpr (std::is_same_v<Value, simd::bfloat16_t>) {
    return simd::f32_to_bf16(value);
  } else if constexpr (std::is_same_v<Value, simd::float8_e4m3_t>) {
    return simd::f32_to_f8e4m3(value);
  } else {
    static_assert(std::is_same_v<Value, simd::float8_e5m2_t>);
    return simd::f32_to_f8e5m2(value);
  }
}

template <class Source, class Destination>
Destination convert_value(Source input) noexcept {
  if constexpr (std::is_same_v<Source, std::int32_t>) {
    static_assert(std::is_same_v<Destination, float>);
    return int32_to_float32(input);
  } else if constexpr (std::is_same_v<Destination, std::uint8_t>) {
    // Bool is truth, not truncation to an integer byte. Inspect floating storage
    // directly so even subnormals remain nonzero under host DAZ/FTZ modes.
    using Bits = std::conditional_t<sizeof(Source) == 4, std::uint32_t, std::uint16_t>;
    Bits bits = 0;
    std::memcpy(&bits, &input, sizeof(input));
    constexpr auto magnitude = sizeof(Source) == 4 ? 0x7fffffffu : 0x7fffu;
    return static_cast<std::uint8_t>((bits & magnitude) != 0);
  } else if constexpr (std::is_same_v<Source, std::uint8_t>) {
    return encode<Destination>(input != 0 ? 1.F : 0.F);
  } else {
    return encode<Destination>(decode(input));
  }
}

template <class Source, class Destination>
void convert_linear(const void *source, const void *, const void *,
                    void *destination, std::size_t count) noexcept {
  if constexpr (std::is_same_v<Source, std::uint8_t> ||
                std::is_same_v<Destination, std::uint8_t> ||
                std::is_same_v<Source, std::int32_t>) {
    const auto *input = static_cast<const Source *>(source);
    auto *output = static_cast<Destination *>(destination);
    for (std::size_t i = 0; i < count; ++i)
      output[i] = convert_value<Source, Destination>(input[i]);
  } else if constexpr (std::is_same_v<Source, Destination>) {
    std::copy_n(static_cast<const Source *>(source), count,
                static_cast<Destination *>(destination));
  } else {
    simd::convert(static_cast<const Source *>(source),
                  static_cast<Destination *>(destination), count);
  }
}

template <class Source, class Destination>
void convert_scalar(const std::byte *source, const std::byte *,
                    const std::byte *, std::byte *destination) noexcept {
  // Public byte strides need not preserve natural C++ alignment. Keep the
  // reference materializer defined for those legitimate layouts.
  Source input{};
  std::memcpy(&input, source, sizeof(input));
  const Destination output = convert_value<Source, Destination>(input);
  std::memcpy(destination, &output, sizeof(output));
}

template <class Source, class Destination>
void select_pair(NativePlan &plan) noexcept {
  plan.linear_elementwise_route = &convert_linear<Source, Destination>;
  plan.scalar_elementwise_route = &convert_scalar<Source, Destination>;
}
bool select_cast(DataType source, DataType destination, NativePlan &plan) noexcept {
  if (source == DataType::kInt32 && destination == DataType::kFloat32) {
    select_pair<std::int32_t, float>(plan);
    return true;
  }
  if (source == DataType::kBool || destination == DataType::kBool) {
    if (source == destination) return true; // The original Copy keeps bytes.
    const bool to_bool = destination == DataType::kBool;
    const auto select = [&]<class Value>() {
      if (to_bool) select_pair<Value, std::uint8_t>(plan);
      else select_pair<std::uint8_t, Value>(plan);
    };
    switch (to_bool ? source : destination) {
    case DataType::kFloat32: select.template operator()<float>(); return true;
    case DataType::kFloat16: select.template operator()<simd::float16_t>(); return true;
    case DataType::kBFloat16: select.template operator()<simd::bfloat16_t>(); return true;
    default: return false;
    }
  }
  // Plain E4M3 Cast is numerical SATFINITE conversion, not an ATen storage
  // reinterpretation. Keep the original generic scalar/linear executor.
  if (destination == DataType::kFloat8E4M3FN) {
    switch (source) {
    case DataType::kFloat32: select_pair<float, simd::float8_e4m3_t>(plan); return true;
    case DataType::kFloat16: select_pair<simd::float16_t, simd::float8_e4m3_t>(plan); return true;
    case DataType::kBFloat16: select_pair<simd::bfloat16_t, simd::float8_e4m3_t>(plan); return true;
    default: break;
    }
  }
  if (source == DataType::kFloat8E4M3FN) {
    switch (destination) {
    case DataType::kFloat32: select_pair<simd::float8_e4m3_t, float>(plan); return true;
    case DataType::kFloat16: select_pair<simd::float8_e4m3_t, simd::float16_t>(plan); return true;
    case DataType::kBFloat16: select_pair<simd::float8_e4m3_t, simd::bfloat16_t>(plan); return true;
    default: break;
    }
  }
  if (source != DataType::kFloat8E5M2)
    return false;
  if (destination == DataType::kBFloat16) {
    select_pair<simd::float8_e5m2_t, simd::bfloat16_t>(plan);
    return true;
  }
  if (destination == DataType::kFloat8E4M3FN) {
    select_pair<simd::float8_e5m2_t, simd::float8_e4m3_t>(plan);
    return true;
  }
  return destination == DataType::kFloat8E5M2;
}

detail::EnqueueResult submit_cast(const NativePlan &plan,
                               std::span<const ConstTensorView> inputs,
                               std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1 ||
      plan.linear_elementwise_route == nullptr ||
      plan.scalar_elementwise_route == nullptr) {
    return completed(Status::Internal("native cast plan is invalid"));
  }
  const std::size_t elements = plan.parameters[0];
  if (elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView input = inputs[0].storage();
  const mem::BufferView output = outputs[0].storage();
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
  if (!aligned_for_storage(source,
                           inputs[0].description().required_alignment()) ||
      !aligned_for_storage(destination,
                           outputs[0].description().required_alignment())) {
    output_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU cast mapping does not satisfy tensor alignment"));
  }

  if (plan.parameters[1] != 0) {
    plan.linear_elementwise_route(source, nullptr, nullptr, destination,
                                  elements);
  } else {
    const TensorDesc &description = inputs[0].description();
    const std::size_t destination_size =
        data_type_size(outputs[0].description().data_type());
    for (std::size_t linear = 0; linear < elements; ++linear) {
      plan.scalar_elementwise_route(
          source + logical_byte_offset(description, linear), nullptr, nullptr,
          destination + linear * destination_size);
    }
  }

  Status terminal = Status::Ok();
  if (!output_properties.has(mem::MemoryProperty::kHostDeviceCoherent))
    preserve_first_error(terminal, output_mapping.flush());
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return completed(std::move(terminal));
}

} // namespace

Status prepare_cast(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    NativePlan &plan) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1 || !inputs[0].is_plain() ||
      !outputs[0].is_plain() || !outputs[0].is_dense()) {
    return Status::Unimplemented(
        "native Cast requires plain input and dense plain output");
  }
  const Cast *cast = operation.get_if<Cast>();
  if (cast == nullptr)
    return Status::Internal("native Cast descriptor is missing");
  if (!select_cast(inputs[0].data_type(), cast->output_type, plan)) {
    return Status::Unimplemented(
        "native Cast storage conversion pair is not admitted");
  }
  if (inputs[0].data_type() == cast->output_type)
    return prepare_transform(Operation(Copy{}), inputs, outputs, plan);
  auto elements = outputs[0].element_count();
  if (!elements)
    return std::move(elements).error();
  plan.parameters[0] = *elements;
  plan.parameters[1] = inputs[0].is_dense() ? 1 : 0;
  plan.submit_route = &submit_cast;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
