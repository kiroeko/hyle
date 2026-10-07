#include "native_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/mem/resource.h"
#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"

namespace hyle::compute::native::internal {
namespace {

template <class Source, class Destination>
void convert_linear(const void *source, const void *, const void *,
                    void *destination, std::size_t count) noexcept {
  if constexpr (std::is_same_v<Source, Destination>) {
    std::copy_n(static_cast<const Source *>(source), count,
                static_cast<Destination *>(destination));
  } else {
    simd::convert(static_cast<const Source *>(source),
                  static_cast<Destination *>(destination), count);
  }
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
void convert_scalar(const std::byte *source, const std::byte *,
                    const std::byte *, std::byte *destination) noexcept {
  // Public byte strides need not preserve natural C++ alignment. Keep the
  // reference materializer defined for those legitimate layouts.
  Source input{};
  std::memcpy(&input, source, sizeof(input));
  const Destination output = encode<Destination>(decode(input));
  std::memcpy(destination, &output, sizeof(output));
}

template <class Source, class Destination>
void select_pair(NativePlan &plan) noexcept {
  plan.linear_elementwise_route = &convert_linear<Source, Destination>;
  plan.scalar_elementwise_route = &convert_scalar<Source, Destination>;
}
bool select_cast(DataType source, DataType destination, NativePlan &plan) noexcept {
  if (source == DataType::kFloat8E4M3FN && destination == DataType::kFloat16) {
    select_pair<simd::float8_e4m3_t, simd::float16_t>(plan);
    return true;
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
        "native Cast currently supports f32/f16/bf16/fp8 storage types");
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
