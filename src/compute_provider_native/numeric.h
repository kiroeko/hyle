#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"

namespace hyle::compute::native::internal::numeric {

template <class Value> float decode(Value value) noexcept {
  if constexpr (std::is_same_v<Value, float>) {
    return value;
  } else if constexpr (std::is_same_v<Value, simd::float16_t>) {
    return simd::f16_to_f32(value);
  } else {
    static_assert(std::is_same_v<Value, simd::bfloat16_t>);
    return simd::bf16_to_f32(value);
  }
}

template <class Value> Value encode(float value) noexcept {
  if constexpr (std::is_same_v<Value, float>) {
    return value;
  } else if constexpr (std::is_same_v<Value, simd::float16_t>) {
    return simd::f32_to_f16(value);
  } else {
    static_assert(std::is_same_v<Value, simd::bfloat16_t>);
    return simd::f32_to_bf16(value);
  }
}

// A legal byte stride need not preserve the scalar's natural alignment.
template <class Value> float load(const std::byte *source) noexcept {
  Value value{};
  std::memcpy(&value, source, sizeof(value));
  return decode(value);
}

template <class Value>
void store(std::byte *destination, float value) noexcept {
  const Value result = encode<Value>(value);
  std::memcpy(destination, &result, sizeof(result));
}

// Dense callers have already checked storage alignment. Only a bounded tile
// is decoded; weights and activations keep their declared storage outside it.
template <class Value>
void load_tile(const void *source, std::size_t offset, float *destination,
               std::size_t count) noexcept {
  const auto *values = static_cast<const Value *>(source) + offset;
  if constexpr (std::is_same_v<Value, float>)
    std::copy_n(values, count, destination);
  else
    simd::convert(values, destination, count);
}

template <class Value>
void store_tile(const float *source, void *destination, std::size_t offset,
                std::size_t count) noexcept {
  auto *values = static_cast<Value *>(destination) + offset;
  if constexpr (std::is_same_v<Value, float>)
    std::copy_n(source, count, values);
  else
    simd::convert(source, values, count);
}

} // namespace hyle::compute::native::internal::numeric
