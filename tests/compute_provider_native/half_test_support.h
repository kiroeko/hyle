#pragma once

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/dtype.h"

namespace half_test {

template <class Value> constexpr hyle::compute::DataType data_type() {
  if constexpr (std::is_same_v<Value, float>)
    return hyle::compute::DataType::kFloat32;
  else if constexpr (std::is_same_v<Value, hyle::simd::float16_t>)
    return hyle::compute::DataType::kFloat16;
  else
    return hyle::compute::DataType::kBFloat16;
}

// An independent mathematical decoder. The half encoder below searches
// adjacent representable values, instead of sharing the product's bit shifts.
template <class Value> double decode_bits(std::uint16_t bits) {
  if constexpr (std::is_same_v<Value, hyle::simd::bfloat16_t>) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
  } else {
    const unsigned exponent = (bits >> 10) & 31U;
    const unsigned fraction = bits & 1023U;
    const double sign = (bits & 0x8000U) != 0 ? -1.0 : 1.0;
    if (exponent == 31)
      return fraction == 0 ? sign * std::numeric_limits<double>::infinity()
                           : std::numeric_limits<double>::quiet_NaN();
    return sign *
           (exponent == 0
                ? std::ldexp(static_cast<double>(fraction), -24)
                : std::ldexp(1.0 + static_cast<double>(fraction) / 1024.0,
                             static_cast<int>(exponent) - 15));
  }
}

template <class Value> double decode(Value value) {
  if constexpr (std::is_same_v<Value, float>)
    return value;
  else
    return decode_bits<Value>(value.bits);
}

template <class Value> Value encode(double value) {
  if constexpr (std::is_same_v<Value, float>) {
    return static_cast<float>(value);
  } else {
    constexpr bool bf16 = std::is_same_v<Value, hyle::simd::bfloat16_t>;
    constexpr std::uint16_t infinity = bf16 ? 0x7f80 : 0x7c00;
    constexpr std::uint16_t nan = bf16 ? 0x7fc0 : 0x7e00;
    constexpr std::uint16_t maximum = infinity - 1;
    const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    if (std::isnan(value))
      return Value::from_bits(nan);
    const double magnitude = std::abs(value);
    const double max_value = decode_bits<Value>(maximum);
    const double step = max_value - decode_bits<Value>(maximum - 1);
    if (magnitude >= max_value + step * 0.5)
      return Value::from_bits(static_cast<std::uint16_t>(sign | infinity));
    std::uint16_t low = 0;
    std::uint16_t high = maximum;
    while (low < high) {
      const auto mid = static_cast<std::uint16_t>((low + high + 1U) / 2U);
      if (decode_bits<Value>(mid) <= magnitude)
        low = mid;
      else
        high = static_cast<std::uint16_t>(mid - 1);
    }
    if (low != maximum) {
      const double below = magnitude - decode_bits<Value>(low);
      const double above =
          decode_bits<Value>(static_cast<std::uint16_t>(low + 1)) - magnitude;
      if (above < below || (above == below && (low & 1U) != 0))
        ++low;
    }
    return Value::from_bits(static_cast<std::uint16_t>(sign | low));
  }
}

template <class Value> hyle::mem::Buffer buffer(std::span<const Value> values) {
  auto result = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                          values.size_bytes(), alignof(Value))
                    .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());
  return result;
}

template <class Value>
std::vector<Value> read(const hyle::mem::Buffer &buffer) {
  std::vector<Value> values(buffer.size() / sizeof(Value));
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(values))).ok());
  return values;
}

template <class Value> void expect_near(Value actual, double expected) {
  const double rounded = decode(encode<Value>(expected));
  if (std::isnan(rounded))
    EXPECT_TRUE(std::isnan(decode(actual)));
  else if (std::isinf(rounded))
    EXPECT_EQ(decode(actual), rounded);
  else {
    constexpr double relative =
        std::is_same_v<Value, hyle::simd::bfloat16_t> ? 0.008 : 0.001;
    const double subnormal = decode_bits<Value>(1);
    EXPECT_NEAR(decode(actual), rounded,
                std::abs(rounded) * relative + subnormal);
  }
}

} // namespace half_test
