#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/apple.h"

namespace apple_tensor_test {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;
using F16 = hyle::simd::float16_t;
using B16 = hyle::simd::bfloat16_t;
inline mx::Stream stream() { return mx::default_stream(mx::Device::gpu); }
inline mx::array ready(mx::array value) {
  mx::async_eval({value});
  mx::synchronize(stream());
  value.wait();
  return value;
}
inline c::DataType logical(mx::Dtype type) {
  if (type == mx::float16)
    return c::DataType::kFloat16;
  if (type == mx::bfloat16)
    return c::DataType::kBFloat16;
  if (type == mx::int32)
    return c::DataType::kInt32;
  if (type == mx::int64)
    return c::DataType::kInt64;
  if (type == mx::bool_)
    return c::DataType::kBool;
  return c::DataType::kFloat32;
}
inline double rounded(double value, mx::Dtype type) {
  if (type == mx::float16)
    return half_test::decode(half_test::encode<F16>(value));
  if (type == mx::bfloat16)
    return half_test::decode(half_test::encode<B16>(value));
  return static_cast<float>(value);
}
inline mx::array make(std::span<const double> values, mx::Shape shape,
                      mx::Dtype type) {
  if (type == mx::float32) {
    std::vector<float> data(values.begin(), values.end());
    return mx::array(data.data(), std::move(shape), mx::float32);
  }
  std::vector<std::uint16_t> data(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    data[i] = type == mx::float16 ? half_test::encode<F16>(values[i]).bits
                                  : half_test::encode<B16>(values[i]).bits;
  return ready(mx::view(mx::array(data.data(), std::move(shape), mx::uint16),
                        type, stream()));
}
inline std::vector<std::uint64_t> bits(const mx::array &array) {
  std::vector<std::uint64_t> result(array.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    std::size_t remaining = index, offset = 0;
    for (std::size_t axis = array.ndim(); axis-- > 0;) {
      offset += (remaining % array.shape()[axis]) * array.strides()[axis];
      remaining /= array.shape()[axis];
    }
    std::memcpy(&result[index], array.data<char>() + offset * array.itemsize(),
                array.itemsize());
  }
  return result;
}
inline std::vector<double> values(const mx::array &array) {
  const auto raw = bits(array);
  std::vector<double> result(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (array.dtype() == mx::float32)
      result[i] = std::bit_cast<float>(static_cast<std::uint32_t>(raw[i]));
    else if (array.dtype() == mx::float16)
      result[i] =
          half_test::decode_bits<F16>(static_cast<std::uint16_t>(raw[i]));
    else if (array.dtype() == mx::bfloat16)
      result[i] =
          half_test::decode_bits<B16>(static_cast<std::uint16_t>(raw[i]));
    else if (array.dtype() == mx::int32)
      result[i] =
          std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(raw[i]));
    else if (array.dtype() == mx::int64)
      result[i] = static_cast<double>(std::bit_cast<std::int64_t>(raw[i]));
    else
      result[i] = raw[i] != 0;
  }
  return result;
}
inline hyle::Result<mx::array> execute(a::Context &context,
                                       c::Operation operation,
                                       std::span<const mx::array> inputs) {
  HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(operation, inputs));
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  return result.get()->arrays()[0];
}
template <class Descriptor>
inline hyle::Result<mx::array> execute(a::Context &context,
                                       Descriptor operation,
                                       std::span<const mx::array> inputs) {
  return execute(context, c::Operation(std::move(operation)), inputs);
}
inline void expect(const mx::array &actual, std::span<const double> expected) {
  const auto observed = values(actual);
  ASSERT_EQ(observed.size(), expected.size());
  const auto type = actual.dtype();
  const double relative = type == mx::bfloat16  ? 0.008
                          : type == mx::float16 ? 0.001
                                                : 6e-6;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto value = rounded(expected[i], type);
    if (std::isnan(value))
      EXPECT_TRUE(std::isnan(observed[i])) << i;
    else if (std::isinf(value))
      EXPECT_EQ(observed[i], value) << i;
    else
      EXPECT_NEAR(observed[i], value,
                  relative * std::abs(value) +
                      (type == mx::float32 ? 2e-6 : 2e-7))
          << i;
  }
}
inline double activation(c::UnaryKind kind, double x) {
  switch (kind) {
  case c::UnaryKind::kAbs:
    return std::abs(x);
  case c::UnaryKind::kNeg:
    return -x;
  case c::UnaryKind::kExp:
    return std::exp(x);
  case c::UnaryKind::kLog:
    return std::log(x);
  case c::UnaryKind::kSqrt:
    return std::sqrt(x);
  case c::UnaryKind::kRsqrt:
    return 1 / std::sqrt(x);
  case c::UnaryKind::kTanh:
    return std::tanh(x);
  case c::UnaryKind::kSigmoid:
    return 1 / (1 + std::exp(-x));
  case c::UnaryKind::kRelu:
    return std::isnan(x) ? x : (x > 0 ? x : 0);
  case c::UnaryKind::kSoftplus:
    return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
  case c::UnaryKind::kSilu:
    return x == -std::numeric_limits<double>::infinity()
               ? 0
               : x / (1 + std::exp(-x));
  case c::UnaryKind::kGelu:
    return x == -std::numeric_limits<double>::infinity()
               ? 0
               : 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
  case c::UnaryKind::kGeluTanh:
    return x == -std::numeric_limits<double>::infinity()
               ? 0
               : 0.5 * x *
                     (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * x *
                                    (1 + 0.044715 * x * x)));
  }
  return std::numeric_limits<double>::quiet_NaN();
}
inline std::vector<double> pattern(std::size_t count, int phase = 0) {
  std::vector<double> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = (static_cast<int>((i * 7 + phase * 3) % 29) - 14) * 0.17;
  return result;
}
} // namespace apple_tensor_test
