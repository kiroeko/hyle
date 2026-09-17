#pragma once

#include <bit>

#include "../compute_provider_native/half_test_support.h"
#include "fixture.h"

namespace vulkan_tensor_test {
namespace c = hyle::compute;
namespace v = c::vulkan;
using F16 = hyle::simd::float16_t;
using B16 = hyle::simd::bfloat16_t;
inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
inline std::size_t offset(const c::TensorDesc &desc, std::size_t linear) {
  std::size_t result = 0;
  for (std::size_t axis = desc.rank(); axis-- > 0;) {
    result += (linear % desc.dimensions()[axis]) * desc.byte_strides()[axis];
    linear /= desc.dimensions()[axis];
  }
  return result;
}
inline std::uint64_t encode(double value, c::DataType type) {
  if (type == c::DataType::kFloat16)
    return half_test::encode<F16>(value).bits;
  if (type == c::DataType::kBFloat16)
    return half_test::encode<B16>(value).bits;
  return std::bit_cast<std::uint32_t>(static_cast<float>(value));
}
inline double decode(std::uint64_t bits, c::DataType type) {
  if (type == c::DataType::kFloat16)
    return half_test::decode_bits<F16>(static_cast<std::uint16_t>(bits));
  if (type == c::DataType::kBFloat16)
    return half_test::decode_bits<B16>(static_cast<std::uint16_t>(bits));
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits));
}
inline double rounded(double value, c::DataType type) {
  return decode(encode(value, type), type);
}
inline std::vector<std::byte> payload(const c::TensorDesc &desc,
                                      std::span<const std::uint64_t> raw) {
  std::vector<std::byte> data(*desc.required_bytes(), std::byte{0xcd});
  for (std::size_t index = 0; index < raw.size(); ++index)
    std::memcpy(data.data() + offset(desc, index), &raw[index],
                c::data_type_size(desc.data_type()));
  return data;
}
inline v::Tensor make(const v::Context &context, const c::TensorDesc &desc,
                      std::span<const double> values) {
  std::vector<std::uint64_t> raw;
  for (double value : values)
    raw.push_back(encode(value, desc.data_type()));
  return context.upload(desc, payload(desc, raw)).value();
}
inline std::vector<std::byte> bytes(const v::Context &context,
                                    const v::Tensor &tensor) {
  std::vector<std::byte> result(*tensor.description().required_bytes());
  EXPECT_TRUE(context.download(tensor, result).ok());
  return result;
}
inline std::vector<std::uint64_t> bits(const v::Context &context,
                                       const v::Tensor &tensor) {
  const auto data = bytes(context, tensor);
  const auto &desc = tensor.description();
  std::vector<std::uint64_t> result(*desc.element_count(), 0);
  for (std::size_t index = 0; index < result.size(); ++index)
    std::memcpy(&result[index], data.data() + offset(desc, index),
                c::data_type_size(desc.data_type()));
  return result;
}
inline std::vector<double> values(const v::Context &context,
                                  const v::Tensor &tensor) {
  std::vector<double> result;
  for (auto raw : bits(context, tensor))
    result.push_back(decode(raw, tensor.description().data_type()));
  return result;
}
inline hyle::Result<v::Tensor> execute(const v::Context &context,
                                       c::Operation operation,
                                       std::span<const v::Tensor> inputs) {
  std::vector<v::Argument> arguments;
  for (const auto &input : inputs)
    arguments.emplace_back(input.description());
  HYLE_ASSIGN_OR_RETURN(
      auto program,
      v::prepare(context, operation, std::span<const v::Argument>(arguments)));
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  return result.get()->tensors()[0];
}
inline void expect(const v::Context &context, const v::Tensor &actual,
                   std::span<const double> expected, bool exact = false) {
  const auto observed = values(context, actual);
  ASSERT_EQ(observed.size(), expected.size());
  const auto type = actual.description().data_type();
  const double relative = type == c::DataType::kBFloat16  ? 0.008
                          : type == c::DataType::kFloat16 ? 0.001
                                                          : 6e-6;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    const auto want = rounded(expected[i], type);
    if (std::isnan(want))
      EXPECT_TRUE(std::isnan(observed[i])) << i;
    else if (std::isinf(want) || exact) {
      EXPECT_EQ(observed[i], want) << i;
      if (want == 0)
        EXPECT_EQ(std::signbit(observed[i]), std::signbit(want)) << i;
    } else
      EXPECT_NEAR(observed[i], want,
                  relative * std::abs(want) +
                      (type == c::DataType::kFloat32 ? 2e-6 : 2e-7))
          << i;
  }
}
inline double unary(c::UnaryKind mode, double x) {
  switch (mode) {
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
                     (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) * x *
                                    (1 + 0.044715 * x * x)));
  }
  return std::numeric_limits<double>::quiet_NaN();
}
} // namespace vulkan_tensor_test
