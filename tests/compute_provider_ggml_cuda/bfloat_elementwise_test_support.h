#pragma once

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/operation.h"

#include <array>
#include <cmath>
#include <limits>

namespace bfloat_elementwise_test {
using BF16 = hyle::simd::bfloat16_t;
namespace c = hyle::compute;

inline std::array<c::Operation, 11> operations() {
  return {c::Operation(c::Binary{c::BinaryKind::kAdd}),
          c::Operation(c::Binary{c::BinaryKind::kSubtract}),
          c::Operation(c::Binary{c::BinaryKind::kMultiply}),
          c::Operation(c::Binary{c::BinaryKind::kDivide}),
          c::Operation(c::Unary{c::UnaryKind::kAbs}),
          c::Operation(c::Unary{c::UnaryKind::kNeg}),
          c::Operation(c::Unary{c::UnaryKind::kExp}),
          c::Operation(c::Unary{c::UnaryKind::kLog}),
          c::Operation(c::Unary{c::UnaryKind::kSqrt}),
          c::Operation(c::Unary{c::UnaryKind::kTanh}),
          c::Operation(c::Unary{c::UnaryKind::kSigmoid})};
}

// Independent FP64 arithmetic followed by the public F32 computation and
// BF16 storage boundaries. Only finite normal test points use this oracle;
// edge cases compare the actual old F32 recipe, including its native flags.
inline BF16 reference(std::size_t operation, BF16 x, BF16 y) {
  const double a = half_test::decode(x), b = half_test::decode(y);
  double result = 0;
  switch (operation) {
  case 0: result = a + b; break;
  case 1: result = a - b; break;
  case 2: result = a * b; break;
  case 3: result = a / b; break;
  case 4: result = std::abs(a); break;
  case 5: result = -a; break;
  case 6: result = std::exp(a); break;
  case 7: result = std::log(a); break;
  case 8: result = std::sqrt(a); break;
  case 9: result = std::tanh(a); break;
  case 10: result = 1.0 / (1.0 + std::exp(-a)); break;
  }
  return half_test::encode<BF16>(static_cast<float>(result));
}

inline void expect_bits(BF16 actual, BF16 expected, std::size_t index) {
  if (std::isnan(half_test::decode(expected)))
    EXPECT_TRUE(std::isnan(half_test::decode(actual))) << index;
  else
    EXPECT_EQ(actual.bits, expected.bits) << index;
}

inline std::array<BF16, 17> edge_values() {
  return {BF16::from_bits(0x0000), BF16::from_bits(0x8000),
          BF16::from_bits(0x0001), BF16::from_bits(0x8001),
          BF16::from_bits(0x0080), BF16::from_bits(0x7f7f),
          BF16::from_bits(0xff7f), BF16::from_bits(0x7f80),
          BF16::from_bits(0xff80), BF16::from_bits(0x7fc1),
          half_test::encode<BF16>(0.25), half_test::encode<BF16>(0.5),
          half_test::encode<BF16>(1.0), half_test::encode<BF16>(2.0),
          half_test::encode<BF16>(4.0), half_test::encode<BF16>(-1.0),
          half_test::encode<BF16>(-0.5)};
}
} // namespace bfloat_elementwise_test
