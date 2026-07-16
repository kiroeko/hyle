#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hyle/simd/native.h"
#include "hyle/simd/runtime.h"

namespace hs = hyle::simd;

TEST(SimdNative, WholeBufferOpsUseCurrentTarget) {
  constexpr std::size_t kCount = 37;
  std::array<float, kCount> x{};
  std::array<float, kCount> y{};
  std::array<float, kCount> out{};
  float expected_dot = 0.0f;
  for (std::size_t i = 0; i < kCount; ++i) {
    x[i] = static_cast<float>(i + 1) * 0.25f;
    y[i] = static_cast<float>(i + 3) * -0.125f;
    expected_dot += x[i] * y[i];
  }

  hs::native::add(x.data(), y.data(), out.data(), out.size());
  for (std::size_t i = 0; i < kCount; ++i)
    EXPECT_FLOAT_EQ(out[i], x[i] + y[i]);

  const float got = hs::native::dot(x.data(), y.data(), x.size());
  EXPECT_NEAR(got, expected_dot, std::abs(expected_dot) * 1e-5f + 1e-5f);
}

TEST(SimdNative, RandomMatchesPortableScalarRuntime) {
  constexpr std::size_t kCount = 73;
  hs::RandomState native_state(555);
  hs::RandomState scalar_state(555);
  std::array<std::uint64_t, kCount> native{};
  std::array<std::uint64_t, kCount> scalar{};

  hs::native::random_fill(native_state, native.data(), native.size());
  const hs::Runtime* scalar_runtime = hs::runtime_for(hs::Isa::kScalar);
  ASSERT_NE(scalar_runtime, nullptr);
  scalar_runtime->random.fill_u64(&scalar_state, scalar.data(), scalar.size());
  EXPECT_EQ(native, scalar);
}

TEST(SimdNative, ScanUsesCurrentTarget) {
  std::array<std::int32_t, 67> inclusive{};
  std::array<std::int32_t, 67> exclusive{};
  for (std::size_t i = 0; i < inclusive.size(); ++i) {
    inclusive[i] = static_cast<std::int32_t>(i % 7) - 3;
    exclusive[i] = inclusive[i];
  }
  hs::native::inclusive_scan(inclusive.data(), inclusive.size());
  hs::native::exclusive_scan(exclusive.data(), exclusive.size());
  std::int32_t sum = 0;
  for (std::size_t i = 0; i < inclusive.size(); ++i) {
    const std::int32_t value = static_cast<std::int32_t>(i % 7) - 3;
    EXPECT_EQ(exclusive[i], sum);
    sum += value;
    EXPECT_EQ(inclusive[i], sum);
  }
}
