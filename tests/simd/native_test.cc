#include <gtest/gtest.h>

#include <array>
#include <bit>
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

TEST(SimdNative, MixedDotsUseCurrentTarget) {
  constexpr std::size_t kCount = 67;
  std::array<hs::bfloat16_t, kCount> bf0{};
  std::array<hs::bfloat16_t, kCount> bf1{};
  std::array<hs::float16_t, kCount> f16a{};
  std::array<hs::float16_t, kCount> f16b{};
  std::array<float, kCount> f32{};
  std::array<std::int16_t, kCount> i16a{};
  std::array<std::int16_t, kCount> i16b{};
  float bf_expected = 0.0f;
  float mixed_expected = 0.0f;
  float f16_expected = 0.0f;
  std::uint64_t i16_bits = 0;
  for (std::size_t i = 0; i < kCount; ++i) {
    const float a = (static_cast<int>(i % 17) - 8) * 0.25f;
    const float b = (static_cast<int>((i * 5) % 19) - 9) * 0.125f;
    bf0[i] = hs::f32_to_bf16(a);
    bf1[i] = hs::f32_to_bf16(b);
    f16a[i] = hs::f32_to_f16(a);
    f16b[i] = hs::f32_to_f16(b);
    f32[i] = a + 0.0001f;
    i16a[i] = (i % 9 == 0) ? std::int16_t{-32768}
                            : static_cast<std::int16_t>(i * 101 - 3000);
    i16b[i] = (i % 11 == 0) ? std::int16_t{-32768}
                             : static_cast<std::int16_t>(2000 - i * 37);
    bf_expected += hs::bf16_to_f32(bf0[i]) * hs::bf16_to_f32(bf1[i]);
    mixed_expected += f32[i] * hs::bf16_to_f32(bf1[i]);
    f16_expected += hs::f16_to_f32(f16a[i]) * hs::f16_to_f32(f16b[i]);
    i16_bits += static_cast<std::uint64_t>(
        static_cast<std::int64_t>(i16a[i]) * i16b[i]);
  }
  EXPECT_NEAR(hs::native::dot(bf0.data(), bf1.data(), kCount), bf_expected,
              std::abs(bf_expected) * 2e-5f + 2e-5f);
  EXPECT_NEAR(hs::native::dot(f32.data(), bf1.data(), kCount), mixed_expected,
              std::abs(mixed_expected) * 2e-5f + 2e-5f);
  EXPECT_NEAR(hs::native::dot(f16a.data(), f16b.data(), kCount), f16_expected,
              std::abs(f16_expected) * 2e-5f + 2e-5f);
  EXPECT_EQ(hs::native::dot(i16a.data(), i16b.data(), kCount),
            std::bit_cast<std::int64_t>(i16_bits));
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
