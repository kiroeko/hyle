#include <gtest/gtest.h>

#include "hyle/simd/dtype_ops.h"
#include "hyle/simd/runtime.h"
#include "hyle/simd/vec.h"
#include "src/simd/dtype/kernels.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hs = hyle::simd;

#if defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SCALABLE_DTYPE(name) hs::detail::name##_sve
constexpr hs::Isa kExpectedDtypeIsa = hs::Isa::kSve;
#elif defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SCALABLE_DTYPE(name) hs::detail::name##_rvv
constexpr hs::Isa kExpectedDtypeIsa = hs::Isa::kRvv;
#endif

namespace {
constexpr std::size_t kMaxLanes = 256;

constexpr std::uint32_t kF32Patterns[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x007FFFFFu, 0x00800000u,
    0x33000000u, 0x33800000u, 0x38000000u, 0x38800000u, 0x3F800000u,
    0x477FE000u, 0x477FF000u, 0x47800000u, 0x7F7FFFFFu, 0x7F800000u,
    0xFF800000u, 0x7F800001u, 0x7FC12345u, 0xBF800000u, 0xC77FE000u,
};

void CheckConversions(bool public_api) {
  const std::size_t w = hs::Lanes(hs::ScalableTag<float>{});
  const std::size_t n = 2 * w + 5;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) float f32[kMaxLanes], f16_out[kMaxLanes], bf16_out[kMaxLanes];
  alignas(64) hs::float16_t f16[kMaxLanes], f16_encoded[kMaxLanes];
  alignas(64) hs::bfloat16_t bf16[kMaxLanes], bf16_encoded[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) {
    f32[i] = hs::detail::bits_to_f32(kF32Patterns[i % std::size(kF32Patterns)]);
    f16[i] = hs::float16_t::from_bits(static_cast<std::uint16_t>(i * 1009u));
    bf16[i] = hs::bfloat16_t::from_bits(static_cast<std::uint16_t>(i * 4051u));
  }

  if (public_api) {
    hs::f16_to_f32_batch(f16, f16_out, n);
    hs::f32_to_f16_batch(f32, f16_encoded, n);
    hs::bf16_to_f32_batch(bf16, bf16_out, n);
    hs::f32_to_bf16_batch(f32, bf16_encoded, n);
  } else {
    HYLE_SCALABLE_DTYPE(f16_to_f32)(f16, f16_out, n);
    HYLE_SCALABLE_DTYPE(f32_to_f16)(f32, f16_encoded, n);
    HYLE_SCALABLE_DTYPE(bf16_to_f32)(bf16, bf16_out, n);
    HYLE_SCALABLE_DTYPE(f32_to_bf16)(f32, bf16_encoded, n);
  }

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(hs::detail::f32_to_bits(f16_out[i]),
              hs::detail::f32_to_bits(hs::f16_to_f32(f16[i]))) << i;
    EXPECT_EQ(f16_encoded[i].bits, hs::f32_to_f16(f32[i]).bits) << i;
    EXPECT_EQ(hs::detail::f32_to_bits(bf16_out[i]),
              hs::detail::f32_to_bits(hs::bf16_to_f32(bf16[i]))) << i;
    EXPECT_EQ(bf16_encoded[i].bits, hs::f32_to_bf16(f32[i]).bits) << i;
  }
}
}  // namespace

TEST(ScalableDtypeProduct, DirectKernelsMatchScalarOracle) { CheckConversions(false); }

TEST(ScalableDtypeProduct, RuntimeAndPublicApiUseScalableTier) {
  EXPECT_EQ(hs::runtime().dtype.isa, kExpectedDtypeIsa);
  const hs::Runtime* exact = hs::runtime_for(kExpectedDtypeIsa);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(exact->dtype.isa, kExpectedDtypeIsa);
  CheckConversions(true);
}

TEST(ScalableDtypeProduct, ExhaustiveStorageAndBroadF32PatternsMatchScalar) {
  constexpr std::size_t kStorageValues = 1u << 16;
  std::vector<hs::float16_t> f16(kStorageValues);
  std::vector<hs::bfloat16_t> bf16(kStorageValues);
  std::vector<float> decoded(kStorageValues);
  for (std::size_t i = 0; i < kStorageValues; ++i) {
    f16[i] = hs::float16_t::from_bits(static_cast<std::uint16_t>(i));
    bf16[i] = hs::bfloat16_t::from_bits(static_cast<std::uint16_t>(i));
  }
  HYLE_SCALABLE_DTYPE(f16_to_f32)(f16.data(), decoded.data(), decoded.size());
  for (std::size_t i = 0; i < kStorageValues; ++i)
    ASSERT_EQ(hs::detail::f32_to_bits(decoded[i]),
              hs::detail::f32_to_bits(hs::f16_to_f32(f16[i]))) << i;
  HYLE_SCALABLE_DTYPE(bf16_to_f32)(bf16.data(), decoded.data(), decoded.size());
  for (std::size_t i = 0; i < kStorageValues; ++i)
    ASSERT_EQ(hs::detail::f32_to_bits(decoded[i]),
              hs::detail::f32_to_bits(hs::bf16_to_f32(bf16[i]))) << i;

  constexpr std::size_t kF32Values = 16387;
  std::vector<float> src(kF32Values);
  std::vector<hs::float16_t> out16(kF32Values);
  std::vector<hs::bfloat16_t> outb(kF32Values);
  std::uint32_t state = 0x12345678u;
  for (float& value : src) {
    state = state * 1664525u + 1013904223u;
    value = hs::detail::bits_to_f32(state);
  }
  HYLE_SCALABLE_DTYPE(f32_to_f16)(src.data(), out16.data(), src.size());
  HYLE_SCALABLE_DTYPE(f32_to_bf16)(src.data(), outb.data(), src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    ASSERT_EQ(out16[i].bits, hs::f32_to_f16(src[i]).bits) << i;
    ASSERT_EQ(outb[i].bits, hs::f32_to_bf16(src[i]).bits) << i;
  }
}

#undef HYLE_SCALABLE_DTYPE
