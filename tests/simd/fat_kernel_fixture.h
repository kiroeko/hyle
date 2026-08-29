#pragma once

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

#if defined(HWY_COMPILE_ONLY_SCALAR) && HWY_STATIC_TARGET != HWY_SCALAR
#error "fat scalar fixture did not select Highway scalar"
#elif defined(HYLE_SIMD_TARGET_SSE4) && HWY_STATIC_TARGET != HWY_SSE4
#error "fat SSE4 fixture did not select Highway SSE4"
#elif defined(HYLE_SIMD_TARGET_AVX2) && HWY_STATIC_TARGET != HWY_AVX2
#error "fat AVX2 fixture did not select Highway AVX2"
#elif defined(HYLE_SIMD_TARGET_AVX512) && HWY_STATIC_TARGET != HWY_AVX3
#error "fat AVX-512 fixture did not select Highway AVX3"
#elif defined(HYLE_SIMD_TARGET_NEON) && HWY_STATIC_TARGET != HWY_NEON && \
    HWY_STATIC_TARGET != HWY_NEON_WITHOUT_AES
#error "fat NEON fixture did not select Highway NEON"
#elif defined(HYLE_SIMD_TARGET_SVE) && HWY_STATIC_TARGET != HWY_SVE
#error "fat SVE fixture did not select Highway SVE"
#elif defined(HYLE_SIMD_TARGET_RVV) && HWY_STATIC_TARGET != HWY_RVV
#error "fat RVV fixture did not select Highway RVV"
#endif

#define HYLE_TEST_FAT_CAT_I(a, b) a##b
#define HYLE_TEST_FAT_CAT(a, b) HYLE_TEST_FAT_CAT_I(a, b)

#if defined(HYLE_SIMD_TARGET_AVX512)
#  define HYLE_TEST_FAT_TIER 30u
#elif defined(HYLE_SIMD_TARGET_AVX2)
#  define HYLE_TEST_FAT_TIER 20u
#elif defined(HYLE_SIMD_TARGET_SSE4)
#  define HYLE_TEST_FAT_TIER 12u
#elif defined(HYLE_SIMD_TARGET_SVE)
#  define HYLE_TEST_FAT_TIER 120u
#elif defined(HYLE_SIMD_TARGET_NEON)
#  define HYLE_TEST_FAT_TIER 110u
#elif defined(HYLE_SIMD_TARGET_RVV)
#  define HYLE_TEST_FAT_TIER 210u
#else
#  define HYLE_TEST_FAT_TIER 0u
#endif

void HYLE_TEST_FAT_CAT(hyle_test_fat_add_one_,
                       HYLE_SIMD_VARIANT_SUFFIX)(
    const float* src, float* dst, std::size_t n,
    std::uint32_t* selected_tier) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const auto one = hn::Set(d, 1.0f);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    hn::StoreU(hn::Add(hn::LoadU(d, src + i), one), d, dst + i);
  }
  for (; i < n; ++i) dst[i] = src[i] + 1.0f;
  *selected_tier = HYLE_TEST_FAT_TIER;
}

#undef HYLE_TEST_FAT_TIER
#undef HYLE_TEST_FAT_CAT
#undef HYLE_TEST_FAT_CAT_I
