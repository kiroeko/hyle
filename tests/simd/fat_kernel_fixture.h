#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/native/algorithm.h"

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
  hyle::simd::native::transform(src, dst, n, [](auto d, auto v) {
    return hyle::simd::Add(v, hyle::simd::Set(d, 1.0f));
  });
  *selected_tier = HYLE_TEST_FAT_TIER;
}

#undef HYLE_TEST_FAT_TIER
#undef HYLE_TEST_FAT_CAT
#undef HYLE_TEST_FAT_CAT_I
