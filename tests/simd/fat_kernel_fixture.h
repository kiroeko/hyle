#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/target.h"
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
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kAvx512;
#elif defined(HYLE_SIMD_TARGET_AVX2)
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kAvx2;
#elif defined(HYLE_SIMD_TARGET_SSE4)
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kSse4;
#elif defined(HYLE_SIMD_TARGET_SVE)
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kSve;
#elif defined(HYLE_SIMD_TARGET_NEON)
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kNeon;
#elif defined(HYLE_SIMD_TARGET_RVV)
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kRvv;
#else
inline constexpr auto kHyleTestCompiledTarget =
    ::hyle::simd::KernelTarget::kScalar;
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
  *selected_tier = static_cast<std::uint32_t>(kHyleTestCompiledTarget);
}

#undef HYLE_TEST_FAT_CAT
#undef HYLE_TEST_FAT_CAT_I
