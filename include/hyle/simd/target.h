#pragma once

#include "hyle/simd/dispatch.h"

namespace hyle::simd {

#if defined(HYLE_SIMD_TARGET_AVX512) || defined(HYLE_SIMD_EXACT_TARGET_AVX512)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kAvx512;
#elif defined(HYLE_SIMD_TARGET_AVX2) || defined(HYLE_SIMD_EXACT_TARGET_AVX2)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kAvx2;
#elif defined(HYLE_SIMD_TARGET_SSE4) || defined(HYLE_SIMD_EXACT_TARGET_SSE4)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kSse42;
#elif defined(HYLE_SIMD_TARGET_SVE) || defined(HYLE_SIMD_EXACT_TARGET_SVE)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kSve;
#elif defined(HYLE_SIMD_TARGET_NEON) || defined(HYLE_SIMD_EXACT_TARGET_NEON)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kNeon;
#elif defined(HYLE_SIMD_TARGET_RVV) || defined(HYLE_SIMD_EXACT_TARGET_RVV)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kRvv;
#elif defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
static constexpr bool kExactTarget = true;
static constexpr Isa kTargetIsa = Isa::kScalar;
#else
static constexpr bool kExactTarget = false;
static constexpr Isa kTargetIsa = Isa::kScalar;
#endif

}  // namespace hyle::simd
