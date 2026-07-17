// HYLE port of Highway vqsort, SVE scalable experiment/product tier.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#ifndef HYLE_SIMD_TARGET_SVE
#define HYLE_SIMD_TARGET_SVE
#endif
// Keep D-aware and V-only operations on the same 128/256-bit power-of-two logical shape. This
// avoids the invalid 12-lane networks at VL=384 while still emitting only SVE instructions.
#define HYLE_SIMD_SVE_DFROMV_MAX_BYTES 32
#define HYLE_SORT_TIER sve
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER
#undef HYLE_SIMD_SVE_DFROMV_MAX_BYTES
#endif
