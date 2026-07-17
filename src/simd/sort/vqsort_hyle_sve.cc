// HYLE port of Highway vqsort, SVE scalable experiment/product tier.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#ifndef HYLE_SIMD_TARGET_SVE
#define HYLE_SIMD_TARGET_SVE
#endif
// Keep D-aware and V-only sorting-network operations on the same 128/256-bit power-of-two logical
// shape. The vqsort body widens its O(n) partition hot path back to the full physical VL; only the
// network/base-case swizzles retain this cap, avoiding invalid 12-lane networks at VL=384.
#define HYLE_SIMD_SVE_DFROMV_MAX_BYTES 32
#define HYLE_SORT_TIER sve
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER
#undef HYLE_SIMD_SVE_DFROMV_MAX_BYTES
#endif
