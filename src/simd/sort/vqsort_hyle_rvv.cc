// HYLE port of Highway vqsort, RVV scalable tier. HYLE currently uses honest LMUL=1 values;
// fractional LMUL is not modeled by the backend's native vector types yet.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#ifndef HYLE_SIMD_TARGET_RVV
#define HYLE_SIMD_TARGET_RVV
#endif
#define HYLE_SORT_TIER rvv
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER
#endif
