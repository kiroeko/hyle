// HYLE port of Highway vqsort, aarch64 NEON tier (128-bit).
// Origin: hwy/contrib/sort (VQSortStatic path), SHA 83eff39371e5e6494b305b253f8a3c06f831faf4.
// NEON is the ARMv8 baseline; sort_dispatch enters these symbols unconditionally on arm64.
#include "hyle/core/environment.h"

#if HYLE_ARCH_ARM64
#define HYLE_SIMD_TARGET_NEON
#define HYLE_SORT_TIER neon
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER
#undef HYLE_SIMD_TARGET_NEON
#endif  // HYLE_ARCH_ARM64
