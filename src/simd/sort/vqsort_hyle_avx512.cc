// HYLE port of Highway vqsort, AVX-512 tier (512-bit).
// Origin: hwy/contrib/sort (VQSortStatic path), SHA 83eff39371e5e6494b305b253f8a3c06f831faf4.
// The target macro selects the 512-bit D-tag backend. Product CMake intentionally
// excludes this TU for cl.exe (matching Highway's HWY_BROKEN_MSVC AVX3 gate);
// supported frontends enter these symbols only when runtime().sort.isa is AVX-512.
#include "hyle/core/environment.h"

#if HYLE_ARCH_X64
#define HYLE_SIMD_TARGET_AVX512
#define HYLE_SORT_TIER avx512
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER
#undef HYLE_SIMD_TARGET_AVX512
#endif  // HYLE_ARCH_X64
