// Highway vqsort borrow core, aarch64 NEON tier (128-bit). The ARM sibling of
// vqsort_highway_avx2.cc -- SAME borrowed VQSortStatic core, lowered to NEON here (Highway's
// vqsort sorting networks target ARMv8 Advanced SIMD). NEON is the ARMv8 baseline, so this TU
// needs NO extra -march (the NEON tier's flag list is empty, like the math NEON borrow) and
// sort_dispatch enters it UNCONDITIONALLY on arm64 -- no runtime gate, NEON is guaranteed present.
//
// In the arm64 SOURCES split only. See src/simd/sort/vqsort_body.inc for the shared definitions.
#include "hyle/platform.h"  // HYLE_ARCH_ARM64 -- MUST precede the guard below

#if defined(HYLE_ARCH_ARM64)

#define VQSORT_ONLY_STATIC 1

#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#define HYLE_SORT_TIER neon
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER

#endif  // HYLE_ARCH_ARM64
