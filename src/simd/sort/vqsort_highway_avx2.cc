// Highway vqsort borrow core, AVX2 tier (256-bit). BORROWS Highway's verified vqsort
// (VQSortStatic / VQPartialSortStatic / VQSelectStatic) -- HYLE does not hand-roll a vectorized
// sort (charter: 借验证核不手搓). Structurally identical to transcendental_highway.cc: static
// single-target dispatch, compiled at the full AVX2 baseline (hyle_set_highway_isa) so
// HWY_STATIC_TARGET resolves to 256-bit AVX2 rather than silently downgrading to SSSE3.
//
// x86-only: this TU is in the x86 SOURCES split and lowers to AVX2 intrinsics. sort_dispatch
// enters it only when cpu_features().avx2, so the AVX2 codegen never runs on an older CPU (falls
// to the std:: scalar floor there). See src/simd/sort/vqsort_body.inc for the shared definitions.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

// VQSORT_ONLY_STATIC=1: use VQSortStatic without linking vqsort.cc (no Fill16BytesSecure). MUST
// be defined BEFORE including vqsort-inl.h. Header-only, no DLLEXPORT -- entropy from stack/clock.
#define VQSORT_ONLY_STATIC 1

#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#define HYLE_SORT_TIER avx2
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER

#endif  // HYLE_ARCH_X64
