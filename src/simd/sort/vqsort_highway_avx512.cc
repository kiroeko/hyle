// Highway vqsort borrow core, AVX-512 tier (512-bit). The 512-bit sibling of
// vqsort_highway_avx2.cc -- SAME borrowed VQSortStatic core, only compiled at the AVX-512
// baseline (hyle_set_avx512_sort_isa) so HWY_STATIC_TARGET resolves to AVX3 (zmm) instead of
// AVX2 (ymm). vqsort is Highway's flagship AVX-512 win (the sorting networks widen to 512-bit),
// so this is the tier that actually shines. sort_dispatch enters it only when the host has the
// AVX-512 group (best_isa() >= kAvx512), else it stays on the 256-bit avx2 core.
//
// NO MSVC EXCLUSION (unlike transcendental_highway_avx512.cc): Highway marks its AVX3 target
// HWY_BROKEN on MSVC only for the FP-transcendental codegen path; vqsort is integer/compare SIMD
// (sorting networks + partition), which MSVC /arch:AVX512 compiles correctly. So the 512 vqsort
// core is defined on ALL compilers -- confirmed by Highway's own vqsort_*.cc shipping an AVX3
// target on MSVC. If a future MSVC probe shows broken AVX3 vqsort codegen, mirror the math TU's
// `&& !defined(_MSC_VER)` guard here + drop the 512 branch in sort_dispatch on MSVC.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#define VQSORT_ONLY_STATIC 1

#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#define HYLE_SORT_TIER avx512
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER

#endif  // HYLE_ARCH_X64
