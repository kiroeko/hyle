// Highway bit_pack borrow core, AVX-512 tier (512-bit). The 512-bit sibling of
// bitpack_highway_avx2.cc -- SAME borrowed Pack8/Pack16/Pack32 cores, only compiled at the
// AVX-512 baseline (hyle_set_avx512_bitpack_isa) so HWY_STATIC_TARGET resolves to AVX3 (zmm)
// instead of AVX2 (ymm). bitpack_dispatch enters it only when the host has the AVX-512 group
// (best_isa() >= kAvx512), else it stays on the 256-bit avx2 core.
//
// NO MSVC EXCLUSION (unlike transcendental_highway_avx512.cc): Highway marks its AVX3 target
// HWY_BROKEN on MSVC only for the FP-transcendental codegen path; bit_pack is integer shift/or/and
// SIMD, which MSVC /arch:AVX512 compiles correctly -- same reasoning as the vqsort AVX3 core. If a
// future MSVC probe shows broken AVX3 bit_pack codegen, mirror the math TU's `&& !defined(_MSC_VER)`
// guard here + drop the 512 branch in bitpack_dispatch on MSVC.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "hwy/highway.h"
#include "hwy/contrib/bit_pack/bit_pack-inl.h"

#define HYLE_BITPACK_TIER avx512
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_BITPACK_TIER

#endif  // HYLE_ARCH_X64
