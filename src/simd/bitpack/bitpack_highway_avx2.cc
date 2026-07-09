// Highway bit_pack borrow core, AVX2 tier (256-bit). BORROWS Highway's verified bit_pack
// (Pack8/Pack16/Pack32 in hwy/contrib/bit_pack/bit_pack-inl.h) -- HYLE does not hand-roll a
// vectorized bit packer (charter: 借核不手搓). Structurally identical to vqsort_highway_avx2.cc:
// static single-target dispatch, compiled at the full AVX2 baseline (hyle_set_highway_isa) so
// HWY_STATIC_TARGET resolves to 256-bit AVX2 rather than silently downgrading to SSSE3.
//
// x86-only: this TU is in the x86 SOURCES split and lowers to AVX2 intrinsics. bitpack_dispatch
// enters it only when cpu_features().avx2, so the AVX2 codegen never runs on an older CPU (falls
// to the scalar bit-stream floor there). See src/simd/bitpack/bitpack_body.inc for the shared body.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "hwy/highway.h"
#include "hwy/contrib/bit_pack/bit_pack-inl.h"

#define HYLE_BITPACK_TIER avx2
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_BITPACK_TIER

#endif  // HYLE_ARCH_X64
