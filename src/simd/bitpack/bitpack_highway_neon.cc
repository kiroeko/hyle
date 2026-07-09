// Highway bit_pack borrow core, aarch64 NEON tier (128-bit, ARMv8 baseline). SAME borrowed
// Pack8/Pack16/Pack32 cores as the x86 tiers, lowering to NEON intrinsics here. bitpack_dispatch
// enters it unconditionally on arm64 (NEON is guaranteed present, like the vqsort NEON borrow) --
// no runtime gate, no SIGILL risk. NEON is the ARMv8 baseline so no extra -march flag is needed
// (empty tier flags in CMake). See src/simd/bitpack/bitpack_body.inc for the shared body.
#include "hyle/platform.h"  // HYLE_ARCH_ARM64 -- MUST precede the guard below

#if defined(HYLE_ARCH_ARM64)

#include "hwy/highway.h"
#include "hwy/contrib/bit_pack/bit_pack-inl.h"

#define HYLE_BITPACK_TIER neon
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_BITPACK_TIER

#endif  // HYLE_ARCH_ARM64
