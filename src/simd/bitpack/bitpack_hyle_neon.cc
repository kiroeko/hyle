// HYLE bit-pack core for the ARMv8 NEON baseline. The target macro selects the 128-bit D-tag
// backend; ARM64 always has NEON, so the dispatcher enters this tier directly.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64)
#define HYLE_SIMD_TARGET_NEON
#define HYLE_SIMD_VARIANT_SUFFIX neon
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_SIMD_VARIANT_SUFFIX
#undef HYLE_SIMD_TARGET_NEON
#endif  // HYLE_ARCH_ARM64
