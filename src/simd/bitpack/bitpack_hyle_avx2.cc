// HYLE bit-pack core for the AVX2 tier. The target macro selects the 256-bit D-tag backend;
// the dispatcher enters these symbols only after its AVX2 CPU-feature gate.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)
#define HYLE_SIMD_TARGET_AVX2
#define HYLE_SIMD_VARIANT_SUFFIX avx2
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_SIMD_VARIANT_SUFFIX
#undef HYLE_SIMD_TARGET_AVX2
#endif  // HYLE_ARCH_X64
