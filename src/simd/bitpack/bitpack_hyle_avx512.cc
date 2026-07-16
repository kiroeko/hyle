// HYLE bit-pack core for the AVX-512 tier. The target macro selects the 512-bit D-tag backend;
// the dispatcher enters these symbols only after its AVX-512 CPU-feature gate.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)
#define HYLE_SIMD_TARGET_AVX512
#define HYLE_SIMD_VARIANT_SUFFIX avx512
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_SIMD_VARIANT_SUFFIX
#undef HYLE_SIMD_TARGET_AVX512
#endif  // HYLE_ARCH_X64
