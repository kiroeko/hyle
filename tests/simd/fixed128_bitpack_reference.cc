// Test-only 128-bit scalar D-tag reference for the scalable bitpack product test. This compiles
// the shared packing semantics through a different backend, so SVE/RVV partial-tag behavior is
// checked byte-for-byte against a fixed-width implementation.
#if defined(HYLE_SIMD_TARGET_SVE)
#undef HYLE_SIMD_TARGET_SVE
#endif
#if defined(HYLE_SIMD_TARGET_RVV)
#undef HYLE_SIMD_TARGET_RVV
#endif
#define HYLE_SIMD_VARIANT_SUFFIX ref128
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_SIMD_VARIANT_SUFFIX
