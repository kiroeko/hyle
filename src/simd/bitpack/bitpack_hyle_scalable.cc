// Length-agnostic SVE/RVV bit-pack product kernel. The physical register is scalable, while the
// active D-tag is capped to a fixed 128-bit logical block in bitpack_body.inc. No scalable vector
// type crosses this TU or the pointer/count/POD public ABI.

#include "hyle/platform.h"

#if (defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)) || \
    (defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT))

#if defined(HYLE_ARCH_ARM64) && !defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SIMD_TARGET_SVE
#define HYLE_SIMD_VARIANT_SUFFIX sve
#elif defined(HYLE_ARCH_RISCV64) && !defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SIMD_TARGET_RVV
#define HYLE_SIMD_VARIANT_SUFFIX rvv
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SIMD_VARIANT_SUFFIX sve
#else
#define HYLE_SIMD_VARIANT_SUFFIX rvv
#endif
#define HYLE_BITPACK_FIXED_128_LAYOUT
#include "src/simd/bitpack/bitpack_body.inc"
#undef HYLE_BITPACK_FIXED_128_LAYOUT
#undef HYLE_SIMD_VARIANT_SUFFIX

#if defined(HYLE_ARCH_ARM64)
#undef HYLE_SIMD_TARGET_SVE
#else
#undef HYLE_SIMD_TARGET_RVV
#endif

#endif
