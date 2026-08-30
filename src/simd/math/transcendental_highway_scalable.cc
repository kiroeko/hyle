// Length-agnostic SVE/RVV batch/SoA façade over the locked Highway mathematics
// core. CMake compiles this TU once for the selected scalable target.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_SIMD_TARGET_SVE 1
#define HYLE_HIGHWAY_MATH_SUFFIX sve
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_SIMD_TARGET_RVV 1
#define HYLE_HIGHWAY_MATH_SUFFIX rvv
#endif

#if defined(HYLE_HIGHWAY_MATH_SUFFIX)

#define HWY_COMPILE_ONLY_STATIC 1
#include "src/simd/math/highway_math_kernel-inl.h"
#undef HWY_COMPILE_ONLY_STATIC
#undef HYLE_HIGHWAY_MATH_SUFFIX
#if defined(HYLE_SIMD_TARGET_SVE)
#undef HYLE_SIMD_TARGET_SVE
#elif defined(HYLE_SIMD_TARGET_RVV)
#undef HYLE_SIMD_TARGET_RVV
#endif

#endif
