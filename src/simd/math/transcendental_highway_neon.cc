// AArch64 NEON batch/SoA façade over the locked Highway mathematics core.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64)

#define HWY_COMPILE_ONLY_STATIC 1
#define HYLE_SIMD_TARGET_NEON 1
#define HYLE_HIGHWAY_MATH_SUFFIX neon
#include "src/simd/math/highway_math_kernel-inl.h"
#undef HYLE_HIGHWAY_MATH_SUFFIX
#undef HYLE_SIMD_TARGET_NEON
#undef HWY_COMPILE_ONLY_STATIC

#endif  // HYLE_ARCH_ARM64
