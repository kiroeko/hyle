// Fixed AVX2 batch/SoA façade over the locked Highway mathematics core.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)

#define HWY_COMPILE_ONLY_STATIC 1
#define HYLE_SIMD_TARGET_AVX2 1
#define HYLE_HIGHWAY_MATH_SUFFIX avx2
#include "src/simd/math/highway_math_kernel-inl.h"
#undef HYLE_HIGHWAY_MATH_SUFFIX
#undef HYLE_SIMD_TARGET_AVX2
#undef HWY_COMPILE_ONLY_STATIC

#endif  // HYLE_ARCH_X64
