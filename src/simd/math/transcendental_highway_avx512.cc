// Fixed AVX-512 batch/SoA façade over the locked Highway mathematics core.
// Highway excludes its AVX3 targets under classic cl.exe, so this TU is only
// part of GCC/Clang builds; MSVC intentionally uses the AVX2 math kernel.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64) && !defined(HYLE_SIMD_CLASSIC_CL_AVX512_FALLBACK)

#define HWY_COMPILE_ONLY_STATIC 1
#define HYLE_SIMD_TARGET_AVX512 1
#define HYLE_HIGHWAY_MATH_SUFFIX avx512
#include "src/simd/math/highway_math_kernel-inl.h"
#undef HYLE_HIGHWAY_MATH_SUFFIX
#undef HYLE_SIMD_TARGET_AVX512
#undef HWY_COMPILE_ONLY_STATIC

#endif // HYLE_ARCH_X64 && !HYLE_SIMD_CLASSIC_CL_AVX512_FALLBACK
