// AVX2 variant of the HYLE bench kernels (compiled with -mavx2 -mfma / MSVC /arch:AVX2).
#define HYLE_SIMD_TARGET_AVX2
#define HYLE_BENCH_VARIANT_SUFFIX avx2
#include "bench/simd/simd_kernels_hyle_impl.h"
#undef HYLE_BENCH_VARIANT_SUFFIX
#undef HYLE_SIMD_TARGET_AVX2
