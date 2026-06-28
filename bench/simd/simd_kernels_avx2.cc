// AVX2 variant of the HYLE bench kernels (compiled with -mavx2 -mfma / MSVC /arch:AVX2).
#define HYLE_SIMD_TARGET_AVX2
#define HYLE_KERNEL_SUFFIX avx2
#include "bench/simd/simd_kernels_hyle_impl.h"
