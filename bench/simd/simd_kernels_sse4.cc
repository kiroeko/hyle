// SSE4.2 variant of the HYLE bench kernels (compiled with -msse4.2 / MSVC default).
#define HYLE_SIMD_TARGET_SSE4
#define HYLE_BENCH_VARIANT_SUFFIX sse4
#include "bench/simd/simd_kernels_hyle_impl.h"
#undef HYLE_BENCH_VARIANT_SUFFIX
#undef HYLE_SIMD_TARGET_SSE4
