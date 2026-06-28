// SSE4.2 variant of the dispatched kernels (compiled -msse4.2 / MSVC default x64).
#define HYLE_SIMD_TARGET_SSE4
#define HYLE_KERNEL_SUFFIX sse4
#include "src/simd/ops/kernel_impl.h"
