// AVX2 variant of the dispatched kernels (compiled -mavx2 -mfma / MSVC /arch:AVX2). Only
// ENTERED when best_isa() >= kAvx2, so the -mavx2 codegen never runs on a non-AVX2 CPU.
#define HYLE_SIMD_TARGET_AVX2
#define HYLE_KERNEL_SUFFIX avx2
#include "src/simd/ops/kernel_impl.h"
