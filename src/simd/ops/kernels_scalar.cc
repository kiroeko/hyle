// Scalar floor variant of the dispatched kernels. NO -march: the portable backend, the
// always-safe fallback the dispatcher uses when the CPU has no usable SIMD tier.
#define HYLE_KERNEL_SUFFIX scalar
#include "src/simd/ops/kernel_impl.h"
