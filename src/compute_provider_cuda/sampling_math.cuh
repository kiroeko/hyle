#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace hyle::compute::cuda_native::sampling::math {
// Evaluate from the original integer words on the smaller side of U. The
// upper endpoint retains a positive exponential even when F32 U rounds to 1.
__device__ inline float gumbel(std::uint32_t hi, std::uint32_t lo) {
  const bool upper = (hi & 0x80000000U) != 0;
  if (upper) { hi = ~hi; lo = ~lo; }
  const auto small = float(hi) * 0x1p-32f + (float(lo) + .5F) * 0x1p-64f;
  const auto exponential = upper ? -log1pf(-small) : -logf(small);
  return -logf(exponential);
}
} // namespace hyle::compute::cuda_native::sampling::math
