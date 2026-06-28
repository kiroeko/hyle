// Scalar baseline kernels -- plain C loops, compiled at the DEFAULT x86-64 baseline (no
// -march), so the compiler auto-vectorizes to at most SSE2. This is the honest "what you
// get for free from the optimizer" reference the perf doc compares against (SIMD vs
// auto-vec = 1.5-4x). NOT compiled with AVX2, or it would just match the HYLE avx2 kernel.
#include "bench/simd/simd_kernels.h"

namespace hyle_bench {

float dot_scalar(const float* x, const float* y, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) sum += x[i] * y[i];
  return sum;
}

void saxpy_scalar(float a, const float* x, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

}  // namespace hyle_bench
