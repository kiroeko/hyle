// simd_kernels.h -- the bench kernels, declared once. Each is implemented in its OWN
// TU compiled with a different ISA target (see simd_kernels_*.cc), so the per-ISA codegen
// is honestly isolated -- no single TU sees two backends. The bench main (simd_bench.cc)
// links them all and races them via nanobench .relative().
#pragma once

#include <cstddef>

namespace hyle_bench {

// dot product: sum(x[i]*y[i]). The canonical reduction + FMA kernel.
float dot_scalar(const float* x, const float* y, std::size_t n);
float dot_sse4(const float* x, const float* y, std::size_t n);
float dot_avx2(const float* x, const float* y, std::size_t n);
#if defined(HYLE_BENCH_HIGHWAY)
float dot_highway(const float* x, const float* y, std::size_t n);
#endif

// saxpy: y[i] = a*x[i] + y[i]. The canonical elementwise FMA kernel (BLAS level 1).
void saxpy_scalar(float a, const float* x, float* y, std::size_t n);
void saxpy_sse4(float a, const float* x, float* y, std::size_t n);
void saxpy_avx2(float a, const float* x, float* y, std::size_t n);
#if defined(HYLE_BENCH_HIGHWAY)
void saxpy_highway(float a, const float* x, float* y, std::size_t n);
#endif

}  // namespace hyle_bench
