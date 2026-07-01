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
// Highway's PRODUCTION dot (contrib/dot::Compute): 4 independent accumulators +
// manual 4x unroll -- the TRUE SOTA yardstick. dot_highway above is the naive
// single-accumulator teaching form; racing against it alone hid that a reduction
// kernel must break the FMA latency chain. This is the honest ceiling.
float dot_highway_contrib(const float* x, const float* y, std::size_t n);
#endif

// saxpy: y[i] = a*x[i] + y[i]. The canonical elementwise FMA kernel (BLAS level 1).
void saxpy_scalar(float a, const float* x, float* y, std::size_t n);
void saxpy_sse4(float a, const float* x, float* y, std::size_t n);
void saxpy_avx2(float a, const float* x, float* y, std::size_t n);
#if defined(HYLE_BENCH_HIGHWAY)
void saxpy_highway(float a, const float* x, float* y, std::size_t n);
#endif

// reduce_max: max(x[i]). A NON-sum horizontal reduction (Max + ReduceMax, no FMA) -- a
// different pattern than dot, to check the abstraction on a reduce that isn't sum.
float reduce_max_scalar(const float* x, std::size_t n);
float reduce_max_sse4(const float* x, std::size_t n);
float reduce_max_avx2(const float* x, std::size_t n);
#if defined(HYLE_BENCH_HIGHWAY)
float reduce_max_highway(const float* x, std::size_t n);
#endif

// scale: y[i] = a*x[i]. The SIMPLEST element-wise map (Mul + Store, no add/FMA) -- checks
// whether the vec.h abstraction is zero-cost on a trivial op the compiler auto-vectorizes
// perfectly, or whether it adds overhead the FMA-heavy dot/saxpy hid.
void scale_scalar(float a, const float* x, float* y, std::size_t n);
void scale_sse4(float a, const float* x, float* y, std::size_t n);
void scale_avx2(float a, const float* x, float* y, std::size_t n);
#if defined(HYLE_BENCH_HIGHWAY)
void scale_highway(float a, const float* x, float* y, std::size_t n);
#endif

}  // namespace hyle_bench
