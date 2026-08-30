// Same-contract peers for the ordinary pointer/count benchmark. HYLE itself is
// always entered through its shipping public API; this file owns only the
// scalar auto-vectorization baseline and optional locked-Highway peer.
#pragma once

#include <cstddef>

namespace hyle_bench {

// dot product: sum(x[i]*y[i]). The canonical reduction + FMA kernel.
float dot_scalar(const float* x, const float* y, std::size_t n);
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
#if defined(HYLE_BENCH_HIGHWAY)
void saxpy_highway(float a, const float* x, float* y, std::size_t n);
#endif

// scale: y[i] = a*x[i], a simple bandwidth-sensitive element-wise map.
void scale_scalar(float a, const float* x, float* y, std::size_t n);
#if defined(HYLE_BENCH_HIGHWAY)
void scale_highway(float a, const float* x, float* y, std::size_t n);
#endif

}  // namespace hyle_bench
