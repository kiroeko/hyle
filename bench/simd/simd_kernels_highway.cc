// Highway reference kernels -- the SOTA yardstick (simd-design names Highway the gold
// standard). Built only with -DHYLE_BENCH_HIGHWAY=ON (locates the cloned ../highway).
// Compiled with the SAME -mavx2 -mfma as the HYLE avx2 kernel, using Highway's STATIC
// dispatch (single target = HWY_STATIC_TARGET), so this is HYLE-avx2 vs Highway-avx2
// head to head: same ISA, same kernel shape, only the abstraction differs.
#if defined(HYLE_BENCH_HIGHWAY)

#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "bench/simd/simd_kernels.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle_bench {

float dot_highway(const float* x, const float* y, std::size_t n) {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  auto acc = hn::Zero(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    acc = hn::MulAdd(hn::LoadU(d, x + i), hn::LoadU(d, y + i), acc);
  float sum = hn::ReduceSum(d, acc);
  for (; i < n; ++i) sum += x[i] * y[i];
  return sum;
}

// Highway's PRODUCTION dot (contrib/dot::Compute) -- 4 accumulators + 4x unroll,
// the true SOTA. kAtLeastOneVector: n=4096 >> N, safe. This is what dot_highway
// (single-acc teaching form) should have been benched against all along.
float dot_highway_contrib(const float* x, const float* y, std::size_t n) {
  const hn::ScalableTag<float> d;
  return hn::Dot::Compute<hwy::HWY_NAMESPACE::Dot::kAtLeastOneVector>(d, x, y, n);
}

void saxpy_highway(float a, const float* x, float* y, std::size_t n) {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  const auto va = hn::Set(d, a);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto r = hn::MulAdd(va, hn::LoadU(d, x + i), hn::LoadU(d, y + i));
    hn::StoreU(r, d, y + i);
  }
  for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

void scale_highway(float a, const float* x, float* y, std::size_t n) {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  const auto va = hn::Set(d, a);
  std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Mul(va, hn::LoadU(d, x + i)), d, y + i);
  for (; i < n; ++i) y[i] = a * x[i];
}

}  // namespace hyle_bench

#endif  // HYLE_BENCH_HIGHWAY
