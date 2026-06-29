// simd_kernels_hyle_impl.h -- shared HYLE kernel bodies, written ONCE against the D-tag
// abstraction (ScalableTag idiom). Each per-ISA TU (simd_kernels_sse4.cc / _avx2.cc)
// #defines HYLE_SIMD_TARGET_* + HYLE_KERNEL_SUFFIX then includes this, so the SAME source
// compiles to dot_sse4/dot_avx2 etc. -- demonstrating the abstraction's portability AND
// giving honest per-ISA codegen (one backend per TU).
//
// This is the whole point: the kernel author writes ONE loop; the backend + -march make
// it 4-wide (SSE) or 8-wide+FMA (AVX2). No intrinsics in the kernel.

#include <cstddef>

#include "hyle/simd/vec.h"
#include "bench/simd/simd_kernels.h"

#ifndef HYLE_KERNEL_SUFFIX
#error "define HYLE_KERNEL_SUFFIX (e.g. sse4) before including"
#endif

#define HYLE_BENCH_CAT_(a, b) a##b
#define HYLE_BENCH_CAT(a, b) HYLE_BENCH_CAT_(a, b)
#define DOT_FN HYLE_BENCH_CAT(dot_, HYLE_KERNEL_SUFFIX)
#define SAXPY_FN HYLE_BENCH_CAT(saxpy_, HYLE_KERNEL_SUFFIX)
#define REDUCE_MAX_FN HYLE_BENCH_CAT(reduce_max_, HYLE_KERNEL_SUFFIX)
#define SCALE_FN HYLE_BENCH_CAT(scale_, HYLE_KERNEL_SUFFIX)

namespace hyle_bench {

float DOT_FN(const float* x, const float* y, std::size_t n) {
  namespace hs = hyle::simd;
  const hs::ScalableTag<float> d;
  const std::size_t w = hs::Lanes(d);
  auto acc = hs::Zero(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    acc = hs::MulAdd(hs::LoadU(d, x + i), hs::LoadU(d, y + i), acc);  // FMA on AVX2
  float sum = hs::ReduceSum(d, acc);
  for (; i < n; ++i) sum += x[i] * y[i];  // tail
  return sum;
}

void SAXPY_FN(float a, const float* x, float* y, std::size_t n) {
  namespace hs = hyle::simd;
  const hs::ScalableTag<float> d;
  const std::size_t w = hs::Lanes(d);
  const auto va = hs::Set(d, a);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto r = hs::MulAdd(va, hs::LoadU(d, x + i), hs::LoadU(d, y + i));
    hs::StoreU(r, d, y + i);
  }
  for (; i < n; ++i) y[i] = a * x[i] + y[i];  // tail
}

float REDUCE_MAX_FN(const float* x, std::size_t n) {
  namespace hs = hyle::simd;
  const hs::ScalableTag<float> d;
  const std::size_t w = hs::Lanes(d);
  if (n == 0) return 0.0f;
  auto acc = hs::Set(d, x[0]);
  std::size_t i = 0;
  for (; i + w <= n; i += w) acc = hs::Max(acc, hs::LoadU(d, x + i));
  float m = hs::ReduceMax(d, acc);
  for (; i < n; ++i) m = x[i] > m ? x[i] : m;  // tail
  return m;
}

void SCALE_FN(float a, const float* x, float* y, std::size_t n) {
  namespace hs = hyle::simd;
  const hs::ScalableTag<float> d;
  const std::size_t w = hs::Lanes(d);
  const auto va = hs::Set(d, a);
  std::size_t i = 0;
  for (; i + w <= n; i += w) hs::StoreU(hs::Mul(va, hs::LoadU(d, x + i)), d, y + i);
  for (; i < n; ++i) y[i] = a * x[i];  // tail
}

}  // namespace hyle_bench
