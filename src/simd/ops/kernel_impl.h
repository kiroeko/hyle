// kernel_impl.h -- the SHARED kernel bodies, written ONCE against the D-tag abstraction
// (ScalableTag idiom, no intrinsics). Each per-ISA TU #defines HYLE_SIMD_TARGET_* +
// HYLE_KERNEL_SUFFIX then includes this, so the SAME source compiles to
// detail::dot_f32_sse4 / detail::dot_f32_avx2 / detail::dot_f32_scalar. The backend +
// -march make the loop 4-wide (SSE), 8-wide+FMA (AVX2), or scalar -- the kernel author
// writes one loop. These are the functions the dispatcher (dispatch_ops.cc) selects among.

#include <cstddef>

#include "hyle/simd/vec.h"
#include "src/simd/ops/kernels.h"

#ifndef HYLE_KERNEL_SUFFIX
#error "define HYLE_KERNEL_SUFFIX (scalar|sse4|avx2) before including kernel_impl.h"
#endif

#define HYLE_SIMD_CAT_(a, b) a##b
#define HYLE_SIMD_CAT(a, b) HYLE_SIMD_CAT_(a, b)
#define DOT_FN HYLE_SIMD_CAT(dot_f32_, HYLE_KERNEL_SUFFIX)
#define SAXPY_FN HYLE_SIMD_CAT(saxpy_f32_, HYLE_KERNEL_SUFFIX)

namespace hyle::simd::detail {

float DOT_FN(const float* x, const float* y, std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t w = Lanes(d);
  auto acc = Zero(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    acc = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc);  // FMA on AVX2
  float sum = ReduceSum(d, acc);
  for (; i < n; ++i) sum += x[i] * y[i];  // scalar tail (n not a multiple of w)
  return sum;
}

void SAXPY_FN(float a, const float* x, float* y, std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t w = Lanes(d);
  const auto va = Set(d, a);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto r = MulAdd(va, LoadU(d, x + i), LoadU(d, y + i));
    StoreU(r, d, y + i);
  }
  for (; i < n; ++i) y[i] = a * x[i] + y[i];  // tail
}

}  // namespace hyle::simd::detail
