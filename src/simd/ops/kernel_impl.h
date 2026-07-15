// kernel_impl.h -- the SHARED kernel bodies, written ONCE against the D-tag abstraction
// (ScalableTag idiom, no intrinsics). Each per-ISA TU #defines HYLE_SIMD_TARGET_* +
// HYLE_KERNEL_SUFFIX then includes this, so the SAME source compiles to
// detail::dot_f32_sse4 / detail::dot_f32_avx2 / detail::dot_f32_scalar. The backend +
// -march make the loop 4-wide (SSE), 8-wide+FMA (AVX2), or scalar -- the kernel author
// writes one loop. These are the functions the dispatcher (dispatch_ops.cc) selects among.
//
// Both f32 and f64 instantiate the same body: a T-generic core does the work, and the
// per-type entry symbols (dot_f32_avx2 / dot_f64_avx2 / ...) forward to it. f64 halves the
// lane count (AVX2: 4 doubles vs 8 floats) but the D-tag loop is identical.

#include <cstddef>

#include "hyle/simd/vec.h"
#include "src/simd/ops/kernels.h"

#ifndef HYLE_KERNEL_SUFFIX
#error "define HYLE_KERNEL_SUFFIX (scalar|sse4|avx2|avx512) before including kernel_impl.h"
#endif

#define HYLE_SIMD_CAT_(a, b) a##b
#define HYLE_SIMD_CAT(a, b) HYLE_SIMD_CAT_(a, b)
#define DOT_F32_FN HYLE_SIMD_CAT(dot_f32_, HYLE_KERNEL_SUFFIX)
#define SAXPY_F32_FN HYLE_SIMD_CAT(saxpy_f32_, HYLE_KERNEL_SUFFIX)
#define DOT_F64_FN HYLE_SIMD_CAT(dot_f64_, HYLE_KERNEL_SUFFIX)
#define SAXPY_F64_FN HYLE_SIMD_CAT(saxpy_f64_, HYLE_KERNEL_SUFFIX)
#define ADD_F32_FN HYLE_SIMD_CAT(add_f32_, HYLE_KERNEL_SUFFIX)
#define ADD_F64_FN HYLE_SIMD_CAT(add_f64_, HYLE_KERNEL_SUFFIX)
#define SUM_F32_ACC_FN HYLE_SIMD_CAT(sum_f32_accurate_, HYLE_KERNEL_SUFFIX)
#define DOT_F32_ACC_FN HYLE_SIMD_CAT(dot_f32_accurate_, HYLE_KERNEL_SUFFIX)

namespace hyle::simd::detail {

// T-generic dot: FOUR independent accumulators, not one. A single acc chains every FMA into
// the next (acc = MulAdd(.., acc)), so throughput is bound by FMA LATENCY (~4 cyc) -- one FMA
// retires every 4 cycles. Modern x86 has 2 FMA ports, so 4 independent chains keep ~8 FMAs in
// flight and approach 2 FMA/cycle (load ports become the limit, not latency). This mirrors
// Highway's contrib/dot ("2 FMA ports * 4 cycle latency = up to 8 in-flight ... unroll 4x").
// fp add is non-associative so the compiler will NOT split one acc into four on its own -- the
// unroll must be explicit. The 4-way partial-sum ordering is a fixed, documented reduction
// (deterministic across runs), just not the same rounding as a strict sequential sum. f64 uses
// the identical body at half the lane width -- the wider accumulator also means less rounding
// drift, which is exactly why solver/geometry codes want the f64 path.
template <typename T>
HYLE_FORCE_INLINE T dot_kernel(const T* x, const T* y, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  auto acc0 = Zero(d), acc1 = Zero(d), acc2 = Zero(d), acc3 = Zero(d);
  std::size_t i = 0;
  const std::size_t w4 = w * 4;
  for (; i + w4 <= n; i += w4) {
    acc0 = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc0);
    acc1 = MulAdd(LoadU(d, x + i + w), LoadU(d, y + i + w), acc1);
    acc2 = MulAdd(LoadU(d, x + i + w * 2), LoadU(d, y + i + w * 2), acc2);
    acc3 = MulAdd(LoadU(d, x + i + w * 3), LoadU(d, y + i + w * 3), acc3);
  }
  for (; i + w <= n; i += w)  // up to 3 whole vectors
    acc0 = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc0);
  T sum = ReduceSum(d, Add(Add(acc0, acc1), Add(acc2, acc3)));  // reduction tree
  for (; i < n; ++i) sum += x[i] * y[i];  // scalar tail (n not a multiple of w)
  return sum;
}

template <typename T>
HYLE_FORCE_INLINE void saxpy_kernel(T a, const T* x, T* y, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  const auto va = Set(d, a);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto r = MulAdd(va, LoadU(d, x + i), LoadU(d, y + i));
    StoreU(r, d, y + i);
  }
  for (; i < n; ++i) y[i] = a * x[i] + y[i];  // tail
}

template <typename T>
HYLE_FORCE_INLINE void add_kernel(const T* x, const T* y, T* z, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) StoreU(Add(LoadU(d, x + i), LoadU(d, y + i)), d, z + i);
  for (; i < n; ++i) z[i] = x[i] + y[i];  // tail
}

// Accurate f32 sum/dot: promote each f32 vector half to f64, accumulate with f64 FMA/Add.
// Final cast to f32. Error ~n·ε_f64 (vs fast path n·ε_f32). Same width loop as the fast path;
// extra cost is the promote + half-width f64 ops, not a scalar walk.
HYLE_FORCE_INLINE float sum_f32_accurate_kernel(const float* x, std::size_t n) noexcept {
  const ScalableTag<float> df;
  const ScalableTag<double> dd;
  const std::size_t wf = Lanes(df);
  auto acc0 = Zero(dd), acc1 = Zero(dd), acc2 = Zero(dd), acc3 = Zero(dd);
  std::size_t i = 0;
  const std::size_t w2 = wf * 2;
  for (; i + w2 <= n; i += w2) {
    const auto v0 = LoadU(df, x + i);
    const auto v1 = LoadU(df, x + i + wf);
    acc0 = Add(acc0, PromoteLowerTo(dd, v0));
    acc1 = Add(acc1, PromoteUpperTo(dd, v0));
    acc2 = Add(acc2, PromoteLowerTo(dd, v1));
    acc3 = Add(acc3, PromoteUpperTo(dd, v1));
  }
  for (; i + wf <= n; i += wf) {
    const auto v = LoadU(df, x + i);
    acc0 = Add(acc0, PromoteLowerTo(dd, v));
    acc1 = Add(acc1, PromoteUpperTo(dd, v));
  }
  double sum = ReduceSum(dd, Add(Add(acc0, acc1), Add(acc2, acc3)));
  for (; i < n; ++i) sum += static_cast<double>(x[i]);
  return static_cast<float>(sum);
}

HYLE_FORCE_INLINE float dot_f32_accurate_kernel(const float* x, const float* y,
                                                std::size_t n) noexcept {
  const ScalableTag<float> df;
  const ScalableTag<double> dd;
  const std::size_t wf = Lanes(df);
  auto acc0 = Zero(dd), acc1 = Zero(dd), acc2 = Zero(dd), acc3 = Zero(dd);
  std::size_t i = 0;
  const std::size_t w2 = wf * 2;
  for (; i + w2 <= n; i += w2) {
    const auto x0 = LoadU(df, x + i), y0 = LoadU(df, y + i);
    const auto x1 = LoadU(df, x + i + wf), y1 = LoadU(df, y + i + wf);
    acc0 = MulAdd(PromoteLowerTo(dd, x0), PromoteLowerTo(dd, y0), acc0);
    acc1 = MulAdd(PromoteUpperTo(dd, x0), PromoteUpperTo(dd, y0), acc1);
    acc2 = MulAdd(PromoteLowerTo(dd, x1), PromoteLowerTo(dd, y1), acc2);
    acc3 = MulAdd(PromoteUpperTo(dd, x1), PromoteUpperTo(dd, y1), acc3);
  }
  for (; i + wf <= n; i += wf) {
    const auto xv = LoadU(df, x + i), yv = LoadU(df, y + i);
    acc0 = MulAdd(PromoteLowerTo(dd, xv), PromoteLowerTo(dd, yv), acc0);
    acc1 = MulAdd(PromoteUpperTo(dd, xv), PromoteUpperTo(dd, yv), acc1);
  }
  double sum = ReduceSum(dd, Add(Add(acc0, acc1), Add(acc2, acc3)));
  for (; i < n; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
  return static_cast<float>(sum);
}

float DOT_F32_FN(const float* x, const float* y, std::size_t n) noexcept { return dot_kernel<float>(x, y, n); }
void SAXPY_F32_FN(float a, const float* x, float* y, std::size_t n) noexcept { saxpy_kernel<float>(a, x, y, n); }
void ADD_F32_FN(const float* x, const float* y, float* z, std::size_t n) noexcept { add_kernel<float>(x, y, z, n); }
double DOT_F64_FN(const double* x, const double* y, std::size_t n) noexcept { return dot_kernel<double>(x, y, n); }
void SAXPY_F64_FN(double a, const double* x, double* y, std::size_t n) noexcept { saxpy_kernel<double>(a, x, y, n); }
void ADD_F64_FN(const double* x, const double* y, double* z, std::size_t n) noexcept { add_kernel<double>(x, y, z, n); }
float SUM_F32_ACC_FN(const float* x, std::size_t n) noexcept { return sum_f32_accurate_kernel(x, n); }
float DOT_F32_ACC_FN(const float* x, const float* y, std::size_t n) noexcept {
  return dot_f32_accurate_kernel(x, y, n);
}

}  // namespace hyle::simd::detail
