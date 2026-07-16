#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/simd/dtype.h"
#include "hyle/simd/vec.h"

// Target-bound whole-buffer operations. These functions have internal linkage on purpose: the
// implementation selected by vec.h is different in each target TU, so no target-dependent symbol
// crosses a translation-unit or shared-library boundary.

namespace hyle::simd::native {
namespace detail {

template <typename T>
static HYLE_FORCE_INLINE T dot_impl(const T* x, const T* y, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  auto acc0 = Zero(d);
  auto acc1 = Zero(d);
  auto acc2 = Zero(d);
  auto acc3 = Zero(d);
  std::size_t i = 0;
  const std::size_t width4 = width * 4;
  for (; i + width4 <= n; i += width4) {
    acc0 = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc0);
    acc1 = MulAdd(LoadU(d, x + i + width), LoadU(d, y + i + width), acc1);
    acc2 = MulAdd(LoadU(d, x + i + width * 2), LoadU(d, y + i + width * 2), acc2);
    acc3 = MulAdd(LoadU(d, x + i + width * 3), LoadU(d, y + i + width * 3), acc3);
  }
  for (; i + width <= n; i += width)
    acc0 = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc0);
  T sum = ReduceSum(d, Add(Add(acc0, acc1), Add(acc2, acc3)));
  for (; i < n; ++i) sum += x[i] * y[i];
  return sum;
}

static HYLE_FORCE_INLINE float dot_bf16_impl(const bfloat16_t* x,
                                             const bfloat16_t* y,
                                             std::size_t n) noexcept {
  using D16 = ScalableTag<std::uint16_t>;
  if constexpr (D16::kMaxLanes < 2) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
      sum0 += bf16_to_f32(x[i]) * bf16_to_f32(y[i]);
      sum1 += bf16_to_f32(x[i + 1]) * bf16_to_f32(y[i + 1]);
    }
    if (i < n) sum0 += bf16_to_f32(x[i]) * bf16_to_f32(y[i]);
    return sum0 + sum1;
  } else {
    static_assert(sizeof(bfloat16_t) == sizeof(std::uint16_t));
    const D16 d16;
    const Repartition<float, D16> df;
    const std::size_t width = Lanes(d16);
    auto sum0 = Zero(df);
    auto sum1 = Zero(df);
    auto sum2 = Zero(df);
    auto sum3 = Zero(df);
    std::size_t i = 0;
    for (; i + 2 * width <= n; i += 2 * width) {
      const auto a0 = LoadU(d16, reinterpret_cast<const std::uint16_t*>(x + i));
      const auto b0 = LoadU(d16, reinterpret_cast<const std::uint16_t*>(y + i));
      const auto a1 = LoadU(d16, reinterpret_cast<const std::uint16_t*>(x + i + width));
      const auto b1 = LoadU(d16, reinterpret_cast<const std::uint16_t*>(y + i + width));
      sum0 = MulEvenAdd(df, a0, b0, sum0);
      sum1 = MulOddAdd(df, a0, b0, sum1);
      sum2 = MulEvenAdd(df, a1, b1, sum2);
      sum3 = MulOddAdd(df, a1, b1, sum3);
    }
    for (; i + width <= n; i += width) {
      const auto a = LoadU(d16, reinterpret_cast<const std::uint16_t*>(x + i));
      const auto b = LoadU(d16, reinterpret_cast<const std::uint16_t*>(y + i));
      sum0 = MulEvenAdd(df, a, b, sum0);
      sum1 = MulOddAdd(df, a, b, sum1);
    }
    float sum = ReduceSum(df, Add(Add(sum0, sum1), Add(sum2, sum3)));
    for (; i < n; ++i) sum += bf16_to_f32(x[i]) * bf16_to_f32(y[i]);
    return sum;
  }
}

template <class D>
static HYLE_FORCE_INLINE VFromD<D> load_bf16_as_f32(D df,
                                                     const bfloat16_t* p) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  static_assert(sizeof(bfloat16_t) == sizeof(std::uint16_t));
  const Rebind<std::uint32_t, D> du32;
#if HYLE_SIMD_HAVE_SCALABLE
  // SVE/RVV do not have a fixed-width partial-register value for Rebind<u16, D>.
  // Load the low half of a full narrow vector and use the native half-widening
  // primitive instead. LoadN zeroes the unused half and, importantly, reads only
  // Lanes(df) bf16 elements at a whole-buffer tail boundary.
  const Simd<std::uint16_t, D::kMaxLanes * 2, 0> db;
  const auto bits16 = LoadN(db, reinterpret_cast<const std::uint16_t*>(p), Lanes(df));
  return BitCast(df, ShiftLeft<16>(PromoteLowerTo(du32, bits16)));
#else
  const Rebind<std::uint16_t, D> db;
  const auto bits16 = LoadU(db, reinterpret_cast<const std::uint16_t*>(p));
  return BitCast(df, ShiftLeft<16>(PromoteTo(du32, bits16)));
#endif
}

static HYLE_FORCE_INLINE float dot_f32_bf16_impl(const float* x,
                                                  const bfloat16_t* y,
                                                  std::size_t n) noexcept {
  using DF = ScalableTag<float>;
  if constexpr (DF::kMaxLanes < 2) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * bf16_to_f32(y[i]);
    return sum;
  } else {
    const DF df;
    const std::size_t width = Lanes(df);
    auto sum0 = Zero(df);
    auto sum1 = Zero(df);
    auto sum2 = Zero(df);
    auto sum3 = Zero(df);
    std::size_t i = 0;
    const std::size_t width4 = width * 4;
    for (; i + width4 <= n; i += width4) {
      sum0 = MulAdd(LoadU(df, x + i), load_bf16_as_f32(df, y + i), sum0);
      sum1 = MulAdd(LoadU(df, x + i + width),
                    load_bf16_as_f32(df, y + i + width), sum1);
      sum2 = MulAdd(LoadU(df, x + i + width * 2),
                    load_bf16_as_f32(df, y + i + width * 2), sum2);
      sum3 = MulAdd(LoadU(df, x + i + width * 3),
                    load_bf16_as_f32(df, y + i + width * 3), sum3);
    }
    for (; i + width <= n; i += width)
      sum0 = MulAdd(LoadU(df, x + i), load_bf16_as_f32(df, y + i), sum0);
    float sum = ReduceSum(df, Add(Add(sum0, sum1), Add(sum2, sum3)));
    for (; i < n; ++i) sum += x[i] * bf16_to_f32(y[i]);
    return sum;
  }
}

static HYLE_FORCE_INLINE float dot_f16_impl(const float16_t* x,
                                             const float16_t* y,
                                             std::size_t n) noexcept {
#if (defined(HYLE_SIMD_TARGET_AVX2) || defined(HYLE_SIMD_TARGET_AVX512)) && \
    (defined(__F16C__) || defined(_MSC_VER))
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256 x0 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    const __m256 y0 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i)));
    const __m256 x1 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 8)));
    const __m256 y1 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i + 8)));
    const __m256 x2 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 16)));
    const __m256 y2 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i + 16)));
    const __m256 x3 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 24)));
    const __m256 y3 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i + 24)));
    sum0 = _mm256_fmadd_ps(x0, y0, sum0);
    sum1 = _mm256_fmadd_ps(x1, y1, sum1);
    sum2 = _mm256_fmadd_ps(x2, y2, sum2);
    sum3 = _mm256_fmadd_ps(x3, y3, sum3);
  }
  for (; i + 8 <= n; i += 8) {
    const __m256 vx = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    const __m256 vy = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i)));
    sum0 = _mm256_fmadd_ps(vx, vy, sum0);
  }
  const __m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1),
                                     _mm256_add_ps(sum2, sum3));
  __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(total),
                              _mm256_extractf128_ps(total, 1));
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  float sum = _mm_cvtss_f32(reduced);
  for (; i < n; ++i) sum += f16_to_f32(x[i]) * f16_to_f32(y[i]);
  return sum;
#elif defined(HYLE_SIMD_TARGET_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const float16x8_t x0 = vreinterpretq_f16_u16(
        vld1q_u16(reinterpret_cast<const std::uint16_t*>(x + i)));
    const float16x8_t y0 = vreinterpretq_f16_u16(
        vld1q_u16(reinterpret_cast<const std::uint16_t*>(y + i)));
    const float16x8_t x1 = vreinterpretq_f16_u16(
        vld1q_u16(reinterpret_cast<const std::uint16_t*>(x + i + 8)));
    const float16x8_t y1 = vreinterpretq_f16_u16(
        vld1q_u16(reinterpret_cast<const std::uint16_t*>(y + i + 8)));
    sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(x0)),
                     vcvt_f32_f16(vget_low_f16(y0)));
    sum1 = vfmaq_f32(sum1, vcvt_f32_f16(vget_high_f16(x0)),
                     vcvt_f32_f16(vget_high_f16(y0)));
    sum2 = vfmaq_f32(sum2, vcvt_f32_f16(vget_low_f16(x1)),
                     vcvt_f32_f16(vget_low_f16(y1)));
    sum3 = vfmaq_f32(sum3, vcvt_f32_f16(vget_high_f16(x1)),
                     vcvt_f32_f16(vget_high_f16(y1)));
  }
  for (; i + 8 <= n; i += 8) {
    const float16x8_t vx = vreinterpretq_f16_u16(
        vld1q_u16(reinterpret_cast<const std::uint16_t*>(x + i)));
    const float16x8_t vy = vreinterpretq_f16_u16(
        vld1q_u16(reinterpret_cast<const std::uint16_t*>(y + i)));
    sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(vx)),
                     vcvt_f32_f16(vget_low_f16(vy)));
    sum1 = vfmaq_f32(sum1, vcvt_f32_f16(vget_high_f16(vx)),
                     vcvt_f32_f16(vget_high_f16(vy)));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1),
                                    vaddq_f32(sum2, sum3)));
  for (; i < n; ++i) sum += f16_to_f32(x[i]) * f16_to_f32(y[i]);
  return sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    sum0 += f16_to_f32(x[i]) * f16_to_f32(y[i]);
    sum1 += f16_to_f32(x[i + 1]) * f16_to_f32(y[i + 1]);
  }
  if (i < n) sum0 += f16_to_f32(x[i]) * f16_to_f32(y[i]);
  return sum0 + sum1;
#endif
}

static HYLE_FORCE_INLINE std::int64_t dot_i16_impl(const std::int16_t* x,
                                                    const std::int16_t* y,
                                                    std::size_t n) noexcept {
  using D16 = ScalableTag<std::int16_t>;
  if constexpr (D16::kMaxLanes < 2) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t product = static_cast<std::int64_t>(x[i]) * y[i];
      sum += static_cast<std::uint64_t>(product);
    }
    return std::bit_cast<std::int64_t>(sum);
  } else {
    using D32 = RepartitionToWide<D16>;
    using D64 = RepartitionToWide<D32>;
    using DU64 = RebindToUnsigned<D64>;
    const D16 d16;
    const D64 d64;
    const DU64 du64;
    auto sum0 = Zero(du64);
    auto sum1 = Zero(du64);
    auto sum2 = Zero(du64);
    auto sum3 = Zero(du64);
    const std::size_t width = Lanes(d16);
    std::size_t i = 0;
    for (; i + width <= n; i += width) {
      const auto a = LoadU(d16, x + i);
      const auto b = LoadU(d16, y + i);
      const auto even = MulEven(a, b);
      const auto odd = MulOdd(a, b);
      sum0 = Add(sum0, BitCast(du64, PromoteLowerTo(d64, even)));
      sum1 = Add(sum1, BitCast(du64, PromoteUpperTo(d64, even)));
      sum2 = Add(sum2, BitCast(du64, PromoteLowerTo(d64, odd)));
      sum3 = Add(sum3, BitCast(du64, PromoteUpperTo(d64, odd)));
    }
    std::uint64_t sum = ReduceSum(du64, Add(Add(sum0, sum1), Add(sum2, sum3)));
    for (; i < n; ++i) {
      const std::int64_t product = static_cast<std::int64_t>(x[i]) * y[i];
      sum += static_cast<std::uint64_t>(product);
    }
    return std::bit_cast<std::int64_t>(sum);
  }
}

template <typename T>
static HYLE_FORCE_INLINE void saxpy_impl(T a, const T* x, T* y,
                                         std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto va = Set(d, a);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    StoreU(MulAdd(va, LoadU(d, x + i), LoadU(d, y + i)), d, y + i);
  }
  for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

template <typename T>
static HYLE_FORCE_INLINE void add_impl(const T* x, const T* y, T* out,
                                       std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width)
    StoreU(Add(LoadU(d, x + i), LoadU(d, y + i)), d, out + i);
  for (; i < n; ++i) out[i] = x[i] + y[i];
}

}  // namespace detail

static HYLE_FORCE_INLINE float dot(const float* x, const float* y,
                                   std::size_t n) noexcept {
  return detail::dot_impl(x, y, n);
}
static HYLE_FORCE_INLINE double dot(const double* x, const double* y,
                                    std::size_t n) noexcept {
  return detail::dot_impl(x, y, n);
}
static HYLE_FORCE_INLINE float dot(const bfloat16_t* x, const bfloat16_t* y,
                                   std::size_t n) noexcept {
  return detail::dot_bf16_impl(x, y, n);
}
static HYLE_FORCE_INLINE float dot(const float* x, const bfloat16_t* y,
                                   std::size_t n) noexcept {
  return detail::dot_f32_bf16_impl(x, y, n);
}
static HYLE_FORCE_INLINE float dot(const bfloat16_t* x, const float* y,
                                   std::size_t n) noexcept {
  return detail::dot_f32_bf16_impl(y, x, n);
}
static HYLE_FORCE_INLINE float dot(const float16_t* x, const float16_t* y,
                                   std::size_t n) noexcept {
  return detail::dot_f16_impl(x, y, n);
}
static HYLE_FORCE_INLINE std::int64_t dot(const std::int16_t* x,
                                          const std::int16_t* y,
                                          std::size_t n) noexcept {
  return detail::dot_i16_impl(x, y, n);
}
static HYLE_FORCE_INLINE void saxpy(float a, const float* x, float* y,
                                    std::size_t n) noexcept {
  detail::saxpy_impl(a, x, y, n);
}
static HYLE_FORCE_INLINE void saxpy(double a, const double* x, double* y,
                                    std::size_t n) noexcept {
  detail::saxpy_impl(a, x, y, n);
}
static HYLE_FORCE_INLINE void add(const float* x, const float* y, float* out,
                                  std::size_t n) noexcept {
  detail::add_impl(x, y, out, n);
}
static HYLE_FORCE_INLINE void add(const double* x, const double* y, double* out,
                                  std::size_t n) noexcept {
  detail::add_impl(x, y, out, n);
}

}  // namespace hyle::simd::native
