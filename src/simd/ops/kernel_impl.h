// Direct-Highway per-target row kernels. Included once by every generated target TU. HYLE owns
// pointer/count and numeric contracts; Highway owns register types and target specialization.

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

#include "hyle/simd/dtype.h"
#include "src/simd/dtype/highway_narrow.h"
#include "src/simd/ops/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including ops/kernel_impl.h"
#endif

#define HYLE_SIMD_OPS_CAT_(a, b) a##b
#define HYLE_SIMD_OPS_CAT(a, b) HYLE_SIMD_OPS_CAT_(a, b)
#define HYLE_SIMD_OPS_FN(name) HYLE_SIMD_OPS_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE T dot_real(const T* x, const T* y, std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(d);
  auto sum1 = hn::Zero(d);
  auto sum2 = hn::Zero(d);
  auto sum3 = hn::Zero(d);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::MulAdd(hn::LoadU(d, x + i), hn::LoadU(d, y + i), sum0);
    sum1 = hn::MulAdd(hn::LoadU(d, x + i + lanes),
                      hn::LoadU(d, y + i + lanes), sum1);
    sum2 = hn::MulAdd(hn::LoadU(d, x + i + lanes * 2),
                      hn::LoadU(d, y + i + lanes * 2), sum2);
    sum3 = hn::MulAdd(hn::LoadU(d, x + i + lanes * 3),
                      hn::LoadU(d, y + i + lanes * 3), sum3);
  }
  for (; lanes <= n - i; i += lanes) {
    sum0 = hn::MulAdd(hn::LoadU(d, x + i), hn::LoadU(d, y + i), sum0);
  }
  T result = hn::ReduceSum(d, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += x[i] * y[i];
  return result;
}

template <typename HNarrow, typename Storage, class D>
HWY_INLINE auto load_narrow_as_f32(D df, const Storage* p) noexcept {
  return highway_narrow::load_as_f32<HNarrow>(df, p);
}

template <typename HNarrow, typename Storage, typename Decode>
HWY_INLINE float dot_narrow(const Storage* x, const Storage* y, std::size_t n,
                            Decode decode) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(df);
  auto sum1 = hn::Zero(df);
  auto sum2 = hn::Zero(df);
  auto sum3 = hn::Zero(df);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::MulAdd(load_narrow_as_f32<HNarrow>(df, x + i),
                      load_narrow_as_f32<HNarrow>(df, y + i), sum0);
    sum1 = hn::MulAdd(load_narrow_as_f32<HNarrow>(df, x + i + lanes),
                      load_narrow_as_f32<HNarrow>(df, y + i + lanes), sum1);
    sum2 = hn::MulAdd(load_narrow_as_f32<HNarrow>(df, x + i + lanes * 2),
                      load_narrow_as_f32<HNarrow>(df, y + i + lanes * 2), sum2);
    sum3 = hn::MulAdd(load_narrow_as_f32<HNarrow>(df, x + i + lanes * 3),
                      load_narrow_as_f32<HNarrow>(df, y + i + lanes * 3), sum3);
  }
  for (; lanes <= n - i; i += lanes) {
    sum0 = hn::MulAdd(load_narrow_as_f32<HNarrow>(df, x + i),
                      load_narrow_as_f32<HNarrow>(df, y + i), sum0);
  }
  float result =
      hn::ReduceSum(df, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += decode(x[i]) * decode(y[i]);
  return result;
}

HWY_INLINE float dot_f32_bf16(const float* x, const bfloat16_t* y,
                              std::size_t n) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(df);
  auto sum1 = hn::Zero(df);
  auto sum2 = hn::Zero(df);
  auto sum3 = hn::Zero(df);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::MulAdd(hn::LoadU(df, x + i),
                      load_narrow_as_f32<hwy::bfloat16_t>(df, y + i), sum0);
    sum1 = hn::MulAdd(hn::LoadU(df, x + i + lanes),
                      load_narrow_as_f32<hwy::bfloat16_t>(df, y + i + lanes), sum1);
    sum2 = hn::MulAdd(hn::LoadU(df, x + i + lanes * 2),
                      load_narrow_as_f32<hwy::bfloat16_t>(df, y + i + lanes * 2), sum2);
    sum3 = hn::MulAdd(hn::LoadU(df, x + i + lanes * 3),
                      load_narrow_as_f32<hwy::bfloat16_t>(df, y + i + lanes * 3), sum3);
  }
  for (; lanes <= n - i; i += lanes) {
    sum0 = hn::MulAdd(hn::LoadU(df, x + i),
                      load_narrow_as_f32<hwy::bfloat16_t>(df, y + i), sum0);
  }
  float result =
      hn::ReduceSum(df, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += x[i] * bf16_to_f32(y[i]);
  return result;
}

HWY_INLINE std::int64_t dot_i16(const std::int16_t* x, const std::int16_t* y,
                                std::size_t n) noexcept {
#if HWY_TARGET == HWY_SCALAR
  {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += static_cast<std::uint64_t>(static_cast<std::int64_t>(x[i]) * y[i]);
    }
    return std::bit_cast<std::int64_t>(sum);
  }
#else
  {
    using D16 = hn::ScalableTag<std::int16_t>;
    using D32 = hn::RepartitionToWide<D16>;
    using D64 = hn::RepartitionToWide<D32>;
    using DU64 = hn::RebindToUnsigned<D64>;
    const D16 d16;
    const D64 d64;
    const DU64 du64;
    auto sum0 = hn::Zero(du64);
    auto sum1 = hn::Zero(du64);
    auto sum2 = hn::Zero(du64);
    auto sum3 = hn::Zero(du64);
    const std::size_t lanes = hn::Lanes(d16);
    std::size_t i = 0;
    for (; lanes <= n - i; i += lanes) {
      const auto a = hn::LoadU(d16, x + i);
      const auto b = hn::LoadU(d16, y + i);
      const auto even = hn::MulEven(a, b);
      const auto odd = hn::MulOdd(a, b);
      sum0 = hn::Add(sum0, hn::BitCast(du64, hn::PromoteLowerTo(d64, even)));
      sum1 = hn::Add(sum1, hn::BitCast(du64, hn::PromoteUpperTo(d64, even)));
      sum2 = hn::Add(sum2, hn::BitCast(du64, hn::PromoteLowerTo(d64, odd)));
      sum3 = hn::Add(sum3, hn::BitCast(du64, hn::PromoteUpperTo(d64, odd)));
    }
    std::uint64_t sum =
        hn::ReduceSum(du64, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
    for (; i < n; ++i) {
      sum += static_cast<std::uint64_t>(static_cast<std::int64_t>(x[i]) * y[i]);
    }
    return std::bit_cast<std::int64_t>(sum);
  }
#endif
}

HWY_INLINE std::int64_t dot_i8(const std::int8_t* x, const std::int8_t* y,
                               std::size_t n) noexcept {
#if HWY_TARGET == HWY_SCALAR
  {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += static_cast<std::uint64_t>(static_cast<std::int64_t>(x[i]) * y[i]);
    }
    return std::bit_cast<std::int64_t>(sum);
  }
#else
  {
    using D8 = hn::ScalableTag<std::int8_t>;
    using D32 = hn::Repartition<std::int32_t, D8>;
    const D8 d8;
    const D32 d32;
    const std::size_t lanes = hn::Lanes(d8);
    const std::size_t lanes4 = lanes * 4;
    auto sum0 = hn::Zero(d32);
    auto sum1 = hn::Zero(d32);
    auto sum2 = hn::Zero(d32);
    auto sum3 = hn::Zero(d32);
    std::uint64_t sum = 0;
    std::size_t blocks = 0;
    std::size_t i = 0;
    alignas(64) std::int32_t lane_values[D32().MaxLanes()];

    constexpr std::size_t kBlocksPerFlush = 1024;
    for (; lanes4 <= n - i; i += lanes4) {
      sum0 = hn::SumOfMulQuadAccumulate(
          d32, hn::LoadU(d8, x + i), hn::LoadU(d8, y + i), sum0);
      sum1 = hn::SumOfMulQuadAccumulate(
          d32, hn::LoadU(d8, x + i + lanes), hn::LoadU(d8, y + i + lanes), sum1);
      sum2 = hn::SumOfMulQuadAccumulate(d32, hn::LoadU(d8, x + i + lanes * 2),
                                        hn::LoadU(d8, y + i + lanes * 2), sum2);
      sum3 = hn::SumOfMulQuadAccumulate(d32, hn::LoadU(d8, x + i + lanes * 3),
                                        hn::LoadU(d8, y + i + lanes * 3), sum3);
      if (++blocks == kBlocksPerFlush) {
        hn::StoreU(hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)), d32,
                   lane_values);
        for (std::size_t lane = 0; lane < hn::Lanes(d32); ++lane) {
          sum += static_cast<std::uint64_t>(
              static_cast<std::int64_t>(lane_values[lane]));
        }
        sum0 = sum1 = sum2 = sum3 = hn::Zero(d32);
        blocks = 0;
      }
    }
    for (; lanes <= n - i; i += lanes) {
      sum0 = hn::SumOfMulQuadAccumulate(
          d32, hn::LoadU(d8, x + i), hn::LoadU(d8, y + i), sum0);
    }
    hn::StoreU(hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)), d32,
               lane_values);
    for (std::size_t lane = 0; lane < hn::Lanes(d32); ++lane) {
      sum += static_cast<std::uint64_t>(
          static_cast<std::int64_t>(lane_values[lane]));
    }
    for (; i < n; ++i) {
      sum += static_cast<std::uint64_t>(static_cast<std::int64_t>(x[i]) * y[i]);
    }
    return std::bit_cast<std::int64_t>(sum);
  }
#endif
}

template <typename T, class VectorOp, class ScalarOp>
HWY_INLINE void binary_row(const T* x, const T* y, T* out, std::size_t n,
                           VectorOp vector_op, ScalarOp scalar_op) noexcept {
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(vector_op(hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d, out + i);
  }
  for (; i < n; ++i) out[i] = scalar_op(x[i], y[i]);
}

template <typename T, class VectorOp, class ScalarOp>
HWY_INLINE void unary_row(const T* x, T* out, std::size_t n, VectorOp vector_op,
                          ScalarOp scalar_op) noexcept {
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(vector_op(hn::LoadU(d, x + i)), d, out + i);
  }
  for (; i < n; ++i) out[i] = scalar_op(x[i]);
}

template <typename T>
HWY_INLINE void scale_row(const T* x, T factor, T* out, std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  const auto vf = hn::Set(d, factor);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::Mul(hn::LoadU(d, x + i), vf), d, out + i);
  }
  for (; i < n; ++i) out[i] = x[i] * factor;
}

template <typename T>
HWY_INLINE void add_scalar_row(const T* x, T value, T* out,
                               std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  const auto vv = hn::Set(d, value);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::Add(hn::LoadU(d, x + i), vv), d, out + i);
  }
  for (; i < n; ++i) out[i] = x[i] + value;
}

template <typename T>
HWY_INLINE void affine_row(const T* x, T factor, T bias, T* out,
                           std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  const auto vf = hn::Set(d, factor);
  const auto vb = hn::Set(d, bias);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::MulAdd(hn::LoadU(d, x + i), vf, vb), d, out + i);
  }
  for (; i < n; ++i) out[i] = x[i] * factor + bias;
}

template <typename T>
HWY_INLINE void saxpy_row(T factor, const T* x, T* y, std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  const auto vf = hn::Set(d, factor);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::MulAdd(vf, hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d,
               y + i);
  }
  for (; i < n; ++i) y[i] = factor * x[i] + y[i];
}

template <typename T>
HWY_INLINE T sum_row(const T* x, std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(d);
  auto sum1 = hn::Zero(d);
  auto sum2 = hn::Zero(d);
  auto sum3 = hn::Zero(d);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::Add(sum0, hn::LoadU(d, x + i));
    sum1 = hn::Add(sum1, hn::LoadU(d, x + i + lanes));
    sum2 = hn::Add(sum2, hn::LoadU(d, x + i + lanes * 2));
    sum3 = hn::Add(sum3, hn::LoadU(d, x + i + lanes * 3));
  }
  for (; lanes <= n - i; i += lanes) sum0 = hn::Add(sum0, hn::LoadU(d, x + i));
  T result = hn::ReduceSum(d, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += x[i];
  return result;
}

}  // namespace

float HYLE_SIMD_OPS_FN(dot_f32_)(const float* x, const float* y,
                                 std::size_t n) noexcept {
  return dot_real(x, y, n);
}
double HYLE_SIMD_OPS_FN(dot_f64_)(const double* x, const double* y,
                                  std::size_t n) noexcept {
  return dot_real(x, y, n);
}
float HYLE_SIMD_OPS_FN(dot_bf16_bf16_)(const bfloat16_t* x,
                                       const bfloat16_t* y,
                                       std::size_t n) noexcept {
  return dot_narrow<hwy::bfloat16_t>(x, y, n, bf16_to_f32);
}
float HYLE_SIMD_OPS_FN(dot_f32_bf16_)(const float* x, const bfloat16_t* y,
                                      std::size_t n) noexcept {
  return dot_f32_bf16(x, y, n);
}
float HYLE_SIMD_OPS_FN(dot_f16_f16_)(const float16_t* x, const float16_t* y,
                                     std::size_t n) noexcept {
  return dot_narrow<hwy::float16_t>(x, y, n, f16_to_f32);
}
std::int64_t HYLE_SIMD_OPS_FN(dot_i16_i16_)(const std::int16_t* x,
                                            const std::int16_t* y,
                                            std::size_t n) noexcept {
  return dot_i16(x, y, n);
}
std::int64_t HYLE_SIMD_OPS_FN(dot_i8_i8_)(const std::int8_t* x,
                                          const std::int8_t* y,
                                          std::size_t n) noexcept {
  return dot_i8(x, y, n);
}

#define HYLE_SIMD_DEFINE_ROW(T, tag)                                           \
  void HYLE_SIMD_OPS_FN(saxpy_##tag##_)(T factor, const T* x, T* y,            \
                                         std::size_t n) noexcept {              \
    saxpy_row(factor, x, y, n);                                                \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(add_##tag##_)(const T* x, const T* y, T* out,          \
                                      std::size_t n) noexcept {                 \
    binary_row(x, y, out, n, [](auto a, auto b) { return hn::Add(a, b); },     \
               [](T a, T b) { return a + b; });                               \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(sub_##tag##_)(const T* x, const T* y, T* out,          \
                                      std::size_t n) noexcept {                 \
    binary_row(x, y, out, n, [](auto a, auto b) { return hn::Sub(a, b); },     \
               [](T a, T b) { return a - b; });                               \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(mul_##tag##_)(const T* x, const T* y, T* out,          \
                                      std::size_t n) noexcept {                 \
    binary_row(x, y, out, n, [](auto a, auto b) { return hn::Mul(a, b); },     \
               [](T a, T b) { return a * b; });                               \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(div_##tag##_)(const T* x, const T* y, T* out,          \
                                      std::size_t n) noexcept {                 \
    binary_row(x, y, out, n, [](auto a, auto b) { return hn::Div(a, b); },     \
               [](T a, T b) { return a / b; });                               \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(scale_##tag##_)(const T* x, T factor, T* out,          \
                                        std::size_t n) noexcept {               \
    scale_row(x, factor, out, n);                                              \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(add_scalar_##tag##_)(const T* x, T value, T* out,      \
                                             std::size_t n) noexcept {          \
    add_scalar_row(x, value, out, n);                                          \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(affine_##tag##_)(const T* x, T factor, T bias,         \
                                         T* out, std::size_t n) noexcept {      \
    affine_row(x, factor, bias, out, n);                                       \
  }                                                                            \
  T HYLE_SIMD_OPS_FN(sum_##tag##_)(const T* x, std::size_t n) noexcept {       \
    return sum_row(x, n);                                                      \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(abs_##tag##_)(const T* x, T* out,                      \
                                      std::size_t n) noexcept {                 \
    unary_row(x, out, n, [](auto v) { return hn::Abs(v); },                    \
              [](T v) { return std::abs(v); });                               \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(neg_##tag##_)(const T* x, T* out,                      \
                                      std::size_t n) noexcept {                 \
    unary_row(x, out, n, [](auto v) { return hn::Neg(v); },                    \
              [](T v) { return -v; });                                        \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(square_##tag##_)(const T* x, T* out,                   \
                                         std::size_t n) noexcept {              \
    unary_row(x, out, n, [](auto v) { return hn::Mul(v, v); },                 \
              [](T v) { return v * v; });                                     \
  }                                                                            \
  void HYLE_SIMD_OPS_FN(sqrt_##tag##_)(const T* x, T* out,                     \
                                       std::size_t n) noexcept {                \
    unary_row(x, out, n, [](auto v) { return hn::Sqrt(v); },                   \
              [](T v) { return std::sqrt(v); });                              \
  }

HYLE_SIMD_DEFINE_ROW(float, f32)
HYLE_SIMD_DEFINE_ROW(double, f64)

#undef HYLE_SIMD_DEFINE_ROW

}  // namespace hyle::simd::detail

#undef HYLE_SIMD_OPS_FN
#undef HYLE_SIMD_OPS_CAT
#undef HYLE_SIMD_OPS_CAT_
