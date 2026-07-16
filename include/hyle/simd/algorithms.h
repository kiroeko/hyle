#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/macros.h"
#include "hyle/simd/bitpack.h"
#include "hyle/simd/dtype_ops.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"
#include "hyle/simd/random.h"
#include "hyle/simd/scan.h"
#include "hyle/simd/sort.h"

// Ordinary C++ entry point for HYLE's whole-buffer SIMD algorithms.
//
// These overloads are header-only views over the pointer/count API. They do not add exported ABI,
// allocate, probe the CPU, or dispatch a second time. Include hyle/simd/vec.h (or native.h) instead
// when writing a fixed-target register kernel; include the individual module header when the raw
// pointer/count boundary is preferable.

namespace hyle::simd {

namespace detail {
inline void same_size(std::size_t a, std::size_t b) noexcept { HYLE_FATAL_ASSERT(a == b); }
inline void same_size(std::size_t a, std::size_t b, std::size_t c) noexcept {
  HYLE_FATAL_ASSERT(a == b && a == c);
}
inline void same_size(std::size_t a, std::size_t b, std::size_t c, std::size_t d) noexcept {
  HYLE_FATAL_ASSERT(a == b && a == c && a == d);
}
}  // namespace detail

// Basic whole-buffer operations.
#define HYLE_SIMD_SPAN_OPS(T)                                                                  \
  inline T dot(std::span<const T> x, std::span<const T> y) noexcept {                           \
    detail::same_size(x.size(), y.size());                                                       \
    return dot(x.data(), y.data(), x.size());                                                    \
  }                                                                                             \
  inline void saxpy(T a, std::span<const T> x, std::span<T> y) noexcept {                       \
    detail::same_size(x.size(), y.size());                                                       \
    saxpy(a, x.data(), y.data(), x.size());                                                      \
  }                                                                                             \
  inline void add(std::span<const T> x, std::span<const T> y, std::span<T> out) noexcept {       \
    detail::same_size(x.size(), y.size(), out.size());                                           \
    add(x.data(), y.data(), out.data(), x.size());                                               \
  }
HYLE_SIMD_SPAN_OPS(float)
HYLE_SIMD_SPAN_OPS(double)
#undef HYLE_SIMD_SPAN_OPS

// Stateful batch RNG. RandomState is intentionally explicit: filling a span advances it once.
#define HYLE_SIMD_SPAN_RANDOM(T)                                                                \
  inline void random_fill(RandomState& state, std::span<T> dst) noexcept {                       \
    random_fill(state, dst.data(), dst.size());                                                  \
  }
HYLE_SIMD_SPAN_RANDOM(std::uint64_t)
HYLE_SIMD_SPAN_RANDOM(float)
HYLE_SIMD_SPAN_RANDOM(double)
#undef HYLE_SIMD_SPAN_RANDOM

// In-place scans.
#define HYLE_SIMD_SPAN_SCAN(T)                                                                  \
  inline void inclusive_scan(std::span<T> data) noexcept {                                      \
    inclusive_scan(data.data(), data.size());                                                    \
  }                                                                                             \
  inline void exclusive_scan(std::span<T> data) noexcept {                                      \
    exclusive_scan(data.data(), data.size());                                                    \
  }
HYLE_SIMD_SPAN_SCAN(std::uint32_t)
HYLE_SIMD_SPAN_SCAN(std::int32_t)
HYLE_SIMD_SPAN_SCAN(std::uint64_t)
HYLE_SIMD_SPAN_SCAN(std::int64_t)
HYLE_SIMD_SPAN_SCAN(float)
HYLE_SIMD_SPAN_SCAN(double)
#undef HYLE_SIMD_SPAN_SCAN

// Supported storage conversions. Unsupported pairs fail overload resolution at compile time.
#define HYLE_SIMD_SPAN_CONVERT(SRC, DST, FN)                                                     \
  inline void convert(std::span<const SRC> src, std::span<DST> dst) noexcept {                   \
    detail::same_size(src.size(), dst.size());                                                   \
    FN(src.data(), dst.data(), src.size());                                                      \
  }
HYLE_SIMD_SPAN_CONVERT(float16_t, float, f16_to_f32_batch)
HYLE_SIMD_SPAN_CONVERT(float, float16_t, f32_to_f16_batch)
HYLE_SIMD_SPAN_CONVERT(bfloat16_t, float, bf16_to_f32_batch)
HYLE_SIMD_SPAN_CONVERT(float, bfloat16_t, f32_to_bf16_batch)
HYLE_SIMD_SPAN_CONVERT(float8_e4m3_t, float, f8e4m3_to_f32_batch)
HYLE_SIMD_SPAN_CONVERT(float, float8_e4m3_t, f32_to_f8e4m3_batch)
HYLE_SIMD_SPAN_CONVERT(float8_e5m2_t, float, f8e5m2_to_f32_batch)
HYLE_SIMD_SPAN_CONVERT(float, float8_e5m2_t, f32_to_f8e5m2_batch)
HYLE_SIMD_SPAN_CONVERT(float16_t, float8_e4m3_t, f16_to_f8e4m3_batch)
HYLE_SIMD_SPAN_CONVERT(float16_t, float8_e5m2_t, f16_to_f8e5m2_batch)
#undef HYLE_SIMD_SPAN_CONVERT

// Unary real math. The overload type selects f32/f64; the underlying semantic kernel stays the
// existing dynamically dispatched batch API.
#define HYLE_SIMD_SPAN_UNARY(NAME, T, FN)                                                        \
  inline void NAME(std::span<const T> src, std::span<T> dst) noexcept {                          \
    detail::same_size(src.size(), dst.size());                                                   \
    FN(src.data(), dst.data(), src.size());                                                      \
  }
#define HYLE_SIMD_SPAN_UNARY_BOTH(NAME)                                                          \
  HYLE_SIMD_SPAN_UNARY(NAME, float, NAME##_f32_batch)                                            \
  HYLE_SIMD_SPAN_UNARY(NAME, double, NAME##_f64_batch)
HYLE_SIMD_SPAN_UNARY_BOTH(exp)
HYLE_SIMD_SPAN_UNARY_BOTH(log)
HYLE_SIMD_SPAN_UNARY_BOTH(tanh)
HYLE_SIMD_SPAN_UNARY_BOTH(sin)
HYLE_SIMD_SPAN_UNARY_BOTH(cos)
HYLE_SIMD_SPAN_UNARY_BOTH(exp2)
HYLE_SIMD_SPAN_UNARY_BOTH(log2)
HYLE_SIMD_SPAN_UNARY_BOTH(erf)
HYLE_SIMD_SPAN_UNARY_BOTH(atan)
HYLE_SIMD_SPAN_UNARY_BOTH(cbrt)
HYLE_SIMD_SPAN_UNARY_BOTH(expm1)
HYLE_SIMD_SPAN_UNARY_BOTH(log1p)
HYLE_SIMD_SPAN_UNARY_BOTH(asinh)
HYLE_SIMD_SPAN_UNARY_BOTH(acosh)
HYLE_SIMD_SPAN_UNARY_BOTH(acos)
HYLE_SIMD_SPAN_UNARY_BOTH(asin)
HYLE_SIMD_SPAN_UNARY_BOTH(sinh)
HYLE_SIMD_SPAN_UNARY_BOTH(cosh)
HYLE_SIMD_SPAN_UNARY_BOTH(tan)
HYLE_SIMD_SPAN_UNARY_BOTH(atanh)
HYLE_SIMD_SPAN_UNARY_BOTH(log10)
HYLE_SIMD_SPAN_UNARY_BOTH(erfc)
HYLE_SIMD_SPAN_UNARY_BOTH(exp10)
HYLE_SIMD_SPAN_UNARY_BOTH(tgamma)
HYLE_SIMD_SPAN_UNARY_BOTH(lgamma)
HYLE_SIMD_SPAN_UNARY(exp_in_range, double, exp_in_range_f64_batch)
HYLE_SIMD_SPAN_UNARY(log_in_range, double, log_in_range_f64_batch)
HYLE_SIMD_SPAN_UNARY(gelu, float, gelu_f32_batch)
HYLE_SIMD_SPAN_UNARY(gelu_tanh, float, gelu_tanh_f32_batch)
HYLE_SIMD_SPAN_UNARY(sigmoid, float, sigmoid_f32_batch)
HYLE_SIMD_SPAN_UNARY(silu, float, silu_f32_batch)
#undef HYLE_SIMD_SPAN_UNARY_BOTH
#undef HYLE_SIMD_SPAN_UNARY

// Binary real math.
#define HYLE_SIMD_SPAN_BINARY(NAME, T, FN)                                                       \
  inline void NAME(std::span<const T> a, std::span<const T> b, std::span<T> dst) noexcept {      \
    detail::same_size(a.size(), b.size(), dst.size());                                           \
    FN(a.data(), b.data(), dst.data(), a.size());                                                \
  }
#define HYLE_SIMD_SPAN_BINARY_BOTH(NAME)                                                         \
  HYLE_SIMD_SPAN_BINARY(NAME, float, NAME##_f32_batch)                                           \
  HYLE_SIMD_SPAN_BINARY(NAME, double, NAME##_f64_batch)
HYLE_SIMD_SPAN_BINARY_BOTH(atan2)
HYLE_SIMD_SPAN_BINARY_BOTH(pow)
HYLE_SIMD_SPAN_BINARY_BOTH(hypot)
HYLE_SIMD_SPAN_BINARY_BOTH(fmod)
HYLE_SIMD_SPAN_BINARY_BOTH(remainder)
HYLE_SIMD_SPAN_BINARY_BOTH(nextafter)
HYLE_SIMD_SPAN_BINARY_BOTH(fdim)
HYLE_SIMD_SPAN_BINARY(hypot_in_range, float, hypot_in_range_f32_batch)
#undef HYLE_SIMD_SPAN_BINARY_BOTH
#undef HYLE_SIMD_SPAN_BINARY

inline void sincos(std::span<const float> src, std::span<float> s, std::span<float> c) noexcept {
  detail::same_size(src.size(), s.size(), c.size());
  sincos_f32_batch(src.data(), s.data(), c.data(), src.size());
}
inline void sincos(std::span<const double> src, std::span<double> s,
                   std::span<double> c) noexcept {
  detail::same_size(src.size(), s.size(), c.size());
  sincos_f64_batch(src.data(), s.data(), c.data(), src.size());
}
inline void softmax(std::span<float> data) noexcept { softmax_f32(data.data(), data.size()); }
inline void log_softmax(std::span<float> data) noexcept {
  log_softmax_f32(data.data(), data.size());
}

// In-place sort/select family.
#define HYLE_SIMD_SPAN_SORTABLE(T)                                                              \
  inline void sort(std::span<T> keys, SortOrder order = SortOrder::kAscending) noexcept {        \
    sort(keys.data(), keys.size(), order);                                                       \
  }                                                                                             \
  inline void partial_sort(std::span<T> keys, std::size_t k,                                    \
                           SortOrder order = SortOrder::kAscending) noexcept {                    \
    partial_sort(keys.data(), keys.size(), k, order);                                            \
  }                                                                                             \
  inline void select(std::span<T> keys, std::size_t k,                                          \
                     SortOrder order = SortOrder::kAscending) noexcept {                         \
    select(keys.data(), keys.size(), k, order);                                                  \
  }
HYLE_SIMD_SPAN_SORTABLE(std::uint32_t)
HYLE_SIMD_SPAN_SORTABLE(std::int32_t)
HYLE_SIMD_SPAN_SORTABLE(std::uint64_t)
HYLE_SIMD_SPAN_SORTABLE(std::int64_t)
HYLE_SIMD_SPAN_SORTABLE(float)
HYLE_SIMD_SPAN_SORTABLE(double)
HYLE_SIMD_SPAN_SORTABLE(KeyValue32)
HYLE_SIMD_SPAN_SORTABLE(KeyValue64)
#undef HYLE_SIMD_SPAN_SORTABLE

#define HYLE_SIMD_SPAN_RADIX(T)                                                                 \
  inline void radix_sort(std::span<T> keys, SortOrder order = SortOrder::kAscending) noexcept {  \
    radix_sort(keys.data(), keys.size(), order);                                                 \
  }
HYLE_SIMD_SPAN_RADIX(std::uint32_t)
HYLE_SIMD_SPAN_RADIX(std::int32_t)
HYLE_SIMD_SPAN_RADIX(std::uint64_t)
HYLE_SIMD_SPAN_RADIX(std::int64_t)
HYLE_SIMD_SPAN_RADIX(float)
#undef HYLE_SIMD_SPAN_RADIX

#define HYLE_SIMD_SPAN_ARGSORT(T)                                                               \
  inline void argsort(std::span<const T> keys, std::span<std::uint32_t> indices,                \
                      SortOrder order = SortOrder::kAscending) noexcept {                        \
    detail::same_size(keys.size(), indices.size());                                              \
    argsort(keys.data(), indices.data(), keys.size(), order);                                    \
  }
HYLE_SIMD_SPAN_ARGSORT(std::uint32_t)
HYLE_SIMD_SPAN_ARGSORT(std::int32_t)
HYLE_SIMD_SPAN_ARGSORT(float)
#undef HYLE_SIMD_SPAN_ARGSORT

// Bit packing. The packed span may be larger than required; undersizing is a caller bug.
#define HYLE_SIMD_SPAN_BITPACK(T)                                                               \
  inline void pack(std::span<const T> raw, std::span<T> packed, unsigned bits) noexcept {        \
    HYLE_FATAL_ASSERT(packed.size() >= packed_size<T>(raw.size(), bits));                         \
    pack(raw.data(), raw.size(), packed.data(), bits);                                           \
  }                                                                                             \
  inline void unpack(std::span<const T> packed, std::span<T> raw, unsigned bits) noexcept {      \
    HYLE_FATAL_ASSERT(packed.size() >= packed_size<T>(raw.size(), bits));                         \
    unpack(packed.data(), raw.size(), raw.data(), bits);                                         \
  }
HYLE_SIMD_SPAN_BITPACK(std::uint8_t)
HYLE_SIMD_SPAN_BITPACK(std::uint16_t)
HYLE_SIMD_SPAN_BITPACK(std::uint32_t)
#undef HYLE_SIMD_SPAN_BITPACK

}  // namespace hyle::simd
