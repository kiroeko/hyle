#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/visibility.h"
#include "hyle/simd/dtype.h"

// Target-neutral numeric row operations. A prebuilt HYLE runtime performs at most one
// whole-buffer target selection per call. Source users who need register-level composition bind
// their TU with hyle_target_simd() and use the centrally locked Highway surface directly.
//
// For operations with an output range, out may exactly equal either complete input range. Any
// other overlap is outside the contract. n == 0 accepts null pointers. All operations are O(n),
// use O(1) extra storage and never allocate or lock.

namespace hyle::simd {

HYLE_SIMD_API float dot(const float* x, const float* y, std::size_t n) noexcept;
HYLE_SIMD_API double dot(const double* x, const double* y, std::size_t n) noexcept;

// Mixed/storage-precision reductions accumulate into f32 without a whole-array
// conversion pass. The symmetric bf16/f32 overload has identical semantics.
HYLE_SIMD_API float dot(const bfloat16_t* x, const bfloat16_t* y,
                        std::size_t n) noexcept;
HYLE_SIMD_API float dot(const float* x, const bfloat16_t* y,
                        std::size_t n) noexcept;
HYLE_SIMD_API float dot(const bfloat16_t* x, const float* y,
                        std::size_t n) noexcept;
HYLE_SIMD_API float dot(const float16_t* x, const float16_t* y,
                        std::size_t n) noexcept;

// Every i16 product is exact. Accumulation is modulo 2^64 and the resulting
// bit pattern is returned as a two's-complement int64_t, so overflow is defined.
HYLE_SIMD_API std::int64_t dot(const std::int16_t* x, const std::int16_t* y,
                               std::size_t n) noexcept;
HYLE_SIMD_API std::int64_t dot(const std::int8_t* x, const std::int8_t* y,
                               std::size_t n) noexcept;

HYLE_SIMD_API void saxpy(float a, const float* x, float* y,
                         std::size_t n) noexcept;
HYLE_SIMD_API void saxpy(double a, const double* x, double* y,
                         std::size_t n) noexcept;

HYLE_SIMD_API void add(const float* x, const float* y, float* out,
                       std::size_t n) noexcept;
HYLE_SIMD_API void add(const double* x, const double* y, double* out,
                       std::size_t n) noexcept;

#define HYLE_SIMD_DECLARE_ROW_ALGEBRA(T)                                      \
  HYLE_SIMD_API void subtract(const T* x, const T* y, T* out,                 \
                              std::size_t n) noexcept;                        \
  HYLE_SIMD_API void multiply(const T* x, const T* y, T* out,                 \
                              std::size_t n) noexcept;                        \
  HYLE_SIMD_API void divide(const T* x, const T* y, T* out,                   \
                            std::size_t n) noexcept;                          \
  HYLE_SIMD_API void scale(const T* x, T factor, T* out,                      \
                           std::size_t n) noexcept;                           \
  HYLE_SIMD_API void add_scalar(const T* x, T value, T* out,                  \
                                std::size_t n) noexcept;                      \
  HYLE_SIMD_API void affine(const T* x, T factor, T bias, T* out,             \
                            std::size_t n) noexcept;                          \
  HYLE_SIMD_API T sum(const T* x, std::size_t n) noexcept;                    \
  HYLE_SIMD_API void abs(const T* x, T* out, std::size_t n) noexcept;         \
  HYLE_SIMD_API void negate(const T* x, T* out, std::size_t n) noexcept;      \
  HYLE_SIMD_API void square(const T* x, T* out, std::size_t n) noexcept;      \
  HYLE_SIMD_API void sqrt(const T* x, T* out, std::size_t n) noexcept;

HYLE_SIMD_DECLARE_ROW_ALGEBRA(float)
HYLE_SIMD_DECLARE_ROW_ALGEBRA(double)

#undef HYLE_SIMD_DECLARE_ROW_ALGEBRA

// Full-precision f32 row primitives used by ordinary numeric pipelines and by
// Compute after it has proved a complete contiguous suffix. reciprocal_sqrt
// evaluates 1.0F / sqrt(x[i]); it is deliberately not an approximate-rsqrt
// contract. fmin/fmax have std::fmin/std::fmax number semantics, including
// one-NaN handling and deterministic signed zero. select_values treats every
// non-zero condition byte as true. Output may exactly alias a value input;
// partial overlap is unsupported, and select_values requires condition storage
// to be disjoint from all float ranges. A zero count permits null pointers.
HYLE_SIMD_API void reciprocal_sqrt(const float* x, float* out,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void fmin(const float* x, const float* y, float* out,
                       std::size_t n) noexcept;
HYLE_SIMD_API void fmax(const float* x, const float* y, float* out,
                       std::size_t n) noexcept;
HYLE_SIMD_API void select_values(const std::uint8_t* condition,
                                const float* when_true,
                                const float* when_false, float* out,
                                std::size_t n) noexcept;

}  // namespace hyle::simd
