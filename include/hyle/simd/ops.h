#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/dtype.h"

// Target-neutral numeric batch operations. These signatures contain only pointers, scalars and
// sizes, so a prebuilt HYLE runtime can select the best compiled implementation for the current
// CPU. Expert source users who want compile-time binding include hyle/simd/native/ops.h and call
// the same overloads in hyle::simd::native.

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

}  // namespace hyle::simd
