#pragma once

#include <cstddef>

#include "hyle/simd/visibility.h"
#include "hyle/simd/dtype.h"

// Discoverable whole-buffer conversion graph for HYLE's numeric storage types. Every unequal
// source/destination pair among f32, f16, bf16, E4M3 and E5M2 is provided. Same-type copying is a
// standard-library operation and is intentionally not duplicated here.
//
// All overloads are O(n), use O(1) extra storage and never allocate or lock. n == 0 accepts null
// pointers. Equal-width conversions allow src and dst to begin at the same address; unequal-width
// ranges must be disjoint. Any other partial overlap is outside the contract. Narrowing preserves
// the scalar contracts in dtype.h: round-to-nearest-even, documented saturation and NaN mapping.

namespace hyle::simd {

HYLE_SIMD_API void convert(const float* src, float16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float* src, bfloat16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept;

HYLE_SIMD_API void convert(const float16_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float16_t* src, bfloat16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept;

HYLE_SIMD_API void convert(const bfloat16_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const bfloat16_t* src, float16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const bfloat16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const bfloat16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept;

HYLE_SIMD_API void convert(const float8_e4m3_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float8_e4m3_t* src, float16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float8_e4m3_t* src, bfloat16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float8_e4m3_t* src, float8_e5m2_t* dst,
                           std::size_t n) noexcept;

HYLE_SIMD_API void convert(const float8_e5m2_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float8_e5m2_t* src, float16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float8_e5m2_t* src, bfloat16_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void convert(const float8_e5m2_t* src, float8_e4m3_t* dst,
                           std::size_t n) noexcept;

}  // namespace hyle::simd
