#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"

// Target-neutral numeric batch operations. These signatures contain only pointers, scalars and
// sizes, so a prebuilt HYLE runtime can select the best compiled implementation for the current
// CPU. Expert source users who want compile-time binding include hyle/simd/native/ops.h and call
// the same overloads in hyle::simd::native.

namespace hyle::simd {

HYLE_SIMD_API float dot(const float* x, const float* y, std::size_t n) noexcept;
HYLE_SIMD_API double dot(const double* x, const double* y, std::size_t n) noexcept;

HYLE_SIMD_API void saxpy(float a, const float* x, float* y,
                         std::size_t n) noexcept;
HYLE_SIMD_API void saxpy(double a, const double* x, double* y,
                         std::size_t n) noexcept;

HYLE_SIMD_API void add(const float* x, const float* y, float* out,
                       std::size_t n) noexcept;
HYLE_SIMD_API void add(const double* x, const double* y, double* out,
                       std::size_t n) noexcept;

}  // namespace hyle::simd
