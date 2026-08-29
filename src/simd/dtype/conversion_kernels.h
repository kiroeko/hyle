#pragma once

#include <cstddef>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

#define HYLE_SIMD_DECLARE_DTYPE_CONVERSION(suffix)                             \
  void f16_to_f32_##suffix(const float16_t*, float*, std::size_t) noexcept;    \
  void f32_to_f16_##suffix(const float*, float16_t*, std::size_t) noexcept;    \
  void bf16_to_f32_##suffix(const bfloat16_t*, float*, std::size_t) noexcept;  \
  void f32_to_bf16_##suffix(const float*, bfloat16_t*, std::size_t) noexcept;  \
  void f16_to_bf16_##suffix(const float16_t*, bfloat16_t*,                    \
                             std::size_t) noexcept;                           \
  void bf16_to_f16_##suffix(const bfloat16_t*, float16_t*,                    \
                             std::size_t) noexcept;

HYLE_SIMD_DECLARE_DTYPE_CONVERSION(scalar)
HYLE_SIMD_DECLARE_DTYPE_CONVERSION(sse4)
HYLE_SIMD_DECLARE_DTYPE_CONVERSION(avx2)
HYLE_SIMD_DECLARE_DTYPE_CONVERSION(avx512)
HYLE_SIMD_DECLARE_DTYPE_CONVERSION(neon)
HYLE_SIMD_DECLARE_DTYPE_CONVERSION(sve)
HYLE_SIMD_DECLARE_DTYPE_CONVERSION(rvv)

#undef HYLE_SIMD_DECLARE_DTYPE_CONVERSION

}  // namespace hyle::simd::detail
