#pragma once

#include <cstddef>

#include "hyle/simd/dtype.h"

// kernels.h (internal) -- per-variant batch-conversion symbols the dtype dispatcher takes
// the address of. fp16 has scalar + f16c variants (distinct -march TUs); bf16 has only a
// scalar variant (no AVX2-tier hardware). Not a public header.

namespace hyle::simd::detail {

// fp16 batch (scalar reference + F16C hardware)
void f16_to_f32_scalar(const float16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f16_scalar(const float* src, float16_t* dst, std::size_t n) noexcept;
void f16_to_f32_f16c(const float16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f16_f16c(const float* src, float16_t* dst, std::size_t n) noexcept;

// bf16 batch (scalar only)
void bf16_to_f32_scalar(const bfloat16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_bf16_scalar(const float* src, bfloat16_t* dst, std::size_t n) noexcept;

}  // namespace hyle::simd::detail
