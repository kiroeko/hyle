#pragma once

#include <cstddef>

#include "hyle/simd/dtype.h"

// kernels.h (internal) -- per-variant batch-conversion symbols the dtype dispatcher takes
// the address of. fp16 has scalar + f16c variants; bf16 has scalar + avx2 variants (bf16
// is pure integer bit-math, vectorizable with plain AVX2 -- no AVX-512-BF16 needed). Each
// hardware variant is a distinct -march TU. Not a public header.

namespace hyle::simd::detail {

// fp16 batch (scalar reference + F16C hardware)
void f16_to_f32_scalar(const float16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f16_scalar(const float* src, float16_t* dst, std::size_t n) noexcept;
void f16_to_f32_f16c(const float16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f16_f16c(const float* src, float16_t* dst, std::size_t n) noexcept;

// bf16 batch (scalar reference + AVX2 8-wide integer)
void bf16_to_f32_scalar(const bfloat16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_bf16_scalar(const float* src, bfloat16_t* dst, std::size_t n) noexcept;
void bf16_to_f32_avx2(const bfloat16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_bf16_avx2(const float* src, bfloat16_t* dst, std::size_t n) noexcept;

}  // namespace hyle::simd::detail
