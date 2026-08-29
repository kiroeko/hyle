#pragma once

#include <cstddef>

#include "hyle/simd/dtype.h"

// Internal symbols for HYLE-only conversion gaps. The common f32/f16/bf16
// graph is generated directly from Highway in conversion_kernel_impl.h.

namespace hyle::simd::detail {

// F16C remains a precise feature specialization for Ivy Bridge-class CPUs:
// they have AVX+F16C but cannot run the generated AVX2 target.
void f16_to_f32_f16c(const float16_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f16_f16c(const float* src, float16_t* dst, std::size_t n) noexcept;

// bf16 f32->bf16 native AVX-512-BF16 (VCVTNEPS2BF16, 16-wide hardware RNE). Only reached
// when cpu_features().avx512bf16. Folds the AVX2 path's ~7-op RNE into one instruction.
// (bf16->f32 has no AVX-512-BF16 win: it's a pure <<16 shift the AVX2 path already does
// optimally, so no avx512 variant for that direction.)
void f32_to_bf16_avx512bf16(const float* src, bfloat16_t* dst, std::size_t n) noexcept;

// fp8 batch: portable HYLE oracle plus the optional AVX10.2 specialization.
void f8e4m3_to_f32_scalar(const float8_e4m3_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f8e4m3_scalar(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept;
void f8e5m2_to_f32_scalar(const float8_e5m2_t* src, float* dst, std::size_t n) noexcept;
void f32_to_f8e5m2_scalar(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept;

// f16 -> fp8 (AI-native shape). scalar = f32_to_f8(f16_to_f32) single-rounding. AVX10.2 hardware
// kernel (cvt_avx10_2.cc, clang-20-only build + runtime avx10_2 gate) is bit-identical to scalar.
void f16_to_f8e4m3_scalar(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept;
void f16_to_f8e5m2_scalar(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept;
#if defined(HYLE_HAVE_AVX10_2)
void f16_to_f8e4m3_avx10_2(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept;
void f16_to_f8e5m2_avx10_2(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept;
// f32 -> fp8 hardware kernel: round-to-odd f32->f16 then hw f16->fp8, bit-identical to the direct
// scalar oracle (Boldo-Melquiond; f32-source inf/NaN fixup). See cvt_avx10_2.cc.
void f32_to_f8e4m3_avx10_2(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept;
void f32_to_f8e5m2_avx10_2(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept;
#endif

}  // namespace hyle::simd::detail
