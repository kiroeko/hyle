// Scalar batch dtype conversions -- the portable reference + fallback. Plain loops over
// the inline element conversions in dtype.h; compiles everywhere, always correct, and is
// the oracle the F16C path is cross-checked against. Compiled at baseline (no -march);
// the compiler auto-vectorizes the bf16 shifts and the fp16 element math as it can.

#include "src/simd/dtype/kernels.h"

#include <cstddef>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

void f16_to_f32_scalar(const float16_t* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}
void f32_to_f16_scalar(const float* src, float16_t* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}
void bf16_to_f32_scalar(const bfloat16_t* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}
void f32_to_bf16_scalar(const float* src, bfloat16_t* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_bf16(src[i]);
}

// fp8: scalar bit-math, both OCP formats, both directions. Auto-vectorized as the compiler can.
void f8e4m3_to_f32_scalar(const float8_e4m3_t* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f8e4m3_to_f32(src[i]);
}
void f32_to_f8e4m3_scalar(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f8e4m3(src[i]);
}
void f8e5m2_to_f32_scalar(const float8_e5m2_t* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f8e5m2_to_f32(src[i]);
}
void f32_to_f8e5m2_scalar(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f8e5m2(src[i]);
}
// f16 -> fp8 (the AI-native shape; AVX10.2 fp8 converts take an f16 source). f16->f32 is EXACT
// (f16 subset of f32), so f32_to_f8(f16_to_f32(h)) rounds the exact f16 value to fp8 ONCE -- no
// double rounding, hence bit-identical to the AVX10.2 hardware (verified exhaustively over all
// 65536 f16: E4M3 0 mismatch; E5M2 differs only on f16 inf/NaN where OCP keeps inf and Intel
// saturates -- the hw kernel restores HYLE's inf semantics with a mask, see cvt_avx10_2.cc).
void f16_to_f8e4m3_scalar(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f8e4m3(f16_to_f32(src[i]));
}
void f16_to_f8e5m2_scalar(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f8e5m2(f16_to_f32(src[i]));
}

}  // namespace hyle::simd::detail
