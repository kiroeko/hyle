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

}  // namespace hyle::simd::detail
