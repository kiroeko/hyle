// Portable fp8 reference and fallback. OCP fp8 value, saturation and NaN
// semantics are HYLE product contracts not supplied by Highway.

#include "src/simd/dtype/kernels.h"

#include <cstddef>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

void f8e4m3_to_f32_scalar(const float8_e4m3_t* src, float* dst,
                          std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f8e4m3_to_f32(src[i]);
}

void f32_to_f8e4m3_scalar(const float* src, float8_e4m3_t* dst,
                          std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f8e4m3(src[i]);
}

void f8e5m2_to_f32_scalar(const float8_e5m2_t* src, float* dst,
                          std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f8e5m2_to_f32(src[i]);
}

void f32_to_f8e5m2_scalar(const float* src, float8_e5m2_t* dst,
                          std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f8e5m2(src[i]);
}

void f16_to_f8e4m3_scalar(const float16_t* src, float8_e4m3_t* dst,
                          std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = f32_to_f8e4m3(f16_to_f32(src[i]));
}

void f16_to_f8e5m2_scalar(const float16_t* src, float8_e5m2_t* dst,
                          std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = f32_to_f8e5m2(f16_to_f32(src[i]));
}

}  // namespace hyle::simd::detail
