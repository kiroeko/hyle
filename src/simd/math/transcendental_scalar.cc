// Scalar libm transcendentals -- the always-available reference + fallback + ULP oracle.
// Plain std::exp/log/tanh per element. Compiled everywhere (no -march, no SLEEF).
#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

namespace hyle::simd::detail {

void exp_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::tanh(src[i]);
}

}  // namespace hyle::simd::detail
