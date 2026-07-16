// AVX-512 (16-wide) sibling of transcendental_xsimd.cc -- same HYLE-ported xsimd
// product kernels, compiled at the AVX-512 tier so ScalableTag lowers to zmm.
#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#define HYLE_SIMD_TARGET_AVX512
#include "hyle/simd/vec.h"
#include "src/simd/math/math_xsimd_hyle-inl.h"
#include "src/simd/math/math_xsimd_body.inc"

namespace hyle::simd::detail {
using namespace hyle::simd;
using namespace xsimd_hyle;
using xsimd_body::unary_loop;
using xsimd_body::binary_loop;

void exp_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp(d, v); },
             [](float x) { return std::exp(x); });
}
void tanh_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tanh(d, v); },
             [](float x) { return std::tanh(x); });
}
void exp2_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp2(d, v); },
             [](float x) { return std::exp2(x); });
}
void erf_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Erf(d, v); },
             [](float x) { return std::erf(x); });
}
void tgamma_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tgamma(d, v); },
             [](float x) { return std::tgamma(x); });
}
void erf_f64_xsimd_avx512(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Erf(d, v); },
             [](double x) { return std::erf(x); });
}
void tgamma_f64_xsimd_avx512(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tgamma(d, v); },
             [](double x) { return std::tgamma(x); });
}
void hypot_in_range_f32_xsimd_avx512(const float* a, const float* b, float* dst,
                                     std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::HypotInRange(d, x, y); },
              [](float x, float y) { return std::sqrt(x * x + y * y); });
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
