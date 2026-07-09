// xsimd-accelerated AVX2 transcendentals (8-wide). BORROWS xsimd's verified math cores
// (xsimd::exp/tanh) -- HYLE does not hand-roll these (charter: 借验证核不手搓). The same-ISA
// bench (math_bench) found xsimd's exp/tanh polynomials fastest at the ~3.5-ULP class
// (Highway wins log), so per-op best-of-breed (math_dispatch.cc) uses xsimd for EXP/TANH
// ONLY -- hence log is not defined here (it would be dead code: Highway serves log).
//
// x86-only: xsimd::batch<float> lowers to the 8-wide AVX2 register here, so the TU compiles
// to nothing on non-x86 (math_dispatch falls to the scalar libm floor there). Header-only +
// vendored in third_party, so it can't be missing -- no opt-in define gates it, only the
// architecture does. Compiled at -mavx2; reached only when cpu_features().avx2.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>

#include "xsimd/xsimd.hpp"

namespace hyle::simd::detail {

namespace {
using batch = xsimd::batch<float>;
using cbatch = xsimd::batch<std::complex<float>>;
constexpr std::size_t kCW = cbatch::size;  // complex lanes per batch (== float batch size)
constexpr std::size_t kW = batch::size;  // 8 at AVX2
}  // namespace

void exp_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::exp(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (scalar oracle)
}
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tanh(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}

// simd-09: xsimd wins exp2/erf at the same ISA (bench-decided; Highway serves the rest).
void exp2_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::exp2(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp2(src[i]);
}
void erf_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::erf(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::erf(src[i]);
}

// xsimd-ONLY borrow: Highway's contrib/math has no gamma, so there is no best-of race -- xsimd
// is the single source. Only TGAMMA is borrowed: xsimd's vectorized LGAMMA has a lane-mixing bug
// (a lane in [1.25,1.5) gets the wrong argument-reduction kernel when it shares a vector with a
// lane in [0.75,1.25); reproduced GCC-x86 + MSVC-x86 + GCC-NEON => algorithm, not codegen). So
// lgamma stays on the scalar libm reference. See docs/debug/xsimd-lgamma-lane-mixing-hunt.md.
void tgamma_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tgamma(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tgamma(src[i]);
}

// ---- complex transcendentals (xsimd-only; SoA planar re/im) -------------------------
// batch<complex>::load_unaligned(re, im) reads the two planes into a complex batch; the store
// writes them back. Tail = scalar std::complex. Highway has no complex path -> single source.
namespace {
using ccf = std::complex<float>;
template <cbatch (*Fn)(const cbatch&), ccf (*Sc)(const ccf&)>
void cunary_x(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kCW <= n; i += kCW)
    Fn(cbatch::load_unaligned(sr + i, si + i)).store_unaligned(dr + i, di + i);
  for (; i < n; ++i) { ccf r = Sc(ccf(sr[i], si[i])); dr[i] = r.real(); di[i] = r.imag(); }
}
cbatch xb_exp(const cbatch& z) { return xsimd::exp(z); }   ccf sc_exp(const ccf& z) { return std::exp(z); }
cbatch xb_log(const cbatch& z) { return xsimd::log(z); }   ccf sc_log(const ccf& z) { return std::log(z); }
cbatch xb_sqrt(const cbatch& z) { return xsimd::sqrt(z); } ccf sc_sqrt(const ccf& z) { return std::sqrt(z); }
cbatch xb_sin(const cbatch& z) { return xsimd::sin(z); }   ccf sc_sin(const ccf& z) { return std::sin(z); }
cbatch xb_cos(const cbatch& z) { return xsimd::cos(z); }   ccf sc_cos(const ccf& z) { return std::cos(z); }
cbatch xb_tanh(const cbatch& z) { return xsimd::tanh(z); } ccf sc_tanh(const ccf& z) { return std::tanh(z); }
}  // namespace
void cexp_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_exp, sc_exp>(sr, si, dr, di, n); }
void clog_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_log, sc_log>(sr, si, dr, di, n); }
void csqrt_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_sqrt, sc_sqrt>(sr, si, dr, di, n); }
void csin_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_sin, sc_sin>(sr, si, dr, di, n); }
void ccos_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_cos, sc_cos>(sr, si, dr, di, n); }
void ctanh_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_tanh, sc_tanh>(sr, si, dr, di, n); }
void cpow_f32_xsimd_avx2(const float* zr, const float* zi, const float* wr, const float* wi, float* dr, float* di, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kCW <= n; i += kCW)
    xsimd::pow(cbatch::load_unaligned(zr + i, zi + i), cbatch::load_unaligned(wr + i, wi + i)).store_unaligned(dr + i, di + i);
  for (; i < n; ++i) { ccf r = std::pow(ccf(zr[i], zi[i]), ccf(wr[i], wi[i])); dr[i] = r.real(); di[i] = r.imag(); }
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
