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
#include "src/simd/math/complex_hyle-inl.h"

namespace hyle::simd::detail {

namespace {
using batch = xsimd::batch<float>;
using cbatch = xsimd::batch<std::complex<float>>;
constexpr std::size_t kCW = cbatch::size;  // complex lanes per batch (== float batch size)
constexpr std::size_t kW = batch::size;  // 8 at AVX2
using dbatch = xsimd::batch<double>;
constexpr std::size_t kDW = dbatch::size;  // 4 at AVX2
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

// f64 (4-wide AVX2) borrow cores. xsimd::batch<double> lowers to the 4-lane ymm. Same verified
// polynomials as the f32 path, just double precision. Winner (xsimd vs Highway) is bench-decided
// per op in math_dispatch -- these exist so best-of can pick them.
void exp_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kDW <= n; i += kDW)
    xsimd::exp(dbatch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kDW <= n; i += kDW)
    xsimd::log(dbatch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kDW <= n; i += kDW)
    xsimd::tanh(dbatch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}
// erf f64: the ONE remaining op xsimd wins at f64 (bench: xsimd 1.3x Highway -- Highway wins the
// other 10). tgamma f64: xsimd-only (Highway has no gamma). lgamma stays scalar (the f32 vectorized
// lgamma lane-mixing bug is an algorithm bug -- assume it affects f64 too; not worth the risk).
void erf_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kDW <= n; i += kDW)
    xsimd::erf(dbatch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::erf(src[i]);
}
void tgamma_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kDW <= n; i += kDW)
    xsimd::tgamma(dbatch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tgamma(src[i]);
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

void hypot_fast_f32_xsimd_avx2(const float* a, const float* b, float* dst,
                               std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::hypot(batch::load_unaligned(a + i), batch::load_unaligned(b + i))
        .store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]);
}

// ---- complex transcendentals (SoA planar re/im) ------------------------------------
// xsimd owns the cexp/clog/csqrt borrow cores. Its ctanh/cpow paths remain same-TU benchmark
// controls for HYLE's SoA candidates. Tail = scalar std::complex; Highway has no complex path.
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
cbatch xb_tanh(const cbatch& z) { return xsimd::tanh(z); } ccf sc_tanh(const ccf& z) { return std::tanh(z); }
}  // namespace
void cexp_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_exp, sc_exp>(sr, si, dr, di, n); }
void clog_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_log, sc_log>(sr, si, dr, di, n); }
void csqrt_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_sqrt, sc_sqrt>(sr, si, dr, di, n); }
void csin_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sincos_hyle<false>(sr, si, dr, di, n);
}
void ccos_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sincos_hyle<true>(sr, si, dr, di, n);
}
void ctanh_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                          std::size_t n) noexcept {
  complex_ctanh_hyle(sr, si, dr, di, n);
}
void ctanh_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary_x<xb_tanh, sc_tanh>(sr, si, dr, di, n); }
void cpow_f32_hyle_avx2(const float* zr, const float* zi, const float* wr, const float* wi,
                         float* dr, float* di, std::size_t n) noexcept {
  complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}
void cpow_f32_xsimd_avx2(const float* zr, const float* zi, const float* wr, const float* wi, float* dr, float* di, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kCW <= n; i += kCW)
    xsimd::pow(cbatch::load_unaligned(zr + i, zi + i), cbatch::load_unaligned(wr + i, wi + i)).store_unaligned(dr + i, di + i);
  for (; i < n; ++i) { ccf r = std::pow(ccf(zr[i], zi[i]), ccf(wr[i], wi[i])); dr[i] = r.real(); di[i] = r.imag(); }
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
