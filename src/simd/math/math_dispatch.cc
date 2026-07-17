#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/runtime.h"
#include "src/simd/math/kernels.h"

// math_dispatch.cc -- baseline dispatcher for the transcendental batch ops, PER-OP
// best-of-breed across the two vendored header-only backends. The same-ISA bench found NO
// single library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh
// polynomials are fastest, Highway's log is fastest. So (fixed at compile time, not
// machine-dependent):
//   exp, tanh : xsimd  (fastest)
//   log       : Highway (fastest)
// Both backends are vendored in-tree (can't be missing) + always compiled on x86; the AVX2
// codegen is confined to the kernel TUs (baseline + LTO-off, same isolation as ops/dtype).
// This dispatcher only enters them when HYLE_ARCH_X64 && cpu_features().avx2 -> a non-x86
// build or an old x86 CPU falls to the scalar libm floor (no SIGILL).

namespace hyle::simd {
namespace {

#if defined(HYLE_ARCH_X64)
// AVX2 x86 host? (the borrow cores are x86 AVX2-only). On non-x86 this is always false.
bool have_avx2() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX2) || defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#elif defined(HYLE_ARCH_X64)
  return runtime().math.isa == Isa::kAvx2 || runtime().math.isa == Isa::kAvx512;
#else
  return false;
#endif
}

// AVX-512 group host? Mirrors dispatch.cc's avx512_group (f+bw+vl+dq+cd, all XGETBV/ZMM-gated
// there). The 16-wide borrow cores are reached only here; an AVX2-but-not-512 host stays on
// the 8-wide cores. best_isa()>=kAvx512 is the same condition, expressed via the features.
bool have_avx512() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#elif defined(HYLE_ARCH_X64)
  return runtime().math.isa == Isa::kAvx512;
#else
  return false;
#endif
}
#endif

#if defined(HYLE_ARCH_ARM64)
bool have_sve_math() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_SVE)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#elif defined(HYLE_SIMD_HAS_SVE_VARIANT)
  return runtime().math.isa == Isa::kSve;
#else
  return false;
#endif
}
#endif

#if defined(HYLE_ARCH_RISCV64)
bool have_rvv_math() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_RVV)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#elif defined(HYLE_SIMD_HAS_RVV_VARIANT)
  return runtime().math.isa == Isa::kRvv;
#else
  return false;
#endif
}
#endif

}  // namespace

#define HYLE_MATH_NAME_I(name, tag, suffix) detail::name##_##tag##_highway_##suffix
#define HYLE_MATH_NAME_II(name, tag, suffix) HYLE_MATH_NAME_I(name, tag, suffix)
#define HYLE_HYLE_NAME_I(name, tag, suffix) detail::name##_##tag##_hyle_##suffix
#define HYLE_HYLE_NAME_II(name, tag, suffix) HYLE_HYLE_NAME_I(name, tag, suffix)
#define HYLE_SCALAR_MATH_NAME_I(name, tag) detail::name##_##tag##_scalar
#define HYLE_SCALAR_MATH_NAME(name, tag) HYLE_SCALAR_MATH_NAME_I(name, tag)
#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_SCALABLE_MATH(name, tag) \
  (have_sve_math() ? HYLE_MATH_NAME_II(name, tag, sve) : HYLE_MATH_NAME_II(name, tag, neon))
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SCALABLE_MATH(name, tag) HYLE_MATH_NAME_II(name, tag, neon)
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_SCALABLE_MATH(name, tag) \
  (have_rvv_math() ? HYLE_MATH_NAME_II(name, tag, rvv) : HYLE_SCALAR_MATH_NAME(name, tag))
#elif defined(HYLE_ARCH_RISCV64)
#define HYLE_SCALABLE_MATH(name, tag) HYLE_SCALAR_MATH_NAME(name, tag)
#endif

void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::exp_f32_xsimd_avx512(src, dst, n); return; }  // 16-wide
  static const bool hw = have_avx2();
  if (hw) {
    detail::exp_f32_xsimd_avx2(src, dst, n); return;       // exp: xsimd fastest
  }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(exp, f32)(src, dst, n); return;
#endif
  detail::exp_f32_scalar(src, dst, n);
}
void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  // Highway's AVX3 is HWY_BROKEN on MSVC (would emit ymm), so the 512 log core exists only
  // on GCC/Clang. On MSVC we fall through to the avx2 highway log -- the widest Highway gives
  // there (no 512 win to be had). See transcendental_highway_avx512.cc MSVC EXCLUSION.
  static const bool hw512 = have_avx512();
  if (hw512) { detail::log_f32_highway_avx512(src, dst, n); return; }  // 16-wide (GCC/Clang)
#endif
  static const bool hw = have_avx2();
  if (hw) {
    detail::log_f32_highway_avx2(src, dst, n); return;     // log: Highway fastest
  }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(log, f32)(src, dst, n); return;
#endif
  detail::log_f32_scalar(src, dst, n);
}
void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::tanh_f32_xsimd_avx512(src, dst, n); return; }  // 16-wide
  static const bool hw = have_avx2();
  if (hw) {
    detail::tanh_f32_xsimd_avx2(src, dst, n); return;      // tanh: xsimd fastest
  }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(tanh, f32)(src, dst, n); return;
#endif
  detail::tanh_f32_scalar(src, dst, n);
}

// simd-09 dispatchers. Two shapes: xsimd-winning ops (exp2/erf) mirror exp (xsimd 512->avx2->
// scalar); Highway-winning ops (the other 9) mirror log (Highway 512 GCC-only ->avx2->scalar,
// MSVC keeps avx2 highway since its AVX3 is broken). SVE/RVV compile the shared D-tag body in
// their scalable product TU. Per-op winner was bench-decided (GCC+MSVC cross-checked) -- see
// math.h.

// xsimd-winning: exp2, erf (written longhand -- only two, mirror exp's 512->avx2->scalar).
void exp2_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::exp2_f32_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::exp2_f32_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(exp2, f32)(src, dst, n); return;
#endif
  detail::exp2_f32_scalar(src, dst, n);
}
void erf_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::erf_f32_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::erf_f32_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(erf, f32)(src, dst, n); return;
#endif
  detail::erf_f32_scalar(src, dst, n);
}

// xsimd-derived gamma family: tgamma is vectorized on x86/NEON/SVE/RVV. Highway has no gamma,
// so the portable path is HYLE's D-tag port. LGAMMA is NOT vectorized: xsimd's vectorized lgamma
// has a lane-mixing bug (wrong
// argument-reduction kernel for a lane in [1.25,1.5) sharing a vector with a lane in [0.75,1.25);
// reproduced GCC-x86 + MSVC-x86 + GCC-NEON => algorithm, not codegen; see
// docs/debug/xsimd-lgamma-lane-mixing-hunt.md). So lgamma is the scalar libm reference on every
// arch until xsimd fixes it -- the API ships correct, just not vectorized.
void lgamma_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
  detail::lgamma_f32_scalar(src, dst, n);
}
void tgamma_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::tgamma_f32_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::tgamma_f32_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(tgamma, f32, sve)(src, dst, n); return; }
#endif
  detail::tgamma_f32_xsimd_neon(src, dst, n); return;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(tgamma, f32, rvv)(src, dst, n); return; }
#endif
  detail::tgamma_f32_scalar(src, dst, n);
}

// Full-range hypot contract: Highway exponent scaling, scalar std::hypot fallback. The separate
// hypot_in_range_f32_batch below exposes xsimd's faster direct formula under a narrower input contract.
void hypot_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::hypot_f32_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::hypot_f32_highway_avx2(a, b, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(hypot, f32)(a, b, dst, n); return;
#endif
  detail::hypot_f32_scalar(a, b, dst, n);
}

// Direct-formula contract: faster xsimd core, no per-lane overflow/underflow checks.
void hypot_in_range_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::hypot_in_range_f32_xsimd_avx512(a, b, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::hypot_in_range_f32_xsimd_avx2(a, b, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(hypot_in_range, f32, sve)(a, b, dst, n); return; }
#endif
  detail::hypot_in_range_f32_xsimd_neon(a, b, dst, n); return;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(hypot_in_range, f32, rvv)(a, b, dst, n); return; }
#endif
  detail::hypot_in_range_f32_scalar(a, b, dst, n);
}

// SoA complex transcendentals. x86 uses AVX2; ARM64 uses SVE then NEON; RISC-V uses RVV. The
// scalable paths compile the same HYLE D-tag bodies and retain scalar std::complex lane fixups.
#define HYLE_CX_UNARY_DISPATCH(name)                                                        \
  void name##_f32_batch(const float* sr, const float* si, float* dr, float* di,             \
                        std::size_t n) noexcept {                                           \
    HYLE_CX_UNARY_BODY(name)                                                                \
  }
#if defined(HYLE_ARCH_X64)
#define HYLE_CX_UNARY_BODY(name)                                                            \
  static const bool hw = have_avx2();                                                       \
  if (hw) { detail::name##_f32_xsimd_avx2(sr, si, dr, di, n); return; }                      \
  detail::name##_f32_scalar(sr, si, dr, di, n);
#elif defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_CX_UNARY_BODY(name)                                                            \
  if (have_sve_math()) { HYLE_HYLE_NAME_II(name, f32, sve)(sr, si, dr, di, n); return; }    \
  detail::name##_f32_hyle_neon(sr, si, dr, di, n);
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_CX_UNARY_BODY(name)                                                            \
  detail::name##_f32_hyle_neon(sr, si, dr, di, n);
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_CX_UNARY_BODY(name)                                                            \
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(name, f32, rvv)(sr, si, dr, di, n); return; }    \
  detail::name##_f32_scalar(sr, si, dr, di, n);
#else
#define HYLE_CX_UNARY_BODY(name) detail::name##_f32_scalar(sr, si, dr, di, n);
#endif
HYLE_CX_UNARY_DISPATCH(cexp)
HYLE_CX_UNARY_DISPATCH(clog)
HYLE_CX_UNARY_DISPATCH(csqrt)
#undef HYLE_CX_UNARY_DISPATCH
#undef HYLE_CX_UNARY_BODY

void csin_f32_batch(const float* sr, const float* si, float* dr, float* di,
                     std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::csin_f32_hyle_avx2(sr, si, dr, di, n); return; }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(csin, f32, sve)(sr, si, dr, di, n); return; }
#endif
  detail::csin_f32_hyle_neon(sr, si, dr, di, n); return;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(csin, f32, rvv)(sr, si, dr, di, n); return; }
#endif
  detail::csin_f32_scalar(sr, si, dr, di, n);
}

void ccos_f32_batch(const float* sr, const float* si, float* dr, float* di,
                     std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::ccos_f32_hyle_avx2(sr, si, dr, di, n); return; }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(ccos, f32, sve)(sr, si, dr, di, n); return; }
#endif
  detail::ccos_f32_hyle_neon(sr, si, dr, di, n); return;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(ccos, f32, rvv)(sr, si, dr, di, n); return; }
#endif
  detail::ccos_f32_scalar(sr, si, dr, di, n);
}

void ctanh_f32_batch(const float* sr, const float* si, float* dr, float* di,
                     std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::ctanh_f32_hyle_avx2(sr, si, dr, di, n); return; }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(ctanh, f32, sve)(sr, si, dr, di, n); return; }
#endif
  detail::ctanh_f32_hyle_neon(sr, si, dr, di, n); return;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(ctanh, f32, rvv)(sr, si, dr, di, n); return; }
#endif
  detail::ctanh_f32_scalar(sr, si, dr, di, n);
}

void cpow_f32_batch(const float* zr, const float* zi, const float* wr, const float* wi,
                    float* dr, float* di, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::cpow_f32_hyle_avx2(zr, zi, wr, wi, dr, di, n); return; }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(cpow, f32, sve)(zr, zi, wr, wi, dr, di, n); return; }
#endif
  detail::cpow_f32_hyle_neon(zr, zi, wr, wi, dr, di, n); return;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(cpow, f32, rvv)(zr, zi, wr, wi, dr, di, n); return; }
#endif
  detail::cpow_f32_scalar(zr, zi, wr, wi, dr, di, n);
}

// Highway-winning: sin cos log2 atan cbrt expm1 log1p asinh acosh. On MSVC the 512 Highway is
// broken (would emit ymm), so the avx512 branch is GCC/Clang-only; MSVC falls to avx2 highway.
#define HYLE_MATH_DISPATCH_HIGHWAY(name)                                         \
  void name##_f32_batch(const float* src, float* dst, std::size_t n) noexcept {  \
    /* x86: 512 (GCC/Clang) -> avx2 -> scalar; arm64: NEON; else scalar */       \
    HYLE_MATH_HIGHWAY_BODY(name)                                                 \
  }
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
#define HYLE_MATH_HIGHWAY_BODY(name)                                             \
  static const bool hw512 = have_avx512();                                       \
  if (hw512) { detail::name##_f32_highway_avx512(src, dst, n); return; }         \
  static const bool hw = have_avx2();                                            \
  if (hw) { detail::name##_f32_highway_avx2(src, dst, n); return; }              \
  detail::name##_f32_scalar(src, dst, n);
#else  // MSVC: no 512 Highway (HWY_BROKEN AVX3), stay on avx2 highway
#define HYLE_MATH_HIGHWAY_BODY(name)                                             \
  static const bool hw = have_avx2();                                            \
  if (hw) { detail::name##_f32_highway_avx2(src, dst, n); return; }              \
  detail::name##_f32_scalar(src, dst, n);
#endif
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
#define HYLE_MATH_HIGHWAY_BODY(name)                                             \
  HYLE_SCALABLE_MATH(name, f32)(src, dst, n);
#else
#define HYLE_MATH_HIGHWAY_BODY(name) detail::name##_f32_scalar(src, dst, n);
#endif

HYLE_MATH_DISPATCH_HIGHWAY(sin)
HYLE_MATH_DISPATCH_HIGHWAY(cos)
HYLE_MATH_DISPATCH_HIGHWAY(log2)
HYLE_MATH_DISPATCH_HIGHWAY(atan)
HYLE_MATH_DISPATCH_HIGHWAY(cbrt)
HYLE_MATH_DISPATCH_HIGHWAY(expm1)
HYLE_MATH_DISPATCH_HIGHWAY(log1p)
HYLE_MATH_DISPATCH_HIGHWAY(asinh)
HYLE_MATH_DISPATCH_HIGHWAY(acosh)
HYLE_MATH_DISPATCH_HIGHWAY(acos)
HYLE_MATH_DISPATCH_HIGHWAY(asin)
HYLE_MATH_DISPATCH_HIGHWAY(sinh)
HYLE_MATH_DISPATCH_HIGHWAY(cosh)
HYLE_MATH_DISPATCH_HIGHWAY(tan)
HYLE_MATH_DISPATCH_HIGHWAY(atanh)
HYLE_MATH_DISPATCH_HIGHWAY(log10)
#undef HYLE_MATH_DISPATCH_HIGHWAY
#undef HYLE_MATH_HIGHWAY_BODY

// f64 transcendentals (CAE/solver/geometry). best-of bench (f64 rows, WSL GCC AVX2) found HIGHWAY
// wins 13 of 14 f64 ops it serves (exp/log/tanh + sin/cos/exp2/log2/atan/cbrt/expm1/log1p/asinh/
// acosh + hypot) -- the near-OPPOSITE of f32 (xsimd wins exp/tanh/exp2 there). xsimd wins only f64
// erf. This is exactly why winners must be benched per precision, not extrapolated from f32.
// Full tiering now: x86 avx512(GCC/Clang)->avx2->scalar; arm64 NEON; else scalar.
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
#define HYLE_MATH_F64_HIGHWAY_BODY(name)                                           \
  static const bool hw512 = have_avx512();                                         \
  if (hw512) { detail::name##_f64_highway_avx512(src, dst, n); return; }           \
  static const bool hw = have_avx2();                                              \
  if (hw) { detail::name##_f64_highway_avx2(src, dst, n); return; }                \
  detail::name##_f64_scalar(src, dst, n);
#else  // MSVC: Highway AVX3 broken -> stay on avx2 highway
#define HYLE_MATH_F64_HIGHWAY_BODY(name)                                           \
  static const bool hw = have_avx2();                                              \
  if (hw) { detail::name##_f64_highway_avx2(src, dst, n); return; }                \
  detail::name##_f64_scalar(src, dst, n);
#endif
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
#define HYLE_MATH_F64_HIGHWAY_BODY(name)                                         \
  HYLE_SCALABLE_MATH(name, f64)(src, dst, n);
#else
#define HYLE_MATH_F64_HIGHWAY_BODY(name) detail::name##_f64_scalar(src, dst, n);
#endif
#define HYLE_MATH_F64_HIGHWAY(name)                                                \
  void name##_f64_batch(const double* src, double* dst, std::size_t n) noexcept {  \
    HYLE_MATH_F64_HIGHWAY_BODY(name)                                               \
  }
HYLE_MATH_F64_HIGHWAY(exp)
HYLE_MATH_F64_HIGHWAY(log)
HYLE_MATH_F64_HIGHWAY(tanh)
HYLE_MATH_F64_HIGHWAY(sin)
HYLE_MATH_F64_HIGHWAY(cos)
HYLE_MATH_F64_HIGHWAY(exp2)
HYLE_MATH_F64_HIGHWAY(log2)
HYLE_MATH_F64_HIGHWAY(atan)
HYLE_MATH_F64_HIGHWAY(cbrt)
HYLE_MATH_F64_HIGHWAY(expm1)
HYLE_MATH_F64_HIGHWAY(log1p)
HYLE_MATH_F64_HIGHWAY(asinh)
HYLE_MATH_F64_HIGHWAY(acosh)
HYLE_MATH_F64_HIGHWAY(acos)
HYLE_MATH_F64_HIGHWAY(asin)
HYLE_MATH_F64_HIGHWAY(sinh)
HYLE_MATH_F64_HIGHWAY(cosh)
HYLE_MATH_F64_HIGHWAY(tan)
HYLE_MATH_F64_HIGHWAY(atanh)
HYLE_MATH_F64_HIGHWAY(log10)
#undef HYLE_MATH_F64_HIGHWAY
#undef HYLE_MATH_F64_HIGHWAY_BODY

// P2-3 narrow contracts: caller guarantees all lanes are normal finite values in the documented
// exp/log domains, so the ISA core may bypass the exceptional-vector screen. The table core and
// tier dispatch remain identical to the full API; old x86 / unsupported hosts use scalar libm.
void exp_in_range_f64_batch(const double* src, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::exp_in_range_f64_highway_avx512(src, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::exp_in_range_f64_highway_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  HYLE_SCALABLE_MATH(exp_in_range, f64)(src, dst, n); return;
#elif defined(HYLE_ARCH_RISCV64)
#if defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { detail::exp_in_range_f64_highway_rvv(src, dst, n); return; }
#endif
  detail::exp_f64_scalar(src, dst, n); return;
#endif
  detail::exp_f64_scalar(src, dst, n);
}
void log_in_range_f64_batch(const double* src, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::log_in_range_f64_highway_avx512(src, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::log_in_range_f64_highway_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  HYLE_SCALABLE_MATH(log_in_range, f64)(src, dst, n); return;
#elif defined(HYLE_ARCH_RISCV64)
#if defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { detail::log_in_range_f64_highway_rvv(src, dst, n); return; }
#endif
  detail::log_f64_scalar(src, dst, n); return;
#endif
  detail::log_f64_scalar(src, dst, n);
}

// hypot f64 (binary, overflow-safe -> Highway all platforms). Same tiering.
void hypot_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::hypot_f64_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::hypot_f64_highway_avx2(a, b, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(hypot, f64)(a, b, dst, n); return;
#endif
  detail::hypot_f64_scalar(a, b, dst, n);
}
// erf f64 (xsimd wins at f64, the one exception to Highway's fixed-width f64 sweep).
void erf_f64_batch(const double* src, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::erf_f64_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::erf_f64_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(erf, f64)(src, dst, n); return;
#endif
  detail::erf_f64_scalar(src, dst, n);
}
// tgamma f64 (xsimd-derived; Highway has no gamma) has SVE/RVV D-tag products. lgamma f64 remains
// scalar on every target because of the vendor algorithm bug documented by the f32 comment.
void tgamma_f64_batch(const double* src, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::tgamma_f64_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::tgamma_f64_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (have_sve_math()) { HYLE_HYLE_NAME_II(tgamma, f64, sve)(src, dst, n); return; }
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  if (have_rvv_math()) { HYLE_HYLE_NAME_II(tgamma, f64, rvv)(src, dst, n); return; }
#endif
  detail::tgamma_f64_scalar(src, dst, n);
}
void lgamma_f64_batch(const double* src, double* dst, std::size_t n) noexcept {
  detail::lgamma_f64_scalar(src, dst, n);  // scalar only (vendor vectorized-lgamma bug)
}

// simd-18 批G binary Highway (atan2/pow) + sincos + xsimd extras (scalar path first)
void atan2_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::atan2_f32_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::atan2_f32_highway_avx2(a, b, dst, n); return; }
  detail::atan2_f32_scalar(a, b, dst, n);
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(atan2, f32)(a, b, dst, n);
#else
  detail::atan2_f32_scalar(a, b, dst, n);
#endif
}
void pow_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::pow_f32_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::pow_f32_highway_avx2(a, b, dst, n); return; }
  detail::pow_f32_scalar(a, b, dst, n);
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(pow, f32)(a, b, dst, n);
#else
  detail::pow_f32_scalar(a, b, dst, n);
#endif
}
void sincos_f32_batch(const float* src, float* s, float* c, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::sincos_f32_highway_avx512(src, s, c, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::sincos_f32_highway_avx2(src, s, c, n); return; }
  detail::sincos_f32_scalar(src, s, c, n);
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(sincos, f32)(src, s, c, n);
#else
  detail::sincos_f32_scalar(src, s, c, n);
#endif
}
void atan2_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::atan2_f64_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::atan2_f64_highway_avx2(a, b, dst, n); return; }
  detail::atan2_f64_scalar(a, b, dst, n);
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(atan2, f64)(a, b, dst, n);
#else
  detail::atan2_f64_scalar(a, b, dst, n);
#endif
}
void pow_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::pow_f64_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::pow_f64_highway_avx2(a, b, dst, n); return; }
  detail::pow_f64_scalar(a, b, dst, n);
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(pow, f64)(a, b, dst, n);
#else
  detail::pow_f64_scalar(a, b, dst, n);
#endif
}
void sincos_f64_batch(const double* src, double* s, double* c, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::sincos_f64_highway_avx512(src, s, c, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::sincos_f64_highway_avx2(src, s, c, n); return; }
  detail::sincos_f64_scalar(src, s, c, n);
#elif defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
  HYLE_SCALABLE_MATH(sincos, f64)(src, s, c, n);
#else
  detail::sincos_f64_scalar(src, s, c, n);
#endif
}

#if defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64)
#undef HYLE_SCALABLE_MATH
#endif
#undef HYLE_MATH_NAME_II
#undef HYLE_MATH_NAME_I
#undef HYLE_SCALAR_MATH_NAME
#undef HYLE_SCALAR_MATH_NAME_I
// simd-18 batch G, now HYLE D-tag code: x86 uses AVX2; SVE/RVV use their scalable product TU.
// remainder/nextafter remain scalar below because the vector algorithms violate exact semantics.
#if defined(HYLE_ARCH_X64)
#define HYLE_XSIMD_UNARY(name)                                                              \
  void name##_f32_batch(const float* s, float* d, std::size_t n) noexcept {                 \
    static const bool hw = have_avx2();                                                     \
    if (hw) { detail::name##_f32_xsimd_avx2(s, d, n); return; }                             \
    detail::name##_f32_scalar(s, d, n);                                                     \
  }                                                                                         \
  void name##_f64_batch(const double* s, double* d, std::size_t n) noexcept {               \
    static const bool hw = have_avx2();                                                     \
    if (hw) { detail::name##_f64_xsimd_avx2(s, d, n); return; }                             \
    detail::name##_f64_scalar(s, d, n);                                                     \
  }
#define HYLE_XSIMD_BINARY(name)                                                             \
  void name##_f32_batch(const float* a, const float* b, float* d, std::size_t n) noexcept { \
    static const bool hw = have_avx2();                                                     \
    if (hw) { detail::name##_f32_xsimd_avx2(a, b, d, n); return; }                          \
    detail::name##_f32_scalar(a, b, d, n);                                                  \
  }                                                                                         \
  void name##_f64_batch(const double* a, const double* b, double* d, std::size_t n) noexcept { \
    static const bool hw = have_avx2();                                                     \
    if (hw) { detail::name##_f64_xsimd_avx2(a, b, d, n); return; }                          \
    detail::name##_f64_scalar(a, b, d, n);                                                  \
  }
#elif defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_XSIMD_UNARY(name)                                                              \
  void name##_f32_batch(const float* s, float* d, std::size_t n) noexcept {                 \
    if (have_sve_math()) { HYLE_HYLE_NAME_II(name, f32, sve)(s, d, n); return; }            \
    detail::name##_f32_scalar(s, d, n);                                                     \
  }                                                                                         \
  void name##_f64_batch(const double* s, double* d, std::size_t n) noexcept {               \
    if (have_sve_math()) { HYLE_HYLE_NAME_II(name, f64, sve)(s, d, n); return; }            \
    detail::name##_f64_scalar(s, d, n);                                                     \
  }
#define HYLE_XSIMD_BINARY(name)                                                             \
  void name##_f32_batch(const float* a, const float* b, float* d, std::size_t n) noexcept { \
    if (have_sve_math()) { HYLE_HYLE_NAME_II(name, f32, sve)(a, b, d, n); return; }         \
    detail::name##_f32_scalar(a, b, d, n);                                                  \
  }                                                                                         \
  void name##_f64_batch(const double* a, const double* b, double* d, std::size_t n) noexcept { \
    if (have_sve_math()) { HYLE_HYLE_NAME_II(name, f64, sve)(a, b, d, n); return; }         \
    detail::name##_f64_scalar(a, b, d, n);                                                  \
  }
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_XSIMD_UNARY(name)                                                              \
  void name##_f32_batch(const float* s, float* d, std::size_t n) noexcept {                 \
    if (have_rvv_math()) { HYLE_HYLE_NAME_II(name, f32, rvv)(s, d, n); return; }            \
    detail::name##_f32_scalar(s, d, n);                                                     \
  }                                                                                         \
  void name##_f64_batch(const double* s, double* d, std::size_t n) noexcept {               \
    if (have_rvv_math()) { HYLE_HYLE_NAME_II(name, f64, rvv)(s, d, n); return; }            \
    detail::name##_f64_scalar(s, d, n);                                                     \
  }
#define HYLE_XSIMD_BINARY(name)                                                             \
  void name##_f32_batch(const float* a, const float* b, float* d, std::size_t n) noexcept { \
    if (have_rvv_math()) { HYLE_HYLE_NAME_II(name, f32, rvv)(a, b, d, n); return; }         \
    detail::name##_f32_scalar(a, b, d, n);                                                  \
  }                                                                                         \
  void name##_f64_batch(const double* a, const double* b, double* d, std::size_t n) noexcept { \
    if (have_rvv_math()) { HYLE_HYLE_NAME_II(name, f64, rvv)(a, b, d, n); return; }         \
    detail::name##_f64_scalar(a, b, d, n);                                                  \
  }
#else
#define HYLE_XSIMD_UNARY(name)                                                              \
  void name##_f32_batch(const float* s, float* d, std::size_t n) noexcept { detail::name##_f32_scalar(s, d, n); } \
  void name##_f64_batch(const double* s, double* d, std::size_t n) noexcept { detail::name##_f64_scalar(s, d, n); }
#define HYLE_XSIMD_BINARY(name)                                                             \
  void name##_f32_batch(const float* a, const float* b, float* d, std::size_t n) noexcept { detail::name##_f32_scalar(a, b, d, n); } \
  void name##_f64_batch(const double* a, const double* b, double* d, std::size_t n) noexcept { detail::name##_f64_scalar(a, b, d, n); }
#endif
HYLE_XSIMD_UNARY(erfc)
HYLE_XSIMD_UNARY(exp10)
HYLE_XSIMD_BINARY(fmod)
// remainder: IEEE round-to-even remainder; xsimd's tie-breaking diverges from libm at half-way
// quotients (e.g. remainder(1.5,0.2): libm +0.1 vs xsimd -0.1). Bit-exact op -> scalar libm.
void remainder_f32_batch(const float* a, const float* b, float* d, std::size_t n) noexcept {
  detail::remainder_f32_scalar(a, b, d, n);
}
void remainder_f64_batch(const double* a, const double* b, double* d, std::size_t n) noexcept {
  detail::remainder_f64_scalar(a, b, d, n);
}
// nextafter: xsimd's vectorized nextafter diverges from libm (2-ULP step vs libm's exact 1-ULP,
// verified vs std::nextafter) -- this is a bit-exact op, so correctness beats vectorization here.
// Route to the scalar libm reference on all targets. (The other 5 xsimd extras match libm.)
void nextafter_f32_batch(const float* a, const float* b, float* d, std::size_t n) noexcept {
  detail::nextafter_f32_scalar(a, b, d, n);
}
void nextafter_f64_batch(const double* a, const double* b, double* d, std::size_t n) noexcept {
  detail::nextafter_f64_scalar(a, b, d, n);
}
HYLE_XSIMD_BINARY(fdim)
#undef HYLE_XSIMD_UNARY
#undef HYLE_XSIMD_BINARY
#undef HYLE_HYLE_NAME_II
#undef HYLE_HYLE_NAME_I
}  // namespace hyle::simd
