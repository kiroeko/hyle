#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
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

// AVX2 x86 host? (the borrow cores are x86 AVX2-only). On non-x86 this is always false.
bool have_avx2() noexcept {
#if defined(HYLE_ARCH_X64)
  return cpu_features().avx2;
#else
  return false;
#endif
}

// AVX-512 group host? Mirrors dispatch.cc's avx512_group (f+bw+vl+dq+cd, all XGETBV/ZMM-gated
// there). The 16-wide borrow cores are reached only here; an AVX2-but-not-512 host stays on
// the 8-wide cores. best_isa()>=kAvx512 is the same condition, expressed via the features.
bool have_avx512() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto& f = cpu_features();
  return f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq && f.avx512cd;
#else
  return false;
#endif
}

}  // namespace

void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::exp_f32_xsimd_avx512(src, dst, n); return; }  // 16-wide
  static const bool hw = have_avx2();
  if (hw) {
    detail::exp_f32_xsimd_avx2(src, dst, n); return;       // exp: xsimd fastest
  }
#elif defined(HYLE_ARCH_ARM64)
  detail::exp_f32_highway_neon(src, dst, n); return;       // NEON Highway (always present)
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
#elif defined(HYLE_ARCH_ARM64)
  detail::log_f32_highway_neon(src, dst, n); return;       // NEON Highway
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
#elif defined(HYLE_ARCH_ARM64)
  detail::tanh_f32_highway_neon(src, dst, n); return;      // NEON Highway
#endif
  detail::tanh_f32_scalar(src, dst, n);
}

// simd-09 dispatchers. Two shapes: xsimd-winning ops (exp2/erf) mirror exp (xsimd 512->avx2->
// scalar); Highway-winning ops (the other 9) mirror log (Highway 512 GCC-only ->avx2->scalar,
// MSVC keeps avx2 highway since its AVX3 is broken). arm64 always routes to the NEON Highway
// core. Per-op winner was bench-decided (GCC+MSVC cross-checked) -- see math.h.

// xsimd-winning: exp2, erf (written longhand -- only two, mirror exp's 512->avx2->scalar).
void exp2_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::exp2_f32_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::exp2_f32_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::exp2_f32_highway_neon(src, dst, n); return;
#endif
  detail::exp2_f32_scalar(src, dst, n);
}
void erf_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::erf_f32_xsimd_avx512(src, dst, n); return; }
  static const bool hw = have_avx2();
  if (hw) { detail::erf_f32_xsimd_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::erf_f32_highway_neon(src, dst, n); return;
#endif
  detail::erf_f32_scalar(src, dst, n);
}

// xsimd-only borrows (gamma family): tgamma is vectorized (xsimd shape 512->avx2->scalar on x86,
// xsimd-NEON on arm64 -- Highway has no gamma so it can't use the Highway-NEON path the other
// ops do). LGAMMA is NOT vectorized: xsimd's vectorized lgamma has a lane-mixing bug (wrong
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
  detail::tgamma_f32_xsimd_neon(src, dst, n); return;
#endif
  detail::tgamma_f32_scalar(src, dst, n);
}

// hypot(a,b): BINARY best-of. math_bench found xsimd ~2.2x faster, BUT xsimd's hypot is NOT
// overflow-safe (returns inf when a^2/b^2 exceed FLT_MAX). Highway's Hypot does exponent scaling
// to match std::hypot's full range -- and overflow-safety is hypot's entire reason to exist over
// naive sqrt(a*a+b*b). So CORRECTNESS wins over the 2.2x: borrow Highway (same MSVC-AVX3-broken
// 512-exclusion as log -- MSVC stays on the avx2 Highway core). scalar std::hypot is the floor.
void hypot_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
#if !defined(_MSC_VER)
  static const bool hw512 = have_avx512();
  if (hw512) { detail::hypot_f32_highway_avx512(a, b, dst, n); return; }
#endif
  static const bool hw = have_avx2();
  if (hw) { detail::hypot_f32_highway_avx2(a, b, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::hypot_f32_highway_neon(a, b, dst, n); return;
#endif
  detail::hypot_f32_scalar(a, b, dst, n);
}

// complex transcendentals (xsimd-only). x86 AVX2 core when available, else scalar std::complex.
// arm64 falls to scalar (Highway has no complex; xsimd-NEON complex not yet verified -> honest
// scalar fallback, not a silent wrong path). All SoA planar (re/im separate arrays).
#define HYLE_CX_UNARY_DISPATCH(name)                                                        \
  void name##_f32_batch(const float* sr, const float* si, float* dr, float* di,             \
                        std::size_t n) noexcept {                                           \
    /* x86: xsimd avx2 when present, else scalar; other arches: scalar oracle */            \
    HYLE_CX_UNARY_BODY(name)                                                                \
  }
#if defined(HYLE_ARCH_X64)
#define HYLE_CX_UNARY_BODY(name)                                                            \
  static const bool hw = have_avx2();                                                       \
  if (hw) { detail::name##_f32_xsimd_avx2(sr, si, dr, di, n); return; }                      \
  detail::name##_f32_scalar(sr, si, dr, di, n);
#else
#define HYLE_CX_UNARY_BODY(name) detail::name##_f32_scalar(sr, si, dr, di, n);
#endif
HYLE_CX_UNARY_DISPATCH(cexp)
HYLE_CX_UNARY_DISPATCH(clog)
HYLE_CX_UNARY_DISPATCH(csqrt)
HYLE_CX_UNARY_DISPATCH(csin)
HYLE_CX_UNARY_DISPATCH(ccos)
HYLE_CX_UNARY_DISPATCH(ctanh)
#undef HYLE_CX_UNARY_DISPATCH
#undef HYLE_CX_UNARY_BODY

void cpow_f32_batch(const float* zr, const float* zi, const float* wr, const float* wi,
                    float* dr, float* di, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::cpow_f32_xsimd_avx2(zr, zi, wr, wi, dr, di, n); return; }
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
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_MATH_HIGHWAY_BODY(name) detail::name##_f32_highway_neon(src, dst, n);
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
#undef HYLE_MATH_DISPATCH_HIGHWAY
#undef HYLE_MATH_HIGHWAY_BODY

Isa active_math_variant() noexcept {
#if defined(HYLE_ARCH_X64)
  static const Isa tier = have_avx512() ? Isa::kAvx512 : (have_avx2() ? Isa::kAvx2 : Isa::kScalar);
  return tier;
#elif defined(HYLE_ARCH_ARM64)
  return Isa::kNeon;  // Highway NEON cores always compiled (ARMv8 baseline)
#else
  return Isa::kScalar;
#endif
}

}  // namespace hyle::simd
