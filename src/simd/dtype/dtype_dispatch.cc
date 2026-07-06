#include "hyle/simd/dtype_ops.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/dtype/kernels.h"

// dtype_dispatch.cc -- baseline dispatcher for the batch dtype conversions. fp16 picks the
// F16C kernel iff the CPU advertises F16C (cpu_features().f16c), else the scalar core.
// Baseline + LTO-off (hyle_simd) keeps the F16C codegen confined to cvt_f16c.cc, so this
// dispatcher never emits an F16C instruction on a CPU that lacks it.

namespace hyle::simd {
namespace {

// The fp16 hardware kernel (cvt_f16c.cc) uses 256-bit YMM ops (_mm256_cvtph_ps etc.), so
// it needs BOTH F16C *and* AVX with OS-saved YMM. Gate on f16c && avx:
//  - f.avx is already XGETBV-gated (os_ymm), so this never runs YMM on a stale OS that
//    doesn't preserve YMM across context switches (red-team caught the missing gate).
//  - it also rules out F16C-only-without-AVX CPUs (none real -- F16C shipped WITH AVX on
//    Ivy Bridge -- but the gate makes the YMM requirement explicit and correct anyway).
// The kernel TU is compiled -mf16c -mavx (NOT -mavx2): F16C+AVX is exactly Ivy Bridge's
// level, so neither the intrinsics nor the auto-vectorized scalar tail can emit an AVX2
// instruction that would SIGILL on an F16C+AVX-but-not-AVX2 CPU.
bool use_f16c() noexcept {
#if defined(HYLE_ARCH_X64)
  return cpu_features().f16c && cpu_features().avx;
#else
  return false;
#endif
}

// The bf16 AVX2 kernel (cvt_avx2.cc) uses 256-bit integer YMM ops (_mm256_cvtepu16_epi32,
// packus, permute4x64), so it needs AVX2 with OS-saved YMM. cpu_features().avx2 is the
// AVX2 flag; .avx is XGETBV-gated (os_ymm), so requiring both is the honest YMM gate.
bool use_avx2() noexcept {
#if defined(HYLE_ARCH_X64)
  return cpu_features().avx2 && cpu_features().avx;
#else
  return false;
#endif
}

// The native bf16 convert kernel (cvt_avx512bf16.cc) uses VCVTNEPS2BF16, gated on the
// avx512bf16 feature. dispatch.cc already AND-folds avx512bf16 with os_zmm (XGETBV), so
// this flag being true means BOTH the CPU has the bf16 extension AND the OS saves ZMM --
// the honest gate for a 512-bit instruction. Only f32->bf16 has a win (the convert folds
// the AVX2 RNE sequence into one op); bf16->f32 stays on the optimal AVX2 <<16 shift.
bool use_avx512bf16() noexcept {
#if defined(HYLE_ARCH_X64)
  return cpu_features().avx512bf16;
#else
  return false;
#endif
}

}  // namespace

void f16_to_f32_batch(const float16_t* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = use_f16c();
  if (hw) { detail::f16_to_f32_f16c(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::f16_to_f32_neon(src, dst, n); return;  // native ARMv8.2 vcvt (always present)
#endif
  detail::f16_to_f32_scalar(src, dst, n);
}

void f32_to_f16_batch(const float* src, float16_t* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = use_f16c();
  if (hw) { detail::f32_to_f16_f16c(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::f32_to_f16_neon(src, dst, n); return;
#endif
  detail::f32_to_f16_scalar(src, dst, n);
}

// bf16: AVX2 8-wide integer kernel iff the CPU has AVX2, else the scalar core. bf16<->f32
// is pure bit-math, so the AVX2 path is bit-identical to scalar (NaN payloads included).
void bf16_to_f32_batch(const bfloat16_t* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = use_avx2();
  if (hw) { detail::bf16_to_f32_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::bf16_to_f32_neon(src, dst, n); return;  // NEON integer, bit-identical to scalar
#endif
  detail::bf16_to_f32_scalar(src, dst, n);
}
void f32_to_bf16_batch(const float* src, bfloat16_t* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool bf16hw = use_avx512bf16();
  if (bf16hw) { detail::f32_to_bf16_avx512bf16(src, dst, n); return; }  // VCVTNEPS2BF16
  static const bool hw = use_avx2();
  if (hw) { detail::f32_to_bf16_avx2(src, dst, n); return; }
#elif defined(HYLE_ARCH_ARM64)
  detail::f32_to_bf16_neon(src, dst, n); return;  // NEON integer RNE, bit-identical
#endif
  detail::f32_to_bf16_scalar(src, dst, n);
}

Isa active_dtype_variant() noexcept {
#if defined(HYLE_ARCH_X64)
  static const Isa tier = use_f16c() ? Isa::kAvx2 : Isa::kScalar;
  return tier;
#elif defined(HYLE_ARCH_ARM64)
  return Isa::kNeon;  // fp16 via native vcvt + bf16 via NEON integer, always compiled
#else
  return Isa::kScalar;
#endif
}

}  // namespace hyle::simd
