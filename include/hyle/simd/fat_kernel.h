#pragma once

#include "hyle/simd/target.h"

namespace hyle::simd {

// Selects one stable-signature whole-kernel function. Call once and cache the returned pointer;
// the kernel itself then pays no feature branch. In an exact-target TU the choice is constexpr
// and no CPU probe is referenced. Vector or mask types must never appear in Fn's signature.
#if defined(HYLE_ARCH_X64)
template <class Fn>
inline Fn select_fat_kernel(Fn scalar, Fn sse4, Fn avx2, Fn avx512) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return avx512;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  return avx2;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
  return sse4;
#elif defined(HYLE_SIMD_EXACT)
  return scalar;
#else
  switch (best_isa()) {
    case Isa::kAvx512: return avx512;
    case Isa::kAvx2: return avx2;
    case Isa::kSse42: return sse4;
    default: return scalar;
  }
#endif
}
#elif defined(HYLE_ARCH_ARM64)
template <class Fn>
inline Fn select_fat_kernel(Fn scalar, Fn neon, Fn sve = nullptr) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_SVE)
  return sve;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
  return neon;
#elif defined(HYLE_SIMD_EXACT)
  return scalar;
#else
  if (best_isa() == Isa::kSve && sve != nullptr) return sve;
  return best_isa() == Isa::kNeon ? neon : scalar;
#endif
}
#elif defined(HYLE_ARCH_RISCV64)
template <class Fn>
inline Fn select_fat_kernel(Fn scalar, Fn rvv) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_RVV)
  return rvv;
#elif defined(HYLE_SIMD_EXACT)
  return scalar;
#else
  return best_isa() == Isa::kRvv ? rvv : scalar;
#endif
}
#else
template <class Fn>
inline Fn select_fat_kernel(Fn scalar) noexcept { return scalar; }
#endif

}  // namespace hyle::simd
