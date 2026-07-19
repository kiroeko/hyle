#pragma once

#include "hyle/simd/target.h"

namespace hyle::simd {

namespace detail {

#if defined(HYLE_ARCH_X64)
template <class Fn>
inline Fn select_x86_fat_kernel_for_isa(Isa isa, Fn scalar, Fn sse4, Fn avx2,
                                         Fn avx512) noexcept {
  // best_isa() has honest detection-only tiers (SSE2/SSSE3/AVX) for which HYLE
  // intentionally does not build a separate kernel. Use the ordered ladder so
  // an AVX-only host takes the SSE4 kernel instead of accidentally falling all
  // the way back to scalar.
  const auto tier = static_cast<std::uint32_t>(isa);
  if (tier >= static_cast<std::uint32_t>(Isa::kAvx512)) return avx512;
  if (tier >= static_cast<std::uint32_t>(Isa::kAvx2)) return avx2;
  if (tier >= static_cast<std::uint32_t>(Isa::kSse42)) return sse4;
  return scalar;
}
#endif

}  // namespace detail

// Selects one stable-signature whole-kernel function. Call once and cache the returned pointer;
// the kernel itself then pays no feature branch. In an exact-target TU the choice is constexpr
// and no CPU probe is referenced. Vector or mask types must never appear in Fn's signature.
#if defined(HYLE_ARCH_X64)
template <class Fn>
inline Fn select_fat_kernel(Fn scalar, Fn sse4, Fn avx2, Fn avx512) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  (void)scalar;
  (void)sse4;
  (void)avx2;
  return avx512;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  (void)scalar;
  (void)sse4;
  (void)avx512;
  return avx2;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
  (void)scalar;
  (void)avx2;
  (void)avx512;
  return sse4;
#elif defined(HYLE_SIMD_EXACT)
  (void)sse4;
  (void)avx2;
  (void)avx512;
  return scalar;
#else
  return detail::select_x86_fat_kernel_for_isa(best_isa(), scalar, sse4,
                                                avx2, avx512);
#endif
}
#elif defined(HYLE_ARCH_ARM64)
template <class Fn>
inline Fn select_fat_kernel(Fn scalar, Fn neon, Fn sve = nullptr) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_SVE)
  (void)scalar;
  (void)neon;
  return sve;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
  (void)scalar;
  (void)sve;
  return neon;
#elif defined(HYLE_SIMD_EXACT)
  (void)neon;
  (void)sve;
  return scalar;
#else
  const Isa isa = best_isa();
  if (isa == Isa::kSve && sve != nullptr) return sve;
  return isa == Isa::kNeon ? neon : scalar;
#endif
}
#elif defined(HYLE_ARCH_RISCV64)
template <class Fn>
inline Fn select_fat_kernel(Fn scalar, Fn rvv) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_RVV)
  (void)scalar;
  return rvv;
#elif defined(HYLE_SIMD_EXACT)
  (void)rvv;
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

// hyle_add_fat_kernel may generate only the canonical variant in an exact HYLE
// build. A function call would still odr-use every argument before its exact
// branch can discard the unused pointers. This macro keeps one call spelling in
// fat and exact builds while preprocessing away symbols that were not generated.
// All arguments must be function-pointer expressions with the same type.
#if defined(HYLE_ARCH_X64)
#  if defined(HYLE_SIMD_FAT_EXACT_VARIANT_AVX512)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, sse4, avx2, avx512) (avx512)
#  elif defined(HYLE_SIMD_FAT_EXACT_VARIANT_AVX2)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, sse4, avx2, avx512) (avx2)
#  elif defined(HYLE_SIMD_FAT_EXACT_VARIANT_SSE4)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, sse4, avx2, avx512) (sse4)
#  elif defined(HYLE_SIMD_FAT_EXACT_VARIANT_SCALAR)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, sse4, avx2, avx512) (scalar)
#  else
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, sse4, avx2, avx512) \
      ::hyle::simd::select_fat_kernel((scalar), (sse4), (avx2), (avx512))
#  endif
#elif defined(HYLE_ARCH_ARM64)
#  if defined(HYLE_SIMD_FAT_EXACT_VARIANT_SVE)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, neon, sve) (sve)
#  elif defined(HYLE_SIMD_FAT_EXACT_VARIANT_NEON)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, neon, sve) (neon)
#  elif defined(HYLE_SIMD_FAT_EXACT_VARIANT_SCALAR)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, neon, sve) (scalar)
#  elif !defined(HYLE_SIMD_HAS_SVE_VARIANT)
// Apple ARM64 and other non-Linux ARM64 builds have NEON but do not emit an SVE
// translation unit. Do not odr-use the syntactically supplied SVE symbol.
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, neon, sve) \
      ::hyle::simd::select_fat_kernel((scalar), (neon))
#  else
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, neon, sve) \
      ::hyle::simd::select_fat_kernel((scalar), (neon), (sve))
#  endif
#elif defined(HYLE_ARCH_RISCV64)
#  if defined(HYLE_SIMD_FAT_EXACT_VARIANT_RVV)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, rvv) (rvv)
#  elif defined(HYLE_SIMD_FAT_EXACT_VARIANT_SCALAR)
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, rvv) (scalar)
#  else
#    define HYLE_SIMD_SELECT_FAT_KERNEL(scalar, rvv) \
      ::hyle::simd::select_fat_kernel((scalar), (rvv))
#  endif
#else
#  define HYLE_SIMD_SELECT_FAT_KERNEL(scalar) (scalar)
#endif
