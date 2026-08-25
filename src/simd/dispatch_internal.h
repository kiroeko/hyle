#pragma once

#include <cstdint>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/target_manifest.h"

// Internal helpers for batch-kernel dispatchers. This deliberately lives outside the public
// dispatch header: callers export fixed POD function signatures, while each module retains its
// own feature gates and fallback contract. These helpers only cover the common full-ISA ladder
// emitted by hyle_add_dispatched_kernel.
namespace hyle::simd::detail {

#if defined(HYLE_ARCH_X64)
// Exact runtime prerequisites for the optional x86 product TUs. These are
// deliberately narrower than best_isa(): a specialty kernel may be useful
// below HYLE's broad Highway-compatible AVX2/AVX-512 tier, but it must still
// satisfy every ISA feature that its own compile flags permit.
struct X86SpecialtySupport {
  bool avx2_int;
  bool f16c;
  bool avx_vnni;
  bool avx512_vnni;
  bool avx512_bf16;
  bool avx10_2;
};

constexpr X86SpecialtySupport x86_specialty_support(
    const CpuFeatures& f) noexcept {
  // -mavx2/-mavxvnni may use the AVX2 integer face and always require an
  // OS-saved YMM state. Keep AVX explicit even for malformed/masked CPUID sets.
  const bool avx2_tu = f.avx && f.avx2;
  // The AVX-512 VNNI/BF16 TUs compile with F+BW+VL+DQ in addition to their
  // specialty extension. Clang also enables FMA/F16C as prerequisites of
  // -mavx512f and may emit them anywhere in the TU (including scalar tails).
  // A hypervisor may mask those legacy bits independently.
  const bool avx512_tu =
      avx2_tu && f.fma && f.f16c && f.avx512f && f.avx512bw &&
      f.avx512vl && f.avx512dq;
  return X86SpecialtySupport{
      .avx2_int = avx2_tu,
      .f16c = f.avx && f.f16c,
      .avx_vnni = avx2_tu && f.avx_vnni,
      .avx512_vnni = avx512_tu && f.avx512vnni,
      .avx512_bf16 = avx512_tu && f.avx512bf16,
      // CpuFeatures::avx10_2 means AVX10.2 version plus OS ZMM state. Current
      // AVX10 supports all vector widths. The -mavx10.2-512 TU also permits
      // FMA/F16C; retain those legacy prerequisites for independently masked
      // virtual CPUs even though conforming AVX10.2 hardware includes them.
      .avx10_2 = avx2_tu && f.fma && f.f16c && f.avx10_2,
  };
}
#endif

#define HYLE_SIMD_SUFFIXED_SYMBOL_I(name, suffix) name##_##suffix
#define HYLE_SIMD_SUFFIXED_SYMBOL(name, suffix) HYLE_SIMD_SUFFIXED_SYMBOL_I(name, suffix)
#if defined(HYLE_SIMD_EXACT)
#define HYLE_SIMD_EXACT_SYMBOL(name) \
  ::hyle::simd::detail::HYLE_SIMD_SUFFIXED_SYMBOL(name, HYLE_SIMD_EXACT_VARIANT)
#endif

inline Isa default_dispatched_isa() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return Isa::kAvx512;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  return Isa::kAvx2;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
  return Isa::kSse42;
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
  return Isa::kSve;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
  return Isa::kNeon;
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
  return Isa::kRvv;
#elif defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
  return Isa::kScalar;
#elif defined(HYLE_ARCH_X64)
  const auto tier = static_cast<std::uint32_t>(best_isa());
  if (tier >= static_cast<std::uint32_t>(Isa::kAvx512)) return Isa::kAvx512;
  if (tier >= static_cast<std::uint32_t>(Isa::kAvx2)) return Isa::kAvx2;
  if (tier >= static_cast<std::uint32_t>(Isa::kSse42)) return Isa::kSse42;
  return Isa::kScalar;
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (best_isa() == Isa::kSve) return Isa::kSve;
#endif
  return best_isa() == Isa::kNeon ? Isa::kNeon : Isa::kScalar;
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  return best_isa() == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
#else
  return Isa::kScalar;
#endif
}

#if defined(HYLE_ARCH_X64)
template <class Fn>
inline Fn resolve_default_variant(Fn scalar, Fn sse4, Fn avx2, Fn avx512) noexcept {
  switch (default_dispatched_isa()) {
    case Isa::kAvx512: return avx512;
    case Isa::kAvx2: return avx2;
    case Isa::kSse42: return sse4;
    default: return scalar;
  }
}
#elif defined(HYLE_ARCH_ARM64)
template <class Fn>
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
inline Fn resolve_default_variant(Fn scalar, Fn neon, Fn sve) noexcept {
  if (default_dispatched_isa() == Isa::kSve) return sve;
  return default_dispatched_isa() == Isa::kNeon ? neon : scalar;
}
#else
inline Fn resolve_default_variant(Fn scalar, Fn neon) noexcept {
  return default_dispatched_isa() == Isa::kNeon ? neon : scalar;
}
#endif
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
template <class Fn>
inline Fn resolve_default_variant(Fn scalar, Fn rvv) noexcept {
  return default_dispatched_isa() == Isa::kRvv ? rvv : scalar;
}
#else
template <class Fn>
inline Fn resolve_default_variant(Fn scalar) noexcept {
  return scalar;
}
#endif

}  // namespace hyle::simd::detail
