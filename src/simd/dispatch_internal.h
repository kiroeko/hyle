#pragma once

#include <cstdint>

#include "hyle/core/environment.h"
#include "hyle/simd/target.h"
#include "src/simd/target_manifest.h"

// Internal helpers for batch-kernel dispatchers. This deliberately lives outside the public
// dispatch header: callers export fixed POD function signatures, while each module retains its
// own feature gates and fallback contract. These helpers only cover the common full-ISA ladder
// emitted by hyle_add_dispatched_kernel.
namespace hyle::simd::detail {

// Highway disables its AVX3 targets under the classic MSVC frontend. The HYLE
// vqsort port has the same compiler boundary: cl.exe can compile the AVX-512
// instantiation, but its generated code corrupts 64-bit, UInt128 and KV data.
// Keep this policy separate from the CPU tier so a process may truthfully
// report an AVX-512 product runtime while sort uses the safe AVX2 product TU.
// clang-cl defines _MSC_VER too, but is not affected and must retain AVX-512.
constexpr KernelTarget sort_target_for_product_target(
    KernelTarget product_target) noexcept {
#if HYLE_COMPILER_MSVC
  if (product_target == KernelTarget::kAvx512)
    return KernelTarget::kAvx2;
#endif
  return product_target;
}

#if HYLE_ARCH_X64
// Exact runtime prerequisites for the optional x86 product TUs. These are
// deliberately narrower than a canonical target group: a specialty kernel may be useful
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
    const platform::CpuFeatureSet& features) noexcept {
  using enum platform::CpuFeature;
  // -mavx2/-mavxvnni may use the AVX2 integer face and always require an
  // OS-saved YMM state. Keep AVX explicit even for malformed/masked CPUID sets.
  const bool avx2_tu = features.has_all(
      platform::CpuFeatureSet{kX86Avx, kX86Avx2});
  // The AVX-512 VNNI/BF16 TUs compile with F+BW+VL+DQ in addition to their
  // specialty extension. Clang also enables FMA/F16C as prerequisites of
  // -mavx512f and may emit them anywhere in the TU (including scalar tails).
  // A hypervisor may mask those legacy bits independently.
  const bool avx512_tu = features.has_all(platform::CpuFeatureSet{
      kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Avx512F,
      kX86Avx512Bw, kX86Avx512Vl, kX86Avx512Dq});
  return X86SpecialtySupport{
      .avx2_int = avx2_tu,
      .f16c = features.has_all(
          platform::CpuFeatureSet{kX86Avx, kX86F16c}),
      .avx_vnni = avx2_tu && features.has(kX86AvxVnni),
      .avx512_vnni = avx512_tu && features.has(kX86Avx512Vnni),
      .avx512_bf16 = avx512_tu && features.has(kX86Avx512Bf16),
      // The -mavx10.2-512 TU requires the precise Platform version/width/OS
      // fact and also permits FMA/F16C; retain those legacy prerequisites for
      // independently masked virtual CPUs.
      .avx10_2 = avx2_tu && features.has_all(
          platform::CpuFeatureSet{
              kX86Fma, kX86F16c, kX86Avx10_2_512}),
  };
}
#endif

#define HYLE_SIMD_SUFFIXED_SYMBOL_I(name, suffix) name##_##suffix
#define HYLE_SIMD_SUFFIXED_SYMBOL(name, suffix) HYLE_SIMD_SUFFIXED_SYMBOL_I(name, suffix)
#if defined(HYLE_SIMD_EXACT)
#define HYLE_SIMD_EXACT_SYMBOL(name) \
  ::hyle::simd::detail::HYLE_SIMD_SUFFIXED_SYMBOL(name, HYLE_SIMD_EXACT_VARIANT)
#endif

inline KernelTarget default_dispatched_target() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return KernelTarget::kAvx512;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  return KernelTarget::kAvx2;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
  return KernelTarget::kSse4;
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
  return KernelTarget::kSve;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
  return KernelTarget::kNeon;
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
  return KernelTarget::kRvv;
#elif defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
  return KernelTarget::kScalar;
#elif HYLE_ARCH_X64
  const auto& features = platform::cpu_features();
  if (kernel_target_supported(KernelTarget::kAvx512, features))
    return KernelTarget::kAvx512;
  if (kernel_target_supported(KernelTarget::kAvx2, features))
    return KernelTarget::kAvx2;
  if (kernel_target_supported(KernelTarget::kSse4, features))
    return KernelTarget::kSse4;
  return KernelTarget::kScalar;
#elif HYLE_ARCH_ARM64
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (kernel_target_supported(KernelTarget::kSve))
    return KernelTarget::kSve;
#endif
  return kernel_target_supported(KernelTarget::kNeon)
             ? KernelTarget::kNeon
             : KernelTarget::kScalar;
#elif HYLE_ARCH_RISCV64 && defined(HYLE_SIMD_HAS_RVV_VARIANT)
  return kernel_target_supported(KernelTarget::kRvv)
             ? KernelTarget::kRvv
             : KernelTarget::kScalar;
#else
  return KernelTarget::kScalar;
#endif
}

#if HYLE_ARCH_X64
template <class Fn>
inline Fn resolve_default_variant(Fn scalar, Fn sse4, Fn avx2, Fn avx512) noexcept {
  switch (default_dispatched_target()) {
    case KernelTarget::kAvx512: return avx512;
    case KernelTarget::kAvx2: return avx2;
    case KernelTarget::kSse4: return sse4;
    default: return scalar;
  }
}
#elif HYLE_ARCH_ARM64
template <class Fn>
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
inline Fn resolve_default_variant(Fn scalar, Fn neon, Fn sve) noexcept {
  if (default_dispatched_target() == KernelTarget::kSve) return sve;
  return default_dispatched_target() == KernelTarget::kNeon ? neon : scalar;
}
#else
inline Fn resolve_default_variant(Fn scalar, Fn neon) noexcept {
  return default_dispatched_target() == KernelTarget::kNeon ? neon : scalar;
}
#endif
#elif HYLE_ARCH_RISCV64 && defined(HYLE_SIMD_HAS_RVV_VARIANT)
template <class Fn>
inline Fn resolve_default_variant(Fn scalar, Fn rvv) noexcept {
  return default_dispatched_target() == KernelTarget::kRvv ? rvv : scalar;
}
#else
template <class Fn>
inline Fn resolve_default_variant(Fn scalar) noexcept {
  return scalar;
}
#endif

}  // namespace hyle::simd::detail
