#pragma once

#include <cstdint>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"

// Internal helpers for batch-kernel dispatchers. This deliberately lives outside the public
// dispatch header: callers export fixed POD function signatures, while each module retains its
// own feature gates and fallback contract. These helpers only cover the common full-ISA ladder
// emitted by hyle_add_dispatched_kernel.
namespace hyle::simd::detail {

inline Isa default_dispatched_isa() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto tier = static_cast<std::uint32_t>(best_isa());
  if (tier >= static_cast<std::uint32_t>(Isa::kAvx512)) return Isa::kAvx512;
  if (tier >= static_cast<std::uint32_t>(Isa::kAvx2)) return Isa::kAvx2;
  if (tier >= static_cast<std::uint32_t>(Isa::kSse42)) return Isa::kSse42;
  return Isa::kScalar;
#elif defined(HYLE_ARCH_ARM64)
  return static_cast<std::uint32_t>(best_isa()) >= static_cast<std::uint32_t>(Isa::kNeon)
             ? Isa::kNeon
             : Isa::kScalar;
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
inline Fn resolve_default_variant(Fn scalar, Fn neon) noexcept {
  return default_dispatched_isa() == Isa::kNeon ? neon : scalar;
}
#else
template <class Fn>
inline Fn resolve_default_variant(Fn scalar) noexcept {
  return scalar;
}
#endif

}  // namespace hyle::simd::detail
