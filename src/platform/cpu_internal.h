#pragma once

#include "hyle/platform/cpu.h"

namespace hyle::platform::detail {

#if HYLE_ARCH_X64
// Canonical x86 dispatch groups. Keep this private and constexpr so the probe
// and synthetic tests share one definition without exposing implementation
// policy as part of the public CPU-feature API.
struct X86IsaGroups {
  bool avx2;
  bool avx512;
};

constexpr X86IsaGroups x86_isa_groups(const CpuFeatures& f) noexcept {
  // The AVX2 product TUs are built with the complete Highway-compatible group.
  // Require AVX explicitly even for malformed or independently masked CPUID
  // sets: it also records that the OS saves YMM state.
  const bool avx2 = f.avx && f.avx2 && f.fma && f.f16c && f.lzcnt &&
                    f.bmi1 && f.bmi2 && f.sse42;
  const bool avx512 = avx2 && f.avx512f && f.avx512bw && f.avx512vl &&
                      f.avx512dq && f.avx512cd;
  return X86IsaGroups{.avx2 = avx2, .avx512 = avx512};
}
#endif

}  // namespace hyle::platform::detail
