#include "hyle/simd/ops.h"

#include <cstddef>

#include "hyle/platform.h"
#include "src/simd/ops/kernels.h"

// dispatch_ops.cc -- the BASELINE dispatcher TU (no -march). It maps best_isa() to one of
// the per-ISA kernel variants and forwards through a cached function pointer. Being
// baseline + LTO-off (hyle_simd sets INTERPROCEDURAL_OPTIMIZATION OFF) is what keeps the
// AVX2 codegen confined to its own TU: this file is the entry the dispatcher reaches FIRST
// on an unknown CPU, so it must contain no instruction above the baseline.

namespace hyle::simd {
namespace {

// Map the host's detected top tier to the BEST variant we actually compiled. On x86 we
// shipped scalar / sse4 / avx2; an AVX-512 host still uses the avx2 variant (we have no
// avx512 kernel yet -- it falls to the next lower compiled tier, never SIGILLs). On a
// non-x86 build only scalar exists, so this returns kScalar. This is the function-pointer
// equivalent of llama.cpp's startup score-and-pick.
Isa selected_tier() noexcept {
#if defined(HYLE_ARCH_X64)
  const Isa best = best_isa();
  const auto v = static_cast<std::uint32_t>(best);
  if (v >= static_cast<std::uint32_t>(Isa::kAvx2)) return Isa::kAvx2;
  if (v >= static_cast<std::uint32_t>(Isa::kSse42)) return Isa::kSse42;
  return Isa::kScalar;
#else
  return Isa::kScalar;  // only the scalar variant is compiled on non-x86 (NEON = simd-03)
#endif
}

using DotFn = float (*)(const float*, const float*, std::size_t) noexcept;
using SaxpyFn = void (*)(float, const float*, float*, std::size_t) noexcept;

DotFn resolve_dot() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx2: return &detail::dot_f32_avx2;
    case Isa::kSse42: return &detail::dot_f32_sse4;
#endif
    default: return &detail::dot_f32_scalar;
  }
}
SaxpyFn resolve_saxpy() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx2: return &detail::saxpy_f32_avx2;
    case Isa::kSse42: return &detail::saxpy_f32_sse4;
#endif
    default: return &detail::saxpy_f32_scalar;
  }
}

}  // namespace

// Each cached once via a function-local static (Meyers: thread-safe init-once, C++11+).
// After the first call the branch is a predictable load + one indirect call -- the kernel
// cost dwarfs it. No static-init-order issue: resolve_*() and best_isa() are all lazy.
float dot_f32(const float* x, const float* y, std::size_t n) noexcept {
  static const DotFn fn = resolve_dot();
  return fn(x, y, n);
}
void saxpy_f32(float a, const float* x, float* y, std::size_t n) noexcept {
  static const SaxpyFn fn = resolve_saxpy();
  fn(a, x, y, n);
}
Isa active_variant() noexcept {
  static const Isa tier = selected_tier();
  return tier;
}

}  // namespace hyle::simd
