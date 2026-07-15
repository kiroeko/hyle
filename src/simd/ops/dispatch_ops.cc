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
// shipped scalar / sse4 / avx2 / avx512; the dispatcher picks the highest tier whose kernel
// exists, else falls to the next lower one (never SIGILLs -- each tier is runtime-gated by
// best_isa()). On a non-x86 build only scalar exists, so this returns kScalar. This is the
// function-pointer equivalent of llama.cpp's startup score-and-pick.
Isa selected_tier() noexcept {
#if defined(HYLE_ARCH_X64)
  const Isa best = best_isa();
  const auto v = static_cast<std::uint32_t>(best);
  if (v >= static_cast<std::uint32_t>(Isa::kAvx512)) return Isa::kAvx512;
  if (v >= static_cast<std::uint32_t>(Isa::kAvx2)) return Isa::kAvx2;
  if (v >= static_cast<std::uint32_t>(Isa::kSse42)) return Isa::kSse42;
  return Isa::kScalar;
#elif defined(HYLE_ARCH_ARM64)
  // aarch64 ships scalar + neon; NEON is the ARMv8 baseline so best_isa() returns kNeon on
  // any real arm64 CPU (cpu.cc). Pick neon when present, else the scalar floor.
  if (static_cast<std::uint32_t>(best_isa()) >= static_cast<std::uint32_t>(Isa::kNeon))
    return Isa::kNeon;
  return Isa::kScalar;
#else
  return Isa::kScalar;  // unknown arch: portable floor
#endif
}

using DotFn = float (*)(const float*, const float*, std::size_t) noexcept;
using SaxpyFn = void (*)(float, const float*, float*, std::size_t) noexcept;
using DotF64Fn = double (*)(const double*, const double*, std::size_t) noexcept;
using SaxpyF64Fn = void (*)(double, const double*, double*, std::size_t) noexcept;

DotFn resolve_dot() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx512: return &detail::dot_f32_avx512;
    case Isa::kAvx2: return &detail::dot_f32_avx2;
    case Isa::kSse42: return &detail::dot_f32_sse4;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &detail::dot_f32_neon;
#endif
    default: return &detail::dot_f32_scalar;
  }
}
SaxpyFn resolve_saxpy() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx512: return &detail::saxpy_f32_avx512;
    case Isa::kAvx2: return &detail::saxpy_f32_avx2;
    case Isa::kSse42: return &detail::saxpy_f32_sse4;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &detail::saxpy_f32_neon;
#endif
    default: return &detail::saxpy_f32_scalar;
  }
}
DotF64Fn resolve_dot_f64() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx512: return &detail::dot_f64_avx512;
    case Isa::kAvx2: return &detail::dot_f64_avx2;
    case Isa::kSse42: return &detail::dot_f64_sse4;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &detail::dot_f64_neon;
#endif
    default: return &detail::dot_f64_scalar;
  }
}
SaxpyF64Fn resolve_saxpy_f64() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx512: return &detail::saxpy_f64_avx512;
    case Isa::kAvx2: return &detail::saxpy_f64_avx2;
    case Isa::kSse42: return &detail::saxpy_f64_sse4;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &detail::saxpy_f64_neon;
#endif
    default: return &detail::saxpy_f64_scalar;
  }
}
using AddF32Fn = void (*)(const float*, const float*, float*, std::size_t) noexcept;
using AddF64Fn = void (*)(const double*, const double*, double*, std::size_t) noexcept;
AddF32Fn resolve_add_f32() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx512: return &detail::add_f32_avx512;
    case Isa::kAvx2: return &detail::add_f32_avx2;
    case Isa::kSse42: return &detail::add_f32_sse4;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &detail::add_f32_neon;
#endif
    default: return &detail::add_f32_scalar;
  }
}
AddF64Fn resolve_add_f64() noexcept {
  switch (selected_tier()) {
#if defined(HYLE_ARCH_X64)
    case Isa::kAvx512: return &detail::add_f64_avx512;
    case Isa::kAvx2: return &detail::add_f64_avx2;
    case Isa::kSse42: return &detail::add_f64_sse4;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &detail::add_f64_neon;
#endif
    default: return &detail::add_f64_scalar;
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
double dot_f64(const double* x, const double* y, std::size_t n) noexcept {
  static const DotF64Fn fn = resolve_dot_f64();
  return fn(x, y, n);
}
void saxpy_f64(double a, const double* x, double* y, std::size_t n) noexcept {
  static const SaxpyF64Fn fn = resolve_saxpy_f64();
  fn(a, x, y, n);
}
void add_f32(const float* x, const float* y, float* z, std::size_t n) noexcept {
  static const AddF32Fn fn = resolve_add_f32();
  fn(x, y, z, n);
}
void add_f64(const double* x, const double* y, double* z, std::size_t n) noexcept {
  static const AddF64Fn fn = resolve_add_f64();
  fn(x, y, z, n);
}
Isa active_variant() noexcept {
  static const Isa tier = selected_tier();
  return tier;
}

}  // namespace hyle::simd
