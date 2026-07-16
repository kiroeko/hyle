#include "hyle/simd/ops.h"

#include <cstddef>

#include "src/simd/dispatch_internal.h"
#include "src/simd/ops/kernels.h"

// dispatch_ops.cc -- the BASELINE dispatcher TU (no -march). It maps best_isa() to one of
// the per-ISA kernel variants and forwards through a cached function pointer. Being
// baseline + LTO-off (hyle_simd sets INTERPROCEDURAL_OPTIMIZATION OFF) is what keeps the
// AVX2 codegen confined to its own TU: this file is the entry the dispatcher reaches FIRST
// on an unknown CPU, so it must contain no instruction above the baseline.

namespace hyle::simd {
namespace {

using DotFn = float (*)(const float*, const float*, std::size_t) noexcept;
using SaxpyFn = void (*)(float, const float*, float*, std::size_t) noexcept;
using DotF64Fn = double (*)(const double*, const double*, std::size_t) noexcept;
using SaxpyF64Fn = void (*)(double, const double*, double*, std::size_t) noexcept;

DotFn resolve_dot() noexcept {
#if defined(HYLE_ARCH_X64)
  return detail::resolve_default_variant<DotFn>(&detail::dot_f32_scalar, &detail::dot_f32_sse4,
                                                &detail::dot_f32_avx2, &detail::dot_f32_avx512);
#elif defined(HYLE_ARCH_ARM64)
  return detail::resolve_default_variant<DotFn>(&detail::dot_f32_scalar, &detail::dot_f32_neon);
#else
  return detail::resolve_default_variant<DotFn>(&detail::dot_f32_scalar);
#endif
}
SaxpyFn resolve_saxpy() noexcept {
#if defined(HYLE_ARCH_X64)
  return detail::resolve_default_variant<SaxpyFn>(&detail::saxpy_f32_scalar, &detail::saxpy_f32_sse4,
                                                  &detail::saxpy_f32_avx2, &detail::saxpy_f32_avx512);
#elif defined(HYLE_ARCH_ARM64)
  return detail::resolve_default_variant<SaxpyFn>(&detail::saxpy_f32_scalar, &detail::saxpy_f32_neon);
#else
  return detail::resolve_default_variant<SaxpyFn>(&detail::saxpy_f32_scalar);
#endif
}
DotF64Fn resolve_dot_f64() noexcept {
#if defined(HYLE_ARCH_X64)
  return detail::resolve_default_variant<DotF64Fn>(&detail::dot_f64_scalar, &detail::dot_f64_sse4,
                                                   &detail::dot_f64_avx2, &detail::dot_f64_avx512);
#elif defined(HYLE_ARCH_ARM64)
  return detail::resolve_default_variant<DotF64Fn>(&detail::dot_f64_scalar, &detail::dot_f64_neon);
#else
  return detail::resolve_default_variant<DotF64Fn>(&detail::dot_f64_scalar);
#endif
}
SaxpyF64Fn resolve_saxpy_f64() noexcept {
#if defined(HYLE_ARCH_X64)
  return detail::resolve_default_variant<SaxpyF64Fn>(&detail::saxpy_f64_scalar,
                                                     &detail::saxpy_f64_sse4,
                                                     &detail::saxpy_f64_avx2,
                                                     &detail::saxpy_f64_avx512);
#elif defined(HYLE_ARCH_ARM64)
  return detail::resolve_default_variant<SaxpyF64Fn>(&detail::saxpy_f64_scalar,
                                                     &detail::saxpy_f64_neon);
#else
  return detail::resolve_default_variant<SaxpyF64Fn>(&detail::saxpy_f64_scalar);
#endif
}
using AddF32Fn = void (*)(const float*, const float*, float*, std::size_t) noexcept;
using AddF64Fn = void (*)(const double*, const double*, double*, std::size_t) noexcept;
AddF32Fn resolve_add_f32() noexcept {
#if defined(HYLE_ARCH_X64)
  return detail::resolve_default_variant<AddF32Fn>(&detail::add_f32_scalar, &detail::add_f32_sse4,
                                                   &detail::add_f32_avx2, &detail::add_f32_avx512);
#elif defined(HYLE_ARCH_ARM64)
  return detail::resolve_default_variant<AddF32Fn>(&detail::add_f32_scalar, &detail::add_f32_neon);
#else
  return detail::resolve_default_variant<AddF32Fn>(&detail::add_f32_scalar);
#endif
}
AddF64Fn resolve_add_f64() noexcept {
#if defined(HYLE_ARCH_X64)
  return detail::resolve_default_variant<AddF64Fn>(&detail::add_f64_scalar, &detail::add_f64_sse4,
                                                   &detail::add_f64_avx2, &detail::add_f64_avx512);
#elif defined(HYLE_ARCH_ARM64)
  return detail::resolve_default_variant<AddF64Fn>(&detail::add_f64_scalar, &detail::add_f64_neon);
#else
  return detail::resolve_default_variant<AddF64Fn>(&detail::add_f64_scalar);
#endif
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
  static const Isa tier = detail::default_dispatched_isa();
  return tier;
}

}  // namespace hyle::simd
