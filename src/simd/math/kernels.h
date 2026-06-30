#pragma once

#include <cstddef>

// kernels.h (internal) -- per-tier transcendental batch symbols the math dispatcher takes
// the address of. scalar (libm) is always defined; the SLEEF avx2 variant is defined only
// when HYLE_SIMD_SLEEF is set (its TU is gated in CMake). Not a public header.

namespace hyle::simd::detail {

// scalar libm reference (always available)
void exp_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;

#if defined(HYLE_SIMD_SLEEF)
// SLEEF AVX2 (8-wide; exp _u10, log/tanh _u35). Only built when SLEEF is wired in.
void exp_f32_sleef_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_sleef_avx2(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_sleef_avx2(const float* src, float* dst, std::size_t n) noexcept;
#endif

#if defined(HYLE_SIMD_HIGHWAY_MATH)
// Highway AVX2 (8-wide, contrib/math ~3.5 ULP). Only built when Highway-math is wired
// in. Preferred over SLEEF when present (faster log, ties exp/tanh; header-only).
void exp_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
#endif

}  // namespace hyle::simd::detail
