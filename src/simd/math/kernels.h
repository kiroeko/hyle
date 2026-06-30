#pragma once

#include <cstddef>

// kernels.h (internal) -- per-tier transcendental batch symbols the math dispatcher takes
// the address of. scalar (libm) is always defined + is the non-x86 / fallback floor. The
// Highway + xsimd AVX2 variants are vendored header-only and wired UNCONDITIONALLY on x86
// (their TUs always compile; math_dispatch enters them only when cpu_features().avx2). Not
// a public header.

namespace hyle::simd::detail {

// scalar libm reference (always available)
void exp_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;

#if defined(HYLE_SIMD_HIGHWAY_MATH)
// Highway AVX2 (8-wide, contrib/math ~3.5 ULP). Dispatcher picks it for log (fastest).
void exp_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
#endif

#if defined(HYLE_SIMD_XSIMD_MATH)
// xsimd AVX2 (8-wide, ~3.5 ULP). Per-op best-of-breed: dispatcher prefers xsimd for
// exp/tanh (fastest), Highway for log (fastest).
void exp_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
#endif

}  // namespace hyle::simd::detail
