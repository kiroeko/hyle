#pragma once

#include <cstddef>

#include "hyle/platform.h"  // HYLE_ARCH_X64 (the borrow cores are x86 AVX2-only)

// kernels.h (internal) -- per-tier transcendental batch symbols the math dispatcher takes
// the address of. scalar (libm) is always defined + is the non-x86 / fallback floor. The
// Highway + xsimd AVX2 variants are vendored header-only and compiled UNCONDITIONALLY on
// x86 (their TUs always compile -- the headers are in-tree, can't be missing; math_dispatch
// enters them only when cpu_features().avx2). x86-only because they lower to AVX2 intrinsics.
// Not a public header.

namespace hyle::simd::detail {

// scalar libm reference (always available)
void exp_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;

#if defined(HYLE_ARCH_X64)
// Per-op best-of-breed borrow cores (x86 AVX2, ~3.5 ULP). The same-ISA bench picked the
// fastest library PER OP, so ONLY that op is borrowed from each lib (the others would be
// dead code): xsimd serves exp/tanh, Highway serves log. math_dispatch enters these only
// when cpu_features().avx2.
void exp_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;

// AVX-512 (16-wide) siblings of the AVX2 borrow cores -- same xsimd/Highway cores, just
// compiled at the AVX-512 tier so they lower to zmm. math_dispatch enters these only when
// best_isa() >= kAvx512; else it stays on the AVX2 cores above (then scalar floor).
void exp_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
// log: Highway's AVX3 target is HWY_BROKEN on MSVC (it would emit ymm, not zmm), so this
// 512 symbol exists ONLY on GCC/Clang. On MSVC, math_dispatch keeps the avx2 highway log
// (the widest Highway gives there) -- see transcendental_highway_avx512.cc MSVC EXCLUSION.
#if !defined(_MSC_VER)
void log_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
#endif
#endif

}  // namespace hyle::simd::detail
