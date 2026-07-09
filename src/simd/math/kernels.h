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
// simd-09 scalar references (always available oracle + fallback)
void sin_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void cos_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void exp2_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void log2_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void erf_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void atan_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void cbrt_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void expm1_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void log1p_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void asinh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void acosh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
// xsimd-only borrows (Highway has no gamma): scalar libm oracle + fallback.
void lgamma_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;
void tgamma_f32_scalar(const float* src, float* dst, std::size_t n) noexcept;

#if defined(HYLE_ARCH_X64)
// Per-op best-of-breed borrow cores (x86 AVX2, ~3.5 ULP). The same-ISA bench picked the
// fastest library PER OP, so ONLY that op is borrowed from each lib (the others would be
// dead code): xsimd serves exp/tanh, Highway serves log. math_dispatch enters these only
// when cpu_features().avx2.
void exp_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
// simd-09 AVX2 borrow cores, per-op winner (bench-decided): xsimd for exp2/erf, Highway for
// the rest. Only the winning lib's TU defines each (the loser would be dead code).
void exp2_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void erf_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
// xsimd-only borrow (no Highway equivalent -> single source). Only TGAMMA is vectorized:
// xsimd's vectorized LGAMMA has a lane-mixing bug (wrong result when a lane in [1.25,1.5) shares
// a vector with a lane in [0.75,1.25) -- reproduced GCC-x86/MSVC-x86/GCC-NEON, see
// docs/debug/xsimd-lgamma-lane-mixing-hunt.md). lgamma therefore stays on the scalar libm
// reference (no _xsimd_ variant); tgamma is unaffected.
void tgamma_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept;
void sin_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void cos_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log2_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void atan_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void cbrt_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void expm1_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void log1p_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void asinh_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;
void acosh_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept;

// AVX-512 (16-wide) siblings of the AVX2 borrow cores -- same xsimd/Highway cores, just
// compiled at the AVX-512 tier so they lower to zmm. math_dispatch enters these only when
// best_isa() >= kAvx512; else it stays on the AVX2 cores above (then scalar floor).
void exp_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
// simd-09 AVX-512 (16-wide) siblings -- xsimd wins exp2/erf.
void exp2_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
void erf_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
// xsimd-only borrow, 16-wide sibling (tgamma only; lgamma stays scalar -- vendor lane-mixing bug).
void tgamma_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept;
// log: Highway's AVX3 target is HWY_BROKEN on MSVC (it would emit ymm, not zmm), so this
// 512 symbol exists ONLY on GCC/Clang. On MSVC, math_dispatch keeps the avx2 highway log
// (the widest Highway gives there) -- see transcendental_highway_avx512.cc MSVC EXCLUSION.
#if !defined(_MSC_VER)
void log_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
// simd-09 Highway AVX-512 siblings (GCC/Clang only, same MSVC-AVX3-broken exclusion as log).
void sin_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void cos_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void log2_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void atan_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void cbrt_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void expm1_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void log1p_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void asinh_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
void acosh_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept;
#endif
#endif

#if defined(HYLE_ARCH_ARM64)
// aarch64 NEON borrow cores: Highway's contrib/math lowers to NEON here (probed: target=NEON,
// 4-wide, matches libm). Unlike x86's per-op xsimd/Highway split (a speed pick from
// benchmarking), arm64 uses Highway for ALL ops -- one verified core, consistent. math_dispatch
// enters these unconditionally on arm64 (NEON is the ARMv8 baseline; no runtime gate needed).
// xsimd-NEON as a future per-op speed tune if a NEON bench shows a win.
void exp_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void log_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void tanh_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
// simd-09 NEON borrow cores (Highway, all ops -- same one-verified-core policy).
void sin_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void cos_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void exp2_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void log2_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void erf_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void atan_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void cbrt_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void expm1_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void log1p_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void asinh_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
void acosh_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept;
// xsimd-only borrows: lgamma/tgamma have NO Highway equivalent, so on arm64 they can't use the
// Highway-NEON path the other ops do. These borrow xsimd's NEON core instead (the first xsimd
// core on arm64) -- verified to lower to NEON + match libm under qemu-aarch64. Defined in
// transcendental_xsimd_neon.cc (arm64 xsimd TU, distinct from the Highway-NEON TU). tgamma only:
// xsimd's vectorized lgamma lane-mixing bug reproduces on NEON too, so lgamma stays scalar.
void tgamma_f32_xsimd_neon(const float* src, float* dst, std::size_t n) noexcept;
#endif

}  // namespace hyle::simd::detail
