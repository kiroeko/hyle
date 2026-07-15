#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// ops.h -- RUNTIME-DISPATCHED numeric primitives (simd-04). This is the half simd-01/02
// left open: simd-01 built the CPUID DETECTOR (best_isa()), simd-02 built the per-ISA
// backends, but nothing connected them -- best_isa() was dead. These entry points close
// that gap: call dot_f32() and it runs the AVX2 kernel on an AVX2 CPU, the SSE4 kernel on
// an older one, the scalar floor otherwise -- chosen ONCE at first use from best_isa().
//
// DISPATCH MODEL (be honest about the cost): this is the FUNCTION-POINTER-TABLE model
// (like Highway's HWY_DYNAMIC_DISPATCH). Each call goes through one cached function
// pointer -> ONE indirect call, then the selected kernel runs at full speed. That single
// indirection is the price of "one binary, any CPU". The ZERO-indirection alternative
// (per-variant DLLs, dlopen the winner, then all calls inside are direct -- llama.cpp's
// GGML_BACKEND_DL) is a heavier, later milestone; conventions §10 documents both. For an
// in-process library entry point, the fn-ptr table is the right first cut.
//
// SAFETY: the AVX2 kernel TU is compiled with -mavx2 but is ONLY ever entered when
// best_isa() >= kAvx2, so an older CPU never executes an instruction it lacks (no SIGILL).
// The variant TUs are isolated (one ISA per TU, LTO off) so the baseline dispatcher never
// inlines a wide instruction.

namespace hyle::simd {

// dot product: sum(x[i]*y[i]) over n floats. Dispatched to the best ISA kernel.
HYLE_SIMD_API float dot_f32(const float* x, const float* y, std::size_t n) noexcept;

// saxpy: y[i] = a*x[i] + y[i] over n floats (BLAS level-1). Dispatched.
HYLE_SIMD_API void saxpy_f32(float a, const float* x, float* y, std::size_t n) noexcept;

// f64 (double) BLAS-1 siblings -- the CAE/solver/geometry path where the wider accumulator
// matters (dot's 4-way partial sum also drifts less than a naive f64 loop). Same D-tag kernel,
// half the lane width (AVX2: 4 doubles vs 8 floats). Dispatched to the best ISA.
HYLE_SIMD_API double dot_f64(const double* x, const double* y, std::size_t n) noexcept;
HYLE_SIMD_API void saxpy_f64(double a, const double* x, double* y, std::size_t n) noexcept;

// Elementwise add: z[i] = x[i] + y[i] (3-buffer, bandwidth-bound like saxpy). The elementwise
// vector-add primitive simd lacked -- a compute-07 host vecadd was hand-rolling it. Dispatched.
HYLE_SIMD_API void add_f32(const float* x, const float* y, float* z, std::size_t n) noexcept;
HYLE_SIMD_API void add_f64(const double* x, const double* y, double* z, std::size_t n) noexcept;

// Accurate f32 reductions (P0-3): accumulate in f64 (vector PromoteLower/Upper + f64 FMA),
// cast back to f32. Error ~O(n·ε_f64) vs fast path O(n·ε_f32). Consumer opts in; does not
// replace dot_f32. See simd-complexity-opportunities.md P0-3.
HYLE_SIMD_API float sum_f32_accurate(const float* x, std::size_t n) noexcept;
HYLE_SIMD_API float dot_f32_accurate(const float* x, const float* y, std::size_t n) noexcept;

// The ISA tier the dispatcher actually selected (resolved once from best_isa()). For
// tests/diagnostics: proves dispatch picked the host's best variant, not just the floor.
HYLE_SIMD_API Isa active_variant() noexcept;

}  // namespace hyle::simd
