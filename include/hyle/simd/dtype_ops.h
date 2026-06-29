#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API
#include "hyle/simd/dtype.h"     // float16_t, bfloat16_t

// dtype_ops.h -- BATCH dtype conversions, runtime-dispatched (simd-05). Converting whole
// weight/activation arrays between fp16/bf16 and fp32 is the AI hot path (KV-cache, mixed
// precision). Same fn-ptr dispatch model as ops.h: fp16 conversions pick the F16C
// hardware kernel when the CPU has it (cpu_features().f16c), else the scalar core.
//
// bf16: no AVX2-tier hardware convert exists (AVX-512-BF16's vcvtne2ps is a later
// milestone), so bf16 batch is the scalar core. HONEST status: this is DEFERRED, not
// "unnecessary" -- the baseline scalar TU auto-vectorizes the bf16 shift to ~4-wide (SSE2),
// so an explicit 8-wide AVX2 bf16 kernel WOULD add throughput on AVX2 hosts. It's just not
// the highest-value next step (fp16 has the dedicated F16C instruction; bf16's real win is
// AVX-512-BF16). Left as a follow-up rather than hand-rolled now.

namespace hyle::simd {

// fp16 <-> fp32, n elements. F16C-accelerated where available.
HYLE_SIMD_API void f16_to_f32_batch(const float16_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f32_to_f16_batch(const float* src, float16_t* dst, std::size_t n) noexcept;

// bf16 <-> fp32, n elements. Scalar core (auto-vectorized); see header note.
HYLE_SIMD_API void bf16_to_f32_batch(const bfloat16_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f32_to_bf16_batch(const float* src, bfloat16_t* dst, std::size_t n) noexcept;

// The ISA tier the fp16 batch conversions selected (kAvx2 if F16C present, else kScalar).
// For tests/diagnostics -- proves the hardware path engages when the CPU supports it.
HYLE_SIMD_API Isa active_dtype_variant() noexcept;

}  // namespace hyle::simd
