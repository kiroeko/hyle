#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API
#include "hyle/simd/dtype.h"     // float16_t, bfloat16_t

// dtype_ops.h -- BATCH dtype conversions, runtime-dispatched (simd-05). Converting whole
// weight/activation arrays between fp16/bf16 and fp32 is the AI hot path (KV-cache, mixed
// precision). Same fn-ptr dispatch model as ops.h: fp16 conversions pick the F16C
// hardware kernel when the CPU has it (cpu_features().f16c), else the scalar core.
//
// bf16: vectorized with plain AVX2 INTEGER ops (cpu_features().avx2), NOT a dedicated
// AVX-512-BF16 instruction. bf16 IS the top 16 bits of f32, so bf16<->f32 is a shift
// (bf16->f32) or a round-to-nearest-even bias + shift (f32->bf16) -- both expressible in
// AVX2 (cvtepu16_epi32/slli, packus/permute4x64). Same approach as Highway. Falls back to
// the scalar core on non-AVX2 CPUs. (AVX-512-BF16's vcvtne2ps would only help on AVX-512
// hosts; the AVX2 path already beats scalar 3.5-6.9x and is bit-identical.)

namespace hyle::simd {

// fp16 <-> fp32, n elements. F16C-accelerated where available.
HYLE_SIMD_API void f16_to_f32_batch(const float16_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f32_to_f16_batch(const float* src, float16_t* dst, std::size_t n) noexcept;

// bf16 <-> fp32, n elements. Scalar core (auto-vectorized); see header note.
HYLE_SIMD_API void bf16_to_f32_batch(const bfloat16_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f32_to_bf16_batch(const float* src, bfloat16_t* dst, std::size_t n) noexcept;

// fp8 <-> fp32, n elements, both OCP formats. SCALAR bit-math (auto-vectorized by the compiler);
// there is no CPU hardware fp8-convert path to dispatch to -- hardware fp8 conversion is AVX10.2 /
// GPU-tensor-core only, absent on every AVX2/AVX-512 CPU and not emulated by Intel SDE. So unlike
// fp16 (F16C) / bf16 (AVX-512-BF16), fp8 has no wide-ISA kernel to gate; the scalar core IS the
// complete implementation until AVX10.2 hardware exists. E4M3 = weights/activations, E5M2 = grads.
HYLE_SIMD_API void f8e4m3_to_f32_batch(const float8_e4m3_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f32_to_f8e4m3_batch(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f8e5m2_to_f32_batch(const float8_e5m2_t* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f32_to_f8e5m2_batch(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept;

// f16 -> fp8 (the AI-native shape: fp8 tensors are typically produced FROM f16 activations, and it
// is what AVX10.2's fp8 converts consume). Bit-identical across scalar and the AVX10.2 hardware
// kernel (f16->f32 exact => single rounding), so unlike a hypothetical f32->fp8 hardware path this
// is a TRANSPARENT dispatch target -- the batch result never changes with the CPU. Auto-dispatches
// to the AVX10.2 kernel when the build has it (clang-20) and the CPU advertises avx10_2, else scalar.
HYLE_SIMD_API void f16_to_f8e4m3_batch(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept;
HYLE_SIMD_API void f16_to_f8e5m2_batch(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept;

// The ISA tier the fp16 batch conversions selected (kAvx2 if F16C present, else kScalar).
// For tests/diagnostics -- proves the hardware path engages when the CPU supports it.
HYLE_SIMD_API Isa active_dtype_variant() noexcept;

}  // namespace hyle::simd
