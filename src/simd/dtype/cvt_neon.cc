// NEON batch dtype conversion (aarch64 baseline -- ARMv8 mandates Advanced SIMD; the
// fp16 path additionally needs ARMv8.2-A FP16 convert, which every Apple-Silicon core has).
// Only REACHED on an aarch64 build via the dtype dispatcher's HYLE_ARCH_ARM64 branch.
//
// fp16: uses the NATIVE hardware convert vcvt_f32_f16 / vcvt_f16_f32 (like x86 uses F16C
// _mm256_cvtph_ps). Both are IEEE round-to-nearest-even hardware, so FINITE values are
// bit-identical to the scalar oracle (dtype.h f16_to_f32/f32_to_f16); NaN PAYLOADS may
// differ (hardware keeps its own qNaN bits) -- the cross-check test compares NaN by
// is-NaN, not bits, exactly as it does for the F16C path.
//
// bf16: pure INTEGER bit-math (bf16 = top 16 bits of f32), so the NEON path replicates the
// scalar bit sequence EXACTLY with integer ops -- bit-identical to scalar including NaN
// payloads (same guarantee the AVX2 bf16 path gives). No hardware bf16 convert needed.
//
// HYLE's float16_t/bfloat16_t are {uint16_t bits} wrappers (not the compiler's __fp16), so
// we LOAD the raw u16 lanes and vreinterpret to float16x4_t for the hardware convert.

#include "src/simd/dtype/kernels.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

// fp16 -> f32: load 4 u16, reinterpret as float16x4, widen to float32x4 (native vcvt).
// 4-wide (the NEON fp16->f32 convert is 4-lane); scalar tail matches the oracle.
void f16_to_f32_neon(const float16_t* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint16x4_t raw = vld1_u16(reinterpret_cast<const std::uint16_t*>(src + i));
    const float16x4_t h = vreinterpret_f16_u16(raw);
    vst1q_f32(dst + i, vcvt_f32_f16(h));  // hardware widen, RNE
  }
  for (; i < n; ++i) dst[i] = f16_to_f32(src[i]);  // tail = scalar oracle
}

// f32 -> fp16: narrow 4 f32 to float16x4 (native vcvt, RNE), reinterpret to u16, store.
void f32_to_f16_neon(const float* src, float16_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t f = vld1q_f32(src + i);
    const float16x4_t h = vcvt_f16_f32(f);  // hardware narrow, RNE
    vst1_u16(reinterpret_cast<std::uint16_t*>(dst + i), vreinterpret_u16_f16(h));
  }
  for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);  // tail = scalar oracle
}

// bf16 -> f32: zero-extend each u16 to u32, shift left 16, reinterpret as f32. Exact
// (mirrors scalar bf16_to_f32 = bits<<16). 4-wide.
void bf16_to_f32_neon(const bfloat16_t* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint16x4_t h = vld1_u16(reinterpret_cast<const std::uint16_t*>(src + i));
    const uint32x4_t w = vshlq_n_u32(vmovl_u16(h), 16);  // widen u16->u32, <<16
    vst1q_f32(dst + i, vreinterpretq_f32_u32(w));
  }
  for (; i < n; ++i) dst[i] = bf16_to_f32(src[i]);  // tail = scalar oracle
}

// f32 -> bf16, round-to-nearest-even. Replicates the scalar f32_to_bf16 bit-for-bit:
//   abs  = u & 0x7FFFFFFF
//   incr = (abs < 0x7F800000) ? 0x7FFF + ((u>>16)&1) : 0     // finite: RNE bias; inf/nan: 0
//   snan = (abs > 0x7F800000) ? (1<<6) : 0                    // SNaN->QNaN sticky bit
//   out  = (snan | ((u + incr) >> 16))                        // low 16 bits
// All integer ops -> bit-identical to scalar (NaN payloads included). 4-wide.
void f32_to_bf16_neon(const float* src, bfloat16_t* dst, std::size_t n) noexcept {
  const uint32x4_t k7fffffff = vdupq_n_u32(0x7FFFFFFFu);
  const uint32x4_t kInf = vdupq_n_u32(0x7F800000u);
  const uint32x4_t k7fff = vdupq_n_u32(0x7FFFu);
  const uint32x4_t kOne = vdupq_n_u32(1u);
  const uint32x4_t kSnanBit = vdupq_n_u32(1u << 6);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint32x4_t u = vreinterpretq_u32_f32(vld1q_f32(src + i));
    const uint32x4_t absv = vandq_u32(u, k7fffffff);
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), kOne);
    // finite = abs < inf  (unsigned compare); incr = (0x7FFF + lsb) & finite_mask
    const uint32x4_t finite = vcltq_u32(absv, kInf);
    const uint32x4_t incr = vandq_u32(vaddq_u32(k7fff, lsb), finite);
    // snan sticky bit only when abs > inf (a NaN); vcgtq_u32 gives the mask
    const uint32x4_t snan = vandq_u32(vcgtq_u32(absv, kInf), kSnanBit);
    const uint32x4_t shifted = vshrq_n_u32(vaddq_u32(u, incr), 16);
    const uint32x4_t out32 = vorrq_u32(snan, shifted);
    // pack the low 16 bits of each u32 lane into 4 u16 lanes
    vst1_u16(reinterpret_cast<std::uint16_t*>(dst + i), vmovn_u32(out32));
  }
  for (; i < n; ++i) dst[i] = f32_to_bf16(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail
