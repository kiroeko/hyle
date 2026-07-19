#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "hyle/c/simd_types.h"
#include "hyle/macros.h"

// dtype.h -- AI numeric storage types fp16 (IEEE-754 binary16) and bf16 (bfloat16) +
// their scalar <-> fp32 conversions (simd-design dtype layer). bf16 is the AI training/
// inference workhorse; clay had ZERO support. fp16 has a hardware convert path (F16C);
// see dtype_ops.h for the dispatched batch conversions.
//
// The scalar conversions here are the ALWAYS-AVAILABLE reference + fallback: branch-exact
// IEEE-754 semantics (subnormal / inf / nan / round-to-nearest-even), transcribed from
// Google Highway's base.h (F32FromF16 / F16FromF32 / F32FromBF16 / BF16FromF32) -- NOT a
// lookup table (clay's old approach) and NOT hand-derived (this bit-math is where bugs
// hide). They are the oracle the F16C hardware path is cross-checked against.

namespace hyle::simd {

// ---- storage types: uint16 bit-carriers (Highway's portable wrapper, sans the C++23
// std::float16_t native path). Trivially copyable, 2-byte, ABI-clean. ----------------

// fp8 storage types (OCP 8-bit float, the low-precision inference/training formats). Two layouts,
// both 1-byte bit-carriers:
//   * E4M3 (a.k.a. E4M3FN): 1 sign, 4 exp (bias 7), 3 mantissa. NO infinity; the only NaN is
//     0x7F/0xFF (exp=1111, mant=111). Max normal = 448. Used for weights/activations (more
//     mantissa precision, smaller range).
//   * E5M2: 1 sign, 5 exp (bias 15), 2 mantissa. IEEE-like: has inf (exp=11111,mant=00) and NaN
//     (exp=11111,mant!=0). Max normal = 57344. Used for gradients (more range, less precision).
// Encoding transcribed from the authoritative NVIDIA CUDA cuda_fp8.hpp algorithm (matches the OCP
// "OCP 8-bit Floating Point" spec). NOT borrowed at runtime (no host fp8 lib to vendor; ggml/
// Highway have none, CUDA's is device-oriented) -- the scalar bit-math here IS the reference, same
// as fp16/bf16 are transcribed from Highway. See dtype_ops.h for the batch conversions.

namespace detail {
HYLE_FORCE_INLINE float bits_to_f32(std::uint32_t b) noexcept {
  float f;
  std::memcpy(&f, &b, sizeof(f));
  return f;
}
HYLE_FORCE_INLINE std::uint32_t f32_to_bits(float f) noexcept {
  std::uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}
}  // namespace detail

// ---- fp16 <-> fp32 (Highway base.h F32FromF16 / F16FromF32) --------------------------

// fp16 -> fp32. Subnormals use a float multiply (exact); normals rebias exp 15->127 and
// shift mantissa <<13; exp==0x1F (inf/nan) maps exp->0xFF preserving the distinction.
HYLE_FORCE_INLINE float f16_to_f32(float16_t h) noexcept {
  const std::uint32_t bits16 = h.bits;
  const std::uint32_t sign = bits16 >> 15;
  const std::uint32_t biased_exp = (bits16 >> 10) & 0x1F;
  const std::uint32_t mantissa = bits16 & 0x3FF;
  if (biased_exp == 0) {  // subnormal or zero
    const float sub = (1.0f / 16384) * (static_cast<float>(mantissa) * (1.0f / 1024));
    return sign ? -sub : sub;
  }
  const std::uint32_t biased_exp32 = (biased_exp == 31) ? 0xFF : biased_exp + (127 - 15);
  const std::uint32_t mantissa32 = mantissa << (23 - 10);
  return detail::bits_to_f32((sign << 31) | (biased_exp32 << 23) | mantissa32);
}

// fp32 -> fp16. Round-to-nearest-even via an odd-bit + (half-ULP - 1) bias added to the
// mantissa before the shift; overflow -> inf (mantissa 0), underflow -> subnormal/zero,
// NaN forced to qNaN mantissa 0x3FF (never collapses to inf).
HYLE_FORCE_INLINE float16_t f32_to_f16(float f) noexcept {
  const std::uint32_t bits32 = detail::f32_to_bits(f);
  const std::uint32_t sign = bits32 >> 31;
  const std::uint32_t biased_exp32 = (bits32 >> 23) & 0xFF;
  constexpr std::uint32_t kMantMask = 0x7FFFFF;
  const std::uint32_t mantissa32 = bits32 & kMantMask;

  // ULP bit index clamped to [13,24]: where fp16's least bit lands within fp32 mantissa.
  int idx = 126 - static_cast<int>(biased_exp32);
  idx = idx < 13 ? 13 : (idx > 24 ? 24 : idx);
  const std::uint32_t odd_bit = ((mantissa32 | 0x800000u) >> idx) & 1u;
  const std::uint32_t rounded =
      mantissa32 + odd_bit + (1u << (idx - 1)) - 1u;
  const bool carry = rounded >= (1u << 23);
  const int exp = static_cast<int>(biased_exp32) - 127 + (carry ? 1 : 0);

  if (exp < -24) return float16_t::from_bits(static_cast<std::uint16_t>(sign << 15));

  const bool is_nan = (biased_exp32 == 255) && (mantissa32 != 0);
  const bool overflowed = exp >= 16;
  int be16 = exp + 15;
  be16 = be16 < 0 ? 0 : (be16 > 31 ? 31 : be16);
  const int sub_exp = (-14 - exp) > 0 ? (-14 - exp) : 0;
  const std::uint32_t shifted_mant = (rounded & kMantMask) >> (23 - 10 + sub_exp);
  const std::uint32_t leading = (sub_exp == 0) ? 0u : (1024u >> sub_exp);
  const std::uint32_t mant16 =
      is_nan ? 0x3FFu : (overflowed ? 0u : (leading + shifted_mant));
  const std::uint16_t out = static_cast<std::uint16_t>(
      (sign << 15) | (static_cast<std::uint32_t>(be16) << 10) | mant16);
  return float16_t::from_bits(out);
}

// ---- bf16 <-> fp32 (Highway base.h F32FromBF16 / BF16FromF32) ------------------------

// bf16 -> fp32: bf16 IS the top 16 bits of fp32 -- just shift up and reinterpret.
HYLE_FORCE_INLINE float bf16_to_f32(bfloat16_t b) noexcept {
  return detail::bits_to_f32(static_cast<std::uint32_t>(b.bits) << 16);
}

// fp32 -> bf16: round-to-nearest-even. Increment = 0x7FFF + (result LSB) for FINITE
// inputs (so ties round to even), 0 for inf/nan (no overflow into a wrong exponent). NaN
// additionally ORs bit 6 so it stays NaN (SNaN -> QNaN), never collapses to inf.
HYLE_FORCE_INLINE bfloat16_t f32_to_bf16(float f) noexcept {
  const std::uint32_t u = detail::f32_to_bits(f);
  const std::uint32_t abs = u & 0x7FFFFFFFu;
  const std::uint32_t incr = (abs < 0x7F800000u) ? (0x7FFFu + ((u >> 16) & 1u)) : 0u;
  const std::uint32_t snan = (abs > 0x7F800000u) ? (1u << 6) : 0u;
  const std::uint16_t out = static_cast<std::uint16_t>(snan | ((u + incr) >> 16));
  return bfloat16_t::from_bits(out);
}

// ---- fp8 <-> fp32 (transcribed from CUDA cuda_fp8.hpp; OCP E4M3FN / E5M2) ------------
// The two formats share the same shape (sign|exp|mantissa, round-to-nearest-even, saturate to
// max-normal on overflow) and differ only in field widths + special-value rules; the helpers
// below are parameterized on those. SATURATE on overflow (OCP inference convention / CUDA
// __NV_SATFINITE): overflow yields max-normal, not inf (E4M3 has no inf at all). NaN inputs map
// to the format's canonical NaN. Round-to-nearest-even on the discarded mantissa bits.

namespace detail {
// Shared decode: fp8 bits -> fp32. kExpBits/kManBits/kBias describe the format; kHasInf selects
// E5M2 (inf at max exp, mant 0) vs E4M3 (max exp+max mant is the sole NaN, no inf).
template <int kExpBits, int kManBits, int kBias, bool kHasInf>
HYLE_FORCE_INLINE float f8_to_f32(std::uint8_t bits) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 7) & 1u;
  const std::uint32_t exp = (static_cast<std::uint32_t>(bits) >> kManBits) & ((1u << kExpBits) - 1u);
  const std::uint32_t man = static_cast<std::uint32_t>(bits) & ((1u << kManBits) - 1u);
  const std::uint32_t kMaxExp = (1u << kExpBits) - 1u;
  const std::uint32_t kManMax = (1u << kManBits) - 1u;
  if (exp == kMaxExp) {
    if (kHasInf) {  // E5M2: exp all-ones -> inf (man 0) or NaN (man != 0)
      const std::uint32_t e = 0xFFu << 23;
      const std::uint32_t m = man << (23 - kManBits);
      return detail::bits_to_f32((sign << 31) | e | m);
    }
    // E4M3: the ONLY encoding at max exp+max mant is NaN; everything else is a normal number.
    if (man == kManMax) {
      return detail::bits_to_f32((sign << 31) | (0xFFu << 23) | (1u << 22));  // qNaN
    }
  }
  if (exp == 0) {  // zero or subnormal: value = (-1)^s * 2^(1-bias) * (man / 2^kManBits)
    if (man == 0) return sign ? -0.0f : 0.0f;
    const float scale = static_cast<float>(1u << kManBits);
    const float sub = std::ldexp(static_cast<float>(man) / scale, 1 - kBias);
    return sign ? -sub : sub;
  }
  // normal: rebias exp to fp32 (127) and left-align mantissa.
  const std::uint32_t e32 = exp - static_cast<std::uint32_t>(kBias) + 127u;
  const std::uint32_t m32 = man << (23 - kManBits);
  return detail::bits_to_f32((sign << 31) | (e32 << 23) | m32);
}

// Shared encode: fp32 -> fp8 bits, round-to-nearest-even, saturate-to-max-normal on overflow.
// kMaxNorm is the largest finite fp8 magnitude for the format (E4M3=448, E5M2=57344); kNanBits is
// the canonical NaN payload to emit for a NaN input.
template <int kExpBits, int kManBits, int kBias, bool kHasInf>
HYLE_FORCE_INLINE std::uint8_t f32_to_f8(float f, float kMaxNorm, std::uint8_t kMaxNormBits,
                                         std::uint8_t kNanBits) noexcept {
  const std::uint32_t u = detail::f32_to_bits(f);
  const std::uint8_t sign = static_cast<std::uint8_t>((u >> 31) << 7);
  const std::uint32_t abs = u & 0x7FFFFFFFu;
  if (abs > 0x7F800000u) return static_cast<std::uint8_t>(sign | kNanBits);  // NaN
  if (abs == 0x7F800000u) {  // +/-inf input
    // E5M2 HAS inf: preserve it (exp all-ones, mant 0). E4M3 has no inf: saturate to max-normal.
    if (kHasInf) return static_cast<std::uint8_t>(sign | (((1u << kExpBits) - 1u) << kManBits));
    return static_cast<std::uint8_t>(sign | kMaxNormBits);
  }
  const float fabs_v = detail::bits_to_f32(abs);
  // FINITE overflow: saturate to max-normal (OCP __NV_SATFINITE) for both formats -- a too-large
  // finite value clamps to the largest representable, it does NOT become inf.
  if (fabs_v > kMaxNorm) return static_cast<std::uint8_t>(sign | kMaxNormBits);

  const std::int32_t e32 = static_cast<std::int32_t>((abs >> 23) & 0xFFu) - 127;
  const std::int32_t exp8 = e32 + kBias;  // biased exponent in fp8 terms
  const std::uint32_t man23 = abs & 0x7FFFFFu;

  if (exp8 >= 1) {  // normal fp8
    const int drop = 23 - kManBits;               // fp32 mantissa bits discarded
    std::uint32_t man = man23 >> drop;             // top kManBits of the mantissa
    const std::uint32_t rem = man23 & ((1u << drop) - 1u);
    const std::uint32_t half = 1u << (drop - 1);
    std::uint32_t be = static_cast<std::uint32_t>(exp8);
    if (rem > half || (rem == half && (man & 1u))) man++;  // round-to-nearest-even
    if (man >> kManBits) { man = 0; be++; }        // mantissa carried out (1.11..+1 -> 10.00): exp++
    if (static_cast<int>(be) >= ((1 << kExpBits) - (kHasInf ? 1 : 0))) {
      return static_cast<std::uint8_t>(sign | kMaxNormBits);  // rounded up into overflow -> saturate
    }
    return static_cast<std::uint8_t>(sign | (be << kManBits) | (man & ((1u << kManBits) - 1u)));
  }
  // subnormal fp8 (or underflow to zero): shift the implicit-1 mantissa right by (1 - exp8).
  const int shift = 1 - exp8;               // how far below the min normal exponent
  if (shift > kManBits + 1) return sign;    // underflow to signed zero
  const std::uint32_t mant_with_implicit = (man23 | 0x800000u);  // 1.man in 24 bits
  const int total_drop = (23 - kManBits) + shift;
  std::uint32_t man = mant_with_implicit >> total_drop;
  const std::uint32_t rem = mant_with_implicit & ((1u << total_drop) - 1u);
  const std::uint32_t half = 1u << (total_drop - 1);
  if (rem > half || (rem == half && (man & 1u))) man++;  // RNE (may carry into exp=1, that's fine)
  return static_cast<std::uint8_t>(sign | (man & 0xFFu));
}
}  // namespace detail

// E4M3 (bias 7, 3 mantissa bits, no inf, NaN=0x7F, max normal 448, max-normal bits 0x7E).
HYLE_FORCE_INLINE float f8e4m3_to_f32(float8_e4m3_t v) noexcept {
  return detail::f8_to_f32<4, 3, 7, false>(v.bits);
}
HYLE_FORCE_INLINE float8_e4m3_t f32_to_f8e4m3(float f) noexcept {
  return float8_e4m3_t::from_bits(detail::f32_to_f8<4, 3, 7, false>(f, 448.0f, 0x7E, 0x7F));
}

// E5M2 (bias 15, 2 mantissa bits, IEEE-like inf/NaN, max normal 57344, max-normal bits 0x7B).
HYLE_FORCE_INLINE float f8e5m2_to_f32(float8_e5m2_t v) noexcept {
  return detail::f8_to_f32<5, 2, 15, true>(v.bits);
}
HYLE_FORCE_INLINE float8_e5m2_t f32_to_f8e5m2(float f) noexcept {
  return float8_e5m2_t::from_bits(detail::f32_to_f8<5, 2, 15, true>(f, 57344.0f, 0x7B, 0x7E));
}

}  // namespace hyle::simd
