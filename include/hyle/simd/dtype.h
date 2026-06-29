#pragma once

#include <cstdint>
#include <cstring>

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
struct float16_t {
  std::uint16_t bits;
  static constexpr float16_t from_bits(std::uint16_t b) noexcept { return float16_t{b}; }
};
struct bfloat16_t {
  std::uint16_t bits;
  static constexpr bfloat16_t from_bits(std::uint16_t b) noexcept { return bfloat16_t{b}; }
};

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

}  // namespace hyle::simd
