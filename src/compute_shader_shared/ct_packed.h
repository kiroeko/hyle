#pragma once

namespace hyle::compute::shader {

// Shared Metal/GLSL CT boundaries. Values are IEEE bits; each dialect supplies
// hyle_clz and the existing hyle_f32_mul. Tags: F32=0, F16=1, BF16=2.
// Narrowing is nearest-even, including gradual underflow and quiet NaNs.
inline constexpr char kCtPacked[] = R"HYLE_SHADER(
uint hyle_ct_narrow(uint bits, uint type) {
  if (type == 0u) return bits;
  if (type == 2u) {
    if ((bits & 0x7fffffffu) > 0x7f800000u)
      return (bits >> 16u) | 0x40u;
    return (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
  }
  uint sign = (bits >> 16u) & 0x8000u;
  uint exponent = (bits >> 23u) & 255u;
  uint fraction = bits & 0x7fffffu;
  if (exponent == 255u)
    return sign | 0x7c00u | (fraction != 0u ? (fraction >> 13u) | 0x200u : 0u);
  if (exponent > 142u) return sign | 0x7c00u;
  if (exponent >= 113u) {
    uint result = ((exponent - 112u) << 10u) | (fraction >> 13u);
    uint tail = fraction & 0x1fffu;
    return sign | (result + uint(tail > 0x1000u ||
        (tail == 0x1000u && (result & 1u) != 0u)));
  }
  if (exponent < 102u) return sign;
  uint shift = 126u - exponent;
  uint significand = fraction | 0x800000u;
  uint result = significand >> shift;
  uint tail = significand & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
  return sign | (result + uint(tail > halfway ||
      (tail == halfway && (result & 1u) != 0u)));
}

uint hyle_ct_widen(uint bits, uint type) {
  if (type == 0u) return bits;
  if (type == 2u) return bits << 16u;
  uint sign = (bits & 0x8000u) << 16u;
  uint exponent = (bits >> 10u) & 31u, fraction = bits & 0x3ffu;
  if (exponent == 31u) return sign | 0x7f800000u | (fraction << 13u);
  if (exponent != 0u) return sign | ((exponent + 112u) << 23u) | (fraction << 13u);
  if (fraction == 0u) return sign;
  uint top = 31u - hyle_clz(fraction);
  return sign | ((top + 103u) << 23u) | ((fraction << (23u - top)) & 0x7fffffu);
}

uint hyle_ct_round(uint bits, uint type) {
  return hyle_ct_widen(hyle_ct_narrow(bits, type), type);
}

uint hyle_ct_product(uint centered_bits, uint scale_bits, uint scale_type,
                     uint decode_type, uint operand_type) {
  uint source = hyle_ct_round(hyle_f32_mul(centered_bits, scale_bits), scale_type);
  return hyle_ct_round(hyle_ct_round(source, decode_type), operand_type);
}
)HYLE_SHADER";

} // namespace hyle::compute::shader
