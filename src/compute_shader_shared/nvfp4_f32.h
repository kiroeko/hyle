#pragma once

namespace hyle::compute::shader {

// Private numerical source shared by the Vulkan and Metal NVFP4 reference
// kernels. Inputs/results are IEEE binary32 bits. Each dialect supplies only
// hyle_mul_hi(uint,uint) and hyle_clz(uint); no native division, float64,
// denormal capability or fast-math assumption is needed for these boundaries.
inline constexpr char kNvfp4F32[] = R"HYLE_SHADER(
uint hyle_shift_jam(uint value, uint distance) {
  if (distance == 0u) return value;
  if (distance >= 32u) return uint(value != 0u);
  return (value >> distance) | uint((value << (32u-distance)) != 0u);
}

uint hyle_f32_pack(uint sign, int exponent, uint significand) {
  if (significand == 0u) return sign;
  if (significand >= 0x08000000u) {
    significand = hyle_shift_jam(significand,1u);
    ++exponent;
  }
  while (significand < 0x04000000u && exponent > 1) {
    significand <<= 1u;
    --exponent;
  }
  if (exponent < 1) {
    significand = hyle_shift_jam(significand,uint(1-exponent));
    exponent = 1;
  }
  uint tail = significand & 7u;
  uint rounded = (significand >> 3u) +
      uint(tail > 4u || (tail == 4u && (significand & 8u) != 0u));
  if (rounded >= 0x01000000u) {
    rounded >>= 1u;
    ++exponent;
  }
  if (exponent >= 255) return sign | 0x7f800000u;
  uint field = exponent == 1 && rounded < 0x00800000u ? 0u : uint(exponent);
  return sign | (field << 23u) | (rounded & 0x007fffffu);
}

struct HyleF32Parts { uint significand; int exponent; };
HyleF32Parts hyle_f32_parts(uint bits) {
  HyleF32Parts value;
  value.significand = bits & 0x007fffffu;
  value.exponent = int((bits >> 23u) & 255u);
  if (value.exponent != 0) value.significand |= 0x00800000u;
  else {
    uint shift = hyle_clz(value.significand)-8u;
    value.significand <<= shift;
    value.exponent = 1-int(shift);
  }
  return value;
}

uint hyle_f32_nan(uint a, uint b) {
  if ((a & 0x7fffffffu) > 0x7f800000u) return a | 0x00400000u;
  if ((b & 0x7fffffffu) > 0x7f800000u) return b | 0x00400000u;
  return 0x7fc00000u;
}

uint hyle_f32_mul(uint a, uint b) {
  uint sign = (a ^ b) & 0x80000000u;
  uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
  if (aa >= 0x7f800000u || bb >= 0x7f800000u) {
    if (aa > 0x7f800000u || bb > 0x7f800000u || aa == 0u || bb == 0u)
      return hyle_f32_nan(a,b);
    return sign | 0x7f800000u;
  }
  if (aa == 0u || bb == 0u) return sign;
  HyleF32Parts left = hyle_f32_parts(a), right = hyle_f32_parts(b);
  uint low = left.significand * right.significand;
  uint high = hyle_mul_hi(left.significand,right.significand);
  uint extended = (high << 12u) | (low >> 20u) | uint((low & 0x000fffffu) != 0u);
  return hyle_f32_pack(sign,left.exponent+right.exponent-127,extended);
}

uint hyle_f32_div(uint a, uint b) {
  uint sign = (a ^ b) & 0x80000000u;
  uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
  if (aa > 0x7f800000u || bb > 0x7f800000u ||
      (aa == 0x7f800000u && bb == 0x7f800000u) || (aa == 0u && bb == 0u))
    return hyle_f32_nan(a,b);
  if (aa == 0x7f800000u || bb == 0u) return sign | 0x7f800000u;
  if (aa == 0u || bb == 0x7f800000u) return sign;
  HyleF32Parts left = hyle_f32_parts(a), right = hyle_f32_parts(b);
  int exponent = left.exponent-right.exponent+127;
  uint remainder = left.significand, quotient = 0u;
  if (remainder < right.significand) {
    remainder <<= 1u;
    --exponent;
  }
  for (uint bit = 0u; bit < 27u; ++bit) {
    uint next = uint(remainder >= right.significand);
    if (next != 0u) remainder -= right.significand;
    quotient = (quotient << 1u) | next;
    if (bit != 26u) remainder <<= 1u;
  }
  quotient |= uint(remainder != 0u);
  return hyle_f32_pack(sign,exponent,quotient);
}

uint hyle_f32_add(uint a, uint b) {
  uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
  if (aa >= 0x7f800000u || bb >= 0x7f800000u) {
    if (aa > 0x7f800000u || bb > 0x7f800000u ||
        (aa == 0x7f800000u && bb == 0x7f800000u && ((a ^ b) & 0x80000000u) != 0u))
      return hyle_f32_nan(a,b);
    return aa == 0x7f800000u ? a : b;
  }
  if (aa == 0u && bb == 0u) return (a & b) & 0x80000000u;
  if (aa == 0u) return b;
  if (bb == 0u) return a;
  // Larger magnitude first keeps subtraction nonnegative, including zeros.
  if (aa < bb) { uint temporary=a; a=b; b=temporary; }
  uint ea = (a >> 23u) & 255u, eb = (b >> 23u) & 255u;
  uint left = (a & 0x007fffffu) | (ea == 0u ? 0u : 0x00800000u);
  uint right = (b & 0x007fffffu) | (eb == 0u ? 0u : 0x00800000u);
  ea = ea == 0u ? 1u : ea;
  eb = eb == 0u ? 1u : eb;
  left <<= 3u;
  right = hyle_shift_jam(right << 3u,ea-eb);
  uint result = ((a ^ b) & 0x80000000u) == 0u ? left+right : left-right;
  // Exact cancellation under round-to-nearest produces positive zero.
  if (result == 0u) return 0u;
  return hyle_f32_pack(a & 0x80000000u,int(ea),result);
}
)HYLE_SHADER";

} // namespace hyle::compute::shader
