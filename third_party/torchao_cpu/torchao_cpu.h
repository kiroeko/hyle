#pragma once

// Copyright (c) Meta Platforms, Inc. and affiliates.
// BSD-3-Clause; see LICENSE. CPU ATen spelling of the original TorchAO tensor
// sequences at a701b6a6058720c21f95908b7ae4a24bf0cae1b6. See NOTICE.md for
// symbol mapping. This is not a HYLE payload conversion/math kernel.
#include <ATen/ATen.h>

namespace torchao_cpu {

inline at::Tensor f32_to_e8m0_rceil(const at::Tensor &value) {
  auto f32 = value.to(at::kFloat);
  auto bits = f32.view(at::kInt);
  auto exponent = at::bitwise_and(at::bitwise_right_shift(bits, 23), 255);
  auto mantissa = at::bitwise_and(bits, 0x7fffff);
  auto up = at::where(exponent == 0, mantissa > 0x400000, mantissa != 0);
  return at::where(at::isfinite(f32), exponent + up.to(at::kInt), 255).to(at::kByte);
}

inline at::Tensor e8m0_scale_to_reciprocal_fp32(const at::Tensor &exponent) {
  auto biased = (254 - exponent.to(at::kInt)).to(at::kByte);
  auto bits = at::bitwise_left_shift(biased.to(at::kInt), 23);
  bits = at::where(biased == 0, 0x00400000, bits);
  bits = at::where(biased == 255, 0x7f800001, bits);
  return bits.view(at::kFloat);
}

// Specialization of _floatx_unpacked_to_f32 for its original ebits=2, mbits=1.
inline at::Tensor f4_unpacked_to_f32(const at::Tensor &x) {
  auto sign = at::bitwise_and(x, 8);
  auto positive = at::bitwise_xor(x, sign);
  auto zero = positive == 0;
  auto denormal = at::logical_and(positive > 0, at::bitwise_right_shift(positive, 1) == 0);
  auto exponent = at::bitwise_right_shift(positive, 1) - 1 + 127;
  auto exponent_f32 = at::bitwise_left_shift(exponent.to(at::kInt), 23);
  auto mantissa_f32 = at::bitwise_left_shift(at::bitwise_and(positive, 1).to(at::kInt), 22);
  auto result = at::bitwise_or(exponent_f32, mantissa_f32);
  result.index_put_({zero}, 0);
  result.index_put_({denormal}, 126 << 23);
  auto sign_f32 = at::bitwise_left_shift(sign.to(at::kInt), 28);
  return at::bitwise_or(result, sign_f32).view(at::kFloat);
}

inline at::Tensor unpack_uint4(const at::Tensor &packed) {
  auto first = at::bitwise_and(packed, 15).to(at::kByte);
  auto second = at::bitwise_right_shift(packed, 4).to(at::kByte);
  auto shape = packed.sizes().vec();
  shape.back() *= 2;
  return at::stack({first, second}, -1).view(shape);
}

// Original _choose_scale_float8 reduction/positive lower-bound path, and
// _quantize_affine_float8 / _RoundToFloat8.forward. ATen owns actual execution.
inline at::Tensor choose_scale_float8(const at::Tensor &groups, float floor) {
  return groups.to(at::kFloat).abs().amax(-1, true).clamp_min(floor) / 448.0;
}
inline at::Tensor quantize_affine_float8(const at::Tensor &groups,
                                       const at::Tensor &scale) {
  return (groups.to(at::kFloat) / scale).clamp(-448, 448).to(at::kFloat8_e4m3fn);
}

} // namespace torchao_cpu
