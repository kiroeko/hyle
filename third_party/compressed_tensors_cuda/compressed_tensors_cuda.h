// Original compressed-tensors0.18 int4 unpack_from_int32/dequantize.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <ATen/ATen.h>

namespace hyle::borrowed::compressed_tensors_cuda {
inline at::Tensor decode_symmetric4(const at::Tensor &packed,
    const at::Tensor &scale, std::int64_t group, const at::Tensor &shifts) {
  auto codes = at::bitwise_and(
      at::bitwise_right_shift(packed.to(at::kLong).unsqueeze(-1), shifts), 15);
  // Four-bit dense packing has no cross-word/padding remainder. The original
  // unsigned offset is8; original dequantization casts to scale dtype first.
  auto signed_codes = (codes - 8).to(at::kChar)
      .reshape({packed.size(0), -1, group}).to(scale.scalar_type());
  return (signed_codes * scale.unsqueeze(-1)).flatten(-2);
}

inline at::Tensor decode_asymmetric4(const at::Tensor &packed,
    const at::Tensor &packed_zero, const at::Tensor &scale,
    std::int64_t group, const at::Tensor &shifts) {
  auto codes = at::bitwise_and(
      at::bitwise_right_shift(packed.to(at::kLong).unsqueeze(-1), shifts), 15);
  auto signed_codes = (codes - 8).to(at::kChar)
      .reshape({packed.size(0), -1, group}).to(scale.scalar_type());
  // Original zero points use packed_dim=0, unlike the K-packed codes.
  // Unpack along N, then discard only the padded output rows.
  auto zeros = at::bitwise_and(at::bitwise_right_shift(
      packed_zero.transpose(0, 1).to(at::kLong).unsqueeze(-1), shifts), 15);
  auto signed_zero = (zeros - 8).to(at::kChar)
      .reshape({scale.size(1), -1}).slice(1, 0, packed.size(0))
      .transpose(0, 1).to(scale.scalar_type());
  // Keep the original two typed operations. An explicit F32 output conversion
  // belongs after the materialized scale-dtype product, not before it.
  auto centered = signed_codes - signed_zero.unsqueeze(-1);
  return (centered * scale.unsqueeze(-1)).flatten(-2);
}
} // namespace hyle::borrowed::compressed_tensors_cuda
