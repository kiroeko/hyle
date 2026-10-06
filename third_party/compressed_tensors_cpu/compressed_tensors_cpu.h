// Original compressed-tensors0.18 symmetric4 unpack_from_int32/dequantize.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <ATen/ATen.h>

namespace hyle::borrowed::compressed_tensors_cpu {
inline at::Tensor decode_symmetric4(const at::Tensor &packed,
    const at::Tensor &scale, std::int64_t group) {
  auto shifts = at::arange(0, 32, 4, at::TensorOptions().dtype(at::kLong));
  auto codes = at::bitwise_and(
      at::bitwise_right_shift(packed.to(at::kLong).unsqueeze(-1), shifts), 15);
  // Four-bit dense packing has no cross-word/padding remainder. The original
  // unsigned offset is8; original dequantization casts to scale dtype first.
  auto signed_codes = (codes - 8).to(at::kChar)
      .reshape({packed.size(0), -1, group}).to(scale.scalar_type());
  return (signed_codes * scale.unsqueeze(-1)).flatten(-2);
}
} // namespace hyle::borrowed::compressed_tensors_cpu
