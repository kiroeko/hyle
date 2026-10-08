// Original compressed-tensors0.18 symmetric4 unpack_from_int32/dequantize.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>
#include <mlx/ops.h>

namespace hyle::borrowed::compressed_tensors_apple {
namespace mx = mlx::core;
inline mx::array decode_symmetric4(const mx::array &packed, const mx::array &scale,
    int group, const mx::array &shifts, mx::Stream stream) {
  auto codes = mx::bitwise_and(mx::right_shift(
      mx::expand_dims(mx::astype(packed, mx::uint32, stream), -1, stream), shifts, stream),
      mx::array(std::uint32_t{15}), stream);
  auto signed_codes = mx::subtract(mx::astype(codes, mx::int32, stream), mx::array(8), stream);
  auto values = mx::astype(mx::reshape(signed_codes, {packed.shape(0), -1, group}, stream),
                           scale.dtype(), stream);
  return mx::reshape(mx::multiply(values, mx::expand_dims(scale, -1, stream), stream),
                      {packed.shape(0), -1}, stream);
}
} // namespace hyle::borrowed::compressed_tensors_apple
