// Original compressed-tensors0.18 four-bit unpack_from_int32/dequantize.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>
#include <mlx/ops.h>

namespace hyle::borrowed::compressed_tensors_apple {
namespace mx = mlx::core;
// View is not an MLX compile-fusable primitive. These same-width views share
// the original buffer, while retaining the typed arithmetic storage boundary
// before a later cast (including F16/BF16 -> F32). No payload copy is introduced.
inline mx::array typed_storage_boundary(const mx::array &value, mx::Stream stream) {
  const auto bits = value.dtype() == mx::float32 ? mx::uint32 : mx::uint16;
  return mx::view(mx::view(value, bits, stream), value.dtype(), stream);
}

inline mx::array decode_asymmetric4(const mx::array &packed,
    const mx::array &zero_points, const mx::array &scale, int group,
    const mx::array &shifts, mx::Stream stream) {
  // Codes are packed along K; zero points are independently packed along N.
  auto codes = mx::bitwise_and(mx::right_shift(
      mx::expand_dims(mx::astype(packed, mx::uint32, stream), -1, stream),
      shifts, stream), mx::array(std::uint32_t{15}), stream);
  auto zeros = mx::bitwise_and(mx::right_shift(
      mx::expand_dims(mx::astype(zero_points, mx::uint32, stream), 1, stream),
      mx::reshape(shifts, {1, 8, 1}, stream), stream),
      mx::array(std::uint32_t{15}), stream);
  zeros = mx::slice(mx::reshape(zeros, {-1, scale.shape(1)}, stream),
      {0, 0}, {packed.shape(0), scale.shape(1)}, stream);
  auto q = mx::astype(mx::subtract(mx::astype(codes, mx::int32, stream),
      mx::array(8), stream), scale.dtype(), stream);
  auto z = mx::astype(mx::subtract(mx::astype(zeros, mx::int32, stream),
      mx::array(8), stream), scale.dtype(), stream);
  auto difference = typed_storage_boundary(mx::subtract(
      mx::reshape(q, {packed.shape(0), -1, group}, stream),
      mx::expand_dims(z, -1, stream), stream), stream);
  auto product = typed_storage_boundary(mx::multiply(difference,
      mx::expand_dims(scale, -1, stream), stream), stream);
  return mx::reshape(product, {packed.shape(0), -1}, stream);
}

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
