// Original bitsandbytes default tensor sequence, expressed with MLX GPU ops.
// Copyright (c) Facebook, Inc. and its affiliates. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <cstdint>
#include <utility>
#include <mlx/ops.h>

namespace hyle::borrowed::bitsandbytes_apple {
namespace mx = mlx::core;
inline mx::array nested_scales(const mx::array &codes, const mx::array &absmax,
    const mx::array &codebook, const mx::array &offset, mx::Stream stream) {
  auto flat = mx::reshape(codes, {-1}, stream);
  auto values = mx::take(codebook, mx::astype(flat, mx::uint32, stream), 0, stream);
  auto scale = mx::slice(mx::repeat(absmax, 256, 0, stream), {0},
                         {static_cast<int>(flat.size())}, stream);
  return mx::add(mx::multiply(values, scale, stream),
                 mx::reshape(offset, {}, stream), stream);
}
inline mx::array decode_nf4(const mx::array &packed, const mx::array &absmax,
    const mx::array &codebook, int group, mx::Shape shape, mx::Dtype dtype,
    mx::Stream stream) {
  auto flat = mx::astype(mx::reshape(packed, {-1}, stream), mx::uint32, stream);
  auto high = mx::right_shift(flat, mx::array(std::uint32_t{4}), stream);
  auto low = mx::bitwise_and(flat, mx::array(std::uint32_t{15}), stream);
  auto codes = mx::reshape(mx::stack({high, low}, 1, stream), {-1}, stream);
  auto values = mx::take(codebook, codes, 0, stream);
  auto decoded = mx::multiply(mx::reshape(values, {-1, group}, stream),
                              mx::reshape(absmax, {-1, 1}, stream), stream);
  return mx::astype(mx::reshape(decoded, std::move(shape), stream), dtype, stream);
}
} // namespace hyle::borrowed::bitsandbytes_apple
