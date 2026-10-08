// Original AutoAWQ unpack_awq/reverse_awq_order/dequantize_gemm tensor sequence.
// Copyright (c) 2023 MIT HAN Lab. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <cstdint>
#include <mlx/ops.h>

namespace hyle::borrowed::awq_apple {
namespace mx = mlx::core;
inline mx::array unpack(const mx::array &words, const mx::array &shifts,
                       const mx::array &order, mx::Stream stream) {
  auto codes = mx::right_shift(
      mx::expand_dims(mx::astype(words, mx::uint32, stream), -1, stream), shifts, stream);
  return mx::astype(mx::reshape(mx::bitwise_and(mx::take(codes, order, -1, stream),
      mx::array(std::uint32_t{15}), stream), {words.shape(0), -1}, stream), mx::int8, stream);
}
inline mx::array decode_gemm4(const mx::array &weight, const mx::array &zeros,
    const mx::array &scales, int group, const mx::array &shifts,
    const mx::array &order, mx::Stream stream) {
  auto values = unpack(weight, shifts, order, stream);
  auto zero = mx::repeat(unpack(zeros, shifts, order, stream), group, 0, stream);
  auto difference = mx::astype(mx::subtract(values, zero, stream), scales.dtype(), stream);
  return mx::transpose(mx::multiply(difference, mx::repeat(scales, group, 0, stream),
                                    stream), {1, 0}, stream);
}
} // namespace hyle::borrowed::awq_apple
