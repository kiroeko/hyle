// Original ggml dequantize_row_q4_K/get_scale_min_k4 sequence on MLX GPU.
// Copyright (c) ggml authors. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <mlx/ops.h>
#include <cstdint>

namespace hyle::borrowed::ggml_q4k_apple {
namespace mx = mlx::core;
inline mx::array decode_rows(const mx::array &rows, int width, mx::Stream s) {
  const int count = rows.shape(0), blocks = width / 256;
  auto bytes = mx::reshape(rows, {count, blocks, 144}, s);
  const auto field = [&](int first, int last) {
    return mx::astype(mx::slice(bytes, {0,0,first}, {count,blocks,last}, s), mx::uint32, s);
  };
  const auto half = [&](int first) {
    auto lo = field(first, first+1), hi = field(first+1, first+2);
    auto word = mx::bitwise_or(lo, mx::left_shift(hi, mx::array(std::uint32_t{8}), s), s);
    return mx::astype(mx::view(mx::astype(word, mx::uint16, s), mx::float16, s), mx::float32, s);
  };
  auto scale_lo = field(4,8), min_lo = field(8,12), high = field(12,16);
  const mx::array mask63(std::uint32_t{63}), mask15(std::uint32_t{15});
  auto scale_hi = mx::bitwise_or(mx::bitwise_and(high, mask15, s),
      mx::left_shift(mx::right_shift(scale_lo, mx::array(std::uint32_t{6}), s),
                      mx::array(std::uint32_t{4}), s), s);
  auto min_hi = mx::bitwise_or(mx::right_shift(high, mx::array(std::uint32_t{4}), s),
      mx::left_shift(mx::right_shift(min_lo, mx::array(std::uint32_t{6}), s),
                      mx::array(std::uint32_t{4}), s), s);
  auto scales = mx::concatenate({mx::bitwise_and(scale_lo, mask63, s), scale_hi}, -1, s);
  auto mins = mx::concatenate({mx::bitwise_and(min_lo, mask63, s), min_hi}, -1, s);
  auto q = mx::reshape(field(16,144), {count,blocks,4,32}, s);
  auto codes = mx::reshape(mx::stack({mx::bitwise_and(q, mask15, s),
      mx::right_shift(q, mx::array(std::uint32_t{4}), s)}, -2, s), {count,blocks,8,32}, s);
  auto d = mx::multiply(half(0), mx::astype(scales, mx::float32, s), s);
  auto minimum = mx::multiply(half(2), mx::astype(mins, mx::float32, s), s);
  auto values = mx::subtract(mx::multiply(mx::expand_dims(d, -1, s),
      mx::astype(codes, mx::float32, s), s), mx::expand_dims(minimum, -1, s), s);
  return mx::reshape(values, {count,width}, s);
}
} // namespace hyle::borrowed::ggml_q4k_apple
