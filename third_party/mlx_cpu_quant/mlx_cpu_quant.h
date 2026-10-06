// Original MLX quantized CPU equations expressed with original ATen operations.
// Copyright © 2023-2026 Apple Inc. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <ATen/ATen.h>

namespace hyle::borrowed::mlx_cpu_quant {
enum class Format { affine4, mxfp4, mxfp8 };

inline at::Tensor unpack(const at::Tensor &words, int bits) {
  auto shifts = at::arange(0, 32, bits, at::TensorOptions().dtype(at::kLong));
  auto codes = at::bitwise_and(
      at::bitwise_right_shift(words.to(at::kLong).unsqueeze(-1), shifts),
      (1 << bits) - 1);
  return codes.flatten(-2);
}
inline at::Tensor fp_values(const at::Tensor &codes, Format format) {
  if (format == Format::mxfp8)
    return codes.to(at::kByte).view(at::kFloat8_e4m3fn).to(at::kFloat);
  // Original FP4_LUT from backend/cpu/quantized.cpp and ops.cpp.
  const auto table = at::tensor({+0.0f,+0.5f,+1.0f,+1.5f,+2.0f,+3.0f,+4.0f,+6.0f,
                                -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f}, at::kFloat);
  return table.index_select(0, codes.reshape({-1})).reshape(codes.sizes());
}

// _qmm_t_simd/fp_qmm_t_simd: F32 accumulation, original group scaling order.
// The adapter checks expert ids. Slice K before index_select, so only the
// selected experts' current group is expanded, never the full float library.
inline at::Tensor matmul(const at::Tensor &activation, const at::Tensor &words,
    const at::Tensor &scales, const at::Tensor &biases, const at::Tensor &ids,
    Format format, std::int64_t group, at::ScalarType output_type) {
  const bool grouped = ids.defined();
  const bool affine = format == Format::affine4;
  const auto k = activation.size(-1), n = words.size(-2);
  const auto rows = activation.numel() / k;
  const auto bits = format == Format::mxfp8 ? 8 : 4;
  const auto pack = 32 / bits;
  auto a = activation.to(output_type).to(at::kFloat).reshape({rows, 1, k});
  auto sum = at::zeros({rows, 1, n}, at::kFloat);
  for (std::int64_t block = 0; block < k / group; ++block) {
    auto w = words.slice(-1, block * group / pack, (block + 1) * group / pack);
    auto scale = scales.select(-1, block);
    at::Tensor bias;
    if (affine) bias = biases.select(-1, block);
    if (grouped) {
      w = w.index_select(0, ids);
      scale = scale.index_select(0, ids);
      if (affine) bias = bias.index_select(0, ids);
    } else {
      w = w.unsqueeze(0);scale = scale.unsqueeze(0);
      if (affine) bias = bias.unsqueeze(0);
    }
    auto codes = unpack(w, bits);
    auto x = a.slice(-1, block * group, (block + 1) * group);
    if (affine) {
      auto values = codes.to(at::kFloat) * scale.to(output_type).to(at::kFloat).unsqueeze(-1);
      values = values + bias.to(output_type).to(at::kFloat).unsqueeze(-1);
      sum = sum + at::matmul(x, values.transpose(-1, -2));
    } else {
      auto values = fp_values(codes, format);
      auto dots = at::matmul(x, values.transpose(-1, -2));
      // Original dequantize_scale<T,32>: cast E8M0 to activation dtype first.
      // This preserves original F16 overflow/subnormal behavior.
      auto scale_values = at::pow(2.0, scale.to(at::kFloat) - 127)
          .to(output_type).to(at::kFloat);
      sum = sum + dots * scale_values.unsqueeze(1);
    }
  }
  auto shape = activation.sizes().vec();shape.back() = n;
  return sum.reshape(shape).to(output_type);
}

// QuantizedEmbedding::__call__: select rows, then original affine_dequantize.
inline at::Tensor embedding(const at::Tensor &words, const at::Tensor &scales,
    const at::Tensor &biases, const at::Tensor &ids, std::int64_t group) {
  auto w = unpack(words.index_select(0, ids), 4).to(scales.scalar_type());
  const auto rows = ids.numel(), k = w.size(-1);
  w = w.reshape({rows, k / group, group});
  auto decoded = w * scales.index_select(0, ids).unsqueeze(-1);
  decoded = decoded + biases.index_select(0, ids).unsqueeze(-1);
  return decoded.reshape({rows, k});
}
} // namespace hyle::borrowed::mlx_cpu_quant
