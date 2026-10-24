// Original MLX quantized CPU equations expressed with original ATen operations.
// Copyright © 2023-2026 Apple Inc. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <ATen/ATen.h>
#include <algorithm>

namespace hyle::borrowed::mlx_cpu_quant {
enum class Format { affine, mxfp4, mxfp8 };

inline at::Tensor unpack(const at::Tensor &words, int bits) {
  if (bits == 5 || bits == 6) {
    const auto count = (words.size(-1) / bits) * 32;
    auto bit = at::arange(count, at::TensorOptions().dtype(at::kLong)) * bits;
    auto index = at::floor_divide(bit, 32);
    auto shift = at::remainder(bit, 32);
    // U32 is carried by ATen I32 storage. Clear its sign extension before
    // joining words, otherwise an arithmetic shift fills the crossing code.
    auto source = at::bitwise_and(words.to(at::kLong), 0xffffffffLL);
    auto low = source.index_select(-1, index);
    auto high = source.index_select(-1, at::clamp_max(index + 1, words.size(-1) - 1));
    // Mask high first so signed-I64 shifts remain representable. Noncrossing
    // high bits disappear under the final mask, including the last code.
    auto joined = at::bitwise_or(at::bitwise_right_shift(low, shift),
        at::bitwise_left_shift(at::bitwise_and(high, (1 << bits) - 1), 32 - shift));
    return at::bitwise_and(joined, (1 << bits) - 1);
  }
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
    Format format, int bits, std::int64_t group, at::ScalarType output_type,
    bool weight_as_stored) {
  const bool grouped = ids.defined();
  const bool affine = format == Format::affine;
  if (weight_as_stored) {
    const auto depth = scales.size(-1) * group, reduction = words.size(-2);
    const auto group_words = (group / 32) * bits;
    const auto accumulate = [&](const at::Tensor &a, const at::Tensor &selected,
                                at::Tensor sum) {
      for (std::int64_t block = 0; block < depth / group; ++block) {
        auto words_group = words.slice(-1, block * group_words, (block + 1) * group_words);
        auto scale_group = scales.select(-1, block);
        auto bias_group = biases.select(-1, block);
        for (std::int64_t first = 0; first < reduction; first += 64) {
          const auto last = (std::min)(first + 64, reduction);
          // Slice both physical axes before selecting experts or unpacking:
          // only Ntile x Kgroup of the selected bank becomes floating point.
          auto w = words_group.slice(-2, first, last);
          auto scale = scale_group.slice(-1, first, last);
          auto bias = bias_group.slice(-1, first, last);
          if (selected.defined()) {
            w = w.index_select(0, selected);
            scale = scale.index_select(0, selected);
            bias = bias.index_select(0, selected);
          } else {
            w = w.unsqueeze(0);scale = scale.unsqueeze(0);bias = bias.unsqueeze(0);
          }
          auto values = unpack(w, bits).to(at::kFloat) *
              scale.to(output_type).to(at::kFloat).unsqueeze(-1);
          values = values + bias.to(output_type).to(at::kFloat).unsqueeze(-1);
          sum.slice(-1, block * group, (block + 1) * group).add_(
              at::matmul(a.slice(-1, first, last), values));
        }
      }
    };
    if (grouped && (activation.size(1) == 1 || activation.stride(1) == 0)) {
      const auto tokens = activation.size(0), slots = ids.numel() / tokens;
      auto a = activation.select(1, 0).to(output_type).to(at::kFloat).unsqueeze(1);
      auto selected = ids.reshape({tokens, slots});
      auto sum = at::zeros({tokens, slots, depth}, at::kFloat);
      for (std::int64_t slot = 0; slot < slots; ++slot)
        accumulate(a, selected.select(1, slot), sum.select(1, slot).unsqueeze(1));
      return sum.to(output_type);
    }
    const auto rows = activation.numel() / reduction;
    auto a = activation.to(output_type).to(at::kFloat).reshape({rows, 1, reduction});
    auto sum = at::zeros({rows, 1, depth}, at::kFloat);
    accumulate(a, ids, sum);
    auto shape = activation.sizes().vec();shape.back() = depth;
    return sum.reshape(shape).to(output_type);
  }
  const auto k = activation.size(-1), n = words.size(-2);
  const auto group_dot = [&](const at::Tensor &x, const at::Tensor &w,
                             const at::Tensor &scale, const at::Tensor &bias) {
    auto codes = unpack(w, bits);
    if (affine) {
      auto values = codes.to(at::kFloat) * scale.to(output_type).to(at::kFloat).unsqueeze(-1);
      values = values + bias.to(output_type).to(at::kFloat).unsqueeze(-1);
      return at::matmul(x, values.transpose(-1, -2));
    }
    auto values = fp_values(codes, format);
    auto dots = at::matmul(x, values.transpose(-1, -2));
    // Original dequantize_scale<T,32>: cast E8M0 to activation dtype first.
    // This preserves original F16 overflow/subnormal behavior.
    auto scale_values = at::pow(2.0, scale.to(at::kFloat) - 127)
        .to(output_type).to(at::kFloat);
    return dots * scale_values.unsqueeze(1);
  };
  if (grouped && (activation.size(1) == 1 || activation.stride(1) == 0)) {
    const auto tokens = activation.size(0), slots = ids.numel() / tokens;
    // Collapse the shared axis before either conversion. Keeping the T batch
    // and iterating selected slots avoids matmul flattening a broadcast A into
    // a T*S*K copy; each dot reads exactly the same original token row.
    auto a = activation.select(1, 0).to(output_type).to(at::kFloat).unsqueeze(1);
    auto selected = ids.reshape({tokens, slots});
    auto sum = at::zeros({tokens, slots, n}, at::kFloat);
    for (std::int64_t block = 0; block < k / group; ++block) {
      const auto group_words = (group / 32) * bits;
      auto words_group = words.slice(-1, block * group_words, (block + 1) * group_words);
      auto scale_group = scales.select(-1, block);
      at::Tensor bias_group;
      if (affine) bias_group = biases.select(-1, block);
      auto x = a.slice(-1, block * group, (block + 1) * group);
      for (std::int64_t slot = 0; slot < slots; ++slot) {
        auto expert = selected.select(1, slot);
        auto w = words_group.index_select(0, expert);
        auto scale = scale_group.index_select(0, expert);
        auto bias = affine ? bias_group.index_select(0, expert) : at::Tensor{};
        sum.select(1, slot).add_(group_dot(x, w, scale, bias).squeeze(1));
      }
    }
    return sum.to(output_type);
  }
  const auto rows = activation.numel() / k;
  auto a = activation.to(output_type).to(at::kFloat).reshape({rows, 1, k});
  auto sum = at::zeros({rows, 1, n}, at::kFloat);
  for (std::int64_t block = 0; block < k / group; ++block) {
    const auto group_words = (group / 32) * bits;
    auto w = words.slice(-1, block * group_words, (block + 1) * group_words);
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
    auto x = a.slice(-1, block * group, (block + 1) * group);
    sum = sum + group_dot(x, w, scale, bias);
  }
  auto shape = activation.sizes().vec();shape.back() = n;
  return sum.reshape(shape).to(output_type);
}

// QuantizedEmbedding::__call__: select rows, then original affine_dequantize.
inline at::Tensor embedding(const at::Tensor &words, const at::Tensor &scales,
    const at::Tensor &biases, const at::Tensor &ids, int bits, std::int64_t group) {
  auto w = unpack(words.index_select(0, ids), bits).to(scales.scalar_type());
  const auto rows = ids.numel(), k = w.size(-1);
  w = w.reshape({rows, k / group, group});
  auto decoded = w * scales.index_select(0, ids).unsqueeze(-1);
  decoded = decoded + biases.index_select(0, ids).unsqueeze(-1);
  return decoded.reshape({rows, k});
}
} // namespace hyle::borrowed::mlx_cpu_quant
