#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
#include "cpu_c5_parameter_fragments.h"

// Actual text-only qwen35moe GGUF post-projection region. All three MRoPE
// position axes coincide; rotate the first64 values by halves, leaving192.
namespace cpu_c5_qwen_attention {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace ds = d5_deepseek_official_attention;
using sem::Fragment;
using sem::bytes;
using sem::dense;
inline constexpr std::int64_t heads = 64, kv_heads = 4, depth = 256;
inline std::vector<float> rotate(std::span<const float> input, std::int32_t position) {
  std::vector<float> result(input.begin(), input.end());
  for (std::size_t row = 0; row < input.size() / depth; ++row)
    for (std::size_t pair = 0; pair < 32; ++pair) {
      const auto first = row * depth + pair, second = first + 32;
      const float angle = float(position) / std::pow(10000000.0F, float(pair) / 32);
      result[first] = input[first] * std::cos(angle) - input[second] * std::sin(angle);
      result[second] = input[first] * std::sin(angle) + input[second] * std::cos(angle);
    }
  return result;
}
inline std::array<std::vector<float>, 2> attention_reference(
    std::span<const float> q, std::span<const float> k,
    std::span<const float> v, std::span<const float> gate, bool half_products) {
  const auto rounded = [&](double value) {
    return half_products ? half_test::decode(half_test::encode<hyle::simd::float16_t>(value)) : value;
  };
  std::array<std::vector<float>, 2> result{
      std::vector<float>(heads * depth), std::vector<float>(heads * depth)};
  for (std::int64_t head = 0; head < heads; ++head) {
    const auto kh = head / (heads / kv_heads);
    std::array<double, 3> score{};
    for (int token = 0; token < 3; ++token)
      for (std::int64_t d = 0; d < depth; ++d)
        score[token] += rounded(q[head * depth + d]) *
                        rounded(k[(token * kv_heads + kh) * depth + d]) / 16;
    const auto maximum = *std::max_element(score.begin(), score.end());
    double sum = 0;
    for (auto &value : score) sum += value = std::exp(value - maximum);
    for (std::int64_t d = 0; d < depth; ++d) {
      double value = 0;
      for (int token = 0; token < 3; ++token)
        value += rounded(score[token] / sum) * rounded(v[(token * kv_heads + kh) * depth + d]);
      const auto index = head * depth + d;
      result[0][index] = float(value);
      result[1][index] = float(value / (1 + std::exp(-double(gate[index]))));
    }
  }
  return result;
}
inline Fragment make(const c::ModelArtifactProfileDescription *identity) {
  Fragment f{.profile = identity, .region = 0x040c0501};
  auto query = f.input(dense({1, 1, heads, depth}));
  auto key = f.input(dense({1, 1, kv_heads, depth}));
  auto value = f.input(dense({1, 1, kv_heads, depth}));
  auto gate = f.input(dense({1, 1, heads, depth}));
  auto past_k = f.input(dense({1, 2, kv_heads, depth}));
  auto past_v = f.input(dense({1, 2, kv_heads, depth}));
  auto position = f.input(dense({1}, c::DataType::kInt32));
  const auto scale = sem::oracle::scale(depth);
  auto weight = f.constant(dense({depth}), bytes(scale));
  auto nq = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}), {query, weight});
  auto nk = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}), {key, weight});
  std::vector<float> frequencies(32);
  for (std::size_t pair = 0; pair < frequencies.size(); ++pair)
    frequencies[pair] = 1.0F / std::pow(10000000.0F, float(pair) / 32);
  auto freq = f.constant(dense({32}), bytes(frequencies));
  const c::Operation rope(c::RoPE{.rotary_dimensions = 64, .sequence_axis = 1,
                                .layout = c::RopeLayout::kHalfSplit});
  auto rq = f.one(rope, {nq, position, freq});
  auto rk = f.one(rope, {nk, position, freq});
  auto keys = ds::concat(f, past_k, rk, 1), values = ds::concat(f, past_v, value, 1);
  auto attended = f.one(c::Operation(c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchSequenceHeadDepth, .scale = 0.0625F,
      .causal = true, .accumulation_type = c::DataType::kFloat32}), {rq, keys, values});
  auto probability = core::unary(f, c::UnaryKind::kSigmoid, gate);
  auto final = core::binary(f, c::BinaryKind::kMultiply, attended, probability);
  f.outputs = {ds::slice(f, keys, 1, 1, 2), ds::slice(f, values, 1, 1, 2), nq, nk, rq, rk, attended, final};
  f.state_ports = {{4, 0}, {5, 1}};
  f.exact_outputs = {1}; // V is a copy; rotated K retains the numeric gate.
  auto history_k = d5_gguf_test::values(2 * kv_heads, depth, 5);
  auto history_v = d5_gguf_test::values(2 * kv_heads, depth, 13);
  for (int step = 0; step < 4; ++step) {
    auto q = d5_gguf_test::values(heads, depth, 3 + step * 7);
    auto k = d5_gguf_test::values(kv_heads, depth, 11 + step * 13);
    auto v = d5_gguf_test::values(kv_heads, depth, 17 + step * 11);
    auto g = d5_gguf_test::values(heads, depth, 23 + step * 17);
    const std::array<std::int32_t, 1> positions{37 + step};
    auto normalized_q = sem::oracle::rms_norm_rows(q, scale, depth, 1.0e-6F);
    auto normalized_k = sem::oracle::rms_norm_rows(k, scale, depth, 1.0e-6F);
    auto rotated_q = rotate(normalized_q, positions[0]), rotated_k = rotate(normalized_k, positions[0]);
    auto all_k = history_k, all_v = history_v;
    all_k.insert(all_k.end(), rotated_k.begin(), rotated_k.end());
    all_v.insert(all_v.end(), v.begin(), v.end());
    auto reference = attention_reference(rotated_q, all_k, all_v, g, false);
    auto native = attention_reference(rotated_q, all_k, all_v, g, true);
    auto next_k = std::vector<float>(all_k.begin() + kv_heads * depth, all_k.end());
    auto next_v = std::vector<float>(all_v.begin() + kv_heads * depth, all_v.end());
    f.replays.push_back({{bytes(q), bytes(k), bytes(v), bytes(g), bytes(history_k), bytes(history_v), bytes<std::int32_t>(positions)},
        {next_k, next_v, normalized_q, normalized_k, rotated_q, rotated_k, reference[0], reference[1]},
        {{}, {}, {}, {}, {}, {}, native[0], native[1]}});
    history_k = std::move(next_k); history_v = std::move(next_v);
  }
  return f;
}
} // namespace cpu_c5_qwen_attention
#endif
