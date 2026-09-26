#pragma once

#include "d5_moe_execution_fragments.h"

// Shared real-shape consumers and independent host oracles, never product math.
namespace d5_deepseek_remaining_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace moe = d5_moe_execution_fragments;
using sem::bytes;
using sem::dense;
using sem::Fragment;

inline std::vector<float> sink_reference(std::span<const float> query,
                                          std::span<const float> key,
                                          std::span<const float> value,
                                          std::span<const float> mask,
                                          std::span<const float> sinks,
                                          std::int64_t tokens, std::int64_t keys,
                                          float scale) {
  const auto heads = std::int64_t(sinks.size());
  constexpr std::int64_t depth = 512;
  std::vector<float> result(tokens * heads * depth);
  std::vector<double> scores(keys), probabilities(keys);
  for (std::int64_t token = 0; token < tokens; ++token)
    for (std::int64_t head = 0; head < heads; ++head) {
      const auto offset = (token * heads + head) * depth;
      double maximum = sinks[head];
      for (std::int64_t row = 0; row < keys; ++row) {
        double score = 0;
        for (std::int64_t d = 0; d < depth; ++d)
          score += double(query[offset + d]) * key[row * depth + d];
        scores[row] = score * scale + mask[token * keys + row];
        maximum = std::max(maximum, scores[row]);
      }
      double denominator = std::exp(double(sinks[head]) - maximum);
      for (std::int64_t row = 0; row < keys; ++row) {
        probabilities[row] = std::exp(scores[row] - maximum);
        denominator += probabilities[row];
      }
      for (std::int64_t d = 0; d < depth; ++d) {
        double output = 0;
        for (std::int64_t row = 0; row < keys; ++row)
          output += probabilities[row] * value[row * depth + d];
        result[offset + d] = float(output / denominator);
      }
    }
  return result;
}

inline Fragment sink_attention(const moe::Case &model, int mode = 0,
                                  std::int64_t tokens = 3, std::int64_t keys = 17) {
  const std::int64_t heads = model.experts == 384 ? 128 : 64;
  constexpr std::int64_t depth = 512;
  const float scale = 1.0F / std::sqrt(float(depth));
  Fragment f{.profile = model.profile, .region = 0x08093301};
  auto query = f.input(dense({1, tokens, heads, depth}));
  auto key = f.input(dense({1, keys, 1, depth}));
  auto value = f.input(dense({1, keys, 1, depth}));
  auto mask = f.input(dense({tokens, keys}));
  auto sink = f.input(dense({heads}));
  f.outputs = {f.one(c::model_ops::sink_attention(scale), {query, key, value, mask, sink})};
  if (mode == 1) f.exact_outputs = {0};
  for (int replay = 0; replay < 3; ++replay) {
    std::vector<float> q(tokens * heads * depth), k(keys * depth), v(keys * depth);
    std::vector<float> masks(tokens * keys), sinks(heads);
    for (std::size_t i = 0; i < q.size(); ++i)
      q[i] = float(std::int64_t((i * 13 + replay * 7) % 43) - 21) / 32.0F;
    for (std::size_t i = 0; i < k.size(); ++i) {
      k[i] = float(std::int64_t((i * 7 + replay * 11) % 37) - 18) / 32.0F;
      v[i] = 0.25F + float(std::int64_t((i * 11 + replay * 3) % 29) - 14) / 32.0F;
    }
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t row = 0; row < keys; ++row)
        masks[token * keys + row] = mode == 1 ||
                  row > keys - tokens + token || (row + replay + token) % 5 == 0
              ? -std::numeric_limits<float>::infinity()
              : float((row + replay) % 3 - 1) / 8.0F;
    for (std::int64_t head = 0; head < heads; ++head)
      sinks[head] = (mode == 2 ? 80.0F : 1.25F) + float(head % 7) / 8.0F + replay / 4.0F;
    auto expected = sink_reference(q, k, v, masks, sinks, tokens, keys, scale);
    f.replays.push_back({{bytes(q), bytes(k), bytes(v), bytes(masks), bytes(sinks)},
                          {std::move(expected)}});
  }
  return f;
}
} // namespace d5_deepseek_remaining_fragments
