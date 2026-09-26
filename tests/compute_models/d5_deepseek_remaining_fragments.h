#pragma once

#include "d5_moe_execution_fragments.h"
#include "d5_kda_full_fragments.h"

// Shared real-shape consumers and independent host oracles, never product math.
namespace d5_deepseek_remaining_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace moe = d5_moe_execution_fragments;
namespace core = d5_moe_semantic_fragments;
namespace linear = d5_kda_full_fragments;
using sem::bytes;
using sem::dense;
using sem::Fragment;
using sem::ValueId;
using core::reshape;
using core::binary;

inline ValueId slice(Fragment &f, ValueId input, std::int32_t axis,
                      std::int64_t first, std::int64_t count) {
  auto view = f.one(c::Operation(c::View::slice(axis, first, count).value()), {input});
  return f.one(c::Operation(c::Copy{}), {view});
}
inline ValueId permute(Fragment &f, ValueId input,
                        std::initializer_list<std::int32_t> axes) {
  auto view = f.one(c::Operation(c::View::permute({axes.begin(), axes.size()}).value()), {input});
  return f.one(c::Operation(c::Copy{}), {view});
}
inline ValueId concat(Fragment &f, ValueId a, ValueId b, std::int32_t axis) {
  return f.one(c::Operation(c::Concat{.axis = axis}), {a, b});
}

// Fixed DeepSeek metadata: tail64, base10000 (raw) /160000 (compressed),
// YaRN factor16/original65536/beta32/1. Its attention magnitude correction
// cancels YaRN's multiplier. These are cold consumer parameters, not kernels.
inline std::vector<float> rope_frequencies(bool compressed, bool inverse = false) {
  const float base = compressed ? 160000.0F : 10000.0F;
  const auto correction = [&](float rotations) {
    return 64.0F * std::log(65536.0F / (rotations * 2.0F * float(std::acos(-1.0)))) /
           (2.0F * std::log(base));
  };
  const float low = std::max(0.0F, std::floor(correction(32)));
  const float high = std::min(63.0F, std::ceil(correction(1)));
  const float theta_scale = std::pow(base, -2.0F / 64.0F);
  float frequency = 1.0F;
  std::vector<float> result(32);
  for (std::size_t i = 0; i < result.size(); ++i) {
    const float mix = 1.0F - std::clamp((float(i) - low) / std::max(0.001F, high - low), 0.0F, 1.0F);
    result[i] = frequency * (compressed ? mix + (1.0F - mix) / 16.0F : 1.0F);
    if (inverse) result[i] = -result[i];
    frequency *= theta_scale;
  }
  return result;
}
inline ValueId tail_rope(Fragment &f, ValueId input, ValueId positions,
                          bool compressed, bool inverse = false) {
  const auto depth = f.description(input).dimensions().back();
  auto prefix = slice(f, input, 3, 0, depth - 64);
  auto tail = slice(f, input, 3, depth - 64, 64);
  auto frequencies = f.constant(dense({32}), bytes(rope_frequencies(compressed, inverse)));
  auto rotated = f.one(c::Operation(c::RoPE{.rotary_dimensions = 64,
      .sequence_axis = 1, .layout = c::RopeLayout::kInterleaved}), {tail, positions, frequencies});
  return concat(f, prefix, rotated, 3);
}
inline std::vector<float> rope_reference(std::span<const float> input,
                                          std::int64_t depth, std::int32_t position,
                                          bool compressed, bool inverse = false) {
  auto frequencies = rope_frequencies(compressed, inverse);
  std::vector<float> output(input.begin(), input.end());
  for (std::size_t row = 0; row < input.size() / depth; ++row)
    for (std::size_t pair = 0; pair < 32; ++pair) {
      const double angle = float(position) * frequencies[pair];
      const auto offset = row * depth + depth - 64 + 2 * pair;
      const double a = input[offset], b = input[offset + 1];
      output[offset] = float(a * std::cos(angle) - b * std::sin(angle));
      output[offset + 1] = float(a * std::sin(angle) + b * std::cos(angle));
    }
  return output;
}

inline std::vector<float> project_rows(const linear::Projection &projection,
                                        std::span<const float> input, std::int64_t width) {
  std::vector<float> result;
  for (std::size_t offset = 0; offset < input.size(); offset += width) {
    auto row = projection.reference(input.subspan(offset, width));
    result.insert(result.end(), row.begin(), row.end());
  }
  return result;
}

// The model compresses each channel independently over time. Ratio4 selects
// the previous projection's first half and the current projection's second
// half; HCA uses all 128 single-width rows. Never softmax over channels.
inline ValueId compress(Fragment &f, ValueId previous_kv, ValueId previous_score,
                         ValueId current_kv, ValueId current_score,
                         ValueId norm, ValueId position, std::int64_t ratio,
                         std::int64_t depth, bool indexer) {
  if (ratio == 4) {
    previous_kv = slice(f, previous_kv, 1, 0, depth);
    previous_score = slice(f, previous_score, 1, 0, depth);
    current_kv = slice(f, current_kv, 1, depth, depth);
    current_score = slice(f, current_score, 1, depth, depth);
  }
  auto values = permute(f, concat(f, previous_kv, current_kv, 0), {1, 0});
  auto scores = permute(f, concat(f, previous_score, current_score, 0), {1, 0});
  auto weights = f.one(c::Operation(c::Softmax{.axis = -1}), {scores});
  auto weighted = binary(f, c::BinaryKind::kMultiply, values, weights);
  auto reduced = reshape(f, moe::sum(f, weighted, -1), {1, depth});
  auto normalized = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                           {reduced, norm});
  auto rotated = tail_rope(f, reshape(f, normalized, {1, 1, 1, depth}),
                           position, true);
  auto result = reshape(f, rotated, {1, depth});
  return indexer ? f.one(c::Operation(c::Hadamard{}), {result}) : result;
}

inline std::vector<float> compress_reference(
    std::span<const float> previous_kv, std::span<const float> previous_score,
    std::span<const float> current_kv, std::span<const float> current_score,
    std::span<const float> norm, std::int64_t ratio, std::int32_t position,
    bool indexer) {
  const auto depth = std::int64_t(norm.size());
  const auto channels = ratio == 4 ? 2 * depth : depth;
  const auto rows = ratio == 4 ? 4 : 64;
  std::vector<float> reduced(depth);
  for (std::int64_t channel = 0; channel < depth; ++channel) {
    const auto current_channel = channel + (ratio == 4 ? depth : 0);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t row = 0; row < rows; ++row) {
      maximum = std::max(maximum, double(previous_score[row * channels + channel]));
      maximum = std::max(maximum, double(current_score[row * channels + current_channel]));
    }
    double numerator = 0, denominator = 0;
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto old_index = row * channels + channel;
      const auto new_index = row * channels + current_channel;
      const double a = std::exp(double(previous_score[old_index]) - maximum);
      const double b = std::exp(double(current_score[new_index]) - maximum);
      numerator += a * previous_kv[old_index] + b * current_kv[new_index];
      denominator += a + b;
    }
    reduced[channel] = float(numerator / denominator);
  }
  double squares = 0;
  for (float value : reduced) squares += double(value) * value;
  const auto inverse_rms = 1.0 / std::sqrt(squares / double(depth) + 1.0e-6);
  for (std::int64_t channel = 0; channel < depth; ++channel)
    reduced[channel] = float(reduced[channel] * inverse_rms * norm[channel]);
  auto result = rope_reference(reduced, depth, position, true);
  if (indexer) {
    // Dense signed-matrix reference, independent of the mature butterfly.
    std::vector<float> transformed(depth);
    for (std::int64_t row = 0; row < depth; ++row) {
      double sum = 0;
      for (std::int64_t column = 0; column < depth; ++column)
        sum += (std::popcount(std::uint64_t(row & column)) % 2 ? -1 : 1) *
               double(result[column]);
      transformed[row] = float(sum / std::sqrt(double(depth)));
    }
    result = std::move(transformed);
  }
  return result;
}

inline Fragment compressor(const moe::Case &model, std::int64_t ratio,
                             bool indexer = false) {
  if ((ratio != 4 && ratio != 128) || (indexer && ratio != 4))
    throw std::runtime_error("invalid DeepSeek compressor role");
  using Storage = linear::Storage;
  const auto storage = model.shared_gate == moe::Storage::kQ8_0
      ? Storage::kQ8_0 : Storage::kBFloat16;
  const std::int64_t depth = indexer ? 128 : 512;
  const auto channels = ratio == 4 ? 2 * depth : depth;
  const std::int64_t chunk = ratio == 4 ? 4 : 64;
  Fragment f{.profile = model.profile,
             .region = std::uint64_t(indexer ? 0x08093403 :
                                    ratio == 4 ? 0x08093401 : 0x08093402),
             .quantized = model.gguf && storage == Storage::kQ8_0};
  std::optional<linear::Projection> kv_projection, gate_projection;
  ValueId new_kv, new_gate;
  if (model.gguf) {
    auto hidden = f.input(dense({chunk, model.hidden}));
    kv_projection = linear::projection(f, hidden, channels, model.hidden, storage, 251);
    gate_projection = linear::projection(f, hidden, channels, model.hidden, storage, 257);
    new_kv = kv_projection->output;
    new_gate = gate_projection->output;
  } else {
    // Official FP8 projections are a separate qualification. This is only
    // their post-projection F32 semantic consumer, never an FP8 substitute.
    new_kv = f.input(dense({chunk, channels}));
    new_gate = f.input(dense({chunk, channels}));
  }
  const auto past_kv_port = f.inputs.size();
  auto past_kv = f.input(dense({chunk, channels}));
  const auto past_score_port = f.inputs.size();
  auto past_score = f.input(dense({chunk, channels}));
  const auto cache_port = f.inputs.size();
  auto cache = f.input(dense({1, depth}));
  auto positions = f.input(dense({1}, c::DataType::kInt32));
  auto ape_rows = f.input(dense({chunk}, c::DataType::kInt32));
  auto publish = f.input(dense({1}, c::DataType::kBool));
  std::vector<float> ape(ratio * channels), norm(depth);
  for (std::size_t i = 0; i < ape.size(); ++i)
    ape[i] = float(std::int64_t((i * 7 + 3) % 23) - 11) / 16.0F;
  for (std::int64_t i = 0; i < depth; ++i)
    norm[i] = 0.75F + float(i % 7) / 16.0F;
  auto stored_ape = f.constant(dense({ratio, channels}), bytes(ape));
  auto selected_ape = f.one(c::Operation(c::Gather{}), {stored_ape, ape_rows});
  auto new_score = binary(f, c::BinaryKind::kAdd, new_gate, selected_ape);
  auto compressed = compress(f, past_kv, past_score, new_kv, new_score,
                              f.constant(dense({depth}), bytes(norm)), positions,
                              ratio, depth, indexer);
  auto next_cache = f.one(c::Operation(c::Select{}), {publish, compressed, cache});
  f.outputs = {next_cache, new_kv, new_score};
  f.state_ports = {{past_kv_port, 1}, {past_score_port, 2}, {cache_port, 0}};

  std::vector<float> old_kv(chunk * channels), old_score(chunk * channels), old_cache(depth);
  for (int step = 0; step < 6; ++step) {
    const bool reset = step % 3 == 0;
    const bool update = ratio == 4 || step % 3 != 1;
    if (reset) {
      for (std::size_t i = 0; i < old_kv.size(); ++i) {
        old_kv[i] = ratio == 4 && step == 0 ? 0.0F
            : float(std::int64_t((i * 3 + step) % 31) - 15) / 16.0F;
        old_score[i] = ratio == 4 && step == 0 ? -std::numeric_limits<float>::infinity()
            : float(std::int64_t((i * 5 + step) % 29) - 14) / 8.0F;
      }
      for (std::size_t i = 0; i < old_cache.size(); ++i)
        old_cache[i] = float(std::int64_t((i + step) % 17) - 8) / 16.0F;
    }
    sem::Replay replay{};
    replay.reset_state = reset;
    std::vector<float> kv, score;
    if (model.gguf) {
      auto hidden = d5_gguf_test::activation(chunk * model.hidden, 11 + 13 * step);
      replay.inputs.push_back(bytes(hidden));
      kv = project_rows(*kv_projection, hidden, model.hidden);
      score = project_rows(*gate_projection, hidden, model.hidden);
    } else {
      kv.resize(chunk * channels);
      score.resize(chunk * channels);
      for (std::size_t i = 0; i < kv.size(); ++i) {
        kv[i] = float(std::int64_t((i * 13 + 7 * step) % 43) - 21) / 16.0F;
        score[i] = float(std::int64_t((i * 11 + 5 * step) % 37) - 18) / 8.0F;
      }
      replay.inputs = {bytes(kv), bytes(score)};
    }
    // Only reset inputs carry initial state. A missing actual-output feedback
    // must fail conspicuously, rather than silently using the oracle's state.
    replay.inputs.push_back(bytes(reset ? old_kv : std::vector<float>(old_kv.size(), 23)));
    replay.inputs.push_back(bytes(reset ? old_score : std::vector<float>(old_score.size(), -19)));
    replay.inputs.push_back(bytes(reset ? old_cache : std::vector<float>(old_cache.size(), 31)));
    const auto position = std::int32_t((step >= 3 ? 65536 : 0) +
        (ratio == 4 ? 4 * (step % 3 + 1) - 1 : 127 + 64 * (step % 3)));
    std::vector<std::int32_t> indices(chunk);
    for (std::int64_t row = 0; row < chunk; ++row) {
      indices[row] = std::int32_t((position - chunk + 1 + row) % ratio);
      for (std::int64_t channel = 0; channel < channels; ++channel)
        score[row * channels + channel] += ape[indices[row] * channels + channel];
    }
    // The pinned cache planner rotates at the compressed block's START,
    // whereas APE indexes each original token's position modulo the ratio.
    const auto compressed_position = std::int32_t((position / ratio) * ratio);
    replay.inputs.push_back(bytes(std::vector<std::int32_t>{compressed_position}));
    replay.inputs.push_back(bytes(indices));
    replay.inputs.push_back(bytes(std::vector<std::uint8_t>{std::uint8_t(update)}));
    if (update)
      old_cache = compress_reference(old_kv, old_score, kv, score, norm, ratio,
                                      compressed_position, indexer);
    replay.outputs = {old_cache, kv, score};
    f.replays.push_back(std::move(replay));
    old_kv = std::move(kv);
    old_score = std::move(score);
  }
  return f;
}

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
