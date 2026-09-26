#pragma once

#include "d5_moe_execution_fragments.h"
#include "d5_kda_full_fragments.h"
#include "d5_glm_mtp_head_test.h"

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

inline std::vector<float> hadamard_reference(std::span<const float> input,
                                              std::int64_t depth) {
  std::vector<float> result(input.size());
  // Dense signed-matrix oracle, independent of the mature butterfly.
  for (std::size_t offset = 0; offset < input.size(); offset += depth)
    for (std::int64_t row = 0; row < depth; ++row) {
      double sum = 0;
      for (std::int64_t column = 0; column < depth; ++column)
        sum += (std::popcount(std::uint64_t(row & column)) % 2 ? -1 : 1) *
               double(input[offset + column]);
      result[offset + row] = float(sum / std::sqrt(double(depth)));
    }
  return result;
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
  return indexer ? hadamard_reference(result, depth) : result;
}

struct CompressorOptions final {
  std::int32_t first_block = 0;
  std::int32_t reset_position_offset = 65536;
  bool positive_hidden = false;
};
inline Fragment compressor(const moe::Case &model, std::int64_t ratio,
                             bool indexer = false, CompressorOptions options = {}) {
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
  // An official semantic input is also new_kv. Publish a copy so Graph's
  // distinct output binding cannot replace that input's storage binding.
  f.outputs = {next_cache, f.one(c::Operation(c::Copy{}), {new_kv}), new_score};
  f.state_ports = {{past_kv_port, 1}, {past_score_port, 2}, {cache_port, 0}};

  std::vector<float> old_kv(chunk * channels), old_score(chunk * channels), old_cache(depth);
  for (int step = 0; step < 6; ++step) {
    const bool reset = step % 3 == 0;
    const bool update = ratio == 4 || step % 3 != 1;
    if (reset) {
      for (std::size_t i = 0; i < old_kv.size(); ++i) {
        old_kv[i] = ratio == 4 && step == 0 && options.first_block == 0 ? 0.0F
            : float(std::int64_t((i * 3 + step) % 31) - 15) / 16.0F;
        old_score[i] = ratio == 4 && step == 0 && options.first_block == 0
            ? -std::numeric_limits<float>::infinity()
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
      if (options.positive_hidden)
        for (auto &value : hidden) value = std::abs(value) + 0.5F;
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
    const auto position = std::int32_t((step >= 3 ? options.reset_position_offset : 0) +
        options.first_block * ratio +
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
// Join compressor recipes at their common hidden activation, retaining each
// one's actual state feedback. This maps a test model graph, not an executor.
inline void join_compressor(Fragment &f, Fragment child) {
  if (f.replays.size() != child.replays.size())
    throw std::runtime_error("compressor replay mismatch");
  const auto input_base = f.inputs.size();
  const auto output_base = f.outputs.size();
  std::vector<ValueId> mapped(child.descriptions.size());
  mapped[child.inputs[0].ordinal()] = f.inputs[0];
  for (std::size_t i = 1; i < child.inputs.size(); ++i)
    mapped[child.inputs[i].ordinal()] = f.input(child.description(child.inputs[i]));
  for (std::size_t i = 0; i < child.constants.size(); ++i) {
    auto id = child.constants[i];
    mapped[id.ordinal()] = f.constant(child.description(id), std::move(child.constant_bytes[i]));
  }
  for (auto node : child.nodes) {
    for (auto &input : node.inputs) input = mapped[input.ordinal()];
    for (auto &output : node.outputs) {
      auto value = f.value(child.description(output), sem::ValueKind::kNodeOutput);
      mapped[output.ordinal()] = value;
      output = value;
    }
    f.nodes.push_back(std::move(node));
  }
  for (auto output : child.outputs) f.outputs.push_back(mapped[output.ordinal()]);
  for (auto [input, output] : child.state_ports)
    f.state_ports.push_back({input_base + input - 1, output_base + output});
  for (std::size_t step = 0; step < f.replays.size(); ++step) {
    auto &to = f.replays[step];
    auto &from = child.replays[step];
    if (to.inputs[0] != from.inputs[0] || to.reset_state != from.reset_state)
      throw std::runtime_error("compressors must consume the same hidden activation");
    for (std::size_t i = 1; i < from.inputs.size(); ++i)
      to.inputs.push_back(std::move(from.inputs[i]));
    for (auto &output : from.outputs) to.outputs.push_back(std::move(output));
  }
}
inline std::vector<float> float_values(std::span<const std::byte> data) {
  if (data.size() % sizeof(float)) throw std::runtime_error("misaligned F32 fixture");
  std::vector<float> result(data.size() / sizeof(float));
  std::memcpy(result.data(), data.data(), data.size());
  return result;
}

inline Fragment attention(const moe::Case &model, std::int64_t ratio) {
  if (!model.gguf || (ratio != 0 && ratio != 4 && ratio != 128))
    throw std::runtime_error("complete DeepSeek attention requires audited GGUF projections");
  using Storage = linear::Storage;
  const auto storage = model.shared_gate == moe::Storage::kQ8_0
      ? Storage::kQ8_0 : Storage::kBFloat16;
  const std::int64_t heads = model.experts == 384 ? 128 : 64;
  const std::int64_t rank = model.experts == 384 ? 1536 : 1024;
  const std::int64_t top = model.experts == 384 ? 1024 : 512;
  const std::int64_t groups = heads / 8;
  constexpr std::int64_t depth = 512, group_width = 4096, out_rank = 1024;
  const std::int64_t chunk = ratio == 4 ? 4 : ratio == 128 ? 64 : 1;
  const std::int64_t capacity = ratio == 4 ? top + 4 : ratio == 128 ? 20 : 0;
  const auto initial_blocks = capacity ? capacity - 3 : 0;
  const CompressorOptions options{std::int32_t(initial_blocks), 0, true};
  Fragment f{.profile = model.profile, .region = 0x08093404,
             .quantized = storage == Storage::kQ8_0};
  if (ratio) {
    f = compressor(model, ratio, false, options);
    if (ratio == 4) join_compressor(f, compressor(model, ratio, true, options));
    f.region = ratio == 4 ? 0x08093405 : 0x08093406;
  } else {
    (void)f.input(dense({1, model.hidden}));
    for (int step = 0; step < 6; ++step) {
      auto hidden = d5_gguf_test::activation(model.hidden, 11 + 13 * step);
      for (auto &value : hidden) value = std::abs(value) + 0.5F;
      sem::Replay replay{{bytes(hidden)}, {}};
      replay.reset_state = step % 3 == 0;
      f.replays.push_back(std::move(replay));
    }
  }
  const auto raw_port = f.inputs.size();
  auto raw_past = f.input(dense({127, depth}));
  auto positions = f.input(dense({chunk}, c::DataType::kInt32));
  auto query_position = slice(f, positions, 0, chunk - 1, 1);
  auto hidden = slice(f, f.inputs[0], 0, chunk - 1, 1);
  auto qa = linear::projection(f, hidden, rank, model.hidden, storage, 263);
  std::vector<float> qnorm(rank), knorm(depth), ones(depth, 1), sinks(heads);
  for (std::int64_t i = 0; i < rank; ++i) qnorm[i] = 0.75F + float(i % 7) / 32.0F;
  for (std::int64_t i = 0; i < depth; ++i) knorm[i] = 0.875F + float(i % 5) / 32.0F;
  for (std::int64_t i = 0; i < heads; ++i) sinks[i] = 0.75F + float(i % 11) / 16.0F;
  auto qr = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                   {qa.output, f.constant(dense({rank}), bytes(qnorm))});
  auto qb = linear::projection(f, qr, heads * depth, rank, storage, 269);
  auto q = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
      {reshape(f, qb.output, {1, 1, heads, depth}), f.constant(dense({depth}), bytes(ones))});
  q = tail_rope(f, q, query_position, ratio != 0);
  auto kv = linear::projection(f, f.inputs[0], depth, model.hidden, storage, 271);
  auto key = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                    {kv.output, f.constant(dense({depth}), bytes(knorm))});
  key = reshape(f, tail_rope(f, reshape(f, key, {1, chunk, 1, depth}),
                             positions, ratio != 0), {chunk, depth});
  auto raw_all = concat(f, raw_past, key, 0);
  auto raw_window = slice(f, raw_all, 0, chunk - 1, 128);
  auto next_raw = slice(f, raw_all, 0, chunk, 127);
  const auto raw_output = f.outputs.size();
  f.outputs.push_back(next_raw);
  f.state_ports.push_back({raw_port, raw_output});
  auto keys = raw_window;
  auto mask = core::filled(f, dense({1, 128}), 0);
  std::size_t comp_port = 0, comp_output = 0, index_port = 0, index_output = 0;
  std::size_t scores_output = 0, ids_output = 0;
  std::optional<linear::Projection> index_q, index_weight;
  if (ratio) {
    comp_port = f.inputs.size();
    auto old_compressed = f.input(dense({capacity, depth}));
    auto write_mask = f.input(dense({capacity, 1}, c::DataType::kBool));
    auto visibility = f.input(dense({1, capacity}));
    auto compressed = f.one(c::Operation(c::Select{}),
                             {write_mask, f.outputs[0], old_compressed});
    comp_output = f.outputs.size();
    f.outputs.push_back(compressed);
    f.state_ports.push_back({comp_port, comp_output});
    auto compressed_mask = visibility;
    if (ratio == 4) {
      index_port = f.inputs.size();
      auto old_index = f.input(dense({capacity, 128}));
      auto index_keys = f.one(c::Operation(c::Select{}),
                               {write_mask, f.outputs[3], old_index});
      index_output = f.outputs.size();
      f.outputs.push_back(index_keys);
      f.state_ports.push_back({index_port, index_output});
      index_q = linear::projection(f, qr, 64 * 128, rank, storage, 277);
      auto iq = tail_rope(f, reshape(f, index_q->output, {1, 1, 64, 128}),
                          query_position, true);
      iq = reshape(f, f.one(c::Operation(c::Hadamard{}), {iq}), {64, 128});
      index_weight = linear::projection(f, hidden, 64, model.hidden, storage, 281);
      auto weight = binary(f, c::BinaryKind::kMultiply, index_weight->output,
          core::filled(f, dense({1}), 1.0F / std::sqrt(8192.0F)));
      auto dot = f.one(c::Operation(c::MatMul{}), {iq, permute(f, index_keys, {1, 0})});
      auto positive = f.one(c::model_ops::clamp(0, std::numeric_limits<float>::max()), {dot});
      auto weighted = binary(f, c::BinaryKind::kMultiply, positive, reshape(f, weight, {64, 1}));
      auto scores = reshape(f, moe::sum(f, weighted, 0), {1, capacity});
      auto selected = f.append(c::Operation(c::TopK{.count = std::size_t(top)}),
          {binary(f, c::BinaryKind::kAdd, scores, visibility)}, 2);
      scores_output = f.outputs.size();
      f.outputs.push_back(f.one(c::Operation(c::Copy{}), {scores}));
      ids_output = f.outputs.size();
      f.outputs.push_back(selected[1]);
      f.unordered_ids = Fragment::UnorderedIds{ids_output, std::size_t(top), capacity, true};
      compressed = reshape(f, f.one(c::Operation(c::Gather{}), {compressed, selected[1]}), {top, depth});
      compressed_mask = reshape(f, f.one(c::Operation(c::Gather{}),
          {reshape(f, visibility, {capacity, 1}), selected[1]}), {1, top});
    }
    keys = concat(f, keys, compressed, 0);
    mask = concat(f, mask, compressed_mask, 1);
  }
  const auto key_count = 128 + (ratio == 4 ? top : capacity);
  keys = reshape(f, keys, {1, key_count, 1, depth});
  const float scale = 1.0F / std::sqrt(float(depth));
  auto context = f.one(c::model_ops::sink_attention(scale),
      {q, keys, keys, mask, f.constant(dense({heads}), bytes(sinks))});
  auto derotated = tail_rope(f, context, query_position, ratio != 0, true);
  auto grouped = reshape(f, derotated, {1, groups, group_width});
  auto wa = moe::weight(groups, out_rank, group_width, model.shared_gate, 283);
  std::vector<std::int32_t> group_ids(groups);
  std::iota(group_ids.begin(), group_ids.end(), 0);
  auto ids = f.constant(dense({1, groups}, c::DataType::kInt32), bytes(group_ids));
  auto oa = f.one(c::model_ops::grouped_matmul(),
      {f.constant(wa.description, std::move(wa.storage)), grouped, ids});
  auto ob = linear::projection(f, reshape(f, oa, {1, groups * out_rank}),
                               model.hidden, groups * out_rank, storage, 293);
  const auto context_output = f.outputs.size();
  f.outputs.push_back(context);
  f.outputs.push_back(derotated);
  f.outputs.push_back(oa);
  f.outputs.push_back(ob.output);
  if (storage == Storage::kBFloat16)
    for (std::size_t i = 0; i < f.outputs.size(); ++i)
      if (ratio != 4 || i != ids_output) f.bfloat_chain_outputs.push_back(i);

  std::vector<float> raw_history(127 * depth), comp_history(capacity * depth);
  std::vector<float> index_history(ratio == 4 ? capacity * 128 : 0);
  std::int64_t visible = initial_blocks;
  for (std::size_t step = 0; step < f.replays.size(); ++step) {
    auto &replay = f.replays[step];
    if (replay.reset_state) {
      visible = initial_blocks;
      for (std::size_t i = 0; i < raw_history.size(); ++i)
        raw_history[i] = float(std::sin(double(i + step * 17) * 0.017) * 0.5);
      for (std::int64_t row = 0; row < capacity; ++row)
        for (std::int64_t d = 0; d < depth; ++d)
          comp_history[row * depth + d] = row < initial_blocks
              ? float(std::sin(double(row * 19 + d + step * 11) * 0.021) * 0.75) : 13.0F;
      for (std::int64_t row = 0; row < (ratio == 4 ? capacity : 0); ++row)
        for (std::int64_t d = 0; d < 128; ++d)
          index_history[row * 128 + d] = row < initial_blocks
              ? (d == 0 ? float(row + 1) / 64.0F : 0.0F) : 32.0F;
    }
    auto x = float_values(replay.inputs[0]);
    auto last = std::span<const float>(x).last(model.hidden);
    const auto position = std::int32_t(ratio
        ? initial_blocks * ratio + (ratio == 4 ? 4 * (step % 3 + 1) - 1 : 127 + 64 * (step % 3))
        : 127 + step % 3);
    std::vector<std::int32_t> token_positions(chunk);
    for (std::int64_t t = 0; t < chunk; ++t) token_positions[t] = std::int32_t(position - chunk + 1 + t);
    replay.inputs.push_back(bytes(replay.reset_state ? raw_history : std::vector<float>(raw_history.size(), 23)));
    replay.inputs.push_back(bytes(token_positions));
    auto qr_values = sem::oracle::rms_norm_rows(qa.reference(last), qnorm, rank, 1.0e-6F);
    auto query = sem::oracle::rms_norm_rows(qb.reference(qr_values), ones, depth, 1.0e-6F);
    query = rope_reference(query, depth, position, ratio != 0);
    auto new_keys = sem::oracle::rms_norm_rows(project_rows(kv, x, model.hidden), knorm, depth, 1.0e-6F);
    for (std::int64_t t = 0; t < chunk; ++t) {
      auto row = rope_reference(std::span<const float>(new_keys).subspan(t * depth, depth),
                                depth, token_positions[t], ratio != 0);
      std::copy(row.begin(), row.end(), new_keys.begin() + t * depth);
    }
    auto full_raw = raw_history;
    full_raw.insert(full_raw.end(), new_keys.begin(), new_keys.end());
    std::vector<float> attended_keys(full_raw.end() - 128 * depth, full_raw.end());
    raw_history.assign(full_raw.end() - 127 * depth, full_raw.end());
    replay.outputs.resize(f.outputs.size());
    replay.outputs[raw_output] = raw_history;
    std::vector<float> attention_mask(128, 0);
    if (ratio) {
      replay.inputs.push_back(bytes(replay.reset_state ? comp_history : std::vector<float>(comp_history.size(), -23)));
      const bool update = ratio == 4 || step % 3 != 1;
      std::vector<std::uint8_t> writing(capacity);
      if (update) {
        writing[visible] = 1;
        std::copy(replay.outputs[0].begin(), replay.outputs[0].end(), comp_history.begin() + visible * depth);
        ++visible;
      }
      std::vector<float> visibility(capacity, -std::numeric_limits<float>::infinity());
      std::fill_n(visibility.begin(), visible, 0.0F);
      replay.inputs.push_back(bytes(writing));
      replay.inputs.push_back(bytes(visibility));
      replay.outputs[comp_output] = comp_history;
      std::vector<std::int32_t> selected;
      if (ratio == 4) {
        replay.inputs.push_back(bytes(replay.reset_state ? index_history : std::vector<float>(index_history.size(), 29)));
        std::copy(replay.outputs[3].begin(), replay.outputs[3].end(), index_history.begin() + (visible - 1) * 128);
        replay.outputs[index_output] = index_history;
        auto iq = hadamard_reference(rope_reference(index_q->reference(qr_values), 128, position, true), 128);
        auto iw = index_weight->reference(last);
        for (auto &value : iw) value *= 1.0F / std::sqrt(8192.0F);
        std::vector<float> scores(capacity);
        for (std::int64_t row = 0; row < capacity; ++row) {
          double sum = 0;
          for (std::int64_t h = 0; h < 64; ++h) {
            double dot = 0;
            for (std::int64_t d = 0; d < 128; ++d)
              dot += double(iq[h * 128 + d]) * index_history[row * 128 + d];
            sum += std::max(0.0, dot) * iw[h];
          }
          scores[row] = float(sum);
        }
        selected.resize(visible);
        std::iota(selected.begin(), selected.end(), 0);
        std::stable_sort(selected.begin(), selected.end(), [&](auto a, auto b) { return scores[a] > scores[b]; });
        selected.resize(top);
        replay.outputs[scores_output] = scores;
        for (auto id : selected) replay.outputs[ids_output].push_back(static_cast<float>(id));
      } else {
        selected.resize(capacity);
        std::iota(selected.begin(), selected.end(), 0);
      }
      for (auto row : selected) {
        attended_keys.insert(attended_keys.end(), comp_history.begin() + row * depth,
                              comp_history.begin() + (row + 1) * depth);
        attention_mask.push_back(visibility[row]);
      }
    }
    auto context_values = sink_reference(query, attended_keys, attended_keys, attention_mask,
                                          sinks, 1, key_count, scale);
    auto derotated_values = rope_reference(context_values, depth, position, ratio != 0, true);
    auto grouped_values = moe::project(derotated_values, wa, group_ids, out_rank, group_width);
    replay.outputs[context_output] = std::move(context_values);
    replay.outputs[context_output + 1] = std::move(derotated_values);
    replay.outputs[context_output + 2] = grouped_values;
    replay.outputs[context_output + 3] = ob.reference(grouped_values);
  }
  return f;
}
inline Fragment hyper_connection(const moe::Case &model) {
  Fragment f{.profile = model.profile, .region = 0x0c0e3401};
  auto residual = f.input(dense({1, 4, model.hidden}));
  auto branch = f.input(dense({1, model.hidden}));
  const auto width = 4 * model.hidden;
  const auto normalize = [&](ValueId value) {
    return f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
        {reshape(f, value, {1, width}), core::filled(f, dense({width}), 1)});
  };
  auto normalized = normalize(residual);
  auto fn = linear::projection(f, normalized, 24, width, linear::Storage::kF32, 307);
  auto child = core::hyper_connection({model.profile, model.hidden});
  std::vector<ValueId> mapped(child.descriptions.size());
  const std::array child_inputs{fn.output, residual, branch};
  for (std::size_t i = 0; i < child.inputs.size(); ++i)
    mapped[child.inputs[i].ordinal()] = child_inputs[i];
  for (std::size_t i = 0; i < child.constants.size(); ++i) {
    auto id = child.constants[i];
    mapped[id.ordinal()] = f.constant(child.description(id), std::move(child.constant_bytes[i]));
  }
  for (auto node : child.nodes) {
    for (auto &input : node.inputs) input = mapped[input.ordinal()];
    for (auto &output : node.outputs) {
      auto value = f.value(child.description(output), sem::ValueKind::kNodeOutput);
      mapped[output.ordinal()] = value;
      output = value;
    }
    f.nodes.push_back(std::move(node));
  }
  auto combined = mapped[child.outputs[0].ordinal()];
  auto collapsed = mapped[child.outputs[1].ordinal()];
  auto expanded = mapped[child.outputs[2].ordinal()];
  auto hn = normalize(expanded);
  auto head = linear::projection(f, hn, 4, width, linear::Storage::kF32, 311);
  std::vector<float> head_base{-0.25F, 0.125F, 0.375F, -0.5F};
  auto gate = binary(f, c::BinaryKind::kAdd,
      binary(f, c::BinaryKind::kMultiply, head.output, core::filled(f, dense({1}), 0.125F)),
      f.constant(dense({4}), bytes(head_base)));
  gate = binary(f, c::BinaryKind::kAdd, core::unary(f, c::UnaryKind::kSigmoid, gate),
                  core::filled(f, dense({1}), 1.0e-6F));
  auto head_value = f.one(c::model_ops::hyper_connection_pre(), {expanded, gate});
  auto final_scale = sem::oracle::scale(model.hidden);
  auto norm_weight = linear::parameter(f, dense({model.hidden}), final_scale, !model.gguf);
  auto final = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}), {head_value, norm_weight});
  f.outputs = {expanded, normalized, fn.output, combined, collapsed, head_value, final};
  f.state_ports = {{0, 0}};
  std::vector<float> scales{0.125F, 0.25F, 0.5F}, bases(24), ones(width, 1);
  for (int i = 0; i < 24; ++i) bases[i] = float(i % 7 - 3) / 8.0F;
  std::vector<float> state;
  for (int step = 0; step < 6; ++step) {
    const bool reset = step % 3 == 0;
    if (reset) state = d5_gguf_test::values(4, model.hidden, 31 + step);
    auto branch_values = d5_gguf_test::activation(model.hidden, 3 + step * 14);
    auto nv = sem::oracle::rms_norm_rows(state, ones, width, 1.0e-6F);
    auto mixes = fn.reference(nv);
    auto values = core::hyper_reference(mixes, state, branch_values, model.hidden, scales, bases);
    auto head_input = sem::oracle::rms_norm_rows(values[2], ones, width, 1.0e-6F);
    auto head_mixes = head.reference(head_input);
    std::vector<float> head_result(model.hidden);
    for (std::int64_t d = 0; d < model.hidden; ++d) {
      double sum = 0;
      for (int stream = 0; stream < 4; ++stream)
        sum += values[2][stream * model.hidden + d] *
            (core::sigmoid(head_mixes[stream] * 0.125 + head_base[stream]) + 1.0e-6);
      head_result[d] = float(sum);
    }
    auto final_values = sem::oracle::rms_norm_rows(head_result, final_scale, model.hidden, 1.0e-6F);
    sem::Replay replay{{bytes(reset ? state : std::vector<float>(state.size(), 37)), bytes(branch_values)},
        {values[2], nv, mixes, values[0], values[1], head_result, final_values}};
    replay.reset_state = reset;
    f.replays.push_back(std::move(replay));
    state = std::move(values[2]);
  }
  return f;
}

template <class Run> void check_hyper_head(const moe::Case &model, Run run) {
  auto fragment = hyper_connection(model);
  std::vector<std::vector<float>> observed, reference;
  for (const auto &replay : fragment.replays) reference.push_back(replay.outputs.back());
  fragment.observe = [&](std::size_t port, std::span<const float> values) {
    if (port == 6) observed.emplace_back(values.begin(), values.end());
  };
  run(std::move(fragment));
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  ASSERT_EQ(observed.size(), reference.size());
  if (!model.gguf) return; // The official FP8/MTP profile remains separate.
  const auto found = std::ranges::find_if(core::profiles::kVocabulary,
      [&](const auto &candidate) { return candidate.profile == model.profile; });
  ASSERT_NE(found, core::profiles::kVocabulary.end());
  ASSERT_EQ(found->vocabulary_rows, 129280);
  d5_glm_mtp_test::check_head(*found, observed, reference);
}
} // namespace d5_deepseek_remaining_fragments
