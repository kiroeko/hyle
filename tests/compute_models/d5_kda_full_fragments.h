#pragma once

#include "d5_moe_semantic_fragments.h"

// Shared test consumer: preserve native projection storage, then compose the
// pinned Kimi/GLM KDA model program through public Compute operations. The host
// calculations below are independent test oracles, never a runtime fallback.
namespace d5_kda_full_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
using d5_kda_projection_cases::Storage;
using sem::Fragment;
using sem::ValueId;
using sem::bytes;
using sem::dense;
using core::binary;
using core::filled;
using core::reshape;
using core::unary;

inline float bf16(float value) {
  return float(sem::q::decode(sem::q::encode(value)));
}
inline ValueId parameter(Fragment &f, c::TensorDesc desc,
                         std::vector<float> &values, bool bfloat) {
  if (!bfloat)
    return f.constant(desc, bytes(values));
  std::vector<std::uint16_t> encoded;
  for (auto &value : values) {
    encoded.push_back(sem::q::encode(value));
    value = float(sem::q::decode(encoded.back()));
  }
  auto stored = f.constant(c::TensorDesc::dense(c::DataType::kBFloat16,
                                               desc.dimensions()).value(),
                            bytes(encoded));
  return f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {stored});
}

struct Projection final {
  sem::Weight weight;
  Storage storage;
  ValueId output;
  std::vector<float> reference(std::span<const float> input) const {
    if (storage != Storage::kBFloat16)
      return sem::project(input, weight);
    std::vector<float> rounded(input.begin(), input.end());
    for (auto &value : rounded) value = bf16(value);
    return sem::project(rounded, weight);
  }
};
inline Projection projection(Fragment &f, ValueId input, std::int64_t n,
                              std::int64_t k, Storage storage, int seed) {
  sem::Weight weight{dense({n, k}), {}, {}};
  if (storage == Storage::kBFloat16) {
    weight.description = dense({n, k}, c::DataType::kBFloat16);
    weight.storage.resize(weight.description.required_bytes().value());
    weight.decoded.resize(n);
    for (std::int64_t row = 0; row < n; ++row) {
      const auto column = (17 * row + seed) % k;
      const auto code = sem::q::encode(0.5 + double((row + seed) % 13) / 32.0);
      std::memcpy(weight.storage.data() + (row * k + column) * sizeof(code),
                  &code, sizeof(code));
      weight.decoded[row].emplace_back(std::size_t(column),
                                       float(sem::q::decode(code)));
    }
    input = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {input});
  } else {
    weight = sem::weight(n, k, c::GgufQuantizedType::kQ8_0,
                         storage == Storage::kF32, seed);
  }
  auto stored = f.constant(weight.description, std::move(weight.storage));
  auto output = f.one(storage == Storage::kBFloat16
                         ? c::model_ops::bfloat_linear_f32()
                         : c::Operation(c::MatMul{}), {input, stored});
  return {std::move(weight), storage, output};
}

inline Fragment complete(const c::ModelArtifactProfileDescription *profile) {
  const bool kimi = profile->model_name == "Kimi-K3";
  const bool official = profile->format_family == "safetensors";
  const auto storage = (official || profile == core::kKdaProfiles[2])
      ? Storage::kBFloat16 : Storage::kQ8_0;
  const auto beta_storage = profile == core::kKdaProfiles[1]
      ? Storage::kF32 : storage;
  const std::int64_t heads = kimi ? 96 : 64;
  const std::int64_t hidden = kimi ? 7168 : 4096;
  constexpr std::int64_t depth = 128;
  const auto width = heads * depth;
  const auto channels = 3 * width;
  Fragment f{.profile = profile, .region = 0x030602,
             .quantized = storage == Storage::kQ8_0};
  auto input = f.input(dense({1, hidden}));
  auto history = f.input(dense({3, channels}));
  auto state_input = f.input(dense({1, heads, depth, depth}));
  auto q = projection(f, input, width, hidden, storage, 173);
  auto k = projection(f, input, width, hidden, storage, 179);
  auto v = projection(f, input, width, hidden, storage, 181);
  auto fa = projection(f, input, depth, hidden, storage, 191);
  auto fb = projection(f, fa.output, width, depth, storage, 193);
  auto b = projection(f, input, heads, hidden, beta_storage, 197);
  std::optional<Projection> ga;
  if (!kimi) ga = projection(f, input, depth, hidden, storage, 199);
  auto g = projection(f, kimi ? input : ga->output, width,
                       kimi ? hidden : depth, storage, 211);

  auto joined = f.one(c::Operation(c::Concat{.axis = 1}), {q.output, k.output});
  joined = f.one(c::Operation(c::Concat{.axis = 1}), {joined, v.output});
  auto window = f.one(c::Operation(c::Concat{.axis = 0}), {history, joined});
  auto next_history = core::rows(f, window, 1, 3);
  d5_qwen38_gdn_fixture::ConvFixture conv(heads, heads);
  for (std::size_t i = 0; i < conv.weights.size(); ++i)
    conv.weights[i] += float(int(i % 7) - 3) * 0.000137F;
  auto conv_weight = parameter(f, conv.weights_desc, conv.weights,
                               official && !kimi);
  auto convolution = f.one(c::model_ops::causal_conv1d(),
                            {reshape(f, window, {1, 4, channels}), conv_weight});
  auto activated = f.one(c::Operation(c::GatedActivation{
                            .inputs = c::GatedInputKind::kSeparate}),
                         {convolution, filled(f, f.description(convolution), 1)});
  auto channel_rows = reshape(f, activated, {channels, 1});
  auto query = reshape(f, core::rows(f, channel_rows, 0, width), {heads, depth});
  auto key = reshape(f, core::rows(f, channel_rows, width, width), {heads, depth});
  auto value = reshape(f, core::rows(f, channel_rows, 2 * width, width),
                        {1, 1, heads, depth});
  auto l2_scale = filled(f, dense({depth}), 1.0F / std::sqrt(float(depth)));
  const c::Operation l2(c::RmsNorm{.epsilon = 1.0e-6F / float(depth)});
  query = reshape(f, f.one(l2, {query, l2_scale}), {1, 1, heads, depth});
  key = reshape(f, f.one(l2, {key, l2_scale}), {1, 1, heads, depth});

  // Official Kimi stores 128 A_log entries; the pinned mature loader consumes
  // only its 96 heads. A conspicuous padded tail detects accidental use.
  std::vector<float> a_source(official && kimi ? 128 : heads), alpha(heads);
  for (std::size_t h = 0; h < a_source.size(); ++h) {
    const float log = float(int(h % 7) - 3) / 8.0F;
    a_source[h] = official ? (h < std::size_t(heads) ? log : 19.0F)
                            : -std::exp(log);
    if (h < std::size_t(heads)) alpha[h] = std::exp(log);
  }
  auto a = f.constant(dense({std::int64_t(a_source.size()), 1}), bytes(a_source));
  if (official) {
    a = core::rows(f, a, 0, heads);
    a = unary(f, c::UnaryKind::kExp, a);
  } else {
    a = binary(f, c::BinaryKind::kMultiply, a,
                filled(f, f.description(a), -1));
  }
  a = reshape(f, a, {1, 1, heads, 1});
  std::vector<float> dt(width);
  for (std::size_t i = 0; i < dt.size(); ++i)
    dt[i] = float(int(i % 11) - 5) / 32.0F;
  auto gate_desc = dense({1, 1, heads, depth});
  auto raw_gate = reshape(f, fb.output, {1, 1, heads, depth});
  auto decay = binary(f, c::BinaryKind::kAdd, raw_gate,
                       f.constant(gate_desc, bytes(dt)));
  decay = unary(f, c::UnaryKind::kSigmoid,
                 binary(f, c::BinaryKind::kMultiply, decay, a));
  decay = binary(f, c::BinaryKind::kMultiply, decay, filled(f, gate_desc, -5));
  auto beta = unary(f, c::UnaryKind::kSigmoid,
                     reshape(f, b.output, {1, 1, heads, 1}));
  auto recurrent = f.append(c::Operation(c::GatedDeltaNet{}),
                            {query, key, value, decay, beta, state_input}, 2);
  auto norm_values = sem::oracle::scale(depth);
  for (std::size_t i = 0; i < norm_values.size(); ++i)
    norm_values[i] += float(i % 5) * 0.0037F;
  auto norm_weight = parameter(f, dense({depth}), norm_values, official && !kimi);
  auto normed = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                      {recurrent[0], norm_weight});
  auto gated = binary(f, c::BinaryKind::kMultiply, normed,
                       unary(f, c::UnaryKind::kSigmoid,
                              reshape(f, g.output, {1, 1, heads, depth})));
  auto out = projection(f, reshape(f, gated, {1, width}), hidden, width,
                         storage, 223);
  f.outputs = {next_history, convolution, recurrent[1], recurrent[0], gated,
               out.output};
  f.state_ports = {{1, 0}, {2, 2}};

  std::vector<float> state(heads * depth * depth);
  for (std::size_t i = 0; i < state.size(); ++i)
    state[i] = float(std::sin(double(i) * 0.013) * 0.0005);
  auto initial_window = conv.samples(3);
  std::vector<float> past(initial_window.begin(), initial_window.begin() + 3 * channels);
  for (int step = 0; step < 8; ++step) {
    auto x = d5_gguf_test::activation(hidden, 3 + 14 * step);
    sem::Replay replay{{bytes(x), bytes(past), bytes(state)}, {}};
    auto projected = q.reference(x);
    auto pk = k.reference(x), pv = v.reference(x);
    projected.insert(projected.end(), pk.begin(), pk.end());
    projected.insert(projected.end(), pv.begin(), pv.end());
    auto samples = past;
    samples.insert(samples.end(), projected.begin(), projected.end());
    past.assign(samples.begin() + channels, samples.end());
    auto conv_values = sem::as_float(conv.oracle(samples));
    std::vector<float> act(channels);
    for (std::size_t i = 0; i < act.size(); ++i)
      act[i] = float(conv_values[i] * core::sigmoid(conv_values[i]));
    auto gate = fb.reference(fa.reference(x));
    auto betas = b.reference(x);
    auto output_gate = kimi ? g.reference(x) : g.reference(ga->reference(x));
    std::array<std::vector<float>, 2> qk{
        std::vector<float>(width), std::vector<float>(width)};
    for (std::int64_t p = 0; p < 2; ++p)
      for (std::int64_t h = 0; h < heads; ++h) {
        double sum = 1.0e-6;
        for (std::int64_t d = 0; d < depth; ++d) {
          const double z = act[p * width + h * depth + d];
          sum += z * z;
        }
        for (std::int64_t d = 0; d < depth; ++d)
          qk[p][h * depth + d] = float(act[p * width + h * depth + d] / std::sqrt(sum));
      }
    std::vector<float> output(width);
    for (std::int64_t h = 0; h < heads; ++h) {
      std::array<double, depth> decay_values;
      for (std::int64_t d = 0; d < depth; ++d) {
        const auto i = h * depth + d;
        decay_values[d] = std::exp(-5.0 * core::sigmoid(
            double(alpha[h]) * (double(gate[i]) + dt[i])));
      }
      for (std::int64_t v_index = 0; v_index < depth; ++v_index) {
        const auto row = (h * depth + v_index) * depth;
        std::array<double, depth> decayed;
        double dot = 0;
        for (std::int64_t d = 0; d < depth; ++d) {
          decayed[d] = state[row + d] * decay_values[d];
          dot += decayed[d] * qk[1][h * depth + d];
        }
        const double delta = core::sigmoid(betas[h]) *
            (act[2 * width + h * depth + v_index] - dot);
        double score = 0;
        for (std::int64_t d = 0; d < depth; ++d) {
          const double next = decayed[d] + delta * qk[1][h * depth + d];
          state[row + d] = float(next);
          score += next * qk[0][h * depth + d];
        }
        output[h * depth + v_index] = float(score / std::sqrt(double(depth)));
      }
    }
    auto expected = sem::oracle::rms_norm_rows(output, norm_values, depth, 1.0e-6F);
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] *= float(core::sigmoid(output_gate[i]));
    auto final = out.reference(expected);
    replay.outputs = {past, std::move(conv_values), state, std::move(output),
                       std::move(expected), std::move(final)};
    f.replays.push_back(std::move(replay));
  }
  return f;
}
} // namespace d5_kda_full_fragments
