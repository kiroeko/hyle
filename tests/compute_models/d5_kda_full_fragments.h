#pragma once

#include "d5_moe_semantic_fragments.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

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
  std::vector<sem::q::BF16> encoded;
  for (auto &value : values) {
    encoded.push_back(sem::q::encode(value));
    value = float(sem::q::decode(encoded.back()));
  }
  auto stored = f.constant(c::TensorDesc::dense(c::DataType::kBFloat16,
                                               desc.dimensions()).value(),
                            bytes(encoded));
  return f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {stored});
}

inline std::vector<float> projection_reference(std::span<const float> input,
    const sem::Weight &weight, Storage storage) {
  if (storage != Storage::kBFloat16) return sem::project(input, weight);
  std::vector<float> rounded(input.begin(), input.end());
  for (auto &value : rounded) value = bf16(value);
  return sem::project(rounded, weight);
}

// A supplied weight is already a Value in this Fragment. Its actual typed
// storage fixes the existing F32 projection boundary, without a profile route.
inline void require_value(const Fragment &f, ValueId value,
                          std::initializer_list<std::int64_t> dimensions) {
  if (value.ordinal() >= f.descriptions.size() ||
      !std::ranges::equal(f.description(value).dimensions(), dimensions))
    throw std::runtime_error("supplied KDA/MLA parameter or input shape differs");
}
inline void require_float_parameter(const Fragment &f, ValueId value,
                                   std::initializer_list<std::int64_t> dimensions) {
  require_value(f, value, dimensions);
  const auto &description = f.description(value);
  if (!description.is_plain() ||
      (description.data_type() != c::DataType::kFloat32 &&
       description.data_type() != c::DataType::kBFloat16))
    throw std::runtime_error("supplied KDA/MLA scalar parameter must be F32 or BF16");
}
inline void require_projection(const Fragment &f, ValueId value,
                               std::int64_t n, std::int64_t k) {
  if (value.ordinal() >= f.descriptions.size())
    throw std::runtime_error("supplied projection is outside this Fragment");
  const auto &description = f.description(value);
  if ((description.data_type() != c::DataType::kFloat32 &&
       description.data_type() != c::DataType::kBFloat16) ||
      (description.data_type() == c::DataType::kBFloat16 && !description.is_plain()))
    throw std::runtime_error("supplied projection needs its original F32/GGUF or BF16 weight");
  // DenseLinearF32 reads BF16 [N,K]; the existing MatMul RHS is logically
  // [K,N] for both transposed F32 storage and GGUF. Do not transpose it here.
  if (description.data_type() == c::DataType::kBFloat16) require_value(f, value, {n, k});
  else require_value(f, value, {k, n});
}
inline ValueId float_parameter(Fragment &f, ValueId value) {
  return f.description(value).data_type() == c::DataType::kBFloat16
      ? f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {value}) : value;
}
inline ValueId project_f32(Fragment &f, ValueId input, ValueId weight) {
  const auto &description = f.description(weight);
  if (description.data_type() == c::DataType::kBFloat16) {
    input = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {input});
    return f.one(c::model_ops::dense_linear_f32(), {input, weight});
  }
  return f.one(c::Operation(c::MatMul{}), {input, weight});
}

struct Projection final {
  sem::Weight weight;
  Storage storage;
  ValueId output;
  std::vector<float> reference(std::span<const float> input) const {
    return projection_reference(input, weight, storage);
  }
};
// Deterministic test initialization, separate from the mathematical append.
// Real source consumers supply their own original ValueIds instead.
struct ProjectionWeight final {
  sem::Weight weight;
  Storage storage;
  ValueId value;
  std::vector<float> reference(std::span<const float> input) const {
    return projection_reference(input, weight, storage);
  }
};
inline ProjectionWeight prepare_projection(Fragment &f, std::int64_t n,
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
  } else {
    weight = sem::weight(n, k, c::GgufQuantizedType::kQ8_0,
                         storage == Storage::kF32, seed);
  }
  auto stored = f.constant(weight.description, std::move(weight.storage));
  return {std::move(weight), storage, stored};
}
inline Projection projection(Fragment &f, ValueId input, std::int64_t n,
                              std::int64_t k, Storage storage, int seed) {
  auto prepared = prepare_projection(f, n, k, storage, seed);
  const auto output = project_f32(f, input, prepared.value);
  return {std::move(prepared.weight), storage, output};
}

enum class DecayEncoding { kLog, kNegativeExp };
struct KdaGeometry final {
  std::int64_t hidden, heads, depth;
  float l2_epsilon, norm_epsilon;
};
struct KdaParameters final {
  ValueId q, k, v, f_a, f_b, beta, gate, output;
  ValueId convolution, a, dt_bias, output_norm;
  std::optional<ValueId> gate_a;
  DecayEncoding decay;
};
struct KdaPorts final {
  ValueId history, convolution, state, recurrent, gated, output;
  std::array<ValueId, 6> outputs() const {
    return {history, convolution, state, recurrent, gated, output};
  }
};

// Model-neutral KDA equation shared by Kimi's direct output gate and GLM's
// low-rank gate. Source binding/geometry is supplied cold; no profile identity,
// generated weight, owner or state publication is created here. All Values
// must belong to f. The caller keeps the original state port declaration.
inline KdaPorts append_kda(Fragment &f, ValueId input, ValueId history,
    ValueId state, const KdaParameters &p, const KdaGeometry &geometry) {
  const auto [hidden, heads, depth, l2_epsilon, norm_epsilon] = geometry;
  if (hidden <= 0 || heads <= 0 || depth != 128 ||
      heads > std::numeric_limits<std::int64_t>::max() / depth / 3 ||
      !(l2_epsilon > 0) || !std::isfinite(l2_epsilon) ||
      !(norm_epsilon > 0) || !std::isfinite(norm_epsilon) ||
      (p.decay != DecayEncoding::kLog && p.decay != DecayEncoding::kNegativeExp))
    throw std::runtime_error("invalid supplied KDA geometry or decay encoding");
  const auto width = heads * depth, channels = 3 * width;
  require_value(f, input, {1, hidden});
  require_value(f, history, {3, channels});
  require_value(f, state, {1, heads, depth, depth});
  for (auto value : {input, history, state})
    if (f.description(value).data_type() != c::DataType::kFloat32 || !f.description(value).is_plain())
      throw std::runtime_error("supplied KDA keeps its original plain F32 input/state boundary");
  for (auto value : {p.q, p.k, p.v}) require_projection(f, value, width, hidden);
  require_projection(f, p.f_a, depth, hidden);
  require_projection(f, p.f_b, width, depth);
  require_projection(f, p.beta, heads, hidden);
  if (p.gate_a) require_projection(f, *p.gate_a, depth, hidden);
  require_projection(f, p.gate, width, p.gate_a ? depth : hidden);
  require_projection(f, p.output, hidden, width);
  require_float_parameter(f, p.convolution, {channels, 4});
  require_float_parameter(f, p.dt_bias, {1, 1, heads, depth});
  require_float_parameter(f, p.output_norm, {depth});
  if (p.a.ordinal() >= f.descriptions.size() ||
      f.description(p.a).rank() != 2 || f.description(p.a).dimensions()[0] < heads)
    throw std::runtime_error("supplied KDA decay source does not cover every head");
  require_float_parameter(f, p.a, {f.description(p.a).dimensions()[0], 1});
  if (p.decay == DecayEncoding::kNegativeExp && f.description(p.a).dimensions()[0] != heads)
    throw std::runtime_error("prepared negative-exp KDA decay must have exactly one value per head");

  const auto q = project_f32(f, input, p.q), k = project_f32(f, input, p.k);
  const auto v = project_f32(f, input, p.v);
  const auto fa = project_f32(f, input, p.f_a), fb = project_f32(f, fa, p.f_b);
  const auto beta_input = project_f32(f, input, p.beta);
  const auto gate_input = p.gate_a ? project_f32(f, input, *p.gate_a) : input;
  const auto gate = project_f32(f, gate_input, p.gate);
  auto joined = f.one(c::Operation(c::Concat{.axis = 1}), {q, k});
  joined = f.one(c::Operation(c::Concat{.axis = 1}), {joined, v});
  const auto window = f.one(c::Operation(c::Concat{.axis = 0}), {history, joined});
  const auto next_history = core::rows(f, window, 1, 3);
  const auto convolution = f.one(c::model_ops::causal_conv1d(),
      {reshape(f, window, {1, 4, channels}), float_parameter(f, p.convolution)});
  const auto activated = f.one(c::Operation(c::GatedActivation{
      .inputs = c::GatedInputKind::kSeparate}),
      {convolution, filled(f, f.description(convolution), 1)});
  const auto channel_rows = reshape(f, activated, {channels, 1});
  auto query = reshape(f, core::rows(f, channel_rows, 0, width), {heads, depth});
  auto key = reshape(f, core::rows(f, channel_rows, width, width), {heads, depth});
  const auto value = reshape(f, core::rows(f, channel_rows, 2 * width, width), {1, 1, heads, depth});
  const auto l2_scale = filled(f, dense({depth}), 1.0F / std::sqrt(float(depth)));
  const c::Operation l2(c::RmsNorm{.epsilon = l2_epsilon / float(depth)});
  query = reshape(f, f.one(l2, {query, l2_scale}), {1, 1, heads, depth});
  key = reshape(f, f.one(l2, {key, l2_scale}), {1, 1, heads, depth});
  auto a = float_parameter(f, p.a);
  if (p.decay == DecayEncoding::kLog) {
    a = core::rows(f, a, 0, heads);
    a = unary(f, c::UnaryKind::kExp, a);
  } else {
    a = binary(f, c::BinaryKind::kMultiply, a, filled(f, f.description(a), -1));
  }
  a = reshape(f, a, {1, 1, heads, 1});
  const auto gate_description = dense({1, 1, heads, depth});
  const auto raw_gate = reshape(f, fb, {1, 1, heads, depth});
  auto decay = binary(f, c::BinaryKind::kAdd, raw_gate, float_parameter(f, p.dt_bias));
  decay = unary(f, c::UnaryKind::kSigmoid, binary(f, c::BinaryKind::kMultiply, decay, a));
  decay = binary(f, c::BinaryKind::kMultiply, decay, filled(f, gate_description, -5));
  const auto beta = unary(f, c::UnaryKind::kSigmoid, reshape(f, beta_input, {1, 1, heads, 1}));
  const auto recurrent = f.append(c::Operation(c::GatedDeltaNet{}), {query, key, value, decay, beta, state}, 2);
  const auto normed = f.one(c::Operation(c::RmsNorm{.epsilon = norm_epsilon}),
      {recurrent[0], float_parameter(f, p.output_norm)});
  const auto gated = binary(f, c::BinaryKind::kMultiply, normed,
      unary(f, c::UnaryKind::kSigmoid, reshape(f, gate, {1, 1, heads, depth})));
  const auto output = project_f32(f, reshape(f, gated, {1, width}), p.output);
  return {next_history, convolution, recurrent[1], recurrent[0], gated, output};
}

using Reference = std::function<std::vector<std::vector<float>>(
    std::span<const float>, std::span<const float>, std::span<const float>)>;

inline Fragment complete(const c::ModelArtifactProfileDescription *profile,
                         Reference *reference = nullptr) {
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
  auto q = prepare_projection(f, width, hidden, storage, 173);
  auto k = prepare_projection(f, width, hidden, storage, 179);
  auto v = prepare_projection(f, width, hidden, storage, 181);
  auto fa = prepare_projection(f, depth, hidden, storage, 191);
  auto fb = prepare_projection(f, width, depth, storage, 193);
  auto b = prepare_projection(f, heads, hidden, beta_storage, 197);
  std::optional<ProjectionWeight> ga;
  if (!kimi) ga = prepare_projection(f, depth, hidden, storage, 199);
  auto g = prepare_projection(f, width, kimi ? hidden : depth, storage, 211);
  auto out = prepare_projection(f, hidden, width, storage, 223);
  d5_qwen38_gdn_fixture::ConvFixture conv(heads, heads);
  for (std::size_t i = 0; i < conv.weights.size(); ++i)
    conv.weights[i] += float(int(i % 7) - 3) * 0.000137F;
  const auto conv_weight = parameter(f, conv.weights_desc, conv.weights, official && !kimi);
  // The fixed source may store padded A_log. The append consumes only heads.
  std::vector<float> a_source(official && kimi ? 128 : heads), alpha(heads);
  for (std::size_t h = 0; h < a_source.size(); ++h) {
    const float log = float(int(h % 7) - 3) / 8.0F;
    a_source[h] = official ? (h < std::size_t(heads) ? log : 19.0F) : -std::exp(log);
    if (h < std::size_t(heads)) alpha[h] = std::exp(log);
  }
  const auto a = f.constant(dense({std::int64_t(a_source.size()), 1}), bytes(a_source));
  std::vector<float> dt(width);
  for (std::size_t i = 0; i < dt.size(); ++i) dt[i] = float(int(i % 11) - 5) / 32.0F;
  const auto dt_value = f.constant(dense({1, 1, heads, depth}), bytes(dt));
  auto norm_values = sem::oracle::scale(depth);
  for (std::size_t i = 0; i < norm_values.size(); ++i) norm_values[i] += float(i % 5) * 0.0037F;
  const auto norm_weight = parameter(f, dense({depth}), norm_values, official && !kimi);
  const KdaParameters parameters{q.value, k.value, v.value, fa.value, fb.value,
      b.value, g.value, out.value, conv_weight, a, dt_value, norm_weight,
      ga ? std::optional<ValueId>(ga->value) : std::nullopt,
      official ? DecayEncoding::kLog : DecayEncoding::kNegativeExp};
  const auto ports = append_kda(f, input, history, state_input, parameters,
      {hidden, heads, depth, core::kda_l2_epsilon(profile), core::kKdaNormEpsilon});
  const auto outputs = ports.outputs();
  f.outputs.assign(outputs.begin(), outputs.end());
  f.state_ports = {{1, 0}, {2, 2}};

  Reference evaluate = [=](std::span<const float> x,
      std::span<const float> prior_history,
      std::span<const float> prior_state) -> std::vector<std::vector<float>> {
    if (x.size() != std::size_t(hidden) || prior_history.size() != std::size_t(3 * channels) ||
        prior_state.size() != std::size_t(heads * depth * depth))
      throw std::runtime_error("KDA reference input extent");
    std::vector<float> state(prior_state.begin(), prior_state.end());
    std::vector<float> past(prior_history.begin(), prior_history.end());
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
        double sum = core::kda_l2_epsilon(profile);
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
    auto expected = sem::oracle::rms_norm_rows(output, norm_values, depth,
                                              core::kKdaNormEpsilon);
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] *= float(core::sigmoid(output_gate[i]));
    auto final = out.reference(expected);
    return {std::move(past), std::move(conv_values), std::move(state),
            std::move(output), std::move(expected), std::move(final)};
  };
  if (reference) {
    *reference = std::move(evaluate);
    return f;
  }
  std::vector<float> state(heads * depth * depth);
  for (std::size_t i = 0; i < state.size(); ++i)
    state[i] = float(std::sin(double(i) * 0.013) * 0.0005);
  auto initial_window = conv.samples(3);
  std::vector<float> past(initial_window.begin(), initial_window.begin() + 3 * channels);
  for (int step = 0; step < 8; ++step) {
    auto x = d5_gguf_test::activation(hidden, 3 + 14 * step);
    sem::Replay replay{{bytes(x), bytes(past), bytes(state)}, evaluate(x, past, state)};
    past = replay.outputs[0];
    state = replay.outputs[2];
    f.replays.push_back(std::move(replay));
  }
  return f;
}
} // namespace d5_kda_full_fragments
