#pragma once

#include "d5_moe_profile_cases.h"
#include "d5_kda_projection_cases.h"
#include "d5_qwen38_24t_semantic_fragments.h"

// Test consumers only: compose public operations, then compare mature Provider
// execution with an independent double-precision oracle. No product kernel lives
// here. Projection storage qualification remains separate from these F32 cores.
namespace d5_moe_semantic_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace profiles = d5_moe_profile_cases;
using sem::bytes;
using sem::dense;
using sem::Fragment;
using sem::ValueId;

inline ValueId reshape(Fragment &f, ValueId input,
                       std::initializer_list<std::int64_t> shape) {
  return f.one(c::Operation(c::View::reshape(
                   c::TensorShape::create({shape.begin(), shape.size()}).value())
                               .value()),
               {input});
}
inline ValueId unary(Fragment &f, c::UnaryKind kind, ValueId input) {
  return f.one(c::Operation(c::Unary{kind}), {input});
}
inline ValueId binary(Fragment &f, c::BinaryKind kind, ValueId a, ValueId b) {
  return f.one(c::Operation(c::Binary{kind}), {a, b});
}
inline ValueId filled(Fragment &f, const c::TensorDesc &desc, float value) {
  return f.constant(desc, bytes(std::vector<float>(*desc.element_count(), value)));
}
inline ValueId rows(Fragment &f, ValueId matrix, std::int64_t first,
                    std::int64_t count) {
  std::vector<std::int32_t> indices(count);
  for (std::int64_t i = 0; i < count; ++i)
    indices[i] = static_cast<std::int32_t>(first + i);
  auto ids = f.constant(dense({count}, c::DataType::kInt32), bytes(indices));
  return f.one(c::Operation(c::Gather{}), {matrix, ids});
}
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

inline Fragment kda_projection(const d5_kda_projection_cases::Case &role) {
  using Storage = d5_kda_projection_cases::Storage;
  if (role.storage == Storage::kBFloat16) {
    // Same public BF16 MatMul boundary and independent storage-rounding
    // oracle as the existing official Qwen consumer, with exact role width.
    auto f = sem::official_projection(
        {role.roles, role.reduction, role.columns, c::GgufQuantizedType::kQ8_0});
    f.profile = role.profile;
    f.region = 0x030601;
    return f;
  }
  const bool f32 = role.storage == Storage::kF32;
  Fragment f{.profile = role.profile, .region = 0x030601, .quantized = !f32};
  auto w = sem::weight(role.columns, role.reduction,
                       c::GgufQuantizedType::kQ8_0, f32, 173);
  auto input = f.input(dense({1, role.reduction}));
  auto stored = f.constant(w.description, std::move(w.storage));
  f.outputs = {f.one(c::Operation(c::MatMul{}), {input, stored})};
  for (int seed : {3, 19}) {
    auto activation = d5_gguf_test::activation(role.reduction, seed);
    f.replays.push_back({{bytes(activation)}, {sem::project(activation, w)}});
  }
  return f;
}

inline constexpr std::array kKdaProfiles{
    &profiles::kimi_k3_official, &profiles::kimi_k3_primary_gguf,
    &profiles::kimi_k3_fidelity_gguf, &profiles::glm53_flash_official,
    &profiles::glm53_flash_primary_gguf, &profiles::glm53_flash_fidelity_gguf};

// llama.cpp@4c9233c, kimi-k3.cpp build_kda_layer; Transformers@c1b80cf,
// glm5_next: bounded per-key decay, not Qwen's scalar Softplus gate. Preserve
// H=96/64, D=128, K=4. Only token count is reduced. Inputs include the causal
// projection window and post-projection gates; R03 checks their weight roles.
inline Fragment kda(const c::ModelArtifactProfileDescription *profile,
                     int steps = 8) {
  const std::int64_t heads = profile->model_name == "Kimi-K3" ? 96 : 64;
  constexpr std::int64_t depth = 128;
  const auto width = heads * depth;
  const auto channels = 3 * width;
  Fragment f{.profile = profile, .region = 0x060601};
  d5_qwen38_gdn_fixture::ConvFixture conv(heads, heads);
  auto samples = f.input(conv.samples_desc);
  auto raw_gate = f.input(dense({1, 1, heads, depth}));
  auto raw_beta = f.input(dense({1, 1, heads, 1}));
  auto initial = f.input(dense({1, heads, depth, depth}));
  auto out_gate = f.input(dense({1, 1, heads, depth}));
  auto weights = f.constant(conv.weights_desc, bytes(conv.weights));
  auto convolution = f.one(c::model_ops::causal_conv1d(), {samples, weights});
  auto ones = filled(f, f.description(convolution), 1.0F);
  auto silu = f.one(c::Operation(c::GatedActivation{
                        .inputs = c::GatedInputKind::kSeparate}),
                    {convolution, ones});
  auto channel_rows = reshape(f, silu, {channels, 1});
  auto query = reshape(f, rows(f, channel_rows, 0, width), {heads, depth});
  auto key = reshape(f, rows(f, channel_rows, width, width), {heads, depth});
  auto value = reshape(f, rows(f, channel_rows, 2 * width, width),
                        {1, 1, heads, depth});
  auto l2_scale = filled(f, dense({depth}), 1.0F / std::sqrt(float(depth)));
  const c::Operation l2(c::RmsNorm{.epsilon = 1.0e-6F / float(depth)});
  query = reshape(f, f.one(l2, {query, l2_scale}), {1, 1, heads, depth});
  key = reshape(f, f.one(l2, {key, l2_scale}), {1, 1, heads, depth});

  std::vector<float> alpha(width), dt(width);
  for (std::int64_t h = 0; h < heads; ++h)
    for (std::int64_t d = 0; d < depth; ++d) {
      // Exact F32 runtime constants. A_log / folded negative-A storage is
      // a separate projection/parameter boundary, not an artifact-wide dtype.
      alpha[h * depth + d] = 0.5F + float(h % 7) / 8.0F;
      dt[h * depth + d] = float(d % 11 - 5) / 32.0F;
    }
  const auto gate_desc = f.description(raw_gate);
  auto a = f.constant(gate_desc, bytes(alpha));
  auto bias = f.constant(gate_desc, bytes(dt));
  auto bounded = unary(f, c::UnaryKind::kSigmoid,
                       binary(f, c::BinaryKind::kMultiply, a,
                              binary(f, c::BinaryKind::kAdd, raw_gate, bias)));
  auto lower = filled(f, gate_desc, -5.0F);
  auto decay = binary(f, c::BinaryKind::kMultiply, bounded, lower);
  auto beta = unary(f, c::UnaryKind::kSigmoid, raw_beta);
  auto recurrent = f.append(c::Operation(c::GatedDeltaNet{}),
                            {query, key, value, decay, beta, initial}, 2);
  auto norm_values = sem::oracle::scale(depth);
  auto norm = f.constant(dense({depth}), bytes(norm_values));
  auto normalized = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                          {recurrent[0], norm});
  auto result = binary(f, c::BinaryKind::kMultiply, normalized,
                       unary(f, c::UnaryKind::kSigmoid, out_gate));
  f.outputs = {convolution, recurrent[0], recurrent[1], result};
  f.state_ports = {{3, 2}};

  std::vector<float> state(heads * depth * depth);
  for (std::size_t i = 0; i < state.size(); ++i)
    state[i] = float(std::sin(double(i) * 0.013) * 0.0005);
  auto window = conv.samples(3);
  for (int step = 0; step < steps; ++step) {
    if (step) {
      std::move(window.begin() + channels, window.end(), window.begin());
      auto fresh = conv.samples(3 + 14 * step);
      std::copy(fresh.end() - channels, fresh.end(), window.end() - channels);
    }
    auto conv_values = sem::as_float(conv.oracle(window));
    std::vector<float> activated(channels), gate(width), z(width), b(heads);
    for (std::size_t i = 0; i < activated.size(); ++i)
      activated[i] = float(conv_values[i] * sigmoid(conv_values[i]));
    for (std::int64_t i = 0; i < width; ++i) {
      gate[i] = float((i * 7 + step * 13) % 81 - 40) / 4.0F;
      z[i] = float((i * 11 + step * 17) % 31 - 15) / 8.0F;
    }
    for (std::int64_t h = 0; h < heads; ++h)
      b[h] = float((h * 3 + step * 5) % 19 - 9) / 4.0F;
    std::array<std::vector<float>, 2> qk{
        std::vector<float>(width), std::vector<float>(width)};
    for (std::int64_t p = 0; p < 2; ++p)
      for (std::int64_t h = 0; h < heads; ++h) {
        double sum = 1.0e-6;
        for (std::int64_t d = 0; d < depth; ++d) {
          double v = activated[p * width + h * depth + d];
          sum += v * v;
        }
        for (std::int64_t d = 0; d < depth; ++d)
          qk[p][h * depth + d] =
              float(activated[p * width + h * depth + d] / std::sqrt(sum));
      }
    sem::Replay replay{{bytes(window), bytes(gate), bytes(b), bytes(state),
                         bytes(z)}, {}};
    std::vector<float> output(width);
    for (std::int64_t h = 0; h < heads; ++h) {
      std::array<double, depth> decay_values;
      for (std::int64_t k = 0; k < depth; ++k) {
        const auto i = h * depth + k;
        decay_values[k] = std::exp(-5.0 * sigmoid(
            double(alpha[i]) * (double(gate[i]) + dt[i])));
      }
      for (std::int64_t v = 0; v < depth; ++v) {
        const auto row = (h * depth + v) * depth;
        std::array<double, depth> decayed;
        double projected = 0;
        for (std::int64_t k = 0; k < depth; ++k) {
          decayed[k] = state[row + k] * decay_values[k];
          projected += decayed[k] * qk[1][h * depth + k];
        }
        const double delta = sigmoid(b[h]) *
            (activated[2 * width + h * depth + v] - projected);
        double score = 0;
        for (std::int64_t k = 0; k < depth; ++k) {
          const double next = decayed[k] + delta * qk[1][h * depth + k];
          state[row + k] = float(next);
          score += next * qk[0][h * depth + k];
        }
        output[h * depth + v] = float(score / std::sqrt(double(depth)));
      }
    }
    auto gated = sem::oracle::rms_norm_rows(output, norm_values, depth, 1.0e-6F);
    for (std::size_t i = 0; i < gated.size(); ++i)
      gated[i] *= float(sigmoid(z[i]));
    replay.outputs = {std::move(conv_values), std::move(output), state,
                       std::move(gated)};
    f.replays.push_back(std::move(replay));
  }
  return f;
}

// Kimi SiTU is not ordinary SwiGLU. Pinned llama.cpp kimi-k3.cpp uses
// 4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25).
inline Fragment situ(const c::ModelArtifactProfileDescription *profile) {
  constexpr std::int64_t width = 3584;
  Fragment f{.profile = profile, .region = 0x020601};
  const auto desc = dense({1, width});
  auto gate = f.input(desc), up = f.input(desc);
  auto four = filled(f, desc, 4.0F), quarter = filled(f, desc, 0.25F);
  auto twenty_five = filled(f, desc, 25.0F), inverse = filled(f, desc, 0.04F);
  auto bounded_gate = binary(f, c::BinaryKind::kMultiply, four,
      unary(f, c::UnaryKind::kTanh,
            binary(f, c::BinaryKind::kMultiply, gate, quarter)));
  auto bounded_up = binary(f, c::BinaryKind::kMultiply, twenty_five,
      unary(f, c::UnaryKind::kTanh,
            binary(f, c::BinaryKind::kMultiply, up, inverse)));
  f.outputs = {binary(f, c::BinaryKind::kMultiply, bounded_up,
      binary(f, c::BinaryKind::kMultiply, bounded_gate,
             unary(f, c::UnaryKind::kSigmoid, gate)))};
  for (int seed : {3, 19}) {
    std::vector<float> g(width), u(width), out(width);
    for (std::int64_t i = 0; i < width; ++i) {
      g[i] = float((i * 7 + seed) % 257 - 128) / 4.0F;
      u[i] = float((i * 13 + seed) % 257 - 128) / 2.0F;
      out[i] = float(4.0 * std::tanh(g[i] / 4.0) * sigmoid(g[i]) *
                     25.0 * std::tanh(u[i] / 25.0));
    }
    f.replays.push_back({{bytes(g), bytes(u)}, {std::move(out)}});
  }
  return f;
}

struct HyperProfile final {
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t hidden;
};
inline constexpr std::array kHyperProfiles{
    HyperProfile{&profiles::deepseek_v4_pro_official, 7168},
    HyperProfile{&profiles::deepseek_v4_flash_official, 4096},
    HyperProfile{&profiles::deepseek_v4_flash_primary_gguf, 4096},
    HyperProfile{&profiles::deepseek_v4_flash_fidelity_gguf, 4096},
    HyperProfile{&profiles::glm53_flash_official, 4096},
    HyperProfile{&profiles::glm53_flash_primary_gguf, 4096},
    HyperProfile{&profiles::glm53_flash_fidelity_gguf, 4096}};

// Native mHC core at the post-projection boundary: affine pre/post gates,
// Sinkhorn(20), collapse and expansion with the original residual. The branch
// input is independently supplied (an attention/FFN consumer), not identity.
// Projection weights and the enclosed attention/FFN have separate receipts.
inline Fragment hyper_connection(const HyperProfile &profile) {
  Fragment f{.profile = profile.profile, .region = 0x0c0601};
  const auto hidden = profile.hidden;
  auto mixes = f.input(dense({1, 24}));
  auto residual = f.input(dense({1, 4, hidden}));
  auto branch = f.input(dense({1, hidden}));
  const std::vector<float> scales{0.125F, 0.25F, 0.5F};
  std::vector<float> bases(24);
  for (int i = 0; i < 24; ++i)
    bases[i] = float(i % 7 - 3) / 8.0F;
  auto scale = f.constant(dense({3}), bytes(scales));
  auto base = f.constant(dense({24}), bytes(bases));
  auto comb = f.one(c::model_ops::hyper_connection_combine(1.0e-6F, 20),
                    {mixes, scale, base});
  auto mixes_rows = reshape(f, mixes, {24, 1});
  const auto gate = [&](int start, float multiplier, float offset) {
    auto x = reshape(f, rows(f, mixes_rows, start, 4), {1, 4});
    auto s = filled(f, dense({1, 4}), scales[start / 4]);
    auto b = f.constant(dense({1, 4}), bytes(std::vector<float>(
        bases.begin() + start, bases.begin() + start + 4)));
    auto y = unary(f, c::UnaryKind::kSigmoid,
                    binary(f, c::BinaryKind::kAdd,
                           binary(f, c::BinaryKind::kMultiply, x, s), b));
    if (multiplier != 1.0F)
      y = binary(f, c::BinaryKind::kMultiply, y,
                  filled(f, dense({1, 4}), multiplier));
    if (offset != 0.0F)
      y = binary(f, c::BinaryKind::kAdd, y,
                  filled(f, dense({1, 4}), offset));
    return y;
  };
  auto pre = gate(0, 1.0F, 1.0e-6F);
  auto post = gate(4, 2.0F, 0.0F);
  auto collapsed = f.one(c::model_ops::hyper_connection_pre(), {residual, pre});
  auto expanded = f.one(c::model_ops::hyper_connection_post(),
                        {branch, residual, post, comb});
  f.outputs = {comb, collapsed, expanded};
  for (int seed : {3, 19}) {
    auto x = d5_gguf_test::values(4, hidden, seed);
    auto branch_values = d5_gguf_test::activation(hidden, seed + 5);
    std::vector<float> m(24), collapsed_values(hidden), expanded_values(4 * hidden);
    for (int i = 0; i < 24; ++i)
      m[i] = float((i * 11 + seed) % 31 - 15) / 4.0F;
    std::array<double, 16> matrix;
    for (int row = 0; row < 4; ++row) {
      double maximum = -INFINITY, sum = 0;
      for (int col = 0; col < 4; ++col) {
        const int i = row * 4 + col;
        matrix[i] = m[8 + i] * double(scales[2]) + bases[8 + i];
        maximum = std::max(maximum, matrix[i]);
      }
      for (int col = 0; col < 4; ++col) {
        auto &v = matrix[row * 4 + col];
        v = std::exp(v - maximum);
        sum += v;
      }
      for (int col = 0; col < 4; ++col)
        matrix[row * 4 + col] = matrix[row * 4 + col] / sum + 1.0e-6;
    }
    const auto normalize = [&](bool columns) {
      for (int i = 0; i < 4; ++i) {
        double sum = 1.0e-6;
        for (int j = 0; j < 4; ++j)
          sum += matrix[columns ? j * 4 + i : i * 4 + j];
        for (int j = 0; j < 4; ++j)
          matrix[columns ? j * 4 + i : i * 4 + j] /= sum;
      }
    };
    normalize(true);
    for (int iteration = 1; iteration < 20; ++iteration) {
      normalize(false);
      normalize(true);
    }
    for (std::int64_t d = 0; d < hidden; ++d) {
      double sum = 0;
      for (int src = 0; src < 4; ++src)
        sum += x[src * hidden + d] *
            (sigmoid(m[src] * double(scales[0]) + bases[src]) + 1.0e-6);
      collapsed_values[d] = float(sum);
      for (int dst = 0; dst < 4; ++dst) {
        double expanded_value = branch_values[d] * 2.0 *
            sigmoid(m[4 + dst] * double(scales[1]) + bases[4 + dst]);
        for (int src = 0; src < 4; ++src)
          expanded_value += x[src * hidden + d] * matrix[src * 4 + dst];
        expanded_values[dst * hidden + d] = float(expanded_value);
      }
    }
    f.replays.push_back({{bytes(m), bytes(x), bytes(branch_values)},
                         {sem::as_float(matrix), std::move(collapsed_values),
                          std::move(expanded_values)}});
  }
  return f;
}
} // namespace d5_moe_semantic_fragments
