#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "d5_qwen35_gguf_fragment_support.h"
#include "d5_qwen38_24t_full_router.h"
#include "d5_qwen38_24t_moe_profiles.h"

// Test-only recipes. Direct consumes Compute operations and inferred tensor
// descriptions; Graph independently lowers the same declared fragment. The
// recipes contain no route executor or product implementation.
namespace d5_qwen38_24t_semantic_fragments {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace q = d5_qwen38_24t_fixture;
namespace recipe = d5_qwen38_24t_full_router;
namespace oracle = d5_qwen35_gguf_fragment;
using ValueId = recipe::ValueId;
using ValueKind = recipe::ValueKind;

struct Replay final {
  std::vector<std::vector<std::byte>> inputs;
  std::vector<std::vector<float>> outputs;
};

struct Fragment final {
  const c::ModelArtifactProfileDescription *profile;
  std::uint64_t region;
  bool quantized = false;
  std::vector<c::TensorDesc> descriptions;
  std::vector<ValueId> inputs, constants, outputs;
  std::vector<std::vector<std::byte>> constant_bytes;
  std::vector<recipe::DirectNode> nodes;
  std::vector<Replay> replays;

  const c::TensorDesc &description(ValueId id) const {
    return descriptions[id.ordinal()];
  }
  ValueId value(c::TensorDesc desc, ValueKind kind) {
    ValueId id{static_cast<std::uint32_t>(descriptions.size()), kind};
    descriptions.push_back(std::move(desc));
    return id;
  }
  ValueId input(c::TensorDesc desc) {
    auto id = value(std::move(desc), ValueKind::kInput);
    inputs.push_back(id);
    return id;
  }
  ValueId constant(c::TensorDesc desc, std::vector<std::byte> bytes) {
    auto id = value(std::move(desc), ValueKind::kConstant);
    constants.push_back(id);
    constant_bytes.push_back(std::move(bytes));
    return id;
  }
  std::vector<ValueId> append(c::Operation operation,
                               std::initializer_list<ValueId> ports,
                               std::size_t count = 1) {
    std::vector<c::TensorDesc> in, out(count);
    for (auto id : ports)
      in.push_back(description(id));
    auto status = c::infer_output_descriptions(operation, in, out);
    if (!status.ok())
      throw std::runtime_error(std::string(status.message()));
    recipe::DirectNode node{operation, ports, {}};
    for (auto &desc : out)
      node.outputs.push_back(value(std::move(desc), ValueKind::kNodeOutput));
    auto ids = node.outputs;
    nodes.push_back(std::move(node));
    return ids;
  }
  ValueId one(c::Operation operation, std::initializer_list<ValueId> ports) {
    return append(operation, ports)[0];
  }
};

template <class T> std::vector<std::byte> bytes(std::span<const T> values) {
  const auto data = std::as_bytes(values);
  return {data.begin(), data.end()};
}
template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  return bytes<T>(std::span<const T>(values));
}
inline c::TensorDesc dense(std::initializer_list<std::int64_t> shape,
                            c::DataType type = c::DataType::kFloat32) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
inline std::vector<float> as_float(std::span<const double> values) {
  std::vector<float> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(),
                   [](double value) { return static_cast<float>(value); });
  return result;
}

struct GraphCase final {
  g::Graph graph;
  std::vector<g::ValueId> inputs, constants, outputs;
};
inline GraphCase make_graph(const Fragment &fragment) {
  auto builder = g::GraphBuilder::create().value();
  std::vector<g::ValueId> values(fragment.descriptions.size());
  GraphCase result;
  for (auto id : fragment.inputs) {
    values[id.ordinal()] = builder.input(fragment.description(id)).value();
    result.inputs.push_back(values[id.ordinal()]);
  }
  for (auto id : fragment.constants) {
    values[id.ordinal()] = builder.constant(fragment.description(id)).value();
    result.constants.push_back(values[id.ordinal()]);
  }
  for (const auto &step : fragment.nodes) {
    std::vector<g::ValueId> inputs;
    for (auto id : step.inputs)
      inputs.push_back(values[id.ordinal()]);
    auto node = builder.add_node(step.operation, inputs).value();
    for (std::size_t i = 0; i < step.outputs.size(); ++i)
      values[step.outputs[i].ordinal()] = builder.result(node, i).value();
  }
  for (auto id : fragment.outputs) {
    auto output = values[id.ordinal()];
    if (!builder.publish(output).ok())
      throw std::runtime_error("semantic fragment publication failed");
    result.outputs.push_back(output);
  }
  result.graph = std::move(builder).finish().value();
  return result;
}

// Every header row below is from unsloth/Qwen3.8-2.4T-A95B-GGUF@567d3e6.
// Storage is a tensor-role property, including F32 in the IQ4 artifact.
struct Projection final {
  std::string_view tensor;
  std::int64_t reduction, columns;
  c::GgufQuantizedType primary;
  bool primary_f32 = false;
};
inline constexpr std::array kProjections{
    Projection{"blk.0.attn_qkv.weight", 8192, 20480, c::GgufQuantizedType::kQ5K},
    Projection{"blk.0.attn_gate.weight", 8192, 16384, c::GgufQuantizedType::kQ5K},
    Projection{"blk.0.ssm_alpha.weight", 8192, 128, c::GgufQuantizedType::kQ8_0, true},
    Projection{"blk.0.ssm_beta.weight", 8192, 128, c::GgufQuantizedType::kQ8_0, true},
    Projection{"blk.0.ssm_out.weight", 16384, 8192, c::GgufQuantizedType::kQ6K},
    Projection{"blk.3.attn_q.weight", 8192, 32768, c::GgufQuantizedType::kQ5K},
    Projection{"blk.3.attn_k.weight", 8192, 1024, c::GgufQuantizedType::kQ5K},
    Projection{"blk.3.attn_v.weight", 8192, 1024, c::GgufQuantizedType::kQ5K},
    Projection{"blk.3.attn_output.weight", 16384, 8192, c::GgufQuantizedType::kQ5K}};

struct Weight final {
  c::TensorDesc description;
  std::vector<std::byte> storage;
  std::vector<std::vector<std::pair<std::size_t, float>>> decoded;
};
// Pack bounded groups of rows. Keep the full physical matrix on the route,
// but never materialize an extra gigabyte-sized decoded host copy.
inline Weight weight(std::int64_t columns, std::int64_t reduction,
                       c::GgufQuantizedType type, bool f32, int seed) {
  auto desc = f32 ? dense({columns, reduction}).transposed().value()
                  : c::gguf_weight_description(
                        type, c::DataType::kFloat32, reduction, columns).value();
  Weight result{desc, std::vector<std::byte>(desc.required_bytes().value()),
                std::vector<std::vector<std::pair<std::size_t, float>>>(columns)};
  constexpr std::int64_t chunk = 32;
  for (std::int64_t first = 0; first < columns; first += chunk) {
    const auto rows = std::min(chunk, columns - first);
    std::vector<float> physical(rows * reduction);
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto column = (17 * (first + row) + seed) % reduction;
      physical[row * reduction + column] =
          0.5F + static_cast<float>((first + row + seed) % 13) / 32.0F;
    }
    const auto row_bytes = result.storage.size() / columns;
    std::vector<std::byte> packed;
    std::vector<float> decoded;
    if (f32) {
      packed = bytes(physical);
      decoded = std::move(physical);
    } else {
      auto native = d5_gguf_test::pack(type, physical, rows, reduction,
                                       rows * row_bytes);
      packed = std::move(native.bytes);
      decoded = std::move(native.decoded);
    }
    std::copy(packed.begin(), packed.end(),
                result.storage.begin() + first * row_bytes);
    for (std::int64_t row = 0; row < rows; ++row)
      for (std::int64_t column = 0; column < reduction; ++column) {
        auto value = decoded[row * reduction + column];
        if (value != 0.0F)
          result.decoded[first + row].emplace_back(column, value);
      }
  }
  return result;
}
inline std::vector<float> project(std::span<const float> input,
                                   const Weight &weight) {
  std::vector<float> result(weight.decoded.size());
  for (std::size_t row = 0; row < result.size(); ++row) {
    double sum = 0;
    for (auto [column, value] : weight.decoded[row])
      sum += static_cast<double>(input[column]) * value;
    result[row] = static_cast<float>(sum);
  }
  return result;
}
inline bool primary(const c::ModelArtifactProfileDescription *profile) {
  return profile == &d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile;
}
inline Fragment projection(const c::ModelArtifactProfileDescription *profile,
                              const Projection &role) {
  const bool f32 = primary(profile) && role.primary_f32;
  Fragment f{.profile = profile, .region = 0x040521, .quantized = !f32};
  auto w = weight(role.columns, role.reduction,
                    primary(profile) ? role.primary : c::GgufQuantizedType::kQ8_0,
                    f32, 173);
  auto input = f.input(dense({1, role.reduction}));
  auto stored = f.constant(w.description, std::move(w.storage));
  f.outputs = {f.one(c::Operation(c::MatMul{}), {input, stored})};
  for (int seed : {3, 19}) {
    auto activation = d5_gguf_test::activation(role.reduction, seed);
    f.replays.push_back({{bytes(activation)}, {project(activation, w)}});
  }
  return f;
}

inline Fragment mtp(const c::ModelArtifactProfileDescription *profile) {
  Fragment f{.profile = profile, .region = 0x0d0e21, .quantized = true};
  // Both deployment artifacts use Q8_0 for blk.92.nextn.eh_proj.weight.
  auto w = weight(q::kHidden, 2 * q::kHidden, c::GgufQuantizedType::kQ8_0,
                    false, 179);
  auto a = f.input(dense({1, q::kHidden}));
  auto b = f.input(dense({1, q::kHidden}));
  const auto a_scale = oracle::scale(q::kHidden);
  auto b_scale = oracle::scale(q::kHidden);
  std::reverse(b_scale.begin(), b_scale.end());
  auto sa = f.constant(dense({q::kHidden}), bytes(a_scale));
  auto sb = f.constant(dense({q::kHidden}), bytes(b_scale));
  auto weight_id = f.constant(w.description, std::move(w.storage));
  auto na = f.one(c::Operation(c::RmsNorm{}), {a, sa});
  auto nb = f.one(c::Operation(c::RmsNorm{}), {b, sb});
  auto joined = f.one(c::Operation(c::Concat{.axis = 1}), {na, nb});
  auto projected = f.one(c::Operation(c::MatMul{}), {joined, weight_id});
  auto normalized = f.one(c::Operation(c::RmsNorm{}), {projected, sa});
  f.outputs = {joined, projected, normalized};
  for (int seed : {5, 23}) {
    auto first = d5_gguf_test::activation(q::kHidden, seed);
    auto second = d5_gguf_test::activation(q::kHidden, seed + 1);
    auto joined_values = oracle::rms_norm(first, a_scale);
    auto tail = oracle::rms_norm(second, b_scale);
    joined_values.insert(joined_values.end(), tail.begin(), tail.end());
    auto product = project(joined_values, w);
    f.replays.push_back({{bytes(first), bytes(second)},
                          {joined_values, product, oracle::rms_norm(product, a_scale)}});
  }
  return f;
}

inline std::vector<float> rope(std::span<const float> input,
                                std::size_t heads) {
  auto output = std::vector<float>(input.begin(), input.end());
  for (std::size_t token = 0; token < 2; ++token)
    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t pair = 0; pair < 32; ++pair) {
        const auto offset = (token * heads + head) * 256 + pair * 2;
        const float angle = static_cast<float>(11 + token) / float(pair + 1);
        output[offset] = input[offset] * std::cos(angle) - input[offset + 1] * std::sin(angle);
        output[offset + 1] = input[offset] * std::sin(angle) + input[offset + 1] * std::cos(angle);
      }
  return output;
}
inline Fragment attention(const c::ModelArtifactProfileDescription *profile) {
  Fragment f{.profile = profile, .region = 0x040421};
  auto query = f.input(dense({1, 2, q::kQueryHeads, 256}));
  auto key = f.input(dense({1, 2, q::kKvHeads, 256}));
  auto value = f.input(dense({1, 2, q::kKvHeads, 256}));
  auto gate = f.input(dense({1, 2, q::kQueryHeads, 256}));
  auto scale = oracle::scale(256);
  auto norm_weight = f.constant(dense({256}), bytes(scale));
  std::vector<std::int32_t> positions{11, 12};
  auto pos = f.constant(dense({2}, c::DataType::kInt32), bytes(positions));
  std::vector<float> frequencies(32);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / float(i + 1);
  auto freq = f.constant(dense({32}), bytes(frequencies));
  auto nq = f.one(c::Operation(c::RmsNorm{}), {query, norm_weight});
  auto nk = f.one(c::Operation(c::RmsNorm{}), {key, norm_weight});
  const c::Operation rotate(c::RoPE{.rotary_dimensions = 64, .sequence_axis = 1,
                                    .layout = c::RopeLayout::kInterleaved});
  auto rq = f.one(rotate, {nq, pos, freq});
  auto rk = f.one(rotate, {nk, pos, freq});
  auto attended = f.one(c::Operation(c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
      .scale = 0.0625F, .causal = true,
      .accumulation_type = c::DataType::kFloat32}), {rq, rk, value});
  auto probability = f.one(c::Operation(c::Unary{c::UnaryKind::kSigmoid}), {gate});
  auto gated = f.one(c::Operation(c::Binary{c::BinaryKind::kMultiply}),
                      {attended, probability});
  // Publish intermediate Q/K so one-token/trivial attention cannot hide a
  // broken norm or RoPE. Two causal positions exercise nontrivial softmax.
  f.outputs = {nq, nk, rq, rk, attended, gated};
  for (int seed : {7, 29}) {
    auto queries = d5_gguf_test::values(2 * q::kQueryHeads, 256, seed);
    auto keys = d5_gguf_test::values(2 * q::kKvHeads, 256, seed + 1);
    auto values = d5_gguf_test::values(2 * q::kKvHeads, 256, seed + 2);
    auto gates = d5_gguf_test::values(2 * q::kQueryHeads, 256, seed + 3);
    auto normalized_q = oracle::rms_norm_rows(queries, scale, 256);
    auto normalized_k = oracle::rms_norm_rows(keys, scale, 256);
    auto rotated_q = rope(normalized_q, q::kQueryHeads);
    auto rotated_k = rope(normalized_k, q::kKvHeads);
    std::vector<float> scores(queries.size()), output(queries.size());
    for (std::size_t token = 0; token < 2; ++token)
      for (std::size_t head = 0; head < q::kQueryHeads; ++head) {
        const auto kv_head = head / (q::kQueryHeads / q::kKvHeads);
        const auto offset = (token * q::kQueryHeads + head) * 256;
        std::array<double, 2> logits{};
        for (std::size_t t = 0; t <= token; ++t)
          for (std::size_t d = 0; d < 256; ++d)
            logits[t] += double(rotated_q[offset + d]) *
                         rotated_k[(t * q::kKvHeads + kv_head) * 256 + d] / 16.0;
        const double maximum = token ? std::max(logits[0], logits[1]) : logits[0];
        double sum = 0;
        for (std::size_t t = 0; t <= token; ++t)
          sum += logits[t] = std::exp(logits[t] - maximum);
        for (std::size_t d = 0; d < 256; ++d) {
          double v = 0;
          for (std::size_t t = 0; t <= token; ++t)
            v += logits[t] / sum * values[(t * q::kKvHeads + kv_head) * 256 + d];
          scores[offset + d] = float(v);
          output[offset + d] = float(v / (1.0 + std::exp(-double(gates[offset + d]))));
        }
      }
    f.replays.push_back({{bytes(queries), bytes(keys), bytes(values), bytes(gates)},
                          {normalized_q, normalized_k, rotated_q, rotated_k, scores, output}});
  }
  return f;
}

inline Fragment gdn(const c::ModelArtifactProfileDescription *profile) {
  namespace gd = d5_qwen38_gdn_fixture;
  Fragment f{.profile = profile, .region = 0x050521};
  gd::ConvFixture conv(q::kGdnKeyHeads, q::kGdnValueHeads);
  auto samples = f.input(conv.samples_desc);
  auto weight = f.constant(conv.weights_desc, bytes(conv.weights));
  auto convolution = f.one(c::model_ops::causal_conv1d(), {samples, weight});
  gd::Fixture fixture(0, q::kGdnKeyHeads, q::kGdnValueHeads);
  std::array<ValueId, 6> ports;
  for (std::size_t i = 0; i < ports.size(); ++i)
    ports[i] = f.input(fixture.inputs[i]);
  auto outputs = f.append(c::Operation(c::GatedDeltaNet{}),
                           {ports[0], ports[1], ports[2], ports[3], ports[4], ports[5]}, 2);
  f.outputs = {convolution, outputs[0], outputs[1]};
  std::vector<float> state;
  for (int seed : {3, 17}) {
    gd::Fixture replay(seed, q::kGdnKeyHeads, q::kGdnValueHeads);
    if (!state.empty())
      replay.data[5] = state;
    auto samples_values = conv.samples(seed);
    Replay expected{{bytes(samples_values)}, {}};
    for (auto &input : replay.data)
      expected.inputs.push_back(bytes(input));
    const auto conv_output = conv.oracle(samples_values);
    expected.outputs.push_back(as_float(conv_output));
    const auto recurrent = replay.oracle();
    for (const auto &output : recurrent)
      expected.outputs.push_back(as_float(output));
    state = expected.outputs.back();
    f.replays.push_back(std::move(expected));
  }
  return f;
}

inline void expect_output(const Fragment &fragment, std::span<const float> actual,
                            std::span<const float> expected) {
  if (fragment.quantized)
    d5_qwen38_24t_moe_profiles::expect_output(fragment.profile, actual, expected);
  else
    oracle::expect_close(actual, expected, 8.0e-5F, 4.0e-5F);
}
inline constexpr std::array kProfiles{
    &d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
    &d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile};
} // namespace d5_qwen38_24t_semantic_fragments
