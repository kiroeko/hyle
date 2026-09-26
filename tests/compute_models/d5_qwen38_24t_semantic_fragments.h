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
  std::vector<std::vector<float>> native_outputs{};
};

struct Fragment final {
  const c::ModelArtifactProfileDescription *profile;
  std::uint64_t region;
  bool quantized = false;
  float absolute_tolerance = 8.0e-5F;
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
    Projection{"blk.0.attn_qkv.weight", 8192, 20480,
               c::GgufQuantizedType::kQ5K},
    Projection{"blk.0.attn_gate.weight", 8192, 16384,
               c::GgufQuantizedType::kQ5K},
    Projection{"blk.0.ssm_alpha.weight", 8192, 128, c::GgufQuantizedType::kQ8_0,
               true},
    Projection{"blk.0.ssm_beta.weight", 8192, 128, c::GgufQuantizedType::kQ8_0,
               true},
    Projection{"blk.0.ssm_out.weight", 16384, 8192, c::GgufQuantizedType::kQ6K},
    Projection{"blk.3.attn_q.weight", 8192, 32768, c::GgufQuantizedType::kQ5K},
    Projection{"blk.3.attn_k.weight", 8192, 1024, c::GgufQuantizedType::kQ5K},
    Projection{"blk.3.attn_v.weight", 8192, 1024, c::GgufQuantizedType::kQ5K},
    Projection{"blk.3.attn_output.weight", 16384, 8192,
               c::GgufQuantizedType::kQ5K}};

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
                  : c::gguf_weight_description(type, c::DataType::kFloat32,
                                               reduction, columns)
                        .value();
  Weight result{
      desc, std::vector<std::byte>(desc.required_bytes().value()),
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
      auto native =
          d5_gguf_test::pack(type, physical, rows, reduction, rows * row_bytes);
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

// The official checkpoint stores every projection in BF16, independently
// of the GGUF role's format. Keep both model input and output storage BF16;
// the final cast only makes the test's observed result uniformly F32.
inline Fragment official_projection(const Projection &role) {
  Fragment f{.profile = &q::kOfficialProfile,
             .region = 0x040522,
             .absolute_tolerance = 2.0e-5F};
  auto input = f.input(dense({1, role.reduction}, c::DataType::kBFloat16));
  const auto desc =
      dense({role.columns, role.reduction}, c::DataType::kBFloat16)
          .transposed()
          .value();
  constexpr int seed = 181;
  auto source =
      q::sparse_output_major_weight(role.columns, role.reduction, seed);
  auto stored = f.constant(desc, bytes(source));
  auto product = f.one(c::Operation(c::MatMul{}), {input, stored});
  f.outputs = {f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {product})};
  for (int replay : {3, 19}) {
    auto activation = q::values(role.reduction, replay);
    auto projected = q::sparse_projection(activation, role.columns, seed);
    // Sparse single-coefficient products retain an exact independent BF16
    // storage oracle without a second full F32 copy of the weight matrix.
    std::vector<float> expected;
    expected.reserve(projected.size());
    for (double value : projected)
      expected.push_back(static_cast<float>(q::decode(q::encode(value))));
    f.replays.push_back({{bytes(activation)}, {std::move(expected)}});
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
    f.replays.push_back(
        {{bytes(first), bytes(second)},
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
        output[offset] = input[offset] * std::cos(angle) -
                         input[offset + 1] * std::sin(angle);
        output[offset + 1] = input[offset] * std::sin(angle) +
                             input[offset + 1] * std::cos(angle);
      }
  return output;
}
inline std::array<std::vector<float>, 2>
attention_reference(std::span<const float> query, std::span<const float> key,
                    std::span<const float> value, std::span<const float> gates,
                    bool half_products) {
  const auto rounded = [&](double value) {
    return half_products ? half_test::decode(
                               half_test::encode<hyle::simd::float16_t>(value))
                         : value;
  };
  std::vector<float> scores(query.size()), output(query.size());
  for (std::size_t token = 0; token < 2; ++token)
    for (std::size_t head = 0; head < q::kQueryHeads; ++head) {
      const auto kv_head = head / (q::kQueryHeads / q::kKvHeads);
      const auto offset = (token * q::kQueryHeads + head) * 256;
      std::array<double, 2> logits{};
      for (std::size_t t = 0; t <= token; ++t)
        for (std::size_t d = 0; d < 256; ++d)
          logits[t] += rounded(query[offset + d]) *
                       rounded(key[(t * q::kKvHeads + kv_head) * 256 + d]) /
                       16.0;
      const double maximum = token ? std::max(logits[0], logits[1]) : logits[0];
      double sum = 0;
      for (std::size_t t = 0; t <= token; ++t)
        sum += logits[t] = std::exp(logits[t] - maximum);
      for (std::size_t d = 0; d < 256; ++d) {
        double v = 0;
        for (std::size_t t = 0; t <= token; ++t)
          v += rounded(logits[t] / sum) *
               rounded(value[(t * q::kKvHeads + kv_head) * 256 + d]);
        scores[offset + d] = float(v);
        output[offset + d] =
            float(v / (1.0 + std::exp(-double(gates[offset + d]))));
      }
    }
  return {std::move(scores), std::move(output)};
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
  const c::Operation rotate(c::RoPE{.rotary_dimensions = 64,
                                    .sequence_axis = 1,
                                    .layout = c::RopeLayout::kInterleaved});
  auto rq = f.one(rotate, {nq, pos, freq});
  auto rk = f.one(rotate, {nk, pos, freq});
  auto attended =
      f.one(c::Operation(c::ScaledDotProductAttention{
                .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                .scale = 0.0625F,
                .causal = true,
                .accumulation_type = c::DataType::kFloat32}),
            {rq, rk, value});
  auto probability =
      f.one(c::Operation(c::Unary{c::UnaryKind::kSigmoid}), {gate});
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
    auto reference =
        attention_reference(rotated_q, rotated_k, values, gates, false);
    auto native =
        attention_reference(rotated_q, rotated_k, values, gates, true);
    f.replays.push_back(
        {{bytes(queries), bytes(keys), bytes(values), bytes(gates)},
         {normalized_q, normalized_k, rotated_q, rotated_k, reference[0],
          reference[1]},
         {{}, {}, {}, {}, native[0], native[1]}});
  }
  return f;
}

inline Fragment gdn(const c::ModelArtifactProfileDescription *profile) {
  namespace gd = d5_qwen38_gdn_fixture;
  const bool official = profile == &q::kOfficialProfile;
  Fragment f{
      .profile = profile, .region = 0x050523, .absolute_tolerance = 2.0e-6F};
  constexpr std::int64_t keys = q::kGdnKeyHeads;
  constexpr std::int64_t heads = q::kGdnValueHeads;
  constexpr std::int64_t depth = q::kGdnHeadDepth;
  constexpr std::int64_t key_width = keys * depth;
  constexpr std::int64_t channels = (2 * keys + heads) * depth;
  gd::ConvFixture conv(keys, heads);
  auto samples = f.input(conv.samples_desc);
  auto weight = f.constant(conv.weights_desc, bytes(conv.weights));
  auto convolution = f.one(c::model_ops::causal_conv1d(), {samples, weight});
  auto activated =
      f.one(c::Operation(c::Unary{c::UnaryKind::kSilu}), {convolution});
  const auto reshape = [&](ValueId input,
                           std::initializer_list<std::int64_t> shape) {
    return f.one(
        c::Operation(
            c::View::reshape(
                c::TensorShape::create({shape.begin(), shape.size()}).value())
                .value()),
        {input});
  };
  auto channel_rows = reshape(activated, {channels, 1});
  const auto take = [&](std::int64_t first, std::int64_t count) {
    std::vector<std::int32_t> indices(count);
    for (std::int64_t i = 0; i < count; ++i)
      indices[i] = static_cast<std::int32_t>(first + i);
    auto ids = f.constant(dense({count}, c::DataType::kInt32), bytes(indices));
    return f.one(c::Operation(c::Gather{}), {channel_rows, ids});
  };
  auto query = reshape(take(0, key_width), {keys, depth});
  auto key = reshape(take(key_width, key_width), {keys, depth});
  auto value =
      reshape(take(2 * key_width, heads * depth), {1, 1, heads, depth});
  // Official V heads are grouped by K head; the pinned GGUF converter has
  // already reordered V, conv, gates and output weights into cyclic order.
  // Expand only the official Q/K boundary. GGUF keeps the native 16/128
  // grouped-head program without a head-replication gather.
  std::vector<float> l2_scale(depth, 1.0F / std::sqrt(float(depth)));
  auto l2 = f.constant(dense({depth}), bytes(l2_scale));
  const c::Operation normalize(c::RmsNorm{.epsilon = 1.0e-6F / float(depth)});
  query = f.one(normalize, {query, l2});
  key = f.one(normalize, {key, l2});
  if (official) {
    std::vector<std::int32_t> expansion(heads);
    for (std::int64_t h = 0; h < heads; ++h)
      expansion[h] = static_cast<std::int32_t>(h / (heads / keys));
    auto expansion_ids =
        f.constant(dense({heads}, c::DataType::kInt32), bytes(expansion));
    query = f.one(c::Operation(c::Gather{}), {query, expansion_ids});
    key = f.one(c::Operation(c::Gather{}), {key, expansion_ids});
  }
  const auto native_key_heads = official ? heads : keys;
  query = reshape(query, {1, 1, native_key_heads, depth});
  key = reshape(key, {1, 1, native_key_heads, depth});

  const auto gate_desc = dense({1, 1, heads, 1});
  auto raw_a = f.input(gate_desc);
  auto raw_b = f.input(gate_desc);
  auto initial_state = f.input(dense({1, heads, depth, depth}));
  auto z = f.input(dense({1, 1, heads, depth}));
  std::vector<float> alog(heads), negative_a(heads), dt(heads);
  for (std::int64_t h = 0; h < heads; ++h) {
    alog[h] = -1.0F - float(h % 7) / 32.0F;
    negative_a[h] = -std::exp(alog[h]);
    dt[h] = float(h % 9 - 4) / 64.0F;
  }
  // Runtime F32 ports are the declared recurrent domain. Official small
  // constants originate as BF16 and are exactly representable here; GGUF
  // ssm_a stores the converter's -exp(A_log), independently of its Q label.
  auto bias = f.constant(gate_desc, bytes(dt));
  auto alpha = f.constant(gate_desc, bytes(official ? alog : negative_a));
  if (official) {
    alpha = f.one(c::Operation(c::Unary{c::UnaryKind::kExp}), {alpha});
    alpha = f.one(c::Operation(c::Unary{c::UnaryKind::kNeg}), {alpha});
  }
  auto gate =
      f.one(c::Operation(c::Binary{c::BinaryKind::kAdd}), {raw_a, bias});
  gate = f.one(c::Operation(c::Unary{c::UnaryKind::kSoftplus}), {gate});
  gate =
      f.one(c::Operation(c::Binary{c::BinaryKind::kMultiply}), {gate, alpha});
  auto beta = f.one(c::Operation(c::Unary{c::UnaryKind::kSigmoid}), {raw_b});
  auto recurrent = f.append(c::Operation(c::GatedDeltaNet{}),
                            {query, key, value, gate, beta, initial_state}, 2);
  auto norm_values = oracle::scale(depth);
  auto norm = f.constant(dense({depth}), bytes(norm_values));
  auto normalized =
      f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}), {recurrent[0], norm});
  auto activated_z = f.one(c::Operation(c::Unary{c::UnaryKind::kSilu}), {z});
  auto result = f.one(c::Operation(c::Binary{c::BinaryKind::kMultiply}),
                      {normalized, activated_z});
  f.outputs = {convolution, gate, beta, recurrent[0], recurrent[1], result};
  std::vector<float> state;
  for (int seed : {3, 17}) {
    gd::Fixture reference(seed, native_key_heads, heads);
    if (!state.empty())
      reference.data[5] = state;
    auto samples_values = conv.samples(seed);
    auto conv_values = as_float(conv.oracle(samples_values));
    auto silu = conv_values;
    for (auto &v : silu)
      v = static_cast<float>(double(v) / (1.0 + std::exp(-double(v))));
    std::array<std::vector<float>, 2> normalized_qk;
    for (std::size_t p = 0; p < normalized_qk.size(); ++p) {
      normalized_qk[p].resize(key_width);
      for (std::int64_t h = 0; h < keys; ++h) {
        double sum = 1.0e-6;
        for (std::int64_t d = 0; d < depth; ++d) {
          const double v = silu[p * key_width + h * depth + d];
          sum += v * v;
        }
        for (std::int64_t d = 0; d < depth; ++d)
          normalized_qk[p][h * depth + d] = static_cast<float>(
              silu[p * key_width + h * depth + d] / std::sqrt(sum));
      }
      for (std::int64_t h = 0; h < native_key_heads; ++h)
        for (std::int64_t d = 0; d < depth; ++d)
          reference.data[p][h * depth + d] =
              normalized_qk[p][(official ? h / (heads / keys) : h) * depth + d];
    }
    std::copy(silu.begin() + 2 * key_width, silu.end(),
              reference.data[2].begin());
    std::vector<float> a(heads), b(heads);
    for (std::int64_t h = 0; h < heads; ++h) {
      a[h] = float((h + seed) % 17 - 8) / 32.0F;
      b[h] = float((3 * h + seed) % 19 - 9) / 16.0F;
      reference.data[3][h] =
          static_cast<float>(-std::exp(double(alog[h])) *
                             std::log1p(std::exp(double(a[h]) + dt[h])));
      reference.data[4][h] =
          static_cast<float>(1.0 / (1.0 + std::exp(-double(b[h]))));
    }
    auto z_values = d5_gguf_test::values(heads, depth, seed + 7);
    Replay expected{{bytes(samples_values), bytes(a), bytes(b),
                     bytes(reference.data[5]), bytes(z_values)},
                    {}};
    const auto outputs = reference.oracle();
    auto core = as_float(outputs[0]);
    state = as_float(outputs[1]);
    auto gated = oracle::rms_norm_rows(core, norm_values, depth, 1.0e-6F);
    for (std::size_t i = 0; i < gated.size(); ++i)
      gated[i] *= static_cast<float>(double(z_values[i]) /
                                     (1.0 + std::exp(-double(z_values[i]))));
    expected.outputs = {std::move(conv_values),
                        reference.data[3],
                        reference.data[4],
                        std::move(core),
                        state,
                        std::move(gated)};
    f.replays.push_back(std::move(expected));
  }
  return f;
}

inline void expect_output(const Fragment &fragment,
                          std::span<const float> actual,
                          std::span<const float> expected,
                          std::span<const float> native_attention = {}) {
  if (!native_attention.empty()) {
    // This pin's Vulkan coopmat2 path rounds both multiplicands to F16.
    // Retain the full F32 accuracy gate as well as an independent rounded
    // reference; normalization, RoPE and GDN do not use this allowance.
    oracle::expect_close(actual, expected, 2.0e-4F, 2.0e-3F);
    oracle::expect_close(actual, native_attention, 8.0e-5F, 4.0e-5F);
  } else if (fragment.quantized)
    d5_qwen38_24t_moe_profiles::expect_output(fragment.profile, actual,
                                              expected);
  else
    oracle::expect_close(actual, expected, fragment.absolute_tolerance,
                         4.0e-5F);
}
inline constexpr std::array kProfiles{
    &d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
    &d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile};
} // namespace d5_qwen38_24t_semantic_fragments
