#pragma once

#include "d5_kimi_remaining_fragments.h"

// Model consumers, not product kernels. Native ggml storage and the exact
// source formulas are fixed by d5_glm_remaining_facts.json and the D5 card.
namespace d5_glm_remaining_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace linear = d5_kda_full_fragments;
namespace moe = d5_moe_execution_fragments;
namespace kimi = d5_kimi_remaining_fragments;
using sem::Fragment;
using sem::ValueId;
using sem::bytes;
using sem::dense;
using core::reshape;
using core::binary;
using core::unary;
using core::filled;
using linear::Storage;

struct Case final {
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t hidden, query_rank;
  bool flash, official;
};
inline constexpr std::array kCases{
    Case{&core::profiles::glm53_official, 6144, 2048, false, true},
    Case{&core::profiles::glm53_primary_gguf, 6144, 2048, false, false},
    Case{&core::profiles::glm53_fidelity_gguf, 6144, 2048, false, false},
    Case{&core::profiles::glm53_flash_official, 4096, 1536, true, true},
    Case{&core::profiles::glm53_flash_primary_gguf, 4096, 1536, true, false},
    Case{&core::profiles::glm53_flash_fidelity_gguf, 4096, 1536, true, false}};
inline constexpr std::int64_t kHeads = 64, kLatent = 512, kValue = 256;
inline constexpr float kNormEpsilon = 1.0e-5F;

inline ValueId permute(Fragment &f, ValueId value,
                       std::initializer_list<std::int32_t> axes) {
  return f.one(c::Operation(c::View::permute({axes.begin(), axes.size()}).value()),
               {value});
}
inline ValueId reduce(Fragment &f, ValueId value, std::int32_t axis,
                      c::ReductionKind kind = c::ReductionKind::kSum) {
  return f.one(c::Operation(c::Reduce{
      .kind = kind, .axes = c::Axes::from(std::span(&axis, 1)).value(),
      .accumulation_type = c::DataType::kFloat32}), {value});
}

// Attach an existing model recipe to actual upstream values. This only maps
// public operation descriptions; execution stays with each native route.
inline std::vector<ValueId> attach(Fragment &f, Fragment child,
                                   std::initializer_list<ValueId> inputs) {
  if (inputs.size() != child.inputs.size())
    throw std::runtime_error("model fragment input count mismatch");
  std::vector<ValueId> mapped(child.descriptions.size());
  std::size_t i = 0;
  for (auto input : inputs) mapped[child.inputs[i++].ordinal()] = input;
  for (i = 0; i < child.constants.size(); ++i) {
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
  std::vector<ValueId> outputs;
  for (auto output : child.outputs) outputs.push_back(mapped[output.ordinal()]);
  return outputs;
}

// GLM Flash mHC explicitly calls fn.float(): weights retain their exact BF16
// values but the input is F32. Do not insert an extra BF16 activation boundary.
inline linear::Projection mhc_projection(Fragment &f, ValueId input, bool official) {
  if (!official)
    return linear::projection(f, input, 24, 16384, Storage::kQ8_0, 233);
  auto weight = sem::weight(24, 16384, c::GgufQuantizedType::kQ8_0, true, 233);
  std::vector<sem::q::BF16> encoded(weight.storage.size() / sizeof(float));
  for (std::size_t i = 0; i < encoded.size(); ++i) {
    float value;
    std::memcpy(&value, weight.storage.data() + i * sizeof(float), sizeof(float));
    encoded[i] = sem::q::encode(value);
  }
  for (auto &row : weight.decoded)
    for (auto &[column, value] : row) value = linear::bf16(value);
  auto stored = f.constant(dense({24, 16384}, c::DataType::kBFloat16), bytes(encoded));
  auto fp32 = f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {stored});
  auto output = f.one(c::Operation(c::MatMul{}), {input, permute(f, fp32, {1, 0})});
  return {std::move(weight), Storage::kF32, output};
}

inline ValueId norm(Fragment &f, ValueId x, std::vector<float> &scales,
                     bool bfloat = false) {
  auto weight = linear::parameter(f, dense({std::int64_t(scales.size())}),
                                   scales, bfloat);
  return f.one(c::Operation(c::RmsNorm{.epsilon = kNormEpsilon}), {x, weight});
}

inline Fragment dense_ffn(const Case &model) {
  if (model.official)
    throw std::runtime_error("official GLM FFN requires native FP8, not a Q8 stand-in");
  Fragment f{.profile = model.profile, .region = 0x02030801, .quantized = true};
  auto x = f.input(dense({1, model.hidden}));
  auto scales = sem::oracle::scale(model.hidden);
  auto normalized = norm(f, x, scales);
  auto g = linear::projection(f, normalized, 12288, model.hidden, Storage::kQ8_0, 173);
  auto u = linear::projection(f, normalized, 12288, model.hidden, Storage::kQ8_0, 179);
  const auto &moe_model = moe::cases::kCases[model.flash ? 11 : 8];
  auto gated = moe::gated(f, moe_model, g.output, u.output);
  auto down = linear::projection(f, gated, model.hidden, 12288, Storage::kQ8_0, 181);
  auto residual = binary(f, c::BinaryKind::kAdd, x, down.output);
  f.outputs = {normalized, g.output, u.output, gated, down.output, residual};
  for (int seed : {3, 17, 29}) {
    auto input = d5_gguf_test::activation(model.hidden, seed);
    auto nx = sem::oracle::rms_norm_rows(input, scales, model.hidden, kNormEpsilon);
    auto gv = g.reference(nx), uv = u.reference(nx);
    auto av = moe::gate_reference(moe_model, gv, uv);
    auto dv = down.reference(av), result = dv;
    for (std::size_t i = 0; i < result.size(); ++i) result[i] += input[i];
    f.replays.push_back({{bytes(input)}, {nx, gv, uv, av, dv, result}});
  }
  return f;
}

// Independent NextN input boundary. The decoder attention/MoE and final head
// still require their own complete-chain receipts; this is not a whole MTP.
inline Fragment mtp_input(const Case &model) {
  Fragment f{.profile = model.profile, .region = 0x0d030801,
             .quantized = !model.official};
  auto hidden = f.input(dense({1, model.hidden}));
  auto embedding = f.input(dense({1, model.hidden}));
  auto hs = sem::oracle::scale(model.hidden), es = hs;
  std::reverse(es.begin(), es.end());
  for (std::size_t i = 0; i < es.size(); ++i) es[i] += float(i % 3) * 0.0037F;
  auto hn = norm(f, hidden, hs, model.official);
  auto en = norm(f, embedding, es, model.official);
  // glm-dsa.cpp uses [normalized token embedding, normalized hidden].
  // The pinned converter preserves the EH columns; do not swap the halves.
  auto joined = f.one(c::Operation(c::Concat{.axis = 1}), {en, hn});
  auto eh = linear::projection(f, joined, model.hidden, 2 * model.hidden,
      model.official ? Storage::kBFloat16 : Storage::kQ8_0, 193);
  auto ns = sem::oracle::scale(model.hidden);
  auto result = norm(f, eh.output, ns, model.official);
  f.outputs = {hn, en, joined, eh.output, result};
  if (model.official) f.bfloat_chain_outputs = {3, 4};
  for (int seed : {5, 23, 37}) {
    auto h = d5_gguf_test::activation(model.hidden, seed);
    auto e = d5_gguf_test::activation(model.hidden, seed + 7);
    auto hv = sem::oracle::rms_norm_rows(h, hs, model.hidden, kNormEpsilon);
    auto ev = sem::oracle::rms_norm_rows(e, es, model.hidden, kNormEpsilon);
    auto j = ev; j.insert(j.end(), hv.begin(), hv.end());
    auto projected = eh.reference(j);
    auto expected = sem::oracle::rms_norm_rows(projected, ns, model.hidden, kNormEpsilon);
    f.replays.push_back({{bytes(h), bytes(e)}, {hv, ev, j, projected, expected}});
  }
  return f;
}

inline Fragment hyper_connection(const Case &model) {
  if (!model.flash) throw std::runtime_error("GLM DSA has no mHC streams");
  Fragment f{.profile = model.profile, .region = 0x0c030801,
             .quantized = !model.official};
  auto residual = f.input(dense({1, 4, model.hidden}));
  auto branch = f.input(dense({1, model.hidden}));
  auto flat = reshape(f, residual, {1, 4 * model.hidden});
  auto ones = filled(f, dense({4 * model.hidden}), 1.0F);
  auto normalized = f.one(c::Operation(c::RmsNorm{.epsilon = kNormEpsilon}),
                          {flat, ones});
  auto fn = mhc_projection(f, normalized, model.official);
  auto core_outputs = attach(f, core::hyper_connection({model.profile, model.hidden}),
                            {fn.output, residual, branch});
  auto mean = reduce(f, core_outputs[2], 1, c::ReductionKind::kMean);
  auto final_scale = sem::oracle::scale(model.hidden);
  auto final = norm(f, mean, final_scale, model.official);
  f.outputs = {core_outputs[2], normalized, fn.output, core_outputs[0],
               core_outputs[1], mean, final};
  f.state_ports = {{0, 0}};
  std::vector<float> scales{0.125F, 0.25F, 0.5F}, bases(24);
  for (int i = 0; i < 24; ++i) bases[i] = float(i % 7 - 3) / 8.0F;
  auto state = d5_gguf_test::values(4, model.hidden, 31);
  const std::vector<float> unweighted(4 * model.hidden, 1.0F);
  for (int seed : {3, 17, 29, 41}) {
    auto b = d5_gguf_test::activation(model.hidden, seed);
    auto nv = sem::oracle::rms_norm_rows(state, unweighted, 4 * model.hidden, kNormEpsilon);
    auto projected = fn.reference(nv);
    auto core_values = core::hyper_reference(projected, state, b, model.hidden, scales, bases);
    std::vector<float> mv(model.hidden);
    for (std::int64_t d = 0; d < model.hidden; ++d) {
      double sum = 0;
      for (int s = 0; s < 4; ++s) sum += core_values[2][s * model.hidden + d];
      mv[d] = float(sum / 4.0);
    }
    auto fv = sem::oracle::rms_norm_rows(mv, final_scale, model.hidden, kNormEpsilon);
    f.replays.push_back({{bytes(state), bytes(b)},
                         {core_values[2], nv, projected, core_values[0], core_values[1], mv, fv}});
    state = std::move(core_values[2]);
  }
  return f;
}

struct HeadProjection final {
  sem::Weight weight;
  std::int64_t n, k;
  ValueId output;
  std::vector<float> reference(std::span<const float> x) const {
    std::vector<std::int32_t> ids(kHeads);
    std::iota(ids.begin(), ids.end(), 0);
    return moe::project(x, weight, ids, n, k);
  }
};
inline HeadProjection head_projection(Fragment &f, ValueId x,
                                       std::int64_t n, std::int64_t k, int seed) {
  auto w = moe::weight(kHeads, n, k, moe::Storage::kQ8_0, seed);
  std::vector<std::int32_t> ids(kHeads);
  std::iota(ids.begin(), ids.end(), 0);
  auto heads = f.constant(dense({1, kHeads}, c::DataType::kInt32), bytes(ids));
  auto weight = f.constant(w.description, std::move(w.storage));
  auto output = f.one(c::model_ops::grouped_matmul(), {weight, x, heads});
  return {std::move(w), n, k, output};
}

inline void rotate_pairs(std::vector<float> &x, std::span<const float> cosine,
                     std::span<const float> sine) {
  for (std::size_t row = 0; row < x.size() / 64; ++row)
    for (std::size_t p = 0; p < 32; ++p) {
      const auto i = row * 64 + 2 * p;
      const double a = x[i], b = x[i + 1];
      x[i] = float(a * cosine[p] - b * sine[p]);
      x[i + 1] = float(a * sine[p] + b * cosine[p]);
    }
}

inline std::vector<float> layer_norm(std::span<const float> input,
                                     std::span<const float> scale,
                                     std::span<const float> bias) {
  double mean = 0, variance = 0;
  for (auto v : input) mean += v;
  mean /= double(input.size());
  for (auto v : input) variance += (v - mean) * (v - mean);
  const double inverse = 1.0 / std::sqrt(variance / double(input.size()) + 1.0e-6);
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = float((input[i] - mean) * inverse * scale[i] + bias[i]);
  return output;
}

// Decode DSA at its actual 32x128 indexer boundary. The cache exceeds top-2048;
// Flash has 513 complete K4 pools (top-512), left padding, and 0..3 tail tokens.
// Only positions/validity are host model policy. Projections, learned pooling,
// scoring, TopK and expansion all execute through mature route operations.
inline Fragment indexer(const Case &model, std::int64_t tail = 0,
                         bool quantized_scores = false) {
  if (model.official && !model.flash)
    throw std::runtime_error("official GLM DSA indexer needs its native FP8 roles");
  if (tail < 0 || tail > 3 || (!model.flash && tail))
    throw std::runtime_error("invalid GLM pool tail fixture");
  constexpr std::int64_t heads = 32, depth = 128, padding = 2, pools = 513;
  const std::int64_t count = padding + 2052 + tail, history_count = count - 1;
  const auto storage = model.official ? Storage::kBFloat16 : Storage::kQ8_0;
  Fragment f{.profile = model.profile, .region = 0x08030801 + std::uint64_t(tail),
             .quantized = !model.official};
  auto x = f.input(dense({1, model.hidden}));
  auto q_resid = f.input(dense({1, model.query_rank}));
  auto past_key = f.input(dense({history_count, depth}));
  ValueId past_gate{}, positions{};
  if (model.flash) past_gate = f.input(dense({history_count, depth}));
  else positions = f.input(dense({1}, c::DataType::kInt32));
  auto wq = linear::projection(f, q_resid, heads * depth, model.query_rank, storage, 239);
  auto wk = linear::projection(f, x, depth, model.hidden, storage, 241);
  const auto score_storage = model.official ? Storage::kBFloat16
      : quantized_scores ? Storage::kQ8_0 : Storage::kF32;
  auto ws = linear::projection(f, x, heads, model.hidden, score_storage, 251);
  auto scales = sem::oracle::scale(depth);
  std::vector<float> bias(depth);
  for (std::int64_t d = 0; d < depth; ++d) bias[d] = float(d % 7 - 3) / 32.0F;
  auto scale = linear::parameter(f, dense({depth}), scales, model.official);
  auto offset = linear::parameter(f, dense({depth}), bias, model.official);
  auto key = f.one(c::Operation(c::LayerNorm{.epsilon = 1.0e-6F}), {wk.output, scale, offset});
  auto query = reshape(f, wq.output, {1, heads, depth});
  std::vector<float> frequencies(32);
  if (!model.flash) {
    for (int p = 0; p < 32; ++p) frequencies[p] = float(1.0 / std::pow(8000000.0, double(p) / 32));
    auto frequency = f.constant(dense({32}), bytes(frequencies));
    auto qp = kimi::slice_heads(f, query, heads, depth, 0, 64);
    auto kp = kimi::slice_heads(f, key, 1, depth, 0, 64);
    c::Operation rope(c::RoPE{.rotary_dimensions = 64, .sequence_axis = 1,
                              .layout = c::RopeLayout::kInterleaved});
    qp = reshape(f, f.one(rope, {reshape(f, qp, {1, 1, heads, 64}), positions, frequency}), {1, heads, 64});
    kp = reshape(f, f.one(rope, {reshape(f, kp, {1, 1, 1, 64}), positions, frequency}), {1, 1, 64});
    query = f.one(c::Operation(c::Concat{.axis = 2}),
        {qp, kimi::slice_heads(f, query, heads, depth, 64, 64)});
    key = reshape(f, f.one(c::Operation(c::Concat{.axis = 2}),
        {kp, kimi::slice_heads(f, key, 1, depth, 64, 64)}), {1, depth});
  }
  auto cache = f.one(c::Operation(c::Concat{.axis = 0}), {past_key, key});
  std::vector<std::int32_t> next_ids;
  for (std::int32_t t = 0; t < count; ++t) if (t != padding) next_ids.push_back(t);
  auto next_index = f.constant(dense({history_count}, c::DataType::kInt32), bytes(next_ids));
  auto next_cache = f.one(c::Operation(c::Gather{}), {cache, next_index});
  auto keys = cache;
  linear::Projection wg{};
  ValueId next_gates{}, pool_index{};
  std::vector<float> ape(4 * depth);
  std::vector<std::int32_t> pool_ids(pools * 4);
  if (model.flash) {
    wg = linear::projection(f, x, depth, model.hidden, storage, 257);
    auto gates = f.one(c::Operation(c::Concat{.axis = 0}), {past_gate, wg.output});
    next_gates = f.one(c::Operation(c::Gather{}), {gates, next_index});
    for (std::int32_t i = 0; i < pools * 4; ++i) pool_ids[i] = std::int32_t(padding) + i;
    pool_index = f.constant(dense({pools, 4}, c::DataType::kInt32), bytes(pool_ids));
    auto grouped_key = f.one(c::Operation(c::Gather{}), {cache, pool_index});
    auto grouped_gate = f.one(c::Operation(c::Gather{}), {gates, pool_index});
    for (std::size_t i = 0; i < ape.size(); ++i) ape[i] = float(int(i * 7 % 19) - 9) / 16.0F;
    auto pos_bias = linear::parameter(f, dense({1, 4, depth}), ape, model.official);
    auto logits = binary(f, c::BinaryKind::kAdd, grouped_gate, pos_bias);
    // ggml Softmax consumes contiguous rows. Put the four pool tokens last,
    // materialize that native layout, and restore the logical key layout.
    auto transposed = f.one(c::Operation(c::Copy{}), {permute(f, logits, {0, 2, 1})});
    auto probabilities = f.one(c::Operation(c::Softmax{}),
        {reshape(f, transposed, {pools * depth, 4})});
    probabilities = f.one(c::Operation(c::Copy{}),
        {permute(f, reshape(f, probabilities, {pools, depth, 4}), {0, 2, 1})});
    keys = reduce(f, binary(f, c::BinaryKind::kMultiply, probabilities, grouped_key), 1);
  }
  const auto candidates = model.flash ? pools : count;
  auto dot = f.one(c::Operation(c::MatMul{}),
      {reshape(f, query, {heads, depth}), permute(f, keys, {1, 0})});
  dot = binary(f, c::BinaryKind::kMultiply, dot,
               filled(f, dense({1}), 1.0F / std::sqrt(float(depth))));
  dot = f.one(c::model_ops::clamp(0.0F, std::numeric_limits<float>::max()), {dot});
  auto weights = binary(f, c::BinaryKind::kMultiply, ws.output,
                        filled(f, dense({1}), 1.0F / std::sqrt(float(heads))));
  auto scores = reshape(f, reduce(f, binary(f, c::BinaryKind::kMultiply,
      reshape(f, weights, {heads, 1}), dot), 0), {1, candidates});
  if (!model.flash) {
    std::vector<float> mask(count);
    std::fill_n(mask.begin(), padding, -1.0e9F);
    scores = binary(f, c::BinaryKind::kAdd, scores,
                     f.constant(dense({1, count}), bytes(mask)));
  }
  const std::size_t top_count = model.flash ? 512 : 2048;
  auto selected = f.append(c::Operation(c::TopK{.count = top_count}), {scores}, 2);
  auto ids = selected[1];
  if (model.flash) {
    ids = reshape(f, f.one(c::Operation(c::Gather{}), {pool_index, ids}), {1, 2048});
    std::vector<std::int32_t> tail_ids(3, -1);
    for (int i = 0; i < tail; ++i) tail_ids[i] = std::int32_t(padding + 2052 + i);
    ids = f.one(c::Operation(c::Concat{.axis = 1}),
                {ids, f.constant(dense({1, 3}, c::DataType::kInt32), bytes(tail_ids))});
  }
  // Diagnostic Q/K publications own storage; their reshape aliases remain
  // internal to the model program and are not separately rebound by Graph.
  auto query_check = f.one(c::Operation(c::Copy{}), {query});
  auto key_check = f.one(c::Operation(c::Copy{}), {key});
  auto score_check = f.one(c::Operation(c::Copy{}), {scores});
  f.outputs = {next_cache, query_check, key_check, weights, keys, score_check, ids};
  f.state_ports = {{2, 0}};
  if (model.flash) { f.outputs.push_back(next_gates); f.state_ports.push_back({3, 7}); }
  f.unordered_ids = Fragment::UnorderedIds{6, model.flash ? 2051U : 2048U,
                                         count, true, model.flash};
  std::vector<float> past(history_count * depth), gate_history(past.size());
  for (std::int64_t t = padding; t < history_count; ++t)
    for (std::int64_t d = 0; d < depth; ++d) {
      past[t * depth + d] = (0.125F + float(t - padding) / 8.0F) *
                            (0.5F + float(d % 11) / 32.0F);
      gate_history[t * depth + d] = float((t * 7 + d * 11) % 29 - 14) / 8.0F;
    }
  for (int step = 0; step < 4; ++step) {
    auto input = d5_gguf_test::activation(model.hidden, 5 + step * 14);
    auto qr = d5_gguf_test::activation(model.query_rank, 11 + step * 14);
    for (auto &value : input) value = std::abs(value) + 0.5F;
    for (auto &value : qr) value = std::abs(value) + 0.5F;
    sem::Replay replay{{bytes(input), bytes(qr), bytes(past)}, {}};
    auto qv = wq.reference(qr), projected_key = wk.reference(input);
    auto kv = layer_norm(projected_key, scales, bias);
    auto wv = ws.reference(input);
    for (auto &value : wv) value *= 1.0F / std::sqrt(float(heads));
    if (!model.flash) {
      const std::int32_t position = 2061 + step;
      replay.inputs.push_back(bytes(std::vector<std::int32_t>{position}));
      std::vector<float> cs(32), sn(32);
      for (int p = 0; p < 32; ++p) {
        cs[p] = float(std::cos(double(position) * frequencies[p]));
        sn[p] = float(std::sin(double(position) * frequencies[p]));
      }
      auto qp = kimi::slice(qv, heads, depth, 0, 64), kp = kimi::slice(kv, 1, depth, 0, 64);
      rotate_pairs(qp, cs, sn); rotate_pairs(kp, cs, sn);
      for (int h = 0; h < heads; ++h)
        std::copy_n(qp.begin() + h * 64, 64, qv.begin() + h * depth);
      std::copy(kp.begin(), kp.end(), kv.begin());
    }
    auto full = past; full.insert(full.end(), kv.begin(), kv.end());
    auto pool_keys = full;
    std::vector<float> next_gate_values;
    if (model.flash) {
      replay.inputs.push_back(bytes(gate_history));
      auto gv = wg.reference(input);
      auto full_gates = gate_history; full_gates.insert(full_gates.end(), gv.begin(), gv.end());
      pool_keys.assign(pools * depth, 0.0F);
      for (std::int64_t p = 0; p < pools; ++p)
        for (std::int64_t d = 0; d < depth; ++d) {
          std::array<double, 4> probability{};
          double maximum = -INFINITY, sum = 0;
          for (int k = 0; k < 4; ++k) {
            probability[k] = full_gates[(padding + p * 4 + k) * depth + d] + ape[k * depth + d];
            maximum = std::max(maximum, probability[k]);
          }
          for (auto &value : probability) sum += value = std::exp(value - maximum);
          double value = 0;
          for (int k = 0; k < 4; ++k)
            value += probability[k] / sum * full[(padding + p * 4 + k) * depth + d];
          pool_keys[p * depth + d] = float(value);
        }
      for (auto index : next_ids)
        next_gate_values.insert(next_gate_values.end(), full_gates.begin() + index * depth,
                               full_gates.begin() + (index + 1) * depth);
    }
    std::vector<float> index_scores(candidates), half_product_scores(candidates);
    const auto half = [](float value) {
      return half_test::decode(half_test::encode<hyle::simd::float16_t>(value));
    };
    for (std::int64_t t = 0; t < candidates; ++t) {
      double sum = 0, half_sum = 0;
      for (int h = 0; h < heads; ++h) {
        double dot_value = 0, half_dot = 0;
        for (int d = 0; d < depth; ++d) {
          dot_value += double(qv[h * depth + d]) * pool_keys[t * depth + d];
          half_dot += half(qv[h * depth + d]) * half(pool_keys[t * depth + d]);
        }
        sum += wv[h] * std::max(0.0, dot_value / std::sqrt(double(depth)));
        half_sum += wv[h] * std::max(0.0, half_dot / std::sqrt(double(depth)));
      }
      index_scores[t] = float(sum) + ((!model.flash && t < padding) ? -1.0e9F : 0.0F);
      half_product_scores[t] = float(half_sum);
    }
    std::vector<std::int32_t> order(candidates);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
      return index_scores[a] > index_scores[b];
    });
    std::vector<float> selected_ids;
    for (std::size_t i = 0; i < top_count; ++i) {
      if (model.flash) for (int j = 0; j < 4; ++j) selected_ids.push_back(float(pool_ids[order[i] * 4 + j]));
      else selected_ids.push_back(float(order[i]));
    }
    if (model.flash) for (int i = 0; i < 3; ++i)
      selected_ids.push_back(i < tail ? float(padding + 2052 + i) : -1.0F);
    past.clear();
    for (auto index : next_ids)
      past.insert(past.end(), full.begin() + index * depth, full.begin() + (index + 1) * depth);
    replay.outputs = {past, qv, kv, wv, pool_keys, index_scores, selected_ids};
    if (model.flash) { replay.outputs.push_back(next_gate_values); gate_history = std::move(next_gate_values); }
    if (model.flash && model.official) {
      // The pinned ggml Vulkan coopmat2 MatMul converts both F32 operands to
      // F16, independently of its F32 accumulation setting. Check only that
      // score boundary against the independent rounded oracle AND the existing
      // full-F32 accuracy gate. Cache, pooling and exact TopK members stay strict.
      replay.native_outputs.resize(replay.outputs.size());
      replay.native_outputs[5] = std::move(half_product_scores);
    }
    f.replays.push_back(std::move(replay));
  }
  return f;
}

// Exact ggml absorbed-MLA program. DSA selection is a separate shared region;
// this cache consumer validates native projection, RoPE and actual feedback.
inline Fragment mla(const Case &model) {
  if (model.official)
    throw std::runtime_error("official GLM MLA requires its mixed FP8 roles");
  const std::int64_t rotary = model.flash ? 0 : 64;
  const auto no_pe = 256 - rotary, cached_width = kLatent + rotary;
  Fragment f{.profile = model.profile, .region = 0x07030801, .quantized = true};
  auto x = f.input(dense({1, model.hidden}));
  auto history = f.input(dense({3, cached_width}));
  ValueId positions{}, frequencies{};
  std::vector<float> frequency_values(32);
  if (rotary) {
    positions = f.input(dense({1}, c::DataType::kInt32));
    for (std::size_t p = 0; p < frequency_values.size(); ++p)
      frequency_values[p] = float(1.0 / std::pow(8000000.0, double(p) / 32.0));
    frequencies = f.constant(dense({32}), bytes(frequency_values));
  }
  auto input_scale = sem::oracle::scale(model.hidden);
  auto nx = norm(f, x, input_scale);
  auto qa = linear::projection(f, nx, model.query_rank, model.hidden, Storage::kQ8_0, 197);
  auto qscale = sem::oracle::scale(model.query_rank);
  auto qr = norm(f, qa.output, qscale);
  auto qb = linear::projection(f, qr, kHeads * 256, model.query_rank, Storage::kQ8_0, 199);
  auto kva = linear::projection(f, nx, cached_width, model.hidden, Storage::kQ8_0, 211);
  auto latent = reshape(f, core::rows(f, reshape(f, kva.output, {cached_width, 1}), 0, kLatent), {1, kLatent});
  auto kvscale = sem::oracle::scale(kLatent);
  latent = norm(f, latent, kvscale);
  auto query_nope = kimi::slice_heads(f, qb.output, kHeads, 256, 0, no_pe);
  auto kb = head_projection(f, query_nope, kLatent, no_pe, 223);
  auto query = kb.output, current = latent;
  if (rotary) {
    auto qp = kimi::slice_heads(f, qb.output, kHeads, 256, no_pe, rotary);
    auto kp = reshape(f, core::rows(f, reshape(f, kva.output, {cached_width, 1}), kLatent, rotary), {1, 1, rotary});
    const c::Operation rope(c::RoPE{.rotary_dimensions = 64, .sequence_axis = 1,
                                    .layout = c::RopeLayout::kInterleaved});
    qp = reshape(f, f.one(rope, {reshape(f, qp, {1, 1, kHeads, 64}), positions, frequencies}), {1, kHeads, 64});
    kp = reshape(f, f.one(rope, {reshape(f, kp, {1, 1, 1, 64}), positions, frequencies}), {1, 1, 64});
    query = f.one(c::Operation(c::Concat{.axis = 2}), {query, qp});
    current = f.one(c::Operation(c::Concat{.axis = 1}),
                     {latent, reshape(f, kp, {1, rotary})});
  }
  auto cache = f.one(c::Operation(c::Concat{.axis = 0}), {history, current});
  auto next_cache = core::rows(f, cache, 1, 3);
  auto cv = kimi::slice_heads(f, cache, 4, cached_width, 0, kLatent);
  auto attention = f.one(c::Operation(c::ScaledDotProductAttention{
      .scale = 1.0F / 16.0F, .causal = true,
      .accumulation_type = c::DataType::kFloat32}),
      {reshape(f, query, {1, 1, kHeads, cached_width}),
       reshape(f, cache, {1, 4, 1, cached_width}), reshape(f, cv, {1, 4, 1, kLatent})});
  auto vb = head_projection(f, reshape(f, attention, {1, kHeads, kLatent}), kValue, kLatent, 227);
  auto out = linear::projection(f, reshape(f, vb.output, {1, kHeads * kValue}),
                               model.hidden, kHeads * kValue, Storage::kQ8_0, 229);
  f.outputs = {next_cache, qb.output, query, attention, vb.output, out.output};
  f.state_ports = {{1, 0}};
  auto past = d5_gguf_test::activation(3 * cached_width, 31);
  for (int step = 0; step < 4; ++step) {
    auto input = d5_gguf_test::activation(model.hidden, 3 + 14 * step);
    sem::Replay replay{{bytes(input), bytes(past)}, {}};
    auto xv = sem::oracle::rms_norm_rows(input, input_scale, model.hidden, kNormEpsilon);
    auto av = qa.reference(xv);
    av = sem::oracle::rms_norm_rows(av, qscale, model.query_rank, kNormEpsilon);
    auto qv = qb.reference(av), kv = kva.reference(xv);
    auto lv = sem::oracle::rms_norm_rows(std::span(kv).first(kLatent), kvscale, kLatent, kNormEpsilon);
    auto qn = kimi::slice(qv, kHeads, 256, 0, no_pe);
    auto absorbed = kb.reference(qn);
    if (rotary) {
      std::vector<float> cs(32), sn(32);
      for (std::size_t p = 0; p < cs.size(); ++p) {
        const double angle = double(13 + step) * frequency_values[p];
        cs[p] = float(std::cos(angle)); sn[p] = float(std::sin(angle));
      }
      replay.inputs.push_back(bytes(std::vector<std::int32_t>{13 + step}));
      auto qp = kimi::slice(qv, kHeads, 256, no_pe, 64);
      std::vector<float> kp(kv.begin() + kLatent, kv.end());
      rotate_pairs(qp, cs, sn); rotate_pairs(kp, cs, sn);
      lv.insert(lv.end(), kp.begin(), kp.end());
      std::vector<float> joined;
      for (std::int64_t h = 0; h < kHeads; ++h) {
        joined.insert(joined.end(), absorbed.begin() + h * kLatent, absorbed.begin() + (h + 1) * kLatent);
        joined.insert(joined.end(), qp.begin() + h * 64, qp.begin() + (h + 1) * 64);
      }
      absorbed = std::move(joined);
    }
    auto full = past; full.insert(full.end(), lv.begin(), lv.end());
    past.assign(full.begin() + cached_width, full.end());
    std::vector<float> attended(kHeads * kLatent);
    for (std::int64_t h = 0; h < kHeads; ++h) {
      std::array<double, 4> score{};
      for (int t = 0; t < 4; ++t)
        for (std::int64_t d = 0; d < cached_width; ++d)
          score[t] += double(absorbed[h * cached_width + d]) * full[t * cached_width + d] / 16.0;
      const double largest = *std::max_element(score.begin(), score.end());
      double total = 0;
      for (auto &value : score) total += value = std::exp(value - largest);
      for (std::int64_t d = 0; d < kLatent; ++d) {
        double sum = 0;
        for (int t = 0; t < 4; ++t) sum += score[t] / total * full[t * cached_width + d];
        attended[h * kLatent + d] = float(sum);
      }
    }
    auto projected = vb.reference(attended);
    replay.outputs = {past, qv, absorbed, attended, projected, out.reference(projected)};
    f.replays.push_back(std::move(replay));
  }
  return f;
}
} // namespace d5_glm_remaining_fragments
