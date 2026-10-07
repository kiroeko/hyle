#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS) || defined(HYLE_GGML_CUDA_IQ1_EXTENSIONS)
#include "cpu_c5_qwen_attention.h"

namespace cpu_c5_qwen_nextn {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace ds = d5_deepseek_official_attention;
namespace in = cpu_c5_inputs;
using sem::Fragment;
using sem::ValueId;
using sem::dense;
using sem::bytes;
inline constexpr std::int64_t hidden = 8192, heads = 64, kv_heads = 4, depth = 256,
                              intermediate = 2048, experts = 512, selected = 10, resident = 8, vocabulary = 248320;
struct Linear {
  in::Case spec;
  cpu_c5_gguf_fixture::Bank bank;
  ValueId output;
  Linear(Fragment &f, ValueId input, std::int64_t n, std::int64_t k,
         c::GgufQuantizedType quant, ggml_type native, bool packed = true)
      : spec{"NextNProjection", in::Kind::kProjection, native, quant, packed, k, n, 1, 0, 0}, bank(spec) {
    auto desc = packed ? c::gguf_weight_description(quant, c::DataType::kFloat32, k, n).value()
                       : dense({n, k}).transposed().value();
    auto weight = f.constant(desc, bank.rows(n));
    output = f.one(c::Operation(c::MatMul{}), {input, weight});
  }
  std::vector<float> reference(std::span<const float> input, std::int64_t shift = 0) const {
    const auto dot = bank.dots(input); std::vector<float> result(spec.n);
    for (std::int64_t row = 0; row < spec.n; ++row) result[row] = dot[(row + shift) % cpu_c5_gguf_fixture::prototypes];
    return result;
  }
};
struct Grouped {
  in::Case spec;
  cpu_c5_gguf_fixture::Bank bank;
  ValueId output;
  Grouped(Fragment &f, ValueId input, ValueId ids, std::int64_t n, std::int64_t k,
          c::GgufQuantizedType quant, ggml_type native)
      : spec{"NextNExperts", in::Kind::kExpert, native, quant, true, k, n, resident, 0, 0}, bank(spec) {
    std::vector<std::byte> stored;
    stored.reserve(resident * n * bank.stride);
    for (std::int64_t expert = 0; expert < resident; ++expert) {
      auto part = bank.rows(n, expert); stored.insert(stored.end(), part.begin(), part.end());
    }
    auto desc = c::gguf_grouped_weight_description(quant, c::DataType::kFloat32, resident, n, k).value();
    auto weight = f.constant(desc, std::move(stored));
    output = f.one(c::model_ops::grouped_matmul(), {weight, input, ids});
  }
  std::vector<float> reference(std::span<const float> input, std::span<const std::int32_t> ids) const {
    std::vector<float> result(ids.size() * spec.n);
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
      const auto dot = bank.dots(input.subspan(slot * spec.k, spec.k));
      for (std::int64_t row = 0; row < spec.n; ++row)
        result[slot * spec.n + row] = dot[(row + ids[slot]) % cpu_c5_gguf_fixture::prototypes];
    }
    return result;
  }
};
inline std::vector<float> gated(std::span<const float> gate, std::span<const float> up) {
  std::vector<float> result(gate.size());
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = float(double(gate[i]) / (1 + std::exp(-double(gate[i]))) * up[i]);
  return result;
}

inline Fragment decoder(const c::ModelArtifactProfileDescription *identity,
                        std::span<const std::vector<float>> actual_embedding,
                        std::span<const std::vector<float>> expected_embedding) {
  const bool mixed = identity->variant == "UD-IQ1_M";
  const auto attention_quant = mixed ? c::GgufQuantizedType::kQ5K : c::GgufQuantizedType::kQ8_0;
  const auto attention_native = mixed ? GGML_TYPE_Q5_K : GGML_TYPE_Q8_0;
  const auto expert_quant = mixed ? c::GgufQuantizedType::kQ2K : c::GgufQuantizedType::kQ8_0;
  const auto expert_native = mixed ? GGML_TYPE_Q2_K : GGML_TYPE_Q8_0;
  Fragment f{.profile = identity, .region = 0x0d0c0510, .quantized = true};
  auto h = f.input(dense({1, hidden})), embedding = f.input(dense({1, hidden}));
  auto past_k = f.input(dense({1, 2, kv_heads, depth})), past_v = f.input(dense({1, 2, kv_heads, depth}));
  auto position = f.input(dense({1}, c::DataType::kInt32));
  auto hs = sem::oracle::scale(hidden), es = hs, as = hs, fs = hs, ns = hs;
  std::reverse(es.begin(), es.end()); std::rotate(as.begin(), as.begin() + 3, as.end());
  std::rotate(fs.begin(), fs.begin() + 5, fs.end()); std::rotate(ns.begin(), ns.begin() + 7, ns.end());
  auto qs = sem::oracle::scale(depth), ks = qs; std::reverse(ks.begin(), ks.end());
  const auto norm = [&](ValueId x, const std::vector<float> &scales) {
    auto w = f.constant(dense({std::int64_t(scales.size())}), bytes(scales));
    return f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}), {x, w});
  };
  auto joined = ds::concat(f, norm(embedding, es), norm(h, hs), 1);
  Linear eh(f, joined, hidden, 2 * hidden, c::GgufQuantizedType::kQ8_0, GGML_TYPE_Q8_0);
  auto normalized = norm(eh.output, as);
  Linear qg(f, normalized, 2 * heads * depth, hidden, attention_quant, attention_native);
  Linear key(f, normalized, kv_heads * depth, hidden, attention_quant, attention_native);
  Linear value(f, normalized, kv_heads * depth, hidden, attention_quant, attention_native);
  auto qg_heads = core::reshape(f, qg.output, {1, 1, heads, 2 * depth});
  auto query = norm(ds::slice(f, qg_heads, 3, 0, depth), qs);
  auto gate = ds::slice(f, qg_heads, 3, depth, depth);
  auto current_k = norm(core::reshape(f, key.output, {1, 1, kv_heads, depth}), ks);
  std::vector<float> frequencies(32);
  for (std::size_t pair = 0; pair < frequencies.size(); ++pair) frequencies[pair] = 1.0F / std::pow(10000000.0F, float(pair) / 32);
  auto freq = f.constant(dense({32}), bytes(frequencies));
  const c::Operation rope(c::RoPE{.rotary_dimensions = 64, .sequence_axis = 1, .layout = c::RopeLayout::kHalfSplit});
  query = f.one(rope, {query, position, freq}); current_k = f.one(rope, {current_k, position, freq});
  auto keys = ds::concat(f, past_k, current_k, 1);
  auto values = ds::concat(f, past_v, core::reshape(f, value.output, {1, 1, kv_heads, depth}), 1);
  auto attention = f.one(c::Operation(c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchSequenceHeadDepth, .scale = 0.0625F,
      .causal = true, .accumulation_type = c::DataType::kFloat32}), {query, keys, values});
  auto attended = core::binary(f, c::BinaryKind::kMultiply, attention, core::unary(f, c::UnaryKind::kSigmoid, gate));
  Linear out(f, core::reshape(f, attended, {1, heads * depth}), hidden, heads * depth, attention_quant, attention_native);
  auto residual = core::binary(f, c::BinaryKind::kAdd, eh.output, out.output);
  auto ffn_input = norm(residual, fs);
  Linear router(f, ffn_input, experts, hidden, c::GgufQuantizedType::kQ8_0, GGML_TYPE_F32, false);
  auto probabilities = f.one(c::Operation(c::Softmax{}), {router.output});
  auto top = f.append(c::Operation(c::TopK{.count = std::size_t(selected)}), {probabilities}, 2);
  auto total = d5_moe_execution_fragments::sum(f, top[0], -1);
  auto weights = core::binary(f, c::BinaryKind::kDivide, top[0], total);
  std::vector<std::int32_t> mapping(experts);
  for (std::int64_t index = 0; index < experts; ++index) mapping[index] = std::int32_t(index % resident);
  auto map = f.constant(dense({experts}, c::DataType::kInt32), bytes(mapping));
  auto local_ids = f.one(c::Operation(c::Gather{}), {map, top[1]});
  auto replicate = f.constant(dense({selected}, c::DataType::kInt32), bytes(std::vector<std::int32_t>(selected)));
  auto slots = core::reshape(f, f.one(c::Operation(c::Gather{}), {ffn_input, replicate}), {1, selected, hidden});
  Grouped eg(f, slots, local_ids, intermediate, hidden, expert_quant, expert_native);
  Grouped eu(f, slots, local_ids, intermediate, hidden, expert_quant, expert_native);
  auto activated = f.one(c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}), {eg.output, eu.output});
  Grouped ed(f, activated, local_ids, hidden, intermediate, expert_quant, expert_native);
  auto weighted = core::binary(f, c::BinaryKind::kMultiply, ed.output, core::reshape(f, weights, {1, selected, 1}));
  auto combined = core::reshape(f, d5_moe_execution_fragments::sum(f, weighted, 1), {1, hidden});
  Linear sg(f, ffn_input, intermediate, hidden, attention_quant, attention_native);
  Linear su(f, ffn_input, intermediate, hidden, attention_quant, attention_native);
  auto shared_activation = f.one(c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}), {sg.output, su.output});
  Linear sd(f, shared_activation, hidden, intermediate, mixed ? c::GgufQuantizedType::kQ6K : attention_quant,
            mixed ? GGML_TYPE_Q6_K : attention_native);
  Linear shared_gate(f, ffn_input, 1, hidden, c::GgufQuantizedType::kQ8_0, GGML_TYPE_F32, false);
  auto shared = core::binary(f, c::BinaryKind::kMultiply, sd.output, core::unary(f, c::UnaryKind::kSigmoid, shared_gate.output));
  auto final = norm(core::binary(f, c::BinaryKind::kAdd, residual,
      core::binary(f, c::BinaryKind::kAdd, combined, shared)), ns);
  f.outputs = {final, ds::slice(f, keys, 1, 1, 2), ds::slice(f, values, 1, 1, 2)};
  f.state_ports = {{0, 0}, {2, 1}, {3, 2}};
  auto state = d5_gguf_test::activation(hidden, 3);
  auto history_k = d5_gguf_test::values(2 * kv_heads, depth, 5), history_v = history_k;
  for (int step = 0; step < 4; ++step) {
    const auto &actual_e = actual_embedding[step]; const auto &expected_e = expected_embedding[step];
    const std::array<std::int32_t, 1> positions{37 + step};
    auto input = sem::oracle::rms_norm_rows(expected_e, es, hidden, 1.0e-6F);
    auto hn = sem::oracle::rms_norm_rows(state, hs, hidden, 1.0e-6F); input.insert(input.end(), hn.begin(), hn.end());
    auto projected = eh.reference(input), n = sem::oracle::rms_norm_rows(projected, as, hidden, 1.0e-6F);
    auto qgv = qg.reference(n); std::vector<float> q(heads * depth), g(q.size());
    for (std::int64_t head = 0; head < heads; ++head) for (std::int64_t d = 0; d < depth; ++d) {
      q[head * depth + d] = qgv[head * 2 * depth + d]; g[head * depth + d] = qgv[head * 2 * depth + depth + d];
    }
    q = cpu_c5_qwen_attention::rotate(sem::oracle::rms_norm_rows(q, qs, depth, 1.0e-6F), positions[0]);
    auto kr = cpu_c5_qwen_attention::rotate(sem::oracle::rms_norm_rows(key.reference(n), ks, depth, 1.0e-6F), positions[0]);
    auto vr = value.reference(n), all_k = history_k, all_v = history_v;
    all_k.insert(all_k.end(), kr.begin(), kr.end()); all_v.insert(all_v.end(), vr.begin(), vr.end());
    auto context = cpu_c5_qwen_attention::attention_reference(q, all_k, all_v, g, false)[1];
    auto rv = out.reference(context); for (std::size_t index = 0; index < rv.size(); ++index) rv[index] += projected[index];
    auto fn = sem::oracle::rms_norm_rows(rv, fs, hidden, 1.0e-6F);
    auto logits = router.reference(fn);
    std::vector<std::int32_t> order(experts); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return logits[a] > logits[b]; });
    std::vector<std::int32_t> ids(selected); std::vector<double> score(selected); double denominator = 0;
    for (std::int64_t slot = 0; slot < selected; ++slot) {
      ids[slot] = order[slot] % resident; score[slot] = std::exp(double(logits[order[slot]]) - logits[order[0]]); denominator += score[slot];
    }
    std::vector<float> replicated; for (std::int64_t slot = 0; slot < selected; ++slot) replicated.insert(replicated.end(), fn.begin(), fn.end());
    auto gv = eg.reference(replicated, ids), uv = eu.reference(replicated, ids);
    auto routed = ed.reference(gated(gv, uv), ids); auto shared_value = sd.reference(gated(sg.reference(fn), su.reference(fn)));
    const auto sigmoid = 1.0 / (1.0 + std::exp(-double(shared_gate.reference(fn)[0])));
    for (std::int64_t d = 0; d < hidden; ++d) {
      double combined_value = 0; for (std::int64_t slot = 0; slot < selected; ++slot) combined_value += routed[slot * hidden + d] * score[slot] / denominator;
      rv[d] += float(combined_value + shared_value[d] * sigmoid);
    }
    auto next = sem::oracle::rms_norm_rows(rv, ns, hidden, 1.0e-6F);
    auto next_k = std::vector<float>(all_k.begin() + kv_heads * depth, all_k.end());
    auto next_v = std::vector<float>(all_v.begin() + kv_heads * depth, all_v.end());
    f.replays.push_back({{bytes(state), bytes(actual_e), bytes(history_k), bytes(history_v), bytes<std::int32_t>(positions)}, {next, next_k, next_v}});
    state = std::move(next); history_k = std::move(next_k); history_v = std::move(next_v);
  }
  return f;
}

template <class Run> void check(unsigned profile, Run run) {
  const auto identity = cpu_c5_gguf_fixture::profile(profile);
  const bool mixed = identity.variant == "UD-IQ1_M";
  const auto quant = mixed ? c::GgufQuantizedType::kQ4K : c::GgufQuantizedType::kQ8_0;
  const auto native = mixed ? GGML_TYPE_Q4_K : GGML_TYPE_Q8_0;
  const in::Case embedding_spec{"NextNEmbedding", in::Kind::kEmbedding, native, quant, true, hidden, vocabulary, 1, 0, 0};
  const cpu_c5_gguf_fixture::Bank bank(embedding_spec);
  auto lookup = cpu_c5_gguf_fixture::lookup(embedding_spec, bank, &identity);
  lookup.replays.clear(); lookup.invalid_index_input.reset();
  std::vector<std::vector<float>> observed_embedding, expected_embedding;
  for (int step = 0; step < 4; ++step) {
    const std::array<std::int32_t, 4> ids{step, std::int32_t(vocabulary - 1), std::int32_t(vocabulary / 2), std::int32_t(vocabulary / 2)};
    std::vector<float> expected(4 * hidden);
    for (int row = 0; row < 4; ++row) std::copy_n(bank.decoded.data() + (ids[row] % cpu_c5_gguf_fixture::prototypes) * hidden, hidden, expected.data() + row * hidden);
    expected_embedding.emplace_back(expected.begin(), expected.begin() + hidden);
    lookup.replays.push_back({{bytes<std::int32_t>(ids)}, {std::move(expected)}});
  }
  lookup.observe = [&](std::size_t, std::span<const float> values) { observed_embedding.emplace_back(values.begin(), values.begin() + hidden); };
  run(std::move(lookup)); ASSERT_FALSE(::testing::Test::HasFatalFailure()); ASSERT_EQ(observed_embedding.size(), 4u);
  auto fragment = decoder(&identity, observed_embedding, expected_embedding);
  std::vector<std::vector<float>> observed_hidden, expected_hidden;
  for (const auto &replay : fragment.replays) expected_hidden.push_back(replay.outputs[0]);
  fragment.observe = [&](std::size_t port, std::span<const float> values) { if (port == 0) observed_hidden.emplace_back(values.begin(), values.end()); };
  run(std::move(fragment)); ASSERT_FALSE(::testing::Test::HasFatalFailure()); ASSERT_EQ(observed_hidden.size(), 4u);
  auto head_spec = embedding_spec; head_spec.kind = in::Kind::kHead;
  for (std::int64_t first = 0; first < vocabulary; first += 4096) {
    auto count = std::min<std::int64_t>(4096, vocabulary - first);
    auto head = cpu_c5_gguf_fixture::projection(head_spec, bank, &identity, count, first); head.quantized = true; head.replays.clear();
    for (std::size_t step = 0; step < observed_hidden.size(); ++step) {
      auto dot = bank.dots(expected_hidden[step]); std::vector<float> expected(count);
      for (std::int64_t row = 0; row < count; ++row) expected[row] = dot[(first + row) % cpu_c5_gguf_fixture::prototypes];
      head.replays.push_back({{bytes(observed_hidden[step])}, {std::move(expected)}});
    }
    run(std::move(head)); ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
} // namespace cpu_c5_qwen_nextn
#endif
