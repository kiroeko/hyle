#pragma once

#include "d5_quantization_roundtrip.h"

// Official DeepSeek V4 consumer recipes. Host arithmetic below is only the
// independent oracle; every executed node calls an existing mature Provider.
namespace d5_deepseek_official_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace moe = d5_moe_execution_fragments;
namespace core = d5_moe_semantic_fragments;
namespace fp8 = d5_dynamic_fp8_fragments;
namespace quant = d5_quantization_roundtrip;
using sem::ValueId;
using sem::Fragment;

inline ValueId bfloat_boundary(Fragment &f, ValueId value) {
  auto stored = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {value});
  return fp8::visible(f, stored, true);
}
inline std::vector<float> bfloat_boundary(std::vector<float> values) {
  for (auto &x : values) x = fp8::bfloat(x);
  return values;
}
inline ValueId roundtrip(Fragment &f, ValueId value, quant::Format format) {
  value = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {value});
  return fp8::visible(f, f.one(c::model_ops::block_quantization_roundtrip(format), {value}), true);
}

// Full expert selection, original A8 boundary, original FP4 gate/up/down and
// shared block128 FP8. The route weight belongs before down's quantization.
inline Fragment moe_chain(const moe::Case &model, bool use_hash = false,
                           moe::MoeReference *reference_out = nullptr,
                           bool native_routing = true) {
  if (model.gguf || (model.profile != moe::cases::kCases[3].profile &&
                    model.profile != moe::cases::kCases[4].profile) ||
      (use_hash && reference_out))
    throw std::runtime_error("official DeepSeek MoE profile/replay mismatch");
  Fragment f{.profile = model.profile, .region = use_hash ? 0x0a0b0602u : 0x0a0b0601u,
              .quantized = true};
  const auto input = f.input(sem::dense({1, model.hidden}));
  auto x = bfloat_boundary(f, input);
  std::optional<moe::HashInputs> lookup;
  if (use_hash)
    lookup = moe::HashInputs{
        f.input(sem::dense({1}, native_routing ? c::DataType::kInt64
                                              : c::DataType::kInt32)),
        f.input(sem::dense({moe::kHashVocabulary, 6}, c::DataType::kInt32))};
  auto route = moe::routing(f, model, x, native_routing, 64.0F, lookup);
  const auto resident = model.top_k + 1;
  auto slots = core::reshape(f, x, {1, 1, model.hidden});
  auto wg = d5_official_fp4_storage::weight(resident, model.intermediate, model.hidden, 71, true);
  auto wu = d5_official_fp4_storage::weight(resident, model.intermediate, model.hidden, 79, true);
  auto wd = d5_official_fp4_storage::weight(resident, model.hidden, model.intermediate, 83, true);
  const auto grouped = [&](sem::Weight &w, ValueId value) {
    const auto stored = f.constant(w.description, std::move(w.storage));
    value = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {value});
    return fp8::visible(f, f.one(c::model_ops::grouped_float4_linear(), {stored, value, route.ids}), true);
  };
  auto gate = grouped(wg, slots), up = grouped(wu, slots);
  auto activated = moe::gated(f, model, gate, up);
  auto weighted = core::binary(f, c::BinaryKind::kMultiply, activated,
                               core::reshape(f, route.weights, {1, model.top_k, 1}));
  auto routed = grouped(wd, weighted);
  auto combined = f.one(c::Operation(c::Copy{}),
      {core::reshape(f, moe::sum(f, routed, 1), {1, model.hidden})});

  auto sg = fp8::model_linear(f, x, model.shared_intermediate, model.hidden, 113, true);
  auto su = fp8::model_linear(f, x, model.shared_intermediate, model.hidden, 127, true);
  auto shared_activated = moe::gated(f, model, sg.output, su.output);
  auto sd = fp8::model_linear(f, shared_activated, model.hidden, model.shared_intermediate, 139, true);
  auto output = bfloat_boundary(f, core::binary(f, c::BinaryKind::kAdd, combined, sd.output));
  f.outputs = {moe::gathered_ids(f, model, route.ids), route.weights, routed, combined,
                sd.output, output, gate, up, activated, sg.output, su.output,
                shared_activated};
  f.exact_outputs = {0};
  // Keep the established full-chain gate and independent source decoding.
  // The failed ggml composition and its extra Q8 conversion remain in F43 evidence.
  auto evaluate = [model, sg, su, sd, wg = std::move(wg), wu = std::move(wu), wd = std::move(wd)]
      (std::span<const float> incoming, std::span<const std::int32_t> hash_ids) -> moe::ReferenceOutputs {
    const auto source = bfloat_boundary(std::vector<float>(incoming.begin(), incoming.end()));
    auto route_ref = moe::route_reference(model, source, 64.0F, hash_ids);
    auto a8 = quant::reference(source, false, 128);
    std::vector<float> replicated;
    for (std::int64_t slot = 0; slot < model.top_k; ++slot)
      replicated.insert(replicated.end(), a8.begin(), a8.end());
    auto gate_ref = bfloat_boundary(moe::project(replicated, wg, route_ref.ids, model.intermediate, model.hidden));
    auto up_ref = bfloat_boundary(moe::project(replicated, wu, route_ref.ids, model.intermediate, model.hidden));
    auto act = moe::gate_reference(model, gate_ref, up_ref);
    auto weighted_ref = act;
    for (std::int64_t slot = 0; slot < model.top_k; ++slot)
      for (std::int64_t i = 0; i < model.intermediate; ++i)
        weighted_ref[slot * model.intermediate + i] *= route_ref.weights[slot];
    auto down_ref = quant::reference(bfloat_boundary(std::move(weighted_ref)), false, 128);
    auto routed_ref = bfloat_boundary(moe::project(down_ref, wd, route_ref.ids, model.hidden, model.intermediate));
    std::vector<float> combined_ref(model.hidden);
    // Official accumulation iterates experts by ID. All terms here are BF16
    // and at most six, so this fixed F32 sum preserves its rounding boundary.
    auto order = route_ref.ids;
    std::sort(order.begin(), order.end());
    for (auto id : order) {
      const auto slot = std::find(route_ref.ids.begin(), route_ref.ids.end(), id) - route_ref.ids.begin();
      for (std::int64_t i = 0; i < model.hidden; ++i)
        combined_ref[i] += routed_ref[slot * model.hidden + i];
    }
    auto shared_gate = sg.reference(source), shared_up = su.reference(source);
    auto shared_act = moe::gate_reference(model, shared_gate, shared_up);
    auto shared = sd.reference(shared_act);
    auto full = combined_ref;
    for (std::int64_t i = 0; i < model.hidden; ++i) full[i] = fp8::bfloat(full[i] + shared[i]);
    return {std::move(route_ref.tags), std::move(route_ref.weights), std::move(routed_ref),
            std::move(combined_ref), std::move(shared), std::move(full), std::move(gate_ref),
            std::move(up_ref), std::move(act), std::move(shared_gate), std::move(shared_up),
            std::move(shared_act)};
  };
  if (reference_out) {
    *reference_out = [evaluate = std::move(evaluate)](std::span<const float> values) {
      return evaluate(values, {});
    };
    return f;
  }
  for (int replay = 0; replay < 3; ++replay) {
    auto values = moe::activation(model, replay);
    if (use_hash) {
      const std::array<std::int64_t, 3> tokens{0, moe::kHashVocabulary - 1, 37};
      const std::array token{tokens[replay]};
      auto table = moe::hash_table(resident, replay);
      auto expected = evaluate(values, std::span<const std::int32_t>(table).subspan(token[0] * 6, 6));
      f.replays.push_back({{sem::bytes(values), moe::hash_tokens(token, native_routing), sem::bytes(table)},
                            std::move(expected)});
    } else f.replays.push_back({{sem::bytes(values)}, evaluate(values, {})});
  }
  return f;
}

inline Fragment projection(const moe::Case &model, bool down, bool shared,
                            std::int64_t tokens) {
  const auto n = down ? model.hidden : model.intermediate;
  const auto k = down ? model.intermediate : model.hidden;
  const auto slots = model.top_k, a = shared ? 1 : slots, experts = slots + 1;
  Fragment f{.profile = model.profile, .region = 0x0b0607};
  auto weight = d5_official_fp4_storage::weight(experts, n, k, down ? 83 : 71, true);
  auto w = f.constant(weight.description, std::move(weight.storage));
  auto input = f.input(sem::dense({tokens, a, k}, c::DataType::kBFloat16));
  auto ids = f.input(sem::dense({tokens, slots}, c::DataType::kInt32));
  auto output = f.one(c::model_ops::grouped_float4_linear(), {w, input, ids});
  f.outputs = {fp8::visible(f, output, true)};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    auto values = bfloat_boundary(d5_gguf_test::activation(tokens * a * k, 17 + replay * 31));
    const auto quantized = quant::reference(values, false, 128);
    std::vector<std::int32_t> selected(tokens * slots);
    std::vector<float> expanded;
    for (std::int64_t slot = 0; slot < tokens * slots; ++slot) {
      selected[slot] = std::int32_t((slot * (replay ? 2 : 1) + replay) % experts);
      const auto row = shared ? slot / slots : slot;
      expanded.insert(expanded.end(), quantized.begin() + row * k, quantized.begin() + (row + 1) * k);
    }
    if (replay) selected[1] = selected[0];
    auto expected = bfloat_boundary(moe::project(expanded, weight, selected, n, k));
    f.replays.push_back({{fp8::input_bytes(values, true), sem::bytes(selected)}, {std::move(expected)}});
  }
  return f;
}
} // namespace d5_deepseek_official_fragments
