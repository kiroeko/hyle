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

// Full expert selection, original A8 boundary, packed FP4 gate/up/down and
// shared block128 FP8. The route weight belongs before down's quantization.
inline Fragment moe_chain(const moe::Case &model, bool use_hash = false,
                           moe::MoeReference *reference_out = nullptr) {
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
        f.input(sem::dense({1}, c::DataType::kInt64)),
        f.input(sem::dense({moe::kHashVocabulary, 6}, c::DataType::kInt32))};
  auto route = moe::routing(f, model, x, true, 64.0F, lookup);
  const auto resident = model.top_k + 1;
  auto quantized_x = roundtrip(f, x, quant::Format::kFloat8E4M3Block128);
  const auto replicate = f.constant(sem::dense({model.top_k}, c::DataType::kInt32),
      sem::bytes(std::vector<std::int32_t>(model.top_k, 0)));
  auto slots = core::reshape(f, f.one(c::Operation(c::Gather{}), {quantized_x, replicate}),
                             {1, model.top_k, model.hidden});
  auto wg = d5_official_fp4_storage::weight(resident, model.intermediate, model.hidden, 71);
  auto wu = d5_official_fp4_storage::weight(resident, model.intermediate, model.hidden, 79);
  auto wd = d5_official_fp4_storage::weight(resident, model.hidden, model.intermediate, 83);
  const auto grouped = [&](sem::Weight &w, ValueId value) {
    const auto stored = f.constant(w.description, std::move(w.storage));
    return bfloat_boundary(f, f.one(c::model_ops::grouped_matmul(), {stored, value, route.ids}));
  };
  auto gate = grouped(wg, slots), up = grouped(wu, slots);
  auto activated = moe::gated(f, model, gate, up);
  auto weighted = core::binary(f, c::BinaryKind::kMultiply, activated,
                               core::reshape(f, route.weights, {1, model.top_k, 1}));
  auto down_input = roundtrip(f, core::reshape(f, weighted, {model.top_k, model.intermediate}),
                              quant::Format::kFloat8E4M3Block128);
  auto routed = grouped(wd, core::reshape(f, down_input, {1, model.top_k, model.intermediate}));
  auto combined = core::reshape(f, moe::sum(f, routed, 1), {1, model.hidden});

  auto sg = fp8::model_linear(f, x, model.shared_intermediate, model.hidden, 113, true);
  auto su = fp8::model_linear(f, x, model.shared_intermediate, model.hidden, 127, true);
  auto shared_activated = moe::gated(f, model, sg.output, su.output);
  auto sd = fp8::model_linear(f, shared_activated, model.hidden, model.shared_intermediate, 139, true);
  auto output = bfloat_boundary(f, core::binary(f, c::BinaryKind::kAdd, combined, sd.output));
  f.outputs = {moe::gathered_ids(f, model, route.ids), route.weights, routed, combined,
                sd.output, output, gate, up, activated, sg.output, su.output,
                shared_activated, quantized_x, down_input};
  f.exact_outputs = {0, 12};
  // Error includes the mature packed FP4 executor's own activation conversion;
  // do not replace the source oracle with that executor's internal Q8 values.
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
            std::move(shared_act), std::move(a8), std::move(down_ref)};
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
      f.replays.push_back({{sem::bytes(values), moe::hash_tokens(token, true), sem::bytes(table)},
                            std::move(expected)});
    } else f.replays.push_back({{sem::bytes(values)}, evaluate(values, {})});
  }
  return f;
}
} // namespace d5_deepseek_official_fragments
