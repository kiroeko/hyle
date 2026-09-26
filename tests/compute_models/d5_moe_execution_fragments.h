#pragma once

#include <numeric>
#include <limits>

#include "d5_moe_execution_cases.h"
#include "d5_moe_semantic_fragments.h"
#include "d5_official_fp4_storage.h"

// Model consumer recipes and independent numerical oracles, never product math.
namespace d5_moe_execution_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace cases = d5_moe_execution_cases;
using cases::Case;
using cases::Storage;
using sem::bytes;
using sem::dense;
using sem::Fragment;
using sem::ValueId;
using core::binary;
using core::filled;
using core::reshape;
using core::unary;

inline ValueId bound(Fragment &f, ValueId input, float limit, bool upper) {
  const auto infinity = std::numeric_limits<float>::infinity();
  return f.one(c::model_ops::clamp(upper ? -infinity : limit,
                                  upper ? limit : infinity), {input});
}

inline Fragment clamp_boundaries() {
  Fragment f{.profile = cases::kCases[0].profile, .region = 0x0a0603};
  const auto inf = std::numeric_limits<float>::infinity();
  std::vector<float> x{-inf, -11, -10, -0.25F, 0.25F, 10, 11, inf};
  auto input = f.input(dense({1, 2, 2, 2}));
  f.outputs = {f.one(c::model_ops::clamp(-10, 10), {input})};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<float> expected;
    for (float value : x)
      expected.push_back(std::clamp(value, -10.0F, 10.0F));
    f.replays.push_back({{bytes(x)}, {std::move(expected)}});
    std::reverse(x.begin(), x.end());
  }
  return f;
}

inline ValueId sum(Fragment &f, ValueId input, std::int32_t axis) {
  return f.one(c::Operation(c::Reduce{
                   .kind = c::ReductionKind::kSum,
                   .axes = c::Axes::from(std::array{axis}).value(),
                   .keep_dimensions = true,
                   .accumulation_type = c::DataType::kFloat32}), {input});
}
inline ValueId gathered_ids(Fragment &f, const Case &model, ValueId ids) {
  std::vector<float> tags(model.experts);
  std::iota(tags.begin(), tags.end(), 0.0F);
  auto stored = f.constant(dense({model.experts, 1}), bytes(tags));
  return f.one(c::Operation(c::Gather{}), {stored, ids});
}

struct Routing final { ValueId ids, weights; };
inline Routing routing(Fragment &f, const Case &model, ValueId input,
                        bool native_routing = false) {
  const auto resident = model.top_k + 1;
  auto desc = dense({model.experts, model.hidden}, model.router_bf16
                    ? c::DataType::kBFloat16 : c::DataType::kFloat32);
  std::vector<std::byte> storage(desc.required_bytes().value());
  for (std::int64_t e = 0; e < resident; ++e) {
    const auto offset = e * model.hidden + e;
    if (model.router_bf16) {
      const auto value = sem::q::encode(64.0);
      std::memcpy(storage.data() + offset * sizeof(value), &value, sizeof(value));
    } else {
      const float value = 64.0F;
      std::memcpy(storage.data() + offset * sizeof(value), &value, sizeof(value));
    }
  }
  ValueId logits;
  if (model.router_bf16) {
    auto x = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {input});
    auto weight = f.constant(desc, std::move(storage));
    logits = f.one(c::model_ops::bfloat_linear_f32(), {x, weight});
  } else {
    auto weight = f.constant(desc.transposed().value(), std::move(storage));
    logits = f.one(c::Operation(c::MatMul{}), {input, weight});
  }
  std::vector<float> bias(model.experts, -4.0F);
  for (std::int64_t e = 0; e < resident; ++e)
    bias[e] = float(e % 3 - 1) / 16.0F;
  if (model.sqrt_softplus)
    for (auto &value : bias)
      value = float(sem::q::decode(sem::q::encode(value)));
  if (model.sqrt_softplus && native_routing) {
    auto correction = f.constant(dense({model.experts}), bytes(bias));
    auto outputs = f.append(c::model_ops::sqrt_softplus_routing(model.scaling),
                            {logits, correction}, 2);
    return {outputs[1], outputs[0]};
  }
  auto scores = model.sqrt_softplus
      ? unary(f, c::UnaryKind::kSqrt, unary(f, c::UnaryKind::kSoftplus, logits))
      : unary(f, c::UnaryKind::kSigmoid, logits);
  auto correction = f.constant(dense({1, model.experts}), bytes(bias));
  auto selection = binary(f, c::BinaryKind::kAdd, scores, correction);
  auto top = f.append(c::Operation(c::TopK{.count = std::size_t(model.top_k)}),
                       {selection}, 2);
  auto score_rows = reshape(f, scores, {model.experts, 1});
  auto selected = reshape(f, f.one(c::Operation(c::Gather{}),
                                    {score_rows, top[1]}), {1, model.top_k});
  auto total = sum(f, selected, -1);
  // DeepSeek's positive-sum normalization retains tiny nonzero scores.
  // sqrt(nonzero F32 softplus) exceeds the F32 normal floor; zero stays zero.
  total = bound(f, total, model.sqrt_softplus
                    ? std::numeric_limits<float>::min() : 6.103515625e-5F, false);
  auto normalized = binary(f, c::BinaryKind::kDivide, selected, total);
  normalized = binary(f, c::BinaryKind::kMultiply, normalized,
                       filled(f, dense({1, model.top_k}), model.scaling));
  return {top[1], normalized};
}

inline std::vector<float> activation(const Case &model, int replay) {
  auto values = d5_gguf_test::activation(model.hidden, 3 + 14 * replay);
  // BF16-representable common input for both F32 and BF16 router weights.
  // Keep the FFN activation scale balanced. The router's 64x weight creates
  // the selected logits, rather than synthetic 3-5 outliers among 0.06 values.
  // Those outliers triggered the native Q8 activation quantizer (retained
  // diagnostic evidence), not a routing or weight-format disagreement.
  for (auto &v : values)
    v = float(sem::q::decode(sem::q::encode(v)));
  for (std::int64_t e = 0; e <= model.top_k; ++e)
    values[e] = (2.0F + float(replay == 1 ? model.top_k - e : e) / 8.0F) / 64.0F;
  if (replay == 2)
    values[0] = 5.0F / 64.0F;
  return values;
}
struct RouteReference final {
  std::vector<std::int32_t> ids;
  std::vector<float> scores, weights, tags;
};
inline RouteReference route_reference(const Case &model,
                                       std::span<const float> input) {
  RouteReference r;
  r.scores.resize(model.experts);
  r.ids.resize(model.experts);
  std::iota(r.ids.begin(), r.ids.end(), 0);
  std::vector<double> selection(model.experts);
  for (std::int64_t e = 0; e < model.experts; ++e) {
    const double logit = e <= model.top_k ? input[e] * 64.0 : 0;
    const double score = model.sqrt_softplus ? std::sqrt(std::log1p(std::exp(logit)))
                                             : core::sigmoid(logit);
    r.scores[e] = float(score);
    selection[e] = score + (e <= model.top_k ? double(e % 3 - 1) / 16.0 : -4.0);
  }
  std::stable_sort(r.ids.begin(), r.ids.end(), [&](auto a, auto b) {
    return selection[a] > selection[b];
  });
  r.ids.resize(model.top_k);
  double denominator = 0;
  for (auto id : r.ids)
    denominator += r.scores[id];
  for (auto id : r.ids) {
    r.weights.push_back(float(r.scores[id] / denominator * model.scaling));
    r.tags.push_back(float(id));
  }
  return r;
}
inline Fragment router(const Case &model, bool native_routing = false) {
  Fragment f{.profile = model.profile, .region = 0x0a0601};
  auto input = f.input(dense({1, model.hidden}));
  auto route = routing(f, model, input, native_routing);
  f.outputs = {gathered_ids(f, model, route.ids), route.weights};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 3; ++replay) {
    auto x = activation(model, replay);
    auto reference = route_reference(model, x);
    f.replays.push_back({{bytes(x)}, {std::move(reference.tags),
                                    std::move(reference.weights)}});
  }
  return f;
}

// The original CUDA program has two launch branches at T=512. Ties have
// intentionally unspecified membership, while non-ties retain exact IDs.
inline Fragment native_router_boundary(std::int64_t experts,
                                        std::int64_t tokens, int mode) {
  const auto &model = experts == 384 ? cases::kCases[3] : cases::kCases[4];
  Fragment f{.profile = model.profile, .region = 0x0a0604};
  auto logits = f.input(dense({tokens, experts}));
  std::vector<float> bias(experts, 0.0F);
  auto correction = f.constant(dense({experts}), bytes(bias));
  auto outputs = f.append(c::model_ops::sqrt_softplus_routing(model.scaling),
                          {logits, correction}, 2);
  f.outputs = {outputs[0], gathered_ids(f, model, outputs[1])};
  f.exact_outputs = {1};
  if (mode == 1 || mode == 2) f.unordered_ids = Fragment::UnorderedIds{1, 6, experts};
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<float> x(tokens * experts), weights(tokens * 6), tags(tokens * 6);
    for (std::int64_t row = 0; row < tokens; ++row) {
      std::vector<std::int32_t> selected(experts);
      std::iota(selected.begin(), selected.end(), 0);
      std::vector<double> scores(experts);
      for (std::int64_t e = 0; e < experts; ++e) {
        const auto index = replay == 0 ? e : experts - 1 - e;
        const auto value = mode == 1 ? 0.0F : mode == 2 ? -1000.0F
                         : mode == 3 ? std::numeric_limits<float>::quiet_NaN()
                         : mode == 4 ? -80.0F + float(index) / 512.0F
                         : mode == 5 ? 1.0F + float(index) / 65536.0F
                         : float((index * 7 + row * 11) % 81 - 40) + float(index) / 512.0F;
        x[row * experts + e] = value;
        const double wide = value;
        const float sp = mode == 3 ? 0.0F
            : float(std::max(wide, 0.0) + std::log1p(std::exp(-std::abs(wide))));
        scores[e] = std::sqrt(double(sp));
      }
      std::stable_sort(selected.begin(), selected.end(), [&](auto a, auto b) {
        return scores[a] > scores[b];
      });
      double denominator = 0;
      for (std::size_t slot = 0; slot < 6; ++slot) denominator += scores[selected[slot]];
      for (std::size_t slot = 0; slot < 6; ++slot) {
        weights[row * 6 + slot] = denominator > 0
            ? float(scores[selected[slot]] / denominator * model.scaling) : 0.0F;
        tags[row * 6 + slot] = mode == 3 ? 0.0F : float(selected[slot]);
      }
    }
    f.replays.push_back({{bytes(x)}, {std::move(weights), std::move(tags)}});
  }
  return f;
}

inline ValueId gated(Fragment &f, const Case &model, ValueId gate, ValueId up) {
  if (model.latent != model.hidden) {
    const auto desc = f.description(gate);
    auto a = unary(f, c::UnaryKind::kTanh,
                    binary(f, c::BinaryKind::kMultiply, gate, filled(f, desc, 0.25F)));
    a = binary(f, c::BinaryKind::kMultiply, a, filled(f, desc, 4.0F));
    a = binary(f, c::BinaryKind::kMultiply, a,
                unary(f, c::UnaryKind::kSigmoid, gate));
    auto b = unary(f, c::UnaryKind::kTanh,
                    binary(f, c::BinaryKind::kMultiply, up, filled(f, desc, 0.04F)));
    b = binary(f, c::BinaryKind::kMultiply, b, filled(f, desc, 25.0F));
    return binary(f, c::BinaryKind::kMultiply, a, b);
  }
  if (model.sqrt_softplus || model.profile->model_name == "GLM-5.3-Flash") {
    gate = bound(f, gate, 10.0F, true);
    up = bound(f, bound(f, up, -10.0F, false), 10.0F, true);
  }
  return f.one(c::Operation(c::GatedActivation{
                   .inputs = c::GatedInputKind::kSeparate}), {gate, up});
}
inline std::vector<float> gate_reference(const Case &model,
                                        std::span<const float> gate,
                                        std::span<const float> up) {
  std::vector<float> out(gate.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    double a = gate[i], b = up[i];
    if (model.latent != model.hidden)
      out[i] = float(4.0 * std::tanh(a / 4.0) * core::sigmoid(a) *
                     25.0 * std::tanh(b / 25.0));
    else {
      if (model.sqrt_softplus || model.profile->model_name == "GLM-5.3-Flash") {
        a = std::min(a, 10.0);
        b = std::clamp(b, -10.0, 10.0);
      }
      out[i] = float(a * core::sigmoid(a) * b);
    }
  }
  return out;
}
inline Fragment activation_core(const Case &model) {
  Fragment f{.profile = model.profile, .region = 0x020602};
  auto desc = dense({1, model.intermediate});
  auto a = f.input(desc), b = f.input(desc);
  f.outputs = {gated(f, model, a, b)};
  for (int seed : {3, 19}) {
    std::vector<float> x(model.intermediate), y(model.intermediate);
    for (std::int64_t i = 0; i < model.intermediate; ++i) {
      x[i] = float((i * 7 + seed) % 257 - 128) / 4.0F;
      y[i] = float((i * 13 + seed) % 257 - 128) / 2.0F;
    }
    f.replays.push_back({{bytes(x), bytes(y)}, {gate_reference(model, x, y)}});
  }
  return f;
}

inline c::GgufQuantizedType quantized_type(Storage storage) {
  switch (storage) {
  case Storage::kQ4K: return c::GgufQuantizedType::kQ4K;
  case Storage::kQ5K: return c::GgufQuantizedType::kQ5K;
  case Storage::kQ8_0: return c::GgufQuantizedType::kQ8_0;
  case Storage::kMxfp4: return c::GgufQuantizedType::kMxfp4;
  case Storage::kBFloat16: break;
  }
  throw std::runtime_error("BF16 has no quantized block identity");
}
inline sem::Weight weight(std::int64_t experts, std::int64_t n, std::int64_t k,
                           Storage storage, int seed) {
  if (storage != Storage::kBFloat16) {
    auto result = sem::weight(experts * n, k, quantized_type(storage), false, seed);
    result.description = c::gguf_grouped_weight_description(
        quantized_type(storage), c::DataType::kFloat32, experts, n, k).value();
    return result;
  }
  auto desc = dense({experts, n, k}, c::DataType::kBFloat16);
  sem::Weight result{desc, std::vector<std::byte>(desc.required_bytes().value()),
                     std::vector<std::vector<std::pair<std::size_t, float>>>(experts * n)};
  for (std::int64_t row = 0; row < experts * n; ++row) {
    const auto column = (17 * row + seed) % k;
    const auto value = sem::q::encode(0.5 + double((row + seed) % 13) / 32.0);
    std::memcpy(result.storage.data() + (row * k + column) * sizeof(value),
                 &value, sizeof(value));
    result.decoded[row].emplace_back(std::size_t(column), float(sem::q::decode(value)));
  }
  return result;
}
inline std::vector<float> project(std::span<const float> input,
                                  const sem::Weight &weight,
                                  std::span<const std::int32_t> ids,
                                  std::int64_t n, std::int64_t k) {
  std::vector<float> out(ids.size() * n);
  for (std::size_t slot = 0; slot < ids.size(); ++slot)
    for (std::int64_t row = 0; row < n; ++row) {
      double value = 0;
      for (auto [column, coefficient] : weight.decoded[ids[slot] * n + row])
        value += double(input[slot * k + column]) * coefficient;
      out[slot * n + row] = float(value);
    }
  return out;
}

inline Fragment official_fp4_projection(const Case &model, bool down) {
  if (model.gguf || (model.profile != cases::kCases[0].profile &&
                    model.profile != cases::kCases[3].profile &&
                    model.profile != cases::kCases[4].profile))
    throw std::runtime_error("profile has no audited official FP4 expert packing");
  Fragment f{.profile = model.profile, .region = 0x0b0602, .quantized = true};
  const auto n = down ? model.latent : model.intermediate;
  const auto k = down ? model.intermediate : model.latent;
  const auto resident = model.top_k + 1;
  auto w = d5_official_fp4_storage::weight(resident, n, k, down ? 83 : 71);
  auto stored = f.constant(w.description, std::move(w.storage));
  auto input = f.input(dense({1, model.top_k, k}));
  auto ids = f.input(dense({1, model.top_k}, c::DataType::kInt32));
  f.outputs = {f.one(c::model_ops::grouped_matmul(), {stored, input, ids})};
  for (int replay = 0; replay < 2; ++replay) {
    auto x = d5_gguf_test::activation(model.top_k * k, 17 + replay * 31);
    std::vector<std::int32_t> selected(model.top_k);
    for (std::int64_t slot = 0; slot < model.top_k; ++slot)
      selected[slot] = std::int32_t(replay ? resident - 1 - slot : slot);
    f.replays.push_back({{bytes(x), bytes(selected)}, {project(x, w, selected, n, k)}});
  }
  return f;
}

inline Fragment moe_chain(const Case &model, bool native_routing,
                            bool official_fp4) {
  if (!model.gguf && !(official_fp4 && model.profile == cases::kCases[0].profile))
    throw std::runtime_error("official expert packing is a separate qualification");
  Fragment f{.profile = model.profile, .region = 0x0a0b0601, .quantized = true};
  auto input = f.input(dense({1, model.hidden}));
  auto route = routing(f, model, input, native_routing);
  const auto resident = model.top_k + 1;
  const std::vector<std::int32_t> zero{0};
  auto zero_id = f.constant(dense({1, 1}, c::DataType::kInt32), bytes(zero));
  const auto grouped = [&](const sem::Weight &w, ValueId x, ValueId ids) {
    auto stored = f.constant(w.description, w.storage);
    return f.one(c::model_ops::grouped_matmul(), {stored, x, ids});
  };
  std::optional<sem::Weight> latent_down, latent_up;
  auto routed_input = reshape(f, input, {1, 1, model.hidden});
  if (model.latent != model.hidden) {
    latent_down = weight(1, model.latent, model.hidden, model.latent_storage, 151);
    latent_up = weight(1, model.hidden, model.latent, model.latent_storage, 157);
    routed_input = grouped(*latent_down, routed_input, zero_id);
  }
  auto flat_input = reshape(f, routed_input, {1, model.latent});
  auto replicate = f.constant(dense({model.top_k}, c::DataType::kInt32),
                               bytes(std::vector<std::int32_t>(model.top_k, 0)));
  auto slots = reshape(f, f.one(c::Operation(c::Gather{}), {flat_input, replicate}),
                        {1, model.top_k, model.latent});
  const auto expert_weight = [&](std::int64_t n, std::int64_t k, Storage type, int seed) {
    return official_fp4 ? d5_official_fp4_storage::weight(resident, n, k, seed)
                         : weight(resident, n, k, type, seed);
  };
  auto wg = expert_weight(model.intermediate, model.latent, model.gate, 71);
  auto wu = expert_weight(model.intermediate, model.latent, model.up, 79);
  auto wd = expert_weight(model.latent, model.intermediate, model.down, 83);
  auto gate = grouped(wg, slots, route.ids), up = grouped(wu, slots, route.ids);
  auto activated = gated(f, model, gate, up);
  auto routed = grouped(wd, activated, route.ids);
  auto weights = reshape(f, route.weights, {1, model.top_k, 1});
  auto combined = sum(f, binary(f, c::BinaryKind::kMultiply, routed, weights), 1);
  std::vector<float> norm_values;
  if (latent_up) {
    norm_values = sem::oracle::scale(model.latent);
    auto scale = f.constant(dense({model.latent}), bytes(norm_values));
    combined = f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-5F}), {combined, scale});
    combined = grouped(*latent_up, combined, zero_id);
  }
  auto sg = weight(1, model.shared_intermediate, model.hidden, model.shared_gate, 113);
  auto su = weight(1, model.shared_intermediate, model.hidden, model.shared_up, 127);
  auto sd = weight(1, model.hidden, model.shared_intermediate, model.shared_down, 139);
  auto shared_input = reshape(f, input, {1, 1, model.hidden});
  auto shared_gate = grouped(sg, shared_input, zero_id);
  auto shared_up = grouped(su, shared_input, zero_id);
  auto shared_activated = gated(f, model, shared_gate, shared_up);
  auto shared = grouped(sd, shared_activated, zero_id);
  auto output = binary(f, c::BinaryKind::kAdd, combined, shared);
  f.outputs = {gathered_ids(f, model, route.ids), route.weights, routed, combined,
                shared, output, gate, up, activated,
                shared_gate, shared_up, shared_activated};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 3; ++replay) {
    auto x = activation(model, replay);
    auto reference = route_reference(model, x);
    auto latent = latent_down ? project(x, *latent_down, zero, model.latent, model.hidden) : x;
    std::vector<float> replicated;
    for (std::int64_t slot = 0; slot < model.top_k; ++slot)
      replicated.insert(replicated.end(), latent.begin(), latent.end());
    auto g = project(replicated, wg, reference.ids, model.intermediate, model.latent);
    auto u = project(replicated, wu, reference.ids, model.intermediate, model.latent);
    auto act = gate_reference(model, g, u);
    auto routed_value = project(act, wd, reference.ids,
                                model.latent, model.intermediate);
    std::vector<float> combined_value(model.latent);
    for (std::int64_t d = 0; d < model.latent; ++d) {
      double value = 0;
      for (std::int64_t slot = 0; slot < model.top_k; ++slot)
        value += routed_value[slot * model.latent + d] * double(reference.weights[slot]);
      combined_value[d] = float(value);
    }
    if (latent_up) {
      auto normalized = sem::oracle::rms_norm_rows(combined_value, norm_values,
                                                    model.latent, 1.0e-5F);
      combined_value = project(normalized, *latent_up, zero, model.hidden, model.latent);
    }
    auto sgate = project(x, sg, zero, model.shared_intermediate, model.hidden);
    auto sup = project(x, su, zero, model.shared_intermediate, model.hidden);
    auto sact = gate_reference(model, sgate, sup);
    auto shared_value = project(sact, sd, zero,
                                model.hidden, model.shared_intermediate);
    std::vector<float> full(model.hidden);
    for (std::int64_t d = 0; d < model.hidden; ++d)
      full[d] = combined_value[d] + shared_value[d];
    f.replays.push_back({{bytes(x)}, {std::move(reference.tags),
        std::move(reference.weights), std::move(routed_value),
        std::move(combined_value), std::move(shared_value), std::move(full),
        std::move(g), std::move(u), std::move(act),
        std::move(sgate), std::move(sup), std::move(sact)}});
  }
  return f;
}

inline Fragment gguf_moe(const Case &model, bool native_routing = false) {
  return moe_chain(model, native_routing, false);
}
inline Fragment official_kimi_moe() {
  return moe_chain(cases::kCases[0], false, true);
}
} // namespace d5_moe_execution_fragments
