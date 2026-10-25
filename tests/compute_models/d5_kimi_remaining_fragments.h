#pragma once

#include "d5_kda_full_fragments.h"
#include "d5_moe_execution_fragments.h"

namespace d5_kimi_remaining_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace linear = d5_kda_full_fragments;
namespace moe = d5_moe_execution_fragments;
using sem::Fragment;
using sem::ValueId;
using sem::bytes;
using sem::dense;
using core::reshape;
using core::binary;
using core::unary;
using core::filled;
using linear::Storage;
inline constexpr std::int64_t kHidden = 7168, kHeads = 96, kLatent = 512;
inline constexpr float kEpsilon = 1.0e-5F;

inline bool official(const c::ModelArtifactProfileDescription *p) {
  return p == core::kKdaProfiles[0];
}
inline Storage storage(const c::ModelArtifactProfileDescription *p) {
  return p == core::kKdaProfiles[1] ? Storage::kQ8_0 : Storage::kBFloat16;
}
inline ValueId selected(Fragment &f, ValueId input,
                         std::vector<std::int32_t> indices) {
  const auto count = std::int64_t(indices.size());
  auto ids = f.constant(dense({count}, c::DataType::kInt32), bytes(indices));
  auto flat = reshape(f, input, {std::int64_t(*f.description(input).element_count()), 1});
  return f.one(c::Operation(c::Gather{}), {flat, ids});
}
inline ValueId slice_heads(Fragment &f, ValueId input, std::int64_t heads,
                           std::int64_t stride, std::int64_t start,
                           std::int64_t count) {
  std::vector<std::int32_t> ids;
  for (std::int64_t h = 0; h < heads; ++h)
    for (std::int64_t d = 0; d < count; ++d)
      ids.push_back(std::int32_t(h * stride + start + d));
  return reshape(f, selected(f, input, std::move(ids)), {1, heads, count});
}
inline std::vector<float> slice(std::span<const float> x, std::int64_t heads,
                                std::int64_t stride, std::int64_t start,
                                std::int64_t count) {
  std::vector<float> result;
  for (std::int64_t h = 0; h < heads; ++h)
    result.insert(result.end(), x.begin() + h * stride + start,
                   x.begin() + h * stride + start + count);
  return result;
}

inline ValueId norm(Fragment &f, ValueId input, std::vector<float> &scale,
                     bool bfloat) {
  auto weight = linear::parameter(f, dense({std::int64_t(scale.size())}), scale, bfloat);
  return f.one(c::Operation(c::RmsNorm{.epsilon = kEpsilon}), {input, weight});
}

// Conversion itself is exact even when full low-precision chains diverge at
// a rounding midpoint. Exercise both tie parities, signs and adjacent floats.
inline Fragment bfloat_rounding_boundaries() {
  Fragment f{.profile = core::kKdaProfiles[0], .region = 0x0e0701};
  std::vector<float> input;
  for (float midpoint : {0.748046875F, 0.21044921875F, 1.00390625F, 1.01171875F})
    for (float sign : {-1.0F, 1.0F}) {
      const float value = sign * midpoint;
      input.push_back(std::nextafter(value, -std::numeric_limits<float>::infinity()));
      input.push_back(value);
      input.push_back(std::nextafter(value, std::numeric_limits<float>::infinity()));
    }
  auto x = f.input(dense({1, std::int64_t(input.size())}));
  auto rounded = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {x});
  f.outputs = {f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {rounded})};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    auto expected = input;
    for (auto &value : expected) value = linear::bf16(value);
    f.replays.push_back({{bytes(input)}, {expected}});
    std::reverse(input.begin(), input.end());
  }
  return f;
}

// Dense first layer: exact 33792 intermediate, SiTU and three native weights.
inline Fragment dense_ffn(const c::ModelArtifactProfileDescription *profile) {
  Fragment f{.profile = profile, .region = 0x02030701,
             .quantized = storage(profile) == Storage::kQ8_0};
  auto x = f.input(dense({1, kHidden}));
  auto scales = sem::oracle::scale(kHidden);
  for (std::size_t i = 0; i < scales.size(); ++i) scales[i] += float(i % 3) * 0.0037F;
  auto normalized = norm(f, x, scales, official(profile));
  auto g = linear::projection(f, normalized, 33792, kHidden, storage(profile), 173);
  auto u = linear::projection(f, normalized, 33792, kHidden, storage(profile), 181);
  auto activated = moe::gated(f, moe::cases::kCases[0], g.output, u.output);
  auto down = linear::projection(f, activated, kHidden, 33792, storage(profile), 191);
  auto residual = binary(f, c::BinaryKind::kAdd, x, down.output);
  auto final_scales = sem::oracle::scale(kHidden);
  std::reverse(final_scales.begin(), final_scales.end());
  auto final_norm = norm(f, residual, final_scales, official(profile));
  f.outputs = {normalized, g.output, u.output, activated, down.output, residual, final_norm};
  if (storage(profile) == Storage::kBFloat16)
    f.bfloat_chain_outputs = {1, 2, 3, 4, 5, 6};
  for (int seed : {3, 17, 29}) {
    auto values = d5_gguf_test::activation(kHidden, seed);
    auto nx = sem::oracle::rms_norm_rows(values, scales, kHidden, kEpsilon);
    auto gv = g.reference(nx), uv = u.reference(nx);
    auto av = moe::gate_reference(moe::cases::kCases[0], gv, uv);
    auto dv = down.reference(av), rv = dv;
    for (std::size_t i = 0; i < rv.size(); ++i) rv[i] += values[i];
    auto fn = sem::oracle::rms_norm_rows(rv, final_scales, kHidden, kEpsilon);
    f.replays.push_back({{bytes(values)}, {nx, gv, uv, av, dv, rv, fn}});
  }
  return f;
}

struct HeadProjectionWeight final {
  sem::Weight weight;
  bool bfloat;
  std::int64_t n, k;
  ValueId value;
  std::vector<float> reference(std::span<const float> input, bool round_input_to_bfloat) const {
    std::vector<float> x(input.begin(), input.end());
    if (bfloat && round_input_to_bfloat) for (auto &v : x) v = linear::bf16(v);
    std::vector<std::int32_t> ids(kHeads);
    std::iota(ids.begin(), ids.end(), 0);
    return moe::project(x, weight, ids, n, k);
  }
};
inline HeadProjectionWeight prepare_head_projection(Fragment &f, bool key,
    const c::ModelArtifactProfileDescription *profile) {
  const auto n = key ? 512 : 128;
  const auto k = key ? 128 : 512;
  const bool bfloat = storage(profile) == Storage::kBFloat16;
  auto weight = moe::weight(kHeads, n, k, bfloat ? moe::Storage::kBFloat16
                                                : moe::Storage::kQ8_0, key ? 199 : 211);
  if (official(profile)) {
    // Original kv_b [H, 128+128, 512] -> K [H,512,128], V [H,128,512].
    // Only move original BF16 bits; the oracle retains the source row/column.
    std::fill(weight.storage.begin(), weight.storage.end(), std::byte{});
    for (auto &row : weight.decoded) row.clear();
    for (std::int64_t h = 0; h < kHeads; ++h)
      for (std::int64_t d = 0; d < 128; ++d) {
        const auto source_row = h * 256 + (key ? d : 128 + d);
        const auto column = (source_row * 17 + 199) % 512;
        const auto code = sem::q::encode(0.5 + double(source_row % 13) / 32.0);
        const auto row = h * n + (key ? column : d);
        const auto reduction = key ? d : column;
        std::memcpy(weight.storage.data() + (row * k + reduction) * sizeof(code),
                      &code, sizeof(code));
        weight.decoded[row].emplace_back(std::size_t(reduction), float(sem::q::decode(code)));
      }
  }
  auto stored = f.constant(weight.description, std::move(weight.storage));
  return {std::move(weight), bfloat, n, k, stored};
}
inline ValueId project_heads(Fragment &f, ValueId input, ValueId weight) {
  std::vector<std::int32_t> ids(kHeads);
  std::iota(ids.begin(), ids.end(), 0);
  const auto head_ids = f.constant(dense({1, kHeads}, c::DataType::kInt32), bytes(ids));
  return f.one(c::model_ops::grouped_matmul(), {weight, input, head_ids});
}
// ggml Metal mul_mv_id_bf16_f32 retains F32 activations; CPU vec-dot
// converts them to BF16. Match that source-defined boundary in the oracle,
// while leaving the identical public operation graph and accuracy gate intact.
struct OrdinaryHistory final {
  std::int64_t capacity, position;
};
inline std::pair<std::int64_t, std::int64_t> history_extents(
    std::optional<OrdinaryHistory> history) {
  if (history && (history->capacity <= 0 || history->capacity > std::numeric_limits<std::int32_t>::max() ||
      history->position < 0 || history->position >= history->capacity))
    throw std::runtime_error("ordinary Kimi MLA history is outside its cold capacity");
  return {history ? history->capacity : 3, history ? history->position + 1 : 4};
}
struct MlaParameters final {
  ValueId input_norm, q_a, q_norm, q_b, kv_a, kv_norm;
  // Prepared original KV-b K/V layouts; they share source identity but are
  // actual independent grouped operands, not a generated replacement bank.
  ValueId absorbed_key, absorbed_value, gate, output;
};
struct MlaPorts final {
  ValueId cache, query, absorbed_query, attention, projected, gated, output, normalized, gate;
  std::array<ValueId, 9> outputs() const {
    return {cache, query, absorbed_query, attention, projected, gated, output, normalized, gate};
  }
};
inline void require_head_projection(const Fragment &f, ValueId value,
                                    std::int64_t n, std::int64_t k) {
  linear::require_value(f, value, {kHeads, n, k});
  const auto &description = f.description(value);
  if ((description.data_type() != c::DataType::kFloat32 &&
       description.data_type() != c::DataType::kBFloat16) ||
      (description.data_type() == c::DataType::kBFloat16 && !description.is_plain()))
    throw std::runtime_error("supplied Kimi absorbed weight needs its original F32/GGUF or BF16 storage");
}

// Fixed Kimi NoPE absorbed-MLA equation. All Values are borrowed from this
// Fragment; original parameter sources and ReadLeases stay with the caller.
// Neither state publication nor model/profile-dependent routing lives here.
inline MlaPorts append_mla(Fragment &f, ValueId input, ValueId past,
    const MlaParameters &p, std::optional<OrdinaryHistory> history = {}) {
  const auto [past_rows, visible] = history_extents(history);
  linear::require_value(f, input, {1, kHidden});
  linear::require_value(f, past, {past_rows, 576});
  for (auto value : {input, past})
    if (f.description(value).data_type() != c::DataType::kFloat32 || !f.description(value).is_plain())
      throw std::runtime_error("supplied Kimi MLA keeps its original plain F32 input/cache boundary");
  linear::require_float_parameter(f, p.input_norm, {kHidden});
  linear::require_float_parameter(f, p.q_norm, {1536});
  linear::require_float_parameter(f, p.kv_norm, {512});
  linear::require_projection(f, p.q_a, 1536, kHidden);
  linear::require_projection(f, p.q_b, kHeads * 192, 1536);
  linear::require_projection(f, p.kv_a, 576, kHidden);
  require_head_projection(f, p.absorbed_key, 512, 128);
  require_head_projection(f, p.absorbed_value, 128, 512);
  linear::require_projection(f, p.gate, 12288, kHidden);
  linear::require_projection(f, p.output, kHidden, 12288);

  const c::Operation normalization(c::RmsNorm{.epsilon = kEpsilon});
  const auto normalized = f.one(normalization, {input, linear::float_parameter(f, p.input_norm)});
  const auto qa = linear::project_f32(f, normalized, p.q_a);
  const auto nq = f.one(normalization, {qa, linear::float_parameter(f, p.q_norm)});
  const auto qb = linear::project_f32(f, nq, p.q_b);
  const auto kva = linear::project_f32(f, normalized, p.kv_a);
  auto latent = reshape(f, core::rows(f, reshape(f, kva, {576, 1}), 0, 512), {1, 512});
  const auto pe = reshape(f, core::rows(f, reshape(f, kva, {576, 1}), 512, 64), {1, 64});
  latent = f.one(normalization, {latent, linear::float_parameter(f, p.kv_norm)});
  const auto current = f.one(c::Operation(c::Concat{.axis = 1}), {latent, pe});
  ValueId cache, next_cache;
  if (history) {
    const auto position = f.constant(dense({1}, c::DataType::kInt32),
        bytes(std::vector<std::int32_t>{std::int32_t(history->position)}));
    next_cache = f.one(c::Operation(c::Scatter{}), {past, position, current});
    cache = core::rows(f, next_cache, 0, visible);
  } else {
    cache = f.one(c::Operation(c::Concat{.axis = 0}), {past, current});
    next_cache = core::rows(f, cache, 1, 3);
  }
  const auto query_nope = slice_heads(f, qb, kHeads, 192, 0, 128);
  const auto query_pe = slice_heads(f, qb, kHeads, 192, 128, 64);
  const auto kb = project_heads(f, query_nope, p.absorbed_key);
  const auto query = f.one(c::Operation(c::Concat{.axis = 2}), {kb, query_pe});
  const auto cached_values = slice_heads(f, cache, visible, 576, 0, 512);
  const auto attention = f.one(c::Operation(c::ScaledDotProductAttention{
      .scale = 1.0F / std::sqrt(192.0F), .causal = true,
      .accumulation_type = c::DataType::kFloat32}),
      {reshape(f, query, {1, 1, kHeads, 576}), reshape(f, cache, {1, visible, 1, 576}),
       reshape(f, cached_values, {1, visible, 1, 512})});
  const auto vb = project_heads(f, reshape(f, attention, {1, kHeads, 512}), p.absorbed_value);
  const auto gate = linear::project_f32(f, normalized, p.gate);
  const auto gated = binary(f, c::BinaryKind::kMultiply, reshape(f, vb, {1, 12288}),
      unary(f, c::UnaryKind::kSigmoid, gate));
  const auto output = linear::project_f32(f, gated, p.output);
  return {next_cache, qb, kb, attention, vb, gated, output, normalized, gate};
}
using MlaReference = std::function<std::vector<std::vector<float>>(
    std::span<const float>, std::span<const float>)>;
inline Fragment mla(const c::ModelArtifactProfileDescription *profile,
                      bool round_grouped_input_to_bfloat = true,
                      std::optional<OrdinaryHistory> history = {},
                      MlaReference *reference = nullptr) {
  const auto [past_rows, visible] = history_extents(history);
  Fragment f{.profile = profile, .region = 0x070701,
             .quantized = storage(profile) == Storage::kQ8_0};
  const auto x = f.input(dense({1, kHidden}));
  const auto past = f.input(dense({past_rows, 576}));
  auto input_scale = sem::oracle::scale(kHidden);
  const auto input_norm = linear::parameter(f, dense({kHidden}), input_scale, official(profile));
  auto qa = linear::prepare_projection(f, 1536, kHidden, storage(profile), 173);
  auto qscale = sem::oracle::scale(1536);
  const auto qnorm = linear::parameter(f, dense({1536}), qscale, official(profile));
  auto qb = linear::prepare_projection(f, 96 * 192, 1536, storage(profile), 179);
  auto kva = linear::prepare_projection(f, 576, kHidden, storage(profile), 181);
  auto kvscale = sem::oracle::scale(512);
  const auto kvnorm = linear::parameter(f, dense({512}), kvscale, official(profile));
  auto kb = prepare_head_projection(f, true, profile);
  auto vb = prepare_head_projection(f, false, profile);
  auto gate = linear::prepare_projection(f, 12288, kHidden, storage(profile), 223);
  auto out = linear::prepare_projection(f, kHidden, 12288, storage(profile), 227);
  const MlaParameters parameters{input_norm, qa.value, qnorm, qb.value, kva.value,
      kvnorm, kb.value, vb.value, gate.value, out.value};
  const auto ports = append_mla(f, x, past, parameters, history);
  const auto outputs = ports.outputs();
  f.outputs.assign(outputs.begin(), outputs.end());
  if (storage(profile) == Storage::kBFloat16)
    f.bfloat_chain_outputs = {0, 1, 2, 3, 4, 5, 6, 8};
  f.state_ports = {{1, 0}};
  MlaReference evaluate = [=](std::span<const float> values,
      std::span<const float> cache_values) -> std::vector<std::vector<float>> {
    if (values.size() != std::size_t(kHidden) || cache_values.size() != std::size_t(past_rows * 576))
      throw std::runtime_error("ordinary Kimi MLA reference input extent");
    auto nx = sem::oracle::rms_norm_rows(values, input_scale, kHidden, kEpsilon);
    auto aq = qa.reference(nx);
    aq = sem::oracle::rms_norm_rows(aq, qscale, 1536, kEpsilon);
    auto bq = qb.reference(aq), kv = kva.reference(nx);
    auto l = sem::oracle::rms_norm_rows(std::span(kv).first(512), kvscale, 512, kEpsilon);
    l.insert(l.end(), kv.begin() + 512, kv.end());
    std::vector<float> full_cache(cache_values.begin(), cache_values.end()), next;
    if (history) {
      std::copy(l.begin(), l.end(), full_cache.begin() + history->position * 576);
      next = full_cache;
      full_cache.resize(std::size_t(visible * 576));
    } else {
      full_cache.insert(full_cache.end(), l.begin(), l.end());
      next.assign(full_cache.begin() + 576, full_cache.end());
    }
    auto no_pe = slice(bq, kHeads, 192, 0, 128);
    auto absorbed = kb.reference(no_pe, round_grouped_input_to_bfloat);
    std::vector<float> attended(kHeads * 512);
    const float scale = 1.0F / std::sqrt(192.0F);
    for (std::int64_t h = 0; h < kHeads; ++h) {
      std::vector<double> scores(std::size_t(visible), 0.0);
      for (std::int64_t t = 0; t < visible; ++t) {
        for (int d = 0; d < 512; ++d)
          scores[t] += double(absorbed[h * 512 + d]) * full_cache[t * 576 + d];
        for (int d = 0; d < 64; ++d)
          scores[t] += double(bq[h * 192 + 128 + d]) * full_cache[t * 576 + 512 + d];
        scores[t] *= scale;
      }
      const double largest = *std::max_element(scores.begin(), scores.end());
      double total = 0;
      for (auto &score : scores) total += score = std::exp(score - largest);
      for (int d = 0; d < 512; ++d) {
        double sum = 0;
        for (std::int64_t t = 0; t < visible; ++t) sum += scores[t] / total * full_cache[t * 576 + d];
        attended[h * 512 + d] = float(sum);
      }
    }
    auto projected = vb.reference(attended, round_grouped_input_to_bfloat), gate_values = gate.reference(nx);
    auto gv = projected;
    for (std::size_t i = 0; i < gv.size(); ++i) gv[i] *= float(core::sigmoid(gate_values[i]));
    auto result = out.reference(gv);
    return {std::move(next), bq, absorbed, attended, projected, gv, result, nx, gate_values};
  };
  if (reference) { *reference = std::move(evaluate); return f; }
  auto cache_values = d5_gguf_test::activation(std::size_t(past_rows * 576), 23);
  for (int step = 0; step < 4; ++step) {
    auto values = d5_gguf_test::activation(kHidden, 3 + 14 * step);
    sem::Replay replay{{bytes(values), bytes(cache_values)}, evaluate(values, cache_values)};
    cache_values = replay.outputs[0];
    f.replays.push_back(std::move(replay));
  }
  return f;
}

inline ValueId residual_mix(Fragment &f, ValueId bank, ValueId current,
                             ValueId mask, ValueId score) {
  auto joined = f.one(c::Operation(c::Concat{.axis = 0}), {bank, current});
  auto normalized = f.one(c::Operation(c::RmsNorm{.epsilon = kEpsilon}),
                           {joined, filled(f, dense({kHidden}), 1)});
  auto scored = binary(f, c::BinaryKind::kMultiply, normalized, score);
  auto sums = moe::sum(f, scored, -1);
  auto logits = binary(f, c::BinaryKind::kAdd, reshape(f, sums, {1, 9}), mask);
  auto probabilities = f.one(c::Operation(c::Softmax{}), {logits});
  return moe::sum(f, binary(f, c::BinaryKind::kMultiply, joined,
                            reshape(f, probabilities, {9, 1})), 0);
}
inline std::vector<float> residual_reference(std::span<const float> bank,
    std::span<const float> current, std::span<const float> score, int count) {
  std::vector<float> raw(bank.begin(), bank.begin() + count * kHidden);
  raw.insert(raw.end(), current.begin(), current.end());
  auto normalized = sem::oracle::rms_norm_rows(raw, std::vector<float>(kHidden, 1), kHidden, kEpsilon);
  std::vector<double> logits(count + 1);
  for (int row = 0; row <= count; ++row)
    for (std::int64_t d = 0; d < kHidden; ++d)
      logits[row] += double(normalized[row * kHidden + d]) * score[d];
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double total = 0;
  for (auto &value : logits) total += value = std::exp(value - maximum);
  std::vector<float> out(kHidden);
  for (std::int64_t d = 0; d < kHidden; ++d) {
    double value = 0;
    for (int row = 0; row <= count; ++row) value += logits[row] / total * raw[row * kHidden + d];
    out[d] = float(value);
  }
  return out;
}

inline Fragment attention_residual(const c::ModelArtifactProfileDescription *profile,
                                    bool boundary) {
  Fragment f{.profile = profile, .region = 0x0c0701};
  auto bank = f.input(dense({8, kHidden})), current = f.input(dense({1, kHidden}));
  auto attn = f.input(dense({1, kHidden})), ffn = f.input(dense({1, kHidden}));
  auto before_mask = f.input(dense({1, 9})), after_mask = f.input(dense({1, 9}));
  auto bank_rows = f.input(dense({8}, c::DataType::kInt32));
  std::array<std::vector<float>, 3> score_values;
  std::array<ValueId, 3> scores;
  for (int role = 0; role < 3; ++role) {
    auto n = sem::oracle::scale(kHidden);
    std::vector<float> p(kHidden);
    for (std::int64_t d = 0; d < kHidden; ++d) {
      n[d] = linear::bf16(n[d] + float(role + 1) * 0.0037F);
      p[d] = linear::bf16(float((d * 7 + role * 13) % 31 - 15) / 2048.0F);
    }
    score_values[role].resize(kHidden);
    for (std::size_t d = 0; d < p.size(); ++d) score_values[role][d] = n[d] * p[d];
    scores[role] = official(profile)
        ? binary(f, c::BinaryKind::kMultiply,
                  linear::parameter(f, dense({kHidden}), n, true),
                  linear::parameter(f, dense({kHidden}), p, true))
        : f.constant(dense({kHidden}), bytes(score_values[role]));
  }
  auto attn_input = residual_mix(f, bank, current, before_mask, scores[0]);
  auto next_bank = boundary
      ? f.one(c::Operation(c::Gather{}),
              {f.one(c::Operation(c::Concat{.axis = 0}), {bank, current}), bank_rows})
      : f.one(c::Operation(c::Copy{}), {bank});
  auto prefix = boundary ? attn : binary(f, c::BinaryKind::kAdd, current, attn);
  auto ffn_input = residual_mix(f, next_bank, prefix, after_mask, scores[1]);
  auto result = binary(f, c::BinaryKind::kAdd, prefix, ffn);
  auto final_mix = residual_mix(f, next_bank, result, after_mask, scores[2]);
  f.outputs = {attn_input, next_bank, ffn_input, result, final_mix};
  f.state_ports = {{0, 1}};
  f.exact_outputs = {1};
  auto bank_values = d5_gguf_test::activation(8 * kHidden, 11);
  for (int step = 0; step < 8; ++step) {
    const int before = boundary ? step : 8, after = boundary ? step + 1 : 8;
    std::vector<float> bm(9, -std::numeric_limits<float>::infinity()), am = bm;
    std::fill_n(bm.begin(), before, 0.0F); std::fill_n(am.begin(), after, 0.0F);
    bm[8] = am[8] = 0;
    auto cur = d5_gguf_test::activation(kHidden, 3 + step * 14);
    auto av = d5_gguf_test::activation(kHidden, 7 + step * 17);
    auto fv = d5_gguf_test::activation(kHidden, 19 + step * 23);
    std::vector<std::int32_t> rows(8);
    std::iota(rows.begin(), rows.end(), 0);
    if (boundary) rows[step] = 8;
    sem::Replay replay{{bytes(bank_values), bytes(cur), bytes(av), bytes(fv),
                         bytes(bm), bytes(am), bytes(rows)}, {}};
    auto ai = residual_reference(bank_values, cur, score_values[0], before);
    if (boundary) std::copy(cur.begin(), cur.end(), bank_values.begin() + step * kHidden);
    auto pv = av;
    if (!boundary) for (std::size_t d = 0; d < pv.size(); ++d) pv[d] += cur[d];
    auto fi = residual_reference(bank_values, pv, score_values[1], after);
    for (std::size_t d = 0; d < pv.size(); ++d) pv[d] += fv[d];
    auto final = residual_reference(bank_values, pv, score_values[2], after);
    replay.outputs = {ai, bank_values, fi, pv, final};
    f.replays.push_back(std::move(replay));
  }
  return f;
}
} // namespace d5_kimi_remaining_fragments
