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

struct HeadProjection final {
  sem::Weight weight;
  bool bfloat;
  std::int64_t n, k;
  ValueId output;
  std::vector<float> reference(std::span<const float> input) const {
    std::vector<float> x(input.begin(), input.end());
    if (bfloat) for (auto &v : x) v = linear::bf16(v);
    std::vector<std::int32_t> ids(kHeads);
    std::iota(ids.begin(), ids.end(), 0);
    return moe::project(x, weight, ids, n, k);
  }
};

inline HeadProjection head_projection(Fragment &f, ValueId input, bool key,
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
  std::vector<std::int32_t> ids(kHeads);
  std::iota(ids.begin(), ids.end(), 0);
  auto head_ids = f.constant(dense({1, kHeads}, c::DataType::kInt32), bytes(ids));
  auto stored = f.constant(weight.description, std::move(weight.storage));
  auto output = f.one(c::model_ops::grouped_matmul(), {stored, input, head_ids});
  return {std::move(weight), bfloat, n, k, output};
}

inline Fragment mla(const c::ModelArtifactProfileDescription *profile) {
  Fragment f{.profile = profile, .region = 0x070701,
             .quantized = storage(profile) == Storage::kQ8_0};
  auto x = f.input(dense({1, kHidden}));
  auto past = f.input(dense({3, 576}));
  auto input_scale = sem::oracle::scale(kHidden);
  auto normalized = norm(f, x, input_scale, official(profile));
  auto qa = linear::projection(f, normalized, 1536, kHidden, storage(profile), 173);
  auto qscale = sem::oracle::scale(1536);
  auto nq = norm(f, qa.output, qscale, official(profile));
  auto qb = linear::projection(f, nq, 96 * 192, 1536, storage(profile), 179);
  auto kva = linear::projection(f, normalized, 576, kHidden, storage(profile), 181);
  auto latent = reshape(f, core::rows(f, reshape(f, kva.output, {576, 1}), 0, 512), {1, 512});
  auto pe = reshape(f, core::rows(f, reshape(f, kva.output, {576, 1}), 512, 64), {1, 64});
  auto kvscale = sem::oracle::scale(512);
  latent = norm(f, latent, kvscale, official(profile));
  auto current = f.one(c::Operation(c::Concat{.axis = 1}), {latent, pe});
  auto cache = f.one(c::Operation(c::Concat{.axis = 0}), {past, current});
  auto next_cache = core::rows(f, cache, 1, 3);
  auto query_nope = slice_heads(f, qb.output, kHeads, 192, 0, 128);
  auto query_pe = slice_heads(f, qb.output, kHeads, 192, 128, 64);
  auto kb = head_projection(f, query_nope, true, profile);
  auto query = f.one(c::Operation(c::Concat{.axis = 2}), {kb.output, query_pe});
  auto cached_values = slice_heads(f, cache, 4, 576, 0, 512);
  auto attention = f.one(c::Operation(c::ScaledDotProductAttention{
                          .scale = 1.0F / std::sqrt(192.0F), .causal = true,
                          .accumulation_type = c::DataType::kFloat32}),
      {reshape(f, query, {1, 1, kHeads, 576}), reshape(f, cache, {1, 4, 1, 576}),
       reshape(f, cached_values, {1, 4, 1, 512})});
  auto vb = head_projection(f, reshape(f, attention, {1, kHeads, 512}), false, profile);
  auto gate = linear::projection(f, normalized, 12288, kHidden, storage(profile), 223);
  auto gated = binary(f, c::BinaryKind::kMultiply,
                       reshape(f, vb.output, {1, 12288}),
                       unary(f, c::UnaryKind::kSigmoid, gate.output));
  auto out = linear::projection(f, gated, kHidden, 12288, storage(profile), 227);
  f.outputs = {next_cache, qb.output, kb.output, attention, vb.output, gated, out.output, normalized, gate.output};
  if (storage(profile) == Storage::kBFloat16)
    f.bfloat_chain_outputs = {0, 1, 2, 3, 4, 5, 6, 8};
  f.state_ports = {{1, 0}};
  auto cache_values = d5_gguf_test::activation(3 * 576, 23);
  for (int step = 0; step < 4; ++step) {
    auto values = d5_gguf_test::activation(kHidden, 3 + 14 * step);
    sem::Replay replay{{bytes(values), bytes(cache_values)}, {}};
    auto nx = sem::oracle::rms_norm_rows(values, input_scale, kHidden, kEpsilon);
    auto aq = qa.reference(nx);
    aq = sem::oracle::rms_norm_rows(aq, qscale, 1536, kEpsilon);
    auto bq = qb.reference(aq), kv = kva.reference(nx);
    auto l = sem::oracle::rms_norm_rows(std::span(kv).first(512), kvscale, 512, kEpsilon);
    l.insert(l.end(), kv.begin() + 512, kv.end());
    auto full_cache = cache_values;
    full_cache.insert(full_cache.end(), l.begin(), l.end());
    cache_values.assign(full_cache.begin() + 576, full_cache.end());
    auto no_pe = slice(bq, kHeads, 192, 0, 128);
    auto absorbed = kb.reference(no_pe);
    std::vector<float> attended(kHeads * 512);
    const float scale = 1.0F / std::sqrt(192.0F);
    for (std::int64_t h = 0; h < kHeads; ++h) {
      std::array<double, 4> scores{};
      for (int t = 0; t < 4; ++t) {
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
        for (int t = 0; t < 4; ++t) sum += scores[t] / total * full_cache[t * 576 + d];
        attended[h * 512 + d] = float(sum);
      }
    }
    auto projected = vb.reference(attended), gate_values = gate.reference(nx);
    auto gv = projected;
    for (std::size_t i = 0; i < gv.size(); ++i) gv[i] *= float(core::sigmoid(gate_values[i]));
    auto result = out.reference(gv);
    replay.outputs = {cache_values, bq, absorbed, attended, projected, gv, result, nx, gate_values};
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
