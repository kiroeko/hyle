#pragma once

#include "d5_glm_remaining_fragments.h"

// Selected generated primary-GGUF/F32 layer43 wrapper, at the actual Flash
// dimensions. This owns no invocation, cache, StateStore, or temporal 4H state.
// The caller inserts its actual MLA/indexer graph between prepare and finish;
// finish returns this token's four-stream result directly for target44 input0.
// Layer43 is not the last target layer: no stream mean or model.norm is added.
// Fixed SGLang06992: communicator_mhc.py:84-108,298-302; glm5_next.py:764-847.
namespace glm_mtp_target43_component {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace glm = d5_glm_remaining_fragments;
using Value = sem::ValueId;
using Outputs = std::vector<std::vector<float>>;
using Reference = std::function<Outputs(std::span<const float>,
                                      std::span<const float>)>;
inline constexpr std::int64_t kHidden = 4096, kStreams = 4;

struct AttentionValues final {
  std::vector<float> normalized, mix, combine, collapsed, input, post;
};
using AttentionReference = std::function<AttentionValues(
    std::span<const float>, std::span<const float>)>;

// All values belong to the supplied Fragment. input is the sole actual
// pre-normalized H-width source for this layer's qa/kv_a/index key/head gate.
// Its qa->qnorm is also the index query projection's source, not a new input.
struct AttentionBoundary final {
  Value residual, normalized, mix, combine, collapsed, input, post_gate;
  AttentionReference evaluate;
};

inline std::vector<float> mix_bases() {
  std::vector<float> values(24);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = float(int(i % 7) - 3) / 8.F;
  return values;
}

inline Value mix_normalize(sem::Fragment &f, Value residual) {
  return f.one(c::Operation(c::RmsNorm{.epsilon = glm::kNormEpsilon}),
      {glm::reshape(f, residual, {1, kStreams * kHidden}),
       core::filled(f, sem::dense({kStreams * kHidden}), 1.F)});
}

inline AttentionBoundary prepare(sem::Fragment &f, Value residual) {
  if (f.description(residual) != sem::dense({1, kStreams, kHidden}))
    throw std::runtime_error("GLM target43 requires actual this-token F32 four-stream input");
  const auto normalized = mix_normalize(f, residual);
  // Distinct generated attention/FFN mixing roles from layer44 and NextN.
  auto projection = glm::linear::projection(f, normalized, 24,
      kStreams * kHidden, glm::linear::Storage::kQ8_0, 313);
  const auto mix = projection.output;
  const auto mhc = core::hyper_connection_prepare(f, mix, residual);
  auto scale = sem::oracle::scale(kHidden);
  std::rotate(scale.begin(), scale.begin() + 7, scale.end());
  const auto input = glm::norm(f, mhc.collapsed, scale);
  AttentionReference reference = [projection = std::move(projection),
      scale = std::move(scale)](std::span<const float> x,
                               std::span<const float> branch) -> AttentionValues {
    if (x.size() != std::size_t(kStreams * kHidden) ||
        branch.size() != std::size_t(kHidden))
      throw std::runtime_error("GLM target43 attention oracle extent");
    const std::vector<float> ones(kStreams * kHidden, 1.F);
    const std::vector<float> scales{.125F, .25F, .5F};
    auto normalized_values = sem::oracle::rms_norm_rows(x, ones,
        kStreams * kHidden, glm::kNormEpsilon);
    auto mix_values = projection.reference(normalized_values);
    auto combined = core::hyper_reference(mix_values, x, branch,
        kHidden, scales, mix_bases());
    auto input_values = sem::oracle::rms_norm_rows(combined[1], scale,
        kHidden, glm::kNormEpsilon);
    return {std::move(normalized_values), std::move(mix_values), std::move(combined[0]),
        std::move(combined[1]), std::move(input_values), std::move(combined[2])};
  };
  return {residual, normalized, mix, mhc.combine,
      mhc.collapsed, input, mhc.post, std::move(reference)};
}

// Ordered diagnostic values and independent oracle outputs share these ports.
// mhc is the only forward boundary required by layer44. The other ports allow
// checking actual consumers without reuploading their observed results.
struct Ports final {
  static constexpr std::size_t mhc = 0, attn_normalized = 1, attn_mix = 2;
  static constexpr std::size_t attn_combine = 3, attn_collapsed = 4, input = 5;
  static constexpr std::size_t attention_branch = 6, attn_post = 7;
  static constexpr std::size_t ffn_normalized = 8, ffn_mix = 9;
  static constexpr std::size_t ffn_combine = 10, ffn_collapsed = 11;
  static constexpr std::size_t ffn_input = 12, ffn_branch = 13, count = 14;
};
inline constexpr std::array<std::string_view, Ports::count> kPortNames{
    "mhc", "attn-normalized", "attn-mix", "attn-combine", "attn-collapsed",
    "attn-input", "attention-branch", "attn-post", "ffn-normalized",
    "ffn-mix", "ffn-combine", "ffn-collapsed", "ffn-input", "ffn-branch"};
struct Tail final {
  Value mhc;
  std::array<Value, Ports::count> values;
  Reference evaluate;
};

inline Tail finish(sem::Fragment &f, AttentionBoundary boundary,
                    Value attention_branch) {
  if (f.description(attention_branch) != sem::dense({1, kHidden}))
    throw std::runtime_error("GLM target43 requires actual MLA output projection F32[1,4096]");
  const auto *profile = &core::profiles::glm53_flash_primary_gguf;
  const auto found = std::ranges::find_if(glm::moe::cases::kCases,
      [&](const auto &candidate) { return candidate.profile == profile; });
  if (found == glm::moe::cases::kCases.end())
    throw std::runtime_error("GLM target43 selected sparse FFN profile missing");
  const auto attn_post = f.one(c::model_ops::hyper_connection_post(),
      {attention_branch, boundary.residual, boundary.post_gate, boundary.combine});
  const auto normalized = mix_normalize(f, attn_post);
  auto projection = glm::linear::projection(f, normalized, 24,
      kStreams * kHidden, glm::linear::Storage::kQ8_0, 337);
  const auto mix = projection.output;
  const auto mhc = core::hyper_connection_prepare(f, mix, attn_post);
  auto scale = sem::oracle::scale(kHidden);
  std::rotate(scale.begin(), scale.begin() + 19, scale.end());
  const auto input = glm::norm(f, mhc.collapsed, scale);
  glm::moe::MoeReference moe_reference;
  // Reuse the existing selected primary GGUF MoE equations/resident expert
  // domain at their original widths, without a miniature replacement FFN.
  const auto ff = glm::attach(f, glm::moe::moe_chain(*found, false, false,
      &moe_reference, .5F), {input});
  const auto branch = glm::reshape(f, ff[5], {1, kHidden});
  const auto result = f.one(c::model_ops::hyper_connection_post(),
      {branch, attn_post, mhc.post, mhc.combine});
  const auto branch_output = f.one(c::Operation(c::Copy{}), {branch});
  Reference reference = [attention = std::move(boundary.evaluate),
      projection = std::move(projection), scale = std::move(scale),
      moe_reference = std::move(moe_reference)](std::span<const float> x,
          std::span<const float> attention_values) -> Outputs {
    auto attn = attention(x, attention_values);
    const std::vector<float> ones(kStreams * kHidden, 1.F), zero(kHidden, 0.F);
    const std::vector<float> scales{.125F, .25F, .5F};
    auto normalized_values = sem::oracle::rms_norm_rows(attn.post, ones,
        kStreams * kHidden, glm::kNormEpsilon);
    auto mix_values = projection.reference(normalized_values);
    auto combined = core::hyper_reference(mix_values, attn.post, zero,
        kHidden, scales, mix_bases());
    auto input_values = sem::oracle::rms_norm_rows(combined[1], scale,
        kHidden, glm::kNormEpsilon);
    auto branch_values = moe_reference(input_values)[5];
    auto expanded = core::hyper_reference(mix_values, attn.post, branch_values,
        kHidden, scales, mix_bases())[2];
    return {std::move(expanded), std::move(attn.normalized), std::move(attn.mix),
        std::move(attn.combine), std::move(attn.collapsed), std::move(attn.input),
        std::vector<float>(attention_values.begin(), attention_values.end()),
        std::move(attn.post), std::move(normalized_values), std::move(mix_values),
        std::move(combined[0]), std::move(combined[1]), std::move(input_values),
        std::move(branch_values)};
  };
  return {result, {result, boundary.normalized, boundary.mix, boundary.combine,
      boundary.collapsed, boundary.input, attention_branch, attn_post,
      normalized, mix, mhc.combine, mhc.collapsed, input,
      branch_output}, std::move(reference)};
}
} // namespace glm_mtp_target43_component
