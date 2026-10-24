#pragma once

#include "d5_glm_remaining_fragments.h"

// Test-only selected primary-GGUF/F32 last target block (layer 44 of 45).
// Input 0 is this token's preceding-layer four-stream residual. It is never
// replaced by the previous token's mHC result. Only inputs 1/2 are temporal
// feedback: raw projected QKV history and the FP32 KDA matrix.
//
// Fixed source: SGLang 06992c92ad4494fe9f9636e75be3e35040512447,
// communicator_mhc.py:84-108,298-302; glm5_next.py:491-557,657-751,1071;
// mhc.py:1766-1821. The tap is after both mHC posts, four-stream mean AND
// target model.norm. NextN will apply its own hnorm to this actual tap and
// combine it with the next token's embedding. This is not the full target.
namespace glm_mtp_target_component {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace core = d5_moe_semantic_fragments;
namespace kda = d5_kda_full_fragments;
namespace glm = d5_glm_remaining_fragments;
using Value = sem::ValueId;
using Outputs = std::vector<std::vector<float>>;
using Reference = std::function<Outputs(std::span<const float>,
                                      std::span<const float>,
                                      std::span<const float>)>;
inline constexpr std::int64_t kHidden = 4096, kStreams = 4;
inline constexpr std::int64_t kHeads = 64, kDepth = 128;
inline constexpr std::int64_t kChannels = 3 * kHeads * kDepth;

// All fields index fragment.outputs and Reference's result in the same order.
struct Ports final {
  std::size_t history = 0, state = 1, mhc = 2, tap = 3;
  std::size_t attn_normalized = 4, attn_mix = 5, attn_combine = 6;
  std::size_t attn_collapsed = 7, attn_input = 8;
  std::size_t kda_convolution = 9, kda_output = 10, kda_gated = 11;
  std::size_t kda_branch = 12, attn_post = 13;
  std::size_t ffn_normalized = 14, ffn_mix = 15, ffn_combine = 16;
  std::size_t ffn_collapsed = 17, ffn_input = 18, ffn_branch = 19;
  std::size_t contract = 20;
  static constexpr std::size_t count = 21;
};
inline constexpr std::array<std::string_view, Ports::count> kPortNames{
    "history", "state", "mhc", "tap", "attn-normalized", "attn-mix",
    "attn-combine", "attn-collapsed", "attn-input", "kda-convolution",
    "kda-output", "kda-gated", "kda-branch", "attn-post", "ffn-normalized",
    "ffn-mix", "ffn-combine", "ffn-collapsed", "ffn-input", "ffn-branch",
    "contract"};
struct Recipe final {
  sem::Fragment fragment;
  Ports ports;
  Reference evaluate;
};

inline Recipe build() {
  const auto *profile = &core::profiles::glm53_flash_primary_gguf;
  const auto found = std::ranges::find_if(glm::moe::cases::kCases,
      [&](const auto &candidate) { return candidate.profile == profile; });
  if (found == glm::moe::cases::kCases.end())
    throw std::runtime_error("GLM target sparse FFN profile missing");
  Recipe r;
  auto &f = r.fragment;
  f.profile = profile;
  f.region = 0x474c4d5441524745ULL;
  f.quantized = true;
  const auto residual = f.input(sem::dense({1, kStreams, kHidden}));
  const auto history = f.input(sem::dense({3, kChannels}));
  const auto state = f.input(sem::dense({1, kHeads, kDepth, kDepth}));
  const auto flatten = [&](Value x) {
    return glm::reshape(f, x, {1, kStreams * kHidden});
  };
  const auto mix_norm = [&](Value x) {
    return f.one(c::Operation(c::RmsNorm{.epsilon = glm::kNormEpsilon}),
        {flatten(x), core::filled(f, sem::dense({kStreams * kHidden}), 1)});
  };

  const auto attn_normalized = mix_norm(residual);
  auto attn_fn = kda::projection(f, attn_normalized, 24,
      kStreams * kHidden, kda::Storage::kQ8_0, 233);
  const auto attn = core::hyper_connection_prepare(f, attn_fn.output, residual);
  auto input_scale = sem::oracle::scale(kHidden);
  const auto attn_input = glm::norm(f, attn.collapsed, input_scale);
  kda::Reference kda_reference;
  const auto attention = glm::attach(f, kda::complete(profile, &kda_reference),
      {attn_input, history, state});
  const auto attn_post = f.one(c::model_ops::hyper_connection_post(),
      {attention[5], residual, attn.post, attn.combine});

  // FFN has its own mixing role and parameters, recomputed from attention's
  // actual expanded result. The distinct fixture seed detects role swapping.
  const auto ffn_normalized = mix_norm(attn_post);
  auto ffn_fn = kda::projection(f, ffn_normalized, 24,
      kStreams * kHidden, kda::Storage::kQ8_0, 241);
  const auto ffn = core::hyper_connection_prepare(f, ffn_fn.output, attn_post);
  auto ffn_scale = sem::oracle::scale(kHidden);
  std::rotate(ffn_scale.begin(), ffn_scale.begin() + 11, ffn_scale.end());
  const auto ffn_input = glm::norm(f, ffn.collapsed, ffn_scale);
  glm::moe::MoeReference moe_reference;
  const auto feed_forward = glm::attach(f,
      glm::moe::moe_chain(*found, false, false, &moe_reference, 0.5F),
      {ffn_input});
  const auto ffn_branch = glm::reshape(f, feed_forward[5], {1, kHidden});
  const auto mhc = f.one(c::model_ops::hyper_connection_post(),
      {ffn_branch, attn_post, ffn.post, ffn.combine});
  const auto contract = glm::reduce(f, mhc, 1, c::ReductionKind::kMean);
  auto final_scale = sem::oracle::scale(kHidden);
  std::reverse(final_scale.begin(), final_scale.end());
  const auto tap = glm::norm(f, contract, final_scale);
  // Materialize the diagnostic reshape; public output bindings own storage.
  const auto ffn_branch_output = f.one(c::Operation(c::Copy{}), {ffn_branch});
  f.outputs = {attention[0], attention[2], mhc, tap, attn_normalized,
      attn_fn.output, attn.combine, attn.collapsed, attn_input,
      attention[1], attention[3], attention[4], attention[5], attn_post,
      ffn_normalized, ffn_fn.output, ffn.combine, ffn.collapsed,
      ffn_input, ffn_branch_output, contract};
  f.state_ports = {{1, r.ports.history}, {2, r.ports.state}};

  r.evaluate = [attn_fn = std::move(attn_fn), ffn_fn = std::move(ffn_fn),
      kda_reference = std::move(kda_reference),
      moe_reference = std::move(moe_reference), input_scale = std::move(input_scale),
      ffn_scale = std::move(ffn_scale), final_scale = std::move(final_scale)]
      (std::span<const float> x, std::span<const float> past,
       std::span<const float> prior) -> Outputs {
    if (x.size() != std::size_t(kStreams * kHidden) ||
        past.size() != std::size_t(3 * kChannels) ||
        prior.size() != std::size_t(kHeads * kDepth * kDepth))
      throw std::runtime_error("GLM target component reference input extent");
    const std::vector<float> ones(kStreams * kHidden, 1), zero(kHidden, 0);
    const std::vector<float> scales{0.125F, 0.25F, 0.5F};
    std::vector<float> bases(24);
    for (std::size_t i = 0; i < bases.size(); ++i)
      bases[i] = float(int(i % 7) - 3) / 8.0F;
    const auto normalized = [&](std::span<const float> residual_values) {
      return sem::oracle::rms_norm_rows(residual_values, ones,
          kStreams * kHidden, glm::kNormEpsilon);
    };
    auto an = normalized(x);
    auto am = attn_fn.reference(an);
    auto ap = core::hyper_reference(am, x, zero, kHidden, scales, bases);
    auto ai = sem::oracle::rms_norm_rows(ap[1], input_scale,
                                        kHidden, glm::kNormEpsilon);
    auto kd = kda_reference(ai, past, prior);
    auto ax = core::hyper_reference(am, x, kd[5], kHidden, scales, bases)[2];
    auto fn = normalized(ax);
    auto fm = ffn_fn.reference(fn);
    auto fp = core::hyper_reference(fm, ax, zero, kHidden, scales, bases);
    auto fi = sem::oracle::rms_norm_rows(fp[1], ffn_scale,
                                        kHidden, glm::kNormEpsilon);
    auto branch = moe_reference(fi)[5];
    auto expanded = core::hyper_reference(fm, ax, branch, kHidden, scales, bases)[2];
    std::vector<float> contracted(kHidden);
    for (std::size_t d = 0; d < contracted.size(); ++d) {
      double sum = 0;
      for (std::size_t s = 0; s < std::size_t(kStreams); ++s)
        sum += expanded[s * kHidden + d];
      contracted[d] = float(sum / double(kStreams));
    }
    auto target_tap = sem::oracle::rms_norm_rows(contracted, final_scale,
                                               kHidden, glm::kNormEpsilon);
    return {std::move(kd[0]), std::move(kd[2]), std::move(expanded),
        std::move(target_tap), std::move(an), std::move(am), std::move(ap[0]),
        std::move(ap[1]), std::move(ai), std::move(kd[1]), std::move(kd[3]),
        std::move(kd[4]), std::move(kd[5]), std::move(ax), std::move(fn),
        std::move(fm), std::move(fp[0]), std::move(fp[1]), std::move(fi),
        std::move(branch), std::move(contracted)};
  };
  return r;
}

// read_output(port) only observes the caller's actual output lease; no result
// is uploaded or used to rebuild a later input. Keep the established selected
// profile accuracy gate; this component introduces no relaxed tolerance.
template <class ReadOutput>
inline void observe(const Recipe &recipe, std::span<const float> residual,
                    std::span<const float> history, std::span<const float> state,
                    ReadOutput &&read_output) {
  const auto expected = recipe.evaluate(residual, history, state);
  if (expected.size() != Ports::count || recipe.fragment.outputs.size() != Ports::count)
    throw std::runtime_error("GLM target component output extent");
  for (std::size_t port = 0; port < expected.size(); ++port) {
    SCOPED_TRACE(kPortNames[port]);
    const auto actual = read_output(port);
    sem::expect_output(recipe.fragment, actual, expected[port]);
  }
}
} // namespace glm_mtp_target_component
