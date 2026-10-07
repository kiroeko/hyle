#pragma once

#include "cuda_iq1_fixture.h"
#include "cpu_c5_gguf_inputs.h"
#include "d5_qwen38_24t_semantic_fragments.h"
#include "d5_moe_profile_cases.h"
#include "d5_qwen_gguf_vocabulary_cases.h"

namespace cuda_iq1_model_contracts {
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace in = cpu_c5_inputs;
namespace c = hyle::compute;
inline bool required(const in::Case &item) {
  return item.native == GGML_TYPE_IQ1_XS || item.native == GGML_TYPE_IQ1_XXS ||
         item.native == GGML_TYPE_IQ1_XXXS;
}
inline c::ModelArtifactProfileDescription identity(const in::Case &item) {
  const auto &p = in::profiles[in::aliases[item.alias_begin]];
  auto result = p.family == 2 ? d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile :
                                d5_moe_profile_cases::kimi_k3_primary_gguf;
  result.artifact_repository = p.repository;
  result.artifact_revision = p.revision;
  result.variant = p.variant;
  result.manifest_digest = p.digest;
  result.format_family = "gguf";
  return result;
}
inline sem::Fragment make(const in::Case &item,
                          const c::ModelArtifactProfileDescription &profile) {
  constexpr std::int64_t resident_experts = 3, slots = 2;
  const cuda_iq1_fixture::Format format{item.quant, item.native};
  const auto w = c::gguf_grouped_weight_description(item.quant, c::DataType::kFloat32,
                    resident_experts, item.n, item.k).value();
  auto packed = cuda_iq1_fixture::weights(format, resident_experts * item.n, item.k);
  const auto sums = cuda_iq1_fixture::sums(format, packed, resident_experts * item.n, item.k);
  sem::Fragment f{.profile=&profile,.region=0x0c050301,.absolute_tolerance=8.0e-5F};
  const auto weight = f.constant(w, std::move(packed));
  const auto a = f.input(sem::dense({1, slots, item.k}));
  const auto ids = f.input(sem::dense({1, slots}, c::DataType::kInt32));
  f.outputs = {f.one(c::model_ops::grouped_matmul(), {weight, a, ids})};
  for (int replay=0;replay<2;++replay) {
    // q8 activation scales are exact powers of two in both original codecs.
    // Retain the common numeric gate instead of widening it for half-scale rounding.
    std::vector<float> activation(slots*item.k,replay ? 127.0F/128 : 127.0F/256);
    const std::vector<std::int32_t> selected = replay ? std::vector<std::int32_t>{1,1} :
                                                     std::vector<std::int32_t>{2,0};
    std::vector<float> expected(slots*item.n);
    for (std::int64_t slot=0;slot<slots;++slot)
      for (std::int64_t n=0;n<item.n;++n)
        expected[slot*item.n+n]=sums[selected[slot]*item.n+n]*(127.0F/64)*(replay?2:1);
    f.replays.push_back({{sem::bytes(activation),sem::bytes(selected)},{std::move(expected)}});
  }
  f.invalid_index_input=1;
  return f;
}
} // namespace cuda_iq1_model_contracts
