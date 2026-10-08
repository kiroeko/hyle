#pragma once

#if defined(HYLE_TEST_GRAPH_ROUTE_APPLE)
#define HYLE_APPLE_BRANCHES AppleParityGraphBranches
#define HYLE_APPLE_SIDECARS AppleParityGraphSidecars
#define HYLE_APPLE_MTP AppleParityGraphMtp
#else
#define HYLE_APPLE_BRANCHES AppleParityDirectBranches
#define HYLE_APPLE_SIDECARS AppleParityDirectSidecars
#define HYLE_APPLE_MTP AppleParityDirectMtp
#endif

class HYLE_APPLE_BRANCHES : public ::testing::TestWithParam<cpu_c4_model_branches::Model> {};
TEST_P(HYLE_APPLE_BRANCHES, DsparkTargetHiddenAndFp32Confidence) {
  const auto &model = GetParam();
  SCOPED_TRACE(model.profile->model_revision);
  run_q24_semantic_fragment(cpu_c4_model_branches::target_hidden(model));
  ASSERT_FALSE(::testing::Test::HasFailure());
  run_q24_semantic_fragment(cpu_c4_model_branches::confidence(model));
}
TEST_P(HYLE_APPLE_BRANCHES, DsparkMarkovFullVocabularyWithLogitBias) {
  const auto &model = GetParam();
  SCOPED_TRACE(model.profile->model_revision);
  cpu_c4_model_branches::check_markov(model, run_q24_semantic_fragment);
}
TEST_P(HYLE_APPLE_BRANCHES, DsparkTargetPrefillAndFiveTokenAttentionCache) {
  auto model = GetParam();
  SCOPED_TRACE(model.profile->model_revision);
  apple_dspark_native_reference::bind(model);
  cpu_c4_model_branches::check_attention(model, run_q24_semantic_fragment);
}
TEST_P(HYLE_APPLE_BRANCHES, OfficialMainCompressionAttentionAndFp4Moe) {
  const auto &model=GetParam();
  SCOPED_TRACE(model.profile->model_revision);
  const auto shape=model.main_case();
  for(std::int64_t ratio:{0,4,128}) {
    SCOPED_TRACE(ratio);
    run_q24_semantic_fragment(d5_deepseek_official_attention::attention(
        shape, ratio, nullptr, apple_dspark_native_reference::main_reference()));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(shape,false,nullptr,false));
  ASSERT_FALSE(::testing::Test::HasFailure());
  run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(shape,true,nullptr,false));
}
TEST_P(HYLE_APPLE_BRANCHES, CompleteOfficialThreeStageDsparkWithFp4Experts) {
  auto model=GetParam();
  SCOPED_TRACE(model.profile->model_revision);
  apple_dspark_native_reference::bind(model);
  cpu_c4_model_branches::check_complete_draft(model,run_q24_semantic_fragment);
}
INSTANTIATE_TEST_SUITE_P(CurrentOfficial, HYLE_APPLE_BRANCHES,
    ::testing::ValuesIn([] {
      auto models = cpu_c4_model_branches::models;
      // Same qualified native block program as the existing Apple GLM
      // consumer: separate F32 product and sum, fixed before building all
      // independent references; no per-element choice of a closer answer.
      for (auto &model : models) model.fused_scale_accumulation = false;
      return models;
    }()),
    [](const auto &info) { return info.param.hidden == 7168 ? "Pro0813" : "Flash0731"; });

TEST(HYLE_APPLE_MTP, QwenQ4NamedSidecarUsesActualQ3Q4Q6AndCompleteMtp) {
  cpu_c4_qwen_mtp::check(run_q24_semantic_fragment);
}
class HYLE_APPLE_SIDECARS : public ::testing::TestWithParam<cpu_c4_flash_sidecars::Model> {};
TEST_P(HYLE_APPLE_SIDECARS, FlashMixedMxfp4AndThreeStageDspark) {
  const auto &model = GetParam();
  SCOPED_TRACE(model.profile->variant);
  cpu_c4_flash_sidecars::check(model, run_q24_semantic_fragment);
}
INSTANTIATE_TEST_SUITE_P(CurrentGguf, HYLE_APPLE_SIDECARS,
    ::testing::ValuesIn(cpu_c4_flash_sidecars::models),
    [](const auto &info) { return info.param.bf16 ? "BF16" : "Q8_0"; });

#undef HYLE_APPLE_BRANCHES
#undef HYLE_APPLE_SIDECARS
#undef HYLE_APPLE_MTP
