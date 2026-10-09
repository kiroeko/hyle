#pragma once

#if defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
#define HYLE_VULKAN_CURRENT_STATE_SUITE VulkanGraphCurrentState
#else
#define HYLE_VULKAN_CURRENT_STATE_SUITE VulkanCurrentState
#endif
namespace {
class HYLE_VULKAN_CURRENT_STATE_SUITE : public ::testing::TestWithParam<cpu_c4_model_branches::Model> {};
TEST_P(HYLE_VULKAN_CURRENT_STATE_SUITE, TargetHiddenFp32ConfidenceAndMarkov) {
  const auto &model=GetParam();SCOPED_TRACE(model.profile->model_revision);
  vulkan_catalogue_test::run(cpu_c4_model_branches::target_hidden(model));
  ASSERT_FALSE(::testing::Test::HasFailure());
  vulkan_catalogue_test::run(cpu_c4_model_branches::confidence(model));
  ASSERT_FALSE(::testing::Test::HasFailure());
  cpu_c4_model_branches::check_markov(model,vulkan_catalogue_test::run);
}
TEST_P(HYLE_VULKAN_CURRENT_STATE_SUITE, PrefillFiveTokenAttentionAndCacheFeedback) {
  const auto &model=GetParam();SCOPED_TRACE(model.profile->model_revision);
  cpu_c4_model_branches::check_attention(model,vulkan_catalogue_test::run);
}
TEST_P(HYLE_VULKAN_CURRENT_STATE_SUITE, OfficialMainCompressionAttentionAndFp4Moe) {
  const auto &model=GetParam();SCOPED_TRACE(model.profile->model_revision);const auto shape=model.main_case();
  for(std::int64_t ratio:{0,4,128}) {
    SCOPED_TRACE(ratio);
    vulkan_catalogue_test::run(d5_deepseek_official_attention::attention(shape,ratio));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  for(bool down:{false,true}) {
    vulkan_catalogue_test::run(d5_deepseek_official_fragments::moe_chain(shape,down,nullptr,false));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST_P(HYLE_VULKAN_CURRENT_STATE_SUITE, CompleteThreeStageDspark) {
  const auto &model=GetParam();SCOPED_TRACE(model.profile->model_revision);
  cpu_c4_model_branches::check_complete_draft(model,vulkan_catalogue_test::run);
}
INSTANTIATE_TEST_SUITE_P(CurrentOfficial,HYLE_VULKAN_CURRENT_STATE_SUITE,
    ::testing::ValuesIn(vulkan_current_state_test::models()),
    [](const auto &item){return item.param.hidden==7168?"Pro0813":"Flash0731";});
TEST(HYLE_VULKAN_CURRENT_STATE_SUITE, QwenNamedQ4SidecarUsesActualQ3Q4Q6Mtp) {
  cpu_c4_qwen_mtp::check(vulkan_catalogue_test::run);
}
TEST(HYLE_VULKAN_CURRENT_STATE_SUITE, FlashQ8AndBfloatSidecarsUseMixedMxfp4ThreeStages) {
  for(const auto &model:cpu_c4_flash_sidecars::models) {
    SCOPED_TRACE(model.profile->variant);
    cpu_c4_flash_sidecars::check(model,vulkan_catalogue_test::run);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
} // namespace
#undef HYLE_VULKAN_CURRENT_STATE_SUITE
