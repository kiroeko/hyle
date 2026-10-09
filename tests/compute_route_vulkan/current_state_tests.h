#pragma once

#if defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
#define HYLE_VULKAN_CURRENT_STATE_SUITE VulkanGraphCurrentState
#else
#define HYLE_VULKAN_CURRENT_STATE_SUITE VulkanCurrentState
#endif
namespace {
#if !defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
TEST(VulkanStateCause, ActualSinkInputsUseOriginalNativeAndIndependentEquation) {
  const auto model=vulkan_current_state_test::models()[1];
  const auto cache=d5_deepseek_official_attention::cache_reference(
      d5_gguf_test::activation(cpu_c4_model_branches::window*512,3),512,false);
  auto fragment=cpu_c4_model_branches::draft_attention(model,cache,cache);
  const auto sink=std::ranges::find_if(fragment.nodes,[](const auto &node){return node.operation.template custom_as<hyle::compute::model_ops::SinkAttention>().has_value();});
  ASSERT_NE(sink,fragment.nodes.end());
  fragment.nodes.resize(static_cast<std::size_t>(sink-fragment.nodes.begin())+1);
  fragment.outputs.clear();fragment.state_ports.clear();fragment.replays.resize(1);fragment.replays[0].outputs.clear();
  auto context=vulkan_test::context().value();bool observed=false;
  vulkan_catalogue_test::execute(context,std::move(fragment),nullptr,
      [&](const auto &operation,std::span<const hyle::compute::vulkan::Tensor> inputs,
          std::span<const hyle::compute::vulkan::Tensor> outputs) {
    const auto spec=operation.template custom_as<hyle::compute::model_ops::SinkAttention>();
    if(!spec) return;
    ASSERT_EQ(inputs.size(),5U);std::array<std::vector<float>,5> values;
    std::array<hyle::compute::TensorDesc,5> descriptions;
    for(std::size_t i=0;i<5;++i) {descriptions[i]=inputs[i].description();values[i]=vulkan_catalogue_test::floats(context,inputs[i]);}
    const auto actual=vulkan_catalogue_test::floats(context,outputs[0]);
    const auto native=vulkan_current_state_test::native_sink(descriptions,values,spec->scale);
    const auto nominal=d5_deepseek_remaining_fragments::sink_reference(values[0],values[1],values[2],values[3],values[4],
        descriptions[0].dimensions()[1],descriptions[1].dimensions()[1],spec->scale);
    const auto rounded=vulkan_current_state_test::rounded_sink(values,descriptions[0].dimensions()[1],descriptions[1].dimensions()[1],spec->scale);
    double native_error=0,nominal_error=0,rounded_error=0;
    ASSERT_EQ(actual.size(),native.size());ASSERT_EQ(actual.size(),nominal.size());
    for(std::size_t i=0;i<actual.size();++i) {
      native_error=std::max(native_error,std::abs(static_cast<double>(actual[i])-native[i]));
      nominal_error=std::max(nominal_error,std::abs(static_cast<double>(actual[i])-nominal[i]));
      rounded_error=std::max(rounded_error,std::abs(static_cast<double>(actual[i])-rounded[i]));
    }
    std::printf("STATE_SINK_CAUSE query_tokens=%lld heads=%lld depth=%lld keys=%lld native_max=%.9g nominal_max=%.9g rounded_max=%.9g\n",
        static_cast<long long>(descriptions[0].dimensions()[1]),static_cast<long long>(descriptions[0].dimensions()[2]),
        static_cast<long long>(descriptions[0].dimensions()[3]),static_cast<long long>(descriptions[1].dimensions()[1]),native_error,nominal_error,rounded_error);
    EXPECT_LE(native_error,1e-6);observed=true;
  });
  EXPECT_TRUE(observed);
}
#endif
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
    vulkan_catalogue_test::run(d5_deepseek_official_attention::attention(shape,ratio,nullptr,vulkan_current_state_test::main_reference()));
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
