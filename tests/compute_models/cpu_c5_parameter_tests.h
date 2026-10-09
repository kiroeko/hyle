#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS) || defined(HYLE_GGML_CUDA_IQ1_EXTENSIONS) || defined(HYLE_TEST_VULKAN_CATALOGUE)
#if defined(HYLE_TEST_VULKAN_CATALOGUE) && defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
#define HYLE_C5_PARAMETER_SUITE VulkanGraphCatalogueParameters
#define HYLE_C5_DEEPSEEK_SUITE VulkanGraphCatalogueNewDeepSeek
#elif defined(HYLE_TEST_VULKAN_CATALOGUE)
#define HYLE_C5_PARAMETER_SUITE VulkanCatalogueParameters
#define HYLE_C5_DEEPSEEK_SUITE VulkanCatalogueNewDeepSeek
#elif defined(HYLE_TEST_GRAPH_ROUTE_CUDA)
#define HYLE_C5_PARAMETER_SUITE CudaGraphC5Parameters
#define HYLE_C5_DEEPSEEK_SUITE CudaGraphC5NewDeepSeek
#elif defined(HYLE_C5_GGUF_DOT_REFERENCE)
#define HYLE_C5_PARAMETER_SUITE CudaDirectC5Parameters
#define HYLE_C5_DEEPSEEK_SUITE CudaDirectC5NewDeepSeek
#elif defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_C5_PARAMETER_SUITE CpuGraphC5Parameters
#define HYLE_C5_DEEPSEEK_SUITE CpuGraphC5NewDeepSeek
#else
#define HYLE_C5_PARAMETER_SUITE CpuC5Parameters
#define HYLE_C5_DEEPSEEK_SUITE CpuC5NewDeepSeek
#endif
class HYLE_C5_PARAMETER_SUITE : public ::testing::TestWithParam<const cpu_c5_parameters::ConvCase *> {};
TEST_P(HYLE_C5_PARAMETER_SUITE, ActualGgufConvolutionStorageFeedsCompleteRecurrentState) {
  const auto &item = *GetParam();
  const auto identity = cpu_c5_gguf_fixture::profile(item.profile);
  SCOPED_TRACE(identity.artifact_repository);
  SCOPED_TRACE(identity.variant);
  run_q24_semantic_fragment(cpu_c5_parameters::convolution(item, &identity));
}
INSTANTIATE_TEST_SUITE_P(C5ActualConvolutionParameters, HYLE_C5_PARAMETER_SUITE,
    ::testing::ValuesIn([] {
      std::vector<const cpu_c5_parameters::ConvCase *> result;
      for (const auto &item : cpu_c5_parameters::convolutions) result.push_back(&item);
      return result;
    }()), [](const auto &parameter) { return std::string(parameter.param->id); });
class HYLE_C5_DEEPSEEK_SUITE : public ::testing::TestWithParam<unsigned> {};
TEST_P(HYLE_C5_DEEPSEEK_SUITE, ActualMixedMainAttentionMoeAndHyperHeadReplay) {
  const auto identity = cpu_c5_gguf_fixture::profile(GetParam());
  SCOPED_TRACE(identity.artifact_repository);
  SCOPED_TRACE(identity.variant);
  const auto model = cpu_c5_parameters::new_deepseek_case(&identity);
  for (std::int64_t ratio : {0, 4, 128}) {
    SCOPED_TRACE(ratio);
#if defined(HYLE_TEST_VULKAN_CATALOGUE)
    run_q24_semantic_fragment(d5_deepseek_remaining_fragments::attention(model, ratio, vulkan_current_state_test::gguf_reference()));
#else
    run_q24_semantic_fragment(d5_deepseek_remaining_fragments::attention(model, ratio));
#endif
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
#if defined(HYLE_TEST_VULKAN_CATALOGUE)
  run_q24_semantic_fragment(d5_deepseek_remaining_fragments::compressor(model, 4, true, {}, vulkan_current_state_test::gguf_reference()));
#else
  run_q24_semantic_fragment(d5_deepseek_remaining_fragments::compressor(model, 4, true));
#endif
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  run_q24_semantic_fragment(d5_moe_execution_fragments::gguf_moe(model));
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  run_q24_semantic_fragment(d5_moe_execution_fragments::hash_moe(model));
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  cpu_c5_parameters::new_deepseek_hyper_head(model, run_q24_semantic_fragment);
}
INSTANTIATE_TEST_SUITE_P(C5ExactNewGgufMain, HYLE_C5_DEEPSEEK_SUITE,
    ::testing::ValuesIn(cpu_c5_parameters::new_deepseek_profiles()),
    [](const auto &parameter) { return "Profile" + std::to_string(parameter.param); });
TEST(HYLE_C5_PARAMETER_SUITE, Qwen24ActualHalfSplitFrequencyNormAndKvFeedback) {
  for (const auto &p : {0u, 22u, 48u, 49u}) {
    const auto identity = cpu_c5_gguf_fixture::profile(p);
    SCOPED_TRACE(identity.variant);
    ASSERT_TRUE(identity.model_repository.starts_with("Qwen/"));
    run_q24_semantic_fragment(cpu_c5_qwen_attention::make(&identity));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
TEST(HYLE_C5_PARAMETER_SUITE, QwenActualWidthsUseDeclaredFloatNormEpsilon) {
  for (const auto &model : {std::pair{0u, 2048LL}, std::pair{22u, 5120LL}, std::pair{47u, 8192LL}}) {
    const auto identity = cpu_c5_gguf_fixture::profile(model.first);
    SCOPED_TRACE(identity.model_name);
    run_q24_semantic_fragment(cpu_c5_qwen_attention::norm_residual(&identity, model.second));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
TEST(HYLE_C5_PARAMETER_SUITE, Qwen24OriginalMtpCodecNormOrderAndCarry) {
  for (unsigned profile : {48u, 49u}) {
    const auto identity = cpu_c5_gguf_fixture::profile(profile);
    SCOPED_TRACE(identity.variant);
    run_q24_semantic_fragment(cpu_c5_qwen_attention::mtp(&identity));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
TEST(HYLE_C5_PARAMETER_SUITE, Qwen24CompleteNextNUsesMixedRolesCacheMoeAndFullHead) {
  for (unsigned profile : {48u, 49u}) {
    SCOPED_TRACE(profile);
    cpu_c5_qwen_nextn::check(profile, run_q24_semantic_fragment);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
#undef HYLE_C5_PARAMETER_SUITE
#undef HYLE_C5_DEEPSEEK_SUITE
#endif
