#pragma once
#if defined(HYLE_TEST_GRAPH_ROUTE_CUDA)
#define HYLE_R1_SUITE CudaGraphCurrentRegions
#else
#define HYLE_R1_SUITE CudaDirectCurrentRegions
#endif
// Same current model programs and source identities as the common catalogue.
// Their execution is through the ordinary CUDA runner, not the CPU witness.
TEST(HYLE_R1_SUITE, DsparkTargetHiddenAndFp32Confidence) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    run_q24_semantic_fragment(cpu_c4_model_branches::target_hidden(model));
    ASSERT_FALSE(::testing::Test::HasFailure());
    run_q24_semantic_fragment(cpu_c4_model_branches::confidence(model));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(HYLE_R1_SUITE, DsparkMarkovFullVocabularyWithLogitBias) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    cpu_c4_model_branches::check_markov(model,run_q24_semantic_fragment);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(HYLE_R1_SUITE, DsparkTargetPrefillAndFiveTokenAttentionCache) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    auto prefill=cpu_c4_model_branches::cache_prefill(model,128);
    std::vector<float> actual_cache;
    prefill.observe=[&](std::size_t port,std::span<const float> values) {
      EXPECT_EQ(port,0U);actual_cache.assign(values.begin(),values.end());
    };
    run_q24_semantic_fragment(std::move(prefill));
    ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_EQ(actual_cache.size(),cpu_c4_model_branches::window*512);
    run_q24_semantic_fragment(cpu_c4_model_branches::cache_prefill(model,129));
    ASSERT_FALSE(::testing::Test::HasFailure());
    // The next program consumes the actual public cache. Give its independent
    // source equation the same initial bytes; prefill keeps its own full gate.
    run_q24_semantic_fragment(cpu_c4_model_branches::draft_attention(model,actual_cache,actual_cache));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(HYLE_R1_SUITE, NewOfficialMainAttentionCompressionAndMoe) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    const auto shape=model.main_case();
    for(std::int64_t ratio:{0,4,128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::attention(shape,ratio));
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
    run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(shape,false,nullptr,false));
    ASSERT_FALSE(::testing::Test::HasFailure());
    run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(shape,true,nullptr,false));
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(HYLE_R1_SUITE, CompleteThreeStageDsparkWithSharedEmbeddingAndConfidence) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    cpu_c4_model_branches::check_complete_draft(model,run_q24_semantic_fragment);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
#undef HYLE_R1_SUITE
