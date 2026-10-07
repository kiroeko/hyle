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
    auto draft=cpu_c4_model_branches::draft_attention(model,actual_cache,actual_cache);
    const auto source_observer=draft.observe;
    const auto source_validator=draft.validate_intermediate;
    const auto weights=draft.preparation;
    constexpr std::size_t k=4096,n=1024;
    // Use the independently generated source weight bank, not provider output.
    // This original projection has sparse weights; retain only its source
    // terms in the oracle rather than scan the whole bank on every replay.
    std::vector<std::vector<std::pair<std::size_t,float>>> terms(model.groups*n);
    const auto &source_weights=weights->replays.front().outputs[0];
    for(std::size_t row=0;row<terms.size();++row)
      for(std::size_t col=0;col<k;++col)
        if(source_weights[row*k+col]!=0)terms[row].emplace_back(col,source_weights[row*k+col]);
    std::vector<float> inverse;
    draft.observe=[&,source_observer](std::size_t port,std::span<const float> values) {
      if(source_observer)source_observer(port,values);
      if(port==2)inverse.assign(values.begin(),values.end());
    };
    draft.validate_intermediate=[&,source_validator](std::size_t port,std::span<const float> values) {
      if(port!=3)return source_validator?source_validator(port,values):false;
      EXPECT_EQ(values.size(),std::size_t(cpu_c4_model_branches::block*model.groups*n));
      for(std::size_t index=0;index<values.size();++index) {
        const auto token=index/(model.groups*n),expert=(index/n)%model.groups,row=index%n;
        double dot=0;
        for(const auto &[col,weight]:terms[expert*n+row]) {
          dot+=weight*double(inverse[(token*model.groups+expert)*k+col]);
        }
        const auto expected=float(half_test::decode(half_test::encode<hyle::simd::bfloat16_t>(float(dot))));
        EXPECT_EQ(values[index],expected)<<index;
      }
      return true;
    };
    run_q24_semantic_fragment(std::move(draft));
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
