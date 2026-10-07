#pragma once
#include <cstdio>
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
    const auto weights=draft.preparation;
    const auto reference_inverse=draft.replays.front().outputs[2];
    std::vector<float> inverse;
    draft.observe=[&,source_observer,weights,reference_inverse](std::size_t port,std::span<const float> values) {
      if(source_observer)source_observer(port,values);
      if(port==2)inverse.assign(values.begin(),values.end());
      if(port==3 && model.groups==16) {
        constexpr std::size_t target=71280,k=4096,n=1024;
        const auto token=target/(model.groups*n),expert=(target/n)%model.groups,row=target%n;
        const auto &w=weights->replays.front().outputs[0];
        double actual_dot=0,reference_dot=0;
        for(std::size_t col=0;col<k;++col) {
          const auto weight=w[(expert*n+row)*k+col];
          const auto index=(token*model.groups+expert)*k+col;
          actual_dot+=weight*double(inverse[index]);
          reference_dot+=weight*double(reference_inverse[index]);
          if(weight!=0)std::fprintf(stderr,"R1 TERM w=%g actual_x=%.9g reference_x=%.9g\n",
              weight,inverse[index],reference_inverse[index]);
        }
        std::fprintf(stderr,"R1 DOT actual-input=%.9g reference-input=%.9g gpu-out=%.9g\n",
            actual_dot,reference_dot,values[target]);
      }
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
