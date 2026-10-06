#pragma once
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_C4_CPU_SUITE CpuGraphC4Branches
#else
#define HYLE_C4_CPU_SUITE CpuC4Branches
#endif
TEST(HYLE_C4_CPU_SUITE, DsparkTargetHiddenAndFp32Confidence) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    run_q24_semantic_fragment(cpu_c4_model_branches::target_hidden(model));
    run_q24_semantic_fragment(cpu_c4_model_branches::confidence(model));
  }
}
TEST(HYLE_C4_CPU_SUITE, DsparkMarkovFullVocabularyWithLogitBias) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    cpu_c4_model_branches::check_markov(model,run_q24_semantic_fragment);
  }
}
TEST(HYLE_C4_CPU_SUITE, DsparkTargetPrefillAndFiveTokenAttentionCache) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    cpu_c4_model_branches::check_attention(model,run_q24_semantic_fragment);
  }
}
TEST(HYLE_C4_CPU_SUITE, NewOfficialMainAttentionCompressionAndMoe) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    const auto shape=model.main_case();
    for(std::int64_t ratio:{0,4,128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::attention(shape,ratio));
    }
    run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(shape,false,nullptr,false));
    run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(shape,true,nullptr,false));
  }
}
TEST(HYLE_C4_CPU_SUITE, CompleteThreeStageDsparkWithSharedEmbeddingAndConfidence) {
  for(const auto &model:cpu_c4_model_branches::models) {
    SCOPED_TRACE(model.profile->model_name);
    cpu_c4_model_branches::check_complete_draft(model,run_q24_semantic_fragment);
  }
}
TEST(HYLE_C4_CPU_SUITE, QwenQ4NamedSidecarUsesActualQ3Q4Q6AndCompleteMtp) {
  cpu_c4_qwen_mtp::check(run_q24_semantic_fragment);
}
#undef HYLE_C4_CPU_SUITE
