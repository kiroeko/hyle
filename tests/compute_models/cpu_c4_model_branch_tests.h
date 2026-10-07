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
TEST(HYLE_C4_CPU_SUITE, FlashSidecarsKeepMixedMxfp4BfloatAndQ8ThreeStagePrograms) {
  for(const auto &model:cpu_c4_flash_sidecars::models) {
    SCOPED_TRACE(model.profile->variant);
    cpu_c4_flash_sidecars::check(model,run_q24_semantic_fragment);
  }
}
TEST(HYLE_C4_CPU_SUITE, SidecarOraclePreservesSelectedQ8MidpointDefinition) {
  ggml_cpu_init();std::array<float,32> input{};
  input[0]=127;input[1]=0.5F;input[2]=1.5F;input[3]=2.5F;
  input[4]=-0.5F;input[5]=-1.5F;input[6]=-2.5F;
  std::array<std::uint64_t,5> encoded{};
  cpu_c4_flash_sidecars::activation_encode(GGML_TYPE_Q8_0,input,encoded.data());
  const auto *codes=reinterpret_cast<const std::int8_t *>(encoded.data())+2;
#if (defined(_M_X64) || defined(__x86_64__)) && (defined(_WIN32) || defined(__linux__))
  if(hyle::compute::ggml_runtime::cpu_avx2_program_eligible(hyle::platform::cpu_features())) {
    EXPECT_EQ(codes[1],0);EXPECT_EQ(codes[2],2);EXPECT_EQ(codes[3],2);
    EXPECT_EQ(codes[4],0);EXPECT_EQ(codes[5],-2);EXPECT_EQ(codes[6],-2);
    return;
  }
#endif
  std::array<std::uint64_t,5> original{};
  ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(input.data(),original.data(),input.size());
  EXPECT_EQ(encoded,original);
}
#undef HYLE_C4_CPU_SUITE
