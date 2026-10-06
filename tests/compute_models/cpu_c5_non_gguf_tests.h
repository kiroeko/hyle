#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_C5_NON_GGUF_SUITE CpuGraphC5ActualNonGguf
#else
#define HYLE_C5_NON_GGUF_SUITE CpuC5ActualNonGguf
#endif
class HYLE_C5_NON_GGUF_SUITE : public ::testing::TestWithParam<const cpu_c5_non_gguf_inputs::Case *> {};
TEST_P(HYLE_C5_NON_GGUF_SUITE, ActualPackingAndWidthUseOriginalProgramAcrossReplay) {
  namespace f=cpu_c5_non_gguf_fixture;namespace in=cpu_c5_non_gguf_inputs;
  const auto &item=*GetParam();SCOPED_TRACE(item.id);
  const auto identity=f::profile(item);SCOPED_TRACE(identity.artifact_repository);
  if(item.format==in::Format::kNf4||item.format==in::Format::kAwq||item.format==in::Format::kCt)
    run_q24_semantic_fragment(f::artifact(item,&identity));
  else if(item.n>32768&&!item.embedding) {
    for(std::int64_t first=0;first<item.n;first+=4096) {
      SCOPED_TRACE(first);
      run_q24_semantic_fragment(f::mlx(item,&identity,std::min<std::int64_t>(4096,item.n-first),first));
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
  } else run_q24_semantic_fragment(f::mlx(item,&identity,item.n));
}
INSTANTIATE_TEST_SUITE_P(C5PackedContracts,HYLE_C5_NON_GGUF_SUITE,
    ::testing::ValuesIn([] {
      std::vector<const cpu_c5_non_gguf_inputs::Case *> result;
      for(const auto &item:cpu_c5_non_gguf_inputs::cases)result.push_back(&item);
      return result;
    }()),[](const auto &parameter){return std::string(parameter.param->id);});
#undef HYLE_C5_NON_GGUF_SUITE
#endif
