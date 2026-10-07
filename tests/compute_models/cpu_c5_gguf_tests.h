#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS) || defined(HYLE_GGML_CUDA_IQ1_EXTENSIONS)
#if defined(HYLE_TEST_GRAPH_ROUTE_CUDA)
#define HYLE_C5_GGUF_SUITE CudaGraphC5ActualGguf
#elif defined(HYLE_C5_GGUF_DOT_REFERENCE)
#define HYLE_C5_GGUF_SUITE CudaDirectC5ActualGguf
#elif defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_C5_GGUF_SUITE CpuGraphC5ActualGguf
#else
#define HYLE_C5_GGUF_SUITE CpuC5ActualGguf
#endif
namespace c5f=cpu_c5_gguf_fixture;
class HYLE_C5_GGUF_SUITE : public ::testing::TestWithParam<const cpu_c5_inputs::Case *> {};
TEST_P(HYLE_C5_GGUF_SUITE, ActualRoleAndWidthUseOriginalCodecAcrossReplay) {
  const auto &item=*GetParam();SCOPED_TRACE(item.id);
  const auto identity=c5f::profile(cpu_c5_inputs::aliases[item.alias_begin]);
  SCOPED_TRACE(identity.artifact_repository);
  SCOPED_TRACE(identity.variant);
  const c5f::Bank bank(item);
  using Kind=cpu_c5_inputs::Kind;
  if(item.kind==Kind::kEmbedding||item.kind==Kind::kPosition||item.kind==Kind::kInteger)
    run_q24_semantic_fragment(c5f::lookup(item,bank,&identity));
  else if(item.kind==Kind::kExpert||item.kind==Kind::kHeadGroup)
    run_q24_semantic_fragment(c5f::grouped(item,bank,&identity));
  else if(item.kind==Kind::kHead) {
    for(std::int64_t first=0;first<item.n;first+=4096) {
      SCOPED_TRACE(first);
      run_q24_semantic_fragment(c5f::projection(item,bank,&identity,std::min<std::int64_t>(4096,item.n-first),first));
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
  } else run_q24_semantic_fragment(c5f::projection(item,bank,&identity,item.n));
}
INSTANTIATE_TEST_SUITE_P(C5HeaderContracts,HYLE_C5_GGUF_SUITE,
    ::testing::ValuesIn([] {
      std::vector<const cpu_c5_inputs::Case *> result;
      for(const auto &item:cpu_c5_inputs::cases)result.push_back(&item);
      return result;
    }()),[](const auto &parameter){return std::string(parameter.param->id);});

TEST(HYLE_C5_GGUF_SUITE, All104ExactArtifactIdentitiesExecuteActualProjection) {
  for(unsigned profile=0;profile<std::size(cpu_c5_inputs::profiles);++profile) {
    const auto identity=c5f::profile(profile);SCOPED_TRACE(identity.artifact_repository);
    SCOPED_TRACE(identity.variant);
    const cpu_c5_inputs::Case *selected=nullptr;
    for(const auto &item:cpu_c5_inputs::cases) {
      if(item.kind!=cpu_c5_inputs::Kind::kProjection)continue;
      const auto aliases=std::span(cpu_c5_inputs::aliases).subspan(item.alias_begin,item.alias_count);
      if(std::ranges::find(aliases,profile)==aliases.end())continue;
      if(!selected||item.k*item.n<selected->k*selected->n)selected=&item;
    }
    ASSERT_NE(selected,nullptr);
    const c5f::Bank bank(*selected);
    run_q24_semantic_fragment(c5f::projection(*selected,bank,&identity,selected->n));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
#undef HYLE_C5_GGUF_SUITE
#endif
