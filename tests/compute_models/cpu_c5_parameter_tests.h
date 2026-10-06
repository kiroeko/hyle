#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_C5_PARAMETER_SUITE CpuGraphC5Parameters
#else
#define HYLE_C5_PARAMETER_SUITE CpuC5Parameters
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
#undef HYLE_C5_PARAMETER_SUITE
#endif
