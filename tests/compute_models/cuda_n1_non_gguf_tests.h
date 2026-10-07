#pragma once
#if defined(HYLE_COMPUTE_TORCH_CUDA)
#if defined(HYLE_TEST_GRAPH_ROUTE_CUDA)
#define HYLE_N1_SUITE CudaGraphN1ActualArtifacts
#else
#define HYLE_N1_SUITE CudaDirectN1ActualArtifacts
#endif
class HYLE_N1_SUITE : public ::testing::TestWithParam<const cpu_c5_non_gguf_inputs::Case *> {};
TEST_P(HYLE_N1_SUITE, OriginalPackedWeightPreparationFeedsMatureProjectionAcrossReplay) {
  namespace f = cpu_c5_non_gguf_fixture;
  const auto &item = *GetParam();
  SCOPED_TRACE(item.id);
  const auto identity = f::profile(item);
  SCOPED_TRACE(identity.artifact_repository);
  run_q24_semantic_fragment(f::artifact(item, &identity));
}
INSTANTIATE_TEST_SUITE_P(FixedPackedContracts, HYLE_N1_SUITE,
    ::testing::ValuesIn([] {
      std::vector<const cpu_c5_non_gguf_inputs::Case *> result;
      for (const auto &item : cpu_c5_non_gguf_inputs::cases)
        if (item.format == cpu_c5_non_gguf_inputs::Format::kNf4 ||
            item.format == cpu_c5_non_gguf_inputs::Format::kAwq ||
            item.format == cpu_c5_non_gguf_inputs::Format::kCt)
          result.push_back(&item);
      return result;
    }()), [](const auto &parameter) { return std::string(parameter.param->id); });
#undef HYLE_N1_SUITE
#endif
