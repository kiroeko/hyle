#pragma once
#if defined(HYLE_COMPUTE_TORCH_CUDA)
#if defined(HYLE_TEST_GRAPH_ROUTE_CUDA)
#define HYLE_N2_SUITE CudaGraphN2ActualMlx
#else
#define HYLE_N2_SUITE CudaDirectN2ActualMlx
#endif
class HYLE_N2_SUITE : public ::testing::TestWithParam<const cpu_c5_non_gguf_inputs::Case *> {};
TEST_P(HYLE_N2_SUITE, OriginalPackedProjectionAndSelectedRowsAcrossReplay) {
  namespace f = cpu_c5_non_gguf_fixture;
  const auto &item = *GetParam();
  SCOPED_TRACE(item.id);
  const auto identity = f::profile(item);
  SCOPED_TRACE(identity.artifact_repository);
  run_q24_semantic_fragment(f::mlx(item, &identity, item.n));
}
INSTANTIATE_TEST_SUITE_P(FixedPackedContracts,HYLE_N2_SUITE,
  ::testing::ValuesIn([] {
    std::vector<const cpu_c5_non_gguf_inputs::Case *> result;
    for (const auto &item : cpu_c5_non_gguf_inputs::cases)
      if (item.format == cpu_c5_non_gguf_inputs::Format::kAffine ||
          item.format == cpu_c5_non_gguf_inputs::Format::kMx4 ||
          item.format == cpu_c5_non_gguf_inputs::Format::kMx8)
        result.push_back(&item);
    return result;
  }()),[](const auto &parameter) { return std::string(parameter.param->id); });
#undef HYLE_N2_SUITE
#endif
