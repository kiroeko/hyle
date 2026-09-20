#include "hyle/graph/providers/ggml_cuda.h"
#include "region_test_support.h"

TEST(GgmlCudaGraph, WholeOriginalRegionUsesIndependentInvocations) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  exercise(
      context, false,
      [&](const graph::Graph &logical, mem::MemoryResourceRef resource) {
        const std::array storage{graph::StorageDomain{.resource = resource}};
        const std::array targets{graph::ExecutionTarget{
            .context = &context, .default_storage = graph::StorageDomainId{0}}};
        const std::array lanes{
            graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
        return graph::compile(
            logical,
            {.storage_domains = storage, .targets = targets, .lanes = lanes});
      });
}
