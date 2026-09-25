#include <gtest/gtest.h>

#include <array>

#include "../compute_models/d5_qwen_gguf_vocabulary_graph_test.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace cuda = c::cuda;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace graph_test = d5_qwen_gguf_vocabulary_graph_test;
inline constexpr std::array kMatureProviders{
    c::ggml_cuda::kProviderId, c::cublas::kProviderId,
    c::cuda_native::kCcclProviderId};

class CudaGraphQwenGgufVocabulary
    : public ::testing::TestWithParam<const cases::Case *> {};

TEST_P(CudaGraphQwenGgufVocabulary, HeadRunsEntireLogicalVocabulary) {
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  graph_test::check_head(
      context, memory.ref(), kMatureProviders, *GetParam(),
      [&](const g::Graph &logical, const c::WorkloadContext &workload) {
        return g::cuda::compile(logical, context, memory.ref(),
                                {.workload = workload})
            .value();
      });
}

TEST_P(CudaGraphQwenGgufVocabulary,
       GatherMapsGlobalRowsAndRejectsOutOfBounds) {
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  graph_test::check_gather(
      context, memory.ref(), kMatureProviders, *GetParam(),
      [&](const g::Graph &logical, const c::WorkloadContext &workload) {
        return g::cuda::compile(logical, context, memory.ref(),
                                {.workload = workload})
            .value();
      });
}

INSTANTIATE_TEST_SUITE_P(
    D5ExactQwenGguf, CudaGraphQwenGgufVocabulary,
    ::testing::Values(&cases::kCases[0], &cases::kCases[1], &cases::kCases[2],
                      &cases::kCases[3], &cases::kCases[4], &cases::kCases[5]));
} // namespace
