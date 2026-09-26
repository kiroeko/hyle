#include <gtest/gtest.h>

#include <array>

#include "../compute_models/d5_moe_profile_cases.h"
#include "../compute_models/d5_qwen_gguf_vocabulary_direct_test.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"

namespace {
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace direct = d5_qwen_gguf_vocabulary_direct_test;
namespace cuda = hyle::compute::cuda;
inline constexpr std::array kMatureProviders{
    hyle::compute::ggml_cuda::kProviderId,
    hyle::compute::cublas::kProviderId,
    hyle::compute::cuda_native::kCcclProviderId};

class CudaQwenGgufVocabulary
    : public ::testing::TestWithParam<const cases::Case *> {};

TEST_P(CudaQwenGgufVocabulary, HeadRunsEntireLogicalVocabulary) {
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  direct::check_head(context, memory.ref(), kMatureProviders, *GetParam());
}

TEST_P(CudaQwenGgufVocabulary, GatherMapsGlobalRowsAndRejectsOutOfBounds) {
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  direct::check_gather(context, memory.ref(), kMatureProviders, *GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    D5ExactQwenGguf, CudaQwenGgufVocabulary,
    ::testing::Values(&cases::kCases[0], &cases::kCases[1], &cases::kCases[2],
                      &cases::kCases[3], &cases::kCases[4], &cases::kCases[5]));

INSTANTIATE_TEST_SUITE_P(
    D5ExactMoeVocabulary, CudaQwenGgufVocabulary,
    ::testing::ValuesIn(d5_moe_profile_cases::kVocabularyPointers));
} // namespace
