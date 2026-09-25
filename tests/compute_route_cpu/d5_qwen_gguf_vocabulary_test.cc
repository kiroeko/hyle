#include <gtest/gtest.h>

#include <array>

#include "../compute_models/d5_qwen_gguf_vocabulary_direct_test.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"

namespace {
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace direct = d5_qwen_gguf_vocabulary_direct_test;
inline constexpr std::array kMatureProviders{
    hyle::compute::ggml_cpu::kProviderId,
    hyle::compute::onednn::kProviderId};

class CpuQwenGgufVocabulary
    : public ::testing::TestWithParam<const cases::Case *> {};

TEST_P(CpuQwenGgufVocabulary, HeadRunsEntireLogicalVocabulary) {
  auto context = hyle::compute::cpu::create_context().value();
  direct::check_head(context, hyle::mem::system_resource(),
                     kMatureProviders, *GetParam());
}

TEST_P(CpuQwenGgufVocabulary, GatherMapsGlobalRowsAndRejectsOutOfBounds) {
  auto context = hyle::compute::cpu::create_context().value();
  direct::check_gather(context, hyle::mem::system_resource(),
                       kMatureProviders, *GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    D5ExactQwenGguf, CpuQwenGgufVocabulary,
    ::testing::Values(&cases::kCases[0], &cases::kCases[1], &cases::kCases[2],
                      &cases::kCases[3], &cases::kCases[4], &cases::kCases[5]));
} // namespace
