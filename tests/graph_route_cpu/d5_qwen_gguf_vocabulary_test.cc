#include <gtest/gtest.h>

#include <array>

#include "../compute_models/d5_moe_profile_cases.h"
#include "../compute_models/d5_qwen_gguf_vocabulary_graph_test.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/providers/onednn.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace graph_test = d5_qwen_gguf_vocabulary_graph_test;
inline constexpr std::array kMatureProviders{
    c::ggml_cpu::kProviderId, c::onednn::kProviderId,
    g::onednn::kProviderId};

class CpuGraphQwenGgufVocabulary
    : public ::testing::TestWithParam<const cases::Case *> {};

TEST_P(CpuGraphQwenGgufVocabulary, HeadRunsEntireLogicalVocabulary) {
  auto context = g::cpu::create_context().value();
  graph_test::check_head(
      context, hyle::mem::system_resource(), kMatureProviders, *GetParam(),
      [&](const g::Graph &logical, const c::WorkloadContext &workload) {
        return g::compile(logical, context, {.workload = workload}).value();
      });
}

TEST_P(CpuGraphQwenGgufVocabulary,
       GatherMapsGlobalRowsAndRejectsOutOfBounds) {
  auto context = g::cpu::create_context().value();
  graph_test::check_gather(
      context, hyle::mem::system_resource(), kMatureProviders, *GetParam(),
      [&](const g::Graph &logical, const c::WorkloadContext &workload) {
        return g::compile(logical, context, {.workload = workload}).value();
      });
}

INSTANTIATE_TEST_SUITE_P(
    D5ExactQwenGguf, CpuGraphQwenGgufVocabulary,
    ::testing::Values(&cases::kCases[0], &cases::kCases[1], &cases::kCases[2],
                      &cases::kCases[3], &cases::kCases[4], &cases::kCases[5]));

INSTANTIATE_TEST_SUITE_P(
    D5ExactMoeVocabulary, CpuGraphQwenGgufVocabulary,
    ::testing::ValuesIn(d5_moe_profile_cases::kVocabularyPointers));
} // namespace
