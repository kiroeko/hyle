#pragma once

#define HYLE_TEST_VULKAN_CATALOGUE 1
#include "../compute_models/cpu_c5_parameter_fragments.h"
#include "../compute_models/cpu_c5_qwen_attention.h"
#include "../compute_models/cpu_c5_qwen_nextn.h"

namespace {
inline void run_q24_semantic_fragment(d5_qwen38_24t_semantic_fragments::Fragment fixture) {
  vulkan_catalogue_test::run(std::move(fixture));
}
#include "../compute_models/cpu_c5_parameter_tests.h"
} // namespace
#undef HYLE_TEST_VULKAN_CATALOGUE
