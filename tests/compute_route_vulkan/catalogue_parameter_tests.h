#pragma once

#define HYLE_TEST_VULKAN_CATALOGUE 1
#include "../compute_models/cpu_c5_parameter_fragments.h"
#include "../compute_models/cpu_c5_qwen_attention.h"
#include "../compute_models/cpu_c5_qwen_nextn.h"

namespace {
inline void run_q24_semantic_fragment(d5_qwen38_24t_semantic_fragments::Fragment fixture) {
  if (fixture.region == 0x040c0501 &&
      fixture.description(fixture.inputs[0]).dimensions()[1] == 1) {
    // This route's two original F32 MatMul calls select F32 matvec at Qseq=1.
    // The common optional F16 coopmat2 reference is a different program.
    // Keep the stricter nominal F32 gate and all intermediate/state checks.
    for (auto &replay : fixture.replays) replay.native_outputs.clear();
  }
  vulkan_catalogue_test::run(std::move(fixture));
}
#include "../compute_models/cpu_c5_parameter_tests.h"
} // namespace
#undef HYLE_TEST_VULKAN_CATALOGUE
