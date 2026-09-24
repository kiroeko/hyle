#pragma once

#include "hyle/compute/workload.h"

namespace workload_test {

// Caller-declared identity of the existing local inventory. These tests run
// small provider fragments, not the full model, and make no optimized-model
// qualification claim. Source: qwen35-2b-local-input.json (2026-09-17).
inline constexpr hyle::compute::ModelArtifactProfileDescription kModel{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.5-2B",
    .model_family = "qwen3_5",
    .model_name = "qwen3.5-2b",
    .model_revision = "15852e8c16360a2fea060d615a32b45270f8a8fc",
    .configuration_digest =
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    .artifact_repository = "Qwen/Qwen3.5-2B",
    .artifact_revision = "15852e8c16360a2fea060d615a32b45270f8a8fc",
    .variant = "official-bf16",
    .manifest_digest =
        "4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155",
    .format_family = "safetensors"};

inline hyle::compute::WorkloadContext
known(hyle::compute::WorkloadPhase phase) {
  return {hyle::compute::ModelArtifactProfile::create(kModel).value(),
          phase,
          {0x48594c4554455354ULL, 1, 1}};
}

} // namespace workload_test
