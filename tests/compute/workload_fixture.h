#pragma once

#include "hyle/compute/workload.h"

namespace workload_test {

// Caller-declared identity of the existing local inventory. These tests run
// small provider fragments, not the full model, and make no optimized-model
// qualification claim. Source: qwen35-2b-local-input.json (2026-09-17).
inline constexpr hyle::compute::ModelDescription kModel{
    "qwen3_5",
    "local-sha256:4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155",
    "config-sha256:ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"};

inline hyle::compute::WorkloadContext
known(hyle::compute::WorkloadPhase phase) {
  return {hyle::compute::ModelIdentity::create(kModel).value(), phase,
          {0x48594c4554455354ULL, 1, 1}};
}

} // namespace workload_test
