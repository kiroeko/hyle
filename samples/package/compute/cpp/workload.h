#pragma once

#include <utility>

#include <hyle/compute/workload.h>

namespace sample_workload {

// An application-declared example fragment, not a known-model optimization.
inline hyle::Result<hyle::compute::WorkloadContext> create() {
  HYLE_ASSIGN_OR_RETURN(
      auto model, hyle::compute::ModelIdentity::create(
                      {"example-family", "example-v1", "example-config-v1"}));
  return hyle::compute::WorkloadContext{
      std::move(model), hyle::compute::WorkloadPhase::kDecode, {1, 1, 1}};
}

} // namespace sample_workload
