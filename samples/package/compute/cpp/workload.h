#pragma once

#include <utility>

#include <hyle/compute/workload.h>

namespace sample_workload {

// An application-declared example fragment, not a known-model optimization.
inline hyle::Result<hyle::compute::WorkloadContext> create() {
  HYLE_ASSIGN_OR_RETURN(
      auto profile,
      hyle::compute::ModelArtifactProfile::create(
          {.model_publisher = "example",
           .model_repository = "example/model",
           .model_family = "example-family",
           .model_name = "example-v1",
           .model_revision = "revision-v1",
           .configuration_digest = "0000000000000000000000000000000000000000000"
                                   "000000000000000000001",
           .artifact_repository = "example/model",
           .artifact_revision = "artifact-v1",
           .variant = "example-bf16",
           .manifest_digest = "000000000000000000000000000000000000000000000000"
                              "0000000000000002",
           .format_family = "example"}));
  return hyle::compute::WorkloadContext{
      std::move(profile), hyle::compute::WorkloadPhase::kDecode, {1, 1, 1}};
}

} // namespace sample_workload
