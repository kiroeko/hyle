#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hyle/compute/batch_execution.h"
#include "hyle/graph/executable.h"

namespace hyle::inference {

struct StepObservation {
  std::string name;
  std::size_t input_tokens = 0, visible = 0, capture_recorded = 0, capture_reused = 0;
  std::int32_t next = 0;
  double seconds = 0, submit_seconds = 0, step_seconds = 0, cold_batch_seconds = 0;
  // Synchronous admission of a previously unseen shape, included in step_seconds.
  // Zero for an already prepared family; unrelated to CUDA Graph recording.
  double shape_admission_seconds = 0;
  graph::ReplayRoute replay_route = graph::ReplayRoute::kUnprepared;
  bool scheduled_progress = false;
  bool prefix_hit = false, graph_executed = true;
  std::size_t reused_tokens = 0, graph_input_tokens = 0;
};
struct PreparationObservation {
  std::size_t slot = 0, phase_slot_prior_uses = 0;
  double layout_seconds = 0, batch_prepare_seconds = 0, replay_prepare_seconds = 0;
  double report_snapshot_seconds = 0;
  std::vector<graph::NodeId> nodes;
  std::vector<compute::BatchPlanReport> reports;
};

} // namespace hyle::inference
