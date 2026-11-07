#pragma once

#include <cstddef>

namespace hyle::inference {

// Immutable request configuration shared by the cold policy stages. Actual
// session counts and draft depth are resolved after assets and schedules load.
struct ColdWorkloadFacts final {
  bool natural_generation = false;
  bool multiple_schedules = false;
  bool prefix_enabled = false;
  bool scheduler_progress = false;
  bool planned_shapes = false;
  bool cpu_assist = false;
  bool greedy = false;
  bool zero_draft_threshold = false;
  bool state_trace = false;
  std::size_t max_sessions = 0;
  std::size_t active_sequences = 0;
  std::size_t capacity = 0;
};

} // namespace hyle::inference
