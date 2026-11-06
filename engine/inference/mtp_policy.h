#pragma once

#include <cstddef>

namespace hyle::inference::mtp {

enum class RequestedMode { kOff, kAuto, kOn };

// Cold configuration for the current prepared default. This policy does not
// assert model or backend support; the selected source is validated separately.
struct DefaultRequest final {
  RequestedMode mode = RequestedMode::kAuto;
  bool natural_generation = false;
  bool multiple_schedules = false;
  bool prefix_enabled = false;
  bool scheduler_progress = false;
  bool warmup_planned_shapes = false;
  std::size_t max_sessions = 0;
  std::size_t active_sequences = 0;
  std::size_t capacity = 0;
  bool default_backend = false;
  bool cpu_ffn = false;
  bool greedy = false;
  bool decode_overlap = false;
  bool default_replay = false;
  std::size_t max_drafts = 0;
  bool automatic_head = false;
  bool automatic_vocabulary = false;
};

// Avoid inspecting an unused artifact, including errors from its filesystem.
constexpr bool inspect_prepared_vocabulary(const DefaultRequest &request) noexcept {
  return request.mode != RequestedMode::kOff && request.max_drafts > 0 &&
      request.default_backend && request.greedy && request.automatic_head &&
      request.automatic_vocabulary;
}

struct DefaultSelection final {
  bool automatic = false;
  bool enabled = false;
  bool prepared_assets = false;
};

constexpr DefaultSelection select_default(const DefaultRequest &request,
    bool prepared_vocabulary_exists) noexcept {
  const bool prepared = inspect_prepared_vocabulary(request) && prepared_vocabulary_exists;
  const bool automatic = prepared && request.natural_generation && !request.multiple_schedules &&
      !request.prefix_enabled && !request.scheduler_progress && !request.warmup_planned_shapes &&
      request.max_sessions <= 1 && request.active_sequences <= 1 && request.capacity >= 4 &&
      request.capacity <= 32768 && !request.cpu_ffn && request.decode_overlap &&
      request.default_replay && request.max_drafts <= 3;
  const bool enabled = request.mode == RequestedMode::kOn || automatic;
  return {automatic, enabled, enabled && prepared};
}

} // namespace hyle::inference::mtp
