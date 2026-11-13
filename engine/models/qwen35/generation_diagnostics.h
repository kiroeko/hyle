#pragma once

#include "generation_state.h"
#include "prefix_cache.h"
#include "decode_route_policy.h"
#include <iosfwd>

namespace hyle::inference { class RequestOutputSlots; }
namespace qwen35 {
struct GenerationContinuationReport final {
  bool unavailable;
  std::size_t successor_submissions, successor_publications, successor_discards, successor_refusals;
};
// Borrowed read-only observation after original work has really retired.
// It cannot escape, reenter execution or become a state/resource owner.
struct GenerationDiagnosticView final {
  const Model & model;
  std::span<const GenerationRequest> requests;
  const inference::RequestOutputSlots * output_deliveries;
  const cache::PrefixCache * prefix;
  std::span<const std::size_t> dispatch;
  std::span<const std::size_t> dispatch_turns;
  std::span<const std::size_t> evictions;
  std::span<const GenerationBatchObservation> batches;
  Clock::time_point startup;
  Clock::time_point ready;
  Clock::time_point start;
  Clock::time_point endpoint;
  std::size_t total_turns;
  std::size_t requested_outputs;
  std::size_t admitted;
  std::size_t completed;
  std::size_t peak_resident;
  std::size_t peak_committed;
  std::size_t actual_prefill_calls;
  std::size_t actual_decode_calls;
  std::size_t actual_fresh_participations;
  std::size_t actual_fresh_target_rows;
  std::size_t frontier_before;
  std::size_t draft_before;
  std::size_t prefix_before;
  std::size_t submitted_draft_begin;
  double prefill_seconds;
  double decode_seconds;
  double actual_step_sum;
  double frontier_seconds_before;
  double cold_prepare_seconds;
  bool mtp;
  bool reuse_ready;
  bool decode_overlap;
  bool prefix_enabled;
  bool first_turn_only;
  StartupPrefillWarmup startup_prefill_warmup;
  std::array<std::size_t, 3> packed_prefill_offset;
  GenerationContinuationReport continuation;
  const qwen35::RouteDecisionFacts & routes;
  const std::filesystem::path & trace;
};

struct GenerationDiagnostics final {
  std::filesystem::path trace;
  bool trace_states = false, prefix_enabled = false, warmup_planned_shapes = false, first_turn_only = false;
  Clock::time_point startup{}, constructed{};
  std::span<const cache::TokenPlanRef> token_plans;
  std::span<const inference::LogitsInputProvider> logits_providers;
  const qwen35::RouteDecisionFacts *routes = nullptr;
  void *context = nullptr;
  void (*completed)(void *, const GenerationDiagnosticView &) = nullptr;
};

void write_generation_report(std::ostream &, const GenerationDiagnosticView &);
} // namespace qwen35
