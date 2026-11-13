#pragma once

#include "model_core.h"
#include "prefix_cache.h"
#include "generation_ledger.h"
#include "../../inference/request_progress.h"
#include "../../inference/request_stop_policy.h"
#include "../../inference/logits_inputs.h"

namespace qwen35 {
struct GenerationRestoreStats final {
  std::size_t graph_calls = 0, target_rows = 0, draft_rows = 0;
  std::size_t submitted_draft_rows = 0;
  std::size_t normalized_lag_rows = 0;
  struct DecisionChange { std::size_t episode; std::int32_t historical, fresh; };
  std::vector<DecisionChange> decision_changes;
  double step_seconds = 0, seconds = 0;
};

enum class GenerationMode { kText, kMtp, kMixed };

struct GenerationRequest {
  std::string id, memory_name;
  std::span<const std::int32_t> prompt;
  PrefillInputPackets prefill_packets;
  std::filesystem::path trace;
  inference::RequestProgress progress;
  inference::RequestKey request_key{};
  std::shared_ptr<Session> session;
  cache::TokenPlanRef input;
  std::shared_ptr<const cache::HostImage> host_image;
  cache::Checkpoint completed_checkpoint;
  std::shared_ptr<cache::ReplayHistory> history;
  std::vector<MtpRoundResult> rounds;
  std::vector<std::vector<std::int32_t>> round_emitted;
  Clock::time_point prefill_start{}, prefill_endpoint{}, endpoint{};
  double prefill_step_seconds = 0, decode_step_sum_seconds = 0;
  std::size_t verified_rows = 0, draft_rows = 0, draft_rounds = 0, agreements = 0, accepts = 0;
  std::size_t visible_tokens = 0, draft_frontier = 0, draft_lag = 0;
  std::size_t reused_tokens = 0, prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0;
  std::size_t prefill_graph_calls = 0, prefill_graph_tokens = 0, prefill_draft_rows = 0;
  std::size_t prefix_sampling_calls = 0;
  std::size_t auxiliary_graph_calls = 0;
  double prefix_sampling_seconds = 0;
  // Immutable while this request owns its slot; replaced only after retirement.
  inference::RequestStopPolicy stop_policy;
  std::unique_ptr<GenerationLedger> ledger;
  struct TurnObservation {
    Clock::time_point start{}, prefill_endpoint{}, endpoint{}, arrival{};
    double prefill_shape_admission_seconds = 0, prefill_phase_prepare_seconds = 0, prefill_slot_initialize_seconds = 0;
    std::size_t prefill_phase_creations = 0, prefill_slot_initializations = 0;
    std::size_t output_begin = 0, output_end = 0;
    std::size_t context_before = 0, context_after = 0;
  };
  std::vector<TurnObservation> turn_observations;
  std::vector<std::shared_ptr<std::vector<std::int32_t>>> warm_inputs;
  std::vector<inference::GenerationTurn> live_turns;
  std::size_t last_idle = 0;
  std::size_t sampling_index;
  // A cold-prepared algorithm owns this request's full state and programs.
  // Null retains the original single-model/diagnostic fast path.
  Model *prepared_model = nullptr;
  cache::PrefixCache *prepared_prefix = nullptr;
  std::size_t execution_profile = 0;
  std::size_t max_draft_tokens = SIZE_MAX;
  std::uint64_t cache_sharing_domain = 0;
  LogitsInputs logits; // One [1,V] snapshot for every conditional row in an output episode.
  inference::LogitsInputProvider logits_provider;
  Clock::time_point admission_start{};
  GenerationRestoreStats restored;
  std::size_t restore_count = 0, idle_evictions = 0;
  // Retired Session submissions plus the live Session delta. Restore submits
  // have their own original call scope and are excluded from that live delta.
  std::size_t retired_draft_rows = 0, session_draft_rows_begin = 0;
  std::size_t actual_draft_rows() const noexcept {
    return retired_draft_rows + restored.submitted_draft_rows +
        (session ? session->submitted_draft_rows - session_draft_rows_begin : 0);
  }
  void retire_draft_rows() noexcept {
    if (session) retired_draft_rows += session->submitted_draft_rows - session_draft_rows_begin;
    session_draft_rows_begin = 0;
  }
  std::size_t checkpoint_resume_hits = 0, checkpoint_resume_misses = 0;
  std::int32_t fresh_next_id = -1;

  GenerationRequest(std::string name, std::span<const std::int32_t> ids,
                       std::filesystem::path output, std::size_t steps, std::size_t sampling = 0,
                       inference::RequestStopPolicy stop = {})
      : id(std::move(name)), memory_name(id.empty() ? "generation" : "generation_" + id),
        prompt(ids), trace(std::move(output)), progress(steps), stop_policy(std::move(stop)), sampling_index(sampling) {
    rounds.reserve(steps - 1);
  }
};

struct StartupPrefillWarmup final {
  std::size_t calls = 0, target_rows = 0, draft_rows = 0, physical_rows = 0;
  double seconds = 0;
  bool retired = false;
};

struct GenerationBatchObservation final {
  std::size_t tokens, active_rows, physical_rows; GraphRole role; bool generation;
  hyle::container::SmallVector<std::size_t, 2> members, input_rows;
};
} // namespace qwen35
