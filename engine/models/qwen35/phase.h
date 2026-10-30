#pragma once

// Qwen's original mutable physical carriers and their one cold preparation.
// ColdResources is the existing retirement scope, not a new owner here.
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "family_factory.h"
#include "../../inference/program_slots.h"
#include "hyle/graph/batch.h"

namespace qwen35 {
struct ColdResources;
using inference::PreparedOutputFacts;

// Physical IO/control carriers are independent of candidate depth. A reader
// may still retain the preceding carrier after its producer publishes.
constexpr std::size_t execution_slot_count = 3;
enum class StateBankKind { kHistory, kRecurrent, kHidden };
struct StateBank final {
  std::size_t layer = 0;
  StateBankKind kind = StateBankKind::kHistory;
  std::size_t source_index = SIZE_MAX; // Only a real paired Gather has a Source.
};
// Mutable input, result and layout carriers belong to one actual Invocation.
// Sharing a finite child/Queue never grants another slot access to these bytes.
struct PhaseIo final {
  c::BatchLayoutVersion attention_layout;
  // Original per-Invocation planning storage. Nodes are fixed at cold
  // preparation; layout aliases exist only during batch::prepare below.
  std::vector<g::batch::NodeLayout> batch_layouts;
  Tensor tokens, positions, last_ids, logits, final_hidden, history_select, prefix_ids;
  std::vector<std::int32_t> history_select_values, last_id_values, prefix_id_values;
  Tensor sampling_identities, draft_positions, catchup_token;
  Tensor draft_hidden_select, draft_token_select, draft_last_ids;
  std::vector<std::int32_t> draft_hidden_indices, draft_token_indices;
  inference::ProgramIo::Readback next_id, draft_id, sampling_valid, sampling_accept;
  inference::ProgramIo::Readback successor_token, successor_catchup;
  std::vector<std::uint8_t> sampling_valid_values, sampling_accept_values;
  std::vector<float> sampling_scratch;
  std::vector<std::int32_t> draft_ids, prefix_candidates;
  std::array<Tensor, 9> probe_outputs;
};
struct Phase final : FamilyPorts {
  std::size_t query_rows = 0;
  c::BatchSchema attention_schema;
  // The selected child is shared immutable model data, not copied per session.
  const inference::PreparedProgram *child = nullptr;
  const Family *family = nullptr;
  bool imported_sources = false;
  std::vector<StateBank> state_banks;
  using ExecutionSlots = inference::ProgramSlots<ColdResources, PhaseIo, execution_slot_count>;
  ExecutionSlots slots;
  const g::Graph &state_graph() const noexcept {
    return child->resource_graph();
  }
  const PreparedOutputFacts *logits_facts = nullptr, *hidden_facts = nullptr;
  std::array<const PreparedOutputFacts *, 9> probe_facts{};
};

struct PhaseSelection final {
  const Family &family;
  const inference::PreparedProgram &child;
  const SamplingConfiguration &sampling;
};
struct PhasePrepareFacts final {
  std::size_t tokens = 0, physical_slots = 1;
  bool diagnostic = false, imported_sources = false;
  std::array<std::size_t, execution_slot_count> planned_uses{};
  std::array<g::ReplayMode, execution_slot_count> replay_modes{};
};
struct PhaseEnvironment final {
  const inference::Backend &execution;
  const ModelGeometry &geometry;
  std::span<const StateSchema> recurrent_descriptions;
  const c::ResourceBudget &allocation_budget;
  // Original options for later SlotBindings::create; ordinary vector metadata
  // inside SlotTransactions::prepare is not newly charged by this handle.
  const c::ResourceBudget &cache_budget;
};

// Synchronously create the original stable IO/slot group; do not initialize an
// Invocation. Model keeps policy, state-layout attachment, scratch and publish.
std::unique_ptr<Phase> prepare_phase(const PhaseSelection &selection,
    const PhasePrepareFacts &preparation, const PhaseEnvironment &environment);

} // namespace qwen35
