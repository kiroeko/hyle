#pragma once

// The original stable auxiliary programs. Each original Graph/literal/carrier,
// IO and (for draft) embedded runtime guard retains its old owner and order.
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "episode_state.h"
#include "mtp_graph.h"
#include "../../inference/slot_transactions.h"

namespace qwen35 {

struct DraftProgram final {
  std::size_t rows = 0, sampling_index = 0;
  bool head = false;
  g::ReplayMode replay_mode = g::ReplayMode::kCommands;
  std::vector<Tensor> literals;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  inference::ProgramCarrier<ColdResources> carrier;
  V tokens, positions, key, value, metadata, hidden, id, identity, log_probabilities, valid, probability;
  V bias, allowed, counts, next_counts;
  std::vector<V> hidden_inputs;
  std::vector<inference::ProgramIo::Input> hidden_bindings;
  inference::ProgramIo::Input bias_input, allowed_input, counts_input;
  StatePorts hidden_state;
  inference::SlotTransactions hidden_transactions;
  StepState guard; // The same original episode, cold-prepared for recursive reuse.
  mtp::AppendNodes append;
  g::NodeId attention;
  Tensor token_values, position_values, identity_values;
  inference::ProgramIo io;
  inference::ProgramIo::Readback id_value, valid_value, probability_value;
  c::LeasePool probability_pool, next_counts_pool;
  inference::ProgramIo::Output probabilities_output, next_counts_output;
};

struct PrefixSamplingKey final {
  std::size_t proposals, sampling_index;
  bool conditional = false;
  friend auto operator<=>(const PrefixSamplingKey &, const PrefixSamplingKey &) = default;
};

struct PrefixSamplingProgram final {
  std::size_t proposals = 0, sampling_index = 0;
  std::vector<Tensor> literals;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  inference::ProgramCarrier<ColdResources> carrier;
  V logits, identities, ids, bias, allowed, counts, continuation, packet;
  struct ConditionalInput final {
    V bias, allowed;
    inference::ProgramIo::Input bias_input, allowed_input;
  };
  std::vector<ConditionalInput> conditional_inputs;
  std::vector<V> q, accepted, final_ids, valid;
  Tensor identity_values, id_values, continuation_values;
  std::vector<float> continuation_host;
  std::vector<std::int32_t> packet_host;
  inference::ProgramIo io;
  inference::ProgramIo::Input logits_input, bias_input, allowed_input, counts_input;
  std::vector<inference::ProgramIo::Input> q_inputs;
  inference::ProgramIo::Readback packet_value;
};

struct SamplingFrontier final {
  std::vector<Tensor> literals;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  inference::ProgramCarrier<ColdResources> carrier;
  g::DimensionId last_row;
  V hidden_port, identity_port, id_port, valid_port, bias_port, allowed_port, counts_port;
  Tensor identity;
  inference::ProgramIo io;
  inference::ProgramIo::Input hidden_input, bias_input, allowed_input, counts_input;
  inference::ProgramIo::Readback id, valid;
};

struct PreparedSampling final {
  const std::size_t index;
  const SamplingConfiguration configuration;
  std::unique_ptr<SamplingFrontier> frontier;
  PreparedSampling(std::size_t ordinal, SamplingConfiguration value) : index(ordinal), configuration(value) {}
};

struct AuxiliaryCompileContext final {
  inference::Backend &execution;
  const c::ResourceBudget &allocation_budget;
  const c::ResourceBudget &execution_budget;
  const c::ResourceBudget &cache_budget;
};
struct DraftProgramRequest final {
  std::size_t rows = 0, sampling_index = 0;
  bool head = false;
  g::ReplayMode replay_mode = g::ReplayMode::kCommands;
  float minimum_probability = 0;
};
struct DraftProgramFacts final {
  const c::BatchSchema &attention_schema;
  std::uint32_t hidden_slots = 0;
  std::size_t active_sequences = 0, candidate_boundaries = 0;
  DraftProjectionView projection;
};
struct PrefixSamplingRequest final {
  std::size_t proposals = 0, sampling_index = 0;
  bool conditional = false;
};
template<class Program> struct AuxiliaryProgramResult final {
  std::unique_ptr<Program> program;
  ExecutionPlanFacts plan;
};

// Synchronous borrowed facts only. The factory creates the original unique_ptr
// before graph construction, never moves its embedded nonmovable inventory, and
// never initializes/submits a carrier or accesses Model/Session/runtime guard.
AuxiliaryProgramResult<DraftProgram> build_draft_program(
    const DraftProgramRequest &build_request, const FamilyModelFacts &model,
    const DraftProgramFacts &draft_facts, const SamplingConfiguration &configuration,
    const AuxiliaryCompileContext &context);
AuxiliaryProgramResult<PrefixSamplingProgram> build_prefix_sampling(
    const PrefixSamplingRequest &build_request, const FamilyModelFacts &model,
    const SamplingConfiguration &configuration, const AuxiliaryCompileContext &context);
AuxiliaryProgramResult<SamplingFrontier> build_sampling_frontier(
    const FamilyModelFacts &model, const SamplingConfiguration &configuration,
    const AuxiliaryCompileContext &context);

} // namespace qwen35
