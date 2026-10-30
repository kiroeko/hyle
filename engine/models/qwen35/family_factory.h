#pragma once

// Qwen-private cold graph/program inventory. Runtime Model/Session admission,
// mutable state, carriers and publication remain with the original caller.
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "graph_builder.h"
#include "target_graph.h"
#include "../../inference/prepared_program_family.h"
#include "../../inference/sampling_graph.h"

namespace qwen35 {
namespace inference = hyle::inference;
namespace sampling = hyle::inference::sampling;

struct SamplingConfiguration final {
  bool enabled = false;
  sampling::Policy policy;
  sampling::LogitsPolicy modifiers;
};
struct PreparedSampling; // Stable owner lives in the original ColdResources.

struct FamilyPorts {
  std::size_t prefix_rows = 1;
  V token_port, position_port, last_ids_port, metadata_port, logits_port, id_port, hidden_port;
  V sampling_identity_port, sampling_valid_port, sampling_accept_port;
  V sequence_offsets_port, history_select_port, prefix_ids_port;
  bool batched_text_sampling = false; // Cold [R,4] identity-port contract.
  std::vector<V> logits_bias_ports, logits_allowed_ports, logits_counts_ports; // Cold per-member inputs only.
  std::vector<g::NodeId> attention_nodes, append_nodes;
  std::vector<V> append_cache_inputs; // Cold same-cohort reader ports; invalid means write-only.
  std::vector<g::slots::SourceBinding> state_sources;
  V draft_key_port, draft_value_port, draft_metadata_port, draft_position_port;
  V carry_input_port, prior_input_port, catchup_token_port, draft_id_port;
  V successor_token_port, successor_catchup_port;
  V draft_hidden_select_port, draft_token_select_port, draft_last_ids_port;
  StatePorts hidden_state;
  g::NodeId draft_attention_node;
  std::vector<V> state_resource_ports;
  std::array<V, 9> probe_ports;
  std::vector<Ports> layers;
};
struct Family final : FamilyPorts {
  const PreparedSampling *sampling = nullptr;
  std::size_t requests = 1;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  bool admitted_exact = false;
  RowPacking packing = RowPacking::kRectangular;
  std::vector<Tensor> literals; // Backing survives every child and its slots.
  std::vector<g::TensorSpec> specs;
  c::BatchSchema attention_schema;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  g::DimensionId last_row;
  std::vector<std::size_t> chunks;
};
// An optional already prepared dense projection. Empty is the normal full-V
// path. The two Tensor children stay with their original cold owner; this view
// is borrowed synchronously and never becomes a second permission/owner.
struct DraftProjectionView final {
  StoredProjection projection;
  Tensor *original_ids = nullptr;
};

struct FamilyBuildRequest final {
  std::span<const std::size_t> chunks; // Original validated finite inventory.
  std::size_t requests = 1, prefix_rows = 1;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  RowPacking packing = RowPacking::kRectangular;
  bool diagnostic = false, admitted_exact = false;
};
struct FamilyModelFacts final {
  const ModelGeometry &geometry;
  PrecisionProfile precision;
  std::map<std::string, LoadedWeight> &weights;
  const ProjectionBindings &projections;
  const WeightMeanings &weight_meanings;
  std::size_t cpu_ffn_layers = 0;
};
struct FamilyPreparedFacts final {
  const c::BatchSchema &attention_schema;
  std::span<const c::BatchSchema> attention_schemas;
  std::span<const StateSchema> recurrent_descriptions;
  std::uint32_t state_slots = 0, hidden_slots = 0;
  bool retain_hidden = false, emit_successor = false;
  DraftProjectionView draft_projection;
};
struct FamilyCompileContext final {
  inference::Backend &execution;
  const c::ResourceBudget &allocation_budget;
  const c::ResourceBudget &execution_budget;
};
struct ExecutionPlanFacts final {
  std::array<std::size_t, 3> workspace_peak{}, workspace_limits{};
  std::size_t domain_count = 0;
};
struct FamilyBuildResult final {
  std::unique_ptr<Family> family; // Empty only for the original cold capability fallback.
  ExecutionPlanFacts plan;
};

// All borrowed inputs are synchronous. The complete original Graph and its
// literals/programs have stable addresses before any child is prepared.
FamilyBuildResult build_family(const FamilyBuildRequest &request,
    const FamilyModelFacts &model, const FamilyPreparedFacts &prepared,
    const SamplingConfiguration &configuration, const FamilyCompileContext &context);

// Shared cold checks/math also used by the original draft/frontier builders.
// Constant spans are inspected here and never retained in returned facts.
ExecutionPlanFacts inspect_execution_plan(const inference::Backend &execution,
    std::size_t cpu_ffn_layers, const g::CompileReport &report,
    std::span<const g::ConstantBinding> constants);
sampling::LogitsInputsFor<V> prepare_logits_inputs(Recipe &recipe, sampling::LogitsPolicy policy);
V draft_scores(Recipe &recipe, V selected_hidden, DraftProjectionView projection);
V original_draft_id(Recipe &recipe, V compact_id, DraftProjectionView projection);

} // namespace qwen35
