#pragma once

// Qwen-private cold target equations. All values and descriptions borrow the
// caller's original graph and storage facts; no runtime state owner is retained.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "execution_mode.h"
#include "hyle/compute/batch_layout.h"
#include "hyle/compute/tensor_desc.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/slots.h"

namespace qwen35 {
namespace c = hyle::compute;
namespace g = hyle::graph;
using V = g::ValueId;

struct StateSchema final {
  c::TensorDesc history, recurrent;
};
struct StatePorts final {
  V pool, source_ids, destination_ids;
  g::NodeId gather, store;
};
struct Ports final {
  V key, value;
  StatePorts history, recurrent;
};

struct Recipe;

struct TargetGraphInputs final {
  V token_ids, positions, attention_metadata, frequencies;
  V sequence_offsets, history_select; // Read only by packed target equations.
  std::size_t requests, prefix_rows;
  RowPacking packing;
  std::uint32_t state_slots;
  std::span<const StateSchema> recurrent_descriptions;
  const c::BatchSchema &plain_attention_schema;
  std::span<const c::BatchSchema> target_attention_schemas;
  std::size_t cpu_ffn_layers;
  g::NodeId preceding_append;
  bool has_preceding_append;
};
struct TargetGraphPorts final {
  std::span<Ports> layers;
  std::span<g::NodeId> attention_nodes;
  std::span<g::NodeId> append_nodes; // Only the target's original 2*N slots.
  std::span<V> append_cache_inputs;
  std::vector<g::slots::SourceBinding> &state_sources;
  std::span<V, 9> layer0_probes;
};
struct TargetGraphResult final {
  V hidden_rows;
  g::NodeId last_append;
  bool has_append;
};
TargetGraphResult emit_target_graph(Recipe &recipe,
    const TargetGraphInputs &inputs, TargetGraphPorts outputs);

} // namespace qwen35
