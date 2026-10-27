#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/batch_ops.h"
#include "hyle/graph/batch.h"
#include "model_geometry.h"

// Consumer-only graph construction for the identified one-block Qwen3.5 MTP
// head. R is the existing Recipe; no model, queue, invocation or payload owner
// is created here. All inputs and outputs are values in its original Graph.
namespace qwen35::mtp {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
using Value = graph::ValueId;
using Dimension = graph::Dimension;
using Type = compute::DataType;

constexpr Dimension fixed(std::int64_t value) noexcept {
  return Dimension::fixed(value);
}

// Main constructs these physical carrier dimensions once; packed_rows is
// requests * sequence_rows. Compact selectors/positions describe actual members.
struct Geometry final {
  std::size_t requests;
  Dimension sequence_rows, packed_rows;
};
struct Projection final {
  Dimension rows;
  Value fused;       // BF16 [rows,H], residual before the MTP block.
  Value normalized;  // Its BF16 input_layernorm boundary.
};
struct KeyValue final {
  Dimension rows;
  Value key_rows, value_rows; // BF16 [rows,Hkv,D]; key already has norm/RoPE.
};
struct AppendNodes final {
  graph::NodeId key, value;
};
// Explicit cold Graph ports for compact member-local previous-row selection.
// Empty ports retain the original rectangular exact-shape construction.
struct PrefillSelection final {
  Value hidden_indices{}, token_indices{};
};
struct PagedInputs final {
  compute::BatchSchema schema;
  Value key, value, metadata;
};
struct Head final {
  Value hidden_rows;      // Actual final-normalized BF16 [rows,H].
  Value selected_hidden;  // Material BF16 [requests,H], after final norm.
  Value logits;           // F32 [requests,V], after the BF16 LM head.
  AppendNodes append_nodes;
  graph::NodeId attention_node;
};

namespace detail {
template<class T> T require(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
inline void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

template<class R>
Value linear_absolute(R &r, Value x, const std::string &name) {
  return r.linear_absolute(x, name);
}

template<class R>
Value norm_absolute(R &r, Value x, const std::string &name) {
  return r.norm_absolute(x, name);
}

inline void geometry(const Geometry &shape) {
  if (!shape.requests || shape.requests > static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()) ||
      !shape.sequence_rows.valid() || !shape.packed_rows.valid() ||
      (shape.sequence_rows.is_fixed() && shape.sequence_rows.fixed_value() <= 0) ||
      (shape.packed_rows.is_fixed() && shape.packed_rows.fixed_value() <= 0) ||
      (shape.requests == 1 && shape.sequence_rows != shape.packed_rows))
    throw std::runtime_error("MTP requires shared positive request-major row dimensions");
  if (shape.sequence_rows.is_fixed() && shape.packed_rows.is_fixed()) {
    const auto requests = static_cast<std::int64_t>(shape.requests);
    if (shape.sequence_rows.fixed_value() > (std::numeric_limits<std::int64_t>::max)() / requests ||
        shape.sequence_rows.fixed_value() * requests != shape.packed_rows.fixed_value())
      throw std::runtime_error("MTP packed rows differ from requests times sequence rows");
  }
}

template<class R>
void coordinates(R &r, const Geometry &shape, Value positions, Value frequencies) {
  geometry(shape);
  const auto p = r.spec(positions), f = r.spec(frequencies);
  const auto batch = fixed(static_cast<std::int64_t>(shape.requests));
  const bool matching_positions = shape.requests == 1
      ? p.rank() == 1 && p.dimensions()[0] == shape.sequence_rows
      : p.rank() == 2 && p.dimensions()[0] == batch && p.dimensions()[1] == shape.sequence_rows;
  if (p.data_type() != Type::kInt32 || !matching_positions ||
      f.data_type() != Type::kFloat32 || f.rank() != 1 || f.dimensions()[0] != fixed(r.geometry.rotary / 2))
    throw std::runtime_error("MTP positions/frequencies differ from the original text geometry");
}

inline void paged_schema(const compute::BatchSchema &schema, const ModelGeometry &geometry, Type activation_type) {
  if (!schema.valid()) throw std::runtime_error("MTP requires its real paged schema");
  const auto capacity = schema.capacity();
  const auto plain = require(compute::CacheStorage::plain(activation_type));
  if (capacity.storage != compute::BatchStorage::kPaged ||
      capacity.query_type != activation_type || capacity.output_type != activation_type ||
      capacity.key_storage != plain || capacity.value_storage != plain ||
      capacity.query_heads != geometry.query_heads || capacity.kv_heads != geometry.kv_heads ||
      capacity.qk_head_dim != geometry.key_depth || capacity.value_head_dim != geometry.value_depth)
    throw std::runtime_error("MTP paged schema differs from the original source attention precision/geometry");
}
} // namespace detail

// hidden contains the preceding *target post-final-RMSNorm* row for each input
// token. Token at position p pairs with target h[p-1], without shifting RoPE p.
// Caller forms the initial zero row / catch-up shift in the same ordinary Graph.
template<class R>
Projection input_projection(R &r, Value tokens, Value hidden) {
  const auto ids = r.spec(tokens), state = r.spec(hidden);
  if (ids.data_type() != Type::kInt32 || ids.rank() != 1 ||
      state.data_type() != r.activation_type() || state.rank() != 2 ||
      ids.dimensions()[0] != state.dimensions()[0] || state.dimensions()[1] != fixed(r.geometry.hidden))
    throw std::runtime_error("MTP requires packed I32 tokens and matching BF16 target hidden rows");
  const auto rows = ids.dimensions()[0];
  const auto embedding = r.embedding(tokens, "model.language_model.embed_tokens.weight");
  const auto embedding_norm = detail::norm_absolute(r, embedding, "mtp.pre_fc_norm_embedding.weight");
  const auto hidden_norm = detail::norm_absolute(r, hidden, "mtp.pre_fc_norm_hidden.weight");
  // The trained projection's first half is embedding, second half is hidden.
  const auto concatenated = r.one(compute::Concat{.axis = 1}, {embedding_norm, hidden_norm},
      r.dense_spec(r.activation_type(), {rows, fixed(2 * r.geometry.hidden)}));
  const auto fused = detail::linear_absolute(r, concatenated, "mtp.fc.weight");
  return {rows, fused, detail::norm_absolute(r, fused, "mtp.layers.0.input_layernorm.weight")};
}

// Only this prefix is needed to catch up the one-layer MTP KV during prefill.
// The original positions and theta=1e7 frequency constant are caller inputs.
template<class R>
KeyValue project_kv(R &r, const Projection &projection, Value positions, Value frequencies,
                    const Geometry &shape) {
  detail::coordinates(r, shape, positions, frequencies);
  if (projection.rows != shape.packed_rows)
    throw std::runtime_error("MTP projection differs from its packed request rows");
  const auto batch = fixed(static_cast<std::int64_t>(shape.requests));
  auto key = r.reshape(detail::linear_absolute(r, projection.normalized,
      "mtp.layers.0.self_attn.k_proj.weight"), {batch, shape.sequence_rows, fixed(r.geometry.kv_heads), fixed(r.geometry.key_depth)});
  key = detail::norm_absolute(r, key, "mtp.layers.0.self_attn.k_norm.weight");
  const compute::RoPE rope{.rotary_dimensions = static_cast<std::size_t>(r.geometry.rotary), .sequence_axis = 1,
                          .layout = compute::RopeLayout::kHalfSplit};
  key = r.same(rope, {key, positions, frequencies}, key);
  const auto value = detail::linear_absolute(r, projection.normalized,
      "mtp.layers.0.self_attn.v_proj.weight");
  return {projection.rows, r.reshape(key, {projection.rows, fixed(r.geometry.kv_heads), fixed(r.geometry.key_depth)}),
      r.reshape(value, {projection.rows, fixed(r.geometry.kv_heads), fixed(r.geometry.value_depth)})};
}

// Multirow prompt maintenance in the original target Graph. Recipe defines
// physical token_rows=last_row+1; its finite family contains capacities>=2. T1
// uses the existing Single/CatchupSingle graph without an empty hidden slice.
// A lag1 draft first consumes the already committed catch-up token at L-1,
// paired with h[L-2]. Target input0 at L then pairs with the old carry h[L-1].
// The caller shares both request-major row geometries with positions. Derived
// dimensions are not interned, so do not recreate sums/products here.
// The caller supplies the actual draft positions; this helper never shifts them.
// Packed carriers use checked selectors to form each member's true previous
// rows. Inactive selector entries name a valid old carry/catch-up row. Original
// append layouts consume only Li (+ catch-up); physical R never advances them.
template<class R>
KeyValue prefill_kv(R &r, Value target_ids, Value normalized_target_rows,
                    Value carry, Value prior, Value catchup_token,
                    const Geometry &target_shape, const Geometry &draft_shape,
                    Value positions, Value frequencies, bool catchup, PrefillSelection selection = {}) {
  detail::geometry(target_shape);
  detail::coordinates(r, draft_shape, positions, frequencies);
  const auto batch = fixed(static_cast<std::int64_t>(target_shape.requests));
  const auto ids = r.spec(target_ids), target = r.spec(normalized_target_rows),
             old_carry = r.spec(carry);
  if (ids.data_type() != Type::kInt32 || ids.rank() != 1 ||
      target_shape.requests != draft_shape.requests || target_shape.sequence_rows != r.token_rows ||
      target_shape.packed_rows != r.packed_rows || ids.dimensions()[0] != target_shape.packed_rows ||
      target.data_type() != r.activation_type() || target.rank() != 2 ||
      target.dimensions()[0] != ids.dimensions()[0] || target.dimensions()[1] != fixed(r.geometry.hidden) ||
      old_carry.data_type() != r.activation_type() || old_carry.rank() != 2 ||
      old_carry.dimensions()[0] != batch || old_carry.dimensions()[1] != fixed(r.geometry.hidden) ||
      (!catchup && (draft_shape.sequence_rows != target_shape.sequence_rows ||
                    draft_shape.packed_rows != target_shape.packed_rows)) ||
      (target_shape.sequence_rows.is_fixed() && target_shape.sequence_rows.fixed_value() < 2))
    throw std::runtime_error("MTP prefill requires matching T>=2 target rows and one old carry row");
  if (catchup) {
    const auto old_prior = r.spec(prior), token = r.spec(catchup_token);
    if (old_prior.data_type() != r.activation_type() || old_prior.rank() != 2 ||
        old_prior.dimensions()[0] != batch || old_prior.dimensions()[1] != fixed(r.geometry.hidden) ||
        token.data_type() != Type::kInt32 || token.rank() != 1 || token.dimensions()[0] != batch)
      throw std::runtime_error("MTP catch-up prefill requires one prior hidden row and token");
  }
  if (selection.token_indices && !selection.hidden_indices)
    throw std::runtime_error("MTP token selection requires its matching hidden selection");
  Value tokens = target_ids, shifted;
  if (selection.hidden_indices) {
    const auto validate_indices = [&](Value value) {
      const auto description = r.spec(value);
      if (description.data_type() != Type::kInt32 || description.rank() != 1 ||
          description.dimensions()[0] != draft_shape.packed_rows)
        throw std::runtime_error("MTP prefill selection differs from its physical draft carrier");
    };
    validate_indices(selection.hidden_indices);
    if (catchup) {
      if (!selection.token_indices)
        throw std::runtime_error("MTP catch-up requires its compact token selection");
      validate_indices(selection.token_indices);
    } else if (selection.token_indices)
      throw std::runtime_error("MTP lag-zero tokens already have the compact target order");
    if (catchup && target_shape.requests > static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)() / 2))
      throw std::runtime_error("MTP hidden prefix row count overflows");
    const auto prefix = fixed(static_cast<std::int64_t>(target_shape.requests) * (catchup ? 2 : 1));
    const auto source_rows = Dimension(detail::require(r.builder.dimension_sum(prefix, target_shape.packed_rows)));
    const auto source_hidden = catchup
        ? r.one(compute::Concat{.axis = 0}, {carry, prior, normalized_target_rows},
            r.dense_spec(r.activation_type(), {source_rows, fixed(r.geometry.hidden)}))
        : r.one(compute::Concat{.axis = 0}, {carry, normalized_target_rows},
            r.dense_spec(r.activation_type(), {source_rows, fixed(r.geometry.hidden)}));
    shifted = r.one(compute::Gather{.axis = 0, .bounds = compute::IndexBoundsPolicy::kChecked},
        {source_hidden, selection.hidden_indices},
        r.dense_spec(r.activation_type(), {draft_shape.packed_rows, fixed(r.geometry.hidden)}));
    if (catchup) {
      const auto source_token_rows = Dimension(detail::require(r.builder.dimension_sum(batch, target_shape.packed_rows)));
      const auto source_tokens = r.one(compute::Concat{.axis = 0}, {catchup_token, target_ids},
          r.dense_spec(Type::kInt32, {source_token_rows}));
      tokens = r.one(compute::Gather{.axis = 0, .bounds = compute::IndexBoundsPolicy::kChecked},
          {source_tokens, selection.token_indices}, r.dense_spec(Type::kInt32, {draft_shape.packed_rows}));
    }
  } else if (target_shape.requests == 1) {
    const auto previous = r.slice(normalized_target_rows, 0, fixed(0), Dimension(r.last_row));
    shifted = catchup
        ? r.one(compute::Concat{.axis = 0}, {prior, carry, previous},
            r.dense_spec(r.activation_type(), {draft_shape.packed_rows, fixed(r.geometry.hidden)}))
        : r.one(compute::Concat{.axis = 0}, {carry, previous},
            r.dense_spec(r.activation_type(), {draft_shape.packed_rows, fixed(r.geometry.hidden)}));
    if (catchup) tokens = r.one(compute::Concat{.axis = 0}, {catchup_token, target_ids},
        r.dense_spec(Type::kInt32, {draft_shape.packed_rows}));
  } else {
    const auto rows = r.reshape(normalized_target_rows, {batch, target_shape.sequence_rows, fixed(r.geometry.hidden)});
    const auto previous = r.slice(rows, 1, fixed(0), Dimension(r.last_row));
    const auto old = r.reshape(carry, {batch, fixed(1), fixed(r.geometry.hidden)});
    auto grouped = catchup
        ? r.one(compute::Concat{.axis = 1},
            {r.reshape(prior, {batch, fixed(1), fixed(r.geometry.hidden)}), old, previous},
            r.dense_spec(r.activation_type(), {batch, draft_shape.sequence_rows, fixed(r.geometry.hidden)}))
        : r.one(compute::Concat{.axis = 1}, {old, previous},
            r.dense_spec(r.activation_type(), {batch, draft_shape.sequence_rows, fixed(r.geometry.hidden)}));
    shifted = r.reshape(grouped, {draft_shape.packed_rows, fixed(r.geometry.hidden)});
    if (catchup) {
      const auto ids_by_request = r.reshape(target_ids, {batch, target_shape.sequence_rows});
      const auto grouped_ids = r.one(compute::Concat{.axis = 1},
          {r.reshape(catchup_token, {batch, fixed(1)}), ids_by_request},
          r.dense_spec(Type::kInt32, {batch, draft_shape.sequence_rows}));
      tokens = r.reshape(grouped_ids, {draft_shape.packed_rows});
    }
  }
  return project_kv(r, input_projection(r, tokens, shifted), positions, frequencies, draft_shape);
}

// Creates only the original typed append nodes. The caller binds actual old
// frames and dirty pages through batch::prepare/bind_append, then take/ready.
template<class R>
AppendNodes append_kv(R &r, const compute::BatchSchema &schema, const KeyValue &kv) {
  detail::paged_schema(schema, r.geometry, r.activation_type());
  AppendNodes nodes;
  r.op(compute::Operation::custom(detail::require(graph::batch::paged_append(
      schema, compute::BatchCacheRole::kKey))), {kv.key_rows}, {}, &nodes.key);
  r.op(compute::Operation::custom(detail::require(graph::batch::paged_append(
      schema, compute::BatchCacheRole::kValue))), {kv.value_rows}, {}, &nodes.value);
  detail::require(r.builder.depends_on(nodes.value, nodes.key));
  return nodes;
}

// Run the trained block and shared LM head for one local row per request.
// Calling this consumes all 15 original MTP parameters in the same Recipe.
// It does not publish outputs, transfer leases, submit, sample on host or commit
// state. Root owns those original Model/StepState transaction responsibilities.
template<class R>
Head full_head(R &r, const Projection &projection, const KeyValue &kv,
               Value positions, Value frequencies, const PagedInputs &paged,
               Dimension last_row, const Geometry &shape) {
  detail::coordinates(r, shape, positions, frequencies);
  if (projection.rows != shape.packed_rows || kv.rows != projection.rows || (last_row.is_fixed() &&
      (last_row.fixed_value() < 0 || (shape.sequence_rows.is_fixed() &&
       last_row.fixed_value() >= shape.sequence_rows.fixed_value()))))
    throw std::runtime_error("MTP selected row is outside its actual input rows");
  const auto batch = fixed(static_cast<std::int64_t>(shape.requests));
  const auto nodes = append_kv(r, paged.schema, kv);
  auto projected = r.reshape(detail::linear_absolute(r, projection.normalized,
      "mtp.layers.0.self_attn.q_proj.weight"), {batch, shape.sequence_rows, fixed(r.geometry.query_heads), fixed(r.geometry.key_depth + r.geometry.value_depth)});
  auto query = detail::norm_absolute(r, r.slice(projected, 3, fixed(0), fixed(r.geometry.key_depth)),
      "mtp.layers.0.self_attn.q_norm.weight");
  const auto gate = r.reshape(r.copy(r.slice(projected, 3, fixed(r.geometry.key_depth), fixed(r.geometry.value_depth))),
      {projection.rows, fixed(r.geometry.attention_width())});
  const compute::RoPE rope{.rotary_dimensions = static_cast<std::size_t>(r.geometry.rotary), .sequence_axis = 1,
                          .layout = compute::RopeLayout::kHalfSplit};
  query = r.same(rope, {query, positions, frequencies}, query);
  graph::NodeId attention_node;
  const auto attention = r.one(compute::Operation::custom(compute::PagedAttention{
      .capacity = paged.schema.capacity(), .scale = 1.F / std::sqrt(static_cast<float>(r.geometry.key_depth)), .causal = true}),
      {r.reshape(query, {projection.rows, fixed(r.geometry.query_heads), fixed(r.geometry.key_depth)}),
       paged.key, paged.value, paged.metadata},
      r.dense_spec(r.activation_type(), {projection.rows, fixed(r.geometry.query_heads), fixed(r.geometry.value_depth)}), &attention_node);
  detail::require(r.builder.depends_on(attention_node, nodes.key));
  detail::require(r.builder.depends_on(attention_node, nodes.value));
  const auto gated = r.binary(compute::BinaryKind::kMultiply,
      r.reshape(attention, {projection.rows, fixed(r.geometry.attention_width())}),
      r.unary(compute::UnaryKind::kSigmoid, gate));
  auto x = r.binary(compute::BinaryKind::kAdd, projection.fused,
      detail::linear_absolute(r, gated, "mtp.layers.0.self_attn.o_proj.weight"));
  const auto ffn_input = detail::norm_absolute(r, x, "mtp.layers.0.post_attention_layernorm.weight");
  const auto ffn_gate = r.unary(compute::UnaryKind::kSilu,
      detail::linear_absolute(r, ffn_input, "mtp.layers.0.mlp.gate_proj.weight"));
  const auto up = detail::linear_absolute(r, ffn_input, "mtp.layers.0.mlp.up_proj.weight");
  const auto product = r.binary(compute::BinaryKind::kMultiply, ffn_gate, up);
  x = r.binary(compute::BinaryKind::kAdd, x,
      detail::linear_absolute(r, product, "mtp.layers.0.mlp.down_proj.weight"));
  const auto hidden_rows = detail::norm_absolute(r, x, "mtp.norm.weight");
  const auto selected = shape.requests == 1
      ? r.copy(r.slice(hidden_rows, 0, last_row, fixed(1)))
      : r.reshape(r.copy(r.slice(r.reshape(hidden_rows, {batch, shape.sequence_rows, fixed(r.geometry.hidden)}),
          1, last_row, fixed(1))), {batch, fixed(r.geometry.hidden)});
  const auto logits = r.scores(selected, std::string(r.geometry.head_role()));
  // Token selection belongs to the caller. Building a greedy TopK here would
  // leave a dead operation when the caller chooses categorical sampling.
  return {hidden_rows, selected, logits, nodes, attention_node};
}
} // namespace qwen35::mtp
