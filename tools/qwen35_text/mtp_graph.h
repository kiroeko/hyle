#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/batch_ops.h"
#include "hyle/graph/batch.h"

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

// Main constructs these shared dimensions once. Rows are request-major;
// packed_rows == requests * sequence_rows in every selected finite child.
struct Geometry final {
  std::size_t requests;
  Dimension sequence_rows, packed_rows;
};
struct Projection final {
  Dimension rows;
  Value fused;       // BF16 [rows,2048], residual before the MTP block.
  Value normalized;  // Its BF16 input_layernorm boundary.
};
struct KeyValue final {
  Dimension rows;
  Value key_rows, value_rows; // BF16 [rows,2,256]; key already has norm/RoPE.
};
struct AppendNodes final {
  graph::NodeId key, value;
};
struct PagedInputs final {
  compute::BatchSchema schema;
  Value key, value, metadata;
};
struct Head final {
  Value hidden_rows;      // Actual final-normalized BF16 [rows,2048].
  Value selected_hidden;  // Material BF16 [requests,2048], after final norm.
  Value logits;           // F32 [requests,248320], after the BF16 LM head.
  Value ids;              // I32 [requests,1], one proposal per request.
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
  const auto original = r.weight_absolute(name);
  const auto source = r.spec(x), weight = r.spec(original);
  if (!source.rank() || weight.rank() != 2 ||
      source.data_type() != weight.data_type() ||
      source.dimensions().back() != weight.dimensions()[1])
    throw std::runtime_error("MTP linear parameter shape/type differs: " + name);
  constexpr std::array<std::int32_t, 2> axes{1, 0};
  const auto transposed = r.one(require(compute::View::permute(axes)), {original});
  auto dimensions = std::vector<Dimension>(source.dimensions().begin(), source.dimensions().end());
  dimensions.back() = weight.dimensions()[0];
  return r.one(compute::MatMul{}, {x, transposed}, r.dense_spec(source.data_type(), dimensions));
}

template<class R>
Value norm_absolute(R &r, Value x, const std::string &name) {
  const auto original = r.weight_absolute(name);
  const auto source = r.spec(x), weight = r.spec(original);
  if (!source.rank() || weight.rank() != 1 ||
      source.dimensions().back() != weight.dimensions()[0] ||
      weight.data_type() != Type::kBFloat16)
    throw std::runtime_error("MTP raw delta norm shape/type differs: " + name);
  // These are original HF delta parameters, not converted GGUF norm weights.
  // Match Recipe::norm: F32(raw+1), F32 RMS, then cast to the input type.
  const auto scale = r.binary(compute::BinaryKind::kAdd,
      r.cast(original, Type::kFloat32), r.scalar(1));
  const auto converted = r.cast(x, Type::kFloat32);
  return r.cast(r.same(compute::RmsNorm{.epsilon = 1.e-6F},
      {converted, scale}, converted), source.data_type());
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
      f.data_type() != Type::kFloat32 || f.rank() != 1 || f.dimensions()[0] != fixed(32))
    throw std::runtime_error("MTP positions/frequencies differ from the original text geometry");
}

inline void paged_schema(const compute::BatchSchema &schema) {
  if (!schema.valid()) throw std::runtime_error("MTP requires its real paged schema");
  const auto capacity = schema.capacity();
  const auto plain = require(compute::CacheStorage::plain(Type::kBFloat16));
  if (capacity.storage != compute::BatchStorage::kPaged ||
      capacity.query_type != Type::kBFloat16 || capacity.output_type != Type::kBFloat16 ||
      capacity.key_storage != plain || capacity.value_storage != plain ||
      capacity.query_heads != 8 || capacity.kv_heads != 2 ||
      capacity.qk_head_dim != 256 || capacity.value_head_dim != 256)
    throw std::runtime_error("MTP paged schema differs from BF16 Hq8/Hkv2/D256");
}
} // namespace detail

// hidden contains the preceding *target post-final-RMSNorm* row for each input
// token. Token at position p pairs with target h[p-1], without shifting RoPE p.
// Caller forms the initial zero row / catch-up shift in the same ordinary Graph.
template<class R>
Projection input_projection(R &r, Value tokens, Value hidden) {
  const auto ids = r.spec(tokens), state = r.spec(hidden);
  if (ids.data_type() != Type::kInt32 || ids.rank() != 1 ||
      state.data_type() != Type::kBFloat16 || state.rank() != 2 ||
      ids.dimensions()[0] != state.dimensions()[0] || state.dimensions()[1] != fixed(2048))
    throw std::runtime_error("MTP requires packed I32 tokens and matching BF16 target hidden rows");
  const auto rows = ids.dimensions()[0];
  const auto embedding = r.one(compute::Gather{.bounds = compute::IndexBoundsPolicy::kUnchecked},
      {r.weight_absolute("model.language_model.embed_tokens.weight"), tokens},
      r.dense_spec(Type::kBFloat16, {rows, fixed(2048)}));
  const auto embedding_norm = detail::norm_absolute(r, embedding, "mtp.pre_fc_norm_embedding.weight");
  const auto hidden_norm = detail::norm_absolute(r, hidden, "mtp.pre_fc_norm_hidden.weight");
  // The trained projection's first half is embedding, second half is hidden.
  const auto concatenated = r.one(compute::Concat{.axis = 1}, {embedding_norm, hidden_norm},
      r.dense_spec(Type::kBFloat16, {rows, fixed(4096)}));
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
      "mtp.layers.0.self_attn.k_proj.weight"), {batch, shape.sequence_rows, fixed(2), fixed(256)});
  key = detail::norm_absolute(r, key, "mtp.layers.0.self_attn.k_norm.weight");
  const compute::RoPE rope{.rotary_dimensions = 64, .sequence_axis = 1,
                          .layout = compute::RopeLayout::kHalfSplit};
  key = r.same(rope, {key, positions, frequencies}, key);
  const auto value = detail::linear_absolute(r, projection.normalized,
      "mtp.layers.0.self_attn.v_proj.weight");
  return {projection.rows, r.reshape(key, {projection.rows, fixed(2), fixed(256)}),
      r.reshape(value, {projection.rows, fixed(2), fixed(256)})};
}

// Multirow prompt maintenance in the original target Graph. Recipe defines
// token_rows=last_row+1; its finite family must contain only T>=2 here. T1
// uses the existing Single/CatchupSingle graph without an empty hidden slice.
// A lag1 draft first consumes the already committed catch-up token at L-1,
// paired with h[L-2]. Target input0 at L then pairs with the old carry h[L-1].
// The caller shares both request-major row geometries with positions. Derived
// dimensions are not interned, so do not recreate sums/products here.
// The caller supplies the actual draft positions; this helper never shifts them.
template<class R>
KeyValue prefill_kv(R &r, Value target_ids, Value normalized_target_rows,
                    Value carry, Value prior, Value catchup_token,
                    const Geometry &target_shape, const Geometry &draft_shape,
                    Value positions, Value frequencies, bool catchup) {
  detail::geometry(target_shape);
  detail::coordinates(r, draft_shape, positions, frequencies);
  const auto batch = fixed(static_cast<std::int64_t>(target_shape.requests));
  const auto ids = r.spec(target_ids), target = r.spec(normalized_target_rows),
             old_carry = r.spec(carry);
  if (ids.data_type() != Type::kInt32 || ids.rank() != 1 ||
      target_shape.requests != draft_shape.requests || target_shape.sequence_rows != r.token_rows ||
      target_shape.packed_rows != r.packed_rows || ids.dimensions()[0] != target_shape.packed_rows ||
      target.data_type() != Type::kBFloat16 || target.rank() != 2 ||
      target.dimensions()[0] != ids.dimensions()[0] || target.dimensions()[1] != fixed(2048) ||
      old_carry.data_type() != Type::kBFloat16 || old_carry.rank() != 2 ||
      old_carry.dimensions()[0] != batch || old_carry.dimensions()[1] != fixed(2048) ||
      (!catchup && (draft_shape.sequence_rows != target_shape.sequence_rows ||
                    draft_shape.packed_rows != target_shape.packed_rows)) ||
      (target_shape.sequence_rows.is_fixed() && target_shape.sequence_rows.fixed_value() < 2))
    throw std::runtime_error("MTP prefill requires matching T>=2 target rows and one old carry row");
  if (catchup) {
    const auto old_prior = r.spec(prior), token = r.spec(catchup_token);
    if (old_prior.data_type() != Type::kBFloat16 || old_prior.rank() != 2 ||
        old_prior.dimensions()[0] != batch || old_prior.dimensions()[1] != fixed(2048) ||
        token.data_type() != Type::kInt32 || token.rank() != 1 || token.dimensions()[0] != batch)
      throw std::runtime_error("MTP catch-up prefill requires one prior hidden row and token");
  }
  Value tokens = target_ids, shifted;
  if (target_shape.requests == 1) {
    const auto previous = r.slice(normalized_target_rows, 0, fixed(0), Dimension(r.last_row));
    shifted = catchup
        ? r.one(compute::Concat{.axis = 0}, {prior, carry, previous},
            r.dense_spec(Type::kBFloat16, {draft_shape.packed_rows, fixed(2048)}))
        : r.one(compute::Concat{.axis = 0}, {carry, previous},
            r.dense_spec(Type::kBFloat16, {draft_shape.packed_rows, fixed(2048)}));
    if (catchup) tokens = r.one(compute::Concat{.axis = 0}, {catchup_token, target_ids},
        r.dense_spec(Type::kInt32, {draft_shape.packed_rows}));
  } else {
    const auto rows = r.reshape(normalized_target_rows, {batch, target_shape.sequence_rows, fixed(2048)});
    const auto previous = r.slice(rows, 1, fixed(0), Dimension(r.last_row));
    const auto old = r.reshape(carry, {batch, fixed(1), fixed(2048)});
    auto grouped = catchup
        ? r.one(compute::Concat{.axis = 1},
            {r.reshape(prior, {batch, fixed(1), fixed(2048)}), old, previous},
            r.dense_spec(Type::kBFloat16, {batch, draft_shape.sequence_rows, fixed(2048)}))
        : r.one(compute::Concat{.axis = 1}, {old, previous},
            r.dense_spec(Type::kBFloat16, {batch, draft_shape.sequence_rows, fixed(2048)}));
    shifted = r.reshape(grouped, {draft_shape.packed_rows, fixed(2048)});
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
  detail::paged_schema(schema);
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
      "mtp.layers.0.self_attn.q_proj.weight"), {batch, shape.sequence_rows, fixed(8), fixed(512)});
  auto query = detail::norm_absolute(r, r.slice(projected, 3, fixed(0), fixed(256)),
      "mtp.layers.0.self_attn.q_norm.weight");
  const auto gate = r.reshape(r.copy(r.slice(projected, 3, fixed(256), fixed(256))),
      {projection.rows, fixed(2048)});
  const compute::RoPE rope{.rotary_dimensions = 64, .sequence_axis = 1,
                          .layout = compute::RopeLayout::kHalfSplit};
  query = r.same(rope, {query, positions, frequencies}, query);
  graph::NodeId attention_node;
  const auto attention = r.one(compute::Operation::custom(compute::PagedAttention{
      .capacity = paged.schema.capacity(), .scale = 0.0625F, .causal = true}),
      {r.reshape(query, {projection.rows, fixed(8), fixed(256)}),
       paged.key, paged.value, paged.metadata},
      r.dense_spec(Type::kBFloat16, {projection.rows, fixed(8), fixed(256)}), &attention_node);
  detail::require(r.builder.depends_on(attention_node, nodes.key));
  detail::require(r.builder.depends_on(attention_node, nodes.value));
  const auto gated = r.binary(compute::BinaryKind::kMultiply,
      r.reshape(attention, {projection.rows, fixed(2048)}),
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
      : r.reshape(r.copy(r.slice(r.reshape(hidden_rows, {batch, shape.sequence_rows, fixed(2048)}),
          1, last_row, fixed(1))), {batch, fixed(2048)});
  const auto logits = r.scores(selected, "model.language_model.embed_tokens.weight");
  const auto ids = r.op(compute::Operation(compute::TopK{.count = 1}), {logits},
      {r.dense_spec(Type::kFloat32, {batch, fixed(1)}),
       r.dense_spec(Type::kInt32, {batch, fixed(1)})})[1];
  return {hidden_rows, selected, logits, ids, nodes, attention_node};
}
} // namespace qwen35::mtp
