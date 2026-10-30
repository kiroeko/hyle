// Cold Qwen target embedding, recurrent/attention blocks and FFN equations.
// Preparation, compilation, state ownership and publication stay with the caller.
#include "target_graph.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph_builder.h"
#include "block_graph.h"
#include "cache_plan.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/sequence_ops.h"
#include "hyle/graph/batch.h"

namespace qwen35 {
TargetGraphResult emit_target_graph(Recipe &r,
    const TargetGraphInputs &inputs, TargetGraphPorts outputs) {
  const auto &geometry = r.geometry;
  auto &assembly = r.assembly;
  const auto t = r.token_rows, packed = r.packed_rows;
  const auto requests = inputs.requests, prefix_rows = inputs.prefix_rows;
  const auto batch = f(static_cast<std::int64_t>(requests));
  const auto state_schema = inputs.recurrent_descriptions;
  const bool packed_verify = packs_verify(inputs.packing), packed_target = packs_target(inputs.packing);
  const auto frequencies = inputs.frequencies;
  const auto target_tokens = inputs.token_ids;
  const auto attention_layers = geometry.attention_layers();
  if (state_schema.size() != geometry.layers.size() || outputs.layers.size() != state_schema.size() ||
      outputs.attention_nodes.size() != attention_layers || outputs.append_nodes.size() != 2 * attention_layers ||
      outputs.append_cache_inputs.size() != 2 * attention_layers ||
      (!inputs.target_attention_schemas.empty() && inputs.target_attention_schemas.size() != attention_layers))
    throw std::runtime_error("target graph cold ports do not match the source geometry");
  auto x = r.embedding(target_tokens, "model.language_model.embed_tokens.weight");
  g::NodeId prior_append = inputs.preceding_append;
  bool has_prior_append = inputs.has_preceding_append;
  for (std::size_t i = 0; i < state_schema.size(); ++i) {
    const auto prefix = "layers." + std::to_string(i) + ".";
    auto normalized = r.norm(x, prefix + "input_layernorm.weight");
    auto &ports = outputs.layers[i];
    V mixed;
    if (geometry.is_gdn(i)) {
      const auto state_port = [&](StatePorts &port, const c::TensorDesc &item) {
        const auto bytes = take(item.required_bytes());
        const auto schema = take(g::slots::Schema::create(item, inputs.state_slots,
            cache::add(bytes, cache::state_pool_padding(bytes))));
        port.pool = r.input(take(schema.capacity_description()));
        port.source_ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        port.destination_ids = r.input(dense(Type::kInt32,
            {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
        auto rows = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
            {port.pool, port.source_ids}, r.dense_spec(item.data_type(),
                {batch, f(schema.item_elements())}), &port.gather);
        return std::pair{schema, rows};
      };
      const auto [history_schema, history_rows] = state_port(ports.history, state_schema[i].history);
      const auto [recurrent_schema, recurrent_rows] = state_port(ports.recurrent, state_schema[i].recurrent);
      const auto history = r.reshape(history_rows,
          {batch, f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())});
      const auto state = r.reshape(recurrent_rows,
          {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)});
      const auto store_state = [&](StatePorts &port, const g::slots::Schema &schema, V rows) {
        r.op(c::Operation::custom(g::slots::Store{schema}), {rows, port.destination_ids}, {}, &port.store);
        if (has_prior_append) check(r.builder.depends_on(port.store, prior_append));
        prior_append = port.store; has_prior_append = true;
        outputs.state_sources.push_back({port.store, port.gather});
      };
      if (packed_target) {
        const auto channels = f(geometry.gdn_qkv_width());
        const auto qkv = r.linear(normalized, prefix + "linear_attn.in_proj_qkv.weight");
        const auto history_count = f(static_cast<std::int64_t>(cache::multiply(requests, geometry.conv_kernel - 1)));
        const auto candidate_count = f(static_cast<std::int64_t>(cache::multiply(requests, prefix_rows)));
        const auto history_outputs = f(static_cast<std::int64_t>(cache::multiply(
            cache::multiply(requests, prefix_rows), geometry.conv_kernel - 1)));
        const auto sample_count = D(take(r.builder.dimension_sum(history_count, packed)));
        const auto raw_samples = r.one(c::Concat{.axis = 0},
            {r.reshape(history, {history_count, channels}), qkv},
            r.dense_spec(r.activation_type(), {sample_count, channels}));
        // Preserve the original activation bytes in persistent history. In
        // particular, F32 convolution must not round-trip NaN payloads through
        // a different storage type just to recover the true history tail.
        const auto next_history = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
            {raw_samples, inputs.history_select},
            r.dense_spec(r.activation_type(), {history_outputs, channels}));
        store_state(ports.history, history_schema,
            r.reshape(next_history, {candidate_count, f(history_schema.item_elements())}));
        const auto convolution = r.op(c::sequence_ops::packed_causal_conv1d(),
            {r.cast(qkv, Type::kFloat32),
             r.cast(r.reshape(r.weight(prefix + "linear_attn.conv1d.weight"),
                 {channels, f(geometry.conv_kernel)}), Type::kFloat32),
             r.cast(history, Type::kFloat32), inputs.sequence_offsets},
            {r.dense_spec(Type::kFloat32, {packed, channels}),
             r.dense_spec(Type::kFloat32, {batch, f(geometry.conv_kernel - 1), channels})});
        const auto activated = r.reshape(r.unary(c::UnaryKind::kSilu,
            r.cast(convolution[0], r.activation_type())), {batch, t, channels});
        auto q = r.reshape(r.slice(activated, 2, f(0), f(geometry.gdn_key_width())),
            {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto k = r.reshape(r.slice(activated, 2, f(geometry.gdn_key_width()), f(geometry.gdn_key_width())),
            {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto v = r.reshape(r.slice(activated, 2, f(2 * geometry.gdn_key_width()), f(geometry.gdn_value_width())),
            {packed, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
        v = r.activation_type() == Type::kFloat32 ? r.copy(v) : r.cast(v, Type::kFloat32);
        q = r.gdn_key_rows(r.l2(q)); k = r.gdn_key_rows(r.l2(k));
        const auto a = r.reshape(r.cast(r.linear(normalized, prefix + "linear_attn.in_proj_a.weight"), Type::kFloat32),
            {packed, f(geometry.gdn_value_heads), f(1)});
        const auto b = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_b.weight"),
            {packed, f(geometry.gdn_value_heads), f(1)});
        const auto dt = r.reshape(r.cast(r.weight(prefix + "linear_attn.dt_bias"), Type::kFloat32),
            {f(1), f(geometry.gdn_value_heads), f(1)});
        const auto decay = r.reshape(r.decay("model.language_model." + prefix + "linear_attn.A_log"),
            {f(1), f(geometry.gdn_value_heads), f(1)});
        const auto g = r.binary(c::BinaryKind::kMultiply, decay,
            r.unary(c::UnaryKind::kSoftplus, r.binary(c::BinaryKind::kAdd, a, dt)));
        const auto beta = r.cast(r.unary(c::UnaryKind::kSigmoid, b), Type::kFloat32);
        const auto recurrent = r.op(packed_verify ? c::sequence_ops::packed_gated_delta_net_prefixes(
            static_cast<std::uint32_t>(prefix_rows)) : c::sequence_ops::packed_gated_delta_net(),
            {r.cast(r.reshape(q, {packed, r.spec(q).dimensions()[2], f(geometry.gdn_key_depth)}), Type::kFloat32),
             r.cast(r.reshape(k, {packed, r.spec(k).dimensions()[2], f(geometry.gdn_key_depth)}), Type::kFloat32),
             r.cast(v, Type::kFloat32), g, beta,
             r.reshape(recurrent_rows, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)}),
             inputs.sequence_offsets},
            {r.dense_spec(Type::kFloat32, {packed, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}),
              packed_verify
                  ? r.dense_spec(Type::kFloat32, {batch, f(static_cast<std::int64_t>(prefix_rows)), f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})
                  : r.dense_spec(Type::kFloat32, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})});
        store_state(ports.recurrent, recurrent_schema,
            r.reshape(recurrent[1], {candidate_count, f(recurrent_schema.item_elements())}));
        mixed = block_graph::emit_gdn_epilogue(r, recurrent[0], normalized,
            "model.language_model." + prefix,
            {packed, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}, packed,
            i == 0 ? std::span<V>(outputs.layer0_probes) : std::span<V>{});
      } else {
        auto qkv = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_qkv.weight"), {batch, t, f(geometry.gdn_qkv_width())});
        auto samples = r.one(c::Concat{.axis = 1}, {history, qkv},
            r.dense_spec(r.activation_type(), {batch, r.sample_rows, f(geometry.gdn_qkv_width())}));
        // K1 records the final boundary; current K>1 families have T==K.
        // The ordinary graph stores raw history before Cast F32. A single
        // window may have a batch gap and therefore needs real materialization.
        std::vector<V> windows;
        windows.reserve(prefix_rows);
        for (std::size_t row = 0; row < prefix_rows; ++row) {
          const auto boundary = prefix_rows == 1 ? t : f(static_cast<std::int64_t>(row + 1));
          auto window = r.slice(samples, 1, boundary, f(geometry.conv_kernel - 1));
          if (prefix_rows == 1) window = r.one(c::Copy{}, {window},
              r.dense_spec(r.activation_type(),
                  {batch, f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())}));
          else window = r.reshape(window,
              {batch, f(1), f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())});
          windows.push_back(window);
        }
        const auto histories = prefix_rows == 1 ? windows[0] : r.concat(windows, 1,
            r.dense_spec(r.activation_type(), {batch, f(static_cast<std::int64_t>(prefix_rows)),
                f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())}));
        store_state(ports.history, history_schema, r.reshape(histories,
            {f(static_cast<std::int64_t>(requests * prefix_rows)), f(history_schema.item_elements())}));
        auto convolution = r.one(c::model_ops::causal_conv1d(), {r.cast(samples, Type::kFloat32),
            r.cast(r.reshape(r.weight(prefix + "linear_attn.conv1d.weight"), {f(geometry.gdn_qkv_width()), f(geometry.conv_kernel)}), Type::kFloat32)},
            r.dense_spec(Type::kFloat32, {batch, t, f(geometry.gdn_qkv_width())}));
        auto activated = r.unary(c::UnaryKind::kSilu, r.cast(convolution, r.activation_type()));
        auto q = r.reshape(r.slice(activated, 2, f(0), f(geometry.gdn_key_width())), {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto k = r.reshape(r.slice(activated, 2, f(geometry.gdn_key_width()), f(geometry.gdn_key_width())), {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto v = r.reshape(r.slice(activated, 2, f(2 * geometry.gdn_key_width()), f(geometry.gdn_value_width())), {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
        auto a = r.reshape(r.cast(r.linear(normalized, prefix + "linear_attn.in_proj_a.weight"), Type::kFloat32), {batch, t, f(geometry.gdn_value_heads), f(1)});
        auto b = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_b.weight"), {batch, t, f(geometry.gdn_value_heads), f(1)});
        auto dt = r.reshape(r.cast(r.weight(prefix + "linear_attn.dt_bias"), Type::kFloat32), {f(1), f(1), f(geometry.gdn_value_heads), f(1)});
        auto decay = r.reshape(r.decay("model.language_model." + prefix + "linear_attn.A_log"), {f(1), f(1), f(geometry.gdn_value_heads), f(1)});
        auto g = r.binary(c::BinaryKind::kMultiply, decay,
            r.unary(c::UnaryKind::kSoftplus, r.binary(c::BinaryKind::kAdd, a, dt)));
        auto beta = r.cast(r.unary(c::UnaryKind::kSigmoid, b), Type::kFloat32);
        auto recurrent = r.op(c::Operation(c::GatedDeltaNet{
            .snapshot_count = static_cast<std::uint32_t>(prefix_rows)}),
            {r.cast(r.gdn_key_rows(r.l2(q)), Type::kFloat32), r.cast(r.gdn_key_rows(r.l2(k)), Type::kFloat32),
             r.cast(v, Type::kFloat32), g, beta, state},
            {r.dense_spec(Type::kFloat32, {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}),
             prefix_rows > 1
                 ? r.dense_spec(Type::kFloat32, {batch, f(static_cast<std::int64_t>(prefix_rows)), f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})
                 : r.dense_spec(Type::kFloat32, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})});
        store_state(ports.recurrent, recurrent_schema, r.reshape(recurrent[1],
            {f(static_cast<std::int64_t>(requests * prefix_rows)), f(recurrent_schema.item_elements())}));
        mixed = block_graph::emit_gdn_epilogue(r, recurrent[0], normalized,
            "model.language_model." + prefix,
            {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}, packed,
            i == 0 ? std::span<V>(outputs.layer0_probes) : std::span<V>{});
      }
    } else {
      const auto layer = geometry.attention_slot(i);
      const auto &schema = inputs.target_attention_schemas.empty() ? inputs.plain_attention_schema : inputs.target_attention_schemas[layer];
      ports.key = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kKey)));
      ports.value = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kValue)));
      outputs.append_cache_inputs[layer * 2] = ports.key;
      outputs.append_cache_inputs[layer * 2 + 1] = ports.value;
      auto projected = r.reshape(r.linear(normalized, prefix + "self_attn.q_proj.weight"), {batch, t, f(geometry.query_heads), f(geometry.key_depth + geometry.value_depth)});
      auto q = r.norm(r.slice(projected, 3, f(0), f(geometry.key_depth)), prefix + "self_attn.q_norm.weight");
      auto gate = r.reshape(r.copy(r.slice(projected, 3, f(geometry.key_depth), f(geometry.value_depth))), {packed, f(geometry.attention_width())});
      auto k = r.norm(r.reshape(r.linear(normalized, prefix + "self_attn.k_proj.weight"), {batch, t, f(geometry.kv_heads), f(geometry.key_depth)}), prefix + "self_attn.k_norm.weight");
      auto v = r.linear(normalized, prefix + "self_attn.v_proj.weight");
      const c::RoPE rope{.rotary_dimensions = static_cast<std::size_t>(geometry.rotary), .sequence_axis = 1, .layout = c::RopeLayout::kHalfSplit};
      q = r.same(rope, {q, inputs.positions, frequencies}, q);
      k = r.same(rope, {k, inputs.positions, frequencies}, k);
      auto &key_append = outputs.append_nodes[layer * 2];
      auto &value_append = outputs.append_nodes[layer * 2 + 1];
      auto key_rows = r.reshape(k, {packed, f(geometry.kv_heads), f(geometry.key_depth)});
      auto value_rows = r.reshape(v, {packed, f(geometry.kv_heads), f(geometry.value_depth)});
      const auto cache_capacity = schema.capacity();
      if (cache_capacity.key_storage.data_type() == Type::kFloat8E4M3FN)
        key_rows = r.one(c::Operation::custom(c::model_ops::CacheEncode{cache_capacity.key_storage}),
            {key_rows}, r.dense_spec(Type::kFloat8E4M3FN, {packed, f(geometry.kv_heads), f(geometry.key_depth)}));
      if (cache_capacity.value_storage.data_type() == Type::kFloat8E4M3FN)
        value_rows = r.one(c::Operation::custom(c::model_ops::CacheEncode{cache_capacity.value_storage}),
            {value_rows}, r.dense_spec(Type::kFloat8E4M3FN, {packed, f(geometry.kv_heads), f(geometry.value_depth)}));
      r.op(c::Operation::custom(take(g::batch::paged_append(schema, c::BatchCacheRole::kKey))),
          {key_rows}, {}, &key_append);
      r.op(c::Operation::custom(take(g::batch::paged_append(schema, c::BatchCacheRole::kValue))),
          {value_rows}, {}, &value_append);
      if (has_prior_append) check(r.builder.depends_on(key_append, prior_append));
      check(r.builder.depends_on(value_append, key_append));
      prior_append = value_append; has_prior_append = true;
      auto attention = r.one(c::Operation::custom(c::PagedAttention{
          .capacity = cache_capacity, .scale = 1.F / std::sqrt(static_cast<float>(geometry.key_depth)), .causal = true}),
          {r.reshape(q, {packed, f(geometry.query_heads), f(geometry.key_depth)}), ports.key, ports.value, inputs.attention_metadata},
          r.dense_spec(r.activation_type(), {packed, f(geometry.query_heads), f(geometry.value_depth)}), &outputs.attention_nodes[layer]);
      check(r.builder.depends_on(outputs.attention_nodes[layer], key_append));
      check(r.builder.depends_on(outputs.attention_nodes[layer], value_append));
      auto gated = r.binary(c::BinaryKind::kMultiply, r.reshape(attention, {packed, f(geometry.attention_width())}), r.unary(c::UnaryKind::kSigmoid, gate));
      mixed = r.linear(gated, prefix + "self_attn.o_proj.weight");
    }
    const auto ffn_input = r.binary(c::BinaryKind::kAdd, x, mixed);
    {
      GraphAssembly::Scope scope(assembly, i < inputs.cpu_ffn_layers ? assembly.cpu() : assembly.primary());
      x = block_graph::emit_ffn(r, ffn_input, "model.language_model." + prefix,
          i == 0 ? std::span<V>(outputs.layer0_probes) : std::span<V>{});
    }
    if (i == 0) outputs.layer0_probes[4] = x;
  }
  return {x, prior_append, has_prior_append};
}
} // namespace qwen35
