// Complete Qwen-private cold Family assembly. No Model/Session or runtime
// admission/publication owner is included or retained by this implementation.
#include "family_factory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <utility>

#include "mtp_graph.h"
#include "../../inference/backend.h"

namespace qwen35 {
using Clock = std::chrono::steady_clock;
namespace {
// Distinguish direct packed contractions from the ordinary Decode/Cast route.
// The latter has per-projection Invocation workspace and may use managed
// provider temporaries; it does not replace the original constant weight bank.
void report_packed_projections(const g::Graph &graph, const g::CompileReport &report,
                               std::span<const PackedProjectionNodes> projections, std::size_t graph_workspace_bytes) {
  std::size_t packed = 0;
  for (const auto &projection : projections) {
    if (projection.count != 3 && projection.count != 4)
      throw std::runtime_error("packed projection requires its actual three/four-node recipe");
    std::size_t containing = 0;
    for (std::size_t step_index = 0; step_index < report.step_count(); ++step_index) {
      const auto step = take(report.step(step_index));
      std::array<bool, 4> included{};
      for (std::size_t i = 0; i < step.node_count; ++i) {
        const auto node = take(report.step_node(step_index, i));
        for (std::size_t j = 0; j < projection.count; ++j) included[j] = included[j] || node == projection.chain[j];
      }
      if (!included[projection.count - 1]) continue;
      ++containing;
      const bool contracted = step.policy == g::kArtifactPackedProjectionPolicy && step.node_count == projection.count &&
          std::ranges::all_of(std::span(included).first(projection.count), [](bool value) { return value; });
      if (contracted) {
        for (std::size_t j = 0; j + 1 < projection.count; ++j) {
          const auto intermediate = take(graph.result(projection.chain[j]));
          for (std::size_t i = 0; i < report.physical_value_count(); ++i)
            if (take(report.physical_value(i)).value == intermediate)
              throw std::runtime_error("packed projection materialized a floating weight intermediate");
        }
        ++packed;
      }
    }
    if (containing != 1) throw std::runtime_error("original packed projection is missing from its selected plan");
  }
  if (!projections.empty())
    std::cerr << "original_quantized_projections=" << projections.size()
              << " packed_contractions=" << packed
              << " decomposed_projections=" << projections.size() - packed
              << " graph_workspace_bytes=" << graph_workspace_bytes << '\n';
}

void merge_execution_plan_facts(ExecutionPlanFacts &result, const ExecutionPlanFacts &facts) {
  result.domain_count = (std::max)(result.domain_count, facts.domain_count);
  for (std::size_t i = 0; i < facts.domain_count; ++i) {
    result.workspace_peak[i] = (std::max)(result.workspace_peak[i], facts.workspace_peak[i]);
    result.workspace_limits[i] = facts.workspace_limits[i];
  }
}
} // namespace

sampling::LogitsInputsFor<V> prepare_logits_inputs(Recipe &r, sampling::LogitsPolicy policy) {
  sampling::LogitsInputsFor<V> result;
  if (policy.bias) result.bias = r.input(dense(Type::kFloat32, {1, r.geometry.vocabulary}));
  if (policy.allowed) result.allowed = r.input(dense(Type::kBool, {1, r.geometry.vocabulary}));
  if (!policy.history.empty()) result.counts = r.input(dense(Type::kInt32, {1, r.geometry.vocabulary}));
  return result;
}

V draft_scores(Recipe &recipe, V selected_hidden, DraftProjectionView projection) {
  if (!projection.original_ids) return recipe.scores(selected_hidden, std::string(recipe.geometry.head_role()));
  return recipe.emit_projection(selected_hidden, projection.projection, true);
}
V original_draft_id(Recipe &recipe, V compact_id, DraftProjectionView projection) {
  if (!projection.original_ids) return compact_id;
  return recipe.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
      {recipe.constant(*projection.original_ids), compact_id}, recipe.spec(compact_id));
}

ExecutionPlanFacts inspect_execution_plan(const inference::Backend &execution,
    std::size_t cpu_ffn_layers, const g::CompileReport &report,
    std::span<const g::ConstantBinding> constants) {
  ExecutionPlanFacts facts;
  if (execution.host_only()) {
    if (report.storage_domain_count() != 1 || report.target_count() != 1)
      throw std::runtime_error("CPU graph lost its single host allocation domain");
    const auto domain = take(report.storage_domain(0));
    if (domain.workspace_budget_bytes == SIZE_MAX || domain.workspace_bytes > domain.workspace_budget_bytes)
      throw std::runtime_error("CPU known workspace lacks its finite domain admission limit");
    facts.workspace_peak[0] = (std::max)(facts.workspace_peak[0], domain.workspace_bytes);
    facts.workspace_limits[0] = domain.workspace_budget_bytes;
    std::cerr << "cpu_plan known_workspace_host=" << domain.workspace_bytes
        << " workspace_host_limit=" << domain.workspace_budget_bytes
        << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_memory_cap=0\n";
    facts.domain_count = 1;
    return facts;
  }
  if (!cpu_ffn_layers) return facts;
  if (report.storage_domain_count() != 3 || report.target_count() != 2)
    throw std::runtime_error("CPU FFN graph lost its configured CPU/GPU domains");
  facts.domain_count = facts.workspace_peak.size();
  for (std::size_t i = 0; i < facts.workspace_peak.size(); ++i) {
    const auto domain = take(report.storage_domain(i));
    if (domain.workspace_budget_bytes == SIZE_MAX || domain.workspace_bytes > domain.workspace_budget_bytes)
      throw std::runtime_error("hybrid known workspace lacks its finite domain admission limit");
    facts.workspace_peak[i] = (std::max)(facts.workspace_peak[i], domain.workspace_bytes);
    facts.workspace_limits[i] = domain.workspace_budget_bytes;
  }
  std::size_t host_constants = 0;
  for (std::size_t i = 0; i < report.constant_count(); ++i) {
    const auto constant = take(report.constant(i));
    const auto source = std::ranges::find(constants, constant.value, &g::ConstantBinding::value);
    if (source == constants.end()) throw std::runtime_error("hybrid constant lost its original binding");
    if (source->tensor.storage().resource() != weight_resource(execution, true)) continue;
    if (constant.storage != g::StorageDomainId{1} || constant.materialized_bytes)
      throw std::runtime_error("CPU FFN constant acquired an unexpected canonical replica");
    for (std::size_t value = 0; value < report.physical_value_count(); ++value) {
      const auto physical = take(report.physical_value(value));
      if (physical.value == constant.value && physical.constant && physical.storage != g::StorageDomainId{1})
        throw std::runtime_error("CPU FFN constant escaped its host allocation domain");
    }
    ++host_constants;
  }
  std::cerr << "hybrid_plan host_constants=" << host_constants << " materialized_constant_bytes="
      << report.materialized_constant_bytes() << " activation_copy_bytes=" << report.copy_bytes()
      << " known_workspace_device=" << take(report.storage_domain(0)).workspace_bytes
      << " known_workspace_host=" << take(report.storage_domain(1)).workspace_bytes
      << " known_workspace_staging=" << take(report.storage_domain(2)).workspace_bytes
      << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_vram_cap=0\n";
  return facts;
}

FamilyBuildResult build_family(const FamilyBuildRequest &build_request,
    const FamilyModelFacts &model, const FamilyPreparedFacts &prepared,
    const SamplingConfiguration &configuration, const FamilyCompileContext &context) {
  if (build_request.chunks.empty()) throw std::runtime_error("text finite family requires known nonempty chunks");
  const auto &geometry = model.geometry;
  const auto precision = model.precision;
  auto &weights = model.weights;
  const auto &projections = model.projections;
  const auto &weight_meanings = model.weight_meanings;
  auto &execution = context.execution;
  const auto &allocation_budget = context.allocation_budget;
  const auto &execution_budget = context.execution_budget;
  const auto &attention_schema = prepared.attention_schema;
  const auto attention_schemas = prepared.attention_schemas;
  const auto state_schema = prepared.recurrent_descriptions;
  const auto requests = build_request.requests, prefix_rows = build_request.prefix_rows;
  const auto role = build_request.role;
  const auto completion = build_request.completion;
  const auto packing = build_request.packing;
  const auto diagnostic = build_request.diagnostic, admitted_exact = build_request.admitted_exact;
  const auto &selected_sampler = configuration;
  const bool packed_prefill = packs_prefill(packing), compact_draft = packs_draft(packing);
  const bool packed_verify = packs_verify(packing), packed_target = packs_target(packing);
  const auto target_kv_layers = geometry.attention_layers(), draft_bank = target_kv_layers * 2;
  const auto hidden = geometry.hidden, vocabulary = geometry.vocabulary;
  const auto ports_per_request = cache::add(76, cache::multiply(38, prefix_rows - 1));
  ExecutionPlanFacts plan_facts;
    auto result = std::make_unique<Family>();
    auto &definition = *result;
    definition.layers.resize(geometry.layers.size()); definition.attention_nodes.resize(target_kv_layers);
    definition.append_nodes.resize(draft_bank + 2);
    definition.append_cache_inputs.resize(draft_bank + 2);
    definition.requests = requests; definition.role = role; definition.prefix_rows = prefix_rows; definition.completion = completion;
    definition.admitted_exact = admitted_exact; definition.packing = packing;
    definition.state_resource_ports.reserve(ports_per_request * requests);
    for (const auto tokens : build_request.chunks) {
      if (std::find(definition.chunks.begin(), definition.chunks.end(), tokens) == definition.chunks.end())
        definition.chunks.push_back(tokens);
    }
    const auto start = Clock::now();
    const auto max_query_rows = *std::max_element(definition.chunks.begin(), definition.chunks.end());
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, max_query_rows, requests, &weight_meanings);
    const auto t = r.token_rows, packed = r.packed_rows;
    // In a packed family [batch,t] partitions the physical carrier only.
    // Original cu_seqlens and explicit positions describe logical members.
    const auto batch = f(static_cast<std::int64_t>(requests));
    g::slots::Schema hidden_schema;
    if (verifies_prefix(role) || has_mtp(role) || prepared.retain_hidden) {
      hidden_schema = take(g::slots::Schema::create(dense(precision.activation_type(), {1, hidden}),
          prepared.hidden_slots,
          cache::add(geometry.hidden_bytes(), cache::state_pool_padding(geometry.hidden_bytes()))));
      definition.hidden_state.destination_ids = r.input(dense(Type::kInt32,
          {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
    }
    definition.last_row = r.last_row;
    definition.attention_schema = attention_schema;
    // A sampler produces [B,1]. Preserve that exact external token carrier
    // for ordinary T1 and flatten only the mathematical embedding indices.
    definition.token_port = role == GraphRole::kText
        ? r.input(r.dense_spec(Type::kInt32, {packed, f(1)}))
        : r.input(r.dense_spec(Type::kInt32,
            {has_mtp(role) && !is_mtp_prefill(role) ? batch : packed}));
    definition.position_port = requests == 1 ? r.input(r.dense_spec(Type::kInt32, {t})) :
        r.input(r.dense_spec(Type::kInt32, {batch, t}));
    if ((requests != 1 || packed_prefill) && !verifies_prefix(role)) definition.last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
    if (packed_target) {
      definition.sequence_offsets_port = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(requests + 1)}));
      definition.history_select_port = r.input(dense(Type::kInt32,
          {static_cast<std::int64_t>(cache::multiply(cache::multiply(requests, prefix_rows), geometry.conv_kernel - 1))}));
    }
    if (packed_verify) definition.prefix_ids_port = r.input(dense(Type::kInt32,
        {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
    definition.metadata_port = r.input(take(definition.attention_schema.metadata_description()));
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    V target_tokens = role == GraphRole::kText
        ? r.reshape(definition.token_port, {packed}) : definition.token_port;
    std::vector<sampling::Distribution> draft_distributions;
    std::vector<V> proposal_ids, draft_valid;
    std::vector<sampling::LogitsInputsFor<V>> logits_inputs(requests);
    if (completion == TextCompletion::kEmitNext && role != GraphRole::kVerify && !selected_sampler.modifiers.empty())
      for (auto &inputs : logits_inputs) {
        inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
        if (inputs.bias) { definition.logits_bias_ports.push_back(*inputs.bias); definition.state_resource_ports.push_back(*inputs.bias); }
        if (inputs.allowed) { definition.logits_allowed_ports.push_back(*inputs.allowed); definition.state_resource_ports.push_back(*inputs.allowed); }
        if (inputs.counts) { definition.logits_counts_ports.push_back(*inputs.counts); definition.state_resource_ports.push_back(*inputs.counts); }
      }
    definition.batched_text_sampling = requests > 1 && role == GraphRole::kText &&
        completion == TextCompletion::kEmitNext && selected_sampler.enabled &&
        selected_sampler.modifiers.empty() && selected_sampler.policy.top_p == 1.F &&
        (!selected_sampler.policy.top_k || selected_sampler.policy.top_k == static_cast<std::size_t>(vocabulary));
    if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify)
      definition.sampling_identity_port = r.input(definition.batched_text_sampling
          ? dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 4})
          : dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 3, 4}));
    const auto sampling_identity = [&](std::size_t request, std::size_t draw) {
      return r.copy(r.reshape(r.slice(r.slice(definition.sampling_identity_port, 0,
          f(static_cast<std::int64_t>(request)), f(1)), 1,
          f(static_cast<std::int64_t>(draw)), f(1)), {f(1), f(4)}));
    };
    const auto sampling_row = [&](V values, std::size_t row) {
      return r.copy(r.slice(values, 0, f(static_cast<std::int64_t>(row)), f(1)));
    };
    D draft_sequence_rows = f(0), draft_packed_rows = f(0);
    if (has_mtp(role)) {
      auto &port = definition.hidden_state;
      const auto source_rows = f(static_cast<std::int64_t>(cache::multiply(requests, draft_lag(role) + 1)));
      port.pool = r.input(take(hidden_schema.capacity_description()));
      port.source_ids = r.input(r.dense_spec(Type::kInt32, {source_rows}));
      const auto gathered = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {port.pool, port.source_ids}, r.dense_spec(precision.activation_type(), {source_rows, f(hidden)}), &port.gather);
      // One fixed pool and private IDs; the original ready leases still prove
      // exactly these rows. Batched/packed order is [all carries, all priors].
      const bool chronological = requests == 1 && role == GraphRole::kMtpCatchupRound &&
          packing == RowPacking::kRectangular;
      // A K1 producer stores [prior, carry]. The lag1 R1 reader uses that same
      // real ID order, in ready and imported modes; only the graph interprets
      // the two chronological rows. No source ticket/ID is manufactured.
      definition.carry_input_port = r.slice(gathered, 0, chronological ? batch : f(0), batch);
      const auto draft_rows = is_mtp_prefill(role)
          ? (draft_lag(role) ? D(take(r.builder.dimension_sum(t, f(1)))) : t)
          : f(static_cast<std::int64_t>(draft_lag(role) + 1));
      const auto packed_draft_rows = !draft_lag(role) && is_mtp_prefill(role) ? packed : requests == 1 ? draft_rows :
          draft_rows.is_fixed() ? f(static_cast<std::int64_t>(cache::multiply(requests, static_cast<std::size_t>(draft_rows.fixed_value())))) :
          D(take(r.builder.dimension_product(draft_rows, batch)));
      draft_sequence_rows = draft_rows; draft_packed_rows = packed_draft_rows;
      if (packed_prefill || compact_draft) {
        definition.draft_hidden_select_port = r.input(r.dense_spec(Type::kInt32, {packed_draft_rows}));
        if (draft_lag(role)) definition.draft_token_select_port = r.input(r.dense_spec(Type::kInt32, {packed_draft_rows}));
      }
      if (compact_draft) definition.draft_last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      definition.draft_position_port = r.input(requests == 1 ? r.dense_spec(Type::kInt32, {draft_rows})
          : r.dense_spec(Type::kInt32, {batch, draft_rows}));
      if (draft_lag(role)) {
        definition.prior_input_port = r.slice(gathered, 0, chronological ? f(0) : batch, batch);
        definition.catchup_token_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      }
      if (drafts_token(role) || !is_mtp_prefill(role)) {
        V ids = definition.token_port, hidden_rows = definition.carry_input_port;
        if (compact_draft) {
          const auto token_source = r.one(c::Concat{.axis = 0}, {definition.catchup_token_port, ids},
              r.dense_spec(Type::kInt32, {packed_draft_rows}));
          const auto hidden_source = r.one(c::Concat{.axis = 0}, {hidden_rows, definition.prior_input_port},
              r.dense_spec(precision.activation_type(), {packed_draft_rows, f(hidden)}));
          ids = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {token_source, definition.draft_token_select_port}, r.dense_spec(Type::kInt32, {packed_draft_rows}));
          hidden_rows = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {hidden_source, definition.draft_hidden_select_port}, r.dense_spec(precision.activation_type(), {packed_draft_rows, f(hidden)}));
        } else if (draft_lag(role)) {
          if (requests == 1) {
            ids = r.one(c::Concat{.axis = 0}, {definition.catchup_token_port, ids}, r.dense_spec(Type::kInt32, {f(2)}));
            hidden_rows = r.one(c::Concat{.axis = 0}, {definition.prior_input_port, hidden_rows}, r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
          } else {
            ids = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.catchup_token_port, {batch, f(1)}), r.reshape(ids, {batch, f(1)})},
                r.dense_spec(Type::kInt32, {batch, f(2)})), {packed_draft_rows});
            hidden_rows = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.prior_input_port, {batch, f(1), f(hidden)}), r.reshape(hidden_rows, {batch, f(1), f(hidden)})},
                r.dense_spec(precision.activation_type(), {batch, f(2), f(hidden)})), {packed_draft_rows, f(hidden)});
          }
        }
        const auto projection = mtp::input_projection(r, ids, hidden_rows);
        const mtp::Geometry row_geometry{requests, draft_rows, packed_draft_rows};
        const auto kv = mtp::project_kv(r, projection, definition.draft_position_port, frequencies, row_geometry);
        mtp::AppendNodes append;
        if (drafts_token(role)) {
          definition.draft_key_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
          definition.draft_value_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
          definition.draft_metadata_port = r.input(take(attention_schema.metadata_description()));
          definition.append_cache_inputs[draft_bank] = definition.draft_key_port;
          definition.append_cache_inputs[draft_bank + 1] = definition.draft_value_port;
          const auto head = mtp::full_head(r, projection, kv, definition.draft_position_port, frequencies,
              {attention_schema, definition.draft_key_port, definition.draft_value_port, definition.draft_metadata_port},
              f(static_cast<std::int64_t>(draft_lag(role))), row_geometry,
              [&](Recipe &recipe, V selected) { return qwen35::draft_scores(recipe, selected, prepared.draft_projection); }, definition.draft_last_ids_port);
          append = head.append_nodes; definition.draft_attention_node = head.attention_node;
          if (!selected_sampler.enabled && selected_sampler.modifiers.empty())
            definition.draft_id_port = qwen35::original_draft_id(r, r.op(c::Operation(c::TopK{.count = 1}), {head.logits},
                {r.dense_spec(Type::kFloat32, {batch, f(1)}), r.dense_spec(Type::kInt32, {batch, f(1)})})[1], prepared.draft_projection);
          else {
            for (std::size_t request = 0; request < requests; ++request) {
              const auto processed = sampling::process_logits(r, sampling_row(head.logits, request), logits_inputs[request], selected_sampler.modifiers.history);
              if (selected_sampler.enabled) {
                draft_distributions.push_back(sampling::processed_distribution(r, processed, selected_sampler.policy));
                proposal_ids.push_back(r.one(c::sampling_ops::categorical(),
                    {draft_distributions.back().log_probabilities, sampling_identity(request, 0)}));
              } else {
                const auto selection = sampling::processed_selection(r, processed, std::optional<V>{}, selected_sampler.policy);
                proposal_ids.push_back(r.reshape(selection.id, {f(1)})); draft_valid.push_back(*selection.valid);
              }
            }
            definition.draft_id_port = r.reshape(r.concat(proposal_ids, 0,
                r.dense_spec(Type::kInt32, {batch})), {batch, f(1)});
          }
          r.publish(definition.draft_id_port);
          target_tokens = requests == 1 ? r.one(c::Concat{.axis = 0},
              {definition.token_port, r.reshape(definition.draft_id_port, {f(1)})}, r.dense_spec(Type::kInt32, {packed})) :
              r.reshape(r.one(c::Concat{.axis = 1},
                  {r.reshape(definition.token_port, {batch, f(1)}), r.reshape(definition.draft_id_port, {batch, f(1)})},
                  r.dense_spec(Type::kInt32, {batch, f(2)})), {packed});
        } else append = mtp::append_kv(r, attention_schema, kv);
        definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      }
    }
    // Writer-only draft maintenance has no attention data edge into target.
    // Order its real cache effects before target's first append as well.
    const auto target = emit_target_graph(r, {
        .token_ids = target_tokens, .positions = definition.position_port,
        .attention_metadata = definition.metadata_port, .frequencies = frequencies,
        .sequence_offsets = definition.sequence_offsets_port, .history_select = definition.history_select_port,
        .requests = requests, .prefix_rows = prefix_rows, .packing = packing,
        .state_slots = prepared.state_slots, .recurrent_descriptions = state_schema,
        .plain_attention_schema = attention_schema, .target_attention_schemas = attention_schemas,
        .cpu_ffn_layers = model.cpu_ffn_layers,
        .preceding_append = has_mtp(role) && !is_mtp_prefill(role)
            ? definition.append_nodes[draft_bank + 1] : g::NodeId{},
        .has_preceding_append = has_mtp(role) && !is_mtp_prefill(role)},
        {definition.layers, definition.attention_nodes,
         std::span<g::NodeId>(definition.append_nodes).first(draft_bank),
         std::span<V>(definition.append_cache_inputs).first(draft_bank),
         definition.state_sources, definition.probe_ports});
    auto x = target.hidden_rows;
    auto prior_append = target.last_append;
    bool has_prior_append = target.has_append;
    if (diagnostic) for (std::size_t i = 0; i < definition.probe_ports.size(); ++i) {
      r.publish(definition.probe_ports[i]);
      auto description = r.spec(definition.probe_ports[i]);
      if (description.data_type() != precision.activation_type()) throw std::runtime_error("layer0 diagnostic differs from source activation precision");
    }
    // The selected child decides caller-written versus generated result storage.
    const auto normalized_last = r.norm(x, "norm.weight");
    if (is_mtp_prefill(role)) {
      const auto kv = mtp::prefill_kv(r, definition.token_port, normalized_last,
          definition.carry_input_port, definition.prior_input_port, definition.catchup_token_port,
          mtp::Geometry{requests, t, packed}, mtp::Geometry{requests, draft_sequence_rows, draft_packed_rows}, definition.draft_position_port,
          frequencies, draft_lag(role) != 0,
          {definition.draft_hidden_select_port, definition.draft_token_select_port});
      const auto append = mtp::append_kv(r, attention_schema, kv);
      if (has_prior_append) check(r.builder.depends_on(append.key, prior_append));
      definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      prior_append = append.value; has_prior_append = true;
    }
    auto last = verifies_prefix(role) ? r.copy(normalized_last) : packed_prefill
        ? r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
            {normalized_last, definition.last_ids_port}, r.dense_spec(precision.activation_type(), {batch, f(hidden)})) :
        (requests == 1 ? r.copy(r.slice(normalized_last, 0, D(r.last_row), f(1))) :
        r.one(c::Gather{}, {normalized_last, definition.last_ids_port},
            r.dense_spec(precision.activation_type(), {batch, f(hidden)})));
    if (hidden_schema.valid()) {
      auto &port = definition.hidden_state;
      // Verifiers retain every target prefix; prefill/single retain only the
      // same final row per member as their existing complete boundary.
      const auto retained = packed_verify
          ? r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {normalized_last, definition.prefix_ids_port}, r.dense_spec(precision.activation_type(),
                  {f(static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))), f(hidden)}))
          : verifies_prefix(role) ? normalized_last : last;
      r.op(c::Operation::custom(g::slots::Store{hidden_schema}),
          {retained, port.destination_ids}, {}, &port.store);
      if (has_prior_append) check(r.builder.depends_on(port.store, prior_append));
      if (has_mtp(role)) definition.state_sources.push_back({port.store, port.gather});
    }
    definition.hidden_port = last;
    const auto bonus_logits_inputs = [&](std::size_t request) {
      auto inputs = logits_inputs[request];
      if (!selected_sampler.modifiers.history.empty())
        inputs.counts = sampling::advance_history_counts(r, *inputs.counts, r.reshape(proposal_ids[request], {f(1)}));
      return inputs;
    };
    if (completion == TextCompletion::kEmitNext) {
      definition.logits_port = r.scores(last, std::string(geometry.head_role()));
      if ((!selected_sampler.enabled && selected_sampler.modifiers.empty()) || role == GraphRole::kVerify) definition.id_port = r.op(c::Operation(c::TopK{.count = 1}), {definition.logits_port},
          {r.dense_spec(Type::kFloat32, {verifies_prefix(role) ? packed : batch, f(1)}),
           r.dense_spec(Type::kInt32, {verifies_prefix(role) ? packed : batch, f(1)})})[1];
      else if (definition.batched_text_sampling) {
        const auto p = sampling::distribution(r, definition.logits_port, selected_sampler.policy);
        definition.id_port = r.reshape(r.one(c::sampling_ops::categorical(),
            {p.log_probabilities, definition.sampling_identity_port}), {batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.reshape(p.valid, {batch}));
        r.publish(definition.sampling_valid_port);
      } else if (!selected_sampler.enabled) {
        std::vector<V> ids, valid;
        V false_value;
        if (drafts_token(role)) false_value = r.cast(r.scalar(0), Type::kBool);
        for (std::size_t request = 0; request < requests; ++request) {
          const auto first_row = drafts_token(role) ? request * 2 : request;
          const auto selection = sampling::processed_selection(r,
              sampling::process_logits(r, sampling_row(definition.logits_port, first_row), logits_inputs[request], selected_sampler.modifiers.history),
              std::optional<V>{}, selected_sampler.policy);
          ids.push_back(r.reshape(selection.id, {f(1)}));
          auto ok = *selection.valid;
          if (drafts_token(role)) {
            const auto bonus = sampling::processed_selection(r,
                sampling::process_logits(r, sampling_row(definition.logits_port, first_row + 1), bonus_logits_inputs(request), selected_sampler.modifiers.history),
                std::optional<V>{}, selected_sampler.policy);
            ids.push_back(r.reshape(bonus.id, {f(1)}));
            ok = r.one(c::Select{}, {ok, draft_valid[request], false_value});
            ok = r.one(c::Select{}, {ok, *bonus.valid, false_value});
          }
          valid.push_back(ok);
        }
        definition.id_port = r.reshape(r.concat(ids, 0,
            r.dense_spec(Type::kInt32, {drafts_token(role) ? packed : batch})),
            {drafts_token(role) ? packed : batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.concat(valid, 0, r.dense_spec(Type::kBool, {batch})));
        r.publish(definition.sampling_valid_port);
      } else {
        std::vector<V> ids, valid, accepted;
        for (std::size_t request = 0; request < requests; ++request) {
          const auto p = sampling::processed_distribution(r, sampling::process_logits(r,
              sampling_row(definition.logits_port, drafts_token(role) ? request * 2 : request), logits_inputs[request], selected_sampler.modifiers.history), selected_sampler.policy);
          if (drafts_token(role)) {
            const auto bonus = sampling::processed_distribution(r, sampling::process_logits(r,
                sampling_row(definition.logits_port, request * 2 + 1), bonus_logits_inputs(request), selected_sampler.modifiers.history), selected_sampler.policy);
            const auto decision = sampling::decision(r, p.log_probabilities,
                draft_distributions[request].log_probabilities, bonus.log_probabilities,
                proposal_ids[request], sampling_identity(request, 1), sampling_identity(request, 2));
            ids.push_back(r.one(c::Select{}, {decision.accepted, proposal_ids[request], decision.final_id}));
            ids.push_back(decision.final_id); accepted.push_back(decision.accepted);
            auto ok = r.one(c::Select{}, {p.valid, draft_distributions[request].valid, r.cast(r.scalar(0), Type::kBool)});
            ok = r.one(c::Select{}, {ok, bonus.valid, r.cast(r.scalar(0), Type::kBool)});
            valid.push_back(r.one(c::Select{}, {ok, decision.valid, r.cast(r.scalar(0), Type::kBool)}));
          } else {
            ids.push_back(r.one(c::sampling_ops::categorical(), {p.log_probabilities, sampling_identity(request, 0)}));
            valid.push_back(p.valid);
          }
        }
        definition.id_port = r.reshape(r.concat(ids, 0, r.dense_spec(Type::kInt32,
            {f(static_cast<std::int64_t>(ids.size()))})), {verifies_prefix(role) ? packed : batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.concat(valid, 0, r.dense_spec(Type::kBool, {batch})));
        r.publish(definition.sampling_valid_port);
        if (drafts_token(role)) {
          definition.sampling_accept_port = r.host_predicate(r.concat(accepted, 0, r.dense_spec(Type::kBool, {batch})));
          r.publish(definition.sampling_accept_port);
        }
      }
    }
    r.publish(definition.hidden_port);
    if (completion == TextCompletion::kEmitNext) { r.publish(definition.logits_port); r.publish(definition.id_port); }
    if (drafts_token(role) && requests == 1 && packing == RowPacking::kRectangular &&
        !selected_sampler.enabled && selected_sampler.modifiers.empty() &&
        prepared.emit_successor) {
      // Two complete, independently owned I32[1] outputs. OutputSource cannot
      // grant a fabricated subview of the two-row target-ID writer to B.
      definition.successor_token_port = r.copy(r.reshape(r.slice(definition.id_port, 0, f(1), f(1)), {f(1)}));
      definition.successor_catchup_port = r.copy(r.reshape(definition.draft_id_port, {f(1)}));
      r.publish(definition.successor_token_port); r.publish(definition.successor_catchup_port);
    }
    definition.graph = take(std::move(r.builder).finish());
    definition.literals = std::move(assembly.literals);
    definition.specs = std::move(r.specs);
    std::vector<std::array<g::ShapeBinding, 1>> bindings;
    bindings.reserve(definition.chunks.size());
    for (const auto tokens : definition.chunks)
      bindings.push_back({g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(tokens - 1)}});
    std::vector<g::ExactShapeVariant> variants;
    variants.reserve(bindings.size());
    for (const auto &point : bindings) variants.push_back({point});
    g::CompileOptions compile_options{.shape_policy = g::CompileShapePolicy::kFiniteExact,
        .exact_variants = variants, .constants = assembly.constants()};
    // The library's cold operation admission cap covers the whole family.
    // Budget the requested finite children explicitly at the per-graph default;
    // this is host search/storage policy, not device or execution capacity.
    const auto per_graph_operations = g::CompileOptions{}.max_operation_proposals;
    if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph_operations)
      throw std::runtime_error("finite text operation proposal budget overflows");
    compile_options.max_operation_proposals = per_graph_operations * variants.size();
    const auto per_graph_preparations = g::CompileOptions{}.max_placement_attempts;
    if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph_preparations)
      throw std::runtime_error("finite text artifact preparation budget overflows");
    compile_options.max_placement_attempts = per_graph_preparations * variants.size();
    std::cerr << "compile full text finite family requests=" << requests << " variants=" << variants.size()
              << " max_query_rows=" << max_query_rows << " nodes=" << definition.graph.node_count()
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts << '\n';
    compile_options.resource_budget = execution_budget;
    auto compiled = execution.compile(definition.graph, assembly.constants(), assembly.options(compile_options));
    if (!compiled && (packed_target || compact_draft) && compiled.error().IsUnimplemented() && !compiled.error().has_native_error())
      return {}; // Cold capability only; no execution/device error is a fallback.
    auto compiled_family = take(std::move(compiled));
    const auto &family_report = compiled_family.report;
    const auto storage = family_report.total_compiled_storage();
    std::cerr << "compiled text finite family variants=" << family_report.exact_variant_count()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " operation_proposals=" << family_report.total_search().operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts
              << " placement_attempts=" << family_report.total_search().placement_attempts
              << " cold_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    const auto hidden_store = definition.hidden_state.store;
    const auto extra_stores = hidden_store.valid() ? std::span(&hidden_store, 1) : std::span<const g::NodeId>{};
    check(definition.programs.prepare(std::move(compiled_family), definition.graph, variants,
        {definition.state_sources, extra_stores}, take(execution_budget.metadata_allocator())));
    for (std::size_t index = 0; index < definition.chunks.size(); ++index) {
      const auto &child = definition.programs.child(index);
      merge_execution_plan_facts(plan_facts, inspect_execution_plan(
          execution, model.cpu_ffn_layers, child.compiled.report, assembly.constants()));
      report_packed_projections(definition.graph, child.compiled.report, r.packed_projection_nodes,
          child.compiled.executable.workspace_bytes());
    }
  return {std::move(result), plan_facts};
}

} // namespace qwen35
