// Complete Qwen-private cold Family assembly. No Model/Session or runtime
// admission/publication owner is included or retained by this implementation.
#include "family_factory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <utility>

#include "mtp_graph.h"
#include "ordinary_io.h"
#include "../../inference/backend.h"

#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
#include "../../../tests/tools/inference_warm_cost_probe.h"
#endif

namespace qwen35 {
using Clock = std::chrono::steady_clock;
namespace {
std::mutex family_log_mutex;
template<class F> void family_log(F &&write) {
  std::lock_guard lock(family_log_mutex);
  write(std::cerr);
}

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
  if (!projections.empty()) family_log([&](std::ostream &out) {
    out << "original_quantized_projections=" << projections.size()
              << " packed_contractions=" << packed
              << " decomposed_projections=" << projections.size() - packed
              << " graph_workspace_bytes=" << graph_workspace_bytes << '\n';
  });
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
  if (!projection.projection.weight) return recipe.scores(selected_hidden, std::string(recipe.geometry.head_role()));
  if (projection.indexed_rows) {
    const auto rows = recipe.spec(selected_hidden).dimensions()[0];
    if (!rows.is_fixed() || rows.fixed_value() != 1)
      throw std::logic_error("indexed draft projection requires its cold single-request carrier");
    const auto weight = recipe.operand(*projection.projection.weight);
    const auto vocabulary = recipe.spec(weight).dimensions()[0];
    const auto hidden = recipe.spec(weight).dimensions()[1];
    const auto mapping = recipe.constant(*projection.original_ids);
    const auto count = recipe.spec(mapping).dimensions()[0];
    // One original output row is one expert with N=1. Only metadata views
    // and the exact BF16->F32 activation widening precede the existing op.
    const auto bank = recipe.reshape(weight, {vocabulary, f(1), hidden});
    const auto activation = recipe.reshape(recipe.cast(selected_hidden, Type::kFloat32),
        {rows, f(1), hidden});
    const auto ids = recipe.reshape(mapping, {rows, count});
    const auto scores = recipe.one(c::model_ops::grouped_matmul(), {bank, activation, ids},
        recipe.dense_spec(Type::kFloat32, {rows, count, f(1)}));
    return recipe.reshape(scores, {rows, count});
  }
  return recipe.emit_projection(selected_hidden, projection.projection, true);
}
V original_draft_id(Recipe &recipe, V compact_id, DraftProjectionView projection) {
  if (!projection.original_ids) return compact_id;
  return recipe.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
      {recipe.constant(*projection.original_ids), compact_id}, recipe.spec(compact_id));
}

DraftChainValues emit_draft_chain(Recipe&r,V tokens,V shifted,V positions,V frequencies,
    std::size_t rows,std::size_t steps,const c::BatchSchema&schema,DraftProjectionView head,
    std::span<const DraftChainControl>controls,V has_lag){
  if(!rows || rows>2 || steps<2 || steps>3 ||
      (!controls.empty() && (controls.size()!=steps || rows!=2 || !has_lag)))
    throw std::logic_error("draft chain requires its bounded original R1 rows and matching device controls");
  DraftChainValues result;result.stages.reserve(steps);
  auto ids=tokens;mtp::Projection prefix;mtp::KeyValue prefix_kv;
  V last_query_lengths;
  if(!controls.empty()){auto&literal=r.assembly.literal_at(GraphAssembly::primary(),dense(Type::kInt32,{1}));
    constexpr std::array<std::int32_t,1>one{1};literal.write<std::int32_t>(one);last_query_lengths=r.constant(literal);}
  std::vector<V>packet{r.copy(r.slice(tokens,0,f(static_cast<std::int64_t>(rows-1)),f(1)))};
  for(std::size_t i=0;i<steps;++i){
    const auto width=static_cast<std::int64_t>(rows+i),local_rows=static_cast<std::int64_t>(i?1:rows);
    const auto cumulative_positions=controls.empty()?r.copy(r.slice(positions,0,f(0),f(width))):controls[i].positions;
    const auto local_positions=!i?cumulative_positions:controls.empty()?
        r.copy(r.slice(cumulative_positions,0,f(width-1),f(1))):
        r.one(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},
            {cumulative_positions,controls[i].last_index},r.dense_spec(Type::kInt32,{f(1)}));
    auto projection=mtp::input_projection(r,ids,shifted);
    auto kv=mtp::project_kv(r,projection,local_positions,frequencies,mtp::Geometry{1,f(local_rows),f(local_rows)});
    if(i){
      const auto pack=[&](V previous,V next){
        std::vector<D>shape(r.spec(previous).dimensions().begin(),r.spec(previous).dimensions().end());shape[0]=f(width);
        const auto spec=r.dense_spec(r.spec(previous).data_type(),shape);
        const auto full=r.one(c::Concat{.axis=0},{previous,next},spec);
        if(controls.empty())return full;
        // At lag0 the old last physical row is padding. Move it behind the
        // newly projected row; only a contiguous active prefix can reach KV.
        const std::array compact{r.copy(r.slice(previous,0,f(0),f(width-2))),next,r.copy(r.slice(previous,0,f(0),f(1)))};
        const auto padded=r.concat(compact,0,spec);
        return r.one(c::Select{},{has_lag,full,padded},spec);
      };
      projection={f(width),pack(prefix.fused,projection.fused),pack(prefix.normalized,projection.normalized)};
      kv={f(width),pack(prefix_kv.key_rows,kv.key_rows),pack(prefix_kv.value_rows,kv.value_rows)};
    }
    DraftChainStage stage;stage.rows=static_cast<std::size_t>(width);
    stage.key=r.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kKey)));
    stage.value=r.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kValue)));
    if(controls.empty())stage.metadata=r.input(take(schema.metadata_description()));
    const auto selected=mtp::full_head(r,projection,kv,cumulative_positions,frequencies,
        {schema,stage.key,stage.value,stage.metadata,controls.empty()?V{}:controls[i].query_lengths,controls.empty()?V{}:controls[i].kv_lengths,last_query_lengths},
        f(width-1),mtp::Geometry{1,f(width),f(width)},
        [&](Recipe&recipe,V hidden){return draft_scores(recipe,hidden,head);},controls.empty()?V{}:controls[i].last_index,i!=0);
    stage.append_key=selected.append_nodes.key;stage.append_value=selected.append_nodes.value;
    stage.attention=selected.attention_node;result.stages.push_back(stage);result.hidden=selected.selected_hidden;
    const auto id=original_draft_id(r,r.op(c::Operation(c::TopK{.count=1}),{selected.logits},
        {r.dense_spec(Type::kFloat32,{f(1),f(1)}),r.dense_spec(Type::kInt32,{f(1),f(1)})})[1],head);
    ids=r.reshape(id,{f(1)});shifted=result.hidden;packet.push_back(ids);prefix=projection;prefix_kv=kv;
  }
  result.packet=r.copy(r.concat(packet,0,r.dense_spec(Type::kInt32,{f(static_cast<std::int64_t>(packet.size()))})));
  return result;
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
    family_log([&](std::ostream &out) { out << "cpu_plan known_workspace_host=" << domain.workspace_bytes
        << " workspace_host_limit=" << domain.workspace_budget_bytes
        << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_memory_cap=0\n"; });
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
  family_log([&](std::ostream &out) { out << "hybrid_plan host_constants=" << host_constants << " materialized_constant_bytes="
      << report.materialized_constant_bytes() << " activation_copy_bytes=" << report.copy_bytes()
      << " known_workspace_device=" << take(report.storage_domain(0)).workspace_bytes
      << " known_workspace_host=" << take(report.storage_domain(1)).workspace_bytes
      << " known_workspace_staging=" << take(report.storage_domain(2)).workspace_bytes
      << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_vram_cap=0\n"; });
  return facts;
}

FamilyBuildResult build_family(const FamilyBuildRequest &build_request,
    const FamilyModelFacts &model, const FamilyPreparedFacts &prepared,
    const SamplingConfiguration &configuration, const FamilyCompileContext &context) {
#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
  inference::testing::record_family_build();
#endif
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
  const auto hidden_row_bytes = cache::multiply(static_cast<std::size_t>(hidden), precision.activation_bytes());
  const bool device_continuation = role == GraphRole::kMtpContinuation;
  const bool continuous_successor = device_continuation && prepared.continuous_device_successor;
  const bool emit_device_decision = prepared.device_acceptance_successor || continuous_successor;
  const bool repair_draft = prepared.recursive_draft_repair;
  const auto chain_steps = prepared.draft_chain_steps;
  if (chain_steps && (requests != 1 || prefix_rows != chain_steps + 1 || chain_steps < 2 || chain_steps > 3 ||
      (role != GraphRole::kMtpRound && role != GraphRole::kMtpCatchupRound && !device_continuation) ||
      packing != RowPacking::kRectangular || selected_sampler.enabled || !selected_sampler.modifiers.empty()))
    throw std::runtime_error("compound MTP round requires its exact fixed greedy R1 family");
  if (repair_draft && (role != GraphRole::kVerify || requests != 1 || prefix_rows < 2 ||
      packing != RowPacking::kRectangular || selected_sampler.enabled || !selected_sampler.modifiers.empty()))
    throw std::runtime_error("fused draft repair requires the original greedy exact R1 verifier");
  if (device_continuation && (requests != 1 || prefix_rows != (chain_steps?chain_steps+1:2) ||
      packing != RowPacking::kRectangular || completion != TextCompletion::kEmitNext ||
      selected_sampler.enabled || !selected_sampler.modifiers.empty() ||
      std::ranges::any_of(build_request.chunks, [&](auto rows) { return rows != prefix_rows; })))
    throw std::runtime_error("device MTP continuation requires its bounded greedy R1 rectangular family");
  const bool successor_eligible = drafts_token(role) && requests == 1 &&
      packing == RowPacking::kRectangular && !selected_sampler.enabled &&
      selected_sampler.modifiers.empty() && (prepared.emit_successor || continuous_successor);
  if (emit_device_decision &&
      (!successor_eligible || vocabulary > (std::int64_t{1} << 24)))
    throw std::runtime_error("device successor decision requires the original greedy R1 K1 family and exact F32 token IDs");
  const auto ports_per_request = cache::add(76, cache::multiply(38, prefix_rows - 1));
  ExecutionPlanFacts plan_facts;
    auto result = std::make_unique<Family>();
    auto &definition = *result;
    definition.continuous_device_successor = continuous_successor;
    definition.recursive_draft_repair = repair_draft;
    definition.draft_chain_steps = chain_steps;
    definition.layers.resize(geometry.layers.size()); definition.attention_nodes.resize(target_kv_layers);
    definition.append_nodes.resize(draft_bank + 2 + 2 * chain_steps);
    definition.append_cache_inputs.resize(draft_bank + 2 + 2 * chain_steps);
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
          cache::add(hidden_row_bytes, cache::state_pool_padding(hidden_row_bytes))));
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
    const auto target_control_rows=device_continuation?D(take(r.builder.dimension_sum(t,f(2)))):D{};
    definition.position_port = device_continuation?r.input(r.dense_spec(Type::kInt32,
        {f(static_cast<std::int64_t>(prefix_rows)),target_control_rows})):
        requests == 1 ? r.input(r.dense_spec(Type::kInt32, {t})) : r.input(r.dense_spec(Type::kInt32, {batch, t}));
    V predecessor_prefix_index{}, predecessor_has_lag{}, target_positions = definition.position_port,
        target_query_lengths{},target_kv_lengths{};
    if (device_continuation) {
      definition.predecessor_prefix_port = r.input(dense(Type::kInt32, {1}));
      auto &choices = assembly.literal_at(assembly.primary(), dense(Type::kInt32, {static_cast<std::int64_t>(prefix_rows)}));
      std::vector<std::int32_t>indices(prefix_rows);for(std::size_t i=0;i<prefix_rows;++i)indices[i]=static_cast<std::int32_t>(i);
      choices.write<std::int32_t>(indices);
      predecessor_prefix_index = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {r.constant(choices), definition.predecessor_prefix_port}, r.dense_spec(Type::kInt32, {f(1)}));
      predecessor_has_lag = r.cast(r.cast(predecessor_prefix_index, Type::kFloat32), Type::kBool);
      const auto selected=r.reshape(r.one(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},
          {definition.position_port,predecessor_prefix_index},r.dense_spec(Type::kInt32,{f(1),target_control_rows})),{target_control_rows});
      target_positions=r.slice(selected,0,f(0),t);
      target_query_lengths=r.slice(selected,0,t,f(1));
      const auto kv_offset=D(take(r.builder.dimension_sum(t,f(1))));
      target_kv_lengths=r.slice(selected,0,kv_offset,f(1));
    }
    if ((requests != 1 || packed_prefill) && !verifies_prefix(role)) definition.last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
    if (packed_target) {
      definition.sequence_offsets_port = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(requests + 1)}));
      definition.history_select_port = r.input(dense(Type::kInt32,
          {static_cast<std::int64_t>(cache::multiply(cache::multiply(requests, prefix_rows), geometry.conv_kernel - 1))}));
    }
    if (packed_verify) definition.prefix_ids_port = r.input(dense(Type::kInt32,
        {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
    if (!device_continuation)
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
    V canonical_draft_positions{},canonical_query_lengths{},canonical_kv_lengths{};
    std::vector<DraftChainControl>device_chain_controls;
    if (repair_draft) {
      auto &port = definition.hidden_state;
      port.pool = r.input(take(hidden_schema.capacity_description()));
      port.source_ids = r.input(dense(Type::kInt32, {2}));
      const auto gathered = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {port.pool, port.source_ids}, r.dense_spec(precision.activation_type(), {f(2), f(hidden)}), &port.gather);
      definition.carry_input_port = r.slice(gathered, 0, f(0), f(1));
      definition.prior_input_port = r.slice(gathered, 0, f(1), f(1));
      definition.draft_position_port = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(prefix_rows)}));
      definition.catchup_token_port = r.input(dense(Type::kInt32, {1}));
      definition.repair_lag_port = r.input(dense(Type::kInt32, {1}));
    }
    if (has_mtp(role)) {
      auto &port = definition.hidden_state;
      const auto source_rows = f(static_cast<std::int64_t>(device_continuation?prefix_rows:
          cache::multiply(requests, draft_lag(role) + 1)));
      port.pool = r.input(take(hidden_schema.capacity_description()));
      port.source_ids = r.input(r.dense_spec(Type::kInt32, {source_rows}));
      const auto gathered = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {port.pool, port.source_ids}, r.dense_spec(precision.activation_type(), {source_rows, f(hidden)}), &port.gather);
      // One fixed pool and private IDs; the original ready leases still prove
      // exactly these rows. Batched/packed order is [all carries, all priors].
      const bool chronological = requests == 1 &&
          (role == GraphRole::kMtpCatchupRound || device_continuation) &&
          packing == RowPacking::kRectangular;
      // A K1 producer stores [prior, carry]. The lag1 R1 reader uses that same
      // real ID order, in ready and imported modes; only the graph interprets
      // the two chronological rows. No source ticket/ID is manufactured.
      definition.carry_input_port = r.slice(gathered, 0, chronological ? batch : f(0), batch);
      if (device_continuation)
        definition.carry_input_port = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
            {gathered, predecessor_prefix_index},
            r.dense_spec(precision.activation_type(), {batch, f(hidden)}));
      const auto draft_rows = chain_steps ? f(static_cast<std::int64_t>(chain_steps + (device_continuation?1:draft_lag(role)))) : is_mtp_prefill(role)
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
      const auto stage_control_columns=chain_steps?4*chain_steps+chain_steps*(chain_steps-1)/2:0;
      const auto draft_control_rows=device_continuation?D(take(r.builder.dimension_sum(draft_rows,
          f(static_cast<std::int64_t>(2+stage_control_columns))))):D{};
      definition.draft_position_port = r.input(device_continuation?r.dense_spec(Type::kInt32,
          {f(static_cast<std::int64_t>(prefix_rows)),draft_control_rows}):
          requests == 1 ? r.dense_spec(Type::kInt32, {draft_rows}) : r.dense_spec(Type::kInt32, {batch, draft_rows}));
      V draft_positions = definition.draft_position_port,draft_query_lengths{},draft_kv_lengths{};
      if (device_continuation) {
        const auto selected=r.reshape(r.one(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},
            {definition.draft_position_port,predecessor_prefix_index},r.dense_spec(Type::kInt32,{f(1),draft_control_rows})),{draft_control_rows});
        draft_positions=r.slice(selected,0,f(0),draft_rows);
        draft_query_lengths=r.slice(selected,0,draft_rows,f(1));
        const auto kv_offset=D(take(r.builder.dimension_sum(draft_rows,f(1))));
        draft_kv_lengths=r.slice(selected,0,kv_offset,f(1));
        if(chain_steps){
          std::int64_t offset=static_cast<std::int64_t>(chain_steps+3);
          for(std::size_t i=0;i<chain_steps;++i){
            const auto rows=static_cast<std::int64_t>(i+2);
            const auto integer=[&](std::int32_t value){auto&literal=assembly.literal_at(assembly.primary(),dense(Type::kInt32,{1}));
              literal.write<std::int32_t>(std::span(&value,1));return r.constant(literal);};
            const auto last=r.one(c::Select{},{predecessor_has_lag,integer(static_cast<std::int32_t>(i+1)),integer(static_cast<std::int32_t>(i))},
                r.dense_spec(Type::kInt32,{f(1)}));
            device_chain_controls.push_back({r.slice(selected,0,f(offset),f(rows)),r.slice(selected,0,f(offset+rows),f(1)),
                r.slice(selected,0,f(offset+rows+1),f(1)),last});offset+=rows+2;
          }
        }
      }
      canonical_draft_positions=draft_positions;canonical_query_lengths=draft_query_lengths;canonical_kv_lengths=draft_kv_lengths;
      if (draft_lag(role)) {
        if(device_continuation){
          std::vector<std::int32_t>indices(prefix_rows);for(std::size_t i=1;i<prefix_rows;++i)indices[i]=static_cast<std::int32_t>(i-1);
          if(std::ranges::all_of(indices,[&](auto value){return value==indices.front();}))
            definition.prior_input_port=r.slice(gathered,0,f(indices.front()),batch);
          else {
            auto&prior_indices=assembly.literal_at(assembly.primary(),dense(Type::kInt32,{static_cast<std::int64_t>(prefix_rows)}));
            prior_indices.write<std::int32_t>(indices);
            const auto selected=r.one(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},
                {r.constant(prior_indices),predecessor_prefix_index},r.dense_spec(Type::kInt32,{f(1)}));
            definition.prior_input_port=r.one(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},
                {gathered,selected},r.dense_spec(precision.activation_type(),{batch,f(hidden)}));
          }
        }else definition.prior_input_port = r.slice(gathered, 0, chronological ? f(0) : batch, batch);
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
        } else if (device_continuation) {
          // Both physical rows exist. The typed attention/Append choice owns
          // whether one or two are active. Pad the inactive row with valid
          // token/hidden values before any embedding or projection reads it.
          const auto lag1_ids = r.one(c::Concat{.axis = 0},
              {definition.catchup_token_port, ids}, r.dense_spec(Type::kInt32, {f(2)}));
          const auto lag0_ids = r.one(c::Concat{.axis = 0}, {ids, ids},
              r.dense_spec(Type::kInt32, {f(2)}));
          ids = r.one(c::Select{}, {predecessor_has_lag, lag1_ids, lag0_ids},
              r.dense_spec(Type::kInt32, {f(2)}));
          const auto lag1_hidden = r.one(c::Concat{.axis = 0},
              {definition.prior_input_port, hidden_rows},
              r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
          const auto lag0_hidden = r.one(c::Concat{.axis = 0}, {hidden_rows, hidden_rows},
              r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
          hidden_rows = r.one(c::Select{}, {predecessor_has_lag, lag1_hidden, lag0_hidden},
              r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
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
        if (chain_steps) {
          auto chain = emit_draft_chain(r, ids, hidden_rows, draft_positions, frequencies,
              device_continuation?2:1+draft_lag(role), chain_steps, attention_schema, prepared.draft_projection,
              device_chain_controls,predecessor_has_lag);
          target_tokens = chain.packet; definition.draft_chain = std::move(chain.stages);
          definition.draft_id_port = r.copy(r.reshape(r.slice(target_tokens, 0, f(1),
              f(static_cast<std::int64_t>(chain_steps))), {f(static_cast<std::int64_t>(chain_steps)), f(1)}));
          r.publish(definition.draft_id_port);
          for (std::size_t i = 0; i < chain_steps; ++i) {
            const auto bank = draft_bank + 2 + 2 * i; const auto &stage = definition.draft_chain[i];
            definition.append_nodes[bank] = stage.append_key; definition.append_nodes[bank + 1] = stage.append_value;
            definition.append_cache_inputs[bank] = stage.key; definition.append_cache_inputs[bank + 1] = stage.value;
          }
        } else {
        const auto projection = mtp::input_projection(r, ids, hidden_rows);
        const mtp::Geometry row_geometry{requests, draft_rows, packed_draft_rows};
        const auto kv = mtp::project_kv(r, projection, draft_positions, frequencies, row_geometry);
        mtp::AppendNodes append;
        if (drafts_token(role)) {
          definition.draft_key_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
          definition.draft_value_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
          if (!device_continuation)
            definition.draft_metadata_port = r.input(take(attention_schema.metadata_description()));
          definition.append_cache_inputs[draft_bank] = definition.draft_key_port;
          definition.append_cache_inputs[draft_bank + 1] = definition.draft_value_port;
          const auto head = mtp::full_head(r, projection, kv, draft_positions, frequencies,
              {attention_schema, definition.draft_key_port, definition.draft_value_port,
               definition.draft_metadata_port, draft_query_lengths, draft_kv_lengths},
              f(static_cast<std::int64_t>(draft_lag(role))), row_geometry,
              [&](Recipe &recipe, V selected) { return qwen35::draft_scores(recipe, selected, prepared.draft_projection); },
              device_continuation ? predecessor_prefix_index : definition.draft_last_ids_port);
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
    }
    // Writer-only draft maintenance has no attention data edge into target.
    // Order its real cache effects before target's first append as well.
    const auto target = emit_target_graph(r, {
        .token_ids = target_tokens, .positions = target_positions,
        .attention_metadata = definition.metadata_port, .frequencies = frequencies,
        .sequence_offsets = definition.sequence_offsets_port, .history_select = definition.history_select_port,
        .requests = requests, .prefix_rows = prefix_rows, .packing = packing,
        .state_slots = prepared.state_slots, .recurrent_descriptions = state_schema,
        .plain_attention_schema = attention_schema, .target_attention_schemas = attention_schemas,
        .cpu_ffn_layers = model.cpu_ffn_layers,
        .preceding_append = has_mtp(role) && !is_mtp_prefill(role) && !chain_steps
            ? definition.append_nodes[draft_bank + 1] : g::NodeId{},
        .has_preceding_append = has_mtp(role) && !is_mtp_prefill(role) && !chain_steps,
        .predecessor_prefix_index = predecessor_prefix_index,
        .query_lengths=target_query_lengths,.kv_lengths=target_kv_lengths},
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
    if (chain_steps) {
      const auto previous = r.copy(r.slice(normalized_last, 0, f(0), f(static_cast<std::int64_t>(chain_steps - 1))));
      std::vector<V> hidden_rows;
      if (draft_lag(role)) hidden_rows.push_back(definition.prior_input_port);
      hidden_rows.push_back(definition.carry_input_port); hidden_rows.push_back(previous);
      const auto width = f(static_cast<std::int64_t>(chain_steps + (device_continuation?1:draft_lag(role))));
      auto canonical_hidden = r.concat(hidden_rows, 0,
          r.dense_spec(precision.activation_type(), {width, f(hidden)}));
      const auto prefix_ids = r.copy(r.slice(target_tokens, 0, f(0), f(static_cast<std::int64_t>(chain_steps))));
      auto canonical_ids = draft_lag(role) ? r.one(c::Concat{.axis = 0},
          {definition.catchup_token_port, prefix_ids}, r.dense_spec(Type::kInt32, {width})) : prefix_ids;
      if(device_continuation){
        const auto lag0_hidden=r.one(c::Concat{.axis=0},
            {definition.carry_input_port,previous,definition.carry_input_port},r.dense_spec(precision.activation_type(),{width,f(hidden)}));
        const auto lag0_ids=r.one(c::Concat{.axis=0},{prefix_ids,definition.token_port},r.dense_spec(Type::kInt32,{width}));
        canonical_hidden=r.one(c::Select{},{predecessor_has_lag,canonical_hidden,lag0_hidden},r.dense_spec(precision.activation_type(),{width,f(hidden)}));
        canonical_ids=r.one(c::Select{},{predecessor_has_lag,canonical_ids,lag0_ids},r.dense_spec(Type::kInt32,{width}));
      }
      const auto projection = mtp::input_projection(r, canonical_ids, canonical_hidden);
      const auto kv = mtp::project_kv(r, projection, canonical_draft_positions, frequencies,
          mtp::Geometry{1, width, width});
      const auto append = mtp::append_kv(r, attention_schema, kv,canonical_query_lengths,canonical_kv_lengths);
      if (has_prior_append) check(r.builder.depends_on(append.key, prior_append));
      definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      prior_append = append.value; has_prior_append = true;
    }
    if (repair_draft) {
      const auto rows = static_cast<std::int64_t>(prefix_rows);
      const auto previous = r.copy(r.slice(normalized_last, 0, f(0), f(rows - 1)));
      const auto lag0_hidden = r.one(c::Concat{.axis = 0}, {definition.carry_input_port, previous},
          r.dense_spec(precision.activation_type(), {f(rows), f(hidden)}));
      const auto lag1_prefix = r.one(c::Concat{.axis = 0},
          {definition.prior_input_port, definition.carry_input_port},
          r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
      const auto lag1_hidden = rows == 2 ? lag1_prefix : r.one(c::Concat{.axis = 0},
          {lag1_prefix, r.copy(r.slice(normalized_last, 0, f(0), f(rows - 2)))},
          r.dense_spec(precision.activation_type(), {f(rows), f(hidden)}));
      const auto lag1_ids = r.one(c::Concat{.axis = 0},
          {definition.catchup_token_port, r.copy(r.slice(target_tokens, 0, f(0), f(rows - 1)))},
          r.dense_spec(Type::kInt32, {f(rows)}));
      const auto lag = r.cast(r.cast(definition.repair_lag_port, Type::kFloat32), Type::kBool);
      const auto ids = r.one(c::Select{}, {lag, lag1_ids, target_tokens}, r.dense_spec(Type::kInt32, {f(rows)}));
      const auto hidden_rows = r.one(c::Select{}, {lag, lag1_hidden, lag0_hidden},
          r.dense_spec(precision.activation_type(), {f(rows), f(hidden)}));
      const auto projection = mtp::input_projection(r, ids, hidden_rows);
      const auto kv = mtp::project_kv(r, projection, definition.draft_position_port, frequencies,
          mtp::Geometry{1, f(rows), f(rows)});
      const auto append = mtp::append_kv(r, attention_schema, kv);
      if (has_prior_append) check(r.builder.depends_on(append.key, prior_append));
      definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      prior_append = append.value; has_prior_append = true;
    }
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
      if (has_mtp(role) || repair_draft) definition.state_sources.push_back({port.store, port.gather});
    }
    definition.hidden_port = last;
    // Private View outputs cannot lend an independent storage lease to a
    // transfer. Materialize only ordinary CUDA host-consumed small results;
    // their original types/values stay exact and Graph owns every Copy.
    const bool caller_small_results = OrdinaryIoCapabilities{
        .backend = execution.kind(), .role = role, .completion = completion,
        .packing = packing, .cpu_assist = execution.has_cpu_assist(), .diagnostic = diagnostic}.caller_small_results();
    const auto publish_valid = [&] {
      if (caller_small_results) definition.sampling_valid_port = r.copy(definition.sampling_valid_port);
      r.publish(definition.sampling_valid_port);
    };
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
        publish_valid();
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
        publish_valid();
      } else {
        std::vector<V> ids, valid, accepted;
        for (std::size_t request = 0; request < requests; ++request) {
          const auto p = sampling::processed_distribution(r, sampling::process_logits(r,
              sampling_row(definition.logits_port, drafts_token(role) ? request * 2 : request), logits_inputs[request], selected_sampler.modifiers.history), selected_sampler.policy,
              drafts_token(role) ? sampling::DistributionConsumer::kFullVocabulary : sampling::DistributionConsumer::kCategoricalOnly);
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
        publish_valid();
        if (drafts_token(role)) {
          definition.sampling_accept_port = r.host_predicate(r.concat(accepted, 0, r.dense_spec(Type::kBool, {batch})));
          r.publish(definition.sampling_accept_port);
        }
      }
    }
    r.publish(definition.hidden_port);
    if (completion == TextCompletion::kEmitNext) {
      // A packed MTP token may be a reshape View of the native TopK result.
      // Keep its published bytes outside reusable Invocation scratch, exactly
      // as ordinary caller-owned small results, without another sampler draw.
      if ((caller_small_results && definition.sampling_valid_port) || (packed_prefill && is_mtp_prefill(role)))
        definition.id_port = r.copy(definition.id_port);
      r.publish(definition.logits_port); r.publish(definition.id_port);
    }
    if (successor_eligible) {
      // Two complete, independently owned I32[1] outputs. OutputSource cannot
      // grant a fabricated subview of the two-row target-ID writer to B.
      definition.successor_token_port = r.copy(r.reshape(r.slice(definition.id_port, 0,
          f(static_cast<std::int64_t>(prefix_rows - 1)), f(1)), {f(1)}));
      definition.successor_catchup_port = r.copy(r.reshape(chain_steps ? r.slice(definition.draft_id_port, 0,
          f(static_cast<std::int64_t>(chain_steps - 1)), f(1)) : definition.draft_id_port, {f(1)}));
      r.publish(definition.successor_token_port); r.publish(definition.successor_catchup_port);
      if (emit_device_decision) {
        const auto integer = [&](std::int32_t value) {
          auto &literal = assembly.literal_at(assembly.primary(), dense(Type::kInt32, {1}));
          literal.write<std::int32_t>(std::span(&value, 1));
          return r.constant(literal);
        };
        const auto select=[&](V predicate,V yes,V no){return r.one(c::Select{},
            {predicate,yes,no},r.dense_spec(Type::kInt32,{f(1)}));};
        const auto proposals=prefix_rows-1;
        std::vector<V>integers,targets,drafts;
        for(std::size_t i=0;i<=prefix_rows;++i)integers.push_back(integer(static_cast<std::int32_t>(i)));
        for(std::size_t i=0;i<prefix_rows;++i)
          targets.push_back(r.reshape(r.slice(definition.id_port,0,f(static_cast<std::int64_t>(i)),f(1)),{f(1)}));
        for(std::size_t i=0;i<proposals;++i)
          drafts.push_back(r.reshape(proposals==1?definition.draft_id_port:
              r.slice(definition.draft_id_port,0,f(static_cast<std::int64_t>(i)),f(1)),{f(1)}));
        const auto absent=integer(-1);
        auto selected=integers[proposals],count=integers[prefix_rows],pending=targets.back(),
            lag=integers[1],catchup=drafts.back();
        // Propagate each complete prefix directly. A rejected one-proposal
        // round reduces to the original Selects without table/Gather kernels;
        // longer chains retain the same first-forward-mismatch semantics.
        // IDs are exactly representable in F32. Stop/quota remain host policy.
        for(std::size_t i=proposals;i-- >0;){
          const auto mismatch=r.cast(r.binary(c::BinaryKind::kSubtract,r.cast(targets[i],Type::kFloat32),r.cast(drafts[i],Type::kFloat32)),Type::kBool);
          selected=select(mismatch,integers[i],selected);
          count=select(mismatch,integers[i+1],count);
          pending=select(mismatch,targets[i],pending);
          lag=select(mismatch,integers[i?1:0],lag);
          catchup=select(mismatch,i?drafts[i-1]:absent,catchup);
        }
        definition.successor_accepted_count_port=r.copy(count);
        definition.successor_prefix_index_port=r.copy(selected);
        definition.successor_pending_port=r.copy(pending);
        definition.successor_lag_port=r.copy(lag);
        definition.successor_selected_catchup_port=r.copy(catchup);
        r.publish(definition.successor_accepted_count_port);
        r.publish(definition.successor_prefix_index_port);
        r.publish(definition.successor_pending_port);
        r.publish(definition.successor_lag_port);
        r.publish(definition.successor_selected_catchup_port);
      }
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
    // Search and admission counters cover the whole finite family. Give every
    // explicitly requested child the library's per-graph cold search budget;
    // storage and device-resource limits remain independent.
    const auto per_child = [&](std::size_t per_graph) {
      if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph)
        throw std::runtime_error("finite text family search budget overflows");
      return per_graph * variants.size();
    };
    const g::CompileOptions defaults{};
    compile_options.max_operation_proposals = per_child(defaults.max_operation_proposals);
    compile_options.max_placement_attempts = per_child(defaults.max_placement_attempts);
    compile_options.max_route_search_states = per_child(defaults.max_route_search_states);
    compile_options.max_region_proposals = per_child(defaults.max_region_proposals);
    compile_options.max_region_proposal_nodes = per_child(defaults.max_region_proposal_nodes);
    family_log([&](std::ostream &out) { out << "compile full text finite family requests=" << requests << " variants=" << variants.size()
              << " max_query_rows=" << max_query_rows << " nodes=" << definition.graph.node_count()
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " max_region_proposals=" << compile_options.max_region_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts << '\n'; });
    compile_options.resource_budget = execution_budget;
#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
    inference::testing::record_compile();
#endif
    auto compiled = execution.compile(definition.graph, assembly.constants(), assembly.options(compile_options));
    if (!compiled && (packed_target || compact_draft) && compiled.error().IsUnimplemented() && !compiled.error().has_native_error())
      return {}; // Cold capability only; no execution/device error is a fallback.
    auto compiled_family = take(std::move(compiled));
    const auto &family_report = compiled_family.report;
    const auto storage = family_report.total_compiled_storage();
    family_log([&](std::ostream &out) { out << "compiled text finite family variants=" << family_report.exact_variant_count()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " operation_proposals=" << family_report.total_search().operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts
              << " placement_attempts=" << family_report.total_search().placement_attempts
              << " cold_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n'; });
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
