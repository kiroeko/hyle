// Original Qwen auxiliary cold construction. Direct calls only; no stored
// callback, model admission or second Invocation/transaction/retirement owner.
#include "auxiliary_programs.h"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

#include "../../inference/backend.h"
#include "../../inference/mtp_acceptance_graph.h"
#include "../../inference/output_pool_capacity.h"

namespace qwen35 {
namespace {
Tensor auxiliary_tensor(const c::ResourceBudget &budget, mem::MemoryResourceRef resource,
                        c::TensorDesc description, bool caller = true) {
  return Tensor(resource, description, caller, budget);
}
} // namespace

AuxiliaryProgramResult<DraftProgram> build_draft_program(
    const DraftProgramRequest &build_request, const FamilyModelFacts &model,
    const DraftProgramFacts &draft_facts, const SamplingConfiguration &configuration,
    const AuxiliaryCompileContext &context) {
  auto &execution = context.execution;
  const auto &allocation_budget = context.allocation_budget;
  const auto &execution_budget = context.execution_budget;
  const auto &geometry = model.geometry;
  const auto &precision = model.precision;
  auto &weights = model.weights;
  const auto &projections = model.projections;
  const auto &weight_meanings = model.weight_meanings;
  const auto &selected_sampler = configuration;
  const auto &cache_budget = context.cache_budget;
  const auto &attention_schema = draft_facts.attention_schema;
  const auto rows = build_request.rows, selected_index = build_request.sampling_index;
  const auto head = build_request.head;
  const auto hidden = geometry.hidden;
    auto prepared = std::make_unique<DraftProgram>(); auto &program = *prepared;
    program.rows = rows; program.head = head; program.sampling_index = selected_index;
    program.replay_mode = build_request.replay_mode;
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, rows, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(rows));
    program.tokens = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.positions = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    V shifted;
    g::slots::Schema hidden_schema;
    if (head) {
      hidden_schema = take(g::slots::Schema::create(dense(precision.activation_type(), {1, hidden}),
          draft_facts.hidden_slots,
          cache::add(geometry.hidden_bytes(), cache::state_pool_padding(geometry.hidden_bytes()))));
      auto &port = program.hidden_state;
      port.pool = r.input(take(hidden_schema.capacity_description()));
      port.source_ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
      port.destination_ids = r.input(dense(Type::kInt32, {1}));
      shifted = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {port.pool, port.source_ids}, r.dense_spec(precision.activation_type(), {count, f(hidden)}), &port.gather);
    } else {
      // This maintenance graph only reads hidden and writes draft KV. Keep
      // its original ready-leaf Commands path; no dummy Store or pool reader.
      for (std::size_t row = 0; row < rows; ++row) {
        program.hidden_inputs.push_back(r.input(dense(precision.activation_type(), {1, hidden})));
      }
      shifted = r.concat(program.hidden_inputs, 0, r.dense_spec(precision.activation_type(), {count, f(hidden)}));
    }
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    const mtp::Geometry row_geometry{1, count, count};
    const auto projection = mtp::input_projection(r, program.tokens, shifted);
    const auto kv = mtp::project_kv(r, projection, program.positions, frequencies, row_geometry);
    if (head) {
      program.key = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
      program.value = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
      program.metadata = r.input(take(attention_schema.metadata_description()));
      const auto result = mtp::full_head(r, projection, kv, program.positions, frequencies,
          {attention_schema, program.key, program.value, program.metadata}, f(static_cast<std::int64_t>(rows - 1)), row_geometry,
          [&](Recipe &recipe, V selected) { return draft_scores(recipe, selected, draft_facts.projection); });
      program.append = result.append_nodes; program.attention = result.attention_node;
      program.hidden = result.selected_hidden;
      const auto inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
      if (inputs.bias) program.bias = *inputs.bias;
      if (inputs.allowed) program.allowed = *inputs.allowed;
      if (inputs.counts) program.counts = *inputs.counts;
      const auto processed = sampling::process_logits(r, result.logits, inputs, selected_sampler.modifiers.history);
      if (!selected_sampler.enabled) {
        if (selected_sampler.modifiers.empty()) program.id = original_draft_id(r,
            r.op(c::Operation(c::TopK{.count = 1}), {result.logits},
                {r.dense_spec(Type::kFloat32, {f(1), f(1)}), r.dense_spec(Type::kInt32, {f(1), f(1)})})[1], draft_facts.projection);
        else if (build_request.minimum_probability == 0) {
          const auto selection = sampling::processed_selection(r, processed, std::optional<V>{}, selected_sampler.policy);
          program.id = selection.id; program.valid = r.host_predicate(*selection.valid);
        }
      }
      if (selected_sampler.enabled || build_request.minimum_probability > 0) {
        const auto distribution = sampling::processed_distribution(r, processed,
            selected_sampler.enabled ? selected_sampler.policy : sampling::Policy{});
        program.log_probabilities = distribution.log_probabilities; program.valid = r.host_predicate(distribution.valid);
        if (selected_sampler.enabled) {
          program.identity = r.input(dense(Type::kUInt32, {1, 4}));
          program.id = r.one(c::sampling_ops::categorical(), {program.log_probabilities, program.identity});
          r.publish(program.log_probabilities);
        } else if (!selected_sampler.modifiers.empty()) {
          // Threshold already requires this one distribution's complete
          // validity. Reuse it for a safe greedy ID instead of validating the
          // same processed row a second time.
          const auto safe = r.one(c::Select{}, {distribution.valid, processed.logits, r.scalar(0)});
          program.id = r.op(c::Operation(c::TopK{.count = 1}), {safe})[1];
        }
        if (build_request.minimum_probability > 0) {
          const auto maximum = r.copy(r.reshape(r.one(c::Reduce{.kind = c::ReductionKind::kMaximum,
              .accumulation_type = Type::kFloat32}, {program.log_probabilities}), {f(1)}));
          program.probability = r.unary(c::UnaryKind::kExp, maximum); r.publish(program.probability);
        }
      }
      if (inputs.counts) {
        program.next_counts = sampling::advance_history_counts(r, *inputs.counts, r.reshape(program.id, {f(1)}));
        r.publish(program.next_counts);
      }
      if (program.valid) r.publish(program.valid);
      r.op(c::Operation::custom(g::slots::Store{hidden_schema}),
          {program.hidden, program.hidden_state.destination_ids}, {}, &program.hidden_state.store);
      check(r.builder.depends_on(program.hidden_state.store, program.append.value));
      r.publish(program.id);
    } else program.append = mtp::append_kv(r, attention_schema, kv);
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, static_cast<std::int64_t>(rows - 1)}};
    const std::array variants{g::ExactShapeVariant{binding}};
    const auto hidden_store = program.hidden_state.store;
    const auto extra_stores = hidden_store.valid() ? std::span(&hidden_store, 1) : std::span<const g::NodeId>{};
    check(program.programs.prepare(take(execution.compile(program.graph, assembly.constants(),
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants(), .resource_budget = execution_budget}))), program.graph, variants,
        {{}, extra_stores}, take(execution_budget.metadata_allocator())));
    const auto &compiled_program = program.programs.child(0).compiled;
    check(program.carrier.prepare(compiled_program, execution.queue_bindings()));
    const auto plan = inspect_execution_plan(execution, model.cpu_ffn_layers, compiled_program.report, assembly.constants());
    program.literals = std::move(assembly.literals);
    program.token_values = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.position_values = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    if (head && selected_sampler.enabled)
      program.identity_values = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kUInt32, {1, 4}));
    check(program.io.prepare(compiled_program, head && selected_sampler.enabled ? 3 : 2,
        program.hidden_inputs.size() + bool(program.bias) + bool(program.allowed) + bool(program.counts),
        head ? 1 + bool(program.valid) + (build_request.minimum_probability > 0) : 0,
        head ? bool(selected_sampler.enabled) + bool(program.next_counts) : 0));
    program.hidden_bindings.reserve(program.hidden_inputs.size());
    for (const auto port : program.hidden_inputs) program.hidden_bindings.push_back(take(program.io.add_input(port)));
    if (program.bias) program.bias_input = take(program.io.add_input(program.bias));
    if (program.allowed) program.allowed_input = take(program.io.add_input(program.allowed));
    if (program.counts) program.counts_input = take(program.io.add_input(program.counts));
    if (head) {
      if (selected_sampler.enabled) program.probabilities_output = take(program.io.add_output(program.log_probabilities));
      if (program.next_counts) program.next_counts_output = take(program.io.add_output(program.next_counts));
    }
    for (const auto &output : program.programs.child(0).outputs()) {
      if (!output.caller_owned()) continue;
      const auto port = output.value;
      if (port == program.next_counts) {
        const auto &facts = program.io.output_facts(program.next_counts_output);
        const auto bytes = take(facts.description.required_bytes());
        const auto pool_capacity = take(inference::checked_output_pool_capacity(
            std::array{bytes}, 2, cache::state_pool_alignment));
        program.next_counts_pool = take(c::LeasePool::create(
            take(execution.resource_for(facts.storage)), cache::add(pool_capacity.backing_bytes, 1),
            {.max_blocks = pool_capacity.max_blocks, .resource_budget = execution_budget}));
      } else if (port == program.log_probabilities) {
        // One verifier cohort retains every member's q until target selection.
        // Keep the original per-member allowance within the existing budget.
        const auto blocks = cache::multiply(draft_facts.active_sequences, draft_facts.candidate_boundaries);
        const auto &facts = program.io.output_facts(program.probabilities_output);
        const auto bytes = take(facts.description.required_bytes());
        const auto pool_capacity = take(inference::checked_output_pool_capacity(
            std::array{bytes}, blocks, cache::state_pool_alignment));
        program.probability_pool = take(c::LeasePool::create(
            take(execution.resource_for(facts.storage)), cache::add(pool_capacity.backing_bytes, 1),
            {.max_blocks = pool_capacity.max_blocks, .resource_budget = execution_budget}));
      }
    }
    check(program.io.add_fixed(program.tokens, program.token_values.read()));
    check(program.io.add_fixed(program.positions, program.position_values.read()));
    if (head) {
      program.id_value = take(program.io.add_readback(program.id, execution.storage_domains(), allocation_budget));
      if (selected_sampler.enabled) check(program.io.add_fixed(program.identity, program.identity_values.read()));
      if (program.valid) program.valid_value = take(program.io.add_readback(program.valid, execution.storage_domains(), allocation_budget));
      if (build_request.minimum_probability > 0)
        program.probability_value = take(program.io.add_readback(program.probability, execution.storage_domains(), allocation_budget));
    }
    check(program.io.seal());
    if (head) {
      const std::array specs{inference::SlotTransactionSpec{program.hidden_state.store,
          rows, 1, {.resource_budget = cache_budget}}};
      check(program.hidden_transactions.prepare(specs));
    }
  return {std::move(prepared), plan};
}

AuxiliaryProgramResult<PrefixSamplingProgram> build_prefix_sampling(
    const PrefixSamplingRequest &build_request, const FamilyModelFacts &model,
    const SamplingConfiguration &configuration, const AuxiliaryCompileContext &context) {
  auto &execution = context.execution;
  const auto &allocation_budget = context.allocation_budget;
  const auto &execution_budget = context.execution_budget;
  const auto &geometry = model.geometry;
  const auto &precision = model.precision;
  auto &weights = model.weights;
  const auto &projections = model.projections;
  const auto &weight_meanings = model.weight_meanings;
  const auto &selected_sampler = configuration;
  const auto proposals = build_request.proposals, sampling_index = build_request.sampling_index;
  const auto conditional = build_request.conditional;
  const auto vocabulary = geometry.vocabulary;
    auto prepared = std::make_unique<PrefixSamplingProgram>(); auto &program = *prepared;
    program.proposals = proposals; program.sampling_index = sampling_index;
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(proposals + 1));
    program.logits = r.input(r.dense_spec(Type::kFloat32, {count, f(vocabulary)}));
    auto shared_policy = selected_sampler.modifiers;
    if (conditional) { shared_policy.bias = false; shared_policy.allowed = false; }
    const auto inputs = prepare_logits_inputs(r, shared_policy);
    std::vector<sampling::LogitsInputsFor<V>> conditional_ports;
    if (conditional) {
      conditional_ports.reserve(proposals + 1); program.conditional_inputs.resize(proposals + 1);
      const sampling::LogitsPolicy policy{.bias = selected_sampler.modifiers.bias, .allowed = selected_sampler.modifiers.allowed};
      for (auto &row : program.conditional_inputs) {
        auto ports = prepare_logits_inputs(r, policy);
        if (ports.bias) row.bias = *ports.bias;
        if (ports.allowed) row.allowed = *ports.allowed;
        conditional_ports.push_back(std::move(ports));
      }
    }
    if (inputs.bias) program.bias = *inputs.bias;
    if (inputs.allowed) program.allowed = *inputs.allowed;
    if (inputs.counts) program.counts = *inputs.counts;
    if (selected_sampler.enabled) {
      program.identities = r.input(r.dense_spec(Type::kUInt32, {count, f(4)}));
      if (proposals) program.ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
      auto assembled = sampling::speculative_prefix(r, program.logits, program.identities,
          program.ids, selected_sampler.policy, inputs, selected_sampler.modifiers.history,
          std::span<const sampling::LogitsInputsFor<V>>(conditional_ports));
      program.q = std::move(assembled.q); program.accepted = std::move(assembled.accepted);
      program.final_ids = std::move(assembled.final_ids); program.valid = std::move(assembled.valid);
    } else {
      if (!selected_sampler.modifiers.history.empty() && proposals)
        program.ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
      auto row_inputs = inputs;
      // A greedy verifier still needs the exact processed target of every row;
      // no proposal-q or RNG ports belong to this program.
      for (std::size_t row = 0; row <= proposals; ++row) {
        if (conditional) {
          row_inputs.bias = conditional_ports[row].bias; row_inputs.allowed = conditional_ports[row].allowed;
        }
        const auto raw = r.copy(r.slice(program.logits, 0, f(static_cast<std::int64_t>(row)), f(1)));
        const auto selection = sampling::processed_selection(r,
            sampling::process_logits(r, raw, row_inputs, selected_sampler.modifiers.history), std::optional<V>{}, selected_sampler.policy);
        program.final_ids.push_back(r.reshape(selection.id, {f(1)})); program.valid.push_back(*selection.valid);
        if (!selected_sampler.modifiers.history.empty() && row < proposals)
          row_inputs.counts = sampling::advance_history_counts(r, *row_inputs.counts,
              r.copy(r.slice(program.ids, 0, f(static_cast<std::int64_t>(row)), f(1))));
      }
    }
    if (selected_sampler.enabled) {
      if (proposals) program.continuation = r.input(dense(Type::kFloat32,
          {static_cast<std::int64_t>(proposals)}));
      program.packet = inference::mtp::sampled_prefix_packet(r, program.ids, program.continuation,
          std::span<const V>(program.accepted), std::span<const V>(program.final_ids), std::span<const V>(program.valid));
    } else {
      program.packet = inference::mtp::validated_ids_packet(r,
          std::span<const V>(program.final_ids), std::span<const V>(program.valid));
    }
    r.publish(program.packet);
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, 0}};
    const std::array variants{g::ExactShapeVariant{binding}};
    check(program.programs.prepare(take(execution.compile(program.graph, assembly.constants(),
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants(), .resource_budget = execution_budget}))), program.graph, variants));
    const auto &compiled_program = program.programs.child(0).compiled;
    check(program.carrier.prepare(compiled_program, execution.queue_bindings()));
    const auto plan = inspect_execution_plan(execution, model.cpu_ffn_layers, compiled_program.report, assembly.constants());
    program.literals = std::move(assembly.literals);
    if (program.identities) {
      program.identity_values = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kUInt32, {static_cast<std::int64_t>(proposals + 1), 4}));
    }
    if (program.ids) program.id_values = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
    if (program.continuation) {
      program.continuation_values = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kFloat32,
          {static_cast<std::int64_t>(proposals)}));
      program.continuation_host.resize(proposals);
    }
    program.packet_host.resize(selected_sampler.enabled ? proposals + 5 : proposals + 2);
    check(program.io.prepare(compiled_program, bool(program.identities) + bool(program.ids) + bool(program.continuation),
        cache::add(1 + program.q.size() + bool(program.bias) + bool(program.allowed) + bool(program.counts),
            cache::multiply(program.conditional_inputs.size(), std::size_t(selected_sampler.modifiers.bias) + std::size_t(selected_sampler.modifiers.allowed))),
        1, 0));
    if (program.identities) check(program.io.add_fixed(program.identities, program.identity_values.read()));
    if (program.ids) check(program.io.add_fixed(program.ids, program.id_values.read()));
    if (program.continuation) check(program.io.add_fixed(program.continuation, program.continuation_values.read()));
    program.packet_value = take(program.io.add_readback(program.packet, execution.storage_domains(), allocation_budget));
    if (program.bias) program.bias_input = take(program.io.add_input(program.bias));
    if (program.allowed) program.allowed_input = take(program.io.add_input(program.allowed));
    if (program.counts) program.counts_input = take(program.io.add_input(program.counts));
    for (auto &row : program.conditional_inputs) {
      if (row.bias) row.bias_input = take(program.io.add_input(row.bias));
      if (row.allowed) row.allowed_input = take(program.io.add_input(row.allowed));
    }
    program.q_inputs.reserve(program.q.size());
    for (const auto port : program.q) program.q_inputs.push_back(take(program.io.add_input(port)));
    program.logits_input = take(program.io.add_input(program.logits));
    check(program.io.seal());
  return {std::move(prepared), plan};
}

AuxiliaryProgramResult<SamplingFrontier> build_sampling_frontier(
    const FamilyModelFacts &model, const SamplingConfiguration &configuration,
    const AuxiliaryCompileContext &context) {
  auto &execution = context.execution;
  const auto &allocation_budget = context.allocation_budget;
  const auto &execution_budget = context.execution_budget;
  const auto &geometry = model.geometry;
  const auto &precision = model.precision;
  auto &weights = model.weights;
  const auto &projections = model.projections;
  const auto &weight_meanings = model.weight_meanings;
  const auto &selected_sampler = configuration;
  const auto hidden = geometry.hidden;
        auto prepared = std::make_unique<SamplingFrontier>();
        GraphAssembly assembly(execution, precision, allocation_budget);
        Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
        prepared->last_row = r.last_row;
        prepared->hidden_port = r.input(dense(precision.activation_type(), {1, hidden}));
        if (selected_sampler.enabled) prepared->identity_port = r.input(dense(Type::kUInt32, {1, 4}));
        const auto inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
        if (inputs.bias) prepared->bias_port = *inputs.bias;
        if (inputs.allowed) prepared->allowed_port = *inputs.allowed;
        if (inputs.counts) prepared->counts_port = *inputs.counts;
        const auto selection = sampling::ordinary_selection(r,
            r.scores(prepared->hidden_port, std::string(geometry.head_role())),
            selected_sampler.enabled ? std::optional<V>(prepared->identity_port) : std::nullopt,
            selected_sampler.policy, inputs, selected_sampler.modifiers.history);
        prepared->id_port = selection.id;
        if (selection.valid) prepared->valid_port = r.host_predicate(*selection.valid);
        r.publish(prepared->id_port); if (prepared->valid_port) r.publish(prepared->valid_port);
        prepared->graph = take(std::move(r.builder).finish());
        // The frontier is a fixed [1,H] projection with no symbolic model rows.
        const std::array shape{g::ShapeBinding{r.last_row, 0}};
        const std::array variants{g::ExactShapeVariant{shape}};
        check(prepared->programs.prepare(take(execution.compile(prepared->graph, assembly.constants(),
            assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
             .constants = assembly.constants(), .resource_budget = execution_budget}))), prepared->graph, variants));
        const auto &compiled_program = prepared->programs.child(0).compiled;
        check(prepared->carrier.prepare(compiled_program, execution.queue_bindings()));
        const auto plan = inspect_execution_plan(execution, model.cpu_ffn_layers, compiled_program.report, assembly.constants());
        prepared->literals = std::move(assembly.literals);
        if (selected_sampler.enabled) prepared->identity = auxiliary_tensor(allocation_budget, execution.device(), dense(Type::kUInt32, {1, 4}));
        check(prepared->io.prepare(compiled_program, bool(prepared->identity_port),
            1 + bool(prepared->bias_port) + bool(prepared->allowed_port) + bool(prepared->counts_port),
            1 + bool(prepared->valid_port), 0));
        if (prepared->identity_port) check(prepared->io.add_fixed(prepared->identity_port, prepared->identity.read()));
        prepared->id = take(prepared->io.add_readback(prepared->id_port, execution.storage_domains(), allocation_budget));
        if (prepared->valid_port) prepared->valid = take(prepared->io.add_readback(prepared->valid_port, execution.storage_domains(), allocation_budget));
        prepared->hidden_input = take(prepared->io.add_input(prepared->hidden_port));
        if (prepared->bias_port) prepared->bias_input = take(prepared->io.add_input(prepared->bias_port));
        if (prepared->allowed_port) prepared->allowed_input = take(prepared->io.add_input(prepared->allowed_port));
        if (prepared->counts_port) prepared->counts_input = take(prepared->io.add_input(prepared->counts_port));
        check(prepared->io.seal());
  return {std::move(prepared), plan};
}

} // namespace qwen35
