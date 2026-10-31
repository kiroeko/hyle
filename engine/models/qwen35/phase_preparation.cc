// Original Qwen physical Phase construction, separate from model policy and
// per-step admission/submission. Only the existing ProgramSlots are prepared.
#include "phase.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <utility>

#include "../../inference/backend.h"

#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
#include "../../../tests/tools/inference_warm_cost_probe.h"
#endif

namespace qwen35 {
namespace {
Tensor phase_tensor(const c::ResourceBudget &budget, mem::MemoryResourceRef resource,
                   c::TensorDesc description, bool caller = true) {
  return Tensor(resource, description, caller, budget);
}
} // namespace

std::unique_ptr<Phase> prepare_phase(const PhaseSelection &selection,
    const PhasePrepareFacts &preparation, const PhaseEnvironment &environment) {
#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
  inference::testing::record_phase_prepare();
#endif
  const auto &definition = selection.family;
  const auto &selected_sampler = selection.sampling;
  auto &execution = environment.execution;
  const auto &geometry = environment.geometry;
  const auto state_schema = environment.recurrent_descriptions;
  const auto &allocation_budget = environment.allocation_budget;
  const auto &execution_budget = environment.execution_budget;
  const auto &cache_budget = environment.cache_budget;
  const auto tokens = preparation.tokens, physical_slots = preparation.physical_slots;
  const auto requests = definition.requests;
  const auto role = definition.role;
  const auto completion = definition.completion;
  const auto packing = definition.packing;
  const auto diagnostic = preparation.diagnostic, imported_sources = preparation.imported_sources;
  const bool packed_prefill = packs_prefill(packing), compact_draft = packs_draft(packing);
  const bool packed_verify = packs_verify(packing), packed_target = packs_target(packing);
    auto result = std::make_unique<Phase>();
    auto &phase = *result;
    phase.imported_sources = imported_sources;
    phase.query_rows = tokens;
    static_cast<FamilyPorts &>(phase) = static_cast<const FamilyPorts &>(definition);
    if (imported_sources) for (auto &source : phase.state_sources) source.mode = g::slots::SourceMode::kImported;
    const auto &planned_slot_uses = preparation.planned_uses;
    phase.attention_schema = definition.attention_schema;
    phase.child = &selection.child;
    phase.family = &definition;
    std::vector<inference::SlotTransactionSpec> transactions;
    phase.state_banks.reserve(phase.state_sources.size() + (phase.hidden_state.store.valid() ? 1u : 0u));
    transactions.reserve(phase.state_banks.capacity());
    const auto add_bank = [&](StateBank bank, const StatePorts &port, std::size_t sources) {
      const auto paired = std::ranges::find(phase.state_sources, port.store, &g::slots::SourceBinding::store);
      if (paired != phase.state_sources.end())
        bank.source_index = static_cast<std::size_t>(paired - phase.state_sources.begin());
      phase.state_banks.push_back(bank);
      transactions.push_back({port.store, imported_sources ? 0 : sources,
          cache::multiply(requests, phase.prefix_rows), {.resource_budget = cache_budget}});
    };
    for (std::size_t layer = 0; layer < state_schema.size(); ++layer) if (geometry.is_gdn(layer))
      for (const bool recurrent : {false, true}) {
        const auto &port = recurrent ? phase.layers[layer].recurrent : phase.layers[layer].history;
        add_bank({.layer = layer, .kind = recurrent ? StateBankKind::kRecurrent : StateBankKind::kHistory},
            port, requests);
      }
    if (phase.hidden_state.store.valid())
      add_bank({.kind = StateBankKind::kHidden}, phase.hidden_state,
          has_mtp(role) ? cache::multiply(requests, draft_lag(role) + 1) : 0);
    std::erase_if(phase.state_resource_ports, [&](V value) {
      return value.kind() != g::ValueKind::kInput && !phase.child->output(value).caller_owned();
    });
    if (imported_sources) {
      // This same ordinary-input inventory serves every episode of the Phase.
      // Prove its separation from the actual cold Source ports once; pending
      // Source permissions and selected reader access are still checked by
      // reader_binding/bind_sources for each original invocation below.
      const auto check_source_input = [&](V value) {
        if (std::ranges::find(phase.state_resource_ports, value) != phase.state_resource_ports.end())
          throw std::runtime_error("ordinary caller input overlaps its imported Source projection");
      };
      check_source_input(phase.token_port);
      if (phase.catchup_token_port) check_source_input(phase.catchup_token_port);
      for (const auto &source : phase.state_sources) {
        check_source_input(take(phase.state_graph().node_input(source.gather, 0)));
        check_source_input(take(phase.state_graph().node_input(source.gather, 1)));
      }
    }
    const auto t = static_cast<std::int64_t>(tokens);
    if (completion == TextCompletion::kEmitNext)
      phase.logits_facts = &phase.child->output(phase.logits_port);
    phase.hidden_facts = &phase.child->output(phase.hidden_port);
    if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
      phase.probe_facts[i] = &phase.child->output(phase.probe_ports[i]);
    check(phase.slots.prepare(phase.child->compiled, execution.queue_bindings(), physical_slots, transactions,
        [&](std::size_t slot, auto &program_slot) -> hyle::Status {
      auto &io = program_slot.data;
      io.batch_layouts.reserve(phase.attention_nodes.size() + drafts_token(role));
      for (auto node : phase.attention_nodes) io.batch_layouts.push_back({node, {}});
      if (drafts_token(role)) io.batch_layouts.push_back({phase.draft_attention_node, {}});
      program_slot.planned_uses = planned_slot_uses[slot];
      program_slot.replay_mode = preparation.replay_modes[slot];
      const auto token_count = has_mtp(role) && !is_mtp_prefill(role)
          ? static_cast<std::int64_t>(requests) : t * static_cast<std::int64_t>(requests);
      const auto token_description = role == GraphRole::kText
          ? dense(Type::kInt32, {token_count, 1}) : dense(Type::kInt32, {token_count});
      const auto position_description = requests == 1 ? dense(Type::kInt32, {t})
          : dense(Type::kInt32, {static_cast<std::int64_t>(requests), t});
      const bool sampling_identity = selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify;
      const bool last_ids = (requests != 1 || packed_prefill) && !verifies_prefix(role);
      const bool packet_inputs = role == GraphRole::kText && (packing == RowPacking::kRectangular || packed_prefill) &&
          !imported_sources && !diagnostic &&
          (execution.kind() == inference::BackendKind::kCpu || execution.kind() == inference::BackendKind::kCuda);
      if (packet_inputs) {
        InputPacket::MetadataDescriptions metadata;
        if (packed_prefill) {
          metadata.last_ids = dense(Type::kInt32, {static_cast<std::int64_t>(requests)});
          const auto count = cache::multiply(cache::multiply(requests, phase.prefix_rows), geometry.conv_kernel - 1);
          metadata.history_select = dense(Type::kInt32, {static_cast<std::int64_t>(count)});
        }
        if (sampling_identity) metadata.sampling_identities = phase.batched_text_sampling
            ? dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 4})
            : dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 3, 4});
        io.input_packet_preparation = io.input_packet.prepare(token_description, position_description,
            execution.device(), allocation_budget, execution_budget, metadata);
        // Packing is optional under a finite requested-byte cap. Release the
        // complete unpublished candidate before the old allocation path; keep
        // its typed refusal, and never mask a native or runtime write error.
        if (!io.input_packet_preparation.ok() && !io.input_packet.allocation_refused())
          return io.input_packet_preparation;
      }
      if (!io.input_packet.prepared()) {
        io.tokens = phase_tensor(allocation_budget, execution.device(), token_description);
        io.positions = phase_tensor(allocation_budget, execution.device(), position_description);
      }
      if (sampling_identity && !io.input_packet.sampling_identities().valid()) {
        io.sampling_identities = phase_tensor(allocation_budget, execution.device(), phase.batched_text_sampling
            ? dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 4})
            : dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 3, 4}));
      }
      if (phase.sampling_valid_port) {
        io.sampling_valid_values.resize(requests);
        bool float_predicates = phase.child->output(phase.sampling_valid_port).description.data_type() == Type::kFloat32;
        if (phase.sampling_accept_port) {
          io.sampling_accept_values.resize(requests);
          float_predicates = float_predicates || phase.child->output(phase.sampling_accept_port).description.data_type() == Type::kFloat32;
        }
        if (float_predicates) io.sampling_scratch.resize(requests);
      }
      if (has_mtp(role)) {
        const auto draft_rows = is_mtp_prefill(role) ? t + static_cast<std::int64_t>(draft_lag(role)) : static_cast<std::int64_t>(draft_lag(role) + 1);
        io.draft_positions = phase_tensor(allocation_budget, execution.device(), requests == 1 ? dense(Type::kInt32, {draft_rows})
            : dense(Type::kInt32, {static_cast<std::int64_t>(requests), draft_rows}));
        if (draft_lag(role)) io.catchup_token = phase_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        if (packs_prefill(definition.packing) || compact_draft) {
          const auto count = cache::multiply(requests, static_cast<std::size_t>(draft_rows));
          const auto description = dense(Type::kInt32, {static_cast<std::int64_t>(count)});
          io.draft_hidden_select = phase_tensor(allocation_budget, execution.device(), description);
          io.draft_hidden_indices.resize(count);
          io.draft_hidden_select.write<std::int32_t>(io.draft_hidden_indices);
          if (draft_lag(role)) {
            io.draft_token_select = phase_tensor(allocation_budget, execution.device(), description);
            io.draft_token_indices.resize(count);
            io.draft_token_select.write<std::int32_t>(io.draft_token_indices);
          }
        }
        if (compact_draft) io.draft_last_ids = phase_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        if (drafts_token(role)) {
          io.draft_ids.resize(requests);
        }
      }
      if (last_ids) {
        if (!io.input_packet.last_ids().valid()) {
          io.last_ids = phase_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
          std::vector<std::int32_t> initial_last_ids(requests);
          for (std::size_t j = 0; j < requests; ++j) initial_last_ids[j] = static_cast<std::int32_t>((j + 1) * tokens - 1);
          io.last_ids.write<std::int32_t>(initial_last_ids);
        }
        if (packed_prefill) io.last_id_values.resize(requests);
      }
      if (packed_target) {
        const auto count = cache::multiply(cache::multiply(requests, phase.prefix_rows), geometry.conv_kernel - 1);
        if (!io.input_packet.history_select().valid())
          io.history_select = phase_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(count)}));
        io.history_select_values.resize(count);
        if (packed_verify) {
          io.last_id_values.resize(requests);
          const auto prefixes = cache::multiply(requests, phase.prefix_rows);
          io.prefix_ids = phase_tensor(allocation_budget, execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(prefixes)}));
          io.prefix_id_values.resize(prefixes);
        }
      }
      if (completion == TextCompletion::kEmitNext) {
        io.logits = phase_tensor(allocation_budget, take(execution.resource_for(phase.logits_facts->storage)),
            phase.logits_facts->description, phase.logits_facts->caller_owned());
      }
      if (verifies_prefix(role)) io.prefix_candidates.resize(cache::multiply(requests, phase.prefix_rows));
      io.final_hidden = phase_tensor(allocation_budget, take(execution.resource_for(phase.hidden_facts->storage)),
          phase.hidden_facts->description, phase.hidden_facts->caller_owned());
      if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
        const auto &output = *phase.probe_facts[i];
        io.probe_outputs[i] = phase_tensor(allocation_budget, take(execution.resource_for(output.storage)), output.description, output.caller_owned());
      }
      std::size_t fixed_count = 2 + sampling_identity + last_ids + packed_target + packed_verify +
          io.logits.caller_owned() + io.final_hidden.caller_owned();
      if (has_mtp(role)) fixed_count += 1 + bool(draft_lag(role)) +
          ((packs_prefill(definition.packing) || compact_draft) ? 1 + bool(draft_lag(role)) : 0) + compact_draft;
      if (diagnostic) for (const auto *output : phase.probe_facts)
        fixed_count += output && output->caller_owned();
      const auto readback_count = bool(phase.sampling_valid_port) + bool(phase.sampling_accept_port) +
          (completion == TextCompletion::kEmitNext) + drafts_token(role) +
          bool(phase.successor_token_port) + bool(phase.successor_catchup_port);
      check(program_slot.io.prepare(phase.child->compiled, fixed_count, 0, readback_count, 0));
      // Register once in the original cold installation order. Tensor owners
      // stay here; each readback's only owner moves into the same shared IO.
      check(program_slot.io.add_fixed(phase.token_port,
          io.input_packet.prepared() ? io.input_packet.tokens() : io.tokens.read()));
      check(program_slot.io.add_fixed(phase.position_port,
          io.input_packet.prepared() ? io.input_packet.positions() : io.positions.read()));
      if (sampling_identity) check(program_slot.io.add_fixed(phase.sampling_identity_port,
          io.input_packet.sampling_identities().valid() ? io.input_packet.sampling_identities() : io.sampling_identities.read()));
      if (phase.sampling_valid_port) {
        io.sampling_valid = take(program_slot.io.add_readback(phase.sampling_valid_port, execution.storage_domains(), allocation_budget));
        if (phase.sampling_accept_port)
          io.sampling_accept = take(program_slot.io.add_readback(phase.sampling_accept_port, execution.storage_domains(), allocation_budget));
      }
      if (last_ids) check(program_slot.io.add_fixed(phase.last_ids_port,
          io.input_packet.last_ids().valid() ? io.input_packet.last_ids() : io.last_ids.read()));
      if (packed_target) check(program_slot.io.add_fixed(phase.history_select_port,
          io.input_packet.history_select().valid() ? io.input_packet.history_select() : io.history_select.read()));
      if (packed_verify) check(program_slot.io.add_fixed(phase.prefix_ids_port, io.prefix_ids.read()));
      if (io.logits.caller_owned()) check(program_slot.io.add_fixed_output(phase.logits_port, io.logits.view()));
      if (completion == TextCompletion::kEmitNext)
        io.next_id = take(program_slot.io.add_readback(phase.id_port, execution.storage_domains(), allocation_budget));
      if (phase.successor_token_port)
        io.successor_token = take(program_slot.io.add_readback(phase.successor_token_port, execution.storage_domains(), allocation_budget));
      if (phase.successor_catchup_port)
        io.successor_catchup = take(program_slot.io.add_readback(phase.successor_catchup_port, execution.storage_domains(), allocation_budget));
      if (io.final_hidden.caller_owned()) check(program_slot.io.add_fixed_output(phase.hidden_port, io.final_hidden.view()));
      if (has_mtp(role)) {
        check(program_slot.io.add_fixed(phase.draft_position_port, io.draft_positions.read()));
        if (draft_lag(role)) check(program_slot.io.add_fixed(phase.catchup_token_port, io.catchup_token.read()));
        if (packs_prefill(definition.packing) || compact_draft) {
          check(program_slot.io.add_fixed(phase.draft_hidden_select_port, io.draft_hidden_select.read()));
          if (draft_lag(role)) check(program_slot.io.add_fixed(phase.draft_token_select_port, io.draft_token_select.read()));
        }
        if (compact_draft) check(program_slot.io.add_fixed(phase.draft_last_ids_port, io.draft_last_ids.read()));
        if (drafts_token(role)) io.draft_id = take(program_slot.io.add_readback(phase.draft_id_port, execution.storage_domains(), allocation_budget));
      }
      if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
        if (const auto *output = phase.probe_facts[i]; output && output->caller_owned())
          check(program_slot.io.add_fixed_output(output->value, io.probe_outputs[i].view()));
      check(program_slot.io.seal());
      // Only these ordinary host-consumed results share a copy episode.
      // Device relay outputs and their full original writers remain unchanged.
      if (role == GraphRole::kText && completion == TextCompletion::kEmitNext &&
          phase.sampling_valid_port && !phase.sampling_accept_port && !diagnostic && !imported_sources &&
          execution.kind() == inference::BackendKind::kCuda && !execution.has_cpu_assist()) {
        const std::array<const inference::PreparedReadback *, 2> results{
            take(program_slot.io.readback(io.sampling_valid)), take(program_slot.io.readback(io.next_id))};
        auto prepared = inference::PreparedReadbackBatch::prepare(execution, results, execution_budget);
        if (prepared) io.small_results = std::move(*prepared);
        else {
          io.small_results_preparation = prepared.error();
          // Qualification/cold capacity may keep the old independent reads.
          // Never turn a native or runtime observation error into a fallback.
          if (prepared.error().has_native_error() ||
              (!prepared.error().IsUnimplemented() && !prepared.error().IsResourceExhausted()))
            return prepared.error();
        }
      }
      return hyle::Status::Ok();
    }));
  return result;
}

} // namespace qwen35
