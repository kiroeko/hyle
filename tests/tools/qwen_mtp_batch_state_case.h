#pragma once

#include "../../src/graph/slots_region.h"
#include "hyle/graph/detail/region_access.h"

// Included after the product Model and the existing host-image helpers.
namespace qwen_mtp_batch_state_case {
namespace q = qwen35;
namespace c = hyle::compute;
namespace gd = hyle::graph::detail;
namespace sd = hyle::graph::slots::detail;
using qwen_host_image_case::require;
using Bytes = std::vector<std::byte>;

inline Bytes read(q::Model &model, const c::ReadLease &lease) {
  Bytes bytes(lease.size());
  if (!bytes.empty()) q::check(model.execution.read(q::take(lease.view()), bytes));
  return bytes;
}
inline Bytes frame(q::Model &model, const c::PagedArenaFrame &value, std::size_t rows) {
  const auto footprint = q::take(value.view());
  const auto host = footprint.layout().host();
  require(host.last_page_lengths.size() == 1 &&
      (host.page_indices.empty() ? rows == 0 :
       (host.page_indices.size() - 1) * q::cache::Plan::page_rows +
           static_cast<std::size_t>(host.last_page_lengths[0]) == rows),
      "B2 state image has an unexpected logical KV frontier");
  Bytes bytes;
  for (const auto id : host.page_indices) {
    const auto part = read(model, q::take(footprint.page_prefix(static_cast<std::uint32_t>(id))));
    bytes.insert(bytes.end(), part.begin(), part.end());
  }
  return bytes;
}
struct StateImage final {
  std::size_t length = 0, frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t pending = -1, catchup = -1;
  q::cache::SamplingCursor sampling;
  std::vector<Bytes> payloads, target;
  friend bool operator==(const StateImage &, const StateImage &) = default;
};
inline StateImage image(q::Model &model, const q::cache::State &state) {
  model.validate_mtp_snapshot(state);
  StateImage result{state.length, state.draft->frontier, state.draft->lag,
      state.next_id, state.draft->catchup_token, state.sampling, {}, {}};
  for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer)
    if (model.geometry.is_gdn(layer)) {
      result.payloads.push_back(read(model, state.state[layer].history));
      result.payloads.push_back(read(model, state.state[layer].recurrent));
    }
  for (std::size_t layer = 0; layer < state.keys.size(); ++layer) {
    result.target.push_back(frame(model, state.keys[layer], state.length));
    result.target.push_back(frame(model, state.values[layer], state.length));
  }
  result.payloads.push_back(frame(model, state.draft->key, state.draft->frontier));
  result.payloads.push_back(frame(model, state.draft->value, state.draft->frontier));
  result.payloads.push_back(read(model, state.carry_hidden));
  result.payloads.push_back(read(model, state.prior_hidden));
  return result;
}
struct Bank final {
  std::size_t layer = 0;
  bool recurrent = false;
  std::array<c::LeaseSlotAddress, 2> sources{};
  std::array<c::LeaseSlotAddress, 6> candidates{};
  std::array<Bytes, 6> bytes;
};
inline Bank inspect(q::Model &model, q::StepState &step, std::size_t layer, bool recurrent,
    bool hidden = false) {
  require(step.members.size() == 2 && step.phase->prefix_rows == 3 && step.phase->state_transactions[step.state_slot].bound() &&
      gd::RegionAccess::successful_outcome(*step.run), "B2 inspection requires its successful original K3 Graph");
  const auto &ports = hidden ? step.phase->hidden_state
      : recurrent ? step.phase->layers[layer].recurrent : step.phase->layers[layer].history;
  const auto item = hidden ? q::dense(model.precision.activation_type(), {1, model.hidden})
      : recurrent ? model.state_schema[layer].recurrent : model.state_schema[layer].history;
  Bank bank{layer, recurrent, {}, {}, {}};
  for (auto &bytes : bank.bytes) bytes.resize(q::take(item.required_bytes()));
  struct Visit final {
    q::Model *model;
    q::StepState *step;
    const q::StatePorts *ports;
    c::TensorDesc item;
    Bank *bank;
    bool hidden = false;
    bool found = false;
  } visit{&model, &step, &ports, item, &bank, hidden};
  q::check(gd::RegionAccess::visit_retired(*step.run, ports.store,
      +[](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
        auto &v = *static_cast<Visit *>(opaque);
        if (v.found || std::ranges::find(view.node_ordinals, v.ports->store.ordinal()) == view.node_ordinals.end() ||
            !view.prepared_vtable || !view.prepared_vtable->query_extension)
          return hyle::Status::InvalidArgument("B2 state has no unique original Store");
        const auto *extension = static_cast<const sd::StoreRegionVTable *>(
            view.prepared_vtable->query_extension(view.prepared, sd::kStoreBindingExtensionId));
        if (!extension || !extension->inspect || !extension->binding)
          return hyle::Status::Unimplemented("B2 selected Store has no transaction extension");
        const hyle::graph::ProviderRegionPrepared prepared{.vtable = view.prepared_vtable, .state = view.prepared};
        HYLE_ASSIGN_OR_RETURN(auto facts, extension->inspect(prepared, view.invocation));
        if (facts.store != v.ports->store || facts.destination_indices != v.ports->destination_ids ||
            facts.destination_count != 6 || (v.hidden && facts.source.count) || (facts.source.count &&
                (facts.source.count != 2 || facts.source.gather != v.ports->gather ||
                 facts.source.capacity != v.ports->pool || facts.source.indices != v.ports->source_ids)))
          return hyle::Status::InvalidArgument("B2 Store facts differ from its actual ports");
        auto *binding = extension->binding(view.invocation);
        if (!binding) return hyle::Status::FailedPrecondition("B2 Store lost its original owner");
        HYLE_RETURN_IF_ERROR(binding->validate_authority());
        const auto *transaction = binding->transaction();
        const std::size_t source_count = v.hidden ? 0 : 2;
        if (!transaction || transaction->source_count() != source_count || transaction->destination_count() != 6 ||
            transaction->sources().size() != source_count || transaction->destinations().size() != 6 ||
            transaction->source_addresses().size() != source_count || transaction->destination_addresses().size() != 6)
          return hyle::Status::FailedPrecondition("B2 Store no longer owns its full original cohort");
        const auto layout = transaction->layout();
        HYLE_ASSIGN_OR_RETURN(auto schema, hyle::graph::slots::Schema::create(
            v.item, layout.slot_count(), layout.stride_bytes()));
        if (layout.description() != v.item || facts.schema != schema)
          return hyle::Status::InvalidArgument("B2 Store schema differs from its original pool");
        for (std::size_t j = 0; j < source_count; ++j) {
          const auto &base = v.step->members[j].base->state[v.bank->layer];
          HYLE_ASSIGN_OR_RETURN(auto address, layout.locate(v.bank->recurrent ? base.recurrent : base.history));
          if (address != transaction->source_addresses()[j])
            return hyle::Status::InvalidArgument("B2 Store source belongs to another member");
          v.bank->sources[j] = address;
        }
        for (std::size_t index = 0; index < 6; ++index) {
          const auto &writer = transaction->destinations()[index];
          HYLE_ASSIGN_OR_RETURN(auto address, layout.locate(writer));
          if (address != transaction->destination_addresses()[index])
            return hyle::Status::InvalidArgument("B2 Store candidate changed slot/generation");
          if (v.hidden) {
            HYLE_ASSIGN_OR_RETURN(auto original, v.model->hidden_layout.locate(writer));
            if (original != address)
              return hyle::Status::InvalidArgument("B2 hidden candidate belongs to another original pool");
          }
          for (std::size_t prior = 0; prior < index; ++prior)
            if (v.bank->candidates[prior].index == address.index)
              return hyle::Status::InvalidArgument("B2 candidate slots overlap");
          for (std::size_t j = 0; j < source_count; ++j)
            if (v.bank->sources[j].index == address.index)
              return hyle::Status::InvalidArgument("B2 candidate overlaps an immutable source");
          HYLE_ASSIGN_OR_RETURN(auto storage, writer.view());
          if (storage.resource() != facts.resource || storage.size() != v.bank->bytes[index].size())
            return hyle::Status::InvalidArgument("B2 candidate raw extent differs");
          HYLE_RETURN_IF_ERROR(v.model->execution.read(hyle::mem::ConstBufferView{storage}, v.bank->bytes[index]));
          v.bank->candidates[index] = address;
        }
        v.found = true;
        return hyle::Status::Ok();
      }, &visit));
  require(visit.found && (hidden || bank.sources[0].index != bank.sources[1].index),
      "B2 state sources must be distinct original slots");
  return bank;
}
inline std::size_t reclaim_unselected(q::Model &model, const Bank &bank,
    const std::array<std::size_t, 2> &accepted) {
  auto &pools = model.state_pools[bank.layer];
  auto &pool = bank.recurrent ? pools.recurrent : pools.history;
  const auto &layout = bank.recurrent ? pools.recurrent_layout : pools.history_layout;
  std::vector<c::WriteLease> writers;
  std::vector<std::size_t> indices;
  bool exhausted = false;
  for (std::size_t attempt = 0; attempt <= pool.stats().max_blocks; ++attempt) {
    auto writer = pool.allocate(bank.bytes[0].size(), q::cache::state_pool_alignment);
    if (!writer) {
      require(writer.error().IsResourceExhausted(), "B2 reclaimed slot allocation failed for an unrelated reason");
      exhausted = true; break;
    }
    indices.push_back(q::take(layout.locate(*writer)).index);
    writers.push_back(std::move(*writer));
  }
  require(exhausted, "B2 bounded free-slot check did not reach the original pool limit");
  std::size_t checked = 0;
  for (std::size_t index = 0; index < 6; ++index) {
    const bool selected = index == accepted[0] - 1 || index == 3 + accepted[1] - 1;
    const bool free = std::ranges::find(indices, bank.candidates[index].index) != indices.end();
    require(selected ? !free : free, "B2 selected state escaped or an unaccepted candidate stayed retained");
    checked += !selected;
  }
  return checked; // Reclaimed permissions, not a claim that backing VRAM was freed.
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.mtp_layers == 1,
      "B2 case requires its cached C16/MTP fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  require(plan.state_slots == 12, "B2 K3 changed the original N+K*R capacity");
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = kind});
  const std::array calls{std::size_t{3}};
  // Session admission uses its R1 call plan; the actual work below is B2.
  model.prepare_family(calls, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 1, q::GraphRole::kMtpPrefill);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kVerify}) {
    model.prepare_family(calls, false, 2, role);
    model.plan_replay_uses(3, 2, 3, role);
  }
  for (std::size_t rows = 1; rows <= 3; ++rows) model.prepare_draft_program(rows, false);
  for (std::size_t rows = 1; rows <= 2; ++rows) model.prepare_draft_program(rows, true);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kVerify}) model.prepare_execution(calls, 2, role);
  std::array sessions{model.create_session(calls, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 7, 11}),
      model.create_session(calls, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 8, 13})};
  const std::array<std::array<std::int32_t, 3>, 2> prompt{{{1, 2, 3}, {3, 2, 1}}};
  const std::array io{q::EvaluationIO{.name = "B2-MTP-state-0", .trace_states = false},
      q::EvaluationIO{.name = "B2-MTP-state-1", .trace_states = false}};
  const std::array outputs{q::evaluation_output(model, sessions[0], io[0]), q::evaluation_output(model, sessions[1], io[1])};
  const auto observe = [&](std::size_t j, const q::CompletedEvaluationView &view) {
    q::observe_completed_evaluation(model, outputs[j], io[j], view);
  };
  const std::array seed{q::EvaluationRequest{sessions[0], prompt[0]}, q::EvaluationRequest{sessions[1], prompt[1]}};
  model.prepare_evaluations(seed, nullptr, q::GraphRole::kMtpPrefill);
  model.submit_evaluations();
  std::array<std::int32_t, 2> next{};
  model.finish_evaluations(outputs, next, observe);
  std::array<q::cache::SnapshotRef, 2> original{sessions[0]->checkpoint(), sessions[1]->checkpoint()};
  const std::array original_images{image(model, *original[0]), image(model, *original[1])};
  // A real checked embedding failure exercises the same cold NextN episode,
  // not a synthetic Completion or a host-preflight failure. The original
  // committed Sessions and this private draft prefix must remain unchanged.
  {
    const std::array epochs{sessions[0]->head.epoch(), sessions[1]->head.epoch()};
    q::DraftWork work; work.state = q::Model::fork_draft(*original[0]->draft);
    const auto frontier = work.state.frontier;
    const auto old_key = frame(model, work.state.key, frontier);
    const auto old_value = frame(model, work.state.value, frontier);
    const std::array shifted{original[0]->carry_hidden};
    const std::array<std::int32_t, 1> invalid{-1};
    const auto identity = original[0]->sampling.identity(0);
    bool rejected = false;
    try { (void)model.execute_draft(work, invalid, shifted, true, identity); }
    catch (const hyle::inference::EvaluationFailure &failure) {
      require(failure.status().IsOutOfRange() || failure.status().IsInvalidArgument(),
          "NextN checked ID failed with an unrelated execution status");
      rejected = true;
    }
    require(rejected, "NextN checked embedding accepted a negative token ID");
    auto &program = *model.draft_programs.at({1, true});
    require(!program.guard.has_completion() && !program.guard.cleanup_pending &&
        !program.hidden_transactions.bound() && !program.retirement.pending(),
        "NextN checked failure did not really retire its original episode");
    require(work.state.frontier == frontier && frame(model, work.state.key, frontier) == old_key &&
        frame(model, work.state.value, frontier) == old_value,
        "NextN failure published a private KV prefix or frontier");
    for (std::size_t j = 0; j < sessions.size(); ++j)
      require(sessions[j]->head.epoch() == epochs[j] && image(model, *sessions[j]->head) == original_images[j] &&
          image(model, *original[j]) == original_images[j],
          "NextN failure changed a committed Session/checkpoint/epoch/RNG");
    const std::array valid{original[0]->next_id};
    const auto proposal = model.execute_draft(work, valid, shifted, true, identity);
    require(proposal.id >= 0 && proposal.id < model.vocabulary && proposal.hidden.valid() &&
        work.state.frontier == frontier + 1 && !program.guard.has_completion() &&
        !program.guard.cleanup_pending && !program.hidden_transactions.bound() && !program.retirement.pending(),
        "NextN original program did not recover for a valid one-row proposal");
    const auto new_key = frame(model, work.state.key, frontier + 1);
    const auto new_value = frame(model, work.state.value, frontier + 1);
    require(new_key.size() * frontier == old_key.size() * (frontier + 1) &&
        new_value.size() * frontier == old_value.size() * (frontier + 1) &&
        std::equal(old_key.begin(), old_key.end(), new_key.begin()) &&
        std::equal(old_value.begin(), old_value.end(), new_value.begin()),
        "NextN valid retry changed its old private KV prefix or logical row count");
    for (std::size_t j = 0; j < sessions.size(); ++j)
      require(sessions[j]->head.epoch() == epochs[j] && image(model, *sessions[j]->head) == original_images[j] &&
          image(model, *original[j]) == original_images[j],
          "NextN private valid retry changed a committed Session/checkpoint/epoch/RNG");
  }
  std::array<q::cache::SnapshotRef, 2> first;
  std::array<StateImage, 2> first_images;
  std::size_t bytes_checked = 0, reclaimed = 0;
  std::vector<Bank> diagnostics;
  for (std::size_t round = 0; round < 2; ++round) {
    const std::array accepted = round ? std::array<std::size_t, 2>{3, 1} : std::array<std::size_t, 2>{1, 3};
    const std::array pending = round ? std::array<std::int32_t, 2>{13, 14} : std::array<std::int32_t, 2>{11, 12};
    const std::array before{image(model, *sessions[0]->head), image(model, *sessions[1]->head)};
    const std::array<std::array<std::int32_t, 3>, 2> ids{{{sessions[0]->head->next_id, 4, 5},
        {sessions[1]->head->next_id, 6, 7}}};
    const std::array requests{q::EvaluationRequest{sessions[0], ids[0]}, q::EvaluationRequest{sessions[1], ids[1]}};
    std::vector<Bank> banks;
    std::array<Bytes, 2> carry, prior;
    model.prepare_evaluations(requests, nullptr, q::GraphRole::kVerify);
    require(model.active_step->members.size() == 2 && model.active_step->phase->prefix_rows == 3,
        "B2 case did not enter one real K3 verifier Graph");
    model.submit_evaluations();
    model.finish_selected_evaluations(outputs, next,
        [&](auto, q::StepState &step, auto selected) {
          for (std::size_t layer = 0; layer < geometry.layers.size(); ++layer) if (geometry.is_gdn(layer))
            for (const bool recurrent : {false, true}) banks.push_back(inspect(model, step, layer, recurrent));
          const auto hidden = inspect(model, step, 0, false, true);
          for (std::size_t j = 0; j < 2; ++j) {
            carry[j] = hidden.bytes[j * step.phase->prefix_rows + accepted[j] - 1];
            if (accepted[j] > 1) prior[j] = hidden.bytes[j * step.phase->prefix_rows + accepted[j] - 2];
            step.members[j].accepted_tokens = accepted[j]; selected[j] = pending[j];
          }
        }, [&](q::StepState &step) { model.repair_recursive_draft(step); }, observe);
    require(!model.active_step && !model.carrier_borrowed, "B2 verifier did not detach its original episode");
    for (std::size_t j = 0; j < 2; ++j) {
      const auto &head = *sessions[j]->head;
      const auto actual = image(model, head);
      require(actual.length == before[j].length + accepted[j] && actual.pending == pending[j] &&
          actual.sampling == before[j].sampling && actual.lag == (accepted[j] == 1 ? 0u : 1u) &&
          actual.frontier + actual.lag == actual.length && actual.catchup == (actual.lag ? ids[j][accepted[j] - 1] : -1),
          "B2 selected length/pending/unchanged controlled RNG/draft frontier differs");
      require(read(model, head.carry_hidden) == carry[j] && read(model, head.prior_hidden) == prior[j],
          "B2 carry/prior did not retain its own exact selected Store row");
      require(actual.target.size() == before[j].target.size(), "B2 target KV layer count changed");
      for (std::size_t bank = 0; bank < actual.target.size(); ++bank)
        require(actual.target[bank].size() * before[j].length == before[j].target[bank].size() * actual.length &&
            std::equal(before[j].target[bank].begin(), before[j].target[bank].end(), actual.target[bank].begin()),
            "B2 visible KV length or old initialized prefix differs");
      for (const auto &bank : banks) {
        const auto &pools = model.state_pools[bank.layer];
        const auto &layout = bank.recurrent ? pools.recurrent_layout : pools.history_layout;
        const auto &lease = bank.recurrent ? head.state[bank.layer].recurrent : head.state[bank.layer].history;
        const auto selected = j * 3 + accepted[j] - 1;
        require(q::take(layout.locate(lease)) == bank.candidates[selected] && read(model, lease) == bank.bytes[selected],
            "B2 member committed another prefix/member/pool candidate");
        bytes_checked += bank.bytes[selected].size();
      }
      require(image(model, *original[j]) == original_images[j], "B2 verifier mutated an original checkpoint");
      if (round) require(image(model, *first[j]) == first_images[j], "B2 continuation mutated its frozen preceding head");
    }
    for (const auto &bank : banks) reclaimed += reclaim_unselected(model, bank, accepted);
    diagnostics.insert(diagnostics.end(), std::make_move_iterator(banks.begin()), std::make_move_iterator(banks.end()));
    if (!round) for (std::size_t j = 0; j < 2; ++j) {
      first[j] = sessions[j]->checkpoint(); first_images[j] = image(model, *first[j]);
    }
  }
  first = {}; original = {};
  for (auto &session : sessions) qwen_host_image_case::retire(session);
  require(model.snapshot_versions.live == 0 && !model.active_step && !model.carrier_borrowed,
      "B2 state case retained a version or original episode");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"requests\":2,\"verify_rows\":3,\"candidate_prefixes\":3,\"rounds\":2,\"accepted\":[[1,3],[3,1]],"
      << "\"checked_selected_bytes\":" << bytes_checked << ",\"reclaimed_unselected_slots\":" << reclaimed
      << ",\"banks\":[";
  for (std::size_t n = 0; n < diagnostics.size(); ++n) {
    const auto &bank = diagnostics[n]; if (n) std::cout << ',';
    std::cout << "{\"layer\":" << bank.layer << ",\"role\":\"" << (bank.recurrent ? "gdn" : "conv")
        << "\",\"source_slot_ids\":[" << bank.sources[0].index << ',' << bank.sources[1].index << "],\"candidate_slot_ids\":[";
    for (std::size_t index = 0; index < bank.candidates.size(); ++index) {
      if (index) std::cout << ','; std::cout << bank.candidates[index].index;
    }
    std::cout << "]}";
  }
  std::cout << "],\"same_width_original_candidates\":true,\"old_snapshots_unchanged\":true,"
      << "\"final_versions\":0,\"controlled_prefix_pending\":true,\"controlled_rng_unchanged\":true,"
      << "\"nextn_checked_id_failure_recovered\":true,\"same_program_valid_retry\":true,"
      << "\"natural_acceptance_quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_mtp_batch_state_case
