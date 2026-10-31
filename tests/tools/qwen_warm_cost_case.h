#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#if defined(HYLE_TEST_INFERENCE_WARM_COST)
#include "inference_warm_cost_probe.h"
#endif

// Included after the product Model. This finite receipt observes its existing
// cold factories, carriers and episode owners; it is not an allocation-free
// claim for the CRT, providers or the whole inference engine.
namespace qwen_warm_cost_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
using h::require;

struct VectorStorage final {
  const void *data = nullptr;
  std::size_t capacity = 0;
  friend bool operator==(const VectorStorage &, const VectorStorage &) = default;
};
template<class T> inline VectorStorage storage(const std::vector<T> &values) noexcept {
  return {values.data(), values.capacity()};
}

struct Scratch final {
  std::array<VectorStorage, 10> vectors;
  std::array<const void *, 8> paged_rows;
  std::array<std::size_t, 3> paged_bounds;
  friend bool operator==(const Scratch &, const Scratch &) = default;
};
inline Scratch scratch(const q::Model &model) noexcept {
  const auto &s = model.input_scratch;
  const auto old = s.paged_appends.old_rows.layout(), next = s.paged_appends.next_rows.layout();
  return {{{storage(s.packed_ids), storage(s.positions), storage(s.draft_positions), storage(s.catchups),
      storage(s.row_inputs), storage(s.query_indptr), storage(s.last_rows), storage(s.sampling_identities),
      storage(s.paged_appends.sources), storage(s.paged_appends.tail_owners)}},
      {{old.query_indptr.data(), old.kv_indptr.data(), old.page_indices.data(), old.last_page_lengths.data(),
        next.query_indptr.data(), next.kv_indptr.data(), next.page_indices.data(), next.last_page_lengths.data()}},
      {{s.paged_appends.max_members(), s.paged_appends.old_rows.max_page_refs(),
        s.paged_appends.next_rows.max_page_refs()}}};
}

struct Carrier final {
  const q::Family *family = nullptr;
  const i::PreparedProgram *child = nullptr;
  const q::Phase *phase = nullptr;
  const void *invocation = nullptr;
  std::array<VectorStorage, 11> io;
  friend bool operator==(const Carrier &, const Carrier &) = default;
};
inline Carrier carrier(q::Phase &phase, std::size_t at) {
  auto &slot = phase.slots[at]; const auto &v = slot.data;
  require(slot.carrier.initialized() && !slot.carrier.closing(),
      "warm receipt lost its original initialized carrier");
  const auto *run = q::take(slot.carrier.borrow());
  return {phase.family, phase.child, &phase, run->object_identity(),
      {{storage(v.batch_layouts), storage(v.history_select_values), storage(v.last_id_values),
        storage(v.prefix_id_values), storage(v.draft_hidden_indices), storage(v.draft_token_indices),
        storage(v.sampling_valid_values), storage(v.sampling_accept_values), storage(v.sampling_scratch),
        storage(v.draft_ids), storage(v.prefix_candidates)}}};
}
inline std::vector<Carrier> carriers(q::Model &model) {
  std::vector<Carrier> result;
  for (auto &[key, phase] : model.phases) {
    (void)key;
    for (std::size_t at = 0; at != phase->slots.size(); ++at) result.push_back(carrier(*phase, at));
  }
  return result;
}

struct Pools final {
  std::size_t capacity = 0, allocated = 0, peak = 0, live_blocks = 0, cached_writers = 0;
  std::uint64_t reclaimed = 0;
};
inline Pools pools(const q::Model &model) noexcept {
  Pools result;
  const auto add = [&](const q::c::LeasePool &pool) {
    if (!pool.valid()) return;
    const auto stats = pool.stats();
    result.capacity += stats.capacity_bytes; result.allocated += stats.allocated_bytes;
    result.peak += stats.peak_bytes; result.live_blocks += stats.live_blocks;
    result.reclaimed += stats.reclaimed_blocks;
  };
  for (const auto &state : model.state_pools) {
    add(state.history); add(state.recurrent);
    result.cached_writers += state.history_reuse.size() + state.recurrent_reuse.size();
  }
  add(model.hidden_pool); return result;
}

template<class T> inline std::size_t vector_bytes(const std::vector<T> &values) noexcept {
  return values.capacity() * sizeof(T);
}
struct Metadata final {
  std::size_t step_owners = 1, incoming_owners = 0, source_owners = 0;
  std::size_t owner_object_bytes = sizeof(q::StepState), members = 0, append_nodes = 0;
  std::size_t member_capacity = 0, member_spill_capacity_bytes = 0, tail_spill_capacity_bytes = 0;
  std::size_t episode_vector_capacity_bytes = 0, incoming_bindings = 0;
  std::size_t source_stores = 0, source_successor_stores = 0, source_pages = 0, source_outputs = 0, source_boundaries = 0;
  std::size_t source_vector_capacity_bytes = 0, state_writer_slots = 0, state_writer_bytes = 0;
};
inline Metadata metadata(q::StepState &step) noexcept {
  Metadata result;
  result.members = step.members.size(); result.append_nodes = step.execution.nodes.size();
  result.member_capacity = step.members.capacity();
  if (result.member_capacity > 1) result.member_spill_capacity_bytes = result.member_capacity * sizeof(q::StepMember);
  for (const auto &member : step.members)
    result.episode_vector_capacity_bytes += vector_bytes(member.verification_inputs) + vector_bytes(member.carry_prefix);
  for (const auto &dirty : step.execution.dirty) result.episode_vector_capacity_bytes += vector_bytes(dirty);
  for (const auto &tails : step.execution.tails)
    if (tails.capacity() > 1) result.tail_spill_capacity_bytes += tails.capacity() * sizeof(q::c::PagedPageAppender);
  if (step.execution.incoming) {
    result.incoming_owners = 1; result.owner_object_bytes += sizeof(q::IncomingStepSources);
    result.incoming_bindings = step.execution.incoming->bindings.size();
    result.episode_vector_capacity_bytes += vector_bytes(step.execution.incoming->bindings);
  }
  if (step.execution.sources) {
    const auto &sources = *step.execution.sources;
    result.source_owners = 1; result.owner_object_bytes += sizeof(q::StepSources);
    result.source_stores = sources.states.size(); result.source_successor_stores = sources.successor_states.size();
    result.source_pages = sources.pages.size(); result.source_outputs = 1 + sources.extra_outputs.size();
    result.source_boundaries = sources.boundaries.size();
    result.source_vector_capacity_bytes = vector_bytes(sources.states) + vector_bytes(sources.pages) +
        vector_bytes(sources.extra_outputs) + vector_bytes(sources.boundaries) +
        vector_bytes(sources.published_states) + vector_bytes(sources.successor_states);
  }
  if (step.phase) {
    auto &transactions = step.phase->slots[step.state_slot].transactions;
    for (std::size_t bank = 0; bank != step.phase->state_banks.size(); ++bank) {
      const auto writers = transactions.writers(bank).size();
      result.state_writer_slots += writers;
      result.state_writer_bytes += writers * transactions.writer_bytes(bank);
    }
  }
  return result;
}

inline void print_metadata(const Metadata &m) {
  std::cout << ",\"StepState_owners\":" << m.step_owners << ",\"incoming_owners\":" << m.incoming_owners
      << ",\"StepSources_owners\":" << m.source_owners << ",\"owner_object_bytes\":" << m.owner_object_bytes
      << ",\"members\":" << m.members << ",\"append_node_descriptors\":" << m.append_nodes
      << ",\"member_capacity\":" << m.member_capacity << ",\"member_spill_capacity_bytes\":" << m.member_spill_capacity_bytes
      << ",\"tail_spill_capacity_bytes\":" << m.tail_spill_capacity_bytes
      << ",\"episode_vector_capacity_bytes\":" << m.episode_vector_capacity_bytes
      << ",\"incoming_bindings\":" << m.incoming_bindings << ",\"Source_store_descriptors\":" << m.source_stores
      << ",\"Source_successor_store_descriptors\":" << m.source_successor_stores
      << ",\"Source_page_descriptors\":" << m.source_pages << ",\"Source_output_descriptors\":" << m.source_outputs
      << ",\"Source_boundaries\":" << m.source_boundaries
      << ",\"Source_vector_capacity_bytes\":" << m.source_vector_capacity_bytes
      << ",\"state_writer_slots\":" << m.state_writer_slots << ",\"state_writer_bytes\":" << m.state_writer_bytes;
}
inline void print_pools(const Pools &before, const Pools &after) {
  std::cout << ",\"state_pool_capacity_bytes\":" << after.capacity
      << ",\"state_pool_before_allocated_bytes\":" << before.allocated
      << ",\"state_pool_after_allocated_bytes\":" << after.allocated
      << ",\"state_pool_peak_bytes\":" << after.peak
      << ",\"state_pool_before_live_blocks\":" << before.live_blocks
      << ",\"state_pool_after_live_blocks\":" << after.live_blocks
      << ",\"state_pool_reclaimed_blocks_delta\":" << after.reclaimed - before.reclaimed
      << ",\"state_pool_cached_writers\":" << after.cached_writers;
}

struct Counts final {
  std::size_t family_builds = 0, compiles = 0, phase_prepares = 0, slot_initializations = 0;
  std::size_t allocation_calls = 0, allocation_bytes = 0, allocation_successes = 0, deallocation_calls = 0;
};
#if defined(HYLE_TEST_INFERENCE_WARM_COST)
inline Counts read_counts() noexcept {
  const auto value = i::testing::cold_work_snapshot();
  return {value.family_builds, value.compiles, value.phase_prepares, value.slot_initializations,
      value.resource_allocate_calls, value.resource_allocate_bytes,
      value.resource_allocate_successes, value.resource_deallocate_calls};
}
#endif
inline Counts delta(const Counts &after, const Counts &before) noexcept {
  return {after.family_builds - before.family_builds, after.compiles - before.compiles,
      after.phase_prepares - before.phase_prepares, after.slot_initializations - before.slot_initializations,
      after.allocation_calls - before.allocation_calls, after.allocation_bytes - before.allocation_bytes,
      after.allocation_successes - before.allocation_successes, after.deallocation_calls - before.deallocation_calls};
}
inline void no_cold_work(const Counts &counts) {
  require(!counts.family_builds && !counts.compiles && !counts.phase_prepares && !counts.slot_initializations,
      "known-shape warm episode entered an actual cold factory, compilation or Invocation initialization");
}
inline void print_counts(const Counts &c) {
  std::cout << ",\"Family_build_calls\":" << c.family_builds << ",\"compile_calls\":" << c.compiles
      << ",\"Phase_prepare_calls\":" << c.phase_prepares
      << ",\"slot_cold_initialization_calls\":" << c.slot_initializations
      << ",\"delegated_allocate_calls\":" << c.allocation_calls
      << ",\"delegated_allocate_requested_bytes\":" << c.allocation_bytes
      << ",\"delegated_allocate_successes\":" << c.allocation_successes
      << ",\"delegated_deallocate_calls\":" << c.deallocation_calls;
}

inline void arm(const std::filesystem::path &input, q::BackendKind kind, std::size_t width,
    bool mtp) {
#if defined(HYLE_TEST_INFERENCE_WARM_COST)
  std::optional<i::testing::CpuResourceCountingScope> cpu_resource_scope;
  if (kind == q::BackendKind::kCpu) cpu_resource_scope.emplace();
  const auto start_counts = read_counts();
  auto source = q::prepare_weight_source(input / "weights.tsv", mtp);
  const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 32, width, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 32), geometry), width, mtp ? 2u : 1u,
      mtp, mtp, 2);
  const auto replay = kind == q::BackendKind::kCuda ? q::ReplayPolicy::kPreferCapture : q::ReplayPolicy::kCommands;
  q::Model model(std::move(source), geometry, 32, plan, {}, false, replay, SIZE_MAX,
      {.enabled = false}, 0, {.primary = kind});
  const auto prefill_role = mtp ? q::GraphRole::kMtpPrefill : q::GraphRole::kText;
  const std::array three{std::size_t{3}};
  const std::array two{std::size_t{2}};
  const std::array ordinary{std::size_t{3}, std::size_t{1}};
  if (mtp) {
    require(model.prepare_family(three, false, 1, prefill_role), "warm K1 prefill Family was not built");
    model.plan_replay_uses(3, 1, 2, prefill_role);
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
      require(model.prepare_family(two, false, 1, role), "warm K1 original round Family was not built");
      model.plan_replay_uses(2, 1, 16, role);
    }
  } else {
    require(model.prepare_family(ordinary), "warm ordinary R1 admission Family was not built");
    for (const auto rows : ordinary) model.plan_replay_uses(rows, 1, 16);
    if (width > 1) {
      require(model.prepare_family(ordinary, false, width), "warm ordinary R2 execution Family was not built");
      for (const auto rows : ordinary) model.plan_replay_uses(rows, width, 16);
    }
  }
  model.freeze_shape_plan(); model.prepare_state_storage();
  if (mtp) {
    model.prepare_execution(three, 1, prefill_role);
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound})
      model.prepare_execution(two, 1, role);
  } else {
    model.prepare_execution(ordinary);
    if (width > 1) model.prepare_execution(ordinary, width);
  }
  const auto cold = delta(read_counts(), start_counts);
  require(cold.family_builds && cold.compiles && cold.phase_prepares && cold.slot_initializations,
      "warm probe did not observe its real cold factories and Invocation initializations");
  if (kind == q::BackendKind::kCpu)
    require(cold.allocation_calls && cold.allocation_successes,
        "CPU warm probe did not observe its real delegated allocations");
  const auto original_carriers = carriers(model); const auto original_scratch = scratch(model);
  const auto before_lookup = read_counts();
  for (const auto &[key, phase] : model.phases) {
    require(&model.prepare_child(key.tokens, false, key.requests, key.role, key.completion,
        key.sampling_index, key.imported_sources, key.packing) == phase.get(),
        "warm lookup replaced an original Phase");
    for (std::size_t at = 0; at != phase->slots.size(); ++at) model.initialize_slot(*phase, at);
  }
  const auto lookup = delta(read_counts(), before_lookup); no_cold_work(lookup);
  require(!lookup.allocation_calls && !lookup.deallocation_calls && carriers(model) == original_carriers &&
      scratch(model) == original_scratch,
      "warm lookup/initialization allocated delegated storage or replaced original owners/scratch");
  const auto arm_name = mtp ? "K1" : width == 1 ? "R1" : "R2";
  const auto resource_counting = kind == q::BackendKind::kCpu;
  std::cout << "{\"case\":\"warm-cost\",\"arm\":\"" << arm_name << "\",\"stage\":\"cold\",\"resource_counting\":"
      << (resource_counting ? "true" : "false");
  print_counts(cold); std::cout << ",\"carriers\":" << original_carriers.size() << "}\n";
  std::cout << "{\"case\":\"warm-cost\",\"arm\":\"" << arm_name << "\",\"stage\":\"warm-initialize\",\"resource_counting\":"
      << (resource_counting ? "true" : "false");
  print_counts(lookup); std::cout << ",\"same_owners_and_scratch\":true}\n";

  std::vector<std::shared_ptr<q::Session>> sessions;
  sessions.reserve(width);
  for (std::size_t at = 0; at != width; ++at) sessions.push_back(model.create_session(three, {}, prefill_role));
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    (void)model.abort_episode();
    for (auto &session : sessions) if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset();
    }
  }};
  const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 3}, {3, 2, 1}}};
  std::array<std::int32_t, 2> pending{}, next{};
  std::vector<q::EvaluationRequest> requests(width);
  std::vector<q::EvaluationOutput> outputs(width);
  std::array<std::array<const q::cache::State *, 2>, 2> state_objects{};
  std::size_t reused = 0, warm_recorded = 0;
  for (std::size_t episode = 0; episode != 9; ++episode) {
    for (std::size_t at = 0; at != width; ++at) {
      pending[at] = sessions[at]->head->next_id;
      requests[at] = {sessions[at], episode ? std::span<const std::int32_t>(&pending[at], 1)
          : std::span<const std::int32_t>(prompts[at])};
      outputs[at] = q::evaluation_output(model, sessions[at], {.name = "warm-cost", .trace_states = false});
    }
    const auto role = !episode ? prefill_role : mtp ? qwen_mtp_relay_case::role(*sessions[0]) : q::GraphRole::kText;
    const auto before = read_counts(); const auto pools_before = pools(model);
    model.prepare_evaluations(requests, nullptr, role);
    auto &step = *model.active_step;
    for (std::size_t at = 0; at != width; ++at) {
      const auto *candidate = step.members[at].next.get();
      require(candidate, "warm episode did not acquire its original candidate state object");
      if (episode < 2) state_objects[at][episode] = candidate;
      else require(candidate == state_objects[at][0] || candidate == state_objects[at][1],
          "warm WorkingState did not reuse its original spare candidate objects");
    }
    const auto before_sources = read_counts();
    if (kind == q::BackendKind::kCuda && episode) (void)model.prepare_submission_sources(step);
    const auto sources = delta(read_counts(), before_sources);
    const auto descriptors = metadata(step);
    const auto recorded = step.capture_recorded, reused_now = step.capture_reused;
    const auto route = step.replay_route;
    require(!step.execution.unsubmitted_reuse_declined(), "ordinary warm producer acquired a typed pending refusal");
    model.submit_evaluations();
    if (mtp && episode) {
      const std::array selections{q::MtpGenerationSelection{sessions[0], pending[0], 2}};
      std::array<q::MtpRoundResult, 1> result;
      model.finish_mtp_rounds(selections, {}, outputs, result, [](auto, const auto &) {});
    } else model.finish_evaluations(outputs, std::span(next).first(width), [](auto, const auto &) {});
    const auto episode_counts = delta(read_counts(), before); const auto pools_after = pools(model);
    no_cold_work(episode_counts);
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
        carriers(model) == original_carriers && scratch(model) == original_scratch,
        "real warm episode changed cold identities, retained a loan or grew its prepared scratch");
    if (episode >= 4) { reused += reused_now; warm_recorded += recorded; }
    std::cout << "{\"case\":\"warm-cost\",\"arm\":\"" << arm_name << "\",\"episode\":" << episode
        << ",\"stage\":\"" << (episode < 4 ? "native-prime" : "warm") << "\",\"capture_recorded\":" << recorded
        << ",\"capture_reused\":" << reused_now << ",\"replay_route\":" << static_cast<unsigned>(route)
        << ",\"resource_counting\":" << (resource_counting ? "true" : "false")
        << ",\"candidate_state_objects\":" << width << ",\"candidate_state_object_bytes\":" << width * sizeof(q::cache::State)
        << ",\"candidate_state_objects_reused\":" << (episode >= 2 ? "true" : "false")
        << ",\"typed_pending_refusals\":0,\"Source_prepare_allocate_calls\":" << sources.allocation_calls
        << ",\"Source_prepare_allocate_requested_bytes\":" << sources.allocation_bytes;
    print_counts(episode_counts); print_metadata(descriptors); print_pools(pools_before, pools_after);
    std::cout << ",\"same_owners_and_scratch\":true}\n";
  }
  if (kind == q::BackendKind::kCuda)
    require(reused && !warm_recorded, "finite CUDA warm producer failed to reuse its original recorded capture");
  for (auto &session : sessions) h::retire(session);
  require(!model.snapshot_versions.live && !model.carrier_borrowed &&
      model.page_inventory.free_ids.size() == model.cache_plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "warm receipt retained an original state version, page or execution loan");
  cleanup.dismiss();
#else
  (void)input; (void)kind; (void)width; (void)mtp;
#endif
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
#if defined(HYLE_TEST_INFERENCE_WARM_COST)
  const auto source = q::prepared::read_roles(input / "weights.tsv", true);
  const auto &g = source.geometry;
  if ((kind != q::BackendKind::kCpu && kind != q::BackendKind::kCuda) || g.hidden != 64 ||
      g.intermediate != 128 || g.vocabulary != 64 || g.layers != "GA" || g.context_limit != 32 || g.mtp_layers != 1) {
    std::cerr << "SKIP: warm-cost requires the cached untrained GA/H64/I128/V64/C32/MTP fixture\n";
    return 77;
  }
  arm(input, kind, 1, false); arm(input, kind, 2, false);
  arm(input, kind, 1, true);
  std::cout << "{\"status\":\"PASS\",\"case\":\"warm-cost\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"arms\":3,\"submitted_graphs\":27,\"cold_entry_checks\":true,"
         "\"per_episode_dynamic_metadata_reported\":true,\"CRT_heap_calls_measured\":false,"
         "\"delegated_resource_scope\":\"CPU_Backend_endpoints\",\"other_memory_resources_measured\":false,"
         "\"native_provider_private_allocations_measured\":false,\"pending_successor_rerun\":false,"
         "\"Graph_Source_private_metadata_bytes_measured\":false,"
         "\"metadata_bytes_exclude_shared_control_blocks_allocator_overhead\":true,"
         "\"performance_claim\":false,\"whole_engine_zero_allocation_claim\":false,\"delegated_warm_initialize_allocation_checked\":"
      << (kind == q::BackendKind::kCpu ? "true" : "false") << ",\"resource_counting\":"
      << (kind == q::BackendKind::kCpu ? "true" : "false") << "}\n";
  return 0;
#else
  (void)input; (void)kind;
  std::cerr << "SKIP: warm-cost test-only entry/resource probes are not compiled\n"; return 77;
#endif
}

} // namespace qwen_warm_cost_case
