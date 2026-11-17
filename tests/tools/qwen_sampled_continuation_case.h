#pragma once

#include "qwen_device_continuation_case.h"
#include "qwen_sampled_chain_math_case.h"
#include "qwen_attention_reader_pair_case.h"

// Full local Qwen3.5-2B consumer, not the small GA fixture. Compare decisions
// to the original route and state to an independently serialized execution of
// the same physical recipe. Native one/two-row GEMMs may round differently;
// that is distinct from transporting or publishing the wrong prepared state.
namespace qwen_sampled_continuation_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace d = qwen_device_continuation_case;
namespace i = hyle::inference;

inline int run(const std::filesystem::path &input, std::size_t proposals = 1,
    q::BackendKind backend = q::BackendKind::kCuda, bool observe_attention = false) {
  std::cout << "PASS legacy_second_teacher_contract checks=" << qwen_legacy_ledger_case::check_contract() << '\n';
  std::cout << "PASS legacy_sampled_semantic_contract checks="
            << qwen_legacy_ledger_case::check_sampled_semantic_contract() << '\n';
  h::require(backend == q::BackendKind::kCuda || backend == q::BackendKind::kApple ||
      backend == q::BackendKind::kVulkan, "sampled continuation requires an explicit GPU-only endpoint");
  const bool native = backend != q::BackendKind::kCuda;
  h::require(proposals >= 1 && proposals <= q::kMaximumDeviceMtpDrafts, "unsupported bounded sampled recipe");
  const auto width = proposals + 1;
  const auto draws = d::episodes * (2 * proposals + 1) + 2;
  constexpr std::size_t capacity = 32768;
  const std::array<std::array<std::int32_t, 3>, 8> prompts{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9},
      {10, 11, 12}, {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}};
  for (const bool random : {false, true}) {
    auto source = q::prepare_weight_source(input / "weights.tsv", true);
    const auto geometry = source.roles.geometry;
    const auto precision = q::PrecisionProfile::from_roles(source.roles);
    h::require(geometry.hidden == 2048 && geometry.vocabulary == 248320 &&
        geometry.context_limit >= capacity && geometry.mtp_layers == 1 &&
        geometry.recurrent_layers() && geometry.attention_layers(),
        "sampled continuation requires the actual local Qwen3.5-2B source");
    const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 1, (std::max)(std::size_t{3}, width),
        q::recurrent_bytes(q::state_schema_for(geometry, capacity, precision), geometry),
        1, width, true, true, 3, proposals > 1 ? 2 : 0, precision, proposals > 1 ? proposals : 0);
    h::require(plan.state_slots == 1 + 3 * width && plan.hidden_slots == 2 + 3 * width + (proposals > 1 ? 2 : 0) && plan.snapshot_slots == 4,
        "sampled continuation changed the existing bounded state pools");
    q::Model model(std::move(source), geometry, capacity, plan, {}, false,
        q::ReplayPolicy::kPreferCapture, SIZE_MAX,
        {.enabled = random, .policy = random ? q::sampling::Policy{.8F, 40, .95F} : q::sampling::Policy{},
         .seed = 42}, 0, {.primary = backend});
    if (random) qwen_sampled_chain_math_case::run(model, proposals);
    model.configure_draft_policy({.max_drafts = 0});
    h::require(!random || !model.supports_device_relay(0), "sampled zero-draft recipe inherited K1 routing");
    model.configure_draft_policy({.max_drafts = proposals});
    h::require(model.supports_device_relay(0), "prepared sampled K1 has no actual relay qualification");
    model.configure_device_mtp_successor(true);
    const std::array three{std::size_t{3}}, round_rows{width}, one{std::size_t{1}};
    model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(3, 1, 32, q::GraphRole::kMtpPrefill);
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound, q::GraphRole::kMtpContinuation}) {
      model.prepare_family(round_rows, false, 1, role);
      model.plan_replay_uses(width, 1, 128, role);
    }
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(one, false, 1, role);
      model.plan_replay_uses(1, 1, 32, role);
    }
    if (proposals > 1) {
      model.prepare_recursive_round_programs(0, false, 32, proposals - 1);
      for (std::size_t rows = 2; rows <= proposals; ++rows) {
        const std::array tail{rows};
        model.prepare_family(tail, false, 1, q::GraphRole::kVerify);
        model.plan_replay_uses(rows, 1, 32, q::GraphRole::kVerify);
      }
    }
    model.freeze_shape_plan();
    model.prepare_execution(three, 1, q::GraphRole::kMtpPrefill);
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound})
      model.prepare_execution(round_rows, 1, role);
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle})
      model.prepare_execution(one, 1, role);
    if (proposals > 1) for (std::size_t rows = 2; rows <= proposals; ++rows) {
      const std::array tail{rows};
      model.prepare_execution(tail, 1, q::GraphRole::kVerify);
    }
    model.prepare_relay_execution(1, 0, q::GraphRole::kMtpContinuation);
    const auto recording = model.prepare_mtp_decode_recordings(0);
    const auto &recording_phase = model.prepare_child(width, false, 1, q::GraphRole::kMtpRound);
    const std::size_t expected_preparations = recording_phase.slots[0].replay_mode == q::g::ReplayMode::kCommands ? 0 : 7;
    std::cout << "sampled_cold_recording_fields preparations=" << recording.preparations
              << " captures=" << recording.captures << " seconds=" << recording.seconds
              << " submitted_draft_rows=" << model.submitted_draft_rows
              << " snapshot_versions=" << model.snapshot_versions.live
              << " active_step=" << bool(model.active_step) << " successor_step=" << bool(model.successor_step)
              << " carrier_borrowed=" << model.carrier_borrowed << '\n' << std::flush;
    h::require(recording.preparations == expected_preparations && (native ? !recording.captures : recording.captures) &&
        !model.submitted_draft_rows &&
        !model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed,
        "sampled cold recording submitted work or retained a state episode");
    std::size_t accepts = 0, rejects = 0, reused = 0, native_prepared = 0, outputs = 0, terminal_stops = 0, legacy_geometry_differences = 0;
    std::size_t legacy_state_cases = 0, legacy_state_observations = 0;
    std::array<std::size_t, q::kMaximumDeviceMtpDrafts + 1> prefixes{};
    for (std::size_t index = 0; index < prompts.size(); ++index) {
      const i::SamplingCursor cursor{42 + index, 7 + index,
          index + 1 == prompts.size() ? i::SamplingCursor::maximum_coordinate - draws : 11};
      std::vector<d::s::StateImage> legacy_states, serial_states, device_states;
      d::AttentionObservation legacy_attention, serial_attention, device_attention;
      const bool inspect = observe_attention && !random && !index;
      const auto legacy = d::journey(model, prompts[index], false, cursor, inspect ? &legacy_states : nullptr, false,
          inspect ? &legacy_attention : nullptr);
      const auto serial = d::journey(model, prompts[index], true, cursor, &serial_states, true,
          inspect ? &serial_attention : nullptr);
      const auto device = d::journey(model, prompts[index], true, cursor, &device_states, false,
          inspect ? &device_attention : nullptr);
      std::cout << "compare_continuation sampling=" << (random ? "random" : "greedy")
                << " case=" << index << " outputs=" << device.ids.size()
                << " accepts=" << device.a2_published << " rejects=" << device.a1_published << '\n' << std::flush;
      const bool first_id_differs = serial.ids.empty() != device.ids.empty() ||
          (!serial.ids.empty() && !device.ids.empty() && serial.ids.front() != device.ids.front());
      const bool first_record_differs = serial.records.empty() != device.records.empty() ||
          (!serial.records.empty() && !device.records.empty() &&
              !qwen_legacy_ledger_case::strict_equal(serial.records.front(), device.records.front()));
      if (first_id_differs || first_record_differs) {
        std::cerr << "prefill_saved_prompt=";
        for (const auto id : prompts[index]) std::cerr << id << ',';
        std::cerr << '\n';
        if (serial_states.empty() || device_states.empty()) std::cerr << "prefill_saved_snapshot_unavailable\n";
        else try { d::diagnose_saved_prefill(model, serial_states.front(), device_states.front()); }
        catch (const std::exception &error) { std::cerr << "prefill_saved_diagnostic_error=" << error.what() << '\n'; }
      }
      d::compare_records(serial, device);
      h::require(legacy.stopped == device.stopped && serial.stopped == device.stopped,
          "continuation changed the natural terminal boundary");
      h::require(serial_states.size() == device_states.size(), "different continuation observation counts");
      for (std::size_t state = 0; state < serial_states.size(); ++state) {
        d::compare_state(model, serial_states[state], device_states[state]);
      }
      d::compare_state(model, *serial.final_state, *device.final_state);
      h::require(device.final_state->sampling == (random ? cursor.advanced(draws) : i::SamplingCursor{}),
          "successor changed the original sampled or greedy cursor contract");
      const auto expected_submissions = inspect ? std::size_t{4} : std::size_t{8};
      const auto expected_successors = inspect ? std::size_t{1} : std::size_t{5};
      h::require(device.submitted == expected_submissions &&
          (native ? device.native_prepared == expected_successors && !device.reused :
          device.reused == expected_successors && !device.native_prepared) && !device.recorded,
          "sampled successor submitted extra work or recorded a hot program");
      for (const auto &record : device.records)
        if (record.generation && record.draft_limit == proposals) ++prefixes.at(record.accepted - 1);
      std::cout << "PASS same_physical_continuation proposals=" << proposals << " random=" << random
                << " case=" << index << " complete_state=equal ledger=equal logical_rng=equal" << '\n' << std::flush;
      if (legacy.ids != device.ids) {
        ++legacy_geometry_differences;
        std::cout << "legacy_geometry_difference proposals=" << proposals << " random=" << random << " case=" << index << " old=";
        for (const auto id : legacy.ids) std::cout << id << ',';
        std::cout << " new=";
        for (const auto id : device.ids) std::cout << id << ',';
        std::cout << '\n' << std::flush;
      }
      // Exact state/identity belongs to the same physical program above.
      // Stochastic proposals may differ with draft GEMM row geometry; the
      // independent equation witness checks the p/q/RNG contract directly.
      if (!random || legacy.ids == device.ids) {
        d::compare_legacy_records(model.source_identity,
            model.sampling_configuration(0).configuration, static_cast<std::int32_t>(geometry.vocabulary), legacy, device);
        // Different attention recipes need a numerical/teacher comparison.
        // The complete-state transport oracle above remains strict between
        // serialized and device-scheduled executions of the SAME recipe.
        // A separate diagnostic can inspect cross-recipe propagation; its
        // byte equality is deliberately not a production acceptance premise.
        if (inspect) {
        h::require(legacy_states.size() == device_states.size() &&
            legacy_states.size() + 1 == legacy.records.size(),
            "legacy complete-state observations omitted a committed boundary");
        for (std::size_t state = 0; state < legacy_states.size(); ++state) {
          try { d::compare_state(model, legacy_states[state], device_states[state]); }
          catch (...) {
            std::cerr << "legacy_complete_state_difference case=" << index << " snapshot=" << state
                      << " sampling=" << (random ? "random" : "greedy") << '\n';
            d::diagnose_saved_prefill(model, legacy_states[state], device_states[state]);
#if defined(HYLE_COMPUTE_VULKAN)
            if (inspect && state == 2 && backend == q::BackendKind::kVulkan) {
              h::require(legacy_attention.captured && device_attention.captured &&
                  legacy_attention.query_lengths == device_attention.query_lengths &&
                  legacy_attention.kv_lengths == device_attention.kv_lengths &&
                  legacy_attention.query_lengths.size() == 1 && legacy_attention.kv_lengths.size() == 1 &&
                  legacy_attention.accepted_rows == legacy_attention.input_rows &&
                  device_attention.accepted_rows == device_attention.input_rows,
                  "actual attention observation did not retain the complete equal live verifier prefix");
              const auto &a = legacy_attention; const auto &b = device_attention;
              std::cerr << "actual_attention_pair before=" << a.before_length << ',' << b.before_length
                        << " input_rows=" << a.input_rows << ',' << b.input_rows
                        << " accepted=" << a.accepted_rows << ',' << b.accepted_rows
                        << " device_lengths=" << a.device_lengths << ',' << b.device_lengths
                        << " Q=" << a.query_lengths[0] << " KV=" << a.kv_lengths[0]
                        << " query_bytes_equal=" << (a.query.bytes == b.query.bytes)
                        << " attention_bytes_equal=" << (a.attention.bytes == b.attention.bytes)
                        << " key_bytes_equal=" << (legacy_states[state].target[0] == device_states[state].target[0])
                        << " value_bytes_equal=" << (legacy_states[state].target[1] == device_states[state].target[1]) << '\n';
              qwen_attention_reader_pair_case::replay("legacy", geometry,
                  static_cast<std::uint32_t>(a.query_lengths[0]), static_cast<std::uint32_t>(a.kv_lengths[0]),
                  a.query.description, a.query.bytes, a.attention.bytes,
                  legacy_states[state].target[0], legacy_states[state].target[1], a.device_lengths);
              qwen_attention_reader_pair_case::replay("continuation", geometry,
                  static_cast<std::uint32_t>(b.query_lengths[0]), static_cast<std::uint32_t>(b.kv_lengths[0]),
                  b.query.description, b.query.bytes, b.attention.bytes,
                  device_states[state].target[0], device_states[state].target[1], b.device_lengths);
            }
#endif
            throw;
          }
        }
        d::compare_state(model, *legacy.final_state, *device.final_state);
        ++legacy_state_cases;
        legacy_state_observations += legacy_states.size() + 1;
        }
      }
      if (!random && !observe_attention) {
        // Native output tape, including every actually published proposal and
        // bonus. The original independent BF16 teacher consumes this prefix;
        // a separate ordinary worker cannot stand in for this continuation.
        std::cout << "native_teacher_tape proposals=" << proposals << " case=" << index << " prompt=";
        for (const auto id : prompts[index]) std::cout << id << ',';
        std::cout << " ids=";
        for (const auto id : device.ids) std::cout << id << ',';
        std::cout << '\n' << std::flush;
      }
      accepts += device.a2_published; rejects += device.a1_published;
      reused += device.reused; native_prepared += device.native_prepared; outputs += device.ids.size();
      terminal_stops += device.stopped;
    }
    std::cout << "real_prefix_inventory proposals=" << proposals << " random=" << random << " counts=";
    for (std::size_t n = 0; n < width; ++n) std::cout << prefixes[n] << ',';
    std::cout << '\n' << std::flush;
    if (observe_attention) std::cout << "legacy_complete_state_inventory proposals=" << proposals << " random=" << random
              << " cases=" << legacy_state_cases << " observations=" << legacy_state_observations
              << " changed_content_cases=" << legacy_geometry_differences
              << " original_kv_hidden_history_exact=1 original_gdn_budget=3e-6" << '\n' << std::flush;
    h::require(accepts && rejects, "real model did not exercise both published K1 prefix choices");
    const auto recovery = d::boundary_journey(model, prompts[1], -1, false, d::BoundaryCase::kRecovery);
    h::require(recovery.ids.size() >= 2 && recovery.ids[0] != recovery.ids[1],
        "real continuation prompt has no post-prefill stop witness");
    const auto stop = recovery.ids[1];
    const auto serial_stop = d::boundary_journey(model, prompts[1], stop, false, d::BoundaryCase::kStop);
    const auto device_stop = d::boundary_journey(model, prompts[1], stop, true, d::BoundaryCase::kStop);
    d::compare_records(serial_stop, device_stop);
    d::compare_state(model, *serial_stop.final_state, *device_stop.final_state);
    // Reuse is a same-route contract. Different legacy/native attention and
    // draft row recipes can follow different stochastic traces; comparing a
    // native post-cancel request to a legacy request cannot isolate reuse.
    const auto native_recovery = d::boundary_journey(model, prompts[1], stop, true, d::BoundaryCase::kRecovery);
    for (const auto failure : {d::BoundaryCase::kCancel, d::BoundaryCase::kHostFailure}) {
      (void)d::boundary_journey(model, prompts[1], stop, true, failure);
      const auto restored = d::boundary_journey(model, prompts[1], stop, true, d::BoundaryCase::kRecovery);
      d::compare_request_records(native_recovery, restored);
      d::compare_state(model, *native_recovery.final_state, *restored.final_state);
      h::require(restored.submitted, "request recovery did not restore its original device successors");
    }
    std::cout << "PASS real_sampled_continuation backend=" << h::backend_name(backend)
              << " hidden=2048 vocabulary=248320 capacity=32768"
              << " proposals=" << proposals << " sampling=" << (random ? "random" : "greedy") << " cases=8 outputs=" << outputs
              << " accepts=" << accepts << " rejects=" << rejects << " captured_successors=" << reused << " native_source_successors=" << native_prepared
              << " natural_terminal_stops=" << terminal_stops
              << " legacy_geometry_differences=" << legacy_geometry_differences
              << " hot_recordings=0 complete_state=equal ledger=equal logical_rng=equal"
              << " stop=passed cancellation=passed failure=passed recovery=passed retired=1\n" << std::flush;
  }
  return 0;
}
}  // namespace qwen_sampled_continuation_case
