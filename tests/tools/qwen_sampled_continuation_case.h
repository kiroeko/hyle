#pragma once

#include "qwen_device_continuation_case.h"

// Full local Qwen3.5-2B consumer, not the small GA fixture. Compare decisions
// to the original route and state to an independently serialized execution of
// the same physical recipe. Native one/two-row GEMMs may round differently;
// that is distinct from transporting or publishing the wrong prepared state.
namespace qwen_sampled_continuation_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace d = qwen_device_continuation_case;
namespace i = hyle::inference;

inline int run(const std::filesystem::path &input) {
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
    const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 1, 3,
        q::recurrent_bytes(q::state_schema_for(geometry, capacity, precision), geometry),
        1, 2, true, true, 3, 0, precision);
    h::require(plan.state_slots == 7 && plan.hidden_slots == 8 && plan.snapshot_slots == 4,
        "sampled continuation changed the existing bounded state pools");
    q::Model model(std::move(source), geometry, capacity, plan, {}, false,
        q::ReplayPolicy::kPreferCapture, SIZE_MAX,
        {.enabled = random, .policy = random ? q::sampling::Policy{.8F, 40, .95F} : q::sampling::Policy{},
         .seed = 42}, 0, {.primary = q::BackendKind::kCuda});
    model.configure_draft_policy({.max_drafts = 0});
    h::require(!random || !model.supports_device_relay(0), "sampled zero-draft recipe inherited K1 routing");
    model.configure_draft_policy({.max_drafts = 1});
    h::require(model.supports_device_relay(0), "prepared sampled K1 has no actual relay qualification");
    model.configure_device_mtp_successor(true);
    const std::array three{std::size_t{3}}, two{std::size_t{2}}, one{std::size_t{1}};
    model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(3, 1, 32, q::GraphRole::kMtpPrefill);
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound, q::GraphRole::kMtpContinuation}) {
      model.prepare_family(two, false, 1, role);
      model.plan_replay_uses(2, 1, 128, role);
    }
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(one, false, 1, role);
      model.plan_replay_uses(1, 1, 32, role);
    }
    model.freeze_shape_plan();
    model.prepare_execution(three, 1, q::GraphRole::kMtpPrefill);
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound})
      model.prepare_execution(two, 1, role);
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle})
      model.prepare_execution(one, 1, role);
    model.prepare_relay_execution(1, 0, q::GraphRole::kMtpContinuation);
    const auto recording = model.prepare_mtp_decode_recordings(0);
    h::require(recording.preparations == 7 && recording.captures && !model.submitted_draft_rows &&
        !model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed,
        "sampled cold recording submitted work or retained a state episode");
    std::size_t accepts = 0, rejects = 0, reused = 0, outputs = 0, terminal_stops = 0;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
      const i::SamplingCursor cursor{42 + index, 7 + index,
          index + 1 == prompts.size() ? i::SamplingCursor::maximum_coordinate - 20 : 11};
      std::vector<d::s::StateImage> serial_states, device_states;
      const auto legacy = d::journey(model, prompts[index], false, cursor);
      const auto serial = d::journey(model, prompts[index], true, cursor, &serial_states, true);
      const auto device = d::journey(model, prompts[index], true, cursor, &device_states);
      std::cout << "compare_continuation sampling=" << (random ? "random" : "greedy")
                << " case=" << index << " outputs=" << device.ids.size()
                << " accepts=" << device.a2_published << " rejects=" << device.a1_published << '\n' << std::flush;
      d::compare_records(legacy, device);
      d::compare_records(serial, device);
      h::require(legacy.stopped == device.stopped && serial.stopped == device.stopped,
          "continuation changed the natural terminal boundary");
      h::require(serial_states.size() == device_states.size(), "different continuation observation counts");
      for (std::size_t state = 0; state < serial_states.size(); ++state) {
        d::compare_state(model, serial_states[state], device_states[state]);
      }
      d::compare_state(model, *serial.final_state, *device.final_state);
      h::require(device.final_state->sampling == (random ? cursor.advanced(20) : i::SamplingCursor{}),
          "successor changed the original sampled or greedy cursor contract");
      h::require(device.submitted == 8 && device.reused == 5 && !device.recorded,
          "sampled successor submitted extra work or recorded a hot program");
      accepts += device.a2_published; rejects += device.a1_published;
      reused += device.reused; outputs += device.ids.size();
      terminal_stops += device.stopped;
    }
    h::require(accepts && rejects, "real model did not exercise both published K1 prefix choices");
    const auto recovery = d::boundary_journey(model, prompts[1], -1, false, d::BoundaryCase::kRecovery);
    h::require(recovery.ids.size() >= 2 && recovery.ids[0] != recovery.ids[1],
        "real continuation prompt has no post-prefill stop witness");
    const auto stop = recovery.ids[1];
    const auto serial_stop = d::boundary_journey(model, prompts[1], stop, false, d::BoundaryCase::kStop);
    const auto device_stop = d::boundary_journey(model, prompts[1], stop, true, d::BoundaryCase::kStop);
    d::compare_records(serial_stop, device_stop);
    d::compare_state(model, *serial_stop.final_state, *device_stop.final_state);
    for (const auto failure : {d::BoundaryCase::kCancel, d::BoundaryCase::kHostFailure}) {
      (void)d::boundary_journey(model, prompts[1], stop, true, failure);
      const auto restored = d::boundary_journey(model, prompts[1], stop, true, d::BoundaryCase::kRecovery);
      d::compare_request_records(recovery, restored);
      h::require(restored.submitted, "request recovery did not restore its original device successors");
    }
    std::cout << "PASS real_sampled_continuation backend=cuda hidden=2048 vocabulary=248320 capacity=32768"
              << " sampling=" << (random ? "random" : "greedy") << " cases=8 outputs=" << outputs
              << " accepts=" << accepts << " rejects=" << rejects << " captured_successors=" << reused
              << " natural_terminal_stops=" << terminal_stops
              << " hot_recordings=0 complete_state=equal ledger=equal logical_rng=equal"
              << " stop=passed cancellation=passed failure=passed recovery=passed retired=1\n" << std::flush;
  }
  return 0;
}
}  // namespace qwen_sampled_continuation_case
