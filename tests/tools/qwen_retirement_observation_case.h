#pragma once

// Included after the real model implementation by the lifecycle consumer.
// Uses the existing cached untrained A fixture. Status assignments below are
// explicit host-policy injections, not native/device failure injection.
namespace qwen_retirement_observation_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

inline void finish_normal(std::shared_ptr<q::Session> &session) {
  (void)session->stop();
  const auto drained = session->drain();
  const auto retired = session->try_retire();
  if (retired.retired) session.reset();
  q::check(drained.status); q::check(retired.status);
  require(retired.retired, "ordinary CPU Session did not actually retire");
}

inline int run(const std::filesystem::path &input_directory) {
  const auto manifest = input_directory / "weights.tsv";
  const auto geometry = q::read_weight_geometry(manifest);
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A",
      "retirement observation requires the cached untrained A/H64/FFN128/V64 fixture");
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, {.target = cache_rows}, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2);
  q::Model model(manifest, geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      16u << 20, {}, 0, {.primary = i::BackendKind::kCpu});
  const std::array shapes{std::size_t{2}, std::size_t{1}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(shapes, false, width);
    for (const auto rows : shapes) model.plan_replay_uses(rows, width, 8);
  }
  for (const auto width : {std::size_t{1}, std::size_t{2}})
    model.prepare_execution(shapes, width);
  const std::array<std::int32_t, 2> prompt{1, 2};
  const std::array<std::int32_t, 1> one{3};
  const auto first_error = hyle::Status::Internal("synthetic first retirement observation");
  const auto later_error = hyle::Status::Unavailable("synthetic later retirement observation");

  // A prior real error remains observable through a later observation and
  // repeated stop, without becoming a substitute for actual retirement proof.
  {
    auto session = model.create_session(shapes);
    q::evaluate(model, *session, prompt, "retirement_error_seed", {}, false);
    const auto *head = session->head.get();
    const auto before = q::generation_boundary(*head);
    session->retirement_status = first_error; // Synthetic host status only.
    model.prepare_evaluation(*session, one); // Prepared, deliberately not submitted.
    model.active_step->terminal = later_error; // Synthetic host status only.
    const auto observed = session->drain();
    require(!observed.retired && observed.status == first_error &&
        session->retirement_status == first_error && !session->stopping &&
        session->head.get() == head && q::generation_boundary(*session->head) == before &&
        !model.active_step && !session->pending && !model.carrier_borrowed,
        "unstopped error observation destroyed the head or replaced the first error");
    require(session->stop() == first_error && session->stop() == first_error,
        "repeated stop hid the original retirement error");
    const auto retired = session->try_retire();
    require(retired.retired && retired.status == first_error && !session->head,
        "actual retirement lost the original error or retained the head");
    require(session->stop() == first_error,
        "stop after actual retirement lost its original error");
    session.reset();
    require(!model.snapshot_versions.live, "retired error Session retained a version");
  }

  // Ordinary drain does not mean stop/release and leaves the Session runnable.
  {
    auto session = model.create_session(shapes);
    q::evaluate(model, *session, prompt, "retirement_live_seed", {}, false);
    const auto *head = session->head.get();
    const auto before = q::generation_boundary(*head);
    const auto observed = session->drain();
    q::check(observed.status);
    require(!observed.retired && !session->stopping && session->head.get() == head &&
        q::generation_boundary(*session->head) == before && session->retirement_status.ok(),
        "ordinary unstopped drain changed the live Session");
    q::evaluate(model, *session, one, "retirement_live_continue", {}, false);
    require(session->length() == prompt.size() + one.size(),
        "unstopped drain prevented ordinary continuation");
    finish_normal(session);
  }

  // Prepare a real shared R2 episode, but do not submit native work. The host
  // terminal assignment is a synthetic pure cancellation, not a device fault.
  {
    auto a = model.create_session(shapes), b = model.create_session(shapes);
    q::evaluate(model, *a, prompt, "retirement_shared_a", {}, false);
    q::evaluate(model, *b, prompt, "retirement_shared_b", {}, false);
    const auto *a_head = a->head.get(), *b_head = b->head.get();
    const auto a_before = q::generation_boundary(*a_head);
    const auto b_before = q::generation_boundary(*b_head);
    const std::array inputs{q::EvaluationRequest{a, one}, q::EvaluationRequest{b, one}};
    model.prepare_evaluations(inputs);
    require(model.active_step && !model.active_step->submitted &&
        a->pending == model.active_step.get() && b->pending == model.active_step.get(),
        "shared cancellation check did not prepare the actual original R2 episode");
    model.active_step->terminal = hyle::Status::Cancelled("synthetic prepared episode cancellation");
    q::check(a->stop());
    const auto observed = b->drain();
    require(!observed.retired && observed.status.IsCancelled() && !observed.status.has_native_error() &&
        b->retirement_status.ok() && !b->stopping && !b->failed &&
        a->head.get() == a_head && b->head.get() == b_head &&
        q::generation_boundary(*a->head) == a_before &&
        q::generation_boundary(*b->head) == b_before &&
        !model.active_step && !a->pending && !b->pending && !model.carrier_borrowed,
        "survivor drain made a peer cancellation sticky or changed a committed head");
    const auto retired_a = a->try_retire();
    require(retired_a.retired && retired_a.status.IsCancelled() &&
        !retired_a.status.has_native_error(), "cancelled A lost either retirement fact");
    a.reset();
    q::evaluate(model, *b, one, "retirement_survivor_continue", {}, false);
    require(b->length() == prompt.size() + one.size() && b->retirement_status.ok(),
        "cancelled peer prevented survivor continuation");
    finish_normal(b);
  }
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
      "retirement observation case retained a version or execution owner");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\","
      "\"fixture\":\"cached_untrained_A_H64_FFN128_V64\","
      "\"synthetic_status_only\":true,\"native_fault_injected\":false,"
      "\"repeated_stop_preserves_error\":true,\"retired_error_releases_head\":true,"
      "\"unstopped_drain_keeps_head\":true,\"first_error_preserved\":true,"
      "\"prepared_R2_survivor_continues\":true,\"final_versions\":0,"
      "\"GPU\":false,\"HTTP\":false}\n";
  return 0;
}
} // namespace qwen_retirement_observation_case
