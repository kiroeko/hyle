#pragma once

// Uses the original complete-state image checker, not a token-only oracle.
#include "qwen_host_image_case.h"
#include "qwen_depth_case.h"

namespace qwen_native_relay_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;

// The first public request can inherit an empty recording Session. Its cursor
// must match a later request and an independently loaded serial executor.
inline void run_recording_rng(const std::filesystem::path &input, bool control_inputs = false,
    q::BackendKind kind = q::BackendKind::kCuda, bool random = true) {
  namespace i = hyle::inference;
  i::ModelLoadConfig config;
  config.assets.inputs = input;
  config.execution = {.backend = kind, .context = 128,
      .resident_sessions = 1, .active_sequences = 1, .prefill_rows = 32};
  config.execution_profiles[0].draft_count = 0;
  config.execution_profiles[0].overlap = true;
  if (random) config.sampling = {.random = true, .temperature = .8F, .top_p = .95F, .top_k = 40};
  auto preparation = q::prepare_execution_profile(config, 0);
  h::require(preparation.overlap && preparation.cache.in_flight_batches == 2,
      "recording RNG regression requires its original prepared relay");
  auto model = q::instantiate_execution_profile(std::move(preparation), 128);
  auto executor = q::make_prepared_executor(*model, false, 32, true);
  const auto vocabulary = static_cast<std::size_t>(model->vocabulary);
  std::array schedules{i::GenerationSessionSchedule{"recording-rng", {{{1, 2, 3}, 16}},
      i::GenerationSessionMode::kFlat, 42}};
  schedules.front().sampling_stream = 7;
  const std::array stops{i::RequestStopPolicy(std::span<const std::int32_t>{}, true)};
  const i::GenerationBatch batch{.sessions = schedules, .stop_policies = stops};
  const auto generate = [&](i::ModelExecutor &prepared) {
    qwen_depth_case::Tokens tokens; tokens.vocabulary = vocabulary;
    i::TokenConsumerRef output(tokens);
    prepared.validate(batch); prepared.execute(batch, output);
    h::require(!tokens.invalid && !tokens.cancelled && tokens.count == 16 && tokens.endings == 1,
        "recording RNG request did not deliver its complete output");
    return tokens.ids;
  };
  const auto first = generate(*executor), reused = generate(*executor);
  h::require(first == reused, "cold recording replaced the first request's logical RNG cursor");
  std::size_t imported_uses = 0;
  for (const auto &[shape, phase] : model->phases) if (phase->imported_sources) {
    (void)shape;
    for (std::size_t index = 0; index < phase->slots.size(); ++index)
      imported_uses += phase->slots[index].preparation_uses;
  }
  h::require(imported_uses >= 16 && !model->active_step && !model->successor_step &&
      !model->carrier_borrowed && !model->snapshot_versions.live,
      "recording RNG regression did not exercise and retire its real relay");
  std::size_t control_slots = 0;
  if (control_inputs) {
    h::require(q::generation_control_storage(model->execution, {}).has_value(),
        "ordinary controls have no original queued staging owner");
    for (const auto &[shape, phase] : model->phases) {
      (void)shape;
      if (phase->family->role != q::GraphRole::kText ||
          phase->family->completion != q::TextCompletion::kEmitNext ||
          phase->family->packing != q::RowPacking::kRectangular) continue;
      h::require(phase->next_id_owner.ref() == model->execution.device(),
          "ordinary successor token writer moved away from its device domain");
      for (std::size_t index = 0; index < phase->slots.size(); ++index) {
        auto &io = phase->slots[index].data;
        h::require(!io.ordinary_io.packet() &&
            io.tokens.storage().resource() == model->execution.device() &&
            io.positions.storage().resource() == model->execution.staging() &&
            (!random || io.sampling_identities.storage().resource() == model->execution.staging()),
            "prepared ordinary controls or device token lost their exact owners");
        h::require(phase->delivery_id_port && !io.ordinary_io.batched_results() &&
            q::take(phase->slots[index].io.readback(io.source_id))->resource() == model->execution.device() &&
            q::take(phase->slots[index].io.readback(io.next_id))->resource() == model->execution.staging() &&
            (!phase->sampling_valid_port ||
             q::take(phase->slots[index].io.readback(io.sampling_valid))->resource() == model->execution.staging()),
            "ordinary scalar delivery escaped the original Graph or moved the device Source");
        ++control_slots;
      }
    }
    h::require(control_slots >= 3, "ordinary controls did not cover the finite physical slots");
  }
  executor.reset(); model.reset();
  config.execution_profiles[0].overlap = false;
  auto serial = q::load_model_executor(config);
  h::require(first == generate(*serial), "cold recorded and serial requests used different RNG draws");
  std::cout << (control_inputs ? "PASS ordinary_control_inputs" : "PASS recording_request_rng")
            << " backend=" << h::backend_name(kind) << " sampling=" << (random ? "random" : "greedy")
            << " seed=42 logical_stream=7 outputs=48"
            << " first_reused_serial_ids=equal imported_uses=" << imported_uses
            << " control_slots=" << control_slots
            << " retired=1\n" << std::flush;
}

inline void run(const std::filesystem::path &input, q::BackendKind kind, bool random) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto precision = q::PrecisionProfile::from_roles(source.roles);
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 32, 3, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 32, precision), geometry),
      1, 1, false, true, 2, 0, precision);
  q::Model model(std::move(source), geometry, 32, plan, {}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = random, .policy = random ? q::sampling::Policy{.8F, 40, .95F} : q::sampling::Policy{},
       .seed = 123}, 0, {.primary = kind});
  h::require(model.execution.queue_bindings().empty(), "native model relay unexpectedly uses a caller Queue");
  h::require(bool(model.execution.storage_owner()), "native Model has no original primary storage owner");
  h::require(!model.execution.storage_owner().context_identity(), "native Model has a synthetic common Context");
  if (!model.supports_prefill_relay(0)) {
    // Native decode imports do not qualify prefill capture. There is no
    // prefill family yet: a refusal must not construct private chunk slots.
    model.bounded_prefill_rows = 2;
    const auto phases = model.phases.size();
    const auto recording = model.prepare_prefill_recordings(0);
    h::require(!recording.preparations && model.phases.size() == phases,
        "unavailable prefill capture materialized unused native chunk storage");
    model.bounded_prefill_rows = 0;
  }
  h::require(q::native_ordinary_io_domains(model.execution,
      {.role = q::GraphRole::kText, .completion = q::TextCompletion::kEmitNext}).has_value(),
      "native Model has no independently owned device and staging domains");
  const std::array rows{std::size_t{3}, std::size_t{1}};
  model.prepare_family(rows);
  for (const auto count : rows) model.plan_replay_uses(count, 1, 8);
  // A larger warm capacity covers later smaller suffixes. Its private native
  // scratch may be allocated on first submit, after Invocation installation;
  // do not retain the superseded small Invocation until that allocation fails.
  const std::array private_prefill_shapes{std::size_t{2}, std::size_t{3}};
  h::require(model.prepare_prefill_family(private_prefill_shapes, q::TextCompletion::kStateOnly),
      "native prefill admission regression has no actual packed programs");
  for (const auto count : private_prefill_shapes)
    model.plan_replay_uses(count, 1, 8, q::GraphRole::kText,
        q::TextCompletion::kStateOnly, 0, q::RowPacking::kPrefill);
  model.prepare_execution(rows);
  model.prepare_relay_execution();
  const std::array small_prefill{std::size_t{2}}, larger_prefill{std::size_t{3}};
  model.prepare_prefill_execution(small_prefill, q::TextCompletion::kStateOnly);
  auto &small = model.prepare_child(2, false, 1, q::GraphRole::kText,
      q::TextCompletion::kStateOnly, 0, false, q::RowPacking::kPrefill);
  h::require(small.slots[0].carrier.initialized(), "small native prefill was not installed");
  model.prepare_prefill_execution(larger_prefill, q::TextCompletion::kStateOnly);
  auto &larger = model.prepare_child(3, false, 1, q::GraphRole::kText,
      q::TextCompletion::kStateOnly, 0, false, q::RowPacking::kPrefill);
  h::require(!small.slots[0].carrier.initialized() && larger.slots[0].carrier.initialized() &&
      small.private_capacity_preferred && larger.private_capacity_preferred,
      "larger private prefill retained its superseded retired Invocation");
  model.prepare_prefill_execution(small_prefill, q::TextCompletion::kStateOnly);
  h::require(small.slots[0].carrier.initialized() && !larger.slots[0].carrier.initialized() &&
      small.private_capacity_preferred && larger.private_capacity_preferred,
      "private prefill eviction changed its prepared-capacity preference or retained the previous workspace");
  const std::array prefill{std::size_t{3}};
  auto session = model.create_session(prefill, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 7, 0});
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  (void)q::evaluate(model, *session, prompt, "native-relay-prefill", {}, false);
  auto original = session->checkpoint();
  auto serial = model.create_session({}, original);
  std::array<std::int32_t, 3> expected{};
  for (auto &id : expected) {
    const auto pending = serial->head->next_id;
    id = q::evaluate(model, *serial, std::span(&pending, 1), "native-relay-serial", {}, false);
  }

  const auto before = q::generation_boundary(*session->head);
  const auto epoch = session->head.epoch();
  const auto pending = session->head->next_id;
  model.prepare_evaluation(*session, std::span(&pending, 1));
  auto *producer = model.active_step.get();
  const bool device_dependency = producer->phase->slots[producer->state_slot].carrier.native_source_dependencies();
  auto sources = model.prepare_submission_sources(*producer);
  model.submit_evaluations(*producer);
  h::require(sources->submission.valid() && sources->primary_output.valid(),
      "native model relay lost its actual output/complete-state Source");
  const auto &output = producer->phase->child->output(producer->phase->id_port);
  h::require(output.caller_owned() &&
      q::take(model.execution.resource_for(output.storage)) == model.execution.device(),
      "native original token writer escaped its device allocation domain");
  auto &slot = producer->phase->slots[producer->state_slot];
  h::require(producer->phase->delivery_id_port &&
      q::take(slot.io.readback(slot.data.source_id))->resource() == model.execution.device() &&
      q::take(slot.io.readback(slot.data.next_id))->resource() == model.execution.staging(),
      "native token delivery replaced the original device Source or lost staging");
  const std::array following{q::EvaluationRequest{session, {}}};
  auto &consumer = model.prepare_relay_evaluations(following, sources, 1);
  const auto report = q::take(consumer.execution.run->input_source_report());
  h::require(report.source_count && (device_dependency
          ? report.device_wait_count && !report.host_wait_count
          : report.host_wait_count && !report.device_wait_count),
      "native model relay differs from its cold-qualified Source dependency route");
  h::require(!consumer.execution.run->submit(),
      "ordinary native submit bypassed the original complete-state Source admission");
  model.submit_evaluations(consumer);
  h::require(session->head.epoch() == epoch && q::generation_boundary(*session->head) == before,
      "native submission published a model state or RNG cursor before owner completion");
  const auto lifetime = std::weak_ptr(sources);
  const auto first = q::finish_evaluation(model, *session, "native-relay-producer", {}, false);
  h::require(first == expected[0] && model.active_step.get() == &consumer,
      "native predecessor publication differs or lost its original successor");
  sources.reset();
  h::require(!lifetime.expired(), "native successor lost its still-borrowed original Source");
  const auto second = q::finish_evaluation(model, *session, "native-relay-consumer", {}, false);
  h::require(second == expected[1] && lifetime.expired() && !model.active_step && !model.successor_step,
      "native successor differs or retained consumed Source permissions after retirement");
  const auto next = session->head->next_id;
  const auto third = q::evaluate(model, *session, std::span(&next, 1), "native-relay-reuse", {}, false);
  h::require(third == expected[2], "ordinary reuse after native Source retirement changed the next token");
  auto storage = q::take(q::inference::HostStateStorage::create(model.execution, 2, 256u << 20));
  auto expected_image = model.export_session(*serial, storage);
  auto actual_image = model.export_session(*session, storage);
  const auto expected_bytes = h::read_image(*expected_image);
  h::same_image(expected_bytes, h::read_image(*actual_image));
  expected_image.reset(); actual_image.reset(); original.reset();
  h::retire(serial); h::retire(session);
  h::require(!storage->live_images() && !model.snapshot_versions.live && !model.carrier_borrowed,
      "native relay retained a complete state, image or physical carrier after retirement");
  std::cout << "PASS native_model_relay backend=" << h::backend_name(kind)
            << " sampling=" << (random ? "random" : "greedy")
            << " original_queue_count=0 original_state_sources=1 token_host_roundtrip=0"
            << " source_device_wait=" << (device_dependency ? 1 : 0)
            << " publication_before_owner_finish=0 serial_ids=3 complete_state_bytes=" << expected_bytes.total_bytes
            << " source_retired=1 next_reuse=equal\n" << std::flush;
}
} // namespace qwen_native_relay_case
