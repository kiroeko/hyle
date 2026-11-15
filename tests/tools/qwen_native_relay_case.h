#pragma once

// Uses the original complete-state image checker, not a token-only oracle.
#include "qwen_host_image_case.h"

namespace qwen_native_relay_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;

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
  h::require(q::native_ordinary_result_storage(model.execution,
      {.role = q::GraphRole::kText, .completion = q::TextCompletion::kEmitNext}).has_value(),
      "native Model has no independently owned original staging result domain");
  const std::array rows{std::size_t{3}, std::size_t{1}};
  model.prepare_family(rows);
  for (const auto count : rows) model.plan_replay_uses(count, 1, 8);
  model.prepare_execution(rows);
  model.prepare_relay_execution();
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
  auto sources = model.prepare_submission_sources(*producer);
  model.submit_evaluations(*producer);
  h::require(sources->submission.valid() && sources->primary_output.valid(),
      "native model relay lost its actual output/complete-state Source");
  const auto &output = producer->phase->child->output(producer->phase->id_port);
  h::require(output.caller_owned() &&
      q::take(model.execution.resource_for(output.storage)) == model.execution.staging(),
      "native token output is not its original explicit external allocation");
  const std::array following{q::EvaluationRequest{session, {}}};
  auto &consumer = model.prepare_relay_evaluations(following, sources, 1);
  const auto report = q::take(consumer.execution.run->input_source_report());
  h::require(report.source_count && report.host_wait_count && !report.device_wait_count,
      "native model relay concealed its original HostWait contract");
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
            << " publication_before_owner_finish=0 serial_ids=3 complete_state_bytes=" << expected_bytes.total_bytes
            << " source_retired=1 next_reuse=equal\n" << std::flush;
}
} // namespace qwen_native_relay_case
