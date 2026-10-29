#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "../../engine/inference/output_delivery.h"

// Included after the actual Model. One finite CPU journey joins the natural
// request owner, partial-prefix/chunk admission, host parking and output credit.
namespace qwen_host_image_request_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

inline void prepare_request(q::GenerationRequest &request,
    std::span<const i::GenerationTurn> turns, std::size_t stream,
    std::size_t imported_records = 0) {
  request.progress.bind_turns(turns);
  request.progress.initial_sampling = {123, stream, 0};
  request.ledger = std::make_unique<q::GenerationLedger>(
      q::generation_record_capacity(turns, 2, imported_records), 1);
  request.round_emitted.resize(request.progress.maximum_outputs() - 1);
  for (auto &emitted : request.round_emitted) emitted.reserve(1);
  request.turn_observations.resize(turns.size());
  for (const auto &turn : turns)
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turn.prompt));
  request.input = q::cache::token_plan(turns.front().prompt, 16);
}

inline bool same_record(const q::GenerationCommitRecord &a,
    const q::GenerationCommitRecord &b) noexcept {
  return a.same_result(b) && a.committed == b.committed && a.imported == b.imported &&
      a.user_output == b.user_output && a.turn == b.turn && a.input_owner == b.input_owner &&
      a.pending_input == b.pending_input && std::ranges::equal(a.prompt_ids, b.prompt_ids) &&
      a.step_seconds == b.step_seconds && a.endpoint == b.endpoint;
}

inline std::vector<std::vector<std::byte>> state_bytes(q::Model &model,
    const q::cache::State &head) {
  std::vector<std::vector<std::byte>> result;
  const auto read = [&](const q::c::ReadLease &lease) {
    const auto view = q::take(lease.view());
    result.emplace_back(view.size());
    q::check(model.execution.read(view, result.back()));
  };
  for (std::size_t layer = 0; layer < head.paged.frames.size() / 2; ++layer)
    for (const auto *frame : {&head.paged.frames[2 * layer], &head.paged.frames[2 * layer + 1]}) {
      const auto footprint = q::take(frame->view());
      std::vector<q::c::ReadLease> pieces(footprint.layout().host().page_indices.size());
      q::check(footprint.page_prefixes(pieces));
      for (const auto &piece : pieces) read(piece);
    }
  if (head.carry_hidden.valid()) read(head.carry_hidden);
  return result;
}

inline std::vector<std::vector<std::byte>> image_bytes(const q::cache::HostImage &image) {
  require(image.ready() && image.payload_count() == image.layout.size(),
      "parked image was visible before its complete payload publication");
  std::vector<std::vector<std::byte>> result;
  for (std::size_t index = 0; index < image.payload_count(); ++index) {
    result.emplace_back(image.layout[index].bytes);
    q::check(q::take(image.payload(index).view()).read(result.back()));
  }
  return result;
}

inline void retired_episode(q::Model &model) {
  const auto observed = model.try_retire_episode();
  q::check(observed.status);
  require(observed.retired && !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "request journey proceeded before its original native episode retired");
}

inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A",
      "host-image request case needs the cached untrained A/H64/FFN128/V64 fixture");
  const q::cache::KvRowBytes rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads),
          static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads),
          static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, {.target = rows}, 16, 1, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu});
  const std::array emit_rows{std::size_t{1}, std::size_t{2}};
  const std::array state_rows{std::size_t{2}};
  model.prepare_family(emit_rows);
  model.prepare_family(state_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kStateOnly);
  for (auto count : emit_rows) model.plan_replay_uses(count, 1, 16);
  model.plan_replay_uses(2, 1, 16, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  model.prepare_execution(emit_rows);
  model.prepare_execution(state_rows, 1, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  require(!model.find_child(1, 2, q::GraphRole::kText, false,
          q::TextCompletion::kStateOnly, 0, q::RowPacking::kPrefill),
      "CPU journey unexpectedly prepared a packed prefill family");

  q::cache::PrefixCache prefix(1);
  std::size_t target_calls = 0, request_waves = 0, a_calls = 0, restored_images = 0;
  const auto submitted = [&] {
    require(model.active_step && model.active_step->submitted &&
        model.active_step->members.size() == 1 &&
        model.active_step->phase->family->packing == q::RowPacking::kRectangular,
        "CPU journey did not submit its actual single-member rectangular Graph");
    ++target_calls;
  };
  const std::array seed_turns{i::GenerationTurn{{1, 2}, 1}};
  q::GenerationRequest seed("prefix", seed_turns[0].prompt, {}, 1);
  prepare_request(seed, seed_turns, 99);
  seed.prompt = seed.progress.begin_turn(*seed.warm_inputs[0]);
  std::array seed_calls{q::prepare_generation_call(model, seed, 1, &prefix, {}, 1, 2)};
  q::execute_generation_calls(model, seed_calls, false, &prefix, {}, submitted);
  q::publish_generation_prompt(model, seed_calls[0], &prefix);
  require(prefix.size() == 1 && seed.session->head->sampling.draw == 1 &&
      seed.ledger->committed_count == 1 && seed.progress.request_done(),
      "real seed Graph did not publish the complete two-token prefix");
  const auto seed_retirement = q::retire_generation_request(seed);
  q::check(seed_retirement.status);
  require(seed_retirement.retired, "prefix publisher did not actually retire its Session");
  seed.session.reset(); retired_episode(model);

  const std::array a_turns{i::GenerationTurn{{1, 2, 3, 4, 5, 6, 7}, 2}};
  const std::array peer_turns{i::GenerationTurn{{8}, 1}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(2);
  requests.emplace_back("A", a_turns[0].prompt, std::filesystem::path{}, 2);
  requests.emplace_back("B", peer_turns[0].prompt, std::filesystem::path{}, 1);
  prepare_request(requests[0], a_turns, 20, 1);
  prepare_request(requests[1], peer_turns, 21);
  auto &a = requests[0]; auto &b = requests[1];
  std::array delivery{i::OutputDelivery(1), i::OutputDelivery(1)};
  i::RequestLoop loop; loop.prepare(2, 1);
  auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  std::vector<std::size_t> evictions; evictions.reserve(1);
  q::GenerationBoundary parked_boundary;
  std::vector<std::vector<std::byte>> parked_state;
  std::array<q::GenerationCommitRecord, 2> parked_records;
  const std::int32_t *original_input = nullptr;
  const q::cache::ReplayHistory *original_history = nullptr;
  std::weak_ptr<q::Session> old_session;
  std::weak_ptr<const q::cache::HostImage> parked_image;
  const auto original_records = [&] {
    require(same_record(a.ledger->records[0], parked_records[0]) &&
        same_record(a.ledger->records[1], parked_records[1]),
        "parking, import or later slices changed an already authoritative ledger record");
  };
  const auto execute = [&](std::span<const i::RequestKey> active) {
    require(active.size() == 1 && active[0] == requests[active[0].slot].request_key &&
        loop.matches(active[0]), "natural execution lost its original output/request key");
    auto &request = requests[active[0].slot];
    const bool is_a = active[0].slot == 0;
    if (is_a && a_calls == 1) {
      require(request.session && !request.host_image && parked_image.expired() &&
          !storage->live_images() && q::generation_boundary(*request.session->head) == parked_boundary &&
          state_bytes(model, *request.session->head) == parked_state &&
          request.progress.input().data() == original_input && request.prompt.data() == original_input &&
          request.progress.input_consumed() == 4 && request.progress.generated.empty() &&
          request.progress.observed_commits() == 2 && request.ledger->committed_count == 2 &&
          request.history.get() == original_history && request.prefix_hits == 1 &&
          !request.restore_count && !request.restored.graph_calls,
          "host restore changed complete state, consumed input, RNG, ledger or request ownership");
      original_records(); ++restored_images;
    }
    const auto credit = loop.credit(active[0], request.progress.generated.size());
    std::array calls{q::prepare_generation_call(model, request, request.progress.output_limit(),
        &prefix, requests, credit, 2)};
    if (is_a && a_calls < 3) {
      const std::array begin{std::size_t{2}, std::size_t{4}, std::size_t{6}};
      require(!calls[0].generation && !calls[0].full_hit && calls[0].input_begin == begin[a_calls] &&
          calls[0].ids().size() == (a_calls < 2 ? 2 : 1) &&
          calls[0].completion == (a_calls < 2 ? q::TextCompletion::kStateOnly : q::TextCompletion::kEmitNext),
          "partial-prefix/chunk admission repeated input or changed its real completion mode");
    }
    q::execute_generation_calls(model, calls, false, &prefix, requests, submitted);
    q::publish_generation_prompt(model, calls[0], &prefix);
    retired_episode(model); ++request_waves;
    if (is_a && a_calls++ == 0) {
      require(a.reused_tokens == 2 && a.prefix_hits == 1 && !a.prefix_full_hits &&
          a.progress.input_consumed() == 4 && !a.progress.prefilled && a.progress.generated.empty() &&
          a.session->head->next_id == -1 && a.session->head->sampling.draw == 0 &&
          a.ledger->committed_count == 2 && a.progress.observed_commits() == 2 &&
          a.ledger->records[0].imported && !a.ledger->records[0].user_output &&
          !a.ledger->records[0].emitted_count && !a.ledger->records[1].user_output,
          "partial prefix or StateOnly chunk emitted historical output or consumed RNG");
      parked_boundary = q::generation_boundary(*a.session->head);
      parked_state = state_bytes(model, *a.session->head);
      std::copy_n(a.ledger->records.begin(), 2, parked_records.begin());
      original_input = a.progress.input().data(); original_history = a.history.get(); old_session = a.session;
      loop.request_yield(); // The already queued peer wins the original FIFO.
    }
  };
  q::SynchronousGenerationWave wave{model, execute};
  q::GenerationRequestAdapter adapter{model, &prefix, std::span(requests), false, &evictions, wave, storage};
  a.request_key = loop.enqueue(0, &delivery[0]); b.request_key = loop.enqueue(1, &delivery[1]);
  const auto key_a = a.request_key, key_b = b.request_key;
  const auto first = loop.pump_once(adapter);
  require(first.admitted == 1 && first.executed && !first.completed && loop.active().empty() &&
      loop.waiting_front() == key_b && target_calls == 2 && !a.host_image &&
      loop.matches(key_a) && loop.matches(key_b),
      "StateOnly yield did not preserve A behind the queued peer");
  const auto peer = loop.pump_once(adapter);
  retired_episode(model);
  require(peer.admitted == 1 && peer.executed && !peer.completed && !peer.waiting_retirement &&
      peer.waiting_output && !a.session && a.host_image && a.host_image->ready() &&
      a.host_image->boundary.length == parked_boundary.length &&
      a.host_image->boundary.next_id == -1 && a.host_image->boundary.sampling == parked_boundary.sampling &&
      storage->live_images() == 1 && old_session.expired() && a.idle_evictions == 1 &&
      evictions == std::vector<std::size_t>{0} && !b.session && b.progress.request_done() &&
      b.progress.generated.size() == 1 && delivery[1].pending(b.progress.generated).size() == 1 &&
      b.ledger->committed_count == 1 && b.ledger->records[0].after.sampling.draw == 1 &&
      loop.matches(key_a) && loop.matches(key_b) && loop.waiting_front() == key_a && target_calls == 3,
      "peer admission did not automatically park and really retire only A's native owner");
  parked_image = a.host_image;
  const auto *image_identity = a.host_image.get(); const auto image_payload = image_bytes(*a.host_image);
  original_records();

  // An actual finite page shortage rejects restoration before any new target
  // submit. Returning the original reservations must permit the same FIFO key.
  require(prefix.evict_one() && !prefix.size() && !model.snapshot_versions.live,
      "peer checkpoint did not release its original committed version");
  std::vector<hyle::inference::PagedPageReservation> blockers; blockers.reserve(plan.max_pages);
  while (!model.page_inventory.free_ids.empty())
    blockers.push_back(q::take(model.page_inventory.acquire()));
  require(blockers.size() == plan.max_pages, "page shortage did not occupy the actual finite original inventory");
  bool capacity_rejected = false;
  try { (void)loop.pump_once(adapter); }
  catch (const i::EvaluationFailure &failure) {
    if (!failure.status().IsResourceExhausted() || failure.status().has_native_error()) throw;
    capacity_rejected = true;
  }
  retired_episode(model);
  require(capacity_rejected && !a.session && a.host_image.get() == image_identity &&
      image_bytes(*a.host_image) == image_payload && storage->live_images() == 1 &&
      !model.snapshot_versions.live && loop.matches(key_a) && loop.waiting_front() == key_a &&
      a.progress.input().data() == original_input && a.progress.input_consumed() == 4 &&
      a.ledger->committed_count == 2 && a.progress.observed_commits() == 2 &&
      a.progress.generated.empty() && target_calls == 3 && request_waves == 2,
      "failed host import cleared its image/key or consumed input, output, RNG or target work");
  original_records(); blockers.clear();
  require(model.page_inventory.free_ids.size() == plan.max_pages,
      "original failed-import page reservations did not return their capabilities");
  const auto resumed = loop.pump_once(adapter);
  require(resumed.admitted == 1 && resumed.executed && !resumed.completed && restored_images == 1 &&
      a.progress.input_consumed() == 6 && a.progress.generated.empty() && !a.progress.prefilled &&
      a.session->head->sampling.draw == 0 && a.ledger->committed_count == 3 &&
      a.progress.observed_commits() == 3 && target_calls == 4,
      "same-key host import did not continue its next StateOnly slice exactly once");
  const auto emitted = loop.pump_once(adapter);
  require(emitted.executed && emitted.waiting_output && !emitted.completed &&
      a.progress.input_consumed() == 7 && a.progress.prefilled && !a.progress.turn_done &&
      a.progress.generated.size() == 1 && a.session->head->sampling.draw == 1 &&
      a.ledger->committed_count == 4 && a.progress.observed_commits() == 4 &&
      !delivery[0].available(1) && delivery[0].pending(a.progress.generated).front() == a.session->head->next_id &&
      loop.active().empty() && loop.matches(key_a) && loop.matches(key_b) && target_calls == 5,
      "final prefill output did not fill only its original request's delivery credit");
  const auto credited_head = q::generation_boundary(*a.session->head);
  const auto credited_epoch = a.session->head.epoch();
  const auto blocked = loop.pump_once(adapter);
  require(!blocked.executed && blocked.waiting_output && !blocked.completed && target_calls == 5 &&
      q::generation_boundary(*a.session->head) == credited_head && a.session->head.epoch() == credited_epoch &&
      a.progress.input_consumed() == 7 && a.progress.observed_commits() == 4 &&
      a.ledger->committed_count == 4 && delivery[0].acknowledged() == 0 && delivery[1].acknowledged() == 0,
      "backpressure advanced the head/RNG, reconsumed input or acknowledged a peer output");
  require(loop.acknowledge(key_a, 1, adapter) && loop.matches(key_b) &&
      !delivery[1].acknowledged(), "A acknowledgement changed its peer's output ownership");
  const auto completed = loop.pump_once(adapter);
  retired_episode(model);
  require(completed.executed && completed.waiting_output && !completed.completed &&
      !completed.waiting_retirement && !a.session && a.progress.request_done() &&
      a.progress.generated.size() == 2 && a.progress.observed_commits() == 5 &&
      a.ledger->committed_count == 5 && a.ledger->records[4].after.length == 8 &&
      a.ledger->records[4].after.sampling.draw == 2 &&
      a.ledger->records[4].pending_input == a.progress.generated[0] &&
      delivery[0].pending(a.progress.generated).size() == 1 &&
      delivery[0].pending(a.progress.generated)[0] == a.progress.generated[1] &&
      loop.matches(key_a) && loop.matches(key_b) && target_calls == 6 && request_waves == 5 &&
      a.prefill_graph_calls == 3 && a.prefill_graph_tokens == 5 && !a.restore_count &&
      !a.restored.graph_calls && !a.host_image && !storage->live_images(),
      "ACK continuation replayed prompt/history or released output keys before native retirement and delivery");
  original_records();
  require(a.ledger->records[2].input_begin == 4 && a.ledger->records[2].ids().size() == 2 &&
      a.ledger->records[2].completion == q::TextCompletion::kStateOnly &&
      a.ledger->records[3].input_begin == 6 && a.ledger->records[3].ids().size() == 1 &&
      a.ledger->records[3].prefill_complete && a.ledger->records[4].generation &&
      a.progress.input().data() == original_input && a.progress.input_consumed() == 7,
      "committed slice boundaries no longer describe the original once-consumed prompt");
  require(loop.acknowledge(key_b, 1, adapter) && loop.acknowledge(key_a, 2, adapter),
      "original terminal keys rejected their own final delivery prefixes");
  const auto released = loop.pump_once(adapter);
  require(released.completed == 2 && !released.executed && !released.waiting_output &&
      !released.waiting_retirement && !loop.matches(key_a) && !loop.matches(key_b) &&
      loop.active().empty() && !loop.waiting_front() && target_calls == 6 &&
      !loop.acknowledge(key_a, 2, adapter),
      "terminal keys were not released exactly once after their original ACK and native retirement");
  while (prefix.evict_one()) {}
  retired_episode(model);
  require(!model.snapshot_versions.live && !storage->live_images(),
      "completed host-image request journey retained a model version or parked image");
  std::cout << "{\"status\":\"PASS\",\"fixture\":\"untrained_A_H64\","
      "\"natural_request_loop\":true,\"N\":1,\"partial_prefix_rows\":2,"
      "\"state_only_chunks\":2,\"idle_evictions\":1,\"host_restores\":1,"
      "\"restore_capacity_status\":\"ResourceExhausted\",\"failed_import_preserved_image\":true,"
      "\"restored_state_byte_equal\":true,\"delivery_capacity\":1,"
      "\"request_waves\":5,\"target_calls\":6,\"A_commits\":5,\"B_commits\":1,"
      "\"A_draw\":2,\"B_draw\":1,\"ledger_rebuild_graph_calls\":0,"
      "\"final_versions\":0,\"final_images\":0}\n";
  return 0;
}
} // namespace qwen_host_image_request_case
