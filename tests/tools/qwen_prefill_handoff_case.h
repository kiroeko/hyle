#pragma once

#include "qwen_host_image_case.h"
#include "qwen_packed_prefill_case.h"
#include <thread>

namespace qwen_prefill_handoff_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
using h::require;

inline std::unique_ptr<q::Model> make_model(const std::filesystem::path &input, std::size_t depth = 3) {
  auto weights = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = weights.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "prefill handoff needs the cached GA fixture");
  const auto plan = q::cache::plan(geometry, weights.cache_row_bytes, 16, 1, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true, depth);
  auto owner = std::make_unique<q::Model>(std::move(weights), geometry, 16, plan, std::filesystem::path{},
      false, q::ReplayPolicy::kCommands, SIZE_MAX, q::Model::Sampling{}, 0,
      q::BackendSelection{.primary = q::BackendKind::kCuda});
  auto &model = *owner;
  require(model.supports_prefill_relay(0) == (depth >= 2),
      "prefill relay cold plan does not reflect the actual in-flight capacity");
  const std::array rows{std::size_t{2}};
  for (const auto completion : {q::TextCompletion::kStateOnly, q::TextCompletion::kEmitNext}) {
    require(model.prepare_prefill_family(rows, completion), "prefill handoff lost packed math");
    model.plan_replay_uses(2, 1, 32, q::GraphRole::kText, completion, 0, q::RowPacking::kPrefill);
  }
  model.bounded_prefill_rows = 2;
  model.freeze_shape_plan();
  model.prepare_shared_execution_workspace();
  return owner;
}

enum class Boundary { kNone, kCancel, kHostFailure };
struct Result final {
  std::vector<std::int32_t> ids;
  std::vector<q::GenerationCommitRecord> records;
  h::ImageBytes state;
  std::size_t successors = 0;
};

inline std::vector<std::vector<std::byte>> initialized_bytes(q::Model &model, const q::cache::State &head) {
  std::vector<std::vector<std::byte>> bytes;
  const auto read = [&](const q::c::ReadLease &lease) {
    const auto view = q::take(lease.view());
    bytes.emplace_back(view.size()); q::check(model.execution.read(view, bytes.back()));
  };
  for (const auto &frame : head.paged.frames) {
    const auto view = q::take(frame.view());
    std::vector<q::c::ReadLease> pages(view.layout().host().page_indices.size());
    q::check(view.page_prefixes(pages));
    for (const auto &page : pages) read(page);
  }
  for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer)
    if (model.geometry.is_gdn(layer)) { read(head.state[layer].history); read(head.state[layer].recurrent); }
  if (head.carry_hidden.valid()) read(head.carry_hidden);
  if (head.prior_hidden.valid()) read(head.prior_hidden);
  return bytes;
}

inline Result journey(q::Model &model, bool continuous, Boundary boundary = Boundary::kNone) {
  const std::array turns{i::GenerationTurn{{1, 2, 3, 5, 7, 11, 13, 2, 4, 6, 8, 10, 12}, 1}};
  std::vector<q::GenerationRequest> requests;
  requests.emplace_back("prefill-handoff", turns[0].prompt, std::filesystem::path{}, 1);
  auto &request = requests.front();
  request.progress.bind_turns(turns);
  request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turns[0].prompt));
  request.prompt = request.progress.begin_turn(*request.warm_inputs[0]);
  request.input = q::cache::token_plan(request.prompt, 16);
  request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns, 2), 1);
  request.turn_observations.resize(1);
  i::RequestWindow keys; keys.prepare(1, 1); request.request_key = keys.enqueue(0);
  q::GenerationContinuation continuation; continuation.prepare(1);
  auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    continuation.clear_pending(); continuation.reset();
    (void)model.abort_episode();
    if (request.session) {
      request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset();
    }
  }};
  Result result;
  bool interrupted = false;
  std::optional<h::ImageBytes> before;
  const q::cache::State *before_head = nullptr;
  std::uint64_t before_epoch = 0;
  const auto after_submit = [&] {
    if (!model.successor_step || !model.successor_step->submitted) return;
    ++result.successors;
    require((!request.ledger->committed_count && !request.progress.input_consumed()) || boundary == Boundary::kNone,
        "boundary test missed the first unpublished pair");
    if (boundary == Boundary::kCancel) {
      q::check(request.session->stop()); interrupted = true;
    } else if (boundary == Boundary::kHostFailure) {
      interrupted = true; throw std::runtime_error("injected prefill successor host failure");
    }
  };
  const auto observed = [&](const q::GenerationBatchFacts &facts) {
    require(facts.members.size() == 1 && !facts.generation, "prefill handoff published a foreign batch");
    const auto &call = facts.members[0];
    if (call.disposition != q::EvaluationDisposition::kPublished) return;
    const auto &record = request.ledger->records[request.ledger->committed_count - 1];
    require(record.before.length == call.input_begin && record.after.length == call.input_begin + call.ids().size() &&
        record.before.sampling == record.after.sampling &&
        (call.prefill_complete || (!record.emitted_count && record.after.next_id == -1)),
        "prefill handoff changed order, length, RNG or state-only output");
  };
  for (std::size_t pumps = 0; !request.progress.turn_done && !interrupted; ++pumps) {
    require(pumps < 256, "prefill handoff did not converge");
    try {
      if (continuation.pending_calls.empty()) {
        std::array calls{q::prepare_generation_call(model, request, 1, nullptr, requests, 1, 2)};
        if (!before) {
          before = qwen_packed_prefill_case::capture(model, *request.session, storage);
          before_head = request.session->head.get();
          before_epoch = request.session->head.epoch();
        }
        q::execute_generation_calls(model, calls, false, nullptr, requests, after_submit,
            &continuation, observed, true);
      } else {
        auto calls = std::span(continuation.pending_calls.data(), continuation.pending_calls.size());
        q::SynchronousGenerationStepAdapter adapter(model, calls, false, true, 0, after_submit, observed, &continuation);
        const auto disposition = i::drive_steps(adapter,
            {.resume = true, .defer_successor = true, .cursor = &continuation.drive_cursor});
        if (disposition == i::StepDriveResult::kCompleted) continuation.clear_pending();
      }
    } catch (const std::runtime_error &error) {
      if (!interrupted || boundary != Boundary::kHostFailure ||
          std::string_view(error.what()) != "injected prefill successor host failure") throw;
    }
    if (!continuation.pending_calls.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (boundary != Boundary::kNone) {
    require(interrupted && result.successors == 1 && before &&
        !model.active_step && !model.successor_step && !request.session->pending &&
        request.session->head.get() == before_head && request.session->head.epoch() == before_epoch && !request.ledger->committed_count &&
        !request.progress.input_consumed() && request.progress.generated.empty(),
        "cancelled/failed prefill pair advanced an unpublished Head or request");
    require(before->payloads == initialized_bytes(model, *request.session->head),
        "cancelled/failed prefill pair changed initialized committed bytes");
  } else {
    require(request.progress.turn_done && request.progress.generated.size() == 1 &&
        request.ledger->committed_count == 7 && request.session->length() == 13 &&
        request.session->head.epoch() == before_epoch + 7 &&
        (!continuous || result.successors >= 5), "prefill chain fell back, skipped a chunk or lost its partial tail");
    result.ids = request.progress.generated;
    result.records.assign(request.ledger->records.begin(), request.ledger->records.begin() + request.ledger->committed_count);
    result.state = qwen_packed_prefill_case::capture(model, *request.session, storage);
  }
  continuation.clear_pending(); continuation.reset();
  h::retire(request.session);
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size(),
      "prefill handoff retained native work, versions or page owners");
  return result;
}

inline int run(const std::filesystem::path &input, q::BackendKind backend) {
  if (backend != q::BackendKind::kCuda) return 77;
  auto serial_owner = make_model(input, 1);
  const auto serial = journey(*serial_owner, false);
  auto owner = make_model(input);
  const auto continuous = journey(*owner, true);
  require(serial.ids == continuous.ids && serial.records.size() == continuous.records.size(),
      "prefill handoff changed generated IDs or committed count");
  for (std::size_t index = 0; index < serial.records.size(); ++index) {
    const auto &a = serial.records[index], &b = continuous.records[index];
    require(a.same_result(b) && std::ranges::equal(a.ids(), b.ids()) &&
        std::ranges::equal(a.committed_ids(), b.committed_ids()), "prefill handoff changed an ordered record");
  }
  require(h::same_boundary(serial.state.boundary, continuous.state.boundary) &&
      serial.state.layout == continuous.state.layout && serial.state.total_bytes == continuous.state.total_bytes &&
      serial.state.payloads == continuous.state.payloads, "prefill handoff changed exact logical state bytes");
  for (const auto boundary : {Boundary::kCancel, Boundary::kHostFailure}) {
    auto interrupted = make_model(input);
    (void)journey(*interrupted, true, boundary);
  }
  std::cout << "prefill-handoff PASS: seven chunks, three slots, partial tail, exact state/IDs/records, cancellation and host failure\n";
  return 0;
}
} // namespace qwen_prefill_handoff_case
