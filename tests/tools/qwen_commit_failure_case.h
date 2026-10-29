#pragma once

#include "qwen_host_image_case.h"
#include "qwen_packed_prefill_case.h"
#include "qwen_logits_preflight_case.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

// A caller-owned observer refuses after a real R2 publication. The secondary
// failure is the actual cold-capacity check, never a simulated device fault.
namespace qwen_commit_failure_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace e = qwen_logits_preflight_case;
using h::require;

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require((kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda) &&
      geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "commit failure requires cached untrained GA/H64/V64/C16 CPU/CUDA weights");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 4, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 0, 1.F}, .seed = 123}, 0, {.primary = kind});
  const std::array shapes{std::size_t{1}, std::size_t{3}};
  model.prepare_family(shapes, false, 2);
  for (const auto rows : shapes) model.plan_replay_uses(rows, 2, 2);
  model.prepare_execution(shapes, 2);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 3}, {3, 2, 1}}};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  i::RequestWindow window; window.prepare(2, 2);
  for (std::size_t member = 0; member != 2; ++member) {
    requests.emplace_back(member ? "commit-peer-B" : "commit-observer-A", prompts[member], std::filesystem::path{}, 3);
    auto &request = requests.back();
    auto prompt = std::make_shared<std::vector<std::int32_t>>(prompts[member].begin(), prompts[member].end());
    request.prompt = *prompt; request.warm_inputs.push_back(std::move(prompt));
    request.input = q::cache::token_plan(request.prompt, 16);
    request.ledger = std::make_unique<q::GenerationLedger>(3, 1);
    request.round_emitted.resize(2); for (auto &row : request.round_emitted) row.reserve(1);
    request.turn_observations.resize(1); request.progress.initial_sampling = {123, 51 + member, 0};
    request.request_key = window.enqueue(member); window.admitted(request.request_key);
  }
  const std::array keys{requests[0].request_key, requests[1].request_key};
  std::size_t submissions = 0, refusal_callbacks = 0;
  const auto prepare = [&](std::size_t member) {
    return q::prepare_generation_call(model, requests[member], 3, nullptr, requests);
  };
  const auto submitted = [&] {
    require(model.active_step && model.active_step->submitted && model.active_step->execution.has_completion() &&
        model.active_step->members.size() == 2 && model.active_step->phase->family->role == q::GraphRole::kText,
        "commit failure did not submit its original ordinary R2 Graph");
    ++submissions;
  };
  {
    std::array calls{prepare(0), prepare(1)};
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted);
  }
  std::array<q::GenerationBoundary, 2> before;
  std::array<std::uint64_t, 2> before_epochs{};
  std::array<std::vector<std::int32_t>, 2> before_outputs;
  for (std::size_t member = 0; member != 2; ++member) {
    const auto &request = requests[member];
    require(request.ledger->committed_count == 1 && request.progress.observed_commits() == 1 &&
        request.progress.generated.size() == 1 && request.rounds.empty() && request.rounds.capacity() >= 2,
        "commit failure did not begin with two legally cold-prepared prefilled requests");
    before[member] = q::generation_boundary(*request.session->head);
    before_epochs[member] = request.session->head.epoch(); before_outputs[member] = request.progress.generated;
  }
  const auto preparations = [&] {
    std::size_t count = 0;
    for (const auto &[key, phase] : model.phases)
      for (const auto &slot : phase->slots.slots()) count += slot.preparation_uses;
    return count;
  };
  struct Published final {
    const q::cache::State *head = nullptr;
    std::uint64_t epoch = 0;
    q::GenerationBoundary boundary;
    std::vector<q::GenerationCommitRecord> records;
    h::ImageBytes image;
  };
  std::array<Published, 2> published;
  std::vector<q::MtpRoundResult> original_rounds;
  struct RestoreRounds final {
    q::GenerationRequest &request;
    std::vector<q::MtpRoundResult> &original;
    bool borrowed = false;
    void restore() noexcept { if (borrowed) { request.rounds.swap(original); borrowed = false; } }
    ~RestoreRounds() { restore(); }
  } restore{requests[0], original_rounds};
  const auto first_status = hyle::Status::FailedPrecondition("finite post-commit batch observer refusal");
  const auto observer = [&](const q::GenerationBatchFacts &batch) {
    ++refusal_callbacks;
    require(batch.members.size() == 2 && batch.generation && batch.input_rows == 1 &&
        !model.active_step && !model.successor_step && !model.carrier_borrowed,
        "observer refusal did not follow complete original R2 publication and retirement");
    for (std::size_t member = 0; member != 2; ++member) {
      auto &request = requests[member]; const auto &record = request.ledger->records[1];
      require(batch.members[member].request == &request && batch.members[member].disposition == q::EvaluationDisposition::kPublished &&
          request.ledger->committed_count == 2 && record.committed && record.requests == 2 && record.member_index == member &&
          record.before == before[member] && record.after == q::generation_boundary(*request.session->head) &&
          record.after.sampling == before[member].sampling.advanced(1) && record.accepted == 1 && record.emitted_count == 1 &&
          request.session->head.epoch() == before_epochs[member] + 1 && !request.session->pending &&
          request.progress.observed_commits() == 1 && request.progress.generated == before_outputs[member] && request.rounds.empty(),
          "observer refusal preceded a real head/ledger commit or consumed its host output early");
      published[member] = {request.session->head.get(), request.session->head.epoch(), record.after,
          request.ledger->records, p::capture(model, *request.session, storage)};
    }
    const auto retired = model.try_retire_episode();
    q::check(retired.status); require(retired.retired, "post-commit refusal lacks original Model retirement");
    // Only now temporarily withdraw A's actual reserved observation container.
    // Its original storage is restored after catching the primary exception.
    requests[0].rounds.swap(original_rounds); restore.borrowed = true;
    require(requests[0].rounds.empty() && !requests[0].rounds.capacity() && original_rounds.capacity() >= 2,
        "observer control did not create the exact secondary cold-capacity refusal");
    i::EvaluationFailure failure(first_status, std::size_t{1});
    failure.observe_retirement(retired); throw failure;
  };
  std::array calls{prepare(0), prepare(1)};
  bool first_preserved = false;
  try { q::execute_generation_calls(model, calls, false, nullptr, requests, submitted, nullptr, observer); }
  catch (const i::EvaluationFailure &failure) {
    first_preserved = failure.status() == first_status && !failure.status().has_native_error() &&
        failure.participant() == std::optional<std::size_t>{1} && failure.retirement_observed() &&
        failure.retirement().retired && failure.retirement().status.ok();
  }
  require(first_preserved && refusal_callbacks == 1 && submissions == 2 && restore.borrowed &&
      !model.active_step && !model.successor_step && !model.carrier_borrowed &&
      requests[0].progress.observed_commits() == 1 && requests[0].progress.generated == before_outputs[0] &&
      requests[0].rounds.empty() && !requests[0].rounds.capacity() && requests[1].progress.observed_commits() == 2 &&
      requests[1].progress.generated.size() == 2 && requests[1].rounds.size() == 1,
      "secondary A observation failure hid the first typed error or skipped B's original committed result");
  const auto preparations_before_retry = preparations();
  const auto auxiliary_before_retry = std::array{model.draft_program_graph_calls,
      model.prefix_decision_graph_calls, model.frontier_sampling_graph_calls};
  restore.restore();
  q::observe_generation_commits(requests[0]); // Consume the original record, never run the call again.
  for (std::size_t member = 0; member != 2; ++member) {
    auto &request = requests[member]; const auto &saved = published[member];
    const auto &record = request.ledger->records[1];
    auto expected = before_outputs[member]; expected.push_back(record.emitted[0]);
    require(request.session->head.get() == saved.head && request.session->head.epoch() == saved.epoch &&
        q::generation_boundary(*request.session->head) == saved.boundary && request.ledger->committed_count == 2 &&
        request.ledger->records.size() == saved.records.size() && request.progress.observed_commits() == 2 &&
        request.progress.generated == expected && request.progress.generated.back() == saved.boundary.next_id &&
        request.rounds.size() == 1 && request.rounds[0].emitted == std::vector<std::int32_t>{record.emitted[0]} &&
        request.request_key == keys[member] && window.matches(keys[member]) && !request.session->pending &&
        !request.session->failed && !request.session->stopping && request.session->retirement_status.ok(),
        "observation-only recovery changed a committed head, epoch, RNG, output, ledger or original request");
    for (std::size_t ordinal = 0; ordinal != saved.records.size(); ++ordinal)
      require(e::same_record(request.ledger->records[ordinal], saved.records[ordinal]),
          "observation-only recovery changed a committed or future ledger slot");
    h::same_image(saved.image, p::capture(model, *request.session, storage));
    q::observe_generation_commits(request);
    require(request.progress.observed_commits() == 2 && request.progress.generated == expected && request.rounds.size() == 1,
        "duplicate observation repeated an original output or family record");
  }
  require(submissions == 2 && preparations() == preparations_before_retry &&
      auxiliary_before_retry == std::array{model.draft_program_graph_calls,
          model.prefix_decision_graph_calls, model.frontier_sampling_graph_calls},
      "observation-only retry prepared or submitted new target, draft or sampling work");
  for (std::size_t member = 0; member != 2; ++member) {
    window.retire_active(keys[member]); h::retire(requests[member].session); window.release(keys[member]);
  }
  q::check(model.execution.collect_retired());
  require(!model.snapshot_versions.live && !storage->live_images() && window.active().empty() && !window.waiting_front() &&
      model.page_inventory.free_ids.size() == plan.max_pages,
      "commit failure retained a native state, original request or image after actual retirement");
  std::cout << "{\"status\":\"PASS\",\"case\":\"commit-failure\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"synthetic_untrained\":true,\"requests\":2,\"target_submissions\":" << submissions
      << ",\"postcommit_refusals\":1,\"first_typed_error_preserved\":true,\"original_participant\":1,"
         "\"actual_retirement_preserved\":true,\"secondary_observer_failure\":true,\"peer_commit_observed\":true,"
         "\"observation_retry_target_submissions\":0,\"duplicate_observation_noop\":true,\"complete_state_unchanged\":true,"
         "\"original_keys_preserved\":true,\"native_device_fault_injection\":false,\"continuous_pipeline\":false,"
         "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_commit_failure_case
