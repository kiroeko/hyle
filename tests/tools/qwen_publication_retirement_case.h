#pragma once

#include "qwen_commit_failure_case.h"

#include <array>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

// An original Complete callback refuses after real Graph retirement and state
// extraction but before publication. The same non-std exception later leaves
// already published peers intact when an original result observer refuses.
namespace qwen_publication_retirement_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace e = qwen_logits_preflight_case;
using h::require;

struct Refusal final { const void *identity; };

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require((kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda) &&
      geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "publication retirement requires cached untrained GA/H64/V64/C16 weights");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 4, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 0, 1.F}, .seed = 123}, 0, {.primary = kind});
  const std::array shapes{std::size_t{1}, std::size_t{3}};
  model.prepare_family(shapes, false, 2);
  for (const auto rows : shapes) model.plan_replay_uses(rows, 2, 4);
  // Read-only checkpoint readers use this existing one-row family; they never
  // submit a separate Graph or create another state version.
  const std::array reader_shape{std::size_t{1}};
  model.prepare_family(reader_shape);
  model.plan_replay_uses(1, 1, 1);
  model.prepare_execution(shapes, 2);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 3}, {3, 2, 1}}};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  i::RequestWindow window; window.prepare(2, 2);
  for (std::size_t member = 0; member != prompts.size(); ++member) {
    requests.emplace_back(member ? "publication-peer-B" : "publication-refusal-A", prompts[member],
        std::filesystem::path{}, 3);
    auto &request = requests.back();
    auto prompt = std::make_shared<std::vector<std::int32_t>>(prompts[member].begin(), prompts[member].end());
    request.prompt = *prompt; request.warm_inputs.push_back(std::move(prompt));
    request.input = q::cache::token_plan(request.prompt, 16);
    request.ledger = std::make_unique<q::GenerationLedger>(3, 1);
    request.round_emitted.resize(2); for (auto &row : request.round_emitted) row.reserve(1);
    request.turn_observations.resize(1); request.progress.initial_sampling = {123, 71 + member, 0};
    request.request_key = window.enqueue(member); window.admitted(request.request_key);
  }
  const std::array keys{requests[0].request_key, requests[1].request_key};
  std::size_t submissions = 0, completion_refusals = 0, observer_refusals = 0, cleared_publication_batches = 0;
  const auto prepare = [&](std::size_t member) {
    return q::prepare_generation_call(model, requests[member], 3, nullptr, requests);
  };
  const auto submitted = [&] {
    require(model.active_step && model.active_step->submitted && model.active_step->execution.has_completion() &&
        model.active_step->members.size() == 2 && model.active_step->phase->family->role == q::GraphRole::kText,
        "publication retirement did not submit its original ordinary R2 Graph");
    ++submissions;
  };
  {
    std::array calls{prepare(0), prepare(1)};
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted);
  }
  std::array before{e::save(model, requests[0], storage), e::save(model, requests[1], storage)};
  for (const auto &saved : before)
    require(saved.commits == 1 && saved.observed == 1 && saved.generated.size() == 1,
        "publication retirement lacks its real prefilled committed boundary");

  // Use the same original prepare/submit/finish stages as execute_generation_calls,
  // exposing only its supported unpublished Complete and published Observe hooks.
  const auto finish_direct = [&](auto complete, auto observe) {
    std::array calls{prepare(0), prepare(1)};
    std::array<q::EvaluationRequest, 2> inputs;
    std::array<q::EvaluationOutput, 2> outputs;
    for (std::size_t member = 0; member != calls.size(); ++member) {
      auto &call = calls[member]; auto &request = *call.request;
      require(call.generation && call.role == q::GraphRole::kText && call.ids().size() == 1,
          "publication refusal must use the actual pending-only ordinary boundary");
      q::prepare_generation_record(*request.ledger, *request.session, call.role, call.ids(), true,
          call.quota, request.progress.turn_index, {}, call.maximum, call.completion,
          call.input_begin, call.prefill_complete);
      inputs[member] = {request.session, call.ids(), &request.ledger->stop_policy(), call.logits,
          call.conditional_logits, request.request_key};
      outputs[member] = q::evaluation_output(model, request.session, {call.name, &request.trace, false},
          &call.step_seconds, &request.endpoint);
      outputs[member].disposition = &call.disposition;
    }
    model.prepare_evaluations(inputs, nullptr, q::GraphRole::kText);
    model.submit_evaluations(); submitted();
    q::GenerationPublicationBatch publication({{requests[0].ledger.get(),nullptr},{requests[1].ledger.get(),nullptr}});
    std::array<std::int32_t, 2> next{};
    bool completed = false;
    try {
      model.finish_selected_evaluations(outputs, next,
          [](std::span<const std::int32_t> ids, q::StepState &step, std::span<std::int32_t> selected) {
            for (std::size_t member = 0; member != step.members.size(); ++member) {
              step.members[member].accepted_tokens = step.members[member].input_tokens;
              selected[member] = ids[member];
            }
          }, [&](q::StepState &step) { complete(step,publication); completed = true; }, std::move(observe), publication);
    } catch (...) {
      if (!completed) {
        for (std::size_t member = 0; member != requests.size(); ++member)
          require(!publication.record(member) && !publication.ticket(member),
              "retired failed finish retained a call-scoped record borrow or publication ticket");
        ++cleared_publication_batches;
      }
      throw;
    }
  };
  const int identity = 19;
  const Refusal refusal{&identity};
  std::shared_ptr<q::StepState> failed_step;
  bool caught = false;
  try {
    finish_direct([&](q::StepState &step,q::GenerationPublicationBatch &publication) {
      ++completion_refusals;
      require(model.active_step.get() == &step && step.submitted && step.members.size() == 2 &&
          !step.execution.run->in_flight(), "Complete refusal must follow actual original Graph retirement");
      failed_step = model.active_step;
      for (std::size_t member = 0; member != step.members.size(); ++member) {
        const auto &original = step.members[member]; const auto &saved = before[member];
        const auto *record = publication.record(member); const auto *ticket = publication.ticket(member);
        require(original.session == requests[member].session && original.base == saved.head &&
            original.base_epoch == saved.epoch && original.next && original.accepted_tokens == 1 &&
            original.next->length == saved.checkpoint->length + 1 && ticket && ticket->valid() &&
            record == &requests[member].ledger->records[saved.commits] &&
            !record->committed && requests[member].ledger->committed_count == saved.commits,
            "Complete refusal did not retain the exact unpublished original state and record");
      }
      throw refusal;
    }, [](std::size_t, const q::CompletedEvaluationView &) {
      throw std::runtime_error("unpublished refusal reached a result observer");
    });
  } catch (const Refusal &actual) { caught = actual.identity == refusal.identity; }
  require(caught && completion_refusals == 1 && !observer_refusals && submissions == 2,
      "unpublished refusal changed its original non-std exception or repeated native work");
  require(failed_step && failed_step->members.size() == 2,
      "unpublished refusal lost its original call-scoped Step alias");
  require(cleared_publication_batches == 1,
      "failed finish did not release its original call-scoped record borrows and publication tickets");
  failed_step.reset();
  const auto retired = model.try_retire_episode(); q::check(retired.status);
  require(retired.retired && !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "unpublished refusal did not retire the actual original native owner");
  q::GenerationCommitRecord empty;
  empty.emitted.clear(); empty.accepted_inputs.clear(); empty.verification_inputs.clear();
  for (std::size_t member = 0; member != requests.size(); ++member) {
    const auto &request = requests[member]; const auto &saved = before[member];
    require(request.session->head.get() == saved.head && request.session->head.epoch() == saved.epoch &&
        request.ledger->committed_count == saved.commits && request.progress.observed_commits() == saved.observed &&
        request.progress.generated == saved.generated && !request.session->pending && !request.ledger->successor_ordinal() &&
        q::generation_boundary(*request.session->head) == q::generation_boundary(*saved.checkpoint),
        "unpublished refusal changed a Head, epoch, RNG, output or committed ledger prefix");
    for (std::size_t ordinal = 0; ordinal != request.ledger->records.size(); ++ordinal)
      require(e::same_record(request.ledger->records[ordinal], ordinal == saved.commits ? empty : saved.records[ordinal]),
          "retired refusal did not clear only its exact original uncommitted record");
    h::same_image(saved.image, p::capture(model, *request.session, storage));
    p::check_old(model, saved.checkpoint, saved.image, storage);
  }
  {
    std::array calls{prepare(0), prepare(1)};
    for (std::size_t member = 0; member != calls.size(); ++member)
      require(calls[member].pending == before[member].checkpoint->next_id,
          "retry changed the original pending-only input");
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted);
  }
  std::array<std::uint64_t, 2> retry_epochs{};
  std::array<q::GenerationBoundary, 2> retry_boundaries;
  std::array<std::vector<q::GenerationCommitRecord>, 2> retry_records;
  for (std::size_t member = 0; member != requests.size(); ++member) {
    auto &request = requests[member];
    require(request.ledger->committed_count == 2 && request.progress.observed_commits() == 2 &&
        request.progress.generated.size() == 2 && request.session->head.epoch() == before[member].epoch + 1 &&
        request.session->head->sampling == before[member].checkpoint->sampling.advanced(1),
        "same-input retry could not prepare and publish its original next record");
    retry_epochs[member] = request.session->head.epoch();
    retry_boundaries[member] = q::generation_boundary(*request.session->head);
    retry_records[member] = request.ledger->records;
    p::check_old(model, before[member].checkpoint, before[member].image, storage);
  }
  caught = false;
  try {
    finish_direct([](q::StepState &,q::GenerationPublicationBatch &) {}, [&](std::size_t member, const q::CompletedEvaluationView &completed) {
      ++observer_refusals;
      require(member == 0 && completed.requests == 2 && !model.active_step && !model.successor_step,
          "late refusal did not observe the original fully published R2 result");
      for (std::size_t peer = 0; peer != requests.size(); ++peer) {
        const auto &request = requests[peer]; const auto &record = request.ledger->records[2];
        require(request.ledger->committed_count == 3 && record.committed && record.requests == 2 &&
            record.member_index == peer && record.before == retry_boundaries[peer] &&
            record.after.sampling == retry_boundaries[peer].sampling.advanced(1) &&
            request.session->head.epoch() == retry_epochs[peer] + 1 &&
            request.progress.observed_commits() == 2 && request.progress.generated.size() == 2,
            "late refusal preceded a peer's real Head/record commit or observed it early");
      }
      throw refusal;
    });
  } catch (const Refusal &actual) { caught = actual.identity == refusal.identity; }
  require(caught && observer_refusals == 1 && submissions == 4 &&
      !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "late observer refusal changed its exception or failed to retire the original result");
  const auto preparations = [&] {
    std::size_t count = 0;
    for (const auto &[key, phase] : model.phases) {
      (void)key;
      for (const auto &slot : phase->slots.slots()) count += slot.preparation_uses;
    }
    return count;
  };
  const auto prepared_before_observation = preparations();
  const auto auxiliary_before_observation = std::array{model.draft_program_graph_calls,
      model.prefix_decision_graph_calls, model.frontier_sampling_graph_calls};
  for (std::size_t member = 0; member != requests.size(); ++member) {
    auto &request = requests[member];
    const auto *head = request.session->head.get(); const auto epoch = request.session->head.epoch();
    const auto boundary = q::generation_boundary(*head); const auto records = request.ledger->records;
    const auto image = p::capture(model, *request.session, storage);
    for (std::size_t ordinal = 0; ordinal != 2; ++ordinal)
      require(e::same_record(records[ordinal], retry_records[member][ordinal]),
          "late observer refusal discarded an already committed peer prefix");
    q::observe_generation_commits(request);
    require(request.progress.observed_commits() == 3 && request.progress.generated.size() == 3 &&
        request.progress.generated.back() == boundary.next_id && request.progress.turn_done && request.rounds.size() == 2 &&
        request.session->head.get() == head && request.session->head.epoch() == epoch &&
        q::generation_boundary(*request.session->head) == boundary && request.ledger->committed_count == 3,
        "original-ordinal observation changed published state or lost an actual output");
    q::observe_generation_commits(request);
    require(request.progress.observed_commits() == 3 && request.progress.generated.size() == 3 && request.rounds.size() == 2,
        "duplicate observation repeated the original committed output");
    for (std::size_t ordinal = 0; ordinal != records.size(); ++ordinal)
      require(e::same_record(request.ledger->records[ordinal], records[ordinal]),
          "observation-only recovery mutated an authoritative or reserved record");
    h::same_image(image, p::capture(model, *request.session, storage));
    p::check_old(model, before[member].checkpoint, before[member].image, storage);
    require(window.matches(keys[member]) && request.request_key == keys[member],
        "publication recovery replaced an original request identity");
  }
  require(submissions == 4 && preparations() == prepared_before_observation &&
      auxiliary_before_observation == std::array{model.draft_program_graph_calls,
          model.prefix_decision_graph_calls, model.frontier_sampling_graph_calls},
      "observation-only recovery prepared or submitted new model or sampling work");
  for (std::size_t member = 0; member != requests.size(); ++member) {
    before[member].checkpoint.reset(); window.retire_active(keys[member]);
    h::retire(requests[member].session); window.release(keys[member]);
  }
  q::check(model.execution.collect_retired());
  require(!model.snapshot_versions.live && !storage->live_images() && window.active().empty() &&
      model.page_inventory.free_ids.size() == plan.max_pages,
      "publication retirement retained an original version, page or host image");
  std::cout << "{\"status\":\"PASS\",\"case\":\"publication-retirement\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"synthetic_untrained\":true,\"requests\":2,\"target_submissions\":" << submissions
      << ",\"unpublished_refusals\":1,\"published_observer_refusals\":1,\"original_nonstd_exception_preserved\":true,"
         "\"actual_native_retirement\":true,\"exact_uncommitted_record_discard\":true,\"same_input_retry\":true,"
         "\"committed_peers_preserved\":true,\"observation_retry_submissions\":0,\"old_snapshots_unchanged\":true,"
         "\"native_fault_injection\":false,\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_publication_retirement_case
