#pragma once

#include "../../engine/inference/request_batch_execution.h"

// One actual cached GA Model. The old three-line bucket copy is an explicit
// component regression witness, not another Model/request implementation.
namespace qwen_generation_wave_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace s = qwen_mtp_batch_state_case;
namespace t = qwen_mtp_stop_case;
namespace p = qwen_packed_prefill_case;
using h::require;

struct Saved final {
  const q::cache::State *head = nullptr;
  std::uint64_t epoch = 0;
  s::StateImage image{};
  i::RequestKey key{};
};

inline void unchanged(q::Model &model, const q::GenerationRequest &request, const Saved &before) {
  require(request.session && request.session->head.get() == before.head &&
      request.session->head.epoch() == before.epoch && request.request_key == before.key &&
      s::image(model, *request.session->head) == before.image &&
      request.ledger && !request.ledger->committed_count && !request.progress.observed_commits() &&
      request.progress.generated.empty() && !request.progress.prefilled &&
      !request.progress.input_consumed() && request.history && !request.history->length,
      "cancelled generation wave changed head, state bytes, pending/RNG, key, committed ledger or prompt history");
  const auto &next = request.ledger->next();
  require(!next.committed && !next.accepted && !next.emitted_count && !next.quota &&
      !next.input_owner && next.pending_input == -1 && next.prompt_ids.empty() &&
      next.accepted_inputs.empty() && next.verification_inputs.empty(),
      "cancelled generation wave retained uncommitted publication authority");
}

struct Policy final {
  q::Model &model;
  std::span<q::GenerationRequest> requests;
  q::cache::PrefixCache &prefix;
  std::array<Saved, 2> &saved;
  bool cancel = true;
  std::size_t expected = 2, prepared = 0, submissions = 0, publications = 0, cancelled = 0;
  q::PreparedGenerationCall prepare(i::RequestKey key) {
    require(!publications && prefix.size() == 0 && key.slot < requests.size(),
        "wave published a prefix before preparing all original request keys");
    auto &request = requests[key.slot];
    require(request.request_key == key, "wave preparation replaced an original RequestKey");
    auto call = q::prepare_generation_call(model, request, request.progress.output_limit(), &prefix, requests);
    require(call.initial_prefix && !call.full_hit && !call.generation &&
        call.role == q::GraphRole::kMtpPrefill && call.ids().size() == 3,
        "wave regression must prepare the actual initial three-row MTP prompt");
    saved[key.slot] = {request.session->head.get(), request.session->head.epoch(),
        s::image(model, *request.session->head), key};
    ++prepared; return call;
  }
  std::size_t limit(const q::PreparedGenerationCall &) const noexcept { return 2; }
  bool compatible(const q::PreparedGenerationCall &a, const q::PreparedGenerationCall &b) const noexcept {
    return a.role == b.role && a.completion == b.completion && a.ids().size() == b.ids().size() &&
        a.request->sampling_index == b.request->sampling_index;
  }
  bool fit(std::span<q::PreparedGenerationCall> batch) {
    return q::make_generation_batch_room(model, &prefix, batch, requests);
  }
  void execute(std::span<q::PreparedGenerationCall> batch, bool entire_wave) {
    require(prepared == expected && publications == 0 && prefix.size() == 0 && entire_wave,
        "wave executed or published before its whole original prepare phase");
    q::execute_generation_calls(model, batch, false, &prefix, requests, [&] {
      require(model.active_step && model.active_step->submitted &&
          model.active_step->execution.completion.valid() && model.active_step->members.size() == expected,
          "wave cancellation did not observe the real target Graph submission");
      ++submissions;
      if (cancel) {
        require(expected == 2 && batch.size() == 2, "cancel witness lost its original B2 cohort");
        (void)requests[0].session->stop();
        require(!model.active_step->cancelled && model.active_step->discard_stopped_members &&
            model.active_step->execution.completion.valid(),
            "MTP prefill withdrawal aborted its surviving member's original submission");
      }
    });
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed,
        "generation wave returned before original native retirement/transaction recovery");
    for (const auto &call : batch) {
      require(call.disposition == (cancel && call.request == &requests[0]
          ? q::EvaluationDisposition::kCancelled : q::EvaluationDisposition::kPublished),
          "executed call lost its real cancellation/publication disposition");
      require(call.draft_log_probabilities.empty() && !call.logits.counts.lease.valid(),
          "retired generation call retained private q/counts leases");
    }
  }
  void publish(q::PreparedGenerationCall &call) {
    require(prepared == expected && submissions == 1,
        "wave prompt publication preceded complete preparation/native execution");
    if (cancel && call.request == &requests[0]) {
      require(call.disposition == q::EvaluationDisposition::kCancelled,
          "move-back failed to return the actual Cancelled disposition");
      ++cancelled;
    }
    q::publish_generation_prompt(model, call, &prefix); ++publications;
  }
};

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers() && geometry.mtp_layers == 1,
      "generation-wave requires the cached untrained GA/H64/V64/C16 MTP fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = false, .policy = {}, .seed = 123}, 0, {.primary = kind});
  model.configure_draft_policy({.max_drafts = 2});
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array shapes{std::size_t{3}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(shapes, false, width, q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(3, width, 8, q::GraphRole::kMtpPrefill);
  }
  i::RequestBatchWorkspace<q::PreparedGenerationCall, 2> workspace;
  workspace.prepare(2, 2);
  bool old_copy_reproduced = false;
  std::size_t native_submissions = 0;
  for (const bool old_copy : {true, false}) {
    const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 3}, {3, 2, 1}}};
    std::vector<q::GenerationRequest> requests; requests.reserve(2);
    i::RequestWindow window; window.prepare(2, 2);
    for (std::size_t member = 0; member != 2; ++member) {
      requests.emplace_back("generation-wave" + std::to_string(member), prompts[member], std::filesystem::path{}, 2);
      auto &request = requests.back(); t::prepare_request(request, 16);
      request.request_key = window.enqueue(member); window.admitted(request.request_key);
    }
    const std::array keys{requests[0].request_key, requests[1].request_key};
    q::cache::PrefixCache prefix(2); std::array<Saved, 2> saved{};
    Policy policy{model, requests, prefix, saved};
    if (old_copy) {
      std::array calls{policy.prepare(keys[0]), policy.prepare(keys[1])};
      // Exact old natural-wave defect: execution updated only these copies.
      auto bucket = calls;
      policy.execute(bucket, true);
      require(calls[0].disposition == q::EvaluationDisposition::kNotPublished &&
          calls[1].disposition == q::EvaluationDisposition::kNotPublished,
          "old component-copy witness unexpectedly updated original call facts");
      try {
        for (auto &call : calls) q::publish_generation_prompt(model, call, &prefix);
      } catch (const std::runtime_error &error) {
        old_copy_reproduced = std::string_view(error.what()) ==
            "committed prefill chunks do not cover the complete prompt";
        if (!old_copy_reproduced) throw;
      }
      require(old_copy_reproduced, "old copied-call witness did not reproduce the incorrect prompt publication");
    } else {
      i::execute_request_batches(workspace, keys, policy);
      require(policy.cancelled == 1 && policy.publications == 2,
          "common wave did not return the original cancelled A and published B before prompt publication");
    }
    native_submissions += policy.submissions;
    require(prefix.size() == (old_copy ? 0u : 1u) && requests[0].session->stopping && !requests[1].session->stopping,
        "withdrawn A published a prefix or poisoned B's original publication");
    unchanged(model, requests[0], saved[0]);
    require(requests[1].request_key == saved[1].key && requests[1].ledger->committed_count == 1 &&
        requests[1].ledger->records[0].committed && requests[1].ledger->records[0].requests == 2 &&
        requests[1].ledger->records[0].member_index == 1 && requests[1].progress.observed_commits() == 1 &&
        requests[1].progress.generated.size() == 1 && requests[1].progress.prefilled &&
        requests[1].session->length() == 3 && requests[1].session->head.epoch() == saved[1].epoch + 1,
        "B did not publish its original B2 result, request key, ledger, output and epoch");
    if (!old_copy) {
      require(requests[1].history->length == 3 && prefix.size() == 1,
          "original surviving B2 result did not publish its complete prompt and prefix");
      // The third Graph is an independent R1 oracle from a fresh request. B
      // itself is never resubmitted to manufacture successful cancellation.
      std::vector<q::GenerationRequest> controls; controls.reserve(1);
      controls.emplace_back("generation-wave-control", prompts[1], std::filesystem::path{}, 2);
      t::prepare_request(controls[0], 16);
      i::RequestWindow control_window; control_window.prepare(1, 1);
      controls[0].request_key = control_window.enqueue(0); control_window.admitted(controls[0].request_key);
      q::cache::PrefixCache control_prefix(1); std::array<Saved, 2> control_saved{};
      Policy control{model, controls, control_prefix, control_saved}; control.cancel = false; control.expected = 1;
      const std::array control_key{controls[0].request_key}; i::execute_request_batches(workspace, control_key, control);
      native_submissions += control.submissions;
      p::StepReport checked;
      p::compare(model, p::capture(model, *controls[0].session, storage),
          p::capture(model, *requests[1].session, storage), checked);
      require(checked.numerically_equivalent && controls[0].progress.generated == requests[1].progress.generated &&
          controls[0].session->head->sampling == requests[1].session->head->sampling &&
          requests[1].ledger->committed_count == 1 && requests[1].session->head.epoch() == saved[1].epoch + 1 &&
          control.submissions == 1 && control.publications == 1 && policy.submissions == 1,
          "surviving original B2 complete state/output/RNG differs from the independent R1 oracle");
      while (control_prefix.evict_one()) {}
      h::retire(controls[0].session);
      unchanged(model, requests[0], saved[0]);
    }
    while (prefix.evict_one()) {}
    for (auto &request : requests) h::retire(request.session);
    require(!model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed &&
        !storage->live_images(),
        "generation-wave cleanup retained an original state/native owner");
  }
  require(native_submissions == 3, "generation-wave changed its finite three original target submissions");
  std::cout << "{\"status\":\"PASS\",\"case\":\"generation-wave\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"native_submissions\":" << native_submissions
      << ",\"old_copy_component_failure_reproduced\":true,\"cancelled_disposition_returned\":true,"
         "\"whole_wave_prepare_before_publish\":true,\"cancelled_heads_ledger_rng_unchanged\":true,"
         "\"survivor_original_submission_published\":true,\"survivor_complete_R1_state_equal\":true,"
         "\"cancelled_prefix_unpublished\":true,\"survivor_prefix_published\":true,"
         "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_generation_wave_case
