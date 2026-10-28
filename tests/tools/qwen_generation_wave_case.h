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
      !request.turn_input_consumed && request.history && !request.history->length,
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
        require(model.active_step->cancelled && !model.active_step->discard_stopped_members,
            "MTP stop changed the original whole-episode cancellation policy");
      }
    });
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed,
        "generation wave returned before original native retirement/transaction recovery");
    for (const auto &call : batch) {
      require(call.disposition == (cancel ? q::EvaluationDisposition::kCancelled : q::EvaluationDisposition::kPublished),
          "executed call lost its real cancellation/publication disposition");
      require(call.draft_log_probabilities.empty() && !call.logits.counts.lease.valid(),
          "retired generation call retained private q/counts leases");
    }
  }
  void publish(q::PreparedGenerationCall &call) {
    require(prepared == expected && submissions == 1,
        "wave prompt publication preceded complete preparation/native execution");
    if (cancel) {
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
      require(policy.cancelled == 2 && policy.publications == 2,
          "common wave did not return both original cancelled calls before prompt publication");
    }
    native_submissions += policy.submissions;
    require(prefix.size() == 0 && requests[0].session->stopping && !requests[1].session->stopping,
        "cancelled wave published a prefix or poisoned its unchanged peer");
    for (std::size_t member = 0; member != 2; ++member) unchanged(model, requests[member], saved[member]);
    if (!old_copy) {
      const auto peer_key = requests[1].request_key;
      Policy retry{model, requests, prefix, saved}; retry.cancel = false; retry.expected = 1;
      const std::array peer{peer_key}; i::execute_request_batches(workspace, peer, retry);
      native_submissions += retry.submissions;
      require(requests[1].request_key == peer_key && requests[1].ledger->committed_count == 1 &&
          requests[1].progress.generated.size() == 1 && requests[1].history->length == 3 &&
          requests[1].session->length() == 3 && requests[1].session->head.epoch() == saved[1].epoch + 1 &&
          prefix.size() == 1 && retry.publications == 1,
          "survivor failed same-key retry or lost its real committed prompt/prefix");
      unchanged(model, requests[0], saved[0]);
    }
    while (prefix.evict_one()) {}
    for (auto &request : requests) h::retire(request.session);
    require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
        "generation-wave cleanup retained an original state/native owner");
  }
  std::cout << "{\"status\":\"PASS\",\"case\":\"generation-wave\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"native_submissions\":" << native_submissions
      << ",\"old_copy_component_failure_reproduced\":true,\"cancelled_disposition_returned\":true,"
         "\"whole_wave_prepare_before_publish\":true,\"cancelled_heads_ledger_rng_unchanged\":true,"
         "\"survivor_same_key_retry\":true,\"prefix_unpublished_on_cancel\":true,"
         "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_generation_wave_case
