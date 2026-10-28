#pragma once

// Actual mixed-width recursive requests. The one failing target distribution
// is supplied only after valid q rows have executed; acceptance is never forced.
namespace qwen_mtp_group_return_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace m = qwen_mtp_logits_case;
namespace s = qwen_mtp_batch_state_case;
namespace t = qwen_mtp_stop_case;
namespace e = qwen_logits_preflight_case;
using h::require;

struct QIdentity final {
  void *handle = nullptr;
  std::size_t offset = 0, bytes = 0;
  std::uint64_t generation = 0;
  friend bool operator==(const QIdentity &, const QIdentity &) = default;
};
inline QIdentity identity(const q::c::ReadLease &lease) {
  const auto view = q::take(lease.view());
  return {view.native_handle(), view.offset(), view.size(), lease.generation()};
}

inline void cold(q::Model &model) {
  const std::array three{std::size_t{3}}, one{std::size_t{1}};
  model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 3, q::GraphRole::kMtpPrefill);
  model.prepare_family(one, false, 1, q::GraphRole::kMtpSingle);
  model.plan_replay_uses(1, 1, 4, q::GraphRole::kMtpSingle);
  const std::array short_verify{std::size_t{2}};
  model.prepare_family(short_verify, false, 1, q::GraphRole::kVerify);
  model.plan_replay_uses(2, 1, 2, q::GraphRole::kVerify);
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(three, false, width, q::GraphRole::kVerify);
    model.plan_replay_uses(3, width, 4, q::GraphRole::kVerify);
  }
  model.prepare_recursive_round_programs();
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers() && geometry.mtp_layers == 1,
      "MTP group-return requires the cached GA/H64/V64/C16 MTP fixture");
  // Three real heads, retained old checkpoints, and a B2 candidate stay bounded.
  // Actual device cohort R is2; the five q rows fit its original2*3 pool bound.
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {.8F, 0, 1.F}, .seed = 123,
       .modifiers = {.bias = true, .allowed = true}}, 0, {.primary = kind});
  model.configure_draft_policy({.max_drafts = 2}); cold(model);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const auto valid = m::payload(model, 0), masked = m::payload(model, 0, true);
  const std::array<std::array<std::int32_t, 3>, 3> prompts{{{1, 2, 3}, {3, 2, 1}, {1, 2, 4}}};
  std::vector<q::GenerationRequest> requests; requests.reserve(3);
  i::RequestWindow window; window.prepare(3, 3);
  for (std::size_t member = 0; member != 3; ++member) {
    requests.emplace_back("group-return" + std::to_string(member), prompts[member], std::filesystem::path{}, 7);
    auto &request = requests.back(); t::prepare_request(request, 16); request.logits = valid.inputs;
    request.progress.initial_sampling = {313, 71 + member, 0};
    request.request_key = window.enqueue(member); window.admitted(request.request_key);
    std::array prefill{q::prepare_generation_call(model, request, 7, nullptr, requests, 3)};
    q::execute_generation_calls(model, prefill, false, nullptr, requests);
    require(request.ledger->committed_count == 1 && request.progress.generated.size() == 1 &&
        request.session->head->sampling == request.progress.initial_sampling.advanced(1),
        "mixed-width seed did not publish exactly its real sampled prefill");
  }
  std::array before{e::save(model, requests[0], storage), e::save(model, requests[1], storage),
      e::save(model, requests[2], storage)};
  const std::array old_payloads{s::image(model, *before[0].checkpoint), s::image(model, *before[1].checkpoint),
      s::image(model, *before[2].checkpoint)};
  const std::array keys{requests[0].request_key, requests[1].request_key, requests[2].request_key};
  std::array calls{
      q::prepare_generation_call(model, requests[0], 7, nullptr, requests, 3),
      q::prepare_generation_call(model, requests[1], 7, nullptr, requests, 2),
      q::prepare_generation_call(model, requests[2], 7, nullptr, requests, 3)};
  const auto &q_program = *model.draft_programs.at({1, true, 0});
  const auto q_baseline = q_program.probability_pool.stats().live_blocks;
  require(q_baseline == 0 && q_program.probability_pool.stats().max_blocks == 6,
      "mixed-width witness changed the actual active2/candidate3 q capacity");
  std::array<std::vector<std::int32_t>, 3> original_ids;
  std::array<std::vector<QIdentity>, 3> original_q;
  std::array<std::string, 3> names;
  for (std::size_t member = 0; member != 3; ++member) {
    auto &call = calls[member];
    require(call.role == q::GraphRole::kVerify && call.maximum == (member == 1 ? 1u : 2u),
        "real output credit did not select the intended K1/K2 proposal bound");
    call.verification_ids = model.propose_recursive(*requests[member].session, call.maximum,
        q::model_eos_tokens, &call.draft_log_probabilities, &requests[member].ledger->stop_policy(), valid.inputs);
    require(call.verification_ids.size() == (member == 1 ? 2u : 3u) &&
        call.draft_log_probabilities.size() + 1 == call.verification_ids.size(),
        "actual proposal/stop policy failed to produce the required distinct verifier widths");
    m::q_rows(model, call.draft_log_probabilities, valid);
    original_ids[member] = call.verification_ids; names[member] = call.name;
    for (const auto &row : call.draft_log_probabilities) original_q[member].push_back(identity(row));
  }
  require(q_program.probability_pool.stats().live_blocks == 5,
      "mixed-width witness did not hold all five real normalized q blocks");
  calls[2].logits = masked.inputs; // q remains the original valid processed distribution.
  const auto drafts_before = model.draft_program_graph_calls;
  std::vector<std::size_t> submitted_widths; submitted_widths.reserve(2);
  const auto diagnose = [&] {
    std::cerr << "group_return submitted_widths=[";
    for (std::size_t j = 0; j != submitted_widths.size(); ++j)
      std::cerr << (j ? "," : "") << submitted_widths[j];
    const auto q_stats = q_program.probability_pool.stats(), hidden_stats = model.hidden_pool.stats();
    std::cerr << "] draft_calls_current=" << model.draft_program_graph_calls << " draft_calls_before=" << drafts_before
        << " expected_successful_peer_repairs=1"
        << " versions=" << model.snapshot_versions.live << "/" << model.snapshot_versions.capacity
        << " q_blocks=" << q_stats.live_blocks << "/" << q_stats.max_blocks
        << " hidden_blocks=" << hidden_stats.live_blocks << "/" << hidden_stats.max_blocks << '\n';
    for (std::size_t layer = 0; layer != model.state_pools.size(); ++layer) if (geometry.is_gdn(layer)) {
      const auto history = model.state_pools[layer].history.stats();
      const auto recurrent = model.state_pools[layer].recurrent.stats();
      std::cerr << "group_return layer=" << layer << " history_blocks=" << history.live_blocks << "/" << history.max_blocks
          << " recurrent_blocks=" << recurrent.live_blocks << "/" << recurrent.max_blocks << '\n';
    }
  };
  bool failed = false;
  try {
    q::execute_generation_calls(model, calls, false, nullptr, requests, [&] {
      const auto &step = *model.active_step;
      require(step.submitted && step.execution.completion.valid() && step.phase->family->role == q::GraphRole::kVerify,
          "mixed-width check failed before actual target submission");
      submitted_widths.push_back(step.input_tokens);
      if (step.input_tokens == 2)
        require(step.members.size() == 1 && step.members[0].session == requests[1].session,
            "narrower real verifier did not execute the original middle request first");
      else require(step.input_tokens == 3 && step.members.size() == 2 &&
          step.members[0].session == requests[0].session && step.members[1].session == requests[2].session,
          "wide child cohort changed the original member mapping");
    });
  } catch (const i::EvaluationFailure &error) {
    const auto native = error.status().native_error();
    const auto retired_native = error.retirement().status.native_error();
    std::cerr << "group_return failure status=" << static_cast<unsigned>(error.status().code())
        << " native=" << static_cast<unsigned>(native.domain) << ":" << native.code
        << " message=" << error.status().message() << " participant=";
    if (error.participant()) std::cerr << *error.participant(); else std::cerr << "unset";
    std::cerr << " retirement_observed=" << error.retirement_observed() << " retired=" << error.retirement().retired
        << " retirement_status=" << static_cast<unsigned>(error.retirement().status.code())
        << " retirement_native=" << static_cast<unsigned>(retired_native.domain) << ":" << retired_native.code
        << " retirement_message=" << error.retirement().status.message() << '\n';
    diagnose();
    failed = error.status().IsInvalidArgument() && !error.status().has_native_error() &&
        error.participant() && *error.participant() == 2 && error.retirement_observed() &&
        error.retirement().retired && error.retirement().status.ok();
  } catch (const std::exception &error) {
    std::cerr << "group_return unexpected exception=" << error.what() << '\n'; diagnose(); throw;
  }
  // The narrow B verifier commits first and repairs its consumed draft prefix
  // with exactly one head=false execute_draft, whether it consumed one or two
  // target rows. That real repair increments the same counter as proposals.
  const auto drafts_after_failure = model.draft_program_graph_calls;
  require(failed && submitted_widths == std::vector<std::size_t>{2, 3} &&
      drafts_after_failure == drafts_before + 1,
      "real failing child lost original participant2/retirement or changed the single successful peer repair budget");
  t::idle(model);
  for (std::size_t member = 0; member != 3; ++member) {
    require(calls[member].request == &requests[member] && requests[member].request_key == keys[member] &&
        calls[member].verification_ids == original_ids[member] && calls[member].name == names[member],
        "exception failed to return the original selected Call metadata to its original index");
    if (member == 1) continue;
    require(calls[member].disposition == q::EvaluationDisposition::kNotPublished &&
        calls[member].draft_log_probabilities.size() == 2,
        "failed wide child lost its actual nonpublication or original q leases");
    for (std::size_t row = 0; row != 2; ++row)
      require(identity(calls[member].draft_log_probabilities[row]) == original_q[member][row],
          "exception returned a replacement q lease instead of the original native block/generation");
    m::q_rows(model, calls[member].draft_log_probabilities, valid);
    m::terminal_failure_unchanged(model, requests[member], before[member], storage);
  }
  require(calls[1].disposition == q::EvaluationDisposition::kPublished && calls[1].draft_log_probabilities.empty() &&
      requests[1].ledger->committed_count == before[1].commits + 1 &&
      requests[1].session->head.epoch() == before[1].epoch + 1 &&
      requests[1].session->head->sampling == before[1].checkpoint->sampling.advanced(3) &&
      requests[1].progress.generated.size() > before[1].generated.size() &&
      q_program.probability_pool.stats().live_blocks == 4,
      "later child failure rolled back the narrow committed peer or lost its actual disposition/q release");
  for (auto &call : calls) call.draft_log_probabilities.clear();
  require(q_program.probability_pool.stats().live_blocks == q_baseline,
      "returned failed calls did not release their own q after actual native retirement");
  // The same original request key parks/restores its unchanged real head, then
  // retries through the ordinary public preparation/proposal/finish contract.
  auto &retry_owner = requests[2]; auto image = model.export_session(*retry_owner.session, storage);
  h::retire(retry_owner.session); retry_owner.session = model.restore_session(image); image.reset();
  require(retry_owner.request_key == keys[2] && window.matches(keys[2]),
      "failed original request lost its key during actual Session retirement/HostImage restore");
  h::same_image(before[2].image, qwen_packed_prefill_case::capture(model, *retry_owner.session, storage));
  retry_owner.logits = valid.inputs; const auto retry_cursor = retry_owner.session->head->sampling;
  const auto retry_epoch = retry_owner.session->head.epoch();
  std::array retry{q::prepare_generation_call(model, retry_owner, 7, nullptr, requests, 3)};
  q::execute_generation_calls(model, retry, false, nullptr, requests);
  require(retry[0].disposition == q::EvaluationDisposition::kPublished && retry[0].draft_log_probabilities.empty() &&
      retry_owner.request_key == keys[2] && window.matches(keys[2]) &&
      retry_owner.ledger->committed_count == before[2].commits + 1 &&
      retry_owner.session->head.epoch() == retry_epoch + 1 && retry_owner.session->head->sampling == retry_cursor.advanced(5),
      "same-key retry failed to publish exactly its own repaired state and five-address RNG reservation");
  for (std::size_t member = 0; member != 3; ++member)
    require(s::image(model, *before[member].checkpoint) == old_payloads[member],
        "mixed-width grouping or retry changed an old complete checkpoint payload/boundary");
  calls = {}; retry = {}; before = {};
  for (auto &request : requests) h::retire(request.session);
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      q_program.probability_pool.stats().live_blocks == 0,
      "group-return consumer retained an original state/native/q owner");
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-group-return\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"actual_active_sequences\":2,\"logical_requests\":3,\"submitted_widths\":[2,3],"
         "\"actual_q_before_grouping\":5,\"actual_q_after_failure\":4,\"original_participant\":2,"
         "\"successful_peer_repairs\":1,\"no_proposal_regeneration\":true,\"draft_program_calls_before_grouping\":" << drafts_before
      << ",\"draft_program_calls_after_failure\":" << drafts_after_failure << ',' <<
         "\"original_calls_and_q_returned\":true,\"earlier_group_commit_preserved\":true,"
         "\"failed_heads_ledger_rng_unchanged\":true,\"actual_retirement\":true,\"same_key_host_restore_retry\":true,"
         "\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_mtp_group_return_case
