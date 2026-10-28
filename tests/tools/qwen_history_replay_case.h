#pragma once

// Included with the existing cached GA consumers after the actual Model.
// Strict replay repeats each recorded role/width/policy and consumed prefix;
// this is not a cross-width distribution or trained-model quality oracle.
namespace qwen_history_replay_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace t = qwen_mtp_stop_case;
namespace m = qwen_mtp_logits_case;
namespace z = qwen_history_logits_case;
using h::require;

inline void same(q::Model &model, const h::ImageBytes &expected, const h::ImageBytes &actual,
    std::vector<p::StepReport> &checks) {
  require(expected.identity == actual.identity, "history replay changed its original model/policy identity");
  t::equal(model, expected, actual, checks);
  if (!checks.back().payloads_equal) {
    std::cerr << "history_replay_state_check\n"; p::write_step(checks.back()); std::cout << '\n';
  }
  require(checks.back().payloads_equal, "same-role history replay changed complete initialized state bytes");
}

inline q::LogitsInputs conditional(q::Model &model, q::GenerationLedger &ledger,
    std::span<const std::int32_t> ids, std::size_t index) {
  const auto committed = z::consumed(ledger);
  const auto expected = z::counts(model, committed, ids);
  auto result = model.history_logits_inputs(ledger, ids, {}, index);
  z::same_counts(model, result.counts, expected);
  return result;
}

inline void teacher(q::Model &model, const std::shared_ptr<q::Session> &session,
    q::GenerationLedger &ledger, std::span<const std::int32_t> ids, q::GraphRole role,
    q::TextCompletion completion, std::size_t begin, std::shared_ptr<const void> owner) {
  const auto ordinal = ledger.committed_count;
  const auto epoch = session->head.epoch();
  const auto cursor = session->head->sampling;
  const auto index = session->sampling_configuration().index;
  const bool emits = completion == q::TextCompletion::kEmitNext;
  q::prepare_generation_record(ledger, *session, role, ids, false, 0, 0,
      std::move(owner), 0, completion, begin, emits);
  auto inputs = emits ? qwen_history_replay_case::conditional(model, ledger, ids, index) : q::LogitsInputs{};
  model.prepare_evaluation(*session, ids, nullptr, role, completion, &ledger.stop_policy(), inputs);
  if (!emits) qwen_mtp_chunked_prefill_case::state_only_ports(*model.active_step->phase);
  model.submit_evaluation(*session);
  const auto output = q::evaluation_output(model, session, {.name = "history-replay-teacher", .trace_states = false},
      nullptr, nullptr, nullptr, nullptr, &ledger);
  const std::array outputs{output}; std::array<std::int32_t, 1> next{-1};
  model.finish_evaluations(outputs, next, [](auto, const auto &) {});
  const bool random = session->sampling_configuration().configuration.enabled;
  const auto expected = random && emits ? cursor.advanced(1) : cursor;
  require(ledger.committed_count == ordinal + 1 && session->head.epoch() == epoch + 1 &&
      session->head->sampling == expected && ledger.records[ordinal].after.sampling == expected &&
      ledger.records[ordinal].committed_ids().size() == ids.size() &&
      (emits ? next[0] >= 0 : next[0] == -1 && !ledger.records[ordinal].emitted_count),
      "history replay teacher changed its actual consumed length, epoch, completion or RNG");
  t::idle(model);
}

inline q::MtpRoundResult commit_round(q::Model &model, const std::shared_ptr<q::Session> &session,
    q::GenerationLedger &ledger, bool recursive) {
  const auto ordinal = ledger.committed_count;
  const auto epoch = session->head.epoch();
  const auto before = z::snapshot(model, *session->head);
  const auto cursor = session->head->sampling;
  const auto pending = session->head->next_id;
  const auto &configuration = session->sampling_configuration();
  const auto role = recursive ? q::GraphRole::kVerify : session->head->draft->lag
      ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
  q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true,
      3, 0, {}, recursive ? 2 : 0);
  auto inputs = qwen_history_replay_case::conditional(model, ledger, std::span(&pending, 1), configuration.index);
  std::vector<q::c::ReadLease> probabilities;
  auto ids = recursive ? model.propose_recursive(*session, 2, q::model_eos_tokens,
      &probabilities, &ledger.stop_policy(), inputs) : std::vector<std::int32_t>{pending};
  require(z::snapshot(model, *session->head) == before && session->head.epoch() == epoch &&
      (!recursive || ids.size() == 3) &&
      probabilities.size() == (recursive && configuration.configuration.enabled ? 2u : 0u),
      "history replay proposals changed the original head or lost a conditional q");
  model.prepare_evaluation(*session, ids, nullptr, role, q::TextCompletion::kEmitNext,
      &ledger.stop_policy(), inputs);
  require(model.active_step->phase->family->sampling == &configuration,
      "history replay selected another cold policy");
  model.submit_evaluation(*session);
  const auto output = q::evaluation_output(model, session, {.name = "history-replay-round", .trace_states = false},
      nullptr, nullptr, nullptr, nullptr, &ledger);
  const std::array outputs{output};
  const std::array selections{q::MtpGenerationSelection{session, pending, 3, probabilities, recursive ? 2u : 1u}};
  std::array<q::MtpRoundResult, 1> results;
  if (recursive) model.finish_recursive_mtp(selections, q::model_eos_tokens, outputs, results, [](auto, const auto &) {});
  else model.finish_mtp_rounds(selections, q::model_eos_tokens, outputs, results, [](auto, const auto &) {});
  const auto expected = configuration.configuration.enabled ? cursor.advanced(recursive ? 5 : 3) : cursor;
  require(ledger.committed_count == ordinal + 1 && session->head.epoch() == epoch + 1 &&
      session->head->sampling == expected && session->length() == before.length + results[0].processed_target &&
      ledger.records[ordinal].after == q::generation_boundary(*session->head) &&
      ledger.records[ordinal].verified_rows == (recursive ? 3u : 2u),
      "history replay round changed its actual width, accepted prefix, epoch or RNG reservation");
  t::idle(model); return std::move(results[0]);
}

inline void journey(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage,
    std::size_t index, std::vector<p::StepReport> &checks) {
  auto prompt = std::make_shared<const std::vector<std::int32_t>>(std::initializer_list<std::int32_t>{1, 2, 1});
  const auto ids = std::span<const std::int32_t>(*prompt);
  const auto initial = index ? q::cache::SamplingCursor{1301, 97, 0} : q::cache::SamplingCursor{};
  const std::array one{std::size_t{1}};
  auto session = model.create_session(one, {}, q::GraphRole::kMtpSingle, initial,
      false, q::TextCompletion::kStateOnly, index);
  q::GenerationLedger ledger(8, 3); ledger.bind_stop_policy({}, model.vocabulary);
  qwen_history_replay_case::teacher(model, session, ledger, ids.first(1), q::GraphRole::kMtpSingle, q::TextCompletion::kStateOnly, 0, prompt);
  require(session->head->next_id == -1 && session->head->sampling == initial,
      "history-only StateOnly consumed a sampling address or invented pending");
  qwen_history_replay_case::teacher(model, session, ledger, ids.subspan(1), q::GraphRole::kMtpPrefill, q::TextCompletion::kEmitNext, 1, prompt);
  auto checkpoint = session->checkpoint(); const auto checkpoint_bytes = z::snapshot(model, *checkpoint);
  (void)qwen_history_replay_case::commit_round(model, session, ledger, false);
  (void)qwen_history_replay_case::commit_round(model, session, ledger, true);
  require(ledger.committed_count == 4 && ledger.records[2].draft_limit == 1 && ledger.records[3].draft_limit == 2,
      "history-only replay ledger did not contain actual K1 and K2 decisions");
  const auto original = p::capture(model, *session, storage);

  // Park the real complete state while retaining the same original ledger.
  // No token history is inferred from a HostImage's scalar length.
  auto parked = model.export_session(*session, storage);
  h::retire(session); session = model.restore_session(parked); parked.reset();
  qwen_history_replay_case::same(model, original, p::capture(model, *session, storage), checks);
  q::GenerationRestoreStats stats;
  auto replay = q::restore_mtp_session(model, ledger, {}, false, &stats);
  require(&replay->sampling_configuration() == &model.sampling_configuration(index) &&
      stats.graph_calls == ledger.committed_count && stats.decision_changes.empty() &&
      q::generation_boundary(*replay->head) == ledger.records[ledger.committed_count - 1].after,
      "history-only strict replay changed the cold configuration, ledger decisions or RNG boundary");
  // Identical role, width and prefix use the original complete byte oracle.
  // No sampling-wrapper correction or GDN tolerance is added for this replay.
  qwen_history_replay_case::same(model, original, p::capture(model, *replay, storage), checks);

  q::GenerationLedger continued(8, 3); continued.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  continued.bind_sampling(model.sampling_configuration(index), model.resources_);
  for (std::size_t at = 0; at != ledger.committed_count; ++at) continued.records[at] = ledger.records[at];
  continued.committed_count = ledger.committed_count;
  for (const bool recursive : {false, true}) {
    const auto expected = qwen_history_replay_case::commit_round(model, session, ledger, recursive);
    const auto actual = qwen_history_replay_case::commit_round(model, replay, continued, recursive);
    h::same_round(expected, actual);
    require(ledger.committed_count == continued.committed_count &&
        ledger.records[ledger.committed_count - 1].same_result(continued.records[continued.committed_count - 1]),
        "history-only replay continuation changed its original ledger decision");
    qwen_history_replay_case::same(model, p::capture(model, *session, storage), p::capture(model, *replay, storage), checks);
  }
  require(z::snapshot(model, *checkpoint) == checkpoint_bytes && !model.resources_->history_counts_pool.stats().live_blocks,
      "history-only replay changed its old checkpoint or retained completed C0");
  checkpoint.reset(); h::retire(session); h::retire(replay);
  require(!model.snapshot_versions.live && !storage->live_images(), "history-only replay retained native state or HostImage owners");
  t::idle(model);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "history replay requires the cached GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 8, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  const q::sampling::LogitsPolicy modifiers{.history = {1.75F, .5F, .125F}};
  const std::array additional{q::SamplingConfiguration{true, {.8F, 8, .9F}, modifiers}};
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = false, .policy = {}, .seed = 123, .modifiers = modifiers}, 0, {.primary = kind}, additional);
  model.configure_draft_policy({.max_drafts = 2}); m::cold(model);
  const std::array one{std::size_t{1}};
  model.prepare_family(one, false, 1, q::GraphRole::kMtpSingle, false, q::TextCompletion::kStateOnly, 0);
  model.plan_replay_uses(1, 1, 8, q::GraphRole::kMtpSingle, q::TextCompletion::kStateOnly, 0);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  std::vector<p::StepReport> checks;
  for (std::size_t index = 0; index != 2; ++index) qwen_history_replay_case::journey(model, storage, index, checks);
  // A fresh key's full hit uses its own history-only policy/frontier and the
  // actual consumed cache key, independently checked against explicit C0.
  z::prefix(model, storage, m::Payload{}, checks);
  require(!model.snapshot_versions.live && !storage->live_images() &&
      model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "history-only strict replay consumer retained an original state/page owner");
  t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"history-replay\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"configurations\":2,\"history_only\":true,\"strict_K1_K2_greedy_random\":true,"
      "\"StateOnly_history\":true,\"HostImage_same_ledger\":true,\"post_replay_K1_K2\":true,"
      "\"full_hit_explicit_C0\":true,\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,"
      "\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at != checks.size(); ++at) { if (at) std::cout << ','; p::write_step(checks[at]); }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_history_replay_case
