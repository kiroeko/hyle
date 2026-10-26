// Explicit cached-weight integration check; never downloads a model and is
// not a performance benchmark. The executable uses the product implementation.
#include "../../engine/models/qwen35/main.cc"
#include "../../engine/models/qwen35/native_gate.h"
#include "../../engine/models/qwen35/probe_geometry.h"

namespace {
namespace q = qwen35;
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
constexpr auto native_backend = hyle::inference::BackendKind::kApple;
#elif defined(HYLE_COMPUTE_CUDA)
constexpr auto native_backend = hyle::inference::BackendKind::kCuda;
#elif defined(HYLE_COMPUTE_VULKAN)
constexpr auto native_backend = hyle::inference::BackendKind::kVulkan;
#else
#error "This native gate probe requires an Apple, CUDA or Vulkan endpoint"
#endif
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void retire(std::shared_ptr<q::Session> &session) {
  if (!session) return;
  (void)session->stop(); const auto drained = session->drain();
  const auto retired = session->try_retire(); if (retired.retired) session.reset();
  q::check(drained.status); q::check(retired.status);
  require(retired.retired, "original session failed to retire");
}
q::MtpRoundResult round(q::Model &model, const std::shared_ptr<q::Session> &session,
    q::GenerationLedger &ledger, const q::GenerationCommitRecord *expected = nullptr) {
  const auto pending = session->head->next_id;
  const auto role = session->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
  q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true, 2, 0);
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role); model.submit_evaluation(*session);
  const q::EvaluationIO output_io{.name = "random_round", .trace_states = false};
  const auto output = q::evaluation_output(model, session, output_io, nullptr, nullptr, nullptr, nullptr, &ledger, expected);
  return q::finish_mtp_round(model, *session, pending, 2, q::model_eos_tokens, &output, output_io);
}
void run(const std::filesystem::path &input) {
  const auto source = q::read_ids(input / "prompt.ids");
  require(source.size() >= 15, "integration check requires fifteen existing prompt tokens");
  const auto prompt = std::span(source).first(15);
  const auto geometry = q::probe::read_two_b_geometry(input);
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, cache_rows, 64, 3, 15,
      q::recurrent_bytes(q::state_schema_for(geometry, 64), geometry), 2, 2, true, true);
  q::Model model(input / "weights.tsv", geometry, 64, plan, {}, true, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.5F, 40, .95F}, .seed = 123}, 0, {.primary=native_backend});
  const std::array prefill{std::size_t{15}}, pair{std::size_t{2}};
  model.prepare_family(prefill, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(15, 1, 12, q::GraphRole::kMtpPrefill);
  for (auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
    model.prepare_family(pair, false, 1, role); model.plan_replay_uses(2, 1, 12, role);
  }
  model.prepare_family(pair, false, 2, q::GraphRole::kMtpRound);
  model.plan_replay_uses(2, 2, 4, q::GraphRole::kMtpRound);
  auto session = model.create_session(prefill, {}, q::GraphRole::kMtpPrefill);
  q::GenerationLedger ledger(2);
  q::prepare_generation_record(ledger, *session, q::GraphRole::kMtpPrefill, prompt, false, 0, 0);
  model.prepare_evaluation(*session, prompt, nullptr, q::GraphRole::kMtpPrefill); model.submit_evaluation(*session);
  (void)q::finish_evaluation(model, *session, "prefill", {}, false, nullptr, nullptr, nullptr, nullptr, {}, &ledger);
  auto base = session->head;
  require(base->sampling == q::cache::SamplingCursor{123, 0, 1}, "prefill did not publish exactly one draw");
  const auto result = round(model, session, ledger);
  const auto final = q::generation_boundary(*session->head);
  require(final.sampling.draw == 4 && ledger.committed_count == 2,
      "K1 round did not atomically publish its three addresses and ledger");
  retire(session);
  q::GenerationRestoreStats stats;
  std::cerr << "random_probe_stage=strict_replay\n";
  session = q::restore_mtp_session(model, ledger, {}, false, &stats);
  require(q::generation_boundary(*session->head) == final && stats.graph_calls == 2,
      "strict real Graph replay changed a sampled decision or RNG boundary");
  retire(session);
  // A real submitted Graph must fail before its head swap, rather than
  // accepting an inconsistent expected decision and advancing its cursor.
  session = model.create_session({}, base, q::GraphRole::kMtpPrefill);
  q::GenerationLedger rejected(1);
  auto damaged = ledger.records[1]; damaged.draft = static_cast<std::int32_t>((damaged.draft + 1) % geometry.vocabulary);
  std::cerr << "random_probe_stage=rejected_commit\n";
  bool failed = false;
  try { (void)round(model, session, rejected, &damaged); }
  catch (const std::runtime_error &error) { failed = true; std::cerr << "random_probe_expected_failure=" << error.what() << '\n'; }
  require(failed && session->head == base && session->head->sampling.draw == 1 &&
      !rejected.committed_count && !model.active_step && !model.carrier_borrowed,
      "failed sampled episode published a partial head, cursor or ledger");
  std::cerr << "random_probe_stage=retry\n";
  const auto retry = round(model, session, rejected, &ledger.records[1]);
  require(retry.emitted == result.emitted && q::generation_boundary(*session->head) == final,
      "same-slot retry did not reproduce the original stochastic decision");
  retire(session);
  // Stall only the backend's original native endpoint. This instrument does
  // not execute model work or manufacture Completion/retirement evidence.
  session = model.create_session({}, base, q::GraphRole::kMtpPrefill);
  std::cerr << "random_probe_stage=cancel_prepare\n";
  const auto pending = base->next_id;
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, q::GraphRole::kMtpRound);
  auto *episode = model.active_step.get(); const auto candidates = model.snapshot_versions.live;
  {
    q::probe::NativeGate gate(model.execution);
    std::cerr << "random_probe_stage=cancel_submit\n";
    q::check(gate.arm()); q::check(gate.wait_until_entered()); model.submit_evaluation(*session);
    q::check(gate.wait_for_model_submission()); q::check(session->stop());
    const auto pending = session->try_retire();
    if (!pending.status.IsCancelled() || pending.status.has_native_error()) q::check(pending.status);
    require(!pending.retired && model.active_step.get() == episode &&
        session->head == base && session->head->sampling.draw == 1 &&
        model.snapshot_versions.live == candidates, "blocked cancellation published or released an accepted native owner");
    gate.release(); q::check(gate.wait());
    require(!gate.timed_out(), "native gate watchdog released the queue");
  }
  const auto drained = session->drain();
  const auto retired = session->try_retire(); if (retired.retired) session.reset();
  if (!drained.status.IsCancelled() || drained.status.has_native_error()) q::check(drained.status);
  if (!retired.status.IsCancelled() || retired.status.has_native_error()) q::check(retired.status);
  require(retired.retired, "cancelled Graph did not actually retire");
  require(base->sampling.draw == 1 && !model.active_step && !model.carrier_borrowed,
      "cancelled sampled work advanced the old cursor");
  // Same state, two independent request streams, one real R2 target/draft
  // Graph. The fresh first tokens are sampled from the cached hidden row.
  std::array sessions{
      model.create_session({}, base, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 0, 0}, true),
      model.create_session({}, base, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 1, 0}, true)};
  require(sessions[0]->head->state_identity() == base.get() && sessions[1]->head->state_identity() == base.get() &&
      model.snapshot_versions.live == 1, "sampler wrappers duplicated or lost the bounded model-state version");
  const std::array ids{sessions[0]->head->next_id, sessions[1]->head->next_id};
  const std::array requests{q::EvaluationRequest{sessions[0], std::span(&ids[0], 1)},
      q::EvaluationRequest{sessions[1], std::span(&ids[1], 1)}};
  const std::array outputs_io{q::EvaluationIO{.name = "row0", .trace_states = false},
      q::EvaluationIO{.name = "row1", .trace_states = false}};
  const std::array outputs{q::evaluation_output(model, sessions[0], outputs_io[0]),
      q::evaluation_output(model, sessions[1], outputs_io[1])};
  const std::array selections{q::MtpGenerationSelection{sessions[0], ids[0], 2},
      q::MtpGenerationSelection{sessions[1], ids[1], 2}};
  std::array<q::MtpRoundResult, 2> results;
  model.prepare_evaluations(requests, nullptr, q::GraphRole::kMtpRound); model.submit_evaluations();
  model.finish_mtp_rounds(selections, q::model_eos_tokens, outputs, results, [&](std::size_t j, const q::CompletedEvaluationView &view) {
        q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
      });
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    require(sessions[i]->head->sampling == q::cache::SamplingCursor{123, i, 4},
        "real R2 mixed independent request cursors or consumed a branch-dependent address count");
    retire(sessions[i]);
  }
  base.reset(); require(model.snapshot_versions.live == 0, "integration check retained a model-state version");
  std::cout << "{\"status\":\"PASS_RANDOM_MTP_TRANSACTION\",\"backend\":\"" << model.execution.name()
      << "\",\"strict_replay\":true,\"failed_commit_retry\":true,\"blocked_native_cancel\":true,"
         "\"real_R2\":true,\"cached_frontier_streams\":true,\"performance_claim\":false}\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 3 || std::string_view(argv[1]) != "--inputs")
      throw std::runtime_error("usage: qwen_random_mtp_probe --inputs EXISTING_PREPARED_MTP_DIR");
    run(argv[2]); return 0;
  } catch (const std::exception &error) { std::cerr << "random MTP integration: " << error.what() << '\n'; return 1; }
}
