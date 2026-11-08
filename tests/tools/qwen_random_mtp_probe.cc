// Explicit cached-weight integration check; never downloads a model and is
// not a performance benchmark. The executable uses the product implementation.
#include "../../engine/models/qwen35/main.cc"
#include "../../engine/models/qwen35/native_gate.h"
#include "../../engine/models/qwen35/probe_geometry.h"
#include "../compute_models/sampling_fixture.h"
#include <cmath>
#include <limits>
#include <numeric>

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
bool expected_replay_mismatch(const std::runtime_error &error) {
  return std::string_view(error.what()).find("MTP replay natural decision or state boundary diverged") != std::string_view::npos;
}

// Independent scalar Philox/Gumbel reference for the CUDA categorical route.
// The Philox helper is separately checked against official known-answer vectors.
std::int32_t reference_draw(std::span<const double> probabilities, std::array<std::uint32_t, 4> identity) {
  double best = -std::numeric_limits<double>::infinity();
  std::int32_t selected = -1;
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    if (!(probabilities[i] > 0)) continue;
    const auto words = sampling_test::philox({static_cast<std::uint32_t>(i / 2), 0, identity[2], identity[3]},
                                              {identity[0], identity[1]});
    auto hi = words[(i % 2) * 2], lo = words[(i % 2) * 2 + 1];
    const bool upper = (hi & 0x80000000u) != 0;
    if (upper) { hi = ~hi; lo = ~lo; }
    const double small = std::ldexp(static_cast<double>(hi), -32) +
        std::ldexp(static_cast<double>(lo) + .5, -64);
    const auto gumbel = -std::log(upper ? -std::log1p(-small) : -std::log(small));
    const auto score = std::log(probabilities[i]) + gumbel;
    if (score > best) { best = score; selected = static_cast<std::int32_t>(i); }
  }
  require(selected >= 0, "reference categorical distribution has no mass");
  return selected;
}

std::vector<double> reference_target(std::span<const float> logits) {
  // Match this probe's fixed temperature=1.5, top-k=40, top-p=.95 policy.
  std::vector<std::size_t> order(logits.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::partial_sort(order.begin(), order.begin() + 40, order.end(), [&](auto a, auto b) {
    return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
  });
  std::vector<double> result(logits.size(), 0.);
  const double maximum = logits[order[0]] / 1.5;
  double total = 0.;
  for (std::size_t i = 0; i < 40; ++i) {
    require(std::isfinite(logits[order[i]]), "nonfinite target logits");
    result[order[i]] = std::exp(logits[order[i]] / 1.5 - maximum); total += result[order[i]];
  }
  double kept = 0.;
  for (std::size_t i = 0; i < 40; ++i) {
    const auto id = order[i];
    if (i && kept / total >= .95) result[id] = 0.; else kept += result[id];
  }
  require(kept > 0., "reference target distribution has no mass");
  for (auto &value : result) value /= kept;
  return result;
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
  const auto output = q::evaluation_output(model, session, output_io);
  return q::finish_mtp_round(model, *session, pending, 2, q::model_eos_tokens, &output, output_io, {&ledger,expected});
}
void run(const std::filesystem::path &input) {
  const auto source = q::read_ids(input / "prompt.ids");
  require(source.size() >= 15, "integration check requires fifteen existing prompt tokens");
  const auto prompt = std::span(source).first(15);
  const auto geometry = q::probe::read_two_b_geometry(input);
  auto weights = q::prepare_weight_source(input / "weights.tsv", true);
  const auto precision = q::PrecisionProfile::from_roles(weights.roles);
  const auto plan = q::cache::plan(geometry, weights.cache_row_bytes, 64, 3, 15,
      q::recurrent_bytes(q::state_schema_for(geometry, 64, precision), geometry), 2, 2, true, true, 1, 0, precision);
  q::Model model(std::move(weights), geometry, 64, plan, {}, true, q::ReplayPolicy::kCommands, SIZE_MAX,
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
  auto base = session->checkpoint();
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
  catch (const std::runtime_error &error) {
    if (!expected_replay_mismatch(error)) throw;
    failed = true; std::cerr << "random_probe_expected_failure=" << error.what() << '\n';
  }
  require(failed && session->head.get() == base.get() && session->head->sampling.draw == 1 &&
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
    const auto waiting_retirement = session->try_retire();
    if (!waiting_retirement.status.IsCancelled() || waiting_retirement.status.has_native_error()) q::check(waiting_retirement.status);
    require(!waiting_retirement.retired && model.active_step.get() == episode &&
        session->head.get() == base.get() && session->head->sampling.draw == 1 &&
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

q::MtpRoundResult recursive_round(q::Model &model, const std::shared_ptr<q::Session> &session,
    q::GenerationLedger &ledger, const q::GenerationCommitRecord *expected = nullptr) {
  const auto pending = session->head->next_id;
  const auto *base = session->head.get();
  const auto before = q::generation_boundary(*base);
  std::vector<q::c::ReadLease> probabilities;
  const auto ids = model.propose_recursive(*session, 3, q::model_eos_tokens, &probabilities);
  require(ids.size() == 4 && probabilities.size() == 3, "K3 did not retain all three actual proposal distributions");
  require(session->head.get() == base && q::generation_boundary(*session->head) == before,
      "draft proposals published state or advanced RNG");
  std::array<std::vector<double>, 3> draft_probabilities;
  if (native_backend == q::BackendKind::kCuda) for (std::size_t row = 0; row < probabilities.size(); ++row) {
    std::vector<float> log_q(static_cast<std::size_t>(model.vocabulary));
    q::check(model.execution.read(q::take(probabilities[row].view()), std::as_writable_bytes(std::span(log_q))));
    auto &distribution = draft_probabilities[row]; distribution.reserve(log_q.size());
    double sum = 0.;
    for (const auto value : log_q) {
      require(!std::isnan(value) && value != std::numeric_limits<float>::infinity(), "invalid actual draft log probability");
      const auto mass = std::exp(static_cast<double>(value)); distribution.push_back(mass); sum += mass;
    }
    require(std::abs(sum - 1.) < 1e-4, "actual draft distribution is not normalized");
    require(reference_draw(distribution, before.sampling.identity(2 * row)) == ids[row + 1],
        "actual draft token differs from independent q/Philox draw");
  }
  q::prepare_generation_record(ledger, *session, q::GraphRole::kVerify,
      std::span(&pending, 1), true, 4, 0, {}, 3);
  model.prepare_evaluation(*session, ids, nullptr, q::GraphRole::kVerify);
  model.submit_evaluation(*session);
  const q::EvaluationIO io{.name = "random_recursive_round", .trace_states = false};
  auto output = q::evaluation_output(model, session, io);
  output.capture_tensors = native_backend == q::BackendKind::kCuda;
  const std::array outputs{output};
  const std::array selections{q::MtpGenerationSelection{session, pending, 4, probabilities, 3}};
  std::array<q::MtpRoundResult, 1> results;
  q::GenerationPublicationBatch publication({{&ledger, expected}});
  std::vector<float> target_logits;
  model.finish_recursive_mtp(selections, q::model_eos_tokens, outputs, results,
      [&](std::size_t j, const q::CompletedEvaluationView &view) {
        if (native_backend == q::BackendKind::kCuda) {
          require(j == 0 && view.logits.description() == q::dense(q::Type::kFloat32, {4, model.vocabulary}),
              "K3 verifier target logits have unexpected geometry");
          target_logits.resize(4 * static_cast<std::size_t>(model.vocabulary));
          q::check(model.execution.read(view.logits.storage(), std::as_writable_bytes(std::span(target_logits))));
        }
        q::observe_completed_evaluation(model, outputs[j], io, view);
      }, publication);
  if (native_backend == q::BackendKind::kCuda) {
    const auto vocabulary = static_cast<std::size_t>(model.vocabulary);
    std::array<std::vector<double>, 4> target;
    for (std::size_t row = 0; row < target.size(); ++row)
      target[row] = reference_target(std::span(target_logits).subspan(row * vocabulary, vocabulary));
    std::vector<std::int32_t> expected_ids;
    bool rejected = false;
    for (std::size_t row = 0; row < 3; ++row) {
      const auto id = static_cast<std::size_t>(ids[row + 1]);
      const auto q_mass = draft_probabilities[row][id];
      require(q_mass > 0., "actual sampled proposal has zero q mass");
      const auto alpha = (std::min)(1., target[row][id] / q_mass);
      const std::array accept_distribution{alpha, 1. - alpha};
      if (reference_draw(accept_distribution, before.sampling.identity(2 * row + 1)) == 0) {
        expected_ids.push_back(ids[row + 1]);
        continue;
      }
      std::vector<double> residual(vocabulary);
      for (std::size_t i = 0; i < vocabulary; ++i)
        residual[i] = (std::max)(0., target[row][i] - draft_probabilities[row][i]);
      expected_ids.push_back(reference_draw(residual, before.sampling.identity(6)));
      rejected = true; break;
    }
    if (!rejected) expected_ids.push_back(reference_draw(target[3], before.sampling.identity(6)));
    require(results[0].emitted == expected_ids, "actual K3 output differs from independent p/q acceptance and draw");
  }
  return std::move(results[0]);
}

void run_recursive(const std::filesystem::path &input) {
  const auto source = q::read_ids(input / "prompt.ids");
  require(source.size() >= 15, "integration check requires fifteen existing prompt tokens");
  const auto prompt = std::span(source).first(15);
  const auto geometry = q::probe::read_two_b_geometry(input);
  auto weights = q::prepare_weight_source(input / "weights.tsv", true);
  const auto precision = q::PrecisionProfile::from_roles(weights.roles);
  const auto plan = q::cache::plan(geometry, weights.cache_row_bytes, 64, 3, 15,
      q::recurrent_bytes(q::state_schema_for(geometry, 64, precision), geometry), 1, 4, true, true, 1, 0, precision);
  q::Model model(std::move(weights), geometry, 64, plan, {}, true, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.5F, 40, .95F}, .seed = 123}, 0, {.primary=native_backend});
  model.configure_draft_policy({.max_drafts = 3});
  const std::array prefill{std::size_t{15}}, verify{std::size_t{4}};
  model.prepare_family(prefill, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(15, 1, 12, q::GraphRole::kMtpPrefill);
  model.prepare_family(verify, false, 1, q::GraphRole::kVerify);
  model.plan_replay_uses(4, 1, 12, q::GraphRole::kVerify);
  model.prepare_recursive_round_programs();
  auto session = model.create_session(prefill, {}, q::GraphRole::kMtpPrefill);
  q::GenerationLedger ledger(2, 4);
  q::prepare_generation_record(ledger, *session, q::GraphRole::kMtpPrefill, prompt, false, 0, 0);
  model.prepare_evaluation(*session, prompt, nullptr, q::GraphRole::kMtpPrefill); model.submit_evaluation(*session);
  (void)q::finish_evaluation(model, *session, "prefill", {}, false, nullptr, nullptr, nullptr, nullptr, {}, &ledger);
  auto base = session->checkpoint();
  require(base->sampling == q::cache::SamplingCursor{123, 0, 1}, "K3 prefill did not publish exactly one draw");
  const auto result = recursive_round(model, session, ledger);
  const auto final = q::generation_boundary(*session->head);
  require(final.sampling.draw == 8 && ledger.committed_count == 2,
      "K3 round did not atomically publish its seven addresses and ledger");
  retire(session);
  q::GenerationRestoreStats stats;
  std::cerr << "random_recursive_stage=strict_replay\n";
  session = q::restore_mtp_session(model, ledger, {}, false, &stats);
  require(q::generation_boundary(*session->head) == final && stats.graph_calls == 2,
      "strict K3 replay changed a sampled decision or RNG boundary");
  retire(session);
  session = model.create_session({}, base, q::GraphRole::kMtpPrefill);
  q::GenerationLedger rejected(1, 4);
  auto damaged = ledger.records[1]; damaged.draft = static_cast<std::int32_t>((damaged.draft + 1) % geometry.vocabulary);
  std::cerr << "random_recursive_stage=rejected_commit\n";
  bool failed = false;
  try { (void)recursive_round(model, session, rejected, &damaged); }
  catch (const std::runtime_error &error) {
    if (!expected_replay_mismatch(error)) throw;
    failed = true; std::cerr << "random_recursive_expected_failure=" << error.what() << '\n';
  }
  require(failed && session->head.get() == base.get() && session->head->sampling.draw == 1 &&
      !rejected.committed_count && !model.active_step && !model.carrier_borrowed,
      "failed K3 episode published a partial head, cursor or ledger");
  std::cerr << "random_recursive_stage=retry\n";
  const auto retry = recursive_round(model, session, rejected, &ledger.records[1]);
  require(retry.emitted == result.emitted && q::generation_boundary(*session->head) == final,
      "K3 retry did not reproduce the original stochastic decision");
  retire(session);
  base.reset(); require(model.snapshot_versions.live == 0, "K3 transaction retained a model-state version");
  std::cout << "{\"status\":\"PASS_RANDOM_K3_TRANSACTION\",\"backend\":\"" << model.execution.name()
      << "\",\"strict_replay\":true,\"failed_commit_retry\":true,\"actual_q_rows\":3,"
         "\"independent_pq_philox_check\":" << (native_backend == q::BackendKind::kCuda ? "true" : "false")
      << ",\"performance_claim\":false,\"quality_claim\":false}\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 3 || std::string_view(argv[1]) != "--inputs")
      throw std::runtime_error("usage: qwen_random_mtp_probe --inputs EXISTING_PREPARED_MTP_DIR");
    run(argv[2]); run_recursive(argv[2]); return 0;
  } catch (const std::exception &error) { std::cerr << "random MTP integration: " << error.what() << '\n'; return 1; }
}
