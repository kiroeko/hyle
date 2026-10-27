#pragma once

#include "qwen_mtp_batch_state_case.h"
#include "qwen_packed_prefill_case.h"

// A finite same-input state comparison, using only the cached untrained GA
// fixture and the original verifier/repair/Single owners. Controlled prefix
// selection checks rollback; it is not a natural draft-acceptance measurement.
namespace qwen_mtp_packed_prefill_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
using h::require;

struct Report final {
  q::GraphRole role = q::GraphRole::kMtpPrefill;
  std::size_t accepted = 0, draft_active = 0, draft_physical = 0;
  std::size_t frontier_before = 0, frontier_after = 0, lag_before = 0, lag_after = 0;
  p::StepReport state;
};

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto exact_source = q::prepare_weight_source(input / "weights.tsv", true);
  auto packed_source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = exact_source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "MTP packed prefill requires the cached C16 GA/MTP fixture");
  // The lag1 draft carrier has five rows even though its target has four.
  const auto plan = q::cache::plan(geometry, exact_source.cache_row_bytes, 16, 4, 5,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 3, true, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123};
  q::Model exact(std::move(exact_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  q::Model packed(std::move(packed_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array lag0_rows{std::size_t{2}, std::size_t{4}};
  const std::array lag1_rows{std::size_t{2}, std::size_t{3}};
  const std::array carrier{std::size_t{4}}, verify{std::size_t{3}}, single{std::size_t{1}};
  exact.prepare_family(lag0_rows, false, 1, q::GraphRole::kMtpPrefill);
  exact.prepare_family(lag1_rows, false, 1, q::GraphRole::kMtpCatchupPrefill);
  for (const auto rows : lag0_rows) exact.plan_replay_uses(rows, 1, 2, q::GraphRole::kMtpPrefill);
  for (const auto rows : lag1_rows) exact.plan_replay_uses(rows, 1, 2, q::GraphRole::kMtpCatchupPrefill);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kMtpCatchupPrefill}) {
    require(packed.prepare_prefill_family(carrier, q::TextCompletion::kEmitNext, 0, 1, role),
        "selected route did not prepare its real packed MTP prefill family");
    packed.plan_replay_uses(4, 1, 4, role, q::TextCompletion::kEmitNext, 0, true);
  }
  for (auto *model : {&exact, &packed}) {
    model->prepare_family(verify, false, 1, q::GraphRole::kVerify);
    model->plan_replay_uses(3, 1, 2, q::GraphRole::kVerify);
    model->prepare_family(single, false, 1, q::GraphRole::kMtpSingle);
    model->plan_replay_uses(1, 1, 1, q::GraphRole::kMtpSingle);
    // Accepting two target rows from a lag0 base repairs one actual draft row.
    model->prepare_draft_program(1, false);
  }
  exact.prepare_execution(lag0_rows, 1, q::GraphRole::kMtpPrefill);
  exact.prepare_execution(lag1_rows, 1, q::GraphRole::kMtpCatchupPrefill);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kMtpCatchupPrefill})
    packed.prepare_prefill_execution(carrier, q::TextCompletion::kEmitNext, 0, 1, role);
  for (auto *model : {&exact, &packed}) {
    model->prepare_execution(verify, 1, q::GraphRole::kVerify);
    model->prepare_execution(single, 1, q::GraphRole::kMtpSingle);
  }
  auto exact_storage = q::take(hyle::inference::HostStateStorage::create(exact.execution, 1, 4u << 20));
  auto packed_storage = q::take(hyle::inference::HostStateStorage::create(packed.execution, 1, 4u << 20));
  const std::array initial{std::size_t{2}};
  const q::cache::SamplingCursor cursor{123, 7, 0};
  auto a = exact.create_session(initial, {}, q::GraphRole::kMtpPrefill, cursor);
  auto b = packed.create_session(initial, {}, q::GraphRole::kMtpPrefill, cursor);
  std::array<const q::Phase *, 2> carriers{};
  std::vector<Report> reports; reports.reserve(7);
  q::cache::SnapshotRef frozen_a, frozen_b;
  s::StateImage frozen_image_a, frozen_image_b;

  const auto advance = [&](std::size_t rows, q::GraphRole role) {
    const bool verification = role == q::GraphRole::kVerify;
    const bool prefill = q::is_mtp_prefill(role);
    const auto accepted = verification ? std::size_t{2} : rows;
    const auto old_a = a->checkpoint(), old_b = b->checkpoint();
    const auto before_a = s::image(exact, *old_a), before_b = s::image(packed, *old_b);
    const auto epoch = a->head.epoch();
    require(a->length() == b->length() && epoch == b->head.epoch() &&
        before_a.sampling == before_b.sampling && before_a.lag == before_b.lag &&
        before_a.frontier == before_b.frontier && before_a.pending == before_b.pending &&
        (!verification || !before_a.lag), "MTP controls do not begin at the same complete boundary");
    std::vector<std::int32_t> ids(rows);
    for (std::size_t row = 0; row < rows; ++row)
      ids[row] = static_cast<std::int32_t>(1 + (a->length() + row) % 15);
    if (verification || !prefill) ids.front() = a->head->next_id;
    Report report; report.role = role; report.accepted = accepted;
    report.frontier_before = before_b.frontier; report.lag_before = before_b.lag;
    report.state = p::StepReport{0, a->length(), rows, prefill ? 4u : rows};
    report.state.prefill = prefill;

    const auto execute = [&](q::Model &model, const std::shared_ptr<q::Session> &session, bool candidate) {
      const std::array requests{q::EvaluationRequest{session, ids}};
      if (candidate && prefill) model.prepare_prefill_evaluations(requests, nullptr, role);
      else model.prepare_evaluations(requests, nullptr, role);
      auto &step = *model.active_step; auto &member = step.members[0];
      require(step.members.size() == 1 && member.session == session && member.input_tokens == rows &&
          member.layout.facts().query_rows == rows && member.layout.facts().kv_rows == before_a.length + rows,
          "MTP target append topology consumed a physical padding row");
      if (prefill || role == q::GraphRole::kMtpSingle) {
        const auto draft_rows = rows + q::draft_lag(role);
        require(member.draft_layout.facts().query_rows == draft_rows &&
            member.draft_layout.facts().kv_rows == before_a.frontier + draft_rows &&
            member.next->draft->frontier == before_a.frontier + draft_rows,
            "MTP draft append/frontier differs from its actual Li plus lag");
        if (candidate) {
          report.draft_active = draft_rows;
          report.draft_physical = prefill ? 4 + q::draft_lag(role) : draft_rows;
          const auto &io = step.phase->io[step.state_slot];
          require(io.draft_positions.description.dimensions().size() == 1 &&
              io.draft_positions.description.dimensions()[0] == static_cast<std::int64_t>(report.draft_physical),
              "MTP draft position carrier no longer has its real cold capacity");
        }
      }
      if (candidate && prefill) {
        require(step.phase->query_rows == 4 && step.phase->family->packed_prefill &&
            step.phase->family->role == role && step.active_input_rows == rows,
            "MTP prefill did not use the original packed R4 target carrier");
        const auto slot = q::draft_lag(role);
        if (carriers[slot]) require(carriers[slot] == step.phase,
            "MTP effective length silently compiled a different physical carrier");
        else carriers[slot] = step.phase;
      }
      model.submit_evaluations();
      const q::EvaluationIO io{.name = candidate ? "mtp-packed" : "mtp-exact", .trace_states = false};
      const std::array outputs{q::evaluation_output(model, session, io)};
      std::array<std::int32_t, 1> next{-1};
      const auto observe = [&](std::size_t member_index, const q::CompletedEvaluationView &view) {
        require(member_index == 0 && view.requests == 1 && view.observation.input_tokens == rows &&
            view.observation.graph_input_tokens == rows, "MTP observation reported physical rather than actual rows");
      };
      if (verification) model.finish_selected_evaluations(outputs, next,
          [](std::span<const std::int32_t> candidates, q::StepState &original, std::span<std::int32_t> selected) {
            require(candidates.size() == 3 && original.members.size() == 1 && original.input_tokens == 3,
                "controlled rejected suffix did not execute the original K3 verifier");
            original.members[0].accepted_tokens = 2; selected[0] = candidates[1];
          }, [&](q::StepState &original) { model.repair_recursive_draft(original); }, observe);
      else model.finish_evaluations(outputs, next, observe);
      require(!model.active_step && !model.successor_step && !session->pending && !model.carrier_borrowed,
          "MTP packed journey did not retire its original Graph episode");
      const auto &before = candidate ? before_b : before_a;
      const auto after = s::image(model, *session->head);
      require(before.target.size() == after.target.size() && before.payloads.size() == after.payloads.size() &&
          before.payloads.size() >= 4, "MTP complete prefix payload count changed");
      for (std::size_t bank = 0; bank < before.target.size(); ++bank)
        require(after.target[bank].size() >= before.target[bank].size() &&
            std::equal(before.target[bank].begin(), before.target[bank].end(), after.target[bank].begin()),
            "MTP target append changed its already initialized prefix bytes");
      for (std::size_t bank = before.payloads.size() - 4; bank < before.payloads.size() - 2; ++bank)
        require(after.payloads[bank].size() >= before.payloads[bank].size() &&
            std::equal(before.payloads[bank].begin(), before.payloads[bank].end(), after.payloads[bank].begin()),
            "MTP draft append or repair changed its already initialized prefix bytes");
      if (candidate && verification) {
        const auto &program = *model.draft_programs.at({1, false});
        report.draft_active = after.frontier - before.frontier;
        report.draft_physical = program.rows;
        require(report.draft_active == 1 && report.draft_physical == 1,
            "controlled target-prefix repair did not execute its original one-row NextN program");
      }
      return next[0];
    };
    report.state.exact_next = execute(exact, a, false);
    report.state.bucket_next = execute(packed, b, true);
    const auto expected_length = before_a.length + accepted;
    const auto expected_frontier = verification ? expected_length - 1 : expected_length;
    const auto expected_cursor = before_a.sampling.advanced(verification ? 0u : 1u);
    for (const auto &session : {a, b}) {
      require(session->length() == expected_length && session->head.epoch() == epoch + 1 &&
          session->head->sampling == expected_cursor && session->head->draft->frontier == expected_frontier &&
          session->head->draft->lag == (verification ? 1u : 0u) &&
          session->head->draft->catchup_token == (verification ? ids[1] : -1),
          "MTP effective-length commit changed actual length/frontier/lag/RNG/epoch");
    }
    report.state.epoch = static_cast<std::size_t>(b->head.epoch());
    report.frontier_after = b->head->draft->frontier; report.lag_after = b->head->draft->lag;
    p::compare(exact, p::capture(exact, *a, exact_storage), p::capture(packed, *b, packed_storage), report.state);
    require(s::image(exact, *old_a) == before_a && s::image(packed, *old_b) == before_b,
        "MTP prefill or rejected suffix mutated its original full checkpoint");
    if (frozen_a) require(s::image(exact, *frozen_a) == frozen_image_a &&
        s::image(packed, *frozen_b) == frozen_image_b, "MTP continuation mutated its independently retained first prefix");
    reports.push_back(std::move(report));
  };

  advance(2, q::GraphRole::kMtpPrefill);
  frozen_a = a->checkpoint(); frozen_b = b->checkpoint();
  frozen_image_a = s::image(exact, *frozen_a); frozen_image_b = s::image(packed, *frozen_b);
  advance(4, q::GraphRole::kMtpPrefill);
  advance(3, q::GraphRole::kVerify);
  advance(2, q::GraphRole::kMtpCatchupPrefill);
  advance(3, q::GraphRole::kVerify);
  advance(3, q::GraphRole::kMtpCatchupPrefill);
  advance(1, q::GraphRole::kMtpSingle);
  require(a->length() == 16 && b->length() == 16 && a->head->sampling.draw == 5,
      "MTP seven-step journey did not consume exactly its real target rows and five draws");
  frozen_a.reset(); frozen_b.reset(); h::retire(a); h::retire(b);
  for (auto *model : {&exact, &packed}) {
    require(!model->snapshot_versions.live && !model->active_step && !model->successor_step && !model->carrier_borrowed &&
        model->page_inventory.free_ids.size() == plan.max_pages && model->draft_inventory.free_ids.size() == plan.max_pages,
        "MTP journey retained a state version, episode or target/draft page permission");
    for (const auto &[key, program] : model->draft_programs) {
      (void)key;
      require(!program->guard.has_completion() && !program->guard.cleanup_pending &&
          !program->hidden_transactions.bound() && !program->retirement.pending(),
          "MTP repair retained a NextN completion or state transaction");
    }
  }
  require(!exact_storage->live_images() && !packed_storage->live_images(), "MTP journey retained a host image");
  bool equal = true, byte_equal = true;
  std::cout << "{\"backend\":\"" << h::backend_name(kind)
      << "\",\"mode\":\"mtp_packed_prefill\",\"synthetic_untrained\":true,\"target_capacity\":4,\"lag1_draft_capacity\":5,\"steps\":[";
  for (std::size_t at = 0; at < reports.size(); ++at) {
    const auto &report = reports[at]; equal = equal && report.state.numerically_equivalent;
    byte_equal = byte_equal && report.state.payloads_equal;
    std::cout << (at ? "," : "") << "{\"role\":" << static_cast<int>(report.role)
        << ",\"accepted_target_rows\":" << report.accepted << ",\"draft_active_rows\":" << report.draft_active
        << ",\"draft_physical_rows\":" << report.draft_physical << ",\"frontier_before\":" << report.frontier_before
        << ",\"frontier_after\":" << report.frontier_after << ",\"lag_before\":" << report.lag_before
        << ",\"lag_after\":" << report.lag_after << ",\"state\":"; p::write_step(report.state); std::cout << '}';
  }
  std::cout << "],\"controlled_verify_width\":3,\"controlled_accepted_rows\":2,\"discarded_target_rows\":2"
      << ",\"old_checkpoints_byte_unchanged\":true,\"rng_and_epochs_equal\":true,\"final_length\":16"
      << ",\"final_frontier\":16,\"final_rng_draws\":5,\"final_versions\":0,\"final_images\":0,\"final_native_episodes\":0"
      << ",\"payloads_byte_equal\":" << (byte_equal ? "true" : "false")
      << ",\"numerically_equivalent\":" << (equal ? "true" : "false")
      << ",\"gdn_state_absolute_budget\":" << p::gdn_state_absolute_budget
      << ",\"numerical_budget_basis\":\"packed_sequence_case.h independent FP64 component oracle\""
      << ",\"natural_acceptance_claim\":false,\"performance_claim\":false,\"status\":\""
      << (equal ? "PASS" : "VALUE_OR_CONTRACT_DIFFERENCE") << "\"}\n";
  require(equal, "MTP packed versus exact complete state exceeded its original numeric or exact metadata/payload contract");
  return 0;
}
} // namespace qwen_mtp_packed_prefill_case
