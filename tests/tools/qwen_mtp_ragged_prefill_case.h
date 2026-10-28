#pragma once

#include "qwen_mtp_packed_prefill_case.h"

// Seven fixed batches in the original Model. Exact R1 members are independent
// controls for packed B2, including the original verifier and draft repair.
namespace qwen_mtp_ragged_prefill_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
namespace m = qwen_mtp_packed_prefill_case;
using h::require;

struct BatchReport final {
  q::GraphRole role = q::GraphRole::kMtpPrefill;
  std::array<std::size_t, 2> order{}, lengths{};
  std::size_t active = 0, physical = 0, accepted = 0, draft_active = 0, draft_physical = 0;
  std::array<m::Report, 2> members;
  std::vector<std::int32_t> hidden_indices, token_indices;
};

inline std::vector<std::int32_t> read_indices(q::Model &model, q::Tensor &tensor) {
  require(tensor.description.data_type() == q::Type::kInt32,
      "MTP selector or row control changed its original I32 type");
  const auto bytes = q::take(tensor.description.required_bytes());
  require(bytes % sizeof(std::int32_t) == 0, "MTP I32 carrier contains a partial element");
  std::vector<std::int32_t> result(bytes / sizeof(std::int32_t));
  q::check(model.execution.read(tensor.read().storage(), std::as_writable_bytes(std::span(result))));
  return result;
}

inline void preserved_prefix(const s::StateImage &before, const s::StateImage &after) {
  require(before.target.size() == after.target.size() && before.payloads.size() == after.payloads.size() &&
      before.payloads.size() >= 4, "ragged MTP complete prefix payload count changed");
  for (std::size_t bank = 0; bank < before.target.size(); ++bank)
    require(after.target[bank].size() >= before.target[bank].size() &&
        std::equal(before.target[bank].begin(), before.target[bank].end(), after.target[bank].begin()),
        "ragged MTP target append changed an already initialized prefix");
  for (std::size_t bank = before.payloads.size() - 4; bank < before.payloads.size() - 2; ++bank)
    require(after.payloads[bank].size() >= before.payloads[bank].size() &&
        std::equal(before.payloads[bank].begin(), before.payloads[bank].end(), after.payloads[bank].begin()),
        "ragged MTP draft append or repair changed an already initialized prefix");
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto exact_source = q::prepare_weight_source(input / "weights.tsv", true);
  auto packed_source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = exact_source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "ragged MTP prefill requires the cached C16 GA/MTP fixture");
  const auto plan = q::cache::plan(geometry, exact_source.cache_row_bytes, 16, 6, 5,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123};
  q::Model exact(std::move(exact_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  q::Model packed(std::move(packed_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array exact_rows{std::size_t{2}, std::size_t{3}, std::size_t{4}};
  const std::array carrier{std::size_t{4}}, verify{std::size_t{3}}, single{std::size_t{1}};
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kMtpCatchupPrefill}) {
    exact.prepare_family(exact_rows, false, 1, role);
    for (const auto rows : exact_rows) exact.plan_replay_uses(rows, 1, 4, role);
    require(packed.prepare_prefill_family(carrier, q::TextCompletion::kEmitNext, 0, 2, role),
        "selected route did not prepare its original B2 packed MTP family");
    packed.plan_replay_uses(4, 2, 4, role, q::TextCompletion::kEmitNext, 0, q::RowPacking::kPrefill);
  }
  for (const auto &[model, width] : {std::pair{&exact, std::size_t{1}}, std::pair{&packed, std::size_t{2}}}) {
    model->prepare_family(verify, false, width, q::GraphRole::kVerify);
    model->plan_replay_uses(3, width, 4, q::GraphRole::kVerify);
    model->prepare_family(single, false, width, q::GraphRole::kMtpSingle);
    model->plan_replay_uses(1, width, 2, q::GraphRole::kMtpSingle);
    model->prepare_draft_program(1, false);
  }
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kMtpCatchupPrefill}) {
    exact.prepare_execution(exact_rows, 1, role);
    packed.prepare_prefill_execution(carrier, q::TextCompletion::kEmitNext, 0, 2, role);
  }
  for (const auto &[model, width] : {std::pair{&exact, std::size_t{1}}, std::pair{&packed, std::size_t{2}}}) {
    model->prepare_execution(verify, width, q::GraphRole::kVerify);
    model->prepare_execution(single, width, q::GraphRole::kMtpSingle);
  }
  auto exact_storage = q::take(hyle::inference::HostStateStorage::create(exact.execution, 1, 4u << 20));
  auto packed_storage = q::take(hyle::inference::HostStateStorage::create(packed.execution, 1, 4u << 20));
  std::array<std::shared_ptr<q::Session>, 2> a, b;
  const std::array admission{std::size_t{2}};
  for (std::size_t member = 0; member < a.size(); ++member) {
    const q::cache::SamplingCursor cursor{123, member + 7, 0};
    a[member] = exact.create_session(admission, {}, q::GraphRole::kMtpPrefill, cursor);
    b[member] = packed.create_session(admission, {}, q::GraphRole::kMtpPrefill, cursor);
  }
  std::array<q::cache::SnapshotRef, 2> frozen_a, frozen_b;
  std::array<s::StateImage, 2> frozen_image_a, frozen_image_b;
  std::array<const q::Phase *, 2> carriers{};
  std::vector<BatchReport> reports; reports.reserve(7);

  const auto advance = [&](std::array<std::size_t, 2> order, std::array<std::size_t, 2> lengths,
      q::GraphRole role, std::initializer_list<std::int32_t> hidden_indices = {},
      std::initializer_list<std::int32_t> token_indices = {}) {
    const bool prefill = q::is_mtp_prefill(role), verification = role == q::GraphRole::kVerify;
    const auto lag = q::draft_lag(role);
    BatchReport report; report.role = role; report.order = order; report.lengths = lengths;
    report.active = lengths[0] + lengths[1]; report.physical = prefill ? 8 : report.active;
    report.hidden_indices.assign(hidden_indices); report.token_indices.assign(token_indices);
    std::array<q::cache::SnapshotRef, 2> old_a, old_b;
    std::array<s::StateImage, 2> before_a, before_b;
    std::array<std::uint64_t, 2> epochs{};
    std::array<std::vector<std::int32_t>, 2> ids;
    std::array<q::EvaluationRequest, 2> requests;
    std::array<q::EvaluationOutput, 2> outputs;
    for (std::size_t j = 0; j < order.size(); ++j) {
      const auto member = order[j];
      old_a[j] = a[member]->checkpoint(); old_b[j] = b[member]->checkpoint();
      before_a[j] = s::image(exact, *old_a[j]); before_b[j] = s::image(packed, *old_b[j]);
      epochs[j] = a[member]->head.epoch();
      require(a[member]->length() == b[member]->length() && epochs[j] == b[member]->head.epoch() &&
          before_a[j].sampling == before_b[j].sampling && before_a[j].lag == lag && before_b[j].lag == lag &&
          before_a[j].frontier == before_b[j].frontier && before_a[j].pending == before_b[j].pending,
          "ragged MTP cohort does not share its original per-member role/lag/boundaries");
      ids[j].resize(lengths[j]);
      for (std::size_t row = 0; row < lengths[j]; ++row)
        ids[j][row] = static_cast<std::int32_t>(1 + (a[member]->length() + row + 7 * member) % 15);
      if (verification || !prefill) ids[j].front() = a[member]->head->next_id;
      requests[j] = {b[member], ids[j]}; outputs[j].session = b[member];
      auto &facts = report.members[j]; facts.role = role;
      facts.accepted = verification ? 2 : lengths[j]; report.accepted += facts.accepted;
      facts.frontier_before = before_b[j].frontier; facts.lag_before = before_b[j].lag;
      facts.state = p::StepReport{member, before_b[j].length, lengths[j], prefill ? 4u : lengths[j]};
      facts.state.prefill = prefill;
      exact.prepare_evaluation(*a[member], ids[j], nullptr, role); exact.submit_evaluation(*a[member]);
      const std::array control_output{q::EvaluationOutput{.session = a[member]}};
      std::array<std::int32_t, 1> expected{-1};
      if (verification) exact.finish_selected_evaluations(control_output, expected,
          [](std::span<const std::int32_t> candidates, q::StepState &step, std::span<std::int32_t> selected) {
            require(candidates.size() == 3 && step.members.size() == 1 && step.input_tokens == 3,
                "ragged MTP control did not execute its original K3 verifier");
            step.members[0].accepted_tokens = 2; selected[0] = candidates[1];
          }, [&](q::StepState &step) { exact.repair_recursive_draft(step); }, [](auto, const auto &) {});
      else exact.finish_evaluations(control_output, expected, [](auto, const auto &) {});
      facts.state.exact_next = expected[0];
    }
    if (prefill) packed.prepare_prefill_evaluations(requests, nullptr, role);
    else packed.prepare_evaluations(requests, nullptr, role);
    auto &step = *packed.active_step;
    require(step.members.size() == 2 && step.phase->family->requests == 2 && step.phase->family->role == role &&
        step.active_input_rows == report.active && step.phase->query_rows == (prefill ? 4 : lengths[0]),
        "ragged MTP did not keep its true B2 owner and actual versus physical target rows");
    for (std::size_t j = 0; j < order.size(); ++j) {
      const auto &member = step.members[j];
      require(member.session == b[order[j]] && member.input_tokens == lengths[j] &&
          member.input_row_begin == (j ? lengths[0] : 0) && member.layout.facts().query_rows == lengths[j] &&
          member.layout.facts().kv_rows == before_b[j].length + lengths[j],
          "ragged MTP target layout mixed member order, length or physical padding");
      if (!verification) {
        require(member.draft_layout.facts().query_rows == lengths[j] + lag &&
            member.draft_layout.facts().kv_rows == before_b[j].frontier + lengths[j] + lag &&
            member.next->draft->frontier == before_b[j].frontier + lengths[j] + lag,
            "ragged MTP draft layout advanced a padding row or another member's frontier");
      }
    }
    if (prefill) {
      require(q::packs_prefill(step.phase->family->packing), "ragged MTP prefill silently used an exact tail family");
      if (carriers[lag]) require(carriers[lag] == step.phase, "ragged MTP compiled a new carrier for Li or member order");
      else carriers[lag] = step.phase;
      auto &carrier_io = step.phase->io[step.state_slot];
      require(read_indices(packed, carrier_io.draft_hidden_select) == report.hidden_indices,
          "actual hidden selector does not match the independently fixed per-member shifted rows");
      if (lag) require(read_indices(packed, carrier_io.draft_token_select) == report.token_indices,
          "actual token selector does not match the independently fixed catch-up/target order");
      else require(report.token_indices.empty() && !carrier_io.draft_token_select.caller_owned(),
          "lag0 prefill unexpectedly introduced a catch-up token selector");
      std::vector<std::int32_t> target_tokens(8), target_positions(8), draft_positions(8 + 2 * lag);
      std::size_t target_row = 0, draft_row = 0;
      for (std::size_t j = 0; j < order.size(); ++j) {
        for (std::size_t row = 0; row < lengths[j]; ++row) {
          target_tokens[target_row] = ids[j][row];
          target_positions[target_row++] = static_cast<std::int32_t>(before_b[j].length + row);
        }
        for (std::size_t row = 0; row < lengths[j] + lag; ++row)
          draft_positions[draft_row++] = static_cast<std::int32_t>(before_b[j].frontier + row);
      }
      const std::vector<std::int32_t> last{static_cast<std::int32_t>(lengths[0] - 1),
          static_cast<std::int32_t>(report.active - 1)};
      require(read_indices(packed, carrier_io.tokens) == target_tokens &&
          read_indices(packed, carrier_io.positions) == target_positions &&
          read_indices(packed, carrier_io.draft_positions) == draft_positions &&
          read_indices(packed, carrier_io.last_ids) == last,
          "ragged MTP actual token/position/last-row carriers differ from their initialized compact prefixes");
    }
    packed.submit_evaluations();
    std::array<std::int32_t, 2> next{-1, -1};
    std::array<std::size_t, 2> observed{};
    const auto observe = [&](std::size_t j, const q::CompletedEvaluationView &view) {
      require(j < order.size() && view.requests == 2 && view.member_index == j &&
          view.observation.input_tokens == lengths[j] && view.observation.graph_input_tokens == lengths[j],
          "ragged MTP observer reported another member's or physical row count");
      ++observed[j];
    };
    if (verification) packed.finish_selected_evaluations(outputs, next,
        [](std::span<const std::int32_t> candidates, q::StepState &original, std::span<std::int32_t> selected) {
          require(candidates.size() == 6 && original.members.size() == 2 && original.input_tokens == 3,
              "ragged MTP rejected suffixes did not execute one original B2/K3 Graph");
          for (std::size_t j = 0; j < 2; ++j) { original.members[j].accepted_tokens = 2; selected[j] = candidates[j * 3 + 1]; }
        }, [&](q::StepState &original) { packed.repair_recursive_draft(original); }, observe);
    else packed.finish_evaluations(outputs, next, observe);
    require(!packed.active_step && !packed.successor_step && !packed.carrier_borrowed &&
        !exact.active_step && !exact.successor_step && !exact.carrier_borrowed,
        "ragged MTP batch retained its original native episode");
    for (std::size_t j = 0; j < order.size(); ++j) {
      const auto member = order[j]; auto &facts = report.members[j];
      const auto expected_length = before_b[j].length + facts.accepted;
      const auto expected_frontier = expected_length - (verification ? 1 : 0);
      for (const auto &session : {a[member], b[member]})
        require(!session->pending && session->length() == expected_length && session->head.epoch() == epochs[j] + 1 &&
            session->head->sampling == before_b[j].sampling.advanced(verification ? 0u : 1u) &&
            session->head->draft->frontier == expected_frontier && session->head->draft->lag == (verification ? 1u : 0u) &&
            session->head->draft->catchup_token == (verification ? ids[j][1] : -1),
            "ragged MTP commit changed real length/frontier/lag/epoch/RNG or member identity");
      require(observed[j] == 1 && b[member]->head->next_id == next[j], "ragged MTP duplicated or lost a member publication");
      facts.state.bucket_next = next[j]; facts.state.epoch = static_cast<std::size_t>(b[member]->head.epoch());
      facts.frontier_after = expected_frontier; facts.lag_after = verification ? 1 : 0;
      facts.draft_active = expected_frontier - before_b[j].frontier;
      facts.draft_physical = verification ? packed.draft_programs.at({1, false, 0})->rows : prefill ? 4 + lag : 1;
      require(!verification || (facts.draft_active == 1 && facts.draft_physical == 1),
          "ragged MTP prefix repair did not reuse the original one-row NextN program");
      report.draft_active += facts.draft_active; report.draft_physical += facts.draft_physical;
      p::compare(exact, p::capture(exact, *a[member], exact_storage), p::capture(packed, *b[member], packed_storage), facts.state);
      preserved_prefix(before_a[j], s::image(exact, *a[member]->head));
      preserved_prefix(before_b[j], s::image(packed, *b[member]->head));
      require(s::image(exact, *old_a[j]) == before_a[j] && s::image(packed, *old_b[j]) == before_b[j],
          "ragged MTP continuation mutated an original complete checkpoint");
      if (frozen_a[member]) require(s::image(exact, *frozen_a[member]) == frozen_image_a[member] &&
          s::image(packed, *frozen_b[member]) == frozen_image_b[member], "ragged MTP changed an independently frozen first prefix");
    }
    reports.push_back(std::move(report));
  };

  advance({0, 1}, {2, 3}, q::GraphRole::kMtpPrefill, {0, 2, 1, 4, 5, 0, 0, 0});
  for (std::size_t member = 0; member < a.size(); ++member) {
    frozen_a[member] = a[member]->checkpoint(); frozen_b[member] = b[member]->checkpoint();
    frozen_image_a[member] = s::image(exact, *frozen_a[member]); frozen_image_b[member] = s::image(packed, *frozen_b[member]);
  }
  advance({1, 0}, {2, 4}, q::GraphRole::kMtpPrefill, {0, 2, 1, 4, 5, 6, 0, 0});
  advance({0, 1}, {3, 3}, q::GraphRole::kVerify);
  advance({1, 0}, {2, 3}, q::GraphRole::kMtpCatchupPrefill,
      {2, 0, 4, 3, 1, 6, 7, 0, 0, 0}, {0, 2, 3, 1, 4, 5, 6, 0, 0, 0});
  advance({0, 1}, {3, 3}, q::GraphRole::kVerify);
  advance({0, 1}, {2, 4}, q::GraphRole::kMtpCatchupPrefill,
      {2, 0, 4, 3, 1, 6, 7, 8, 0, 0}, {0, 2, 3, 1, 4, 5, 6, 7, 0, 0});
  advance({1, 0}, {1, 1}, q::GraphRole::kMtpSingle);
  for (std::size_t member = 0; member < a.size(); ++member) {
    require(a[member]->length() == 16 && b[member]->length() == 16 &&
        a[member]->head->sampling.draw == 5 && b[member]->head->sampling.draw == 5,
        "ragged MTP journey did not consume exactly C16 and five independent draws per member");
    frozen_a[member].reset(); frozen_b[member].reset(); h::retire(a[member]); h::retire(b[member]);
  }
  for (auto *model : {&exact, &packed}) {
    require(!model->snapshot_versions.live && !model->active_step && !model->successor_step && !model->carrier_borrowed &&
        model->page_inventory.free_ids.size() == plan.max_pages && model->draft_inventory.free_ids.size() == plan.max_pages,
        "ragged MTP retirement retained a version, episode or target/draft page permission");
    for (const auto &[key, program] : model->draft_programs) {
      (void)key;
      require(!program->guard.execution.has_completion() && !program->guard.execution.cleanup_pending &&
          !program->hidden_transactions.bound() && !program->carrier.pending(), "ragged MTP repair did not really retire");
    }
  }
  require(!exact_storage->live_images() && !packed_storage->live_images(), "ragged MTP retained a host image");
  bool equal = true, byte_equal = true;
  std::size_t active = 0, physical = 0, accepted = 0, draft_active = 0, draft_physical = 0;
  for (const auto &report : reports) {
    active += report.active; physical += report.physical; accepted += report.accepted;
    draft_active += report.draft_active; draft_physical += report.draft_physical;
    for (const auto &member : report.members) {
      equal = equal && member.state.numerically_equivalent; byte_equal = byte_equal && member.state.payloads_equal;
    }
  }
  require(active == 36 && physical == 46 && accepted == 32 && draft_active == 32 && draft_physical == 42,
      "ragged MTP actual, physical or discarded work accounting changed");
  std::cout << "{\"backend\":\"" << h::backend_name(kind)
      << "\",\"mode\":\"mtp_ragged_prefill\",\"synthetic_untrained\":true,\"width\":2,\"target_capacity\":8,\"lag1_draft_capacity\":10,\"batches\":[";
  const auto write_indices = [](const std::vector<std::int32_t> &values) {
    std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i]; std::cout << ']';
  };
  for (std::size_t at = 0; at < reports.size(); ++at) {
    const auto &report = reports[at];
    std::cout << (at ? "," : "") << "{\"role\":" << static_cast<int>(report.role)
        << ",\"order\":[" << report.order[0] << ',' << report.order[1] << "],\"Li\":[" << report.lengths[0] << ',' << report.lengths[1]
        << "],\"active_rows\":" << report.active << ",\"physical_rows\":" << report.physical << ",\"accepted_rows\":" << report.accepted
        << ",\"draft_active_rows\":" << report.draft_active << ",\"draft_physical_rows\":" << report.draft_physical << ",\"hidden_indices\":";
    write_indices(report.hidden_indices); std::cout << ",\"token_indices\":"; write_indices(report.token_indices); std::cout << ",\"members\":[";
    for (std::size_t j = 0; j < report.members.size(); ++j) {
      const auto &member = report.members[j];
      std::cout << (j ? "," : "") << "{\"frontier_before\":" << member.frontier_before << ",\"frontier_after\":" << member.frontier_after
          << ",\"lag_before\":" << member.lag_before << ",\"lag_after\":" << member.lag_after << ",\"state\":";
      p::write_step(member.state); std::cout << '}';
    }
    std::cout << "]}";
  }
  std::cout << "],\"actual_target_rows\":" << active << ",\"physical_target_rows\":" << physical << ",\"committed_target_rows\":" << accepted
      << ",\"actual_draft_rows\":" << draft_active << ",\"physical_draft_rows\":" << draft_physical
      << ",\"controlled_verify_width\":3,\"controlled_accepted_rows\":2,\"discarded_target_rows\":4"
      << ",\"actual_i32_selectors_checked\":true,\"old_checkpoints_byte_unchanged\":true,\"rng_and_epochs_equal\":true"
      << ",\"final_lengths\":[16,16],\"final_frontiers\":[16,16],\"final_rng_draws\":[5,5],\"final_versions\":0,\"final_images\":0,\"final_native_episodes\":0"
      << ",\"payloads_byte_equal\":" << (byte_equal ? "true" : "false") << ",\"numerically_equivalent\":" << (equal ? "true" : "false")
      << ",\"gdn_state_absolute_budget\":" << p::gdn_state_absolute_budget
      << ",\"natural_acceptance_claim\":false,\"performance_claim\":false,\"status\":\"" << (equal ? "PASS" : "VALUE_OR_CONTRACT_DIFFERENCE") << "\"}\n";
  require(equal, "ragged MTP versus exact complete state exceeded its original numerical or exact payload/boundary contract");
  return 0;
}
} // namespace qwen_mtp_ragged_prefill_case
