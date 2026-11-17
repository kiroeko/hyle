#pragma once

#include "qwen_host_image_case.h"
#include "qwen_mtp_batch_state_case.h"
#include "qwen_packed_prefill_case.h"
#include "qwen_legacy_ledger_case.h"
#include "qwen_attention_observation_case.h"
#include "../../engine/models/qwen35/request_installation.h"

// Native Model-only correctness consumer. Uses existing local GA/MTP weights,
// the original strict greedy acceptance and original ordered ledger/Head
// publication. It does not replace acceptance, weights, Sources or retirement.
namespace qwen_device_continuation_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace s = qwen_mtp_batch_state_case;
namespace p = qwen_packed_prefill_case;
using h::require;
using AttentionObservation = qwen_attention_observation_case::AttentionObservation;
constexpr std::size_t episodes = 6;

struct Result final {
  std::vector<std::int32_t> ids;
  std::vector<q::GenerationCommitRecord> records;
  std::optional<s::StateImage> final_state;
  std::size_t submitted = 0, a1_published = 0, a2_published = 0, reused = 0, recorded = 0;
  std::size_t draft_rows = 0, native_prepared = 0;
  bool stopped = false;
};

inline q::GraphRole role(const q::Session &session) {
  return session.head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
}
inline void stage(q::GenerationLedger &ledger, const std::shared_ptr<q::Session> &session,
           std::size_t quota, q::GraphRole selected, std::size_t maximum = 1) {
  const auto pending = session->head->next_id;
  q::prepare_generation_record(ledger, *session, selected, std::span(&pending, 1), true, quota, 0, {}, maximum);
}
inline q::MtpRoundResult finish(q::Model &model, const std::shared_ptr<q::Session> &session,
                         q::GenerationLedger &ledger, std::int32_t pending, std::size_t quota,
                         AttentionObservation *observation = nullptr) {
  if (!observation) return q::finish_mtp_round(model, *session, pending, quota, q::model_eos_tokens, nullptr,
      {.name = "device-continuous-model-correctness", .trace_states = false}, {&ledger, nullptr});
  const std::array selections{q::MtpGenerationSelection{session, pending, quota}};
  const q::EvaluationIO io{.name = "device-continuous-model-correctness", .trace_states = false};
  const std::array outputs{q::evaluation_output(model, session, io)};
  std::array<q::MtpRoundResult, 1> results;
  q::GenerationPublicationBatch original({{&ledger, nullptr}});
  qwen_attention_observation_case::ObservedPublication publication(original, model, *observation);
  model.finish_mtp_rounds(selections, q::model_eos_tokens, outputs, results,
      [&](std::size_t j, const q::CompletedEvaluationView &view) {
        q::observe_completed_evaluation(model, outputs[j], io, view);
      }, publication);
  return std::move(results[0]);
}
inline void append(Result &result, const q::MtpRoundResult &round) {
  require(!round.stopped && !round.emitted.empty(), "continuation journey unexpectedly reached an EOS boundary");
  result.ids.insert(result.ids.end(), round.emitted.begin(), round.emitted.end());
}

inline void compare_state(q::Model &model, const s::StateImage &a, const s::StateImage &b) {
  require(std::tie(a.length, a.frontier, a.lag, a.pending, a.catchup, a.sampling) ==
      std::tie(b.length, b.frontier, b.lag, b.pending, b.catchup, b.sampling),
      "device choice changed the complete final boundary or logical RNG");
  require(a.target == b.target && a.payloads.size() == b.payloads.size(),
      "device choice changed target KV bytes or the complete payload inventory");
  std::vector<std::size_t> recurrent;
  for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer)
    if (model.geometry.is_gdn(layer)) recurrent.push_back(layer);
  for (std::size_t part = 0; part < a.payloads.size(); ++part) {
    if (a.payloads[part] == b.payloads[part]) continue;
    // Preserve the existing complete-state oracle: only newly evaluated F32
    // GDN admits its existing 3e-6 budget. KV/history/hidden stay byte exact.
    if (!(part < recurrent.size() * 2 && part % 2 == 1)) {
      std::cerr << "continuation_payload_difference part=" << part << " recurrent_payloads=" << recurrent.size() * 2
                << " bytes=" << a.payloads[part].size() << '\n';
      const auto first = std::mismatch(a.payloads[part].begin(), a.payloads[part].end(), b.payloads[part].begin()).first;
      const auto offset = static_cast<std::size_t>(first - a.payloads[part].begin());
      std::cerr << "first_different_byte=" << offset << '\n';
      if (part >= recurrent.size() * 2 && model.precision.cache_type() == q::Type::kBFloat16) {
        std::uint16_t x, y;
        std::memcpy(&x, a.payloads[part].data() + (offset / 2) * 2, 2);
        std::memcpy(&y, b.payloads[part].data() + (offset / 2) * 2, 2);
        std::cerr << "bf16_serial=" << std::bit_cast<float>(std::uint32_t(x) << 16)
                  << " bf16_device=" << std::bit_cast<float>(std::uint32_t(y) << 16) << '\n';
      }
      throw std::runtime_error("device choice changed draft KV, history or hidden bytes");
    }
    q::cache::HostPayloadDesc layout{};
    layout.role = q::cache::HostPayloadRole::kGdnState;
    layout.layer = static_cast<std::uint32_t>(recurrent[part / 2]);
    layout.bytes = a.payloads[part].size();
    const auto difference = p::compare_payload(model, layout, a.payloads[part], b.payloads[part]);
    require(!difference.nonfinite_differences && difference.maximum_absolute <= p::gdn_state_absolute_budget,
        "device choice changed F32 GDN beyond the existing absolute state budget");
  }
}

// Failure-only diagnostics of bytes already captured by journey(). No state
// image, device read, synchronization or publication is performed here.
inline void diagnose_saved_prefill(q::Model &model, const s::StateImage &a, const s::StateImage &b) {
  std::cerr << "prefill_saved_boundary old_length=" << a.length << " new_length=" << b.length
            << " old_frontier=" << a.frontier << " new_frontier=" << b.frontier
            << " old_lag=" << a.lag << " new_lag=" << b.lag
            << " old_catchup=" << a.catchup << " new_catchup=" << b.catchup << '\n';
  std::cerr << "prefill_saved_pending old=" << a.pending << " new=" << b.pending << '\n';
  std::cerr << "prefill_saved_logical_rng old_seed=" << a.sampling.seed << " new_seed=" << b.sampling.seed
            << " old_stream=" << a.sampling.stream << " new_stream=" << b.sampling.stream
            << " old_draw=" << a.sampling.draw << " new_draw=" << b.sampling.draw << '\n';
  std::cerr << "prefill_saved_inventory old_target=" << a.target.size() << " new_target=" << b.target.size()
            << " old_payloads=" << a.payloads.size() << " new_payloads=" << b.payloads.size()
            << " target_bytes_equal=" << (a.target == b.target)
            << " payload_bytes_equal=" << (a.payloads == b.payloads) << '\n';
  const auto payload = [&](const char *name, q::cache::HostPayloadRole role, std::size_t layer,
                           const s::Bytes &old, const s::Bytes &current) {
    std::cerr << "prefill_saved_payload role=" << name << " layer=" << layer
              << " old_bytes=" << old.size() << " new_bytes=" << current.size();
    if (old.size() != current.size()) { std::cerr << " numeric_unavailable=extent_mismatch\n"; return; }
    q::cache::HostPayloadDesc layout{};
    layout.role = role; layout.layer = static_cast<std::uint32_t>(layer); layout.bytes = old.size();
    p::PayloadDifference difference{layout, old.size()};
    if (old != current) difference = p::compare_payload(model, layout, old, current);
    std::cerr << " first_different_byte=" << difference.first_byte
              << " different_bytes=" << difference.different_bytes
              << " maximum_absolute=" << difference.maximum_absolute
              << " nonfinite_differences=" << difference.nonfinite_differences;
    if (role == q::cache::HostPayloadRole::kGdnState)
      std::cerr << " existing_gdn_absolute_budget=" << p::gdn_state_absolute_budget
                << " within_existing_budget=" << (!difference.nonfinite_differences &&
                    difference.maximum_absolute <= p::gdn_state_absolute_budget);
    std::cerr << '\n';
  };
  for (std::size_t bank = 0; bank < (std::min)(a.target.size(), b.target.size()) &&
      bank / 2 < model.attention_layer_ids.size(); ++bank)
    payload(bank % 2 ? "target_value" : "target_key",
        bank % 2 ? q::cache::HostPayloadRole::kTargetValue : q::cache::HostPayloadRole::kTargetKey,
        model.attention_layer_ids[bank / 2], a.target[bank], b.target[bank]);
  std::size_t part = 0;
  const auto next = [&](const char *name, q::cache::HostPayloadRole role, std::size_t layer) {
    if (part < a.payloads.size() && part < b.payloads.size())
      payload(name, role, layer, a.payloads[part], b.payloads[part]);
    ++part;
  };
  for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer) if (model.geometry.is_gdn(layer)) {
    next("history", q::cache::HostPayloadRole::kConvHistory, layer);
    next("gdn_state_f32", q::cache::HostPayloadRole::kGdnState, layer);
  }
  next("draft_key", q::cache::HostPayloadRole::kDraftKey, q::cache::HostPayloadDesc::no_layer);
  next("draft_value", q::cache::HostPayloadRole::kDraftValue, q::cache::HostPayloadDesc::no_layer);
  next("carry_hidden", q::cache::HostPayloadRole::kCarryHidden, q::cache::HostPayloadDesc::no_layer);
  next("prior_hidden", q::cache::HostPayloadRole::kPriorHidden, q::cache::HostPayloadDesc::no_layer);
}

inline void compare_record(const q::GenerationCommitRecord &x, const q::GenerationCommitRecord &y,
    std::size_t ordinal) {
    const bool equal = qwen_legacy_ledger_case::strict_equal(x, y);
    if (!equal) {
      // Keep the exact oracle. Report original host ledger facts only after
      // its failure; no extra device read or synchronization can hide it.
      std::cerr << "continuation_record_difference ordinal=" << ordinal << '\n';
      std::cerr << "ledger_context length=" << x.before.length << " pending=" << x.before.next_id
                << " draft=" << x.draft << " accepted=" << x.accepted
                << " consumed_drafts=" << x.consumed_drafts << " agreed=" << x.agreed
                << " accepted_draft=" << x.accepted_draft << " target_first=" << x.target_first
                << " target_second_old=" << x.target_second << " target_second_new=" << y.target_second << '\n';
      std::cerr << "ledger_verification_inputs=";
      for (const auto id : x.verification_inputs) std::cerr << id << ',';
      std::cerr << '\n';
      const auto scalar = [](const char *name, auto old, auto current) {
        if (old != current) std::cerr << "ledger_field=" << name << " old=" << old << " new=" << current << '\n';
      };
      scalar("role", static_cast<unsigned>(x.role), static_cast<unsigned>(y.role));
      scalar("completion", static_cast<unsigned>(x.completion), static_cast<unsigned>(y.completion));
      scalar("generation", x.generation, y.generation);
      scalar("committed", x.committed, y.committed);
      scalar("user_output", x.user_output, y.user_output);
      scalar("imported", x.imported, y.imported);
      scalar("input_deferred", x.input_deferred, y.input_deferred);
      scalar("prefill_complete", x.prefill_complete, y.prefill_complete);
      scalar("requests", x.requests, y.requests);
      scalar("member_index", x.member_index, y.member_index);
      scalar("input_begin", x.input_begin, y.input_begin);
      scalar("turn", x.turn, y.turn);
      scalar("quota", x.quota, y.quota);
      scalar("accepted", x.accepted, y.accepted);
      scalar("draft_rows", x.draft_rows, y.draft_rows);
      scalar("draft_limit", x.draft_limit, y.draft_limit);
      scalar("verified_rows", x.verified_rows, y.verified_rows);
      scalar("consumed_drafts", x.consumed_drafts, y.consumed_drafts);
      scalar("auxiliary_graph_calls", x.auxiliary_graph_calls, y.auxiliary_graph_calls);
      scalar("draft", x.draft, y.draft);
      scalar("target_first", x.target_first, y.target_first);
      scalar("target_second", x.target_second, y.target_second);
      scalar("agreed", x.agreed, y.agreed);
      scalar("accepted_draft", x.accepted_draft, y.accepted_draft);
      scalar("stopped", x.stopped, y.stopped);
      scalar("emitted_count", x.emitted_count, y.emitted_count);
      const auto boundary = [](const char *name, const q::GenerationBoundary &value) {
        std::cerr << "ledger_boundary=" << name << " length=" << value.length << " frontier=" << value.frontier
                  << " lag=" << value.lag << " pending=" << value.next_id << " catchup=" << value.catchup_token
                  << " rotary_offset=" << value.rotary_offset << " seed=" << value.sampling.seed
                  << " stream=" << value.sampling.stream << " draw=" << value.sampling.draw << '\n';
      };
      boundary("before_old", x.before); boundary("before_new", y.before);
      boundary("after_old", x.after); boundary("after_new", y.after);
      const auto tokens = [](const char *name, const auto &old, const auto &current) {
        if (std::ranges::equal(old, current)) return;
        std::cerr << "ledger_tokens=" << name << " old=";
        for (const auto id : old) std::cerr << id << ',';
        std::cerr << " new=";
        for (const auto id : current) std::cerr << id << ',';
        std::cerr << '\n';
      };
      tokens("emitted", x.emitted, y.emitted);
      tokens("accepted_inputs", x.accepted_inputs, y.accepted_inputs);
      tokens("verification_inputs", x.verification_inputs, y.verification_inputs);
      tokens("ids", x.ids(), y.ids());
      tokens("committed_ids", x.committed_ids(), y.committed_ids());
    }
    require(equal,
        "device choice changed an actual role, quota, token, prefix, frontier or RNG ledger fact");
}

inline void compare_records(const Result &a, const Result &b) {
  if (a.ids != b.ids || a.records.size() != b.records.size()) {
    // Failure-only host observations: preserve the strict oracle and do not
    // read device payloads or change the completed journeys' synchronization.
    std::cerr << "continuation_complete_difference old_ids=" << a.ids.size() << " new_ids=" << b.ids.size()
              << " old_records=" << a.records.size() << " new_records=" << b.records.size() << '\n';
    const auto ids = [](const char *name, const auto &tokens) {
      std::cerr << name << '=';
      for (const auto id : tokens) std::cerr << id << ',';
      std::cerr << '\n';
    };
    ids("continuation_ids_old", a.ids); ids("continuation_ids_new", b.ids);
    std::size_t first_id = 0;
    while (first_id < (std::min)(a.ids.size(), b.ids.size()) && a.ids[first_id] == b.ids[first_id]) ++first_id;
    std::cerr << "continuation_first_different_id ordinal=" << first_id << '\n';
    std::size_t ordinal = 0;
    while (ordinal < (std::min)(a.records.size(), b.records.size()) &&
        qwen_legacy_ledger_case::strict_equal(a.records[ordinal], b.records[ordinal])) ++ordinal;
    std::cerr << "continuation_first_different_record ordinal=" << ordinal << '\n';
    if (ordinal < (std::min)(a.records.size(), b.records.size())) {
      const auto prefix = [&](const char *name, const Result &result) {
        std::cerr << name << '=';
        for (std::size_t index = 0; index < ordinal; ++index)
          for (const auto id : result.records[index].committed_ids()) std::cerr << id << ',';
        for (const auto id : result.records[ordinal].verification_inputs) std::cerr << id << ',';
        std::cerr << '\n';
      };
      prefix("continuation_verifier_prefix_old", a); prefix("continuation_verifier_prefix_new", b);
      compare_record(a.records[ordinal], b.records[ordinal], ordinal);
    }
  }
  require(a.ids == b.ids && a.records.size() == b.records.size(),
      "device choice changed the complete IDs or ledger length");
  for (std::size_t ordinal = 0; ordinal < a.records.size(); ++ordinal)
    compare_record(a.records[ordinal], b.records[ordinal], ordinal);
}

inline void compare_legacy_records(std::string_view source, const q::SamplingConfiguration &sampling,
    std::int32_t vocabulary, const Result &a, const Result &b) {
  require(a.ids == b.ids && a.records.size() == b.records.size(),
      "legacy geometry changed the complete IDs or ledger length");
  for (std::size_t ordinal = 0; ordinal < a.records.size(); ++ordinal) {
    const auto &expected = a.records[ordinal];
    const auto &actual = b.records[ordinal];
    if (!qwen_legacy_ledger_case::strict_equal(expected, actual) &&
        qwen_legacy_ledger_case::cross_geometry_compound_semantic_equal(
            sampling.enabled, vocabulary, expected, actual)) {
      // Only the cross-recipe comparison projects the unconsumed proposal tail
      // and unpublished second diagnostic. Keep every original record intact.
      std::cout << "legacy_unselected_compound_rows ordinal=" << ordinal
                << " consumed_prefix=" << expected.accepted << " old_draft=" << expected.draft
                << " new_draft=" << actual.draft << " old_second=" << expected.target_second
                << " new_second=" << actual.target_second << " emitted_count=" << expected.emitted_count
                << " published_first=" << expected.target_first << " old_verification_inputs=";
      for (const auto id : expected.verification_inputs) std::cout << id << ',';
      std::cout << " new_verification_inputs=";
      for (const auto id : actual.verification_inputs) std::cout << id << ',';
      std::cout << " raw_records_preserved=1" << '\n';
      continue;
    }
    const auto normalized = qwen_legacy_ledger_case::legacy_rejected_second(
        source, !sampling.enabled && sampling.modifiers.empty(), a.records, b.records, ordinal);
    compare_record(a.records[ordinal], normalized ? *normalized : b.records[ordinal], ordinal);
    if (normalized) std::cout << "legacy_unconsumed_second_teacher source=" << source
        << " prefix=19,20,21,18,14,16 old=" << a.records[ordinal].target_second
        << " new=" << b.records[ordinal].target_second << " teacher_top1=24 teacher_top2=17 raw_gap="
        << qwen_legacy_ledger_case::teacher_first_score - qwen_legacy_ledger_case::teacher_second_score << '\n';
  }
}

inline void compare_request_records(const Result &a, const Result &b) {
  require(a.ids == b.ids && a.records.size() == b.records.size(),
      "recovered request changed complete IDs or ledger length");
  for (std::size_t index = 0; index < a.records.size(); ++index) {
    const auto &expected = a.records[index];
    auto actual = b.records[index];
    // The queued successor reserves a smaller quota and records its real
    // imported Source. Those are scheduling facts, not different outputs.
    require(!expected.generation || (actual.quota > 0 && actual.quota <= expected.quota),
        "recovered successor exceeded its synchronous output quota");
    actual.quota = expected.quota;
    require(expected.same_result(actual) && expected.committed == actual.committed &&
        expected.user_output == actual.user_output && expected.turn == actual.turn &&
        std::ranges::equal(expected.ids(), actual.ids()) &&
        std::ranges::equal(expected.committed_ids(), actual.committed_ids()),
        "recovered request changed a committed decision, boundary or RNG fact");
  }
}

// Require only real, already captured producer views. In particular a
// continuation's own pending/prefix/catchup outputs describe its cB decision,
// independently of the cA input selector that it consumed.
inline void require_source(q::Model &model, const q::StepState &step,
                    const std::shared_ptr<q::StepSources> &source) {
  require(source && step.execution.sources == source && source->mtp_device_choice &&
      source->mtp_full_candidate && source->submission.valid() &&
      source->primary_output.valid() && source->extra_outputs.size() == 2 &&
      source->extra_outputs[0].valid() && source->extra_outputs[1].valid() &&
      source->successor_states.size() == model.geometry.recurrent_layers() * 2 &&
      std::ranges::all_of(source->successor_states, [&](const auto &state) {
        return state.valid() && state.destination_count() == model.max_drafts() + 1;
      }) && source->pages.size() == model.draft_bank + 2 &&
      std::ranges::all_of(source->pages, [](const auto &page) { return page.valid(); }) &&
      source->boundaries.size() == 1 && source->boundaries[0].candidate == step.members[0].next.get() &&
      source->boundaries[0].epoch == step.members[0].base_epoch + 1 &&
      source->boundaries[0].length == step.members[0].next->length &&
      source->boundaries[0].draft_frontier == step.members[0].next->draft->frontier,
      "producer did not retain its own three outputs, full GDN2 and real raw-envelope PageSources");
}

inline Result journey(q::Model &model, std::span<const std::int32_t> prompt, bool continuous,
    std::optional<i::SamplingCursor> cursor = {}, std::vector<s::StateImage> *observations = nullptr,
    bool wait_before_successor = false, AttentionObservation *attention_observation = nullptr) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "continuous journey started with a retained original episode");
  const auto submitted_begin = model.submitted_draft_rows;
  const std::array prefill{prompt.size()};
  auto session = model.create_session(prefill, {}, q::GraphRole::kMtpPrefill, cursor);
  const auto proposals = model.max_drafts(), width = proposals + 1;
  // A diagnostic observation stops after the first real continuation. It has
  // no outgoing Source/third episode, so the ordinary retired visitor can
  // read original results without closing or bypassing an export's authority.
  // The selected second verifier still consumes the original pending Source.
  const auto episode_count = attention_observation ? std::size_t{2} : episodes;
  q::GenerationLedger ledger(episodes + 2, width);
  Result result;
  std::array<std::weak_ptr<q::StepState>, episodes> lifetimes;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    (void)model.abort_episode();
    if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset();
    }
  }};
  q::prepare_generation_record(ledger, *session, q::GraphRole::kMtpPrefill, prompt, false, 1, 0);
  model.prepare_evaluation(*session, prompt, nullptr, q::GraphRole::kMtpPrefill);
  model.submit_evaluations(); ++result.submitted;
  result.ids.push_back(q::finish_evaluation(model, *session, "device-continuous-prefill", {}, false,
      nullptr, nullptr, nullptr, nullptr, {}, &ledger));
  if (observations) observations->push_back(s::image(model, *session->head));

  std::shared_ptr<q::StepSources> current_source;
  if (continuous) {
    const auto pending = session->head->next_id;
    stage(ledger, session, 2 * width, role(*session), proposals);
    model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role(*session));
    lifetimes[0] = model.active_step;
    current_source = model.prepare_submission_sources(*model.active_step);
    model.submit_evaluations(); ++result.submitted;
    require_source(model, *model.active_step, current_source);
  }

  for (std::size_t index = 0; index < episode_count; ++index) {
    const bool has_child = index + 1 < episode_count;
    const auto quota = has_child ? 2 * width : width;
    const auto before = q::generation_boundary(*session->head);
    const auto epoch = session->head.epoch();
    const auto ordinal = ledger.committed_count;
    const auto pending = before.next_id;
    if (!continuous) {
      stage(ledger, session, quota, role(*session), proposals);
      model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role(*session));
      lifetimes[index] = model.active_step;
      model.submit_evaluations(); ++result.submitted;
    }
    auto *const current = model.active_step.get();
    require(current && current->submitted && !current->cancelled && session->pending == current &&
        !model.successor_step && ledger.next().quota == quota && !ledger.next().input_deferred &&
        ledger.next().before == before && ledger.next().role == role(*session),
        "current episode lost its actual input role, pending ledger or synchronized quota");
    const auto prepared = ledger.next();
    if (continuous && index) {
      require(current->device_input_choice && current->phase->family->role == q::GraphRole::kMtpContinuation &&
          current->state_slot == (index - 1) % q::execution_slot_count &&
          current->members[0].selected_input_choice && !current->draft_rows_deferred,
          "promoted continuation did not rotate its real three-slot carrier and resolve the parent choice");
    }

    q::StepState *child = nullptr;
    std::shared_ptr<q::StepSources> child_source;
    if (continuous && has_child) {
      if (wait_before_successor) q::check(current->execution.wait());
      require_source(model, *current, current_source);
      (void)ledger.reserve_successor(ordinal, width, 0);
      const std::array future{q::EvaluationRequest{session, {}}};
      const auto next_slot = index ? (current->state_slot + 1) % q::execution_slot_count
          : role(*session) == q::GraphRole::kMtpCatchupRound ? std::size_t{1} : std::size_t{0};
      child = &model.prepare_relay_evaluations(future, current_source, next_slot, q::GraphRole::kMtpContinuation);
      lifetimes[index + 1] = model.successor_step;
      const auto &carrier = child->phase->slots[child->state_slot].carrier;
      const bool native_source = carrier.queues().empty() && carrier.native_source_dependencies();
      if (native_source) {
        const auto report = q::take(child->execution.run->input_source_report());
        require(report.source_count && report.device_wait_count && !report.host_wait_count,
            "native MTP successor lost its actual pending device Source edge");
      }
      require(child->device_input_choice && child->draft_rows_deferred &&
          !child->members[0].selected_input_choice && !child->active_draft_rows &&
          child->state_slot == next_slot && !child->capture_recorded && (child->capture_reused || native_source),
          "successor was not an unobserved prepared device-choice continuation");
      result.native_prepared += native_source;
      result.reused += child->capture_reused; result.recorded += child->capture_recorded;
      // Every nonterminal continuation exports its own actual cB outputs and
      // filled physical envelope before submission, so it can in turn feed C.
      if (index + 2 < episode_count) child_source = model.prepare_submission_sources(*child);
      model.submit_evaluations(*child); ++result.submitted;
      if (child_source) require_source(model, *child, child_source);
      require(model.active_step.get() == current && model.successor_step.get() == child &&
          current->submitted && child->submitted && child->execution.has_completion() &&
          child->execution.incoming && child->execution.incoming->owner == current_source &&
          child->execution.predecessor_episode && session->pending == current &&
          session->head.epoch() == epoch && q::generation_boundary(*session->head) == before &&
          ledger.committed_count == ordinal && !ledger.records[ordinal].committed &&
          ledger.records[ordinal].same_result(prepared) && ledger.records[ordinal + 1].input_deferred,
          "successor was not truly submitted before the current host read and ordered publication");
    } else if (continuous) {
      require(!current_source && !current->execution.sources,
          "terminal quota2 episode retained an outgoing Source or attempted to reserve C");
    }

    const auto round = finish(model, session, ledger, pending, quota, index == 1 ? attention_observation : nullptr);
    append(result, round);
    if (observations) observations->push_back(s::image(model, *session->head));
    require(session->head.epoch() == epoch + 1 && ledger.committed_count == ordinal + 1 &&
        ledger.records[ordinal].committed && !ledger.records[ordinal].input_deferred,
        "one strict K1 finish did not publish exactly one original model and ledger epoch");
    if (continuous && has_child) {
      require(model.active_step.get() == child && !child->cancelled && !model.successor_step &&
          child->members[0].selected_input_choice == round.processed_target - 1 &&
          child->members[0].draft_input_rows == proposals + (round.processed_target > 1) &&
          child->active_draft_rows == proposals + (round.processed_target > 1) && !child->draft_rows_deferred,
          "promoted successor did not retain the exact selected input and real submitted draft row count");
      auto &record = ledger.bind_successor(q::generation_boundary(*session->head),
          ledger.committed_count, 0, round.emitted.back(), true);
      require(record.role == role(*session) && record.pending_input == session->head->next_id &&
          record.quota == width && !record.input_deferred,
          "successor ledger did not bind its actual Round/Catchup role and pending token");
      if (index + 2 < episode_count) {
        const auto remaining = width * (episode_count - index - 1);
        auto &renewed = ledger.renew_pending_quota(q::generation_boundary(*session->head),
            ledger.committed_count, 0, remaining, width, width);
        require(renewed.quota == 2 * width && !ledger.successor_ordinal(),
            "bound successor did not renew its original two-output reservation before reserving a child");
        require_source(model, *child, child_source);
      }
      if (round.processed_target == 1) ++result.a1_published;
      else if (round.processed_target <= width) ++result.a2_published;
      else throw std::runtime_error("strict K1 predecessor published a non-K1 prefix");
      current_source = std::move(child_source);
    } else {
      require(!model.active_step && !model.successor_step,
          "terminal or synchronous K1 episode retained submitted successor work");
    }
  }
  current_source.reset();
  require(std::ranges::all_of(lifetimes, [](const auto &lifetime) { return lifetime.expired(); }),
      "completed continuous chain retained an original producer or successor episode");

  // The same final one-output remainder uses the original Single/CatchupSingle
  // graph; it is outside the six K1 episodes and grants no uncredited child.
  const auto pending = session->head->next_id;
  const auto tail_role = session->head->draft->lag ? q::GraphRole::kMtpCatchupSingle : q::GraphRole::kMtpSingle;
  stage(ledger, session, 1, tail_role);
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, tail_role);
  model.submit_evaluations(); ++result.submitted;
  const auto tail = finish(model, session, ledger, pending, 1);
  require(tail.emitted.size() == 1 && tail.processed_target == 1 && !tail.proposal_count,
      "one-output quota remainder did not use its original single-row math");
  // A natural EOS at the final quota boundary has no outgoing successor.
  // Preserve that terminal result rather than replacing or filtering it.
  result.ids.insert(result.ids.end(), tail.emitted.begin(), tail.emitted.end());
  result.stopped = tail.stopped;
  result.final_state = s::image(model, *session->head);
  require(result.final_state->length == prompt.size() + result.ids.size() - 1,
      "complete emitted IDs do not match the final evaluated pending boundary");
  result.records.assign(ledger.records.begin(), ledger.records.begin() + ledger.committed_count);
  result.draft_rows = model.submitted_draft_rows - submitted_begin;
  std::size_t ledger_rows = 0;
  for (const auto &record : result.records) ledger_rows += record.draft_rows;
  require(result.draft_rows == ledger_rows && session->submitted_draft_rows == ledger_rows,
      "actual submitted draft rows differ from the complete ordered native ledger");
  h::retire(session);
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "continuous journey retained native work, a state version or physical page owner");
  cleanup.dismiss();
  return result;
}

enum class BoundaryCase { kStop, kCancel, kHostFailure, kRecovery };

// Exercise the actual request adapter with a submitted device successor.
// The V64 fixture has no model EOS ID, so its stop case uses the same request
// stop predicate with a real generated token; it is not a real-model EOS test.
inline Result boundary_journey(q::Model &model, std::span<const std::int32_t> prompt,
    std::int32_t stop_id, bool continuous, BoundaryCase selected) {
  const auto outputs = 2 * (model.max_drafts() + 1) + 4;
  const std::array extra{stop_id};
  const i::RequestStopPolicy policy(selected == BoundaryCase::kStop
      ? std::span<const std::int32_t>(extra) : std::span<const std::int32_t>{});
  const i::GenerationSessionSchedule schedule{"device-boundary",
      {{std::vector<std::int32_t>(prompt.begin(), prompt.end()), outputs}}};
  std::vector<q::GenerationRequest> requests;
  requests.push_back(q::install_generation_request({.schedule = schedule, .visible_turns = schedule.turns,
      .model = model, .output_capacity = outputs, .turn_capacity = 1, .record_capacity = outputs, .stop = policy}));
  auto &request = requests.front();
  request.input = q::cache::token_plan(request.prompt, model.capacity);
  i::RequestWindow keys; keys.prepare(1, 1); request.request_key = keys.enqueue(0);
  q::GenerationContinuation continuation; continuation.enabled = continuous; continuation.prepare(1);
  const auto unused_wave = [](i::RequestWave, std::span<std::size_t>) -> i::RequestWaveResult {
    throw std::logic_error("boundary admission must not execute a request wave");
  };
  q::GenerationRequestAdapter admission{model, nullptr, std::span<q::GenerationRequest>(requests),
      false, nullptr, unused_wave, {}, &continuation};
  admission.admission_preflight(request.request_key);
  admission.admit(request.request_key, {});
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    continuation.reset(); (void)model.abort_episode();
    if (request.session) {
      request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset();
    }
  }};
  bool observed_successor = false, interrupted = false;
  std::optional<s::StateImage> before;
  std::vector<std::int32_t> before_ids;
  std::size_t before_commits = 0;
  std::uint64_t before_epoch = 0;
  while (!request.progress.turn_done && !interrupted) {
    std::array calls{q::prepare_generation_call(model, request, outputs, nullptr, requests)};
    if (calls[0].generation && (selected == BoundaryCase::kCancel || selected == BoundaryCase::kHostFailure)) {
      before = s::image(model, *request.session->head);
      before_ids = request.progress.generated; before_commits = request.ledger->committed_count;
      before_epoch = request.session->head.epoch();
    }
    try {
      q::execute_generation_calls(model, calls, false, nullptr, requests, [&] {
        if (!model.successor_step || !model.successor_step->submitted) return;
        observed_successor = true;
        if (selected == BoundaryCase::kCancel) {
          q::check(request.session->stop()); interrupted = true;
        } else if (selected == BoundaryCase::kHostFailure) {
          interrupted = true; throw std::runtime_error("injected device-successor host failure");
        }
      }, &continuation);
    } catch (const std::runtime_error &failure) {
      if (!interrupted || selected != BoundaryCase::kHostFailure ||
          std::string_view(failure.what()) != "injected device-successor host failure") throw;
    }
    require(request.ledger->committed_count <= outputs, "device boundary exceeded its finite ledger");
  }
  if (selected == BoundaryCase::kStop) {
    require(request.progress.turn_done && request.progress.generated.size() >= 2 &&
        request.progress.generated.back() == stop_id, "generated stop was not published at its actual boundary");
    if (continuous) require(observed_successor && continuation.successor_discards == 1 &&
        continuation.successor_submissions == 1 && !continuation.successor_publications,
        "a stop in A failed to retire its already submitted device successor");
  } else if (selected == BoundaryCase::kRecovery) {
    require(request.progress.turn_done && request.progress.generated.size() == outputs,
        "a request after cancellation or failure did not finish its original output quota");
  } else {
    require(interrupted && observed_successor && before && !model.active_step && !model.successor_step &&
        request.session->head.epoch() == before_epoch && request.progress.generated == before_ids &&
        request.ledger->committed_count == before_commits,
        "cancelled or failed device successor published a pending head or output");
    compare_state(model, *before, s::image(model, *request.session->head));
  }
  continuation.reset();
  Result result; result.ids = request.progress.generated;
  result.submitted = continuation.successor_submissions;
  result.records.assign(request.ledger->records.begin(), request.ledger->records.begin() + request.ledger->committed_count);
  result.final_state = s::image(model, *request.session->head);
  if (selected == BoundaryCase::kCancel) {
    // A real native cancellation may remain the stopped Session's terminal
    // outcome. It does not prove retirement or permit any other native error.
    const auto check_cancelled = [](const hyle::Status &status) {
      if (status.IsCancelled() && !status.has_native_error()) return;
      q::check(status);
    };
    require(request.session->stopping, "cancel boundary lost the original stopped owner");
    check_cancelled(request.session->stop());
    const auto drained = request.session->drain(); check_cancelled(drained.status);
    require(drained.retired, "cancel boundary did not drain its original native work");
    const auto observed = request.session->try_retire(); check_cancelled(observed.status);
    require(observed.retired && !request.session->head && !request.session->pending,
        "cancel boundary did not retire the original Session head and pending owner");
    request.session.reset();
  } else h::retire(request.session);
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "device boundary left a native episode, state version or page owner alive");
  cleanup.dismiss();
  return result;
}

inline std::unique_ptr<q::Model> make_model(const std::filesystem::path &input, q::BackendKind backend) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers() && geometry.mtp_layers == 1,
      "device Model probe requires the existing local H64/V64 GA/MTP fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 1, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 2, true, true, 3);
  require(plan.state_slots == 7 && plan.hidden_slots == 8 && plan.snapshot_slots == 4 &&
      plan.in_flight_batches == 3, "continuous probe changed the original exact N1/R1/D3 pool bound");
  auto owner = std::make_unique<q::Model>(std::move(source), geometry, 16, plan, std::filesystem::path{}, false,
      q::ReplayPolicy::kPreferCapture, SIZE_MAX, q::Model::Sampling{}, 0, q::BackendSelection{.primary = backend});
  auto &model = *owner;
  model.configure_draft_policy({.max_drafts = 1});
  model.configure_device_mtp_successor(true);
  require(model.device_mtp_continuous_enabled(), "D3 did not enable the continuous device family");
  const std::array three{std::size_t{3}}, two{std::size_t{2}}, one{std::size_t{1}};
  model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 32, q::GraphRole::kMtpPrefill);
  for (const auto selected : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound, q::GraphRole::kMtpContinuation}) {
    model.prepare_family(two, false, 1, selected); model.plan_replay_uses(2, 1, 128, selected);
  }
  for (const auto selected : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
    model.prepare_family(one, false, 1, selected); model.plan_replay_uses(1, 1, 32, selected);
  }
  model.freeze_shape_plan(); model.prepare_execution(three, 1, q::GraphRole::kMtpPrefill);
  for (const auto selected : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) model.prepare_execution(two, 1, selected);
  for (const auto selected : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) model.prepare_execution(one, 1, selected);
  model.prepare_relay_execution(1, 0, q::GraphRole::kMtpContinuation);
  const auto recorded = model.prepare_mtp_decode_recordings(0);
  require(recorded.preparations == 7 && model.device_mtp_continuous_enabled() && recorded.captures > 0 &&
      !model.submitted_draft_rows && !model.draft_program_graph_calls && !model.prefix_decision_graph_calls &&
      !model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "cold recording did not leave seven dormant never-submitted Round/B0/B1/B2/B0/Catchup/B1 tuples");
  return owner;
}

inline int run(const std::filesystem::path &input, q::BackendKind backend) {
  if (backend != q::BackendKind::kCuda) {
    std::cerr << "SKIP: device-continuation requires CUDA\n";
    return 77;
  }
  auto owner = make_model(input, backend);
  auto &model = *owner;
  constexpr std::array<std::array<std::int32_t, 3>, 8> prompts{{
      {1, 2, 3}, {2, 5, 7}, {3, 1, 4}, {7, 8, 9},
      {11, 2, 17}, {19, 23, 29}, {31, 37, 41}, {43, 47, 53}}};
  std::size_t a1 = 0, a2 = 0, submits = 0, reused = 0;
  std::int32_t boundary_stop = -1;
  for (std::size_t index = 0; index < prompts.size(); ++index) {
    const auto serial = journey(model, prompts[index], false);
    const auto candidate = journey(model, prompts[index], true);
    if (index == 5) {
      require(serial.ids.size() >= 2 && serial.ids[0] != serial.ids[1], "fixture lacks a post-prefill stop");
      boundary_stop = serial.ids[1];
    }
    compare_records(serial, candidate);
    compare_state(model, *serial.final_state, *candidate.final_state);
    require(serial.draft_rows == candidate.draft_rows && serial.submitted == candidate.submitted &&
        candidate.a1_published + candidate.a2_published == episodes - 1 && !candidate.recorded,
        "continuous chain was replaced by fallback, extra math, a discarded successor or hot capture");
    a1 += candidate.a1_published; a2 += candidate.a2_published;
    submits += serial.submitted + candidate.submitted; reused += candidate.reused;
    std::cout << "prompt=" << index << " ids=";
    for (const auto id : candidate.ids) std::cout << id << ',';
    std::cout << " a1_published=" << candidate.a1_published << " a2_published=" << candidate.a2_published << '\n';
  }
  const auto serial_stop = boundary_journey(model, prompts[5], boundary_stop, false, BoundaryCase::kStop);
  const auto device_stop = boundary_journey(model, prompts[5], boundary_stop, true, BoundaryCase::kStop);
  compare_records(serial_stop, device_stop);
  compare_state(model, *serial_stop.final_state, *device_stop.final_state);
  (void)boundary_journey(model, prompts[5], boundary_stop, true, BoundaryCase::kCancel);
  const auto serial_recovery = boundary_journey(model, prompts[5], boundary_stop, false, BoundaryCase::kRecovery);
  const auto cancel_recovery = boundary_journey(model, prompts[5], boundary_stop, true, BoundaryCase::kRecovery);
  compare_request_records(serial_recovery, cancel_recovery);
  compare_state(model, *serial_recovery.final_state, *cancel_recovery.final_state);
  require(cancel_recovery.submitted > 0, "cancel recovery did not restore device successors");
  owner.reset();
  auto failure_owner = make_model(input, backend);
  (void)boundary_journey(*failure_owner, prompts[5], boundary_stop, true, BoundaryCase::kHostFailure);
  const auto failure_recovery = boundary_journey(*failure_owner, prompts[5], boundary_stop, true, BoundaryCase::kRecovery);
  compare_request_records(serial_recovery, failure_recovery);
  compare_state(*failure_owner, *serial_recovery.final_state, *failure_recovery.final_state);
  require(failure_recovery.submitted > 0, "host failure recovery did not restore device successors");
  std::cout << "PASS device_successor_boundaries generated_stop=1 cancel=1 host_failure=1 actual_model_eos=0"
      << " cancel_recovery_successors=" << cancel_recovery.submitted
      << " failure_recovery_successors=" << failure_recovery.submitted << '\n';
  std::cout << "coverage a1_published=" << a1 << " a2_published=" << a2 << " native_submissions=" << submits << '\n';
  std::cout << "PASS device_continuous_model prompts=" << prompts.size() << " journeys=" << prompts.size() * 2
      << " native_submissions=" << submits << " before_current_host_read_links=" << a1 + a2
      << " a1_successor_published=" << a1 << " a2_successor_published=" << a2 << " reused_capture_ranges=" << reused
      << " ids_and_ledger_equal=1 complete_state_oracle_equal=1 gdn_abs_budget=0.000003 controlled_acceptance=0"
         " k1_episodes=6 k1_quotas=4,4,4,4,4,2 continuation_slots=0,1,2,0,1"
         " cold_preparations=7 state_slots=7 hidden_slots=8 snapshots=4"
         " natural_acceptance_coverage_observed_only=1 quota1_tail_covered=1 manual_journey_EOS_cancel_covered=0"
         " page128_covered=0 natural_RequestLoop=0"
         " performance_claim=0 final_versions=0\n";
  return 0;
}
} // namespace qwen_device_continuation_case
