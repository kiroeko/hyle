#pragma once

// Actual cached Model consumers. The independent host oracle counts only the
// ledger's committed tokens plus this call's real input, never emitted pending
// tokens or padded execution rows. No trained-model quality or speed claim.
namespace qwen_history_logits_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace m = qwen_mtp_logits_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
namespace t = qwen_mtp_stop_case;
namespace u = qwen_mtp_sampling_case;
using h::require;

inline s::StateImage snapshot(q::Model &model, const q::cache::State &state) {
  if (model.cache_plan.draft_cache) return s::image(model, state);
  s::StateImage result{state.length, 0, 0, state.next_id, -1, state.sampling, {}, {}};
  for (std::size_t layer = 0; layer != model.geometry.layers.size(); ++layer) if (model.geometry.is_gdn(layer)) {
    result.payloads.push_back(s::read(model, state.state[layer].history));
    result.payloads.push_back(s::read(model, state.state[layer].recurrent));
  }
  for (std::size_t layer = 0; layer != state.paged.frames.size() / 2; ++layer) {
    result.target.push_back(s::frame(model, state.paged.frames[2 * layer], state.length));
    result.target.push_back(s::frame(model, state.paged.frames[2 * layer + 1], state.length));
  }
  result.payloads.push_back(s::read(model, state.carry_hidden));
  result.payloads.push_back(s::read(model, state.prior_hidden));
  return result;
}

inline std::vector<std::int32_t> counts(q::Model &model, std::span<const std::int32_t> consumed,
    std::span<const std::int32_t> input = {}) {
  std::vector<std::int32_t> result(static_cast<std::size_t>(model.vocabulary));
  for (const auto ids : {consumed, input}) for (const auto id : ids) {
    require(id >= 0 && static_cast<std::size_t>(id) < result.size(), "history oracle received an invalid token");
    ++result[static_cast<std::size_t>(id)];
  }
  return result;
}
inline std::vector<std::int32_t> consumed(const q::GenerationLedger &ledger) {
  std::vector<std::int32_t> result;
  for (std::size_t at = 0; at != ledger.committed_count; ++at) {
    const auto &record = ledger.records.at(at);
    require(record.committed, "history oracle saw a reserved record as committed");
    const auto ids = record.committed_ids(); result.insert(result.end(), ids.begin(), ids.end());
  }
  return result;
}
inline void same_counts(q::Model &model, const q::LogitsInput &value, std::span<const std::int32_t> expected) {
  require(value.lease.valid() && value.description == q::dense(q::Type::kInt32, {1, model.vocabulary}),
      "history C0 lost its exact readonly I32 descriptor");
  std::vector<std::int32_t> actual(expected.size());
  q::check(model.execution.read(q::take(value.lease.view()), std::as_writable_bytes(std::span(actual))));
  require(std::ranges::equal(actual, expected), "history C0 counted pending, padding, rejected or another member's tokens");
}
inline m::Payload with_counts(q::Model &model, const m::Payload &base, std::span<const std::int32_t> values) {
  auto result = base; result.inputs.counts = m::ready<std::int32_t>(model, q::Type::kInt32, values); return result;
}

inline void raw_rounds(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage,
    const m::Payload &base, std::vector<p::StepReport> &checks) {
  const std::array planned{std::size_t{3}};
  const std::array<std::int32_t, 3> prompt{1, 2, 1};
  for (std::size_t index = 0; index != 2; ++index) {
    const auto cursor = index ? q::cache::SamplingCursor{313, 17, 0} : q::cache::SamplingCursor{};
    auto control = model.create_session(planned, {}, q::GraphRole::kMtpPrefill, cursor, false, q::TextCompletion::kEmitNext, index);
    auto candidate = model.create_session(planned, {}, q::GraphRole::kMtpPrefill, cursor, false, q::TextCompletion::kEmitNext, index);
    auto history = std::vector<std::int32_t>(prompt.begin(), prompt.end());
    auto first = with_counts(model, base, counts(model, history));
    require(m::raw(model, control, prompt, q::GraphRole::kMtpPrefill, first.inputs) ==
        m::raw(model, candidate, prompt, q::GraphRole::kMtpPrefill, first.inputs), "history prefill changed identical policy/cursor output");
    auto old = candidate->checkpoint(); const auto old_bytes = snapshot(model, *old);
    for (const bool recursive : {false, true}) {
      const auto pending = control->head->next_id;
      require(pending == candidate->head->next_id, "history controls reached different pending inputs");
      const auto c0 = counts(model, history, std::span(&pending, 1)); auto value = with_counts(model, base, c0);
      const auto expected = m::round(model, control, recursive, value), actual = m::round(model, candidate, recursive, value);
      h::same_round(expected, actual); same_counts(model, value.inputs.counts, c0);
      u::same(model, p::capture(model, *control, storage), p::capture(model, *candidate, storage), checks);
      require(expected.processed_target >= 1 && expected.processed_target <= expected.emitted.size(),
          "history oracle cannot identify the actually consumed accepted prefix");
      history.push_back(pending);
      history.insert(history.end(), expected.emitted.begin(), expected.emitted.begin() + expected.processed_target - 1);
      require(history.size() == control->length(), "history raw oracle length differs from canonical accepted state");
    }
    require(snapshot(model, *old) == old_bytes, "history work mutated an old complete checkpoint");
    old.reset(); h::retire(control); h::retire(candidate);
  }
}

struct Journey final {
  std::array<std::vector<h::ImageBytes>, 2> images;
  std::array<std::vector<std::int32_t>, 2> generated;
};
inline Journey requests(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage,
    const m::Payload &base, bool manual, std::vector<p::StepReport> &checks) {
  const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 1}, {2, 1, 2}}};
  std::vector<q::GenerationRequest> owners; owners.reserve(2);
  i::RequestWindow window; window.prepare(2, 2);
  for (std::size_t member = 0; member != 2; ++member) {
    owners.emplace_back("history-member" + std::to_string(member), prompts[member], std::filesystem::path{}, 7, 1);
    auto &owner = owners.back(); t::prepare_request(owner, 16); owner.progress.initial_sampling = {171, 301 + member, 0};
    owner.logits = base.inputs; owner.request_key = window.enqueue(member); window.admitted(owner.request_key);
  }
  Journey result;
  const auto prepare = [&](std::size_t member, std::size_t quantum) {
    auto &owner = owners[member]; return q::prepare_generation_call(model, owner, 7, nullptr, owners, 3, quantum);
  };
  const auto c0 = [&](q::PreparedGenerationCall &call) {
    const auto prior = consumed(*call.request->ledger);
    return counts(model, prior, call.generation ? std::span<const std::int32_t>(&call.pending, 1) : call.prompt);
  };
  for (std::size_t chunk = 0; chunk != 3; ++chunk) for (std::size_t member = 0; member != 2; ++member) {
    auto &owner = owners[member]; std::array calls{prepare(member, 1)}; const auto expected = c0(calls[0]);
    const bool emit = chunk == 2;
    require(calls[0].completion == (emit ? q::TextCompletion::kEmitNext : q::TextCompletion::kStateOnly),
        "history chunk changed its real completion");
    if (manual && emit) calls[0].logits.counts = m::ready<std::int32_t>(model, q::Type::kInt32, expected);
    q::execute_generation_calls(model, calls, false, nullptr, owners, [&] {
      if (emit) same_counts(model, calls[0].logits.counts, expected);
      else {
        require(!model.active_step->members[0].logits.counts.lease.valid(), "StateOnly retained a history input or draw");
        qwen_mtp_chunked_prefill_case::state_only_ports(*model.active_step->phase);
      }
    });
    require(!calls[0].logits.counts.lease.valid(), "completed natural call retained its C0 pool block");
    result.images[member].push_back(p::capture(model, *owner.session, storage));
    require(owner.session->length() == chunk + 1 && owner.progress.generated.size() == std::size_t(emit),
        "history chunk consumed padding or emitted before complete input");
    if (!chunk) {
      auto image = model.export_session(*owner.session, storage); const auto before = h::read_image(*image);
      const auto key = owner.request_key; owner.last_idle = window.pause(key); h::retire(owner.session);
      owner.session = model.restore_session(image); image.reset();
      u::same(model, before, p::capture(model, *owner.session, storage), checks); window.wake(key); window.admitted(key);
    }
  }
  std::array old{owners[0].session->checkpoint(), owners[1].session->checkpoint()};
  const std::array old_bytes{snapshot(model, *old[0]), snapshot(model, *old[1])};
  std::array calls{prepare(0, SIZE_MAX), prepare(1, SIZE_MAX)};
  const std::array expected{c0(calls[0]), c0(calls[1])};
  for (std::size_t member = 0; member != 2; ++member)
    if (manual) calls[member].logits.counts = m::ready<std::int32_t>(model, q::Type::kInt32, expected[member]);
  bool submitted = false;
  q::execute_generation_calls(model, calls, false, nullptr, owners, [&] {
    require(model.active_step->members.size() == 2 && model.active_step->input_tokens == (model.cache_plan.draft_cache ? 3 : 1),
        "history B2 was split instead of one original shared submission");
    for (std::size_t member = 0; member != 2; ++member) {
      same_counts(model, calls[member].logits.counts, expected[member]);
      require(calls[member].draft_log_probabilities.size() == (model.cache_plan.draft_cache ? 2 : 0),
          "history B2 changed its actual conditional q row count");
      m::q_rows(model, calls[member].draft_log_probabilities, base);
    }
    submitted = true;
  });
  require(submitted, "history B2 did not reach native submission");
  for (std::size_t member = 0; member != 2; ++member) {
    auto &owner = owners[member];
    require(!calls[member].logits.counts.lease.valid() && calls[member].draft_log_probabilities.empty() &&
        snapshot(model, *old[member]) == old_bytes[member], "history B2 retained C0/q or mutated a prior checkpoint");
    result.images[member].push_back(p::capture(model, *owner.session, storage)); result.generated[member] = owner.progress.generated;
    m::allowed(result.generated[member], base);
    window.retire_active(owner.request_key); h::retire(owner.session); window.release(owner.request_key); old[member].reset();
  }
  require(!model.snapshot_versions.live && !storage->live_images(), "history journey retained a state or HostImage owner");
  t::idle(model); return result;
}

inline void prefix(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage,
    const m::Payload &base, std::vector<p::StepReport> &checks) {
  const std::array<std::int32_t, 3> prompt{1, 2, 1};
  std::vector<q::GenerationRequest> owners; owners.reserve(2);
  i::RequestWindow window; window.prepare(2, 2); q::cache::PrefixCache directory(1);
  for (std::size_t member = 0; member != 2; ++member) {
    owners.emplace_back("history-prefix" + std::to_string(member), prompt, std::filesystem::path{}, 1, 1 - member);
    auto &owner = owners.back(); t::prepare_request(owner, 16); owner.logits = base.inputs;
    owner.progress.initial_sampling = member ? q::cache::SamplingCursor{} : q::cache::SamplingCursor{811, 9, 0};
    owner.request_key = window.enqueue(member); window.admitted(owner.request_key);
  }
  std::array publish{q::prepare_generation_call(model, owners[0], 1, &directory, owners, 1)};
  q::execute_generation_calls(model, publish, false, &directory, owners); q::publish_generation_prompt(model, publish[0], &directory);
  require(directory.size() == 1, "history prefix publisher did not enter the original cache");
  auto checkpoint = owners[0].session->checkpoint(); const auto bytes = snapshot(model, *checkpoint);
  std::array hit{q::prepare_generation_call(model, owners[1], 1, &directory, owners, 1)};
  require(hit[0].full_hit, "history prefix did not use the existing complete consumed key");
  q::execute_generation_calls(model, hit, false, &directory, owners);
  const auto explicit_counts = with_counts(model, base, counts(model, prompt));
  auto control = model.create_session({}, checkpoint, model.cache_plan.draft_cache ? q::GraphRole::kMtpPrefill : q::GraphRole::kText, {}, true,
      q::TextCompletion::kEmitNext, 0, explicit_counts.inputs);
  u::same(model, p::capture(model, *control, storage), p::capture(model, *owners[1].session, storage), checks);
  require(owners[1].prefix_sampling_calls == 1 && !hit[0].logits.counts.lease.valid() &&
      snapshot(model, *checkpoint) == bytes && owners[1].ledger->committed_count == 1,
      "history full hit reused publisher pending, retained C0 or changed the publisher");
  h::retire(control);
  for (auto &owner : owners) { window.retire_active(owner.request_key); h::retire(owner.session); window.release(owner.request_key); }
  checkpoint.reset(); directory.evict_one(); t::idle(model);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind, bool draft = true) {
  auto source = q::prepare_weight_source(input / "weights.tsv", draft); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "history requires the cached GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 12, 4,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, draft ? 3 : 1, draft, true);
  const q::sampling::HistoryPolicy history{1.75F, .5F, .125F};
  const q::sampling::LogitsPolicy modifiers{.bias = true, .allowed = true, .history = history};
  const std::array additional{q::SamplingConfiguration{true, {.8F, 0, 1.F}, modifiers}};
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = false, .policy = {}, .seed = 123, .modifiers = modifiers}, 0, {.primary = kind}, additional);
  if (draft) {
    model.configure_draft_policy({.max_drafts = 2}); m::cold(model);
    inference_history_math_case::finite_math(model, kind);
  } else {
    const std::array shapes{std::size_t{1}, std::size_t{3}};
    const std::array one{std::size_t{1}};
    for (std::size_t index = 0; index != 2; ++index) {
      model.prepare_family(shapes, false, 1, q::GraphRole::kText, false, q::TextCompletion::kEmitNext, index);
      for (const auto rows : shapes) model.plan_replay_uses(rows, 1, 12, q::GraphRole::kText, q::TextCompletion::kEmitNext, index);
    }
    model.prepare_family(one, false, 1, q::GraphRole::kText, false, q::TextCompletion::kStateOnly, 1);
    model.plan_replay_uses(1, 1, 12, q::GraphRole::kText, q::TextCompletion::kStateOnly, 1);
    model.prepare_family(one, false, 2, q::GraphRole::kText, false, q::TextCompletion::kEmitNext, 1);
    model.plan_replay_uses(1, 2, 4, q::GraphRole::kText, q::TextCompletion::kEmitNext, 1);
  }
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const auto base = m::payload(model, 0); std::vector<p::StepReport> checks;
  if (draft) raw_rounds(model, storage, base, checks);
  const auto natural = requests(model, storage, base, false, checks), manual = requests(model, storage, base, true, checks);
  for (std::size_t member = 0; member != 2; ++member) {
    require(natural.generated[member] == manual.generated[member] && natural.images[member].size() == manual.images[member].size(),
        "derived history changed the manual conditional-token oracle");
    for (std::size_t step = 0; step != natural.images[member].size(); ++step)
      u::same(model, manual.images[member][step], natural.images[member][step], checks);
  }
  prefix(model, storage, base, checks);
  require(!model.snapshot_versions.live && !storage->live_images() && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(), "history consumer retained state/page/native owners");
  t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"history-logits\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"configurations\":2,\"draft\":" << (draft ? "true" : "false")
      << ",\"K1_K2_greedy_random\":" << (draft ? "true" : "false") << ",\"derived_vs_manual_B2\":true,"
      "\"StateOnly_no_counts_or_draw\":true,\"same_key_HostImage\":true,\"full_hit_seed_before_frontier\":true,"
      "\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at != checks.size(); ++at) { if (at) std::cout << ','; p::write_step(checks[at]); }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_history_logits_case
