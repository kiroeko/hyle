#pragma once

// Included after the real model implementation by qwen_request_lifecycle_test.cc.
namespace qwen_sampling_policy_case {
namespace q = qwen35;
namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejected(F f) {
  bool caught = false;
  try { f(); } catch (const std::runtime_error &error) {
    caught = true; std::cerr << "sampling_policy_expected_rejection=" << error.what() << '\n';
  }
  require(caught, "expected explicit sampling-policy preflight rejection");
}
void preflight(q::Model &model) {
  for (const bool invalid_token : {false, true}) {
    auto owner = std::make_shared<std::vector<std::int32_t>>(1,
        invalid_token ? static_cast<std::int32_t>(model.vocabulary) : 0);
    q::GenerationRequest request("invalid-policy-request", *owner, {}, 1, 1);
    request.warm_inputs.push_back(owner); request.ledger = std::make_unique<q::GenerationLedger>(1, 1);
    rejected([&] { (void)q::prepare_generation_call(model, request, 1, nullptr, {}, invalid_token ? SIZE_MAX : 0); });
    require(!request.session && !request.ledger->sampling && request.ledger->sampling_owner.expired() && !request.ledger->committed_count,
        "failed policy request changed ledger identity or admitted a Session");
  }
}
void boundaries(q::Model &model, std::span<const std::int32_t> prompt, std::int32_t suffix_id) {
  auto owner = std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end());
  q::GenerationRequest request("policy-boundary", *owner, {}, 2, 2);
  request.warm_inputs.push_back(owner); request.progress.initial_sampling = {123, 9, 0};
  request.round_emitted.resize(1); request.round_emitted[0].reserve(1);
  request.ledger = std::make_unique<q::GenerationLedger>(2, 1);
  std::array calls{q::prepare_generation_call(model, request, 2, nullptr, {})};
  q::execute_generation_calls(model, calls, false);
  require(request.progress.generated.size() == 1 && request.ledger->committed_count == 1 && request.session->head->sampling.draw == 1,
      "random producer did not publish its one actual sample");
  auto original = request.session->head; const auto *canonical = original->state_identity();
  const auto producer_id = original->next_id; const auto producer_cursor = original->sampling;
  const auto empty = std::span<const std::size_t>{}; const std::array one{std::size_t{1}};
  std::int32_t greedy = -1;
  {
    auto selected = model.create_session(empty, original, q::GraphRole::kText, q::cache::SamplingCursor{}, true, q::TextCompletion::kEmitNext, 0);
    greedy = selected->head->next_id;
    require(selected->head->state_identity() == canonical && selected->head->carry_hidden.valid() && selected->head->sampling == q::cache::SamplingCursor{},
        "new greedy full hit copied state or inherited random cursor");
  }
  {
    const q::cache::SamplingCursor cursor{456, 10, 0};
    auto selected = model.create_session(empty, original, q::GraphRole::kText, cursor, true, q::TextCompletion::kEmitNext, 1);
    require(selected->head->state_identity() == canonical && selected->head->next_id == greedy && selected->head->sampling == cursor.advanced(1),
        "new random-k1 full hit failed to use selected policy or cursor");
  }
  {
    auto selected = model.create_session(one, original, q::GraphRole::kText, q::cache::SamplingCursor{}, false, q::TextCompletion::kEmitNext, 0);
    require(selected->head->sampling == q::cache::SamplingCursor{} && selected->head->next_id == -1,
        "new greedy partial hit inherited random producer pending or cursor");
    const std::array suffix{suffix_id}; model.prepare_evaluation(*selected, suffix); model.submit_evaluation(*selected);
    (void)q::finish_evaluation(model, *selected, "greedy-policy-suffix", {}, false);
    require(selected->length() == prompt.size() + 1 && selected->head->sampling == q::cache::SamplingCursor{},
        "greedy partial suffix sampled or lost context");
  }
  const auto historical = static_cast<std::int32_t>((static_cast<std::int64_t>(producer_id) + 1) % model.vocabulary);
  model.restore_pending_boundary(*request.session, producer_cursor, historical);
  {
    auto same = model.create_session(empty, request.session->head, q::GraphRole::kText, {}, false, q::TextCompletion::kEmitNext, 2);
    require(same->head->next_id == historical && same->head->sampling == producer_cursor && same->head->state_identity() == canonical,
        "same-request restore discarded historical pending or cursor");
  }
  require(original->next_id == producer_id && original->sampling == producer_cursor, "policy forks mutated an old reader");
  calls = {}; q::retire_generation_request(request);
  {
    auto rebuilt = q::rebuild_generation_session(model, *request.ledger);
    require(&rebuilt->sampling_configuration() == &model.sampling_configuration(2) &&
        rebuilt->head->next_id == producer_id && rebuilt->head->sampling == producer_cursor,
        "ordinary ledger rebuild lost its selected policy or historical boundary");
  }
  original.reset();
  require(model.snapshot_versions.live == 0 && !model.active_step && !model.carrier_borrowed, "policy boundaries leaked live state");
}
}
inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  auto prompt = q::read_ids(input / "prompt.ids", geometry.vocabulary, static_cast<std::size_t>(geometry.context_limit));
  require(prompt.size() >= 2 && geometry.vocabulary > 1, "sampling policy case requires two prompt tokens and a nontrivial vocabulary");
  const auto suffix = prompt.size() > 2 ? prompt[2] : prompt[0]; prompt.resize(2);
  constexpr std::size_t capacity = 16;
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, cache_rows, capacity, 2, 5,
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 2, 1, false, true);
  const std::array additional{q::SamplingConfiguration{true, {1.F, 1, 1.F}},
      q::SamplingConfiguration{true, {.7F, (std::min)(std::size_t{8}, static_cast<std::size_t>(geometry.vocabulary)), .9F}}};
  q::Model model(input / "weights.tsv", geometry, capacity, plan, {}, false, q::ReplayPolicy::kCommands, 1073741824,
      {.enabled = false, .policy = {}, .seed = 123}, 0, {.primary = q::BackendKind::kCuda}, additional);
  const std::array<std::size_t, 6> policy{0, 0, 2, 2, 1, 1};
  std::vector<hyle::inference::GenerationSessionSchedule> schedules;
  std::vector<q::cache::TokenPlanRef> tokens;
  for (std::size_t index = 0; index < policy.size(); ++index) {
    schedules.push_back({"policy" + std::to_string(index), {{prompt, 3}, {{suffix}, 2}}});
    tokens.push_back(q::cache::token_plan(prompt, capacity));
  }
  const auto start = q::Clock::now();
  require(q::run_generation_sessions<q::GenerationMode::kText>(model, schedules, {}, false, start, start, true, tokens, policy) == 0,
      "mixed-policy natural requests failed");
  require(model.snapshot_versions.live == 0 && !model.active_step && !model.carrier_borrowed,
      "mixed-policy natural requests leaked a head or carrier");
  require(!model.cache_plan.draft_cache && !model.physical_mtp_roles && model.draft_programs.empty() && model.prefix_sampling_programs.empty(),
      "ordinary sampling policies prepared NextN resources");
  for (const auto &[key, family] : model.families)
    require(family->role == q::GraphRole::kText && family->sampling == &model.sampling_configuration(key.sampling_index),
        "family sampling binding differs from its cold key");
  preflight(model); boundaries(model, prompt, suffix);
  std::cout << "{\"status\":\"PASS\",\"case\":\"sampling-policies\",\"ordinary\":true,\"same_model\":true,"
      "\"configurations\":3,\"requests\":6,\"turns\":12,\"N\":2,\"R\":2,\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_sampling_policy_case
