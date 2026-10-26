#pragma once

// Included after the real Model. A finite CPU owner checks the request ledger
// boundary; raw Session APIs remain independently usable without that owner.
namespace qwen_request_ledger_case {
namespace q = qwen35;
namespace i = hyle::inference;
inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> inline void missing_ledger(F &&action) {
  bool refused = false;
  try { action(); } catch (const std::runtime_error &error) {
    refused = std::string_view(error.what()) == "generation request requires its cold-prepared commit ledger";
  }
  require(refused, "missing request ledger was not rejected before admission");
}
inline void retire(q::GenerationRequest &request) {
  const auto retired = q::retire_generation_request(request);
  if (retired.retired) request.session.reset();
  q::check(retired.status);
  require(retired.retired, "ledger case Session did not actually retire");
}
// The only strong owner of B is its unpinned directory entry. In particular,
// no Hit or temporary SnapshotRef may hide the old admission/eviction defect.
inline void duplicate_session_before_admission(q::Model &model) {
  require(model.cache_plan.resident_sessions == 2 && !model.snapshot_versions.live,
      "duplicate admission case needs an empty N2 version inventory");
  q::cache::PrefixCache prefix(1);
  std::weak_ptr<const q::cache::State> b_boundary;
  const std::array<std::int32_t, 2> b_prompt{3, 4};
  {
    const std::array seed_rows{std::size_t{2}};
    auto seed = model.create_session(seed_rows, {}, q::GraphRole::kText,
        q::cache::SamplingCursor{123, 11, 0});
    const auto next = q::evaluate(model, *seed, b_prompt, "duplicate-prefix-B", {}, false);
    const std::array chunks{q::cache::ConsumedChunk{b_prompt, next, false}};
    const auto checkpoint = q::cache::consumed_checkpoint(chunks, *seed->head, model.capacity);
    require(prefix.publish(checkpoint.input, seed->checkpoint(), checkpoint.replay),
        "duplicate admission case did not publish the real B boundary");
    b_boundary = seed->checkpoint();
    q::check(seed->stop());
    const auto retired = seed->try_retire(); q::check(retired.status);
    require(retired.retired, "duplicate admission B seed did not retire");
    seed.reset();
  }
  require(prefix.size() == 1 && !prefix.evictions() && b_boundary.use_count() == 1 &&
      model.snapshot_versions.live == 1 && !model.active_step && !model.carrier_borrowed,
      "B must be retained only by its evictable prefix directory entry");

  const std::array<std::int32_t, 2> a_prompt{1, 2};
  std::vector<q::GenerationRequest> requests; requests.reserve(1);
  requests.emplace_back("duplicate-session-A", a_prompt, std::filesystem::path{}, 1);
  auto &a = requests[0];
  a.input = q::cache::token_plan(a_prompt, model.capacity);
  a.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(a_prompt.begin(), a_prompt.end()));
  a.prompt = *a.warm_inputs[0];
  a.progress.initial_sampling = {123, 10, 0};
  a.ledger = std::make_unique<q::GenerationLedger>(1, 1);
  auto call = q::prepare_generation_call(model, a, 1, &prefix, requests);
  require(!call.generation && !call.full_hit && a.session && a.session->length() == 0 &&
      a.progress.turn_started() && !a.progress.prefilled && !a.progress.turn_done &&
      a.progress.generated.empty() && model.snapshot_versions.live == 2 &&
      prefix.size() == 1 && !prefix.evictions() && b_boundary.use_count() == 1,
      "A must be prepared beside the independent B-only prefix before rejection");
  const auto *head = a.session->head.get();
  const auto boundary = q::generation_boundary(*head);
  const auto ledger_record = a.ledger->records[0];
  const auto generated = a.progress.generated;
  const auto input_turn = a.input_turn, input_consumed = a.turn_input_consumed;
  const auto turn_index = a.progress.turn_index, output_begin = a.progress.turn_output_begin;
  const auto prefix_size = prefix.size(), evictions = prefix.evictions();
  const auto live = model.snapshot_versions.live;
  // Two actual calls repeat the very same mutable Session, rather than merely
  // sharing a canonical immutable prefix. The old query counts demand=2 and
  // replaceable=1, evicts B at live=2, and only later rejects the duplicate.
  std::array duplicate{call, call};
  std::string rejection;
  try { q::execute_generation_calls(model, duplicate, false, &prefix, requests); }
  catch (const std::runtime_error &error) { rejection = error.what(); }
  std::cerr << "duplicate_session_rejection=\"" << rejection << "\" prefix_before=" << prefix_size
      << " prefix_after=" << prefix.size() << " evictions_before=" << evictions
      << " evictions_after=" << prefix.evictions() << " live_before=" << live
      << " live_after=" << model.snapshot_versions.live << " B_strong_owners=" << b_boundary.use_count() << '\n';
  const auto &record = a.ledger->records[0];
  require(prefix.size() == prefix_size && prefix.evictions() == evictions &&
      b_boundary.use_count() == 1 && model.snapshot_versions.live == live &&
      a.session->head.get() == head && q::generation_boundary(*a.session->head) == boundary &&
      !a.session->pending && !a.session->failed && !a.session->stopping &&
      !a.ledger->committed_count && record.same_result(ledger_record) &&
      record.committed == ledger_record.committed && record.user_output == ledger_record.user_output &&
      record.imported == ledger_record.imported && record.turn == ledger_record.turn &&
      record.prompt_ids.data() == ledger_record.prompt_ids.data() &&
      record.prompt_ids.size() == ledger_record.prompt_ids.size() && record.input_owner == ledger_record.input_owner &&
      a.progress.generated == generated && !a.progress.observed_commits() &&
      a.progress.turn_started() && !a.progress.prefilled && !a.progress.turn_done &&
      a.progress.turn_index == turn_index && a.progress.turn_output_begin == output_begin &&
      a.progress.remaining_outputs() == 1 && a.input_turn == input_turn && a.turn_input_consumed == input_consumed &&
      !model.active_step && !model.carrier_borrowed,
      "duplicate rejection evicted B or changed A head/RNG/ledger/turn outputs");
  require(rejection == "generation batch requires distinct Session owners",
      "duplicate Session did not report its real early input error");

  // Retry the original single prepared call without repairing the request,
  // cache or ledger. A correct rejection left all their original facts intact.
  std::array retry{std::move(call)};
  q::execute_generation_calls(model, retry, false, &prefix, requests);
  require(a.progress.request_done() && a.progress.generated.size() == 1 &&
      a.progress.observed_commits() == 1 && a.ledger->committed_count == 1 &&
      a.ledger->records[0].committed && a.ledger->records[0].requests == 1 &&
      a.ledger->records[0].before == boundary && a.ledger->records[0].after.sampling.draw == 1 &&
      q::generation_boundary(*a.session->head) == a.ledger->records[0].after &&
      a.session->length() == 2 && a.progress.generated[0] == a.session->head->next_id &&
      prefix.size() == prefix_size && prefix.evictions() == evictions &&
      b_boundary.use_count() == 1 && model.snapshot_versions.live == 2,
      "single A retry did not commit exactly once while preserving the B-only prefix");
  retire(a);
  require(model.snapshot_versions.live == 1 && prefix.evict_one() && prefix.size() == 0 &&
      b_boundary.expired() && !model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
      "duplicate admission case did not retire A and release its final B version");
}
inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A", "ledger case requires cached A/H64/FFN128/V64");
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, cache_rows, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = i::BackendKind::kCpu});
  const std::array rows{std::size_t{1}, std::size_t{2}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(rows, false, width);
    for (const auto count : rows) model.plan_replay_uses(count, width, 8);
  }
  for (const auto width : {std::size_t{1}, std::size_t{2}}) model.prepare_execution(rows, width);
  const std::array<std::int32_t, 2> prompt{1, 2};
  q::GenerationRequest missing("missing-ledger", prompt, {}, 2);
  missing_ledger([&] { (void)q::prepare_generation_call(model, missing, 2, nullptr, {}); });
  require(!missing.session && !missing.progress.turn_started() && missing.progress.generated.empty() &&
      !missing.progress.observed_commits() && missing.input_turn == SIZE_MAX && !missing.turn_input_consumed &&
      !model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "missing ledger preparation changed a request or Model admission");

  std::vector<q::GenerationRequest> requests;
  requests.reserve(2);
  const auto impossible_trace = input / "weights.tsv";
  require(std::filesystem::is_regular_file(impossible_trace), "late IO needs the existing manifest file");
  requests.emplace_back("trace-failure-A", prompt, impossible_trace, 2);
  requests.emplace_back("survivor-B", prompt, std::filesystem::path{}, 2);
  for (auto &request : requests) {
    auto owner = std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end());
    request.prompt = *owner;
    request.warm_inputs.push_back(std::move(owner));
    request.progress.initial_sampling = {123, 10, 0};
    request.ledger = std::make_unique<q::GenerationLedger>(2, 1);
    request.round_emitted.resize(1); request.round_emitted[0].reserve(1);
  }
  auto &a = requests[0]; auto &b = requests[1];
  const auto prepare = [&](q::GenerationRequest &request) {
    return q::prepare_generation_call(model, request, request.progress.output_limit(), nullptr, requests);
  };
  std::array calls{prepare(a), prepare(b)};
  const auto *a_head = a.session->head.get(); const auto *b_head = b.session->head.get();
  const auto a_base = q::generation_boundary(*a_head), b_base = q::generation_boundary(*b_head);
  q::PreparedGenerationCall missing_call; missing_call.request = &missing;
  std::array invalid{std::move(calls[0]), std::move(missing_call)};
  missing_ledger([&] { q::execute_generation_calls(model, invalid, false, nullptr, requests); });
  require(a.session->head.get() == a_head && b.session->head.get() == b_head &&
      q::generation_boundary(*a.session->head) == a_base && q::generation_boundary(*b.session->head) == b_base &&
      !a.ledger->committed_count && !b.ledger->committed_count && a.ledger->records[0].prompt_ids.empty() &&
      !a.ledger->records[0].input_owner && a.progress.generated.empty() && b.progress.generated.empty() &&
      !model.active_step && !model.carrier_borrowed,
      "later missing-ledger member mutated the first prepared record or submitted work");
  calls[0] = std::move(invalid[0]);

  // Original real Graph publication happens for the whole R2 batch before A's
  // entry observer tries to create a directory at an existing regular file.
  bool late_io = false;
  try { q::execute_generation_calls(model, calls, false, nullptr, requests); }
  catch (const std::filesystem::filesystem_error &) { late_io = true; }
  require(late_io && !model.active_step && !model.carrier_borrowed,
      "expected post-publication entry IO error did not return the actual carrier");
  for (std::size_t member = 0; member < requests.size(); ++member) {
    auto &request = requests[member]; const auto &record = request.ledger->records[0];
    require(record.committed && request.ledger->committed_count == 1 && record.requests == 2 &&
        record.member_index == member && request.progress.observed_commits() == 1 &&
        request.progress.generated.size() == 1 && request.progress.prefilled && !request.progress.turn_done &&
        request.progress.generated[0] == request.session->head->next_id &&
        request.fresh_next_id == record.after.next_id && q::generation_boundary(*request.session->head) == record.after && record.after.sampling.draw == 1 &&
        request.turn_input_consumed == 2 && !request.session->pending && !request.session->failed &&
        !request.session->stopping && request.session->retirement_status.ok(),
        "late IO lost a member's committed output, state, ledger or request observation");
    q::observe_generation_commits(request);
    require(request.progress.observed_commits() == 1 && request.progress.generated.size() == 1,
        "repeated post-IO observation duplicated output");
  }
  const auto a_published = q::generation_boundary(*a.session->head);
  const auto b_published = q::generation_boundary(*b.session->head);
  a.trace.clear(); // Only the failed diagnostic destination changes.
  std::array continuation{prepare(a), prepare(b)};
  q::execute_generation_calls(model, continuation, false, nullptr, requests);
  for (std::size_t member = 0; member < requests.size(); ++member) {
    auto &request = requests[member]; const auto &record = request.ledger->records[1];
    const auto &prior = member ? b_published : a_published;
    require(request.progress.request_done() && request.progress.generated.size() == 2 &&
        request.ledger->committed_count == 2 && request.progress.observed_commits() == 2 &&
        record.committed && record.before == prior && record.pending_input == prior.next_id &&
        record.after.sampling.draw == 2 && q::generation_boundary(*request.session->head) == record.after &&
        request.session->length() == 3 && request.progress.generated.back() == request.session->head->next_id &&
        request.fresh_next_id == record.after.next_id,
        "post-IO continuation repeated prompt, skipped pending or resampled an output");
  }
  const auto a_ids = a.progress.generated, b_ids = b.progress.generated;
  retire(a); retire(b);
  duplicate_session_before_admission(model);

  // Request-ledger enforcement does not change the lower raw Session API.
  const std::array raw_rows{std::size_t{2}};
  auto raw = model.create_session(raw_rows, {}, q::GraphRole::kText,
      q::cache::SamplingCursor{123, 10, 0});
  const auto raw_id = q::evaluate(model, *raw, prompt, "raw-without-request-ledger", {}, false);
  require(raw->length() == 2 && raw->head->next_id == raw_id && raw->head->sampling.draw == 1,
      "raw Session evaluate unexpectedly required a request ledger");
  raw->stop().IgnoreError(); const auto raw_retired = raw->try_retire(); q::check(raw_retired.status);
  require(raw_retired.retired, "raw Session did not retire"); raw.reset();
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      model.draft_programs.empty() && model.prefix_sampling_programs.empty(), "ledger case EOF retained state or used MTP");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\",\"ordinary\":true,"
      "\"missing_prepare_atomic\":true,\"missing_later_execute_member_atomic\":true,"
      "\"duplicate_Session_before_prefix_eviction\":true,\"same_request_retry\":true,"
      "\"real_R2_late_trace_error\":true,\"both_ledgers_observed\":true,\"duplicate_observation_noop\":true,"
      "\"both_pending_continuations\":true,\"raw_Session_without_request_ledger\":true,\"A_ids\":["
      << a_ids[0] << ',' << a_ids[1] << "],\"B_ids\":[" << b_ids[0] << ',' << b_ids[1]
      << "],\"final_versions\":0,\"GPU\":false,\"HTTP\":0}\n";
  return 0;
}
} // namespace qwen_request_ledger_case
