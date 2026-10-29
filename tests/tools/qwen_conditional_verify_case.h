#pragma once

// Actual target Verify logits and complete target/draft publication. The
// cached GA model is untrained; no selector callback forces its acceptance.
namespace qwen_conditional_verify_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace k = qwen_conditional_logits_case;
namespace m = qwen_mtp_logits_case;
namespace s = qwen_mtp_batch_state_case;
namespace p = qwen_packed_prefill_case;
namespace r = qwen_mtp_ragged_prefill_case;
namespace t = qwen_mtp_stop_case;
namespace g = qwen_mtp_group_return_case;
namespace e = qwen_logits_preflight_case;
using h::require;

inline void owner_contract() {
  const std::array<q::LogitsInputs,3> rows{};
  i::ConditionalLogitsInputs original{std::span<const q::LogitsInputs>(rows)};
  require(original.rows().size() == 3 && original.rows().data() != rows.data(),
      "conditional owner borrowed mutable caller metadata");
  const auto prefix = q::take(original.prefix(2));
  require(prefix.rows().size() == 2 && prefix.rows().data() == original.rows().data(),
      "conditional prefix copied storage instead of sharing its immutable owner");
  require(!original.prefix(4) && original.prefix(4).error().IsInvalidArgument(),
      "conditional prefix admitted an out-of-bounds target row");
  auto moved = std::move(original);
  require(original.empty() && original.rows().empty() && q::take(original.prefix(0)).empty() &&
      !original.prefix(1) && original.prefix(1).error().IsInvalidArgument() && moved.rows().size() == 3 &&
      prefix.rows().data() == moved.rows().data(),"moving a conditional owner retained stale row authority");
  i::ConditionalLogitsInputs assigned; assigned = std::move(moved);
  require(moved.empty() && moved.rows().empty() && assigned.rows().size() == 3 &&
      prefix.rows().data() == assigned.rows().data(),"conditional move assignment broke its retained prefix owner");
}

struct Proposal final {
  std::vector<std::int32_t> ids;
  std::vector<q::c::ReadLease> q;
  std::array<g::QIdentity,2> identities;
  std::array<std::vector<double>,2> laws;
};
inline Proposal propose(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::span<const q::LogitsInputs> rows, const std::array<m::Payload,3> &payloads) {
  const auto before = s::image(model,*session->head); const auto epoch = session->head.epoch();
  Proposal result;
  result.ids = model.propose_recursive(*session,2,{},&result.q,nullptr,{},rows);
  require(result.ids.size() == 3 && result.q.size() == 2 && result.ids.front() == before.pending &&
      session->head.epoch() == epoch && s::image(model,*session->head) == before,
      "real conditional proposer changed its head/RNG or lost actual target/q rows");
  for (std::size_t at = 0; at != 2; ++at) {
    k::q_support(model,result.q[at],payloads[at]); result.identities[at] = g::identity(result.q[at]);
    result.laws[at] = k::probabilities(k::read(model,result.q[at]));
  }
  t::idle(model); return result;
}
inline void same_q(q::Model &model, const Proposal &proposal, const std::array<m::Payload,3> &payloads) {
  require(proposal.q.size() == 2,"conditional Verify lost an original q owner");
  for (std::size_t at = 0; at != 2; ++at) {
    require(g::identity(proposal.q[at]) == proposal.identities[at],"conditional Verify replaced an original q lease");
    k::q_support(model,proposal.q[at],payloads[at]);
  }
}

inline int run(const std::filesystem::path &input,q::BackendKind kind) {
  owner_contract();
  require(kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda,
      "conditional Verify finite consumer is currently checked on CPU/CUDA only");
  auto source = q::prepare_weight_source(input/"weights.tsv",true); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.mtp_layers == 1 && geometry.recurrent_layers() && geometry.attention_layers(),
      "conditional Verify requires the original cached untrained GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry,source.cache_row_bytes,16,6,3,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),2,3,true,true);
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled = true,.policy = {k::temperature,0,1.F},.seed = 123,.modifiers = {.bias = true,.allowed = true}},
      0,{.primary = kind});
  model.configure_draft_policy({.max_drafts = 2});
  const std::array three{std::size_t{3}};
  model.prepare_family(three,false,1,q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3,1,1,q::GraphRole::kMtpPrefill);
  model.prepare_family(three,false,1,q::GraphRole::kVerify,false,q::TextCompletion::kEmitNext,0,q::RowPacking::kVerify);
  model.plan_replay_uses(3,1,3,q::GraphRole::kVerify,q::TextCompletion::kEmitNext,0,q::RowPacking::kVerify);
  model.prepare_recursive_round_programs(0,true); model.freeze_shape_plan();
  model.prepare_execution(three,1,q::GraphRole::kMtpPrefill);
  model.prepare_execution(three,1,q::GraphRole::kVerify,q::TextCompletion::kEmitNext,0,q::RowPacking::kVerify);
  const auto storage = q::take(i::HostStateStorage::create(model.execution,1,4u << 20));
  const std::array payloads{k::row(model,{0,1},-.5F),k::row(model,{1,2},.25F),k::row(model,{2,3},-.75F)};
  const auto prefill = k::row(model,{8,9},.5F), masked = k::row(model,{},0.F);
  const std::array<q::LogitsInputs,3> expected_rows{payloads[0].inputs,payloads[1].inputs,payloads[2].inputs};
  std::vector<q::LogitsInputs> caller_rows(expected_rows.begin(),expected_rows.end());
  const i::ConditionalLogitsInputs frozen{std::span<const q::LogitsInputs>(caller_rows)};
  caller_rows[1] = masked.inputs; caller_rows.clear(); caller_rows.shrink_to_fit();
  const std::array<std::int32_t,3> prompt{1,2,3};
  std::vector<q::GenerationRequest> owners; owners.reserve(1);
  owners.emplace_back("conditional-verify",prompt,std::filesystem::path{},4);
  auto &owner = owners[0]; t::prepare_request(owner,16); owner.logits = prefill.inputs;
  owner.progress.initial_sampling = {123,17,0};
  i::RequestWindow window; window.prepare(1,1); owner.request_key = window.enqueue(0); const auto key = owner.request_key;
  window.admitted(key);
  std::array initial{q::prepare_generation_call(model,owner,4,nullptr,owners,4)};
  q::execute_generation_calls(model,initial,false,nullptr,owners);
  require(owner.ledger->committed_count == 1 && owner.progress.generated.size() == 1,
      "conditional Verify did not start from an actual committed prefill");
  auto original = owner.session->checkpoint(); const auto old_image = s::image(model,*original);
  const auto original_cursor = owner.session->head->sampling; const auto original_epoch = owner.session->head.epoch();
  auto control = model.create_session({},original,q::GraphRole::kMtpPrefill);
  auto control_q = propose(model,control,frozen.rows(),payloads);
  auto actual_q = propose(model,owner.session,frozen.rows(),payloads);
  require(control_q.ids == actual_q.ids,"same checkpoint/cursor proposed different conditional verifier inputs");
  std::size_t target_submissions = 1, observed_raw_targets = 0, independent_decisions = 0;
  const auto inspect = [&](const std::shared_ptr<q::Session> &session,std::span<const q::LogitsInputs> supplied) {
    const auto &step = *model.active_step;
    require(step.prepared && !step.submitted && step.members.size() == 1 && step.input_tokens == 3 &&
        step.active_input_rows == 3 && step.phase->family->role == q::GraphRole::kVerify &&
        step.phase->family->packing == q::RowPacking::kVerify && step.members[0].session == session &&
        step.members[0].conditional_logits.rows().size() == 3,
        "conditional Verify did not use the original bounded target carrier");
    const auto held = step.members[0].conditional_logits.rows();
    for (std::size_t at = 0; at != 3; ++at)
      require(held[at].bias.description == supplied[at].bias.description && held[at].allowed.description == supplied[at].allowed.description &&
          g::identity(held[at].bias.lease) == g::identity(supplied[at].bias.lease) &&
          g::identity(held[at].allowed.lease) == g::identity(supplied[at].allowed.lease),
          "target Step lost its immutable row metadata/ReadLease after caller request destruction");
  };
  const auto finish = [&](const std::shared_ptr<q::Session> &session,q::GenerationLedger &ledger,
      const Proposal &proposal,q::MtpRoundResult &result) {
    auto output = q::evaluation_output(model,session,{.name = "conditional-verify",.trace_states = false},
        nullptr,nullptr,nullptr,nullptr,&ledger);
    output.capture_tensors = true;
    const std::array outputs{output};
    const std::array selections{q::MtpGenerationSelection{session,proposal.ids.front(),3,proposal.q,2}};
    model.finish_recursive_mtp(selections,{},outputs,std::span(&result,1),[&](std::size_t member,const q::CompletedEvaluationView &view) {
      require(member == 0 && view.requests == 1 && view.member_index == 0 && view.observation.input_tokens == 3 &&
          view.logits.description() == q::dense(q::Type::kFloat32,{3,model.vocabulary}),
          "conditional oracle did not observe the actual original F32 target rows");
      std::vector<float> logits(static_cast<std::size_t>(3*model.vocabulary));
      q::check(model.execution.read(view.logits.storage(),std::as_writable_bytes(std::span(logits)))); ++observed_raw_targets;
      std::array<std::vector<double>,3> laws;
      for (std::size_t at = 0; at != 3; ++at)
        laws[at] = m::normalized(std::span<const float>(logits).subspan(at*static_cast<std::size_t>(model.vocabulary),
            static_cast<std::size_t>(model.vocabulary)),payloads[at].bias,payloads[at].allowed,k::temperature);
      const auto reference = k::reference(laws,proposal.laws,std::span<const std::int32_t>(proposal.ids).subspan(1),
          original_cursor,3,{},kind);
      require(result.emitted == reference.emitted && result.processed_target == reference.emitted.size() &&
          result.consumed_drafts == reference.consumed_drafts && result.next_id == reference.emitted.back() &&
          result.agreed == reference.first_agrees && result.stopped == reference.stopped,
          "actual conditional Verify differs from independent real-logit p/q acceptance/residual/bonus mathematics");
      ++independent_decisions;
    });
    t::idle(model);
  };
  q::GenerationLedger control_ledger(1,3); control_ledger.bind_stop_policy({},model.vocabulary);
  const auto pending = original->next_id;
  q::prepare_generation_record(control_ledger,*control,q::GraphRole::kVerify,std::span(&pending,1),true,3,0,{},2);
  {
    const std::array requests{q::EvaluationRequest{.session = control,.ids = control_q.ids,.stop_policy = &control_ledger.stop_policy(),
        .conditional_logits = frozen}};
    model.prepare_verification_evaluations(requests,3);
  }
  inspect(control,frozen.rows()); model.submit_evaluations(); ++target_submissions;
  q::MtpRoundResult expected; finish(control,control_ledger,control_q,expected); same_q(model,control_q,payloads); control_q.q.clear();
  const auto expected_state = p::capture(model,*control,storage);
  q::prepare_generation_record(*owner.ledger,*owner.session,q::GraphRole::kVerify,std::span(&pending,1),true,3,0,{},2);
  auto before = e::save(model,owner,storage);
  const auto target_pages = model.page_inventory.free_ids.size(), draft_pages = model.draft_inventory.free_ids.size();
  const auto versions = model.snapshot_versions.live;
  const auto drafts_before = model.draft_program_graph_calls, terminals_before = model.prefix_decision_graph_calls;
  const auto unchanged = [&](bool retired_failed_slot) {
    require(owner.session->head.get() == before.head && owner.session->head.epoch() == before.epoch &&
        owner.progress.generated == before.generated && owner.progress.observed_commits() == before.observed &&
        owner.progress.input_consumed() == before.consumed && owner.progress.input_turn() == before.input_turn &&
        owner.history.get() == before.history && (!owner.history || owner.history->length == before.history_length) &&
        owner.ledger->committed_count == before.commits && owner.ledger->sampling == before.sampling &&
        !owner.ledger->successor_ordinal() && owner.ledger->records.size() == before.records.size(),
        "conditional failure changed its original head, epoch, progress, RNG or ledger");
    for (std::size_t at = 0; at != before.records.size(); ++at) {
      const auto &record = owner.ledger->records[at];
      if (retired_failed_slot && at == before.commits) {
        q::GenerationCommitRecord empty; empty.emitted.clear(); empty.accepted_inputs.clear(); empty.verification_inputs.clear();
        require(e::same_record(empty,record) && record.emitted.capacity() >= before.records[at].emitted.capacity() &&
            record.accepted_inputs.capacity() >= before.records[at].accepted_inputs.capacity() &&
            record.verification_inputs.capacity() >= before.records[at].verification_inputs.capacity(),
            "retired conditional failure retained uncommitted authority or shrank cold metadata storage");
      } else require(e::same_record(before.records[at],record),"conditional failure changed another ledger record");
    }
    h::same_image(before.image,p::capture(model,*owner.session,storage));
    require(s::image(model,*before.checkpoint) == old_image && s::image(model,*original) == old_image,
        "conditional failure changed an old complete checkpoint"); t::idle(model);
  };
  for (std::size_t malformed = 0; malformed != 2; ++malformed) {
    auto bad_rows = expected_rows;
    if (!malformed) bad_rows[2].allowed.description = q::dense(q::Type::kBool,{model.vocabulary});
    const i::ConditionalLogitsInputs bad{std::span<const q::LogitsInputs>(bad_rows)};
    const std::array requests{q::EvaluationRequest{.session = owner.session,.ids = actual_q.ids,.stop_policy = &owner.ledger->stop_policy(),
        .conditional_logits = bad,.request_key = key}};
    bool rejected = false;
    try {
      if (!malformed) model.prepare_verification_evaluations(requests,3);
      else model.prepare_evaluations(requests,nullptr,q::GraphRole::kMtpPrefill);
    } catch (const i::RequestExecutionFailure &failure) {
      const auto &facts = failure.evaluation();
      rejected = failure.request_key() == key && facts.status().IsInvalidArgument() && !facts.status().has_native_error() &&
          !facts.participant() && !facts.retirement_observed();
    }
    require(rejected && !model.active_step && model.draft_program_graph_calls == drafts_before &&
        model.prefix_decision_graph_calls == terminals_before && model.snapshot_versions.live == versions &&
        model.page_inventory.free_ids.size() == target_pages && model.draft_inventory.free_ids.size() == draft_pages,
        "bad conditional metadata/role lost its original key or reached a target candidate");
    unchanged(false); same_q(model,actual_q,payloads);
  }
  {
    std::vector<q::LogitsInputs> bad_rows(expected_rows.begin(),expected_rows.end()); bad_rows[1] = masked.inputs;
    const i::ConditionalLogitsInputs bad{std::span<const q::LogitsInputs>(bad_rows)};
    const std::array requests{q::EvaluationRequest{.session = owner.session,.ids = actual_q.ids,.stop_policy = &owner.ledger->stop_policy(),
        .conditional_logits = bad,.request_key = key}};
    model.prepare_verification_evaluations(requests,3); inspect(owner.session,bad.rows());
    bad_rows[1] = expected_rows[1]; bad_rows.clear(); bad_rows.shrink_to_fit();
  }
  model.submit_evaluations(); ++target_submissions;
  bool numeric_failed = false; q::MtpRoundResult ignored;
  try { finish(owner.session,*owner.ledger,actual_q,ignored); }
  catch (const i::EvaluationFailure &failure) {
    numeric_failed = failure.status().IsInvalidArgument() && !failure.status().has_native_error() &&
        failure.participant() && *failure.participant() == 0 && failure.retirement_observed() &&
        failure.retirement().retired && failure.retirement().status.ok();
  }
  require(numeric_failed && model.prefix_decision_graph_calls == terminals_before+1 &&
      model.draft_program_graph_calls == drafts_before && owner.request_key == key && window.matches(key),
      "submitted bad conditional row did not fail after actual retirement or regenerated original q");
  unchanged(true); same_q(model,actual_q,payloads);
  require(model.snapshot_versions.live == versions && model.page_inventory.free_ids.size() == target_pages &&
      model.draft_inventory.free_ids.size() == draft_pages,"retired conditional failure retained candidate states or append pages");
  q::prepare_generation_record(*owner.ledger,*owner.session,q::GraphRole::kVerify,std::span(&pending,1),true,3,0,{},2);
  {
    const std::array requests{q::EvaluationRequest{.session = owner.session,.ids = actual_q.ids,.stop_policy = &owner.ledger->stop_policy(),
        .conditional_logits = frozen,.request_key = key}};
    model.prepare_verification_evaluations(requests,3);
  }
  inspect(owner.session,frozen.rows()); model.submit_evaluations(); ++target_submissions;
  q::MtpRoundResult actual; finish(owner.session,*owner.ledger,actual_q,actual); same_q(model,actual_q,payloads); actual_q.q.clear();
  qwen_mtp_mixed_lag_case::same_round(expected,actual); q::observe_generation_commits(owner);
  const auto &record = owner.ledger->records[1]; const auto after = s::image(model,*owner.session->head);
  require(owner.request_key == key && window.matches(key) && owner.ledger->committed_count == 2 &&
      owner.progress.observed_commits() == 2 && owner.rounds.size() == 1 && record.committed &&
      record.same_result(control_ledger.records[0]) && record.accepted == actual.processed_target &&
      record.verified_rows == 3 && record.verification_inputs.size() == 3 &&
      std::ranges::equal(record.verification_inputs,actual_q.ids) &&
      std::ranges::equal(record.accepted_inputs,std::span<const std::int32_t>(actual_q.ids).first(record.accepted)) &&
      std::ranges::equal(record.emitted,actual.emitted) && owner.session->head.epoch() == original_epoch+1 &&
      after.length == old_image.length+actual.processed_target && after.sampling == original_cursor.advanced(5) &&
      after.pending == actual.next_id && after.lag == (actual.processed_target == 1 ? 0u : 1u) &&
      after.frontier+after.lag == after.length && after.catchup == (after.lag ? actual_q.ids[actual.processed_target-1] : -1),
      "conditional Verify publication lost its accepted ledger/state/RNG/frontier or same request key");
  r::preserved_prefix(old_image,after);
  require(s::image(model,*original) == old_image && s::image(model,*before.checkpoint) == old_image,
      "conditional Verify failure/retry mutated an immutable complete checkpoint");
  std::vector<p::StepReport> state_checks;
  t::equal(model,expected_state,p::capture(model,*owner.session,storage),state_checks);
  state_checks.back().prefix = old_image.length; state_checks.back().active = 3; state_checks.back().physical = 3;
  state_checks.back().epoch = owner.session->head.epoch();
  require(target_submissions == 4 && observed_raw_targets == 2 && independent_decisions == 2 &&
      model.prefix_decision_graph_calls == 3 && model.draft_program_graph_calls == drafts_before+1,
      "conditional Verify did not execute its two real successful targets, one retired failure and one actual repair");
  before.checkpoint.reset(); h::retire(control); window.retire_active(key); h::retire(owner.session); window.release(key);
  original.reset();
  require(!storage->live_images() && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "conditional Verify retained a state, target/draft page or HostImage"); t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"conditional-verify\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"host_owner_move_prefix\":true,\"conditional_rows\":3,\"target_submissions\":" << target_submissions
      << ",\"actual_target_F32_oracles\":" << independent_decisions << ",\"actual_private_q_rows\":2,\"same_q_retry\":true,"
      "\"metadata_and_role_key_zero_target\":true,\"numeric_failure_participant\":0,\"numeric_failure_retired\":true,"
      "\"immutable_owner_survives_caller\":true,\"old_checkpoints_unchanged\":true,\"accepted_target_rows\":" << actual.processed_target
      << ",\"complete_target_draft_state\":true,\"RNG_reserved_draws\":5,\"natural_conditional_supplier\":false,"
      "\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at != state_checks.size(); ++at) { if (at) std::cout << ','; p::write_step(state_checks[at]); }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_conditional_verify_case
