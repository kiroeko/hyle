#pragma once

#include <sstream>

// Natural RequestLoop providers on the cached untrained GA fixture. The
// singleton constraints are actual policy inputs, never forced selectors.
namespace qwen_natural_logits_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace k = qwen_conditional_logits_case;
namespace s = qwen_mtp_batch_state_case;
namespace n = qwen_mtp_natural_relay_case;
namespace p = qwen_packed_prefill_case;
namespace t = qwen_mtp_stop_case;
namespace e = qwen_logits_preflight_case;
using h::require;

struct Call final {
  std::vector<std::int32_t> outputs,hypotheses;
  bool failed = false;
};
struct Trace final {
  std::int32_t first = 8;
  std::vector<Call> calls;
  std::size_t audited = 0;
  bool fail_bonus = false;
  const std::vector<q::c::ReadLease> *private_q = nullptr;
  std::size_t failed_q_leases = 0;
};
using Payloads = std::vector<qwen_mtp_logits_case::Payload>;
inline std::shared_ptr<Payloads> payloads(q::Model &model,std::int32_t first) {
  auto result = std::make_shared<Payloads>(); result->reserve(8);
  for (std::size_t at = 0; at != 8; ++at)
    result->push_back(k::row(model,{first+static_cast<std::int32_t>(at)},.125F*static_cast<float>(at)-.25F));
  return result;
}
inline i::LogitsInputProvider provider(const std::shared_ptr<Trace> &trace,const std::shared_ptr<Payloads> &values) {
  return [trace,values](std::span<const std::int32_t> outputs,std::span<const std::int32_t> hypotheses) -> hyle::Result<q::LogitsInputs> {
    for (std::size_t at = 0; at != outputs.size(); ++at)
      require(outputs[at] == trace->first+static_cast<std::int32_t>(at),"provider received another turn's outputs or an uncommitted token");
    for (std::size_t at = 0; at != hypotheses.size(); ++at)
      require(hypotheses[at] == trace->first+static_cast<std::int32_t>(outputs.size()+at),
          "provider future duplicated the already emitted pending token or another hypothesis prefix");
    const bool fail = trace->fail_bonus && hypotheses.size() == 2;
    trace->calls.push_back({{outputs.begin(),outputs.end()},{hypotheses.begin(),hypotheses.end()},fail});
    if (fail) {
      require(trace->private_q && trace->private_q->size() == 2 &&
          std::ranges::all_of(*trace->private_q,[](const auto &lease) { return lease.valid(); }),
          "late provider failure did not follow two real stochastic q probability leases");
      trace->failed_q_leases = trace->private_q->size();
      return std::unexpected(hyle::Status::InvalidArgument("finite late provider failure"));
    }
    const auto position = outputs.size()+hypotheses.size();
    require(position < values->size(),"natural provider exceeded its finite token support");
    const auto result = values->at(position).inputs;
    require(!result.counts.lease.valid(),"test provider attempted to own engine C0 history");
    return result;
  };
}
inline void audit(Trace &trace,const q::GenerationRequest &request,const q::GenerationCommitRecord &record) {
  require(record.committed && request.session && request.session->head &&
      record.after == q::generation_boundary(*request.session->head) && record.turn == request.progress.turn_index,
      "provider observer did not see the original fully published commit boundary");
  const auto outputs = std::span<const std::int32_t>(request.progress.generated).subspan(request.progress.turn_output_begin);
  const auto wanted = record.completion == q::TextCompletion::kStateOnly ? 0u
      : record.role == q::GraphRole::kVerify ? record.verified_rows : std::size_t{1};
  require(trace.calls.size()-trace.audited == wanted,"natural provider duplicated p_i callbacks or sampled a StateOnly chunk");
  for (std::size_t row = 0; row != wanted; ++row) {
    const auto &call = trace.calls[trace.audited+row];
    const auto future = record.role == q::GraphRole::kVerify
        ? std::span<const std::int32_t>(record.verification_inputs).subspan(1,row) : std::span<const std::int32_t>{};
    require(!call.failed && std::ranges::equal(call.outputs,outputs) && std::ranges::equal(call.hypotheses,future),
        "natural provider did not use the current turn's exact committed outputs and future-only hypotheses");
  }
  if (record.generation && record.draft_limit)
    require(record.role == q::GraphRole::kVerify,"dynamic K1 bypassed the original conditional Verify path");
  const auto emitted = std::span<const std::int32_t>(record.emitted).first(record.emitted_count);
  for (std::size_t at = 0; at != emitted.size(); ++at)
    require(emitted[at] == trace.first+static_cast<std::int32_t>(outputs.size()+at),
        "natural model output ignored the actual singleton conditional policy");
  trace.audited += wanted;
}
struct Study final {
  std::string report;
  std::size_t callbacks = 0,state_only = 0,verifiers = 0,turn_row0 = 0;
  std::size_t late_private_draft_calls = 0,late_q_leases = 0;
  std::vector<p::StepReport> states;
};

inline void late_failure(q::Model &model,const std::shared_ptr<Payloads> &values,Study &study) {
  const std::array<std::int32_t,2> prompt{1,2};
  std::vector<q::GenerationRequest> owners; owners.reserve(1); owners.emplace_back("provider-retry",prompt,std::filesystem::path{},4);
  auto &owner = owners[0]; t::prepare_request(owner,16); owner.progress.initial_sampling = {123,37,0};
  const auto trace = std::make_shared<Trace>(); owner.logits_provider = provider(trace,values);
  i::RequestWindow window; window.prepare(1,1); owner.request_key = window.enqueue(0); const auto key = owner.request_key; window.admitted(key);
  std::array prefill{q::prepare_generation_call(model,owner,4,nullptr,owners,4,2)};
  q::execute_generation_calls(model,prefill,false,nullptr,owners);
  require(owner.progress.generated == std::vector<std::int32_t>{8},"late provider witness did not emit its actual prefill");
  trace->audited = trace->calls.size();
  const auto storage = q::take(i::HostStateStorage::create(model.execution,1,4u << 20));
  auto before = e::save(model,owner,storage);
  const auto old = s::image(model,*before.checkpoint);
  const auto versions = model.snapshot_versions.live;
  const auto target_pages = model.page_inventory.free_ids.size(),
      draft_pages = model.draft_inventory.free_ids.size();
  const auto drafted = model.draft_program_graph_calls, terminals = model.prefix_decision_graph_calls;
  trace->fail_bonus = true; bool failed = false, target_submitted = false;
  std::array bad{q::prepare_generation_call(model,owner,4,nullptr,owners,3,2)};
  trace->private_q = &bad[0].draft_log_probabilities;
  try { q::execute_generation_calls(model,bad,false,nullptr,owners,[&] { target_submitted = true; }); }
  catch (const i::RequestExecutionFailure &failure) {
    const auto &actual = failure.evaluation();
    failed = failure.request_key() == key && actual.status().IsInvalidArgument() && !actual.status().has_native_error() &&
        !actual.participant() && actual.retirement_observed() && actual.retirement().retired && actual.retirement().status.ok();
  }
  require(failed && !target_submitted && model.draft_program_graph_calls == drafted+2 &&
      model.prefix_decision_graph_calls == terminals && owner.request_key == key && window.matches(key) &&
      owner.session->head.get() == before.head && owner.session->head.epoch() == before.epoch &&
      owner.progress.generated == before.generated && owner.progress.observed_commits() == before.observed &&
      owner.ledger->committed_count == before.commits && owner.ledger->sampling == before.sampling &&
      owner.history.get() == before.history && (!owner.history || owner.history->length == before.history_length) &&
      owner.progress.input_consumed() == before.consumed && owner.progress.input_turn() == before.input_turn &&
      bad[0].verification_ids.empty() && bad[0].draft_log_probabilities.empty() && bad[0].conditional_logits.empty(),
      "late provider error changed the head/ledger/RNG/key or escaped actual private retirement");
  for (std::size_t at = 0; at != before.records.size(); ++at)
    require(e::same_record(before.records[at],owner.ledger->records[at]),"late provider failure changed an original ledger record");
  require(model.snapshot_versions.live == versions && model.page_inventory.free_ids.size() == target_pages &&
      model.draft_inventory.free_ids.size() == draft_pages && s::image(model,*owner.session->head) == old &&
      s::image(model,*before.checkpoint) == old,"late provider failure retained private pages or mutated an old checkpoint");
  h::same_image(before.image,p::capture(model,*owner.session,storage)); t::idle(model);
  require(trace->calls.size()-trace->audited == 3 && trace->calls.back().failed,
      "late provider error did not follow two real q rows and the one bonus callback");
  study.late_private_draft_calls = model.draft_program_graph_calls-drafted;
  study.late_q_leases = trace->failed_q_leases;
  trace->private_q = nullptr; trace->fail_bonus = false; trace->audited = trace->calls.size();
  std::array retry{q::prepare_generation_call(model,owner,4,nullptr,owners,3,2)};
  const auto observer = [&](const q::GenerationBatchFacts &facts) {
    require(facts.members.size() == 1 && facts.members[0].request == &owner,"provider retry replaced its original request");
    audit(*trace,owner,owner.ledger->records[owner.ledger->committed_count-1]);
  };
  q::execute_generation_calls(model,retry,false,nullptr,owners,nullptr,nullptr,observer);
  require(owner.request_key == key && window.matches(key) && owner.ledger->committed_count == before.commits+1 &&
      owner.progress.observed_commits() == before.observed+1 && owner.progress.generated == std::vector<std::int32_t>({8,9,10,11}) &&
      owner.session->head.epoch() == before.epoch+1 && owner.session->length() == old.length+3 &&
      owner.session->head->sampling == old.sampling.advanced(5) && s::image(model,*before.checkpoint) == old &&
      owner.session->head->draft->lag == 1 && owner.session->head->draft->frontier+1 == owner.session->length(),
      "same-key provider retry did not publish its actual constrained target/draft state");
  window.retire_active(key); h::retire(owner.session); window.release(key); before.checkpoint.reset();
  require(!model.snapshot_versions.live && !storage->live_images(),"late provider retry retained a state or HostImage"); t::idle(model);
}

inline Study mtp(const std::filesystem::path &input,q::BackendKind kind,std::size_t maximum) {
  auto source = q::prepare_weight_source(input/"weights.tsv",true); const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry,source.cache_row_bytes,16,2,3,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),1,maximum+1,true,true);
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled = maximum == 2,.policy = {maximum == 2 ? k::temperature : 1.F,0,1.F},
          .seed = 123,.modifiers = {.bias = true,.allowed = true}},0,{.primary = kind});
  model.configure_draft_policy({.max_drafts = maximum});
  const auto values = payloads(model,8);
  const std::array traces{std::make_shared<Trace>(),std::make_shared<Trace>()};
  const std::array providers{provider(traces[0],values),provider(traces[1],values)};
  const std::vector<std::int32_t> prompt{1,2,3,4};
  const std::array<i::GenerationSessionSchedule,2> schedules{{{"natural-a",{{prompt,4},{{5,6},3}}},
      {"natural-b",{{prompt,4},{{5,6},3}}}}};
  const std::array<q::cache::TokenPlanRef,0> tokens{};
  std::array<std::vector<s::StateImage>,2> states;
  std::array<std::vector<q::GenerationCommitRecord>,2> records;
  std::array<std::vector<std::size_t>,2> active,physical;
  std::array<std::uint64_t,2> draws{};
  Study result;
  const auto observer = [&](const q::GenerationBatchFacts &facts) {
    require(facts.members.size() == 1,"finite natural provider unexpectedly changed R1 geometry");
    const auto &request = *facts.members[0].request; const auto index = request.request_key.slot;
    require(index < 2 && request.ledger->committed_count == records[index].size()+1,
        "natural observer skipped or duplicated an original commit");
    const auto &record = request.ledger->records[request.ledger->committed_count-1];
    audit(*traces[index],request,record);
    result.state_only += record.completion == q::TextCompletion::kStateOnly;
    result.verifiers += record.role == q::GraphRole::kVerify;
    result.turn_row0 += !record.generation && record.completion == q::TextCompletion::kEmitNext;
    const auto delta = record.completion == q::TextCompletion::kStateOnly ? std::uint64_t{0}
        : record.role == q::GraphRole::kVerify ? 2*(record.verified_rows-1)+1 : std::uint64_t{1};
    const auto cursor = maximum == 2 ? q::cache::SamplingCursor{123,index,draws[index]} : q::cache::SamplingCursor{};
    require(record.before.sampling == cursor && record.after.sampling == (maximum == 2 ? cursor.advanced(delta) : cursor),
        "natural provider changed an independent request's seed, stream or actual draw interval");
    if (maximum == 2) draws[index] += delta;
    states[index].push_back(s::image(model,*request.session->head)); records[index].push_back(record);
    active[index].push_back(facts.active_rows); physical[index].push_back(facts.physical_rows);
  };
  const auto now = q::Clock::now(); std::ostringstream natural_report; auto *old_out = std::cout.rdbuf(natural_report.rdbuf());
  hyle::util::ScopeGuard restore{[&]() noexcept { std::cout.rdbuf(old_out); }};
  require(q::run_generation_sessions<q::GenerationMode::kMtp>(model,schedules,{},false,now,now,false,tokens,{}, {},2,false,false,
      providers,observer) == 0,"natural conditional MTP RequestLoop failed");
  std::cout.rdbuf(old_out); restore.dismiss(); result.report = natural_report.str();
  require(states[0].size() == states[1].size() && !states[0].empty() && result.state_only && result.verifiers && result.turn_row0 == 4,
      "natural conditional request omitted StateOnly, Verify or a new-turn row0");
  for (std::size_t at = 0; at != states[0].size(); ++at) {
    auto expected_record = records[0][at]; auto expected_state = states[0][at];
    // Both original cursors were checked above. Only this expected copy is
    // translated to the other known stream; no actual result is rewritten.
    if (maximum == 2) {
      expected_record.before.sampling.stream = 1; expected_record.after.sampling.stream = 1;
      expected_state.sampling.stream = 1;
    }
    require(expected_record.same_result(records[1][at]),"identical natural policies changed original ledger outcomes");
    auto checked = n::compare_state(model,expected_state,states[1][at],active[1][at]);
    checked.sequence = at; checked.prefix = records[1][at].before.length; checked.physical = physical[1][at];
    checked.prefill = !records[1][at].generation; result.states.push_back(std::move(checked));
  }
  for (const auto &trace : traces) {
    require(trace->audited == trace->calls.size(),"natural provider retained unaudited speculative callbacks");
    result.callbacks += trace->calls.size();
  }
  if (maximum == 2) late_failure(model,values,result);
  require(!model.snapshot_versions.live && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),"natural MTP provider retained state/page owners");
  t::idle(model); return result;
}

inline Study cache(const std::filesystem::path &input,q::BackendKind kind) {
  auto source = q::prepare_weight_source(input/"weights.tsv",false); const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry,source.cache_row_bytes,16,2,3,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),1,1,false,true);
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled = false,.policy = {},.seed = 123,.modifiers = {.bias = true,.allowed = true}},0,{.primary = kind});
  const std::array traces{std::make_shared<Trace>(),std::make_shared<Trace>(),std::make_shared<Trace>()};
  std::array<i::LogitsInputProvider,3> providers;
  for (std::size_t at = 0; at != 3; ++at) {
    traces[at]->first = 8+static_cast<std::int32_t>(4*at);
    providers[at] = provider(traces[at],payloads(model,traces[at]->first));
  }
  const std::vector<std::int32_t> a{1,2,3,4},b{4,3,2,1};
  const std::array<i::GenerationSessionSchedule,3> schedules{{{"cache-a",{{a,1}}},{"cache-b",{{b,1}}},{"cache-new",{{a,1}}}}};
  const std::array tokens{q::cache::token_plan(a,16),q::cache::token_plan(b,16),q::cache::token_plan(a,16)};
  Study result;
  const auto observer = [&](const q::GenerationBatchFacts &facts) {
    for (const auto &member : facts.members) {
      const auto &request = *member.request; const auto &record = request.ledger->records[request.ledger->committed_count-1];
      require(request.request_key.slot < 3,"cache observer changed its original request key");
      audit(*traces[request.request_key.slot],request,record);
      result.state_only += record.completion == q::TextCompletion::kStateOnly;
    }
  };
  const auto now = q::Clock::now(); std::ostringstream natural_report; auto *old_out = std::cout.rdbuf(natural_report.rdbuf());
  hyle::util::ScopeGuard restore{[&]() noexcept { std::cout.rdbuf(old_out); }};
  require(q::run_generation_sessions<q::GenerationMode::kText>(model,schedules,{},false,now,now,true,tokens,{}, {},2,false,false,
      providers,observer) == 0,"natural cache/provider RequestLoop failed");
  std::cout.rdbuf(old_out); restore.dismiss(); result.report = natural_report.str();
  require(model.frontier_sampling_graph_calls == 1 && result.state_only > 0 &&
      result.report.find("\"prefix_full_hits\":1") != std::string::npos,
      "natural cache provider did not use an actual full-hit frontier after bounded prefill");
  for (const auto &trace : traces) {
    require(trace->calls.size() == 1 && trace->calls[0].outputs.empty() && trace->calls[0].hypotheses.empty() && !trace->calls[0].failed,
        "cached/ordinary prefill called its provider more than once or leaked a previous request's outputs");
    result.callbacks += trace->calls.size();
  }
  const auto consumer = result.report.find("\"session_id\":\"cache-new\",\"result\":");
  const auto generated = result.report.find("\"generated_ids\":[",consumer);
  require(consumer != std::string::npos && generated != std::string::npos &&
      result.report.compare(generated,std::string("\"generated_ids\":[16]").size(),"\"generated_ids\":[16]") == 0,
      "cache full hit reused its publisher's policy output instead of the current request's provider");
  require(!model.snapshot_versions.live && model.page_inventory.free_ids.size() == plan.max_pages,
      "natural cache provider retained committed state or target pages"); t::idle(model); return result;
}

inline void write(const Study &study,std::size_t drafts) {
  std::cout << "{\"maximum_drafts\":" << drafts << ",\"provider_calls\":" << study.callbacks << ",\"StateOnly_commits\":" << study.state_only
      << ",\"Verify_commits\":" << study.verifiers << ",\"turn_row0_calls\":" << study.turn_row0 << ",\"late_private_draft_calls\":" << study.late_private_draft_calls
      << ",\"late_probability_leases\":" << study.late_q_leases
      << ",\"natural_report\":" << study.report << ",\"state_checks\":[";
  for (std::size_t at = 0; at != study.states.size(); ++at) { if (at) std::cout << ','; p::write_step(study.states[at]); }
  std::cout << "]}";
}
inline int run(const std::filesystem::path &input,q::BackendKind kind) {
  const auto geometry = q::read_weight_geometry(input/"weights.tsv");
  require((kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda) && geometry.hidden == 64 &&
      geometry.vocabulary == 64 && geometry.context_limit >= 16 && geometry.mtp_layers == 1 && geometry.recurrent_layers() && geometry.attention_layers(),
      "natural providers require the cached untrained GA/H64/V64/C16 CPU/CUDA fixture");
  const auto k1 = mtp(input,kind,1),k2 = mtp(input,kind,2),ordinary = cache(input,kind);
  require(k2.late_private_draft_calls == 2 && k2.late_q_leases == 2,
      "late provider witness did not execute both original private stochastic q rows");
  std::cout << "{\"status\":\"PASS\",\"case\":\"natural-logits\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":3,\"natural_RequestLoop\":true,\"provider_future_excludes_pending\":true,\"current_turn_outputs\":true,"
      "\"p_rows_reuse_q_inputs\":true,\"StateOnly_zero_calls\":true,\"dynamic_K1_Verify\":true,\"cache_current_payload\":true,"
      "\"stochastic_K2\":true,\"independent_request_RNG\":true,"
      "\"late_provider_retired\":true,\"same_key_retry\":true,\"old_checkpoint_unchanged\":true,\"synthetic_untrained\":true,"
      "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0,\"journeys\":[";
  write(k1,1); std::cout << ','; write(k2,2); std::cout << ','; write(ordinary,0); std::cout << "]}\n"; return 0;
}
} // namespace qwen_natural_logits_case
