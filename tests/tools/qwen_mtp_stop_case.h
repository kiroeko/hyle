#pragma once

// Real cached V64 GA/MTP mathematics and original request/ledger/cache owners.
// Matching verifier inputs come from an actual target control, never a forced
// acceptance/selected-output callback. This fixture has no model EOS IDs.
namespace qwen_mtp_stop_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
using h::require;

inline void idle(q::Model &model) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed,
      "MTP stop check retained its original execution episode");
}
inline std::int32_t raw(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::span<const std::int32_t> ids, q::GraphRole role) {
  model.prepare_evaluation(*session,ids,nullptr,role); model.submit_evaluation(*session);
  const std::array outputs{q::EvaluationOutput{.session=session}};
  std::array<std::int32_t,1> next{-1};
  model.finish_evaluations(outputs,next,[](auto,const auto &){}); idle(model); return next[0];
}
inline void equal(q::Model &model, const h::ImageBytes &expected, const h::ImageBytes &actual,
    std::vector<p::StepReport> &checks) {
  p::StepReport checked; p::compare(model,expected,actual,checked);
  checked.prefix=expected.boundary.length;checked.exact_next=expected.boundary.next_id;
  checked.bucket_next=actual.boundary.next_id;
  if (!checked.numerically_equivalent) {
    std::cerr<<"mtp_stop_state_check index="<<checks.size()<<" sequence="<<checked.sequence
        <<" prefix="<<checked.prefix<<" active="<<checked.active<<" physical="<<checked.physical
        <<" epoch="<<checked.epoch<<" prefill="<<checked.prefill<<" exact_next="<<checked.exact_next
        <<" bucket_next="<<checked.bucket_next<<" boundary_equal="<<checked.boundary_equal
        <<" payloads_byte_equal="<<checked.payloads_equal
        <<" numerically_equivalent="<<checked.numerically_equivalent<<'\n';
    const auto boundary = [](const char *name,const q::cache::HostImageBoundary &value) {
      std::cerr<<"mtp_stop_state_boundary side="<<name<<" length="<<value.length
          <<" evaluated_next="<<value.evaluated_next_id<<" next="<<value.next_id
          <<" sampling_wrapper="<<value.sampling_wrapper<<" has_carry="<<value.has_carry
          <<" has_prior="<<value.has_prior<<" evaluated_seed="<<value.evaluated_sampling.seed
          <<" evaluated_stream="<<value.evaluated_sampling.stream<<" evaluated_draw="<<value.evaluated_sampling.draw
          <<" seed="<<value.sampling.seed<<" stream="<<value.sampling.stream<<" draw="<<value.sampling.draw
          <<" has_draft="<<value.draft.has_value();
      if (value.draft) std::cerr<<" frontier="<<value.draft->frontier<<" lag="<<value.draft->lag
          <<" catchup_token="<<value.draft->catchup_token;
      std::cerr<<'\n';
    };
    boundary("expected",expected.boundary);boundary("actual",actual.boundary);
    for (const auto &part:checked.differences)
      std::cerr<<"mtp_stop_state_payload role="<<static_cast<int>(part.layout.role)
          <<" layer="<<part.layout.layer<<" page="<<part.layout.page<<" rows="<<part.layout.initialized_rows
          <<" bytes="<<part.layout.bytes<<" first_byte="<<part.first_byte<<" different_bytes="<<part.different_bytes
          <<" maximum_absolute="<<part.maximum_absolute<<" maximum_relative="<<part.maximum_relative
          <<" nonfinite_differences="<<part.nonfinite_differences<<'\n';
  }
  require(checked.numerically_equivalent,"MTP stop changed a complete initialized state or logical boundary");
  checks.push_back(std::move(checked));
}
inline std::shared_ptr<q::Session> fork(q::Model &model, const q::cache::SnapshotRef &base) {
  const std::array one{std::size_t{1}};
  return model.create_session(one,base,q::GraphRole::kMtpSingle);
}
struct Calibration final {
  std::array<std::int32_t,3> prompt{};
  q::cache::SnapshotRef base;
  s::StateImage old;
  std::int32_t pending=-1,target=-1,bonus=-1,draft=-1;
  std::size_t attempts=0;
};
inline Calibration calibrate(q::Model &model) {
  constexpr std::array<std::array<std::int32_t,3>,4> prompts{{{1,2,3},{3,2,1},{1,2,4},{4,5,6}}};
  const std::array three{std::size_t{3}};
  for (std::size_t at=0;at<prompts.size();++at) {
    auto session=model.create_session(three,{},q::GraphRole::kMtpPrefill);
    const auto pending=raw(model,session,prompts[at],q::GraphRole::kMtpPrefill);
    auto base=session->checkpoint(); const auto old=s::image(model,*base);
    const auto proposals=model.propose_recursive(*session,2,q::model_eos_tokens);
    require(proposals.size()==3 && proposals[0]==pending && s::image(model,*base)==old,
        "finite calibration did not run the original two NextN proposals without publication");
    const auto target=raw(model,session,std::span(&pending,1),q::GraphRole::kMtpSingle);
    const auto bonus=raw(model,session,std::span(&target,1),q::GraphRole::kMtpSingle);
    std::cerr<<"mtp_stop_calibration attempt="<<at+1<<" pending="<<pending<<" target="<<target
        <<" bonus="<<bonus<<" native_draft="<<proposals[1]<<'\n';
    h::retire(session);
    if (pending!=target && pending!=bonus && target!=bonus && proposals[1]!=pending && proposals[1]!=target)
      return {prompts[at],std::move(base),old,pending,target,bonus,proposals[1],at+1};
    base.reset();
  }
  throw std::runtime_error("four actual tiny prompts did not expose the distinct stop/bonus/rejected-proposal boundaries");
}

struct Selected final { q::MtpRoundResult result; q::GenerationCommitRecord record; };
inline Selected select(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::span<const std::int32_t> ids, q::GraphRole role, std::size_t quota,
    const i::RequestStopPolicy &policy) {
  q::GenerationLedger ledger(1,3); ledger.bind_stop_policy(policy,model.vocabulary);
  const auto pending=session->head->next_id;
  q::prepare_generation_record(ledger,*session,role,std::span(&pending,1),true,quota,0,{},ids.size()-1);
  const std::array input{q::EvaluationRequest{session,ids,&ledger.stop_policy()}};
  model.prepare_evaluations(input,nullptr,role); model.submit_evaluations();
  const auto output=q::evaluation_output(model,session,{.name="mtp-stop-raw",.trace_states=false},
      nullptr,nullptr,nullptr,nullptr,&ledger);
  const std::array outputs{output}; const std::array selections{q::MtpGenerationSelection{session,pending,quota}};
  std::array<q::MtpRoundResult,1> results;
  if (role==q::GraphRole::kVerify)
    model.finish_recursive_mtp(selections,q::model_eos_tokens,outputs,results,[](auto,const auto &){});
  else model.finish_mtp_rounds(selections,q::model_eos_tokens,outputs,results,[](auto,const auto &){});
  idle(model); require(ledger.committed_count==1,"MTP stop raw episode did not commit its original ledger");
  return {std::move(results[0]),ledger.records[0]};
}

inline void prepare_request(q::GenerationRequest &request, std::size_t capacity) {
  auto owner=std::make_shared<std::vector<std::int32_t>>(request.prompt.begin(),request.prompt.end());
  request.prompt=*owner; request.warm_inputs.push_back(std::move(owner));
  request.input=q::cache::token_plan(request.prompt,capacity);
  request.ledger=std::make_unique<q::GenerationLedger>(8,3);
  request.round_emitted.resize(request.progress.maximum_outputs()-1);
  for (auto &row:request.round_emitted) row.reserve(3);
  request.turn_observations.resize(1);
}

inline int run(const std::filesystem::path &input,q::BackendKind kind) {
  auto source=q::prepare_weight_source(input/"weights.tsv",true); const auto geometry=source.roles.geometry;
  require(geometry.hidden==64 && geometry.vocabulary==64 && geometry.context_limit>=16 &&
      geometry.recurrent_layers() && geometry.attention_layers() && geometry.mtp_layers==1,
      "MTP stop consumer requires the original cached C16 V64 GA/MTP fixture");
  constexpr std::size_t capacity=16;
  // Calibration, two original heads/checkpoints and one published prefix are
  // finite concurrent readers. No second execution or completion owner exists.
  const auto plan=q::cache::plan(geometry,source.cache_row_bytes,capacity,8,3,
      q::recurrent_bytes(q::state_schema_for(geometry,capacity),geometry),2,3,true,true);
  q::Model model(std::move(source),geometry,capacity,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled=false,.policy={},.seed=123},0,{.primary=kind});
  model.configure_draft_policy({.max_drafts=2});
  const std::array three{std::size_t{3}},two{std::size_t{2}},one{std::size_t{1}};
  for (const auto width:{std::size_t{1},std::size_t{2}}) {
    model.prepare_family(three,false,width,q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(3,width,12,q::GraphRole::kMtpPrefill);
    model.prepare_family(two,false,width,q::GraphRole::kVerify);
    model.plan_replay_uses(2,width,8,q::GraphRole::kVerify);
    model.prepare_family(one,false,width,q::GraphRole::kMtpSingle);
    model.plan_replay_uses(1,width,24,q::GraphRole::kMtpSingle);
  }
  model.prepare_family(two,false,1,q::GraphRole::kMtpRound);
  model.plan_replay_uses(2,1,4,q::GraphRole::kMtpRound);
  model.prepare_draft_program(1,true); model.prepare_draft_program(1,false);
  for (const auto width:{std::size_t{1},std::size_t{2}}) {
    model.prepare_execution(three,width,q::GraphRole::kMtpPrefill);
    model.prepare_execution(two,width,q::GraphRole::kVerify);
    model.prepare_execution(one,width,q::GraphRole::kMtpSingle);
  }
  model.prepare_execution(two,1,q::GraphRole::kMtpRound);
  const auto storage=q::take(i::HostStateStorage::create(model.execution,1,4u<<20));
  auto facts=calibrate(model);
  const std::array target_stop{facts.target},draft_stop{facts.draft},bonus_stop{facts.bonus};
  const i::RequestStopPolicy stop_target(target_stop),stop_draft(draft_stop),stop_bonus(bonus_stop),no_stop;
  std::vector<p::StepReport> checks;

  // Original fused K1: the actual proposal is rejected, and the target's new
  // output stop remains pending. Compare against target-only actual state.
  {
    auto control=fork(model,facts.base),candidate=fork(model,facts.base);
    const auto expected=raw(model,control,std::span(&facts.pending,1),q::GraphRole::kMtpSingle);
    const auto epoch=candidate->head.epoch();
    const auto result=select(model,candidate,std::span(&facts.pending,1),q::GraphRole::kMtpRound,2,stop_target);
    require(expected==facts.target && result.result.draft==facts.draft && !result.result.agreed &&
        result.result.stopped && result.result.emitted==std::vector<std::int32_t>{facts.target} &&
        result.result.processed_target==1 && result.record.stopped && result.record.accepted==1 &&
        candidate->head.epoch()==epoch+1 && candidate->length()==facts.old.length+1 &&
        candidate->head->next_id==facts.target && candidate->head->sampling==facts.old.sampling,
        "K1 stop changed the actual rejection, pending/consumed state or original greedy cursor");
    equal(model,p::capture(model,*control,storage),p::capture(model,*candidate,storage),checks);
    h::retire(control);h::retire(candidate);
  }
  // Same real verifier/math with and without an extra-stop on its actual
  // bonus. No callback assigns accepted_tokens or the selected output.
  {
    auto control=fork(model,facts.base),candidate=fork(model,facts.base);
    const std::array ids{facts.pending,facts.target};
    const auto ordinary=select(model,control,ids,q::GraphRole::kVerify,2,no_stop);
    const auto stopped=select(model,candidate,ids,q::GraphRole::kVerify,2,stop_bonus);
    require(ordinary.result.agreed && stopped.result.agreed && ordinary.result.accepted_draft &&
        stopped.result.accepted_draft && !ordinary.result.stopped && stopped.result.stopped &&
        stopped.result.emitted==std::vector<std::int32_t>({facts.target,facts.bonus}) &&
        stopped.result.processed_target==2 && stopped.result.consumed_drafts==1 &&
        stopped.record.accepted==2 && stopped.record.stopped && candidate->head->next_id==facts.bonus &&
        candidate->length()==facts.old.length+2 && candidate->head->draft->lag==1,
        "actual accepted-prefix bonus did not remain pending at the stop boundary");
    h::same_image(p::capture(model,*control,storage),p::capture(model,*candidate,storage));
    h::retire(control);h::retire(candidate);
  }

  std::vector<q::GenerationRequest> requests; requests.reserve(4);
  requests.emplace_back("accepted-stop-A",facts.prompt,std::filesystem::path{},4,0,stop_target);
  requests.emplace_back("rejected-draft-stop-B",facts.prompt,std::filesystem::path{},4,0,stop_draft);
  for (auto &request:requests) prepare_request(request,capacity);
  auto &a=requests[0];auto &b=requests[1];
  i::RequestWindow window;window.prepare(2,2);
  a.request_key=window.enqueue(0);b.request_key=window.enqueue(1);
  const auto key_a=a.request_key,key_b=b.request_key;
  window.admitted(key_a);window.admitted(key_b);
  std::array delivery{i::OutputDelivery(4),i::OutputDelivery(4)};
  const auto prepare=[&](q::GenerationRequest &request,std::size_t credit=SIZE_MAX,q::cache::PrefixCache *prefix=nullptr) {
    return q::prepare_generation_call(model,request,request.progress.output_limit(),prefix,requests,credit);
  };
  std::array initial{prepare(a),prepare(b)};
  q::execute_generation_calls(model,initial,false,nullptr,requests);
  require(a.progress.generated==std::vector<std::int32_t>{facts.pending} && b.progress.generated==a.progress.generated &&
      !a.progress.turn_done && !b.progress.turn_done,"B2 prefill did not reproduce its actual calibration pending");
  delivery[0].acknowledge_prefix(a.progress.generated.size(),1);delivery[1].acknowledge_prefix(b.progress.generated.size(),1);
  std::array checkpoints{a.session->checkpoint(),b.session->checkpoint()};
  const std::array before{s::image(model,*checkpoints[0]),s::image(model,*checkpoints[1])};
  const std::array epochs{a.session->head.epoch(),b.session->head.epoch()};
  std::array<h::ImageBytes,2> expected;
  for (std::size_t member=0;member<2;++member) {
    auto control=fork(model,checkpoints[member]);
    require(raw(model,control,std::span(&facts.pending,1),q::GraphRole::kMtpSingle)==facts.target,
        "original target-only member does not agree with the actual verifier candidate");
    expected[member]=p::capture(model,*control,storage);h::retire(control);
  }
  const auto proposals=model.propose_recursive(*b.session,2,q::model_eos_tokens,nullptr,&b.ledger->stop_policy());
  require(proposals==std::vector<std::int32_t>({facts.pending,facts.draft}) &&
      s::image(model,*b.session->head)==before[1],
      "per-request stop did not truncate the actual NextN proposal without publishing state/RNG");
  std::array calls{prepare(a,delivery[0].available(a.progress.generated.size())),
      prepare(b,delivery[1].available(b.progress.generated.size()))};
  require(calls[0].role==q::GraphRole::kVerify && calls[1].role==q::GraphRole::kVerify,
      "stop requests did not use their original recursive role");
  // A's matching candidate is a real target-only result. B's candidate below
  // remains the original native proposal, generated by execute_generation_calls.
  calls[0].verification_ids={facts.pending,facts.target};
  bool submitted_b2=false;
  q::execute_generation_calls(model,calls,false,nullptr,requests,[&] {
    const auto &step=*model.active_step;
    submitted_b2=step.submitted && step.members.size()==2 && step.input_tokens==2 &&
        step.members[0].verification_inputs==std::vector<std::int32_t>({facts.pending,facts.target}) &&
        step.members[1].verification_inputs==proposals &&
        step.members[0].stop_policy==&a.ledger->stop_policy() && step.members[1].stop_policy==&b.ledger->stop_policy();
    require(submitted_b2,"different stop policies split or replaced the original B2 verifier");
  });
  require(submitted_b2 && a.progress.generated==std::vector<std::int32_t>({facts.pending,facts.target}) &&
      b.progress.generated==a.progress.generated && a.progress.turn_done && !b.progress.turn_done &&
      a.ledger->records[1].stopped && !b.ledger->records[1].stopped && a.ledger->records[1].agreed &&
      !b.ledger->records[1].agreed && a.ledger->records[1].accepted==1 && b.ledger->records[1].accepted==1 &&
      a.session->head->next_id==facts.target && b.session->head->next_id==facts.target &&
      a.session->head.epoch()==epochs[0]+1 && b.session->head.epoch()==epochs[1]+1,
      "accepted stop consumed its pending token or rejected draft-stop terminated its peer");
  for (std::size_t member=0;member<2;++member) {
    auto &request=requests[member];
    require(request.session->length()==facts.old.length+1 && request.session->head->draft->frontier==request.session->length() &&
        !request.session->head->draft->lag && request.session->head->sampling==before[member].sampling &&
        s::image(model,*checkpoints[member])==before[member],"B2 stop changed an old complete prefix, epoch or cursor");
    equal(model,expected[member],p::capture(model,*request.session,storage),checks);
  }

  q::cache::PrefixCache prefix(1);q::publish_generation_completed(model,a,&prefix);
  require(prefix.size()==1 && a.completed_checkpoint.input && a.completed_checkpoint.input->ids.size()==facts.old.length+1,
      "accepted stop failed to publish exactly its consumed prefix");
  window.release_turn(key_a,false);
  require(window.matches(key_a) && delivery[0].pending(a.progress.generated).size()==1 &&
      delivery[0].pending(a.progress.generated)[0]==facts.target,
      "stopped request lost its final pending output before acknowledgement");
  delivery[0].acknowledge_prefix(a.progress.generated.size(),2);h::retire(a.session);window.release(key_a);
  // A's source prefix stopped on T. A new request must classify its reused
  // pending with its own immutable policy, including ledger import/full hit.
  for (const bool stopped:{false,true}) {
    requests.emplace_back(stopped?"new-stopping-hit":"new-continuing-hit",a.completed_checkpoint.input->ids,
        std::filesystem::path{},2,0,stopped?stop_target:no_stop);
    auto &hit=requests.back();prepare_request(hit,capacity);hit.input=a.completed_checkpoint.input;
    std::array hit_call{prepare(hit,2,&prefix)};
    require(hit_call[0].full_hit,"new MTP request did not use the actual completed prefix");
    q::execute_generation_calls(model,hit_call,false,&prefix,requests);
    require(hit.progress.generated==std::vector<std::int32_t>{facts.target} && hit.progress.turn_done==stopped &&
        hit.ledger->records[hit.ledger->committed_count-1].stopped==stopped &&
        hit.ledger->stop_policy()==hit.stop_policy && hit.session->length()==facts.old.length+1,
        "MTP full hit inherited the old publisher's stop policy or consumed its pending");
    h::retire(hit.session);
  }
  auto b_state=p::capture(model,*b.session,storage);const auto b_boundary=q::generation_boundary(*b.session->head);
  const auto b_ids=b.progress.generated;const auto b_observed=b.progress.observed_commits();
  b.last_idle=window.pause(key_b);h::retire(b.session);
  q::restore_generation_session(model,b,nullptr,false,requests);
  require(b.request_key==key_b && window.matches(key_b) && b.progress.generated==b_ids &&
      b.progress.observed_commits()==b_observed && b.ledger->stop_policy()==stop_draft &&
      q::generation_boundary(*b.session->head)==b_boundary && b.restore_count==1,
      "MTP same-request rebuild changed its original key, policy, pending or observations");
  const auto rebuilt_state=p::capture(model,*b.session,storage);
  require(!b_state.boundary.sampling_wrapper && rebuilt_state.boundary.sampling_wrapper,
      "committed-token rebuild did not preserve the explicit historical pending wrapper");
  // Rebuild restores the ledger's pending/cursor through a canonical sampling
  // wrapper. Only this expected representation bit changes; the complete
  // evaluated/historical boundaries and every initialized payload still use
  // the original comparison below.
  b_state.boundary.sampling_wrapper=true;
  equal(model,b_state,rebuilt_state,checks);
  window.wake(key_b);window.admitted(key_b);
  delivery[1].acknowledge_prefix(b.progress.generated.size(),2);
  h::ImageBytes continued_state;
  {
    auto control=fork(model,b.session->checkpoint());
    require(raw(model,control,std::span(&facts.target,1),q::GraphRole::kMtpSingle)==facts.bonus,
        "restored peer's original target-only control changed its next ID");
    continued_state=p::capture(model,*control,storage);h::retire(control);
  }
  std::array continued{prepare(b,1)};
  require(continued[0].role==q::GraphRole::kMtpSingle,"finite peer credit did not choose the original target-only tail");
  q::execute_generation_calls(model,continued,false,nullptr,requests);
  require(b.request_key==key_b && b.progress.generated==std::vector<std::int32_t>({facts.pending,facts.target,facts.bonus}) &&
      b.ledger->records[2].stopped==stop_draft.stopped(facts.bonus,false) && b.session->length()==facts.old.length+2 &&
      b.session->head->sampling==facts.old.sampling,"same-key MTP peer lost its own stop policy after rebuild");
  equal(model,continued_state,p::capture(model,*b.session,storage),checks);
  // The finite owner ends this check after one resumed tail even when B still
  // has quota. Retire its active identity; do not invent a completed turn.
  window.retire_active(key_b);
  require(window.matches(key_b) && delivery[1].pending(b.progress.generated).size()==1,
      "continued peer dropped its final pending delivery");
  delivery[1].acknowledge_prefix(b.progress.generated.size(),3);h::retire(b.session);window.release(key_b);
  for (std::size_t member=0;member<2;++member)
    require(s::image(model,*checkpoints[member])==before[member],"stop/prefix/rebuild mutated an old B2 checkpoint");
  require(s::image(model,*facts.base)==facts.old,"MTP stop journeys mutated the original calibration checkpoint");
  prefix.evict_one();checkpoints={};facts.base.reset();idle(model);
  require(!model.snapshot_versions.live && !storage->live_images() && model.page_inventory.free_ids.size()==plan.max_pages &&
      model.draft_inventory.free_ids.size()==model.draft_inventory.owners.size() &&
      window.active().empty() && !window.waiting_front(),
      "MTP stop journey retained a state/page/native/host-image/request owner");
  std::cout<<"{\"status\":\"PASS\",\"backend\":\""<<h::backend_name(kind)
      <<"\",\"case\":\"mtp-stop-policy\",\"synthetic_untrained\":true,\"calibration_attempts\":"<<facts.attempts
      <<",\"pending\":"<<facts.pending<<",\"target\":"<<facts.target<<",\"bonus\":"<<facts.bonus<<",\"native_draft\":"<<facts.draft
      <<",\"actual_B2\":true,\"accepted_extra_stop_pending\":true,\"native_rejected_draft_stop\":true,"
      "\"fused_K1_stop\":true,\"actual_bonus_stop\":true,\"new_request_prefix_policy\":true,\"same_key_rebuild_policy\":true,"
      "\"rebuild_sampling_wrapper\":true,"
      "\"old_checkpoints_byte_unchanged\":true,\"full_state_numerically_equivalent\":true,\"greedy_rng_draws\":0,"
      "\"controlled_target_matching_candidate\":true,\"real_model_EOS_checked\":false,\"ignoreEOS_host_only\":true,"
      "\"final_versions\":0,\"quality_claim\":false,\"performance_claim\":false,\"state_checks\":[";
  for(std::size_t at=0;at<checks.size();++at){if(at)std::cout<<',';p::write_step(checks[at]);}
  std::cout<<"]}\n";return 0;
}
} // namespace qwen_mtp_stop_case
