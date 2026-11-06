#include "../../engine/models/qwen35/main.cc"
#include "qwen_host_image_case.h"
#include "qwen_mtp_batch_state_case.h"
#include "qwen_device_continuation_case.h"

namespace {
namespace q=qwen35;
namespace s=qwen_mtp_batch_state_case;
using qwen_host_image_case::require;
struct Result {
  std::vector<s::StateImage> states;
  std::vector<std::vector<std::int32_t>> emitted;
};
void cold_policy_facts() {
  const q::GreedyChainCapabilities valid{3, 0, true, 1, 2, 3, true, true};
  require(valid.available(), "complete K3 capability was refused");
  auto changed = valid;
  changed.max_drafts = 1; require(!changed.available(), "K1 selected the multi-stage chain");
  changed = valid; changed.max_drafts = 4; require(!changed.available(), "unsupported depth selected the chain");
  changed = valid; changed.minimum_probability = 0.1f; require(!changed.available(), "threshold selected the greedy chain");
  changed = valid; changed.device_relay = false; require(!changed.available(), "missing device relay selected the chain");
  changed = valid; changed.active_sequences = 2; require(!changed.available(), "multi-sequence selected the chain");
  changed = valid; changed.auxiliary_hidden_rows = 1; require(!changed.available(), "missing hidden rows selected the chain");
  changed = valid; changed.private_draft_stages = 2; require(!changed.available(), "missing private stage selected the chain");
  changed = valid; changed.capture_allowed = false; require(!changed.available(), "commands-only replay selected the chain");
  changed = valid; changed.supported_projection = false; require(!changed.available(), "unsupported projection selected the chain");
  changed = valid; changed.max_drafts = 2; changed.private_draft_stages = 2;
  require(changed.available(), "complete K2 capability was refused");
  {
    hyle::inference::Backend uninitialized;
    bool rejected = false;
    try {
      q::GraphAssembly invalid(uninitialized);
    } catch (const hyle::inference::EvaluationFailure &failure) {
      rejected = failure.status().IsFailedPrecondition();
    }
    require(rejected, "uninitialized backend was silently treated as an unsupported capability");
  }
#if defined(HYLE_COMPUTE_CPU)
  auto cpu_backend = q::take(hyle::inference::Backend::create(
      {.primary = hyle::inference::BackendKind::kCpu}));
  auto *context = q::take(cpu_backend.common_context());
  require(context->provider_count() > 0, "CPU capability test needs a provider");
  { q::GraphAssembly live(cpu_backend); }
  q::check(context->close());
  bool rejected = false;
  try {
    q::GraphAssembly closed(cpu_backend);
  } catch (const hyle::inference::EvaluationFailure &failure) {
    rejected = failure.status().IsFailedPrecondition();
  }
  require(rejected, "closed provider context was silently treated as an unsupported capability");
#endif
}
void zero_draft_cold(const std::filesystem::path &input) {
  for(const auto boundaries:{std::size_t{1},std::size_t{4}}){
  auto source=q::prepare_weight_source(input/"weights.tsv",true);const auto geometry=source.roles.geometry;
  const auto plan=q::cache::plan(geometry,source.cache_row_bytes,16,1,4,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),1,boundaries,true,true);
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled=false,.modifiers={.bias=true,.allowed=true}},0,{.primary=q::BackendKind::kCuda});
  auto capacity = model.attention_capacity();
  require(q::supports_mtp_attention(model.execution, capacity, model.resources_->attention_storage),
      "actual backend refused the supported model attention capacity");
  capacity.qk_head_dim = capacity.value_head_dim = 96;
  require(!q::supports_mtp_attention(model.execution, capacity, model.resources_->attention_storage),
      "unsupported backend head dimensions were admitted");
  model.configure_draft_policy({0,0});
  model.prepare_recursive_round_programs(0,true);
  require(model.draft_programs.size()==boundaries+2 && model.prefix_sampling_programs.size()==boundaries &&
      model.draft_programs.contains(q::DraftProgramKey{1,false,0}) &&
      model.draft_programs.contains(q::DraftProgramKey{boundaries,false,0}) &&
      model.draft_programs.contains(q::DraftProgramKey{1,true,0}) &&
      model.draft_programs.contains(q::DraftProgramKey{2,true,0}) &&
      !model.snapshot_versions.live && !model.active_step && !model.successor_step,
      "zero-draft dynamic policy lost its original auxiliary inventory or published state");
  }
}
Result run(const std::filesystem::path &input,bool fused,bool compound=false,bool parallel=false,bool restart=false) {
  auto source=q::prepare_weight_source(input/"weights.tsv",true);
  const auto geometry=source.roles.geometry;
  require(geometry.context_limit>=32 && geometry.vocabulary>=16,"contract needs existing C32 fixture");
  const auto plan=q::cache::plan(geometry,source.cache_row_bytes,32,4,4,
      q::recurrent_bytes(q::state_schema_for(geometry,32),geometry),1,4,true,false,3,2,{},3);
  q::Model model(std::move(source),geometry,32,plan,{},false,
      fused?q::ReplayPolicy::kPreferCapture:q::ReplayPolicy::kCommands,SIZE_MAX,{},0,{.primary=q::BackendKind::kCuda});
  model.configure_draft_policy({3,0});
  if(parallel)require(model.begin_cold_families(),"parallel cold contract did not start its original worker pool");
  const std::array seed_rows{std::size_t{3}};
  const auto prepare_families=[&]{
  model.prepare_family(seed_rows,false,1,q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3,1,32,q::GraphRole::kMtpPrefill);
  for(std::size_t width=2;width<=4;++width){
    const std::array rows{width};model.prepare_family(rows,false,1,q::GraphRole::kVerify);
    model.plan_replay_uses(width,1,64,q::GraphRole::kVerify);
  }
  if(compound)for(const auto role:{q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound}){
    const std::array rows{std::size_t{4}};model.prepare_family(rows,false,1,role);
    model.plan_replay_uses(4,1,64,role);
  }
  };
  prepare_families();
  if(restart){
    bool refused=false;
    try{model.prepare_recursive_round_programs(0,false,64,0);}
    catch(const std::logic_error&){refused=true;}
    require(refused,"invalid auxiliary width was accepted during cold family work");
    model.cancel_cold_families();
    require(model.families.empty() && model.draft_programs.empty() && !model.active_step &&
        !model.successor_step && !model.snapshot_versions.live,"failed cold work published model or execution state");
    prepare_families();
  }
  model.prepare_recursive_round_programs(0,false,64);
  if(parallel)model.finish_cold_families();
  if(compound)model.prepare_shared_execution_workspace();
  Result result;
  for(const bool lag:{false,true})for(std::size_t accepted=1;accepted<=4;++accepted){
    auto session=model.create_session(seed_rows,{},q::GraphRole::kMtpPrefill);
    const std::array<std::int32_t,3>prompt{1,2,3};
    model.prepare_evaluation(*session,prompt,nullptr,q::GraphRole::kMtpPrefill);
    model.submit_evaluation(*session);
    (void)q::finish_evaluation(model,*session,"seed",{},false);
    const auto prepare=[&](const std::vector<std::int32_t> &ids){
      if(compound){
        const std::array pending{session->head->next_id};
        model.prepare_evaluation(*session,pending,nullptr,session->head->draft->lag
            ?q::GraphRole::kMtpCatchupRound:q::GraphRole::kMtpRound);
        require(model.active_step->phase->draft_chain_steps==3,"test missed complete compound round Graph");
      }else model.prepare_evaluation(*session,ids,nullptr,q::GraphRole::kVerify);
    };
    const auto controlled=[&](std::size_t count){
      const auto ids=model.propose_recursive(*session,3);prepare(ids);
      require(compound || model.active_step->phase->recursive_draft_repair==fused,"test missed requested real repair route");
      model.submit_evaluation(*session);
      const std::array outputs{q::EvaluationOutput{.session=session}};
      std::array<std::int32_t,1>next;
      model.finish_selected_evaluations(outputs,next,[&](auto,q::StepState &step,auto selected){
        step.members[0].accepted_tokens=count;selected[0]=7;
      },[&](q::StepState &step){model.repair_recursive_draft(step);},
      [](std::size_t,const q::CompletedEvaluationView&){});
    };
    if(lag)controlled(2);
    auto old=session->checkpoint();const auto before=s::image(model,*old);
    if(fused && !lag && accepted==1){
      bool rejected=false;
      try{
        (void)model.propose_recursive(*session,3,{},nullptr,nullptr,{}, {},nullptr,
            [](std::span<const std::int32_t>,const std::shared_ptr<q::StepSources>&,bool){
              throw std::runtime_error("intentional verifier preparation rejection");
            });
      }catch(const std::runtime_error &error){
        if(std::string_view(error.what())!="intentional verifier preparation rejection")throw;
        rejected=true;
      }
      require(rejected,"private chain did not enter its real submitted packet export");
      require(s::image(model,*session->head)==before && s::image(model,*old)==before,
          "failed compound proposal changed a committed checkpoint/state/RNG");
      const auto retry=model.propose_recursive(*session,3);
      require(retry.size()==4 && model.private_publications_retired(),
          "compound proposal did not recover its real COW/Store/Source cohort for retry");
      require(s::image(model,*session->head)==before,"valid private retry published approximate draft state");
      const auto no_drafts=model.propose_recursive(*session,0);
      require(no_drafts.size()==1 && session->proposal_graph_calls==0,
          "zero-width proposal retained the preceding Graph work count");
    }
    // A real submitted verifier is withdrawn through the original retirement
    // path. No selected prefix, old snapshot or RNG cursor may be published.
    {
      const auto ids=model.propose_recursive(*session,3);prepare(ids);
      model.submit_evaluation(*session);
      const auto retired=model.abort_episode();q::check(retired.status);
      require(retired.retired && !model.active_step && !model.carrier_borrowed,"withdrawal did not retire real verifier");
      require(s::image(model,*session->head)==before && s::image(model,*old)==before,
          "withdrawal changed a committed checkpoint/state/RNG");
    }
    controlled(accepted);
    result.states.push_back(s::image(model,*session->head));
    require(session->head->draft->lag==(accepted==1?0u:1u),"selected repair lag differs");
    require(session->head->draft->frontier+session->head->draft->lag==session->length(),"selected frontier differs");
    require(s::image(model,*old)==before,"controlled repair changed old checkpoint");
    for(int round=0;round<2;++round){
      const auto pending=session->head->next_id;
      auto inputs=model.propose_recursive(*session,3);
      prepare(inputs);
      model.submit_evaluation(*session);
      const std::array selections{q::MtpGenerationSelection{session,pending,4,{},3}};
      const std::array outputs{q::EvaluationOutput{.session=session}};
      std::array<q::MtpRoundResult,1>decisions;
      model.finish_recursive_mtp(selections,{},outputs,decisions,[](std::size_t,const q::CompletedEvaluationView&){});
      result.states.push_back(s::image(model,*session->head));
      result.emitted.push_back(decisions[0].emitted);
      require(s::image(model,*old)==before,"natural continuation changed old checkpoint");
    }
    old.reset();qwen_host_image_case::retire(session);
    require(model.snapshot_versions.live==0 && !model.active_step && !model.carrier_borrowed,
        "case retained version/admission after retirement");
  }
  if(fused)require(model.draft_chain_graph_calls>0,"contract never exercised compound draft Graph");
  require(model.private_publications_retired(),"contract retained private Graph publication");
  return result;
}
qwen_device_continuation_case::Result continuous_chain(const std::filesystem::path&input,std::size_t steps,bool device,
    const qwen_device_continuation_case::Result*expected=nullptr){
  namespace d=qwen_device_continuation_case;
  auto source=q::prepare_weight_source(input/"weights.tsv",true);const auto geometry=source.roles.geometry;
  const auto width=steps+1;
  const auto plan=q::cache::plan(geometry,source.cache_row_bytes,32,1,4,
      q::recurrent_bytes(q::state_schema_for(geometry,32),geometry),1,width,true,false,3,2,{},steps);
  q::Model model(std::move(source),geometry,32,plan,{},false,q::ReplayPolicy::kPreferCapture,SIZE_MAX,{},0,{.primary=q::BackendKind::kCuda});
  model.configure_draft_policy({steps,0});if(device)model.configure_device_mtp_successor(true);
  const std::array seed{std::size_t{3}},rows{width};
  model.prepare_family(seed,false,1,q::GraphRole::kMtpPrefill);model.plan_replay_uses(3,1,64,q::GraphRole::kMtpPrefill);
  for(const auto role:{q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound}){
    model.prepare_family(rows,false,1,role);model.plan_replay_uses(width,1,64,role);
  }
  if(device){model.prepare_family(rows,false,1,q::GraphRole::kMtpContinuation);model.plan_replay_uses(width,1,64,q::GraphRole::kMtpContinuation);}
  model.prepare_shared_execution_workspace();if(device)(void)model.prepare_mtp_decode_recordings(0);
  static constexpr std::array<std::int32_t,3>prompt{1,2,3};
  const auto prefill=[&]{auto session=model.create_session(seed,{},q::GraphRole::kMtpPrefill);
    model.prepare_evaluation(*session,prompt,nullptr,q::GraphRole::kMtpPrefill);model.submit_evaluations();
    (void)q::finish_evaluation(model,*session,"multi-capacity-seed",{},false);return session;};
  if(device){
    auto session=prefill();auto old=session->checkpoint();const auto before=s::image(model,*old);
    const std::array pending{session->head->next_id};model.prepare_evaluation(*session,pending,nullptr,d::role(*session));
    auto a=model.prepare_submission_sources();std::weak_ptr<q::StepState>alife=model.active_step;model.submit_evaluations();
    const std::array future{q::EvaluationRequest{session,{}}};
    auto&child=model.prepare_relay_evaluations(future,a,0,q::GraphRole::kMtpContinuation);
    auto b=model.prepare_submission_sources(child);std::weak_ptr<q::StepState>blife=model.successor_step;
    model.submit_evaluations(child);require(model.active_step->submitted && child.submitted,"withdrawal missed both actual submissions");
    require(s::image(model,*old)==before && s::image(model,*session->head)==before,"unobserved successor published before withdrawal");
    a.reset();b.reset();const auto retired=model.abort_episode();q::check(retired.status);
    require(retired.retired && alife.expired() && blife.expired() && !model.active_step && !model.successor_step,
        "multi-stage withdrawal did not retire both original Source/Append/Store episodes");
    require(s::image(model,*old)==before && s::image(model,*session->head)==before,"withdrawal changed old state, KV or RNG");
    old.reset();qwen_host_image_case::retire(session);model.restore_mtp_decode_recordings(0);
    require(!model.snapshot_versions.live && model.private_publications_retired(),"withdrawal did not recover its original cold capture inventory");
  }
  auto session=prefill();q::GenerationLedger ledger(6,width);d::Result result;
  const auto draft_begin=model.submitted_draft_rows;
  std::vector<std::weak_ptr<q::StepState>>lifetimes;
  hyle::util::ScopeGuard cleanup{[&]()noexcept{(void)model.abort_episode();if(session){session->stop().IgnoreError();
    (void)session->drain();(void)session->try_retire();session.reset();}}};
  const auto stage=[&](std::size_t quota){const auto pending=session->head->next_id;
    q::prepare_generation_record(ledger,*session,d::role(*session),std::span(&pending,1),true,quota,0,{},steps);};
  std::shared_ptr<q::StepSources>current_source;
  if(device){stage(2*width);const std::array pending{session->head->next_id};
    model.prepare_evaluation(*session,pending,nullptr,d::role(*session));lifetimes.push_back(model.active_step);
    current_source=model.prepare_submission_sources();model.submit_evaluations();}
  for(std::size_t index=0;index<6;++index){
    const bool has_child=index+1<6;const auto quota=has_child?2*width:width;
    const auto pending=session->head->next_id;const auto before=q::generation_boundary(*session->head);
    if(!device){stage(quota);model.prepare_evaluation(*session,std::span(&pending,1),nullptr,d::role(*session));
      lifetimes.push_back(model.active_step);model.submit_evaluations();}
    q::StepState*child=nullptr;std::shared_ptr<q::StepSources>child_source;
    if(device && has_child){
      require(current_source && current_source->draft_stage_sources.size()==2*steps,"producer lost private-stage Source reader bundles");
      for(const auto&source_view:current_source->draft_stage_sources)require(source_view.valid(),"private-stage Source was not captured");
      (void)ledger.reserve_successor(ledger.committed_count,width,0);
      const auto slot=index?(model.active_step->state_slot+1)%q::execution_slot_count:0;
      const std::array future{q::EvaluationRequest{session,{}}};
      child=&model.prepare_relay_evaluations(future,current_source,slot,q::GraphRole::kMtpContinuation);
      lifetimes.push_back(model.successor_step);
      require(child->draft_rows_deferred && !child->active_draft_rows && child->capture_reused && !child->capture_recorded,
          "multi-stage successor observed its parent or missed original capture reuse");
      if(index+2<6)child_source=model.prepare_submission_sources(*child);
      model.submit_evaluations(*child);
      require(model.active_step->submitted && child->submitted && q::generation_boundary(*session->head)==before &&
          ledger.records[ledger.committed_count+1].input_deferred,"successor did not precede host selection/publication");
    }
    const auto round=d::finish(model,session,ledger,pending,quota);d::append(result,round);
    if(device && has_child){
      require(model.active_step.get()==child && child->members[0].selected_input_choice==round.processed_target-1 &&
          child->members[0].draft_input_rows==steps+session->head->draft->lag && !child->draft_rows_deferred,
          "multi-stage successor selected the wrong canonical prefix or active work");
      (void)ledger.bind_successor(q::generation_boundary(*session->head),ledger.committed_count,0,round.emitted.back(),true);
      if(index+2<6)(void)ledger.renew_pending_quota(q::generation_boundary(*session->head),ledger.committed_count,0,
          2*width*(6-index-1),width,width);
      current_source=std::move(child_source);
    }
  }
  current_source.reset();require(std::ranges::all_of(lifetimes,[](const auto&x){return x.expired();}),"continuous chain retained an original producer");
  result.records.assign(ledger.records.begin(),ledger.records.begin()+ledger.committed_count);
  result.final_state=s::image(model,*session->head);result.draft_rows=model.submitted_draft_rows-draft_begin;
  std::size_t rows_sum=0;for(const auto&record:result.records)rows_sum+=record.draft_rows;
  require(rows_sum==result.draft_rows,"continuous chain actual work differs from original committed ledger");
  if(expected){d::compare_records(*expected,result);d::compare_state(model,*expected->final_state,*result.final_state);}
  qwen_host_image_case::retire(session);require(!model.snapshot_versions.live && !model.active_step && !model.successor_step &&
      model.draft_inventory.free_ids.size()==model.draft_inventory.owners.size() &&
      model.page_inventory.free_ids.size()==model.page_inventory.owners.size(),"continuous chain retained native or policy page owners");
  cleanup.dismiss();return result;
}
}
int main(int argc,char**argv)try{
  cold_policy_facts();
  if(argc==1){std::cerr<<"SKIP: pass prepared local C32 MTP inputs\n";return 77;}
  if(argc!=2)throw std::runtime_error("usage: qwen_mtp_chain_test PREPARED_LOCAL_INPUTS");
  zero_draft_cold(argv[1]);
  require(qwen_device_continuation_case::run(argv[1],q::BackendKind::kCuda)==0,
      "generic prefix selection failed the original K1 continuation contract");
  auto original=run(argv[1],false),fused=run(argv[1],true),compound=run(argv[1],true,true),
      parallel=run(argv[1],true,true,true),restarted=run(argv[1],true,true,true,true);
  for(const auto steps:{2u,3u}){const auto serial=continuous_chain(argv[1],steps,false);
    (void)continuous_chain(argv[1],steps,true,&serial);}
  for(const auto *candidate:{&fused,&compound,&parallel,&restarted}){
    require(original.emitted==candidate->emitted,"natural continuation IDs differ");
    require(original.states.size()==candidate->states.size(),"state count differs");
    for(std::size_t index=0;index<original.states.size();++index){
      if(original.states[index]!=candidate->states[index]){
        std::cerr<<"state byte mismatch at controlled/continuation boundary "<<index<<'\n';return 1;
      }
    }
  }
  std::cout<<"PASS original Commands versus captured draft and complete round CUDA Graphs: 8 controlled lag0/lag1 x prefix1..4 cases, "
    <<original.states.size()<<" exact complete state boundaries, 16 natural continuations, "
      "private-chain packet-export failure/retry, submitted-verifier withdrawal/retry, "
      "unchanged checkpoints/RNG and real final retirement; K2/K3 six-round continuous chains match serial final state and ordered ledgers, "
      "unobserved A/B withdrawal preserves snapshots and retries original captures; caller-owned auxiliary construction overlaps "
      "original cold family workers, and rejected cold auxiliary work joins/discards pending families before a serial retry; "
      "both preserve the same complete states; configured zero-draft dynamic policies retain their original cold programs.\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
