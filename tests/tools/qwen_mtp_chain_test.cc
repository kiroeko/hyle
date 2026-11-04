#include "../../engine/models/qwen35/main.cc"
#include "qwen_host_image_case.h"
#include "qwen_mtp_batch_state_case.h"

namespace {
namespace q=qwen35;
namespace s=qwen_mtp_batch_state_case;
using qwen_host_image_case::require;
struct Result {
  std::vector<s::StateImage> states;
  std::vector<std::vector<std::int32_t>> emitted;
};
Result run(const std::filesystem::path &input,bool fused) {
  auto source=q::prepare_weight_source(input/"weights.tsv",true);
  const auto geometry=source.roles.geometry;
  require(geometry.context_limit>=32 && geometry.vocabulary>=16,"contract needs existing C32 fixture");
  const auto plan=q::cache::plan(geometry,source.cache_row_bytes,32,4,4,
      q::recurrent_bytes(q::state_schema_for(geometry,32),geometry),1,4,true,false,3,2);
  q::Model model(std::move(source),geometry,32,plan,{},false,
      fused?q::ReplayPolicy::kPreferCapture:q::ReplayPolicy::kCommands,SIZE_MAX,{},0,{.primary=q::BackendKind::kCuda});
  model.configure_draft_policy({3,0});
  const std::array seed_rows{std::size_t{3}};
  model.prepare_family(seed_rows,false,1,q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3,1,32,q::GraphRole::kMtpPrefill);
  for(std::size_t width=2;width<=4;++width){
    const std::array rows{width};model.prepare_family(rows,false,1,q::GraphRole::kVerify);
    model.plan_replay_uses(width,1,64,q::GraphRole::kVerify);
  }
  model.prepare_recursive_round_programs(0,false,64);
  Result result;
  for(const bool lag:{false,true})for(std::size_t accepted=1;accepted<=4;++accepted){
    auto session=model.create_session(seed_rows,{},q::GraphRole::kMtpPrefill);
    const std::array<std::int32_t,3>prompt{1,2,3};
    model.prepare_evaluation(*session,prompt,nullptr,q::GraphRole::kMtpPrefill);
    model.submit_evaluation(*session);
    (void)q::finish_evaluation(model,*session,"seed",{},false);
    const auto controlled=[&](std::size_t count){
      const std::array<std::int32_t,4>ids{session->head->next_id,4,5,6};
      model.prepare_evaluation(*session,ids,nullptr,q::GraphRole::kVerify);
      require(model.active_step->phase->recursive_draft_repair==fused,"test missed requested real repair route");
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
      const std::array<std::int32_t,4>ids{session->head->next_id,4,5,6};
      model.prepare_evaluation(*session,ids,nullptr,q::GraphRole::kVerify);
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
      model.prepare_evaluation(*session,inputs,nullptr,q::GraphRole::kVerify);
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
}
int main(int argc,char**argv)try{
  if(argc==1){std::cerr<<"SKIP: pass prepared local C32 MTP inputs\n";return 77;}
  if(argc!=2)throw std::runtime_error("usage: qwen_mtp_chain_test PREPARED_LOCAL_INPUTS");
  auto original=run(argv[1],false),fused=run(argv[1],true);
  require(original.emitted==fused.emitted,"natural continuation IDs differ");
  require(original.states.size()==fused.states.size(),"state count differs");
  for(std::size_t index=0;index<original.states.size();++index){
    if(original.states[index]!=fused.states[index]){
      std::cerr<<"state byte mismatch at controlled/continuation boundary "<<index<<'\n';return 1;
    }
  }
  std::cout<<"PASS original Commands versus compound captured CUDA: 8 controlled lag0/lag1 x prefix1..4 cases, "
    <<original.states.size()<<" exact complete state boundaries, 16 natural continuations, "
      "private-chain packet-export failure/retry, submitted-verifier withdrawal/retry, "
      "unchanged checkpoints/RNG and real final retirement.\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
