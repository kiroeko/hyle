#pragma once

#include "hyle/compute/backends/cuda.h"

// Included after the real model implementation by qwen_request_lifecycle_test.cc.
namespace qwen_dual_model_case {
namespace q=qwen35;
namespace i=hyle::inference;
namespace {
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
struct Endpoint final {
  std::uintptr_t wrapper=0,context_owner=0,queue_wrapper=0,primary=0,stream=0,cold_resources=0;
  int device_ordinal=0;
  friend bool operator==(const Endpoint &,const Endpoint &)=default;
};
Endpoint endpoint(q::Model &model){
  auto *context=q::take(model.execution.common_context());auto *queue=q::take(model.execution.common_queue());
  return {reinterpret_cast<std::uintptr_t>(context),
      reinterpret_cast<std::uintptr_t>(model.execution.storage_owner().context_identity()),
      reinterpret_cast<std::uintptr_t>(queue),
      reinterpret_cast<std::uintptr_t>(q::take(q::c::cuda::primary_context(*context))),
      reinterpret_cast<std::uintptr_t>(q::take(q::c::cuda::stream(*queue))),
      reinterpret_cast<std::uintptr_t>(model.resources_.get()),q::take(q::c::cuda::device_ordinal(context->device()))};
}
void normal_retire(q::GenerationRequest &request){
  (void)request.session->stop();const auto drained=request.session->drain();
  const auto retired=request.session->try_retire();if(retired.retired)request.session.reset();
  q::check(drained.status);q::check(retired.status);require(retired.retired,"normal payload did not retire");
}
void expected_cancel(const hyle::Status &status){
  if(!status.IsCancelled()||status.has_native_error())q::check(status);
}
}
inline int run(const std::filesystem::path &input){try{
  const auto geometry=q::read_weight_geometry(input/"weights.tsv");
  const auto all=q::read_ids(input/"prompt.ids",geometry.vocabulary,192);require(all.size()==154,"expected original P154 input");
  const std::vector<std::int32_t> prompt(all.begin(),all.begin()+15);
  const auto plan=q::cache::plan(geometry,32,1,15,q::recurrent_bytes(q::state_schema_for(geometry,32),geometry),1,1,false,true);
  const auto make_model=[&]{
    return std::make_unique<q::Model>(input/"weights.tsv",geometry,32,plan,std::filesystem::path{},true,
        q::ReplayPolicy::kCommands,268435456,q::Model::Sampling{.enabled=true,.policy={1,8,.9F},.seed=123},
        0,q::BackendSelection{.primary=q::BackendKind::kCuda});
  };
  std::cerr<<"dual_model_stage=load_A\n";auto model_a=make_model();
  std::cerr<<"dual_model_stage=load_B\n";auto model_b=make_model();
  const auto a_endpoint=endpoint(*model_a),b_endpoint=endpoint(*model_b);
  require(a_endpoint.wrapper!=b_endpoint.wrapper&&a_endpoint.context_owner!=b_endpoint.context_owner&&
      a_endpoint.queue_wrapper!=b_endpoint.queue_wrapper&&a_endpoint.stream&&b_endpoint.stream&&a_endpoint.stream!=b_endpoint.stream&&
      a_endpoint.cold_resources!=b_endpoint.cold_resources,"models do not own independent HYLE Context/Queue/ColdResources");
  require(a_endpoint.device_ordinal==0&&b_endpoint.device_ordinal==0,"models did not use the same requested CUDA device");
  require(model_a->source_identity==model_b->source_identity&&model_a->math_identity()!=model_b->math_identity(),"source and model identities were conflated");
  for(auto *model:{model_a.get(),model_b.get()}){
    require(model->physical_text_roles==320&&!model->physical_mtp_roles&&!model->cache_plan.draft_cache,"model loaded draft state/weights");
    const std::array shapes{std::size_t{15},std::size_t{1}};model->prepare_family(shapes);
    for(auto rows:shapes)model->plan_replay_uses(rows,1,4);model->prepare_execution(shapes);
  }
  const std::array turns{i::GenerationTurn{prompt,3}};
  q::GenerationRequest a("A",prompt,{},3),b("B",prompt,{},3);
  for(auto *request:{&a,&b}){
    request->progress.bind_turns(turns);request->progress.initial_sampling={123,request==&a?100ULL:101ULL,0};
    request->ledger=std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns),1);
    request->round_emitted.resize(2);for(auto &row:request->round_emitted)row.reserve(1);
    request->turn_observations.resize(1);request->warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt));
    request->prompt=request->progress.begin_turn(*request->warm_inputs[0]);
  }
  const auto prepare=[&](q::Model &model,q::GenerationRequest &request){return q::prepare_generation_call(model,request,request.progress.output_limit(),nullptr,{},1);};
  const auto execute=[&](q::Model &model,q::GenerationRequest &request){std::array calls{prepare(model,request)};q::execute_generation_calls(model,calls,false);};
  std::cerr<<"dual_model_stage=prefill_A_B\n";execute(*model_a,a);execute(*model_b,b);
  require(a.session->head->sampling.draw==1&&b.session->head->sampling.draw==1&&a.ledger->committed_count==1&&b.ledger->committed_count==1,"independent prefill did not commit once");
  const auto a_boundary=q::generation_boundary(*a.session->head),b_boundary=q::generation_boundary(*b.session->head);
  const auto *head_a=a.session->head.get(),*head_b=b.session->head.get();
  bool foreign_rejected=false;
  try{const auto pending=a.session->head->next_id;model_b->prepare_evaluation(*a.session,std::span(&pending,1));}
  catch(const std::runtime_error &error){foreign_rejected=std::string_view(error.what())=="batch session belongs to another model";}
  require(foreign_rejected&&!model_a->active_step&&!model_b->active_step&&a.session->head.get()==head_a&&b.session->head.get()==head_b,"foreign Model preparation did not fail without effects");
  auto a_call=prepare(*model_a,a),b_call=prepare(*model_b,b);
  for(auto *call:{&a_call,&b_call})q::prepare_generation_record(*call->request->ledger,*call->request->session,q::GraphRole::kText,
      call->ids(),true,call->quota,0,{},0,call->completion,call->input_begin,call->prefill_complete);
  std::cerr<<"dual_model_stage=submit_independent_A_B\n";
  model_a->prepare_evaluation(*a.session,a_call.ids());model_a->submit_evaluation(*a.session);
  model_b->prepare_evaluation(*b.session,b_call.ids());model_b->submit_evaluation(*b.session);
  require(model_a->active_step&&model_b->active_step&&model_a->active_step.get()!=model_b->active_step.get()&&
      a.session->pending==model_a->active_step.get()&&b.session->pending==model_b->active_step.get(),"two Models did not retain independent original episodes");
  const bool a_inflight=model_a->active_step->run->in_flight(),b_inflight=model_b->active_step->run->in_flight();
  auto *b_pending=b.session->pending;std::weak_ptr<q::ColdResources> a_resources=model_a->resources_;
  std::cerr<<"dual_model_stage=cancel_retire_destroy_A\n";
  const auto stop=a.session->stop();expected_cancel(stop);const auto abort=model_a->abort_episode();expected_cancel(abort.status);require(abort.retired,"A episode did not actually retire");
  require(!model_a->active_step&&!a.session->pending&&a.session->head.get()==head_a&&q::generation_boundary(*a.session->head)==a_boundary&&
      a.ledger->committed_count==1&&!a.ledger->records[1].committed,"cancelled A published an unobserved result");
  const auto retired_a=a.session->try_retire();if(retired_a.retired)a.session.reset();
  expected_cancel(retired_a.status);require(retired_a.retired,"A did not actually retire");model_a.reset();
  const bool a_scope_released=a_resources.expired();
  require(a_scope_released,"ModelA ColdResources still retained after actual retirement/destruction");
  require(model_b->active_step&&b.session->pending==b_pending&&b.session->head.get()==head_b&&q::generation_boundary(*b.session->head)==b_boundary&&
      b.ledger->committed_count==1&&!b.ledger->records[1].committed&&b.progress.generated.size()==1&&b.session->retirement_status.ok()&&endpoint(*model_b)==b_endpoint,
      "A teardown invalidated B original carrier/head/ledger/RNG/endpoint");
  std::cerr<<"dual_model_stage=finish_B_submitted_before_A_destruction\n";
  (void)q::finish_evaluation(*model_b,*b.session,"Bdecode1",{},false,&b_call.step_seconds,&b.endpoint,nullptr,nullptr,{},b.ledger.get());
  q::finish_generation_call(*model_b,b_call);
  require(b.ledger->committed_count==2&&b.progress.observed_commits()==2&&b.progress.generated.size()==2&&b.session->head->sampling.draw==2&&
      b.ledger->records[1].pending_input==b_boundary.next_id&&b.session->length()==16,"B original submitted episode failed after A destruction");
  const auto pending=b.session->head->next_id;std::cerr<<"dual_model_stage=B_new_submission_after_A_destruction\n";execute(*model_b,b);
  require(b.progress.request_done()&&b.ledger->committed_count==3&&b.progress.observed_commits()==3&&b.session->head->sampling.draw==3&&
      b.session->length()==17&&b.ledger->records[2].pending_input==pending&&b.session->retirement_status.ok()&&endpoint(*model_b)==b_endpoint,
      "B could not continue on its original resources after A destruction");
  const auto ids=b.progress.generated;const auto weight_bytes=model_b->device_weight_bytes;normal_retire(b);model_b.reset();
  std::cout<<"{\"status\":\"PASS\",\"ordinary\":true,\"models\":2,\"physical_text_roles_each\":320,\"physical_mtp_roles_each\":0,\"weight_bytes_each\":"<<weight_bytes
      <<",\"cache_limit_each\":268435456,\"P\":15,\"C\":32,\"N_each\":1,\"R_each\":1,\"independent_HYLE_contexts\":true,\"independent_native_streams\":true,\"independent_ColdResources\":true,\"same_CUDA_primary_context\":"<<(a_endpoint.primary==b_endpoint.primary?"true":"false")
      <<",\"primary_A\":"<<a_endpoint.primary<<",\"primary_B\":"<<b_endpoint.primary<<",\"stream_A\":"<<a_endpoint.stream<<",\"stream_B\":"<<b_endpoint.stream
      <<",\"HYLE_context_A\":"<<a_endpoint.context_owner<<",\"HYLE_context_B\":"<<b_endpoint.context_owner<<",\"device_ordinal_A\":"<<a_endpoint.device_ordinal<<",\"device_ordinal_B\":"<<b_endpoint.device_ordinal
      <<",\"A_logical_inflight_at_cancel\":"<<(a_inflight?"true":"false")<<",\"B_logical_inflight_before_A_teardown\":"<<(b_inflight?"true":"false")<<",\"physical_GPU_completion_at_A_teardown\":\"unknown\",\"A_scope_released\":"<<(a_scope_released?"true":"false")
      <<",\"foreign_Model_rejected\":true,\"A_committed_records\":1,\"A_draw\":1,\"B_committed_records\":3,\"B_draw\":3,\"B_ids\":[";
  for(std::size_t at=0;at<ids.size();++at)std::cout<<(at?",":"")<<ids[at];std::cout<<"]}\n";return 0;
}catch(const std::exception &error){std::cerr<<"dual_model_failure="<<error.what()<<'\n';return 1;}}

} // namespace qwen_dual_model_case
