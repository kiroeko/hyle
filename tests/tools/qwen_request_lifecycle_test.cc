#include "../../engine/models/qwen35/main.cc"
#include "../../engine/inference/output_delivery.h"
#include <cstdlib>
#include <memory>
#include <string>
#include "qwen_chunked_prefill_case.h"
#include "qwen_dual_model_case.h"
#include "qwen_sampling_policy_case.h"
#include "qwen_logits_modifier_case.h"
#include "qwen_turn_context_case.h"
#include "qwen_retirement_observation_case.h"
#include "qwen_midturn_resume_case.h"
#include "qwen_participant_failure_case.h"
#include "qwen_request_stop_case.h"
#include "qwen_sampling_budget_case.h"
#include "qwen_request_ledger_case.h"
#include "qwen_e4_cache_case.h"
namespace q = qwen35;
namespace i = hyle::inference;
namespace {
std::string input_directory_from_environment(const char *name = "HYLE_QWEN_REQUEST_INPUTS") {
#if defined(_MSC_VER)
  char *value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, name))
    throw std::runtime_error("cannot read " + std::string(name));
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto *value = std::getenv(name);
  return value ? value : "";
#endif
}
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
void retirement_status_contract() {
  const auto cancelled = hyle::Status::Cancelled("requested episode cancellation");
  const auto failure = hyle::Status::Internal("retirement failure");
  const auto native = hyle::Status::FromNative(hyle::StatusCode::Unavailable,
      {hyle::NativeErrorDomain::CudaDriver, 999}, "synthetic native failure");
  const auto native_cancelled = hyle::Status::FromNative(hyle::StatusCode::Cancelled,
      {hyle::NativeErrorDomain::CudaDriver, 999}, "synthetic native cancellation failure");
  auto status = hyle::Status::Ok();
  i::record_retirement_status(status, cancelled);
  require(status == cancelled, "initial cancellation was lost");
  i::record_retirement_status(status, hyle::Status::Ok());
  require(status == cancelled, "success cleared prior cancellation");
  i::record_retirement_status(status, failure);
  require(status == failure, "prior cancellation hid retirement failure");
  i::record_retirement_status(status, cancelled);
  require(status == failure, "later cancellation replaced real failure");
  status = cancelled; i::record_retirement_status(status, native);
  require(status == native && status.has_native_error(), "prior cancellation hid native failure");
  status = cancelled; i::record_retirement_status(status, native_cancelled);
  require(status == native_cancelled && status.has_native_error(), "native cancelled code was mistaken for a normal cancellation");
  i::record_retirement_status(status, failure);
  require(status == native_cancelled, "later failure replaced an already native first error");
}
void event(const char *name){std::cerr<<"live_owner_stage="<<name<<'\n';}
void finish_payload(q::GenerationRequest &request){
  const auto stopped=request.session->stop();
  std::cerr<<"live_owner_payload_stop id="<<request.id<<" ok="<<stopped.ok()<<" cancelled="<<stopped.IsCancelled()<<" message="<<stopped.message()<<'\n';
  const auto drained=request.session->drain();const auto retired=request.session->try_retire();
  if(retired.retired)request.session.reset();q::check(stopped);q::check(drained.status);q::check(retired.status);
  require(retired.retired,"request payload did not actually retire");
}
}
int main(int argc,char **argv){
 try{
  if(argc==2 && std::string_view(argv[1])=="--retirement-status-only") {
    retirement_status_contract();
    std::cout<<"retirement status: 7 host checks passed; no device error injected\n";return 0;
  }
  const bool chunked=argc>1 && std::string_view(argv[1])=="--chunked-prefill";
  const bool dual=argc>1 && std::string_view(argv[1])=="--dual-model";
  const bool sampling=argc>1 && std::string_view(argv[1])=="--sampling-policies";
  const bool logits=argc>1 && std::string_view(argv[1])=="--logits-modifiers";
  const bool turn_context=argc>1 && std::string_view(argv[1])=="--turn-context";
  const bool retirement=argc>1 && std::string_view(argv[1])=="--retirement-observation";
  const bool midturn=argc>1 && std::string_view(argv[1])=="--midturn-resume";
  const bool participant_failure=argc>1 && std::string_view(argv[1])=="--participant-failure";
  const bool stop_policy=argc>1 && std::string_view(argv[1])=="--stop-policies";
  const bool sampling_budget=argc>1 && std::string_view(argv[1])=="--sampling-budget";
  const bool request_ledger=argc>1 && std::string_view(argv[1])=="--request-ledger";
  const bool e4_cache=argc>1 && std::string_view(argv[1])=="--e4-cache";
  const int argument=(chunked||dual||sampling||logits||turn_context||retirement||midturn||participant_failure||stop_policy||sampling_budget||request_ledger||e4_cache)?2:1;
  if(argc>argument+1)throw std::runtime_error("usage: qwen_request_lifecycle_test [--chunked-prefill|--dual-model|--sampling-policies|--logits-modifiers|--turn-context|--retirement-observation|--midturn-resume|--participant-failure|--stop-policies|--sampling-budget|--request-ledger|--e4-cache] [prepared_input_directory]");
  const char *input_environment=e4_cache?"HYLE_QWEN_E4_INPUTS":"HYLE_QWEN_REQUEST_INPUTS";
  const std::string input_directory=argc==argument+1?argv[argument]:input_directory_from_environment(input_environment);
  if(input_directory.empty()){std::cerr<<"SKIP: pass prepared input directory or set "<<input_environment<<'\n';return 77;}
  const std::filesystem::path input=input_directory;
  if(chunked)return qwen_chunked_prefill_case::run(input);
  if(dual)return qwen_dual_model_case::run(input);
  if(sampling)return qwen_sampling_policy_case::run(input);
  if(logits)return qwen_logits_modifier_case::run(input);
  if(turn_context)return qwen_turn_context_case::run(input);
  if(retirement)return qwen_retirement_observation_case::run(input);
  if(midturn)return qwen_midturn_resume_case::run(input);
  if(participant_failure)return qwen_participant_failure_case::run(input);
  if(stop_policy)return qwen_request_stop_case::run(input);
  if(sampling_budget)return qwen_sampling_budget_case::run(input);
  if(request_ledger)return qwen_request_ledger_case::run(input);
  if(e4_cache)return qwen_e4_cache_case::run(input);
  const auto geometry=q::read_weight_geometry(input/"weights.tsv");
  const auto prompt=q::read_ids(input/"prompt.ids",geometry.vocabulary,192);require(prompt.size()==154,"expected original P154 prompt");
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan=q::cache::plan(geometry, cache_rows,192,3,154,q::recurrent_bytes(q::state_schema_for(geometry,192),geometry),2,1,false,true);
  event("load_original_ordinary");
  q::Model model(input/"weights.tsv",geometry,192,plan,{},true,q::ReplayPolicy::kCommands,1073741824,
      {.enabled=true,.policy={1,8,.9F},.seed=123},0,{.primary=q::BackendKind::kCuda});
  require(model.physical_text_roles==320 && !model.physical_mtp_roles && !model.cache_plan.draft_cache,"ordinary loaded draft resources");
  const std::array shapes{std::size_t{154},std::size_t{1}};
  for(auto width:{std::size_t{1},std::size_t{2}}){
    model.prepare_family(shapes,false,width,q::GraphRole::kText);
    model.plan_replay_uses(154,width,4,q::GraphRole::kText);model.plan_replay_uses(1,width,16,q::GraphRole::kText);
  }
  for(auto width:{std::size_t{1},std::size_t{2}})model.prepare_execution(shapes,width,q::GraphRole::kText);
  const std::array a_turns{i::GenerationTurn{prompt,3}};
  const std::array b_turns{i::GenerationTurn{prompt,3},i::GenerationTurn{{},1}};
  const std::array c_turns{i::GenerationTurn{prompt,1}};
  std::vector<q::GenerationRequest> requests;requests.reserve(3);
  requests.emplace_back("A",prompt,std::filesystem::path{},3);requests.emplace_back("B",prompt,std::filesystem::path{},4);requests.emplace_back("C",prompt,std::filesystem::path{},1);
  requests[0].progress.bind_turns(a_turns);requests[1].progress.bind_open_turns(std::span(b_turns).first(1),2,154);requests[2].progress.bind_turns(c_turns);
  for(std::size_t index=0;index<requests.size();++index){
    auto &request=requests[index];const auto outputs=request.progress.maximum_outputs();
    request.progress.initial_sampling={123,100+index,0};
    request.ledger=std::make_unique<q::GenerationLedger>(154+outputs,1);request.round_emitted.resize(outputs-1);
    for(auto &row:request.round_emitted)row.reserve(1);
    request.turn_observations.resize(index==1?2:1);request.warm_inputs.reserve(index==1?2:1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt));
    if(index==1)request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(1,-1));
  }
  i::RequestWindow window;window.prepare(3,2);
  std::array delivery{i::OutputDelivery(1),i::OutputDelivery(1),i::OutputDelivery(1)};
  auto &a=requests[0];auto &b=requests[1];auto &c=requests[2];
  a.request_key=window.enqueue(0);b.request_key=window.enqueue(1);
  const auto key_a=a.request_key,key_b=b.request_key;
  const auto ack=[&](std::size_t index,i::RequestKey key,std::size_t through){
    auto &request=requests[index];if(key!=request.request_key || !window.matches(key))return false;
    delivery[index].acknowledge_prefix(request.progress.generated.size(),through);return true;
  };
  const auto prepare=[&](std::size_t index){
    auto &request=requests[index];const auto credit=delivery[index].available(request.progress.generated.size());
    require(credit==1,"ordinary episode must have exactly one real output credit");
    return q::prepare_generation_call(model,request,request.progress.output_limit(),nullptr,requests,credit);
  };
  for(auto index:{std::size_t{0},std::size_t{1}}){
    auto &request=requests[index];window.admitted(request.request_key);request.prompt=request.progress.begin_turn(*request.warm_inputs[0]);
  }
  event("original_R2_prefill");std::array calls{prepare(0),prepare(1)};
  q::execute_generation_calls(model,calls,false,nullptr,requests);
  require(!model.active_step && a.ledger->committed_count==1 && b.ledger->committed_count==1,"R2 prefill was not observed atomically");
  require(a.session->head->sampling.draw==1 && b.session->head->sampling.draw==1 && !a.progress.turn_done && !b.progress.turn_done,"natural prefill did not leave two live random requests");
  require(ack(0,key_a,1) && ack(1,key_b,1),"initial actual transport acknowledgements failed");
  const auto before_a=q::generation_boundary(*a.session->head),before_b=q::generation_boundary(*b.session->head);
  const auto *head_a=a.session->head.get(),*head_b=b.session->head.get();
  const auto base_b_outputs=b.progress.generated;const auto base_b_state_count=b.session->head->state.size();
  auto cancelled_a=prepare(0),cancelled_b=prepare(1);
  for(auto *call:{&cancelled_a,&cancelled_b})q::prepare_generation_record(*call->request->ledger,*call->request->session,q::GraphRole::kText,call->ids(),true,call->quota,0);
  const std::array batch{q::EvaluationRequest{a.session,cancelled_a.ids()},q::EvaluationRequest{b.session,cancelled_b.ids()}};
  event("submit_R2_then_cancel_A");model.prepare_evaluations(batch,nullptr,q::GraphRole::kText);model.submit_evaluations();
  require(model.active_step && model.active_step->submitted && a.session->pending==b.session->pending && model.active_step->members.size()==2,"original submitted R2 episode was not shared");
  const bool logical_inflight=model.active_step->run->in_flight();
  const auto cancel_status=a.session->stop();
  std::cerr<<"live_owner_cancel_status stop_ok="<<cancel_status.ok()<<" stop_cancelled="<<cancel_status.IsCancelled()<<" message="<<cancel_status.message()<<'\n';
  if(!cancel_status.IsCancelled())q::check(cancel_status);
  const auto abort_status=model.abort_episode();
  std::cerr<<"live_owner_cancel_status abort_ok="<<abort_status.status.ok()<<" abort_cancelled="<<abort_status.status.IsCancelled()<<" message="<<abort_status.status.message()<<'\n';
  if(!abort_status.status.IsCancelled()||abort_status.status.has_native_error())q::check(abort_status.status);
  require(abort_status.retired,"cancelled shared native episode did not actually retire");
  require(!model.active_step && !model.carrier_borrowed && !a.session->pending && !b.session->pending,"cancelled shared native episode did not completely retire");
  require(a.session->head.get()==head_a && b.session->head.get()==head_b && q::generation_boundary(*a.session->head)==before_a && q::generation_boundary(*b.session->head)==before_b && b.session->head->state.size()==base_b_state_count,"cancelled R2 changed a committed full head or random cursor");
  require(a.ledger->committed_count==1 && b.ledger->committed_count==1 && !a.ledger->records[1].committed && !b.ledger->records[1].committed && b.progress.generated==base_b_outputs && b.progress.observed_commits()==1 && !b.session->stopping && !b.session->failed,"cancelled batch published or stopped its surviving request");
  const bool survivor_retirement_ok=b.session->retirement_status.ok();
  std::cerr<<"live_owner_survivor_status ok="<<survivor_retirement_ok<<" cancelled="<<b.session->retirement_status.IsCancelled()<<" message="<<b.session->retirement_status.message()<<'\n';
  const auto retired_a=a.session->try_retire();
  if(retired_a.retired){a.session.reset();window.retire_active(key_a);window.release(key_a);}
  if(!retired_a.status.IsCancelled()||retired_a.status.has_native_error())q::check(retired_a.status);
  require(retired_a.retired,"cancelled A payload was not retired");
  window.requeue(key_b);require(window.waiting_front()==key_b && window.matches(key_b),"survivor lost its original key or FIFO continuation");window.admitted(key_b);
  event("invalid_C_request_local_preflight");c.request_key=window.enqueue(2);const auto old_c_key=c.request_key;window.admitted(old_c_key);
  const std::array chunks{std::size_t{154}};c.session=model.create_session(chunks,{},q::GraphRole::kText,c.progress.initial_sampling);
  const auto *c_empty=c.session->head.get();const auto visible_before_error=q::generation_boundary(*b.session->head);
  const std::int32_t pending=b.session->head->next_id,bad_id=static_cast<std::int32_t>(geometry.vocabulary);
  const std::array bad_batch{q::EvaluationRequest{b.session,std::span(&pending,1)},q::EvaluationRequest{c.session,std::span(&bad_id,1)}};
  bool input_rejected=false;
  try{model.prepare_evaluations(bad_batch,nullptr,q::GraphRole::kText);}catch(const std::runtime_error &error){input_rejected=std::string_view(error.what())=="token ID outside vocabulary";}
  require(input_rejected && !model.active_step && !model.carrier_borrowed && b.session->head.get()==head_b && c.session->head.get()==c_empty && q::generation_boundary(*b.session->head)==visible_before_error && b.ledger->committed_count==1 && !c.ledger->committed_count,"bad C token mutated a valid request or admitted native work");
  finish_payload(c);window.retire_active(old_c_key);window.release(old_c_key);c.request_key=window.enqueue(2);
  require(c.request_key.generation>old_c_key.generation && !window.matches(old_c_key),"request-local input retry reused a stale generation");
  event("survivor_continue_then_credit_pause");std::array survivor{prepare(1)};q::execute_generation_calls(model,survivor,false,nullptr,requests);
  require(b.progress.generated.size()==2 && b.session->head->sampling.draw==2 && b.ledger->committed_count==2 && b.ledger->records[1].pending_input==before_b.next_id && b.request_key==key_b,"survivor did not continue the original pending and RNG");
  require(!delivery[1].can_accept(b.progress.generated.size(),1),"small delivery credit did not block another episode");
  const auto paused_boundary=q::generation_boundary(*b.session->head);const auto paused_ledger=b.ledger->committed_count;
  bool zero_rejected=false;try{(void)q::prepare_generation_call(model,b,b.progress.output_limit(),nullptr,requests,0);}catch(const std::runtime_error &){zero_rejected=true;}
  require(zero_rejected && !model.active_step && q::generation_boundary(*b.session->head)==paused_boundary && b.ledger->committed_count==paused_ledger,"zero-credit call produced side effects");
  b.last_idle=window.pause(key_b);require(window.waiting_front()==c.request_key && window.matches(key_b),"credit pause blocked the next logical arrival");
  event("C_runs_while_B_paused");window.admitted(c.request_key);c.prompt=c.progress.begin_turn(*c.warm_inputs[0]);std::array c_call{prepare(2)};q::execute_generation_calls(model,c_call,false,nullptr,requests);
  require(c.progress.generated.size()==1 && c.progress.request_done() && c.session->head->sampling.draw==1 && q::generation_boundary(*b.session->head)==paused_boundary,"new C failed to run independently of paused B");
  window.release_turn(c.request_key,false);
  require(!delivery[2].pending(c.progress.generated).empty() && window.matches(c.request_key),"final pending output was released prematurely");
  require(!ack(2,old_c_key,1) && !delivery[2].acknowledged(),"stale C acknowledgement advanced a new generation");
  require(ack(2,c.request_key,1) && delivery[2].pending(c.progress.generated).empty(),"final C output was not drained");finish_payload(c);window.release(c.request_key);
  require(ack(1,key_b,2),"B delivery acknowledgement failed");window.wake(key_b);require(window.waiting_front()==key_b,"awakened B did not use the original FIFO");window.admitted(key_b);
  event("B_finishes_known_turn_then_waits_for_input");std::array third{prepare(1)};q::execute_generation_calls(model,third,false,nullptr,requests);
  require(b.progress.generated.size()==3 && b.progress.waiting_for_input() && !b.progress.request_done() && b.session->head->sampling.draw==3,"open request incorrectly completed its final known turn");
  window.pause(key_b);const auto before_append=q::generation_boundary(*b.session->head);const auto old_ledger=b.ledger->committed_count;
  b.progress.extend_turns(b_turns);require(ack(1,key_b,3),"B final-known output did not drain before resume");window.wake(key_b);window.admitted(key_b);
  b.progress.advance_turn();b.prompt=b.progress.begin_turn(*b.warm_inputs[1],b.session->head->next_id);b.progress.close_input();
  require(b.prompt.size()==1 && b.prompt[0]==before_append.next_id && b.ledger->committed_count==old_ledger,"appended empty suffix changed history before its actual pending input");
  event("pending_only_append_close_and_final_drain");std::array appended{prepare(1)};q::execute_generation_calls(model,appended,false,nullptr,requests);
  require(b.progress.request_done() && b.progress.generated.size()==4 && b.ledger->committed_count==4 && b.progress.observed_commits()==4 && b.session->head->sampling.draw==4 && b.session->length()==157 && b.ledger->records[3].ids().size()==1 && b.ledger->records[3].ids()[0]==before_append.next_id && b.ledger->records[3].before.length==156 && b.ledger->records[3].after.length==157,"pending-only continuation duplicated input/output or RNG");
  window.release_turn(key_b,false);require(window.matches(key_b) && delivery[1].pending(b.progress.generated).size()==1,"terminal pending output did not retain its request slot");
  bool occupied=true;try{(void)window.enqueue(1);occupied=false;}catch(const std::logic_error &){}require(occupied,"terminal slot was reused before output drain");
  require(ack(1,key_b,4) && delivery[1].pending(b.progress.generated).empty(),"final B output did not drain");
  const auto b_ids=b.progress.generated,c_ids=c.progress.generated;const bool b_status_before_stop=b.session->retirement_status.ok();finish_payload(b);window.release(key_b);
  require(!window.matches(key_b) && !ack(1,key_b,4) && window.active().empty() && !window.waiting_front() && !model.active_step && !model.carrier_borrowed,"final retirement left native admission or accepted a stale ACK");
  require(model.draft_programs.empty() && model.prefix_sampling_programs.empty() && !model.draft_program_graph_calls && !model.prefix_decision_graph_calls,"ordinary owner used MTP paths");
  std::cout<<"{\"status\":\"PASS\",\"ordinary\":true,\"backend\":\"cuda\",\"physical_text_roles\":"<<model.physical_text_roles<<",\"physical_mtp_roles\":"<<model.physical_mtp_roles<<",\"device_weight_bytes\":"<<model.device_weight_bytes<<",\"cache_limit\":1073741824,\"cache_backing_bytes\":"<<model.known_cache_backing_bytes<<",\"submitted_R2_inflight_fact\":"<<(logical_inflight?"true":"false")<<",\"hardware_completion_at_cancel\":\"unknown\",\"survivor_retirement_status_after_abort_ok\":"<<(survivor_retirement_ok?"true":"false")<<",\"survivor_final_status_before_stop_ok\":"<<(b_status_before_stop?"true":"false")<<",\"B_key_generation\":"<<key_b.generation<<",\"C_old_generation\":"<<old_c_key.generation<<",\"C_new_generation\":"<<c.request_key.generation<<",\"B_sampling_draw\":4,\"B_committed_records\":4,\"B_ids\":[";
  for(std::size_t x=0;x<b_ids.size();++x)std::cout<<(x?",":"")<<b_ids[x];std::cout<<"],\"C_ids\":[";
  for(std::size_t x=0;x<c_ids.size();++x)std::cout<<(x?",":"")<<c_ids[x];std::cout<<"],\"scope\":\"one_original_model_host_owner_process_not_online_CLI\"}\n";
  return 0;
 }catch(const std::exception &error){std::cerr<<"live_owner_failure="<<error.what()<<'\n';return 1;}
}
