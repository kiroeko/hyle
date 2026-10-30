#pragma once
#include "../../engine/inference/output_delivery.h"

// Real ordinary CPU Model/Graph on the retained tiny fixture. Request stopping
// changes no weights, logits, sampling graph, accepted state or native executor.
namespace qwen_request_stop_case {
namespace q=qwen35;
namespace i=hyle::inference;
inline void require(bool value,const char *message) {
  if(!value)throw std::runtime_error(message);
}
inline void retire(q::GenerationRequest &request) {
  const auto retired=q::retire_generation_request(request);
  if(retired.retired)request.session.reset();
  q::check(retired.status);require(retired.retired,"stop-policy Session did not actually retire");
}
inline int run(const std::filesystem::path &input) {
  const auto geometry=q::read_weight_geometry(input/"weights.tsv");
  require(geometry.hidden==64 && geometry.intermediate==128 && geometry.vocabulary==64 &&
      geometry.layers=="A","stop-policy case requires cached A/H64/FFN128/V64");
  // Two active sessions plus one retained completed-prefix version.
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan=q::cache::plan(geometry, {.target = cache_rows},16,3,6,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),2,1,false,true);
  q::Model model(input/"weights.tsv",geometry,16,plan,{},false,q::ReplayPolicy::kCommands,
      16u<<20,{.enabled=true,.policy={1.F,8,.9F},.seed=123},0,{.primary=i::BackendKind::kCpu});
  const std::array rows{std::size_t{1},std::size_t{2},std::size_t{6}};
  for(const auto width:{std::size_t{1},std::size_t{2}}) {
    model.prepare_family(rows,false,width);
    for(const auto count:rows)model.plan_replay_uses(count,width,24);
  }
  for(const auto width:{std::size_t{1},std::size_t{2}})model.prepare_execution(rows,width);
  const std::array a_turns{i::GenerationTurn{{1,2},3}};
  const std::array b_turns{i::GenerationTurn{{1,2},1},i::GenerationTurn{{3,4,5,6,7},3}};
  const std::array hit_turns{i::GenerationTurn{{1,2},2}};
  const std::array<std::int32_t,1> stop_two{2},stop_fortyseven{47},invalid_stop{64};
  std::vector<q::GenerationRequest> requests;requests.reserve(5);
  requests.emplace_back("stop-A",a_turns[0].prompt,std::filesystem::path{},3,0,i::RequestStopPolicy(stop_two));
  requests.emplace_back("resume-B",b_turns[0].prompt,std::filesystem::path{},4,0,i::RequestStopPolicy(stop_fortyseven));
  requests.emplace_back("new-policy-hit",hit_turns[0].prompt,std::filesystem::path{},2,0,i::RequestStopPolicy({},true));
  requests.emplace_back("stopped-policy-hit",hit_turns[0].prompt,std::filesystem::path{},2,0,i::RequestStopPolicy(stop_two));
  requests.emplace_back("invalid-stop",hit_turns[0].prompt,std::filesystem::path{},2,0,i::RequestStopPolicy(invalid_stop));
  for(std::size_t index=0;index<requests.size();++index) {
    auto &request=requests[index];
    const auto turns=index==0?std::span<const i::GenerationTurn>(a_turns):index==1?
        std::span<const i::GenerationTurn>(b_turns):std::span<const i::GenerationTurn>(hit_turns);
    request.progress.bind_turns(turns);request.progress.initial_sampling={123,10,0};
    request.ledger=std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns),1);
    request.round_emitted.resize(request.progress.maximum_outputs()-1);
    for(auto &emitted:request.round_emitted)emitted.reserve(1);
    request.turn_observations.resize(turns.size());
    for(std::size_t turn=0;turn<turns.size();++turn) {
      auto owner=std::make_shared<std::vector<std::int32_t>>(turns[turn].prompt.size()+(turn?1:0));
      std::copy(turns[turn].prompt.begin(),turns[turn].prompt.end(),owner->begin()+(turn?1:0));
      request.warm_inputs.push_back(std::move(owner));
    }
  }
  auto &a=requests[0],&b=requests[1],&c=requests[2],&d=requests[3],&invalid=requests[4];
  bool invalid_refused=false;
  try{(void)q::prepare_generation_call(model,invalid,2,nullptr,requests);}
  catch(const std::out_of_range &){invalid_refused=true;}
  require(invalid_refused && !invalid.session && !invalid.progress.turn_started() &&
      !invalid.ledger->committed_count && invalid.progress.input_turn()==SIZE_MAX && !model.active_step &&
      !model.snapshot_versions.live,"invalid stop ID changed high-level admission facts");
  const auto begin=[](q::GenerationRequest &request) {
    if(!request.progress.turn_started())request.prompt=request.progress.begin_turn(
        *request.warm_inputs[request.progress.turn_index],request.progress.turn_index?
        std::optional<std::int32_t>(request.session->head->next_id):std::nullopt);
  };
  const auto prepare=[&](q::GenerationRequest &request,q::cache::PrefixCache *prefix=nullptr,std::size_t credit=1) {
    return q::prepare_generation_call(model,request,request.progress.output_limit(),prefix,requests,credit);
  };
  const auto execute=[&](q::GenerationRequest &request,q::cache::PrefixCache *prefix=nullptr,std::size_t credit=1) {
    std::array calls{prepare(request,prefix,credit)};q::execute_generation_calls(model,calls,false,prefix,requests);
  };
  i::RequestWindow window;window.prepare(2,2);
  const auto key_a=window.enqueue(0),key_b=window.enqueue(1);a.request_key=key_a;b.request_key=key_b;
  begin(a);begin(b);window.admitted(key_a);window.admitted(key_b);
  std::array first{prepare(a),prepare(b)};q::execute_generation_calls(model,first,false,nullptr,requests);
  require(a.progress.generated==std::vector<std::int32_t>{2} && b.progress.generated==a.progress.generated &&
      a.progress.turn_done && a.progress.generated.size()<a.progress.output_limit() &&
      a.ledger->records[0].stopped && !b.ledger->records[0].stopped &&
      a.session->head->sampling.draw==1 && b.session->head->sampling.draw==1,
      "same Model/R2 did not preserve distinct request stop decisions");
  q::cache::PrefixCache prefix(1);q::publish_generation_completed(model,a,&prefix);
  require(prefix.size()==1,"actual complete stopped prefix was not published");
  b.completed_checkpoint=a.completed_checkpoint;
  c.input=a.completed_checkpoint.input;d.input=c.input;
  window.release_turn(key_a,false);retire(a);window.release(key_a);
  b.last_idle=window.release_turn(key_b,true);b.progress.advance_turn();
  // The source publisher stopped on2. C's new policy must not inherit that
  // fact, even though the full consumed state/pending can be reused.
  begin(c);execute(c,&prefix);
  require(c.prefix_full_hits==1 && c.progress.generated==std::vector<std::int32_t>{2} &&
      !c.progress.turn_done && !c.ledger->records[c.ledger->committed_count-1].stopped &&
      c.progress.observed_commits()==c.ledger->committed_count,
      "full-prefix import retained the publisher's request stop decision");
  retire(c);
  begin(d);execute(d,&prefix);
  require(d.prefix_full_hits==1 && d.ledger && d.progress.generated==std::vector<std::int32_t>{2} &&
      d.progress.turn_done && d.progress.generated.size()<d.progress.output_limit(),
      "ledger-backed full hit lost its new request stop policy");
  retire(d);
  require(window.waiting_front()==key_b,"finished peer changed B's original FIFO/key");
  window.admitted(key_b);begin(b);i::OutputDelivery delivery(1);
  delivery.acknowledge_prefix(b.progress.generated.size(),1);execute(b,nullptr,delivery.available(1));
  require(b.progress.generated==std::vector<std::int32_t>({2,10}) && !b.progress.turn_done &&
      b.progress.remaining_outputs()==2 && b.session->length()==8,
      "actual nonterminal suffix did not reach the expected immutable stop boundary");
  const auto boundary=q::generation_boundary(*b.session->head);const auto observed=b.progress.observed_commits();
  const auto output=b.progress.generated;const auto consumed=b.progress.input_consumed();
  b.last_idle=window.pause(key_b);retire(b);
  require(window.matches(key_b) && !b.session && delivery.acknowledged()==1,
      "pause/retirement released the logical request or pending delivery");
  q::restore_generation_session(model,b,&prefix,false,requests);
  require(q::generation_boundary(*b.session->head)==boundary && b.progress.generated==output &&
      b.progress.observed_commits()==observed && b.progress.input_consumed()==consumed &&
      b.ledger->stop_policy()==b.stop_policy && b.restore_count==1 && b.checkpoint_resume_misses==1 &&
      b.progress.turn_started(),"rebuild changed policy/progress/input/pending before reinstall");
  delivery.acknowledge_prefix(b.progress.generated.size(),2);window.wake(key_b);window.admitted(key_b);
  execute(b,nullptr,delivery.available(2));
  require(b.progress.generated==std::vector<std::int32_t>({2,10,47}) && b.progress.request_done() &&
      b.progress.generated.size()<b.progress.output_limit() && b.ledger->records[2].stopped &&
      b.session->length()==9 && b.session->head->sampling.draw==3 &&
      b.ledger->records[2].pending_input==10 && b.session->head->next_id==47,
      "restored ordinary request did not stop early on its original extra token");
  window.release_turn(key_b,false);
  require(window.matches(key_b) && delivery.pending(b.progress.generated).size()==1,
      "stopped request released its key before final pending delivery");
  delivery.acknowledge_prefix(b.progress.generated.size(),3);retire(b);window.release(key_b);prefix.evict_one();

  // Original raw Model+EvaluationRequest+ledger misuse. The empty ledger has
  // not cold-bound the extra-ID policy. No post-submit vector copy is allowed.
  const std::array chunks{std::size_t{2}};
  auto raw=model.create_session(chunks,{},q::GraphRole::kText,q::cache::SamplingCursor{123,10,0});
  const auto *raw_base=raw->head.get();const auto raw_boundary=q::generation_boundary(*raw->head);
  const i::RequestStopPolicy extra(stop_two),bad(invalid_stop);
  const std::array bad_inputs{q::EvaluationRequest{raw,a_turns[0].prompt,&bad}};
  bool raw_invalid=false;try{model.prepare_evaluations(bad_inputs);}catch(const std::out_of_range &){raw_invalid=true;}
  require(raw_invalid && !model.active_step && raw->head.get()==raw_base,
      "raw stop vocabulary rejection changed its actual episode/head");
  q::GenerationLedger ledger(1,1);
  const auto prepare_raw=[&] {
    q::prepare_generation_record(ledger,*raw,q::GraphRole::kText,a_turns[0].prompt,false,0,0);
    const std::array inputs{q::EvaluationRequest{raw,a_turns[0].prompt,&extra}};
    model.prepare_evaluations(inputs);model.submit_evaluations();
  };
  const auto finish_raw=[&] {
    const q::EvaluationIO io{.name="direct-stop-policy"};
    const std::array outputs{q::evaluation_output(model,raw,io)};
    q::GenerationPublicationBatch publication({{&ledger,nullptr}});
    std::array<std::int32_t,1> next{};
    model.finish_evaluations(outputs,next,[](std::size_t,const q::CompletedEvaluationView &){},publication);return next[0];
  };
  prepare_raw();bool unbound_rejected=false;
  try{(void)finish_raw();}catch(const std::runtime_error &error) {
    unbound_rejected=std::string_view(error.what())=="generation ledger stop policy differs from its admitted episode";
  }
  require(unbound_rejected && !ledger.committed_count && raw->head.get()==raw_base &&
      q::generation_boundary(*raw->head)==raw_boundary && !raw->pending && !model.active_step &&
      !model.carrier_borrowed && raw->retirement_status.ok(),
      "unbound nondefault ledger published or failed to retire its original episode");
  ledger.bind_stop_policy(extra,model.vocabulary);prepare_raw();const auto raw_id=finish_raw();
  require(raw_id==2 && ledger.committed_count==1 && ledger.records[0].stopped &&
      raw->head->sampling.draw==1 && raw->length()==2,"cold-bound direct retry lost its stop fact");
  raw->stop().IgnoreError();const auto raw_retired=raw->try_retire();q::check(raw_retired.status);
  require(raw_retired.retired,"raw policy case did not actually retire");raw.reset();
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      window.active().empty() && !window.waiting_front(),"stop-policy EOF retained a request/version/carrier");
  std::cout<<"{\"status\":\"PASS\",\"backend\":\"cpu\",\"same_model_R2_distinct_stops\":true,"
      "\"new_policy_full_hit\":true,\"ledger_full_hit_new_stop_policy\":true,\"pause_rebuild_preserved_policy\":true,"
      "\"unbound_direct_ledger_refused_before_publish\":true,\"same_session_cold_bind_retry\":true,"
      "\"invalid_stop_ID_preflight\":true,\"B_ids\":[2,10,47],\"B_draw\":3,\"final_versions\":0,"
      "\"GPU\":false,\"HTTP\":0}\n";
  return 0;
}
} // namespace qwen_request_stop_case
