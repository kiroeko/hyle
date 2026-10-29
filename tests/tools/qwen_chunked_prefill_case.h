#pragma once

// Included after the real model implementation by qwen_request_lifecycle_test.cc.
namespace qwen_chunked_prefill_case {
namespace q=qwen35;
namespace i=hyle::inference;
namespace {
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
void retire(q::GenerationRequest &request){
  (void)request.session->stop();const auto drained=request.session->drain();
  const auto retired=request.session->try_retire();if(retired.retired)request.session.reset();
  q::check(drained.status);q::check(retired.status);require(retired.retired,"payload did not retire");
}
std::vector<std::vector<std::byte>> full_state_bytes(q::Model &model,const q::cache::State &head){
  std::vector<std::vector<std::byte>> result;
  const auto add=[&](const q::c::ReadLease &lease){
    const auto view=q::take(lease.view());result.emplace_back(view.size());q::check(model.execution.read(view,result.back()));
  };
  for(std::size_t layer=0;layer<head.paged.frames.size() / 2;++layer)for(const auto *frame:{&head.paged.frames[2 * layer],&head.paged.frames[2 * layer + 1]}){
    const auto footprint=q::take(frame->view());std::vector<q::c::ReadLease> pieces(footprint.layout().host().page_indices.size());
    q::check(footprint.page_prefixes(pieces));for(const auto &piece:pieces)add(piece);
  }
  for(std::size_t layer=0;layer<model.geometry.layers.size();++layer)if(model.geometry.is_gdn(layer)){
    add(head.state[layer].history);add(head.state[layer].recurrent);
  }
  if(head.carry_hidden.valid())add(head.carry_hidden);return result;
}
}
inline int run(const std::filesystem::path &input){try{
  const auto geometry=q::read_weight_geometry(input/"weights.tsv");
  const auto prompt=q::read_ids(input/"prompt.ids",geometry.vocabulary,512);require(prompt.size()==154,"expected original P154");
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan=q::cache::plan(geometry, {.target = cache_rows},512,3,128,q::recurrent_bytes(q::state_schema_for(geometry,512),geometry),2,1,false,true);
  q::Model model(input/"weights.tsv",geometry,512,plan,{},true,q::ReplayPolicy::kCommands,1073741824,
      {.enabled=true,.policy={1,8,.9F},.seed=123},0,{.primary=q::BackendKind::kCuda});
  require(model.physical_text_roles==320&&!model.physical_mtp_roles&&!model.cache_plan.draft_cache,"not original ordinary Model");
  // Real consumption is A154 = 64+64+26; the next turn's pending+130
  // input is 64+64+3. Both final tails use the same cold physical R64.
  const std::array state_shapes{std::size_t{64}};
  const std::array emit_shapes{std::size_t{1}}; // Original exact decode only.
  model.prepare_family(emit_shapes);
  require(model.prepare_prefill_family(state_shapes,q::TextCompletion::kStateOnly)&&model.prepare_prefill_family(state_shapes),"real CUDA consumer did not prepare packed state/emit prefill");
  for(auto rows:emit_shapes)model.plan_replay_uses(rows,1,16);
  for(auto rows:state_shapes)for(auto completion:{q::TextCompletion::kStateOnly,q::TextCompletion::kEmitNext})model.plan_replay_uses(rows,1,16,q::GraphRole::kText,completion,0,q::RowPacking::kPrefill);
  model.prepare_execution(emit_shapes);
  model.prepare_prefill_execution(state_shapes,q::TextCompletion::kStateOnly);model.prepare_prefill_execution(state_shapes);
  for(auto rows:state_shapes){
    const auto *state_family=model.find_family(1,rows,q::GraphRole::kText,false,q::TextCompletion::kStateOnly,0,q::RowPacking::kPrefill);
    require(state_family&&!state_family->logits_port&&!state_family->id_port&&!state_family->sampling_identity_port&&!state_family->sampling_valid_port,"state-only graph has a score/sampler/ID port");
  }
  std::vector<i::GenerationTurn> turns_a{{prompt,1},{std::vector<std::int32_t>(prompt.begin(),prompt.begin()+130),1},{{},1}};
  std::vector<i::GenerationTurn> turns_b{{std::vector<std::int32_t>(prompt.begin(),prompt.begin()+16),3}};
  std::vector<q::GenerationRequest> requests;requests.reserve(2);requests.emplace_back("A",prompt,std::filesystem::path{},3);requests.emplace_back("B",turns_b[0].prompt,std::filesystem::path{},3);
  for(std::size_t index=0;index<2;++index){
    auto &request=requests[index];const auto &turns=index?turns_b:turns_a;request.progress.bind_turns(turns);
    request.progress.initial_sampling={123,100+index,0};request.ledger=std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns,64),1);
    request.round_emitted.resize(2);for(auto &row:request.round_emitted)row.reserve(1);request.turn_observations.resize(turns.size());
    request.warm_inputs.reserve(turns.size());
    for(std::size_t turn=0;turn<turns.size();++turn){
      auto owner=std::make_shared<std::vector<std::int32_t>>(turns[turn].prompt.size()+(turn?1:0));
      std::copy(turns[turn].prompt.begin(),turns[turn].prompt.end(),owner->begin()+(turn?1:0));request.warm_inputs.push_back(std::move(owner));
    }
  }
  i::RequestWindow window;window.prepare(2,2);
  for(std::size_t index=0;index<2;++index){auto &r=requests[index];r.request_key=window.enqueue(index);window.admitted(r.request_key);r.prompt=r.progress.begin_turn(*r.warm_inputs[0]);}
  auto &a=requests[0];auto &b=requests[1];const auto key_a=a.request_key,key_b=b.request_key;
  std::vector<std::string> dispatch;dispatch.reserve(16);std::size_t token_rows=0,physical_rows=0;
  std::vector<std::size_t> physical_dispatch;physical_dispatch.reserve(16);
  const auto prepare=[&](std::size_t index){return q::prepare_generation_call(model,requests[index],requests[index].progress.output_limit(),nullptr,requests,1,64);};
  const auto execute=[&](std::size_t index){
    auto call=prepare(index);require(call.planned_rows()<=64,"token quantum exceeded");const auto rows=call.ids().size();
    const auto prefill=!call.generation;
    const auto observed=[&]{require(model.active_step&&model.active_step->input_tokens==rows,"submitted episode lost actual input rows");const auto carrier=model.active_step->phase->query_rows;require(carrier>=rows&&(!prefill||(carrier==64&&q::packs_prefill(model.active_step->phase->family->packing))),"physical carrier does not use its actual cold packed prefill");physical_rows+=carrier;physical_dispatch.push_back(carrier);};
    std::array calls{std::move(call)};q::execute_generation_calls(model,calls,false,nullptr,requests,observed);
    token_rows+=rows;dispatch.push_back(requests[index].id+":"+std::to_string(rows));
  };
  execute(1);require(b.progress.generated.size()==1&&b.session->head->sampling.draw==1,"B is not a ready actual decode");window.requeue(key_b);
  execute(0);require(a.progress.input_consumed()==64&&!a.progress.prefilled&&a.progress.generated.empty()&&a.session->head->next_id==-1&&a.session->head->sampling.draw==0,"first state-only chunk generated output or RNG");
  const auto first_boundary=q::generation_boundary(*a.session->head);const auto first_bytes=full_state_bytes(model,*a.session->head);window.requeue(key_a);
  require(window.waiting_front()==key_b,"ready decode did not precede prefill survivor");window.admitted(key_b);execute(1);require(b.progress.generated.size()==2,"ready short decode was starved");window.requeue(key_b);
  require(window.waiting_front()==key_a,"chunk continuation lost FIFO");window.admitted(key_a);
  retire(a);q::restore_generation_session(model,a,nullptr,false,requests);const auto &restored=a.restored;
  require(q::generation_boundary(*a.session->head)==first_boundary&&full_state_bytes(model,*a.session->head)==first_bytes&&restored.target_rows==64&&a.progress.input_consumed()==64&&a.progress.generated.empty(),"partial state-only rebuild changed full state/KV/cursor/offset");
  const auto *cancel_base=a.session->head.get();const auto before_b=q::generation_boundary(*b.session->head);auto cancelled=prepare(0);
  q::prepare_generation_record(*a.ledger,*a.session,cancelled.role,cancelled.ids(),false,0,0,a.warm_inputs[0],0,cancelled.completion,cancelled.input_begin,cancelled.prefill_complete);
  const std::array inputs{q::EvaluationRequest{a.session,cancelled.ids()}};model.prepare_prefill_evaluations(inputs,nullptr,q::GraphRole::kText,cancelled.completion);require(q::packs_prefill(model.active_step->phase->family->packing)&&model.active_step->phase->query_rows==64,"cancel arm did not submit the real packed prefill carrier");model.submit_evaluations();
  const auto cancelled_physical=model.active_step->phase->query_rows;
  const bool submitted_inflight=model.active_step->execution.run->in_flight();const auto abort=model.abort_episode();if(!abort.status.IsCancelled()||abort.status.has_native_error())q::check(abort.status);require(abort.retired,"cancelled chunk did not actually retire");
  require(!model.active_step&&!model.carrier_borrowed&&!a.session->pending&&a.session->head.get()==cancel_base&&a.progress.input_consumed()==64&&a.ledger->committed_count==1&&a.session->head->sampling.draw==0&&q::generation_boundary(*b.session->head)==before_b,"cancelled chunk published or advanced input");
  require(full_state_bytes(model,*a.session->head)==first_bytes,"cancelled packed prefill changed the committed full state/KV prefix");
  execute(0);require(a.progress.input_consumed()==128&&!a.progress.prefilled&&a.progress.generated.empty()&&a.session->head->sampling.draw==0,"second state-only chunk changed RNG/output");window.requeue(key_a);
  require(window.waiting_front()==key_b,"decode lost second FIFO turn");window.admitted(key_b);execute(1);require(b.progress.request_done()&&b.session->head->sampling.draw==3,"B decode did not complete");window.release_turn(key_b,false);retire(b);window.release(key_b);
  window.admitted(key_a);execute(0);require(a.progress.input_consumed()==154&&a.progress.turn_done&&a.progress.generated.size()==1&&a.session->head->sampling.draw==1&&a.ledger->committed_count==3,"real 26-row final chunk did not emit exactly once");
  // Only the complete final boundary enters the existing directory; keep actual chunk modes.
  q::cache::PrefixCache prefix(1);q::publish_generation_completed(model,a,&prefix);require(prefix.size()==1&&a.completed_checkpoint.replay.chunks==3,"final prefix omitted actual state-only history");
  auto hit=prefix.lookup(*a.completed_checkpoint.input,154,154,&model,model.math_identity());require(bool(hit),"complete final prefix not reusable");
  require(hit.payload.replay.history->chunks[0].state_only&&hit.payload.replay.history->chunks[1].state_only&&!hit.payload.replay.history->chunks[2].state_only&&hit.payload.replay.history->chunks[2].count==26,"prefix history lost real chunk modes or tail length");hit.reset();prefix.evict_one();
  window.release_turn(key_a,true);a.progress.advance_turn();window.admitted(key_a);const auto pending=a.session->head->next_id;a.prompt=a.progress.begin_turn(*a.warm_inputs[1],pending);
  execute(0);require(a.progress.input_consumed()==64&&a.session->head->sampling.draw==1&&!a.progress.prefilled&&a.session->head->next_id==-1&&a.ledger->records[3].ids()[0]==pending,"long suffix duplicated pending or sampled early");window.requeue(key_a);window.admitted(key_a);
  execute(0);require(a.progress.input_consumed()==128&&a.session->head->sampling.draw==1&&!a.progress.prefilled,"long suffix intermediate sampled");window.requeue(key_a);window.admitted(key_a);
  execute(0);require(a.progress.input_consumed()==131&&a.progress.generated.size()==2&&a.session->head->sampling.draw==2&&a.progress.turn_done&&a.ledger->committed_count==6&&a.ledger->records[5].ids().size()==3,"real 3-row suffix final failed");
  window.release_turn(key_a,true);a.progress.advance_turn();window.admitted(key_a);const auto pending_only=a.session->head->next_id;a.prompt=a.progress.begin_turn(*a.warm_inputs[2],pending_only);execute(0);
  require(a.progress.request_done()&&a.progress.input_consumed()==1&&a.progress.generated.size()==3&&a.session->head->sampling.draw==3&&a.session->length()==286&&a.ledger->committed_count==7&&a.ledger->records[6].ids().size()==1&&a.ledger->records[6].ids()[0]==pending_only,"pending-only final consumed history twice");
  const auto ids=a.progress.generated;window.release_turn(key_a,false);retire(a);window.release(key_a);require(window.active().empty()&&!window.waiting_front()&&!model.active_step,"request resources remain");
  require(token_rows==304,"fixed A/B journeys lost or duplicated input rows");
  std::cout<<"{\"status\":\"PASS\",\"ordinary\":true,\"physical_text_roles\":"<<model.physical_text_roles<<",\"physical_mtp_roles\":"<<model.physical_mtp_roles<<",\"Q\":64,\"dispatch\":[";
  for(std::size_t at=0;at<dispatch.size();++at)std::cout<<(at?",":"")<<'"'<<dispatch[at]<<'"';
  std::cout<<"],\"physical_dispatch\":[";for(std::size_t at=0;at<physical_dispatch.size();++at)std::cout<<(at?",":"")<<physical_dispatch[at];
  std::cout<<"],\"actual_committed_input_rows\":"<<token_rows<<",\"physical_committed_input_rows\":"<<physical_rows<<",\"cancelled_input_rows\":"<<cancelled.ids().size()<<",\"cancelled_physical_rows\":"<<cancelled_physical<<",\"A_commits\":7,\"B_commits\":3,\"A_draw\":3,\"B_draw\":3,\"partial_rebuild_rows\":"<<restored.target_rows<<",\"partial_rebuild_graph_calls\":"<<restored.graph_calls<<",\"partial_gdn_and_carry_exact\":true,\"partial_state_kv_and_carry_byte_equal\":true,\"cancelled_state_kv_and_carry_byte_unchanged\":true,\"submitted_inflight_at_abort\":"<<(submitted_inflight?"true":"false")<<",\"hardware_completion_at_abort\":\"unknown\",\"A_ids\":[";
  for(std::size_t at=0;at<ids.size();++at)std::cout<<(at?",":"")<<ids[at];std::cout<<"],\"B_ids\":[";for(std::size_t at=0;at<b.progress.generated.size();++at)std::cout<<(at?",":"")<<b.progress.generated[at];std::cout<<"]}\n";return 0;
}catch(const std::exception &error){std::cerr<<"chunked_failure="<<error.what()<<'\n';return 1;}}

} // namespace qwen_chunked_prefill_case
