#pragma once

#include "../../src/graph_batch/native_device_append_regions.h"
#include "qwen_device_continuation_case.h"

// Prepare the real native successor recipe independently of automatic policy.
// This is a cold consumer check, not an availability override or inference pass.
namespace qwen_native_mtp_plan_case {
namespace q=qwen35;
inline int loader_contract(const std::filesystem::path &inputs,q::BackendKind backend){
  hyle::inference::ModelLoadConfig config;
  config.assets.inputs=inputs;config.execution.backend=backend;config.execution.context=256;
  config.execution.prefill_rows=32;config.execution_profiles.front().draft_count=1;
  auto prepared=q::prepare_execution_profile(config,0);
  const bool overlap=prepared.overlap;
  std::cerr<<"NATIVE_LOADER_PLAN overlap="<<overlap<<" in_flight="<<prepared.cache.in_flight_batches
      <<" active="<<prepared.cache.active_sequences<<" residents="<<prepared.cache.resident_sessions<<'\n';
  auto model=q::instantiate_execution_profile(std::move(prepared),config.execution.context);
  std::cerr<<"NATIVE_LOADER_MODEL relay="<<model->supports_device_relay(0)<<" attention="<<model->device_mtp_attention_available()
      <<" supports_device_mtp="<<model->supports_device_mtp()<<'\n';
  model->prepare_default_device_mtp_successor(overlap);
  std::cerr<<"NATIVE_LOADER_SELECTED device_mtp="<<model->device_mtp_successor_enabled()
      <<" continuous="<<model->device_mtp_continuous_enabled()<<'\n';
  if(!overlap || !model->device_mtp_successor_enabled())throw std::runtime_error("formal native loader did not select its qualified K1 route");
  return 0;
}
inline std::vector<std::vector<std::byte>> projection_rows(q::Model &model,std::size_t rows,
    std::int32_t pending,const std::vector<std::byte> &hidden_row) {
  q::GraphAssembly assembly(model.execution,model.precision,model.allocation_budget);
  q::Recipe r(model.weights,model.projections,assembly,model.geometry,rows,1,&model.weight_meanings);
  const auto ids=r.input(q::dense(q::Type::kInt32,{static_cast<std::int64_t>(rows)}));
  const auto hidden=r.input(q::dense(model.precision.activation_type(),{static_cast<std::int64_t>(rows),model.hidden}));
  const auto en=r.norm_absolute(r.embedding(ids,"model.language_model.embed_tokens.weight"),"mtp.pre_fc_norm_embedding.weight");
  const auto hn=r.norm_absolute(hidden,"mtp.pre_fc_norm_hidden.weight");
  const auto joined=r.one(q::c::Concat{.axis=1},{en,hn},r.dense_spec(model.precision.activation_type(),
      {q::f(static_cast<std::int64_t>(rows)),q::f(2*model.hidden)}));
  const auto fused=r.linear_absolute(joined,"mtp.fc.weight");
  const auto normalized=r.norm_absolute(fused,"mtp.layers.0.input_layernorm.weight");
  const auto key=r.linear_absolute(normalized,"mtp.layers.0.self_attn.k_proj.weight");
  const auto value=r.linear_absolute(normalized,"mtp.layers.0.self_attn.v_proj.weight");
  const std::array ports{en,hn,joined,fused,normalized,key,value};
  for(const auto port:ports)r.publish(port);
  const auto graph=q::take(std::move(r.builder).finish());
  const std::array shape{q::g::ShapeBinding{r.last_row,static_cast<std::int64_t>(rows-1)}};
  const std::array variants{q::g::ExactShapeVariant{shape}};
  auto compiled=q::take(model.execution.compile(graph,assembly.constants(),assembly.options({
      .shape_policy=q::g::CompileShapePolicy::kFiniteExact,.exact_variants=variants,
      .constants=assembly.constants(),.resource_budget=model.execution_budget})));
  compiled.executable=q::take(compiled.executable.exact_variant(0));compiled.report=q::take(compiled.report.exact_variant(0));
  auto tokens=model.caller_tensor(model.execution.device(),q::dense(q::Type::kInt32,{static_cast<std::int64_t>(rows)}));
  auto states=model.caller_tensor(model.execution.device(),q::dense(model.precision.activation_type(),{static_cast<std::int64_t>(rows),model.hidden}));
  const std::vector<std::int32_t> values(rows,pending);tokens.write<std::int32_t>(values);
  std::vector<std::byte> repeated;for(std::size_t row=0;row<rows;++row)repeated.insert(repeated.end(),hidden_row.begin(),hidden_row.end());
  states.write<std::byte>(repeated);
  std::vector<q::PreparedReadback> outputs;
  for(const auto port:ports)outputs.push_back(q::take(q::PreparedReadback::prepare(compiled,port,model.execution.storage_domains(),model.allocation_budget)));
  auto run=q::take(compiled.executable.create_invocation({.resources=q::g::ResourceOptions{}}));
  q::check(run.bind(ids,tokens.read()));q::check(run.bind(hidden,states.read()));
  for(auto &output:outputs)q::check(output.bind(run));
  q::check(run.prepare_replay(model.execution.queue_bindings(),{.mode=q::g::ReplayMode::kCommands,.host_wait=q::g::HostWaitPolicy::kAllow}));
  auto done=q::take(run.submit(model.execution.queue_bindings()));q::check(done.wait());
  std::vector<std::vector<std::byte>> result;
  for(auto &output:outputs){
    std::vector<std::byte> bytes(q::take(output.facts.description.required_bytes()));
    q::check(output.read(run,bytes));bytes.resize(bytes.size()/rows);result.push_back(std::move(bytes));
  }
  return result;
}
inline int run(const std::filesystem::path &inputs,q::BackendKind backend,bool execute=false,bool inspect=false) {
  if(backend!=q::BackendKind::kApple)
    throw std::runtime_error("native Metal MTP plan probe requires the actual Apple endpoint");
  auto source=q::prepare_weight_source(inputs/"weights.tsv",true);
  const auto geometry=source.roles.geometry;
  const auto precision=q::PrecisionProfile::from_roles(source.roles);
  constexpr std::size_t capacity=32768;
  const auto plan=q::cache::plan(geometry,source.cache_row_bytes,capacity,1,execute?3:2,
      q::recurrent_bytes(q::state_schema_for(geometry,capacity,precision),geometry),1,2,true,true,3,0,precision);
  q::Model model(std::move(source),geometry,capacity,plan,{},false,q::ReplayPolicy::kPreferCapture,SIZE_MAX,
      q::Model::Sampling{},0,{.primary=backend});
  model.configure_draft_policy({.max_drafts=1});
  // The actual backend qualifies the original attention configuration. This
  // consumer must not replace that answer with a structural recipe override.
  model.configure_device_mtp_successor(true);
  const std::array pair{std::size_t{2}},prefill{std::size_t{3}},one{std::size_t{1}};
  if(execute){
    model.prepare_family(prefill,false,1,q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(3,1,8,q::GraphRole::kMtpPrefill);
    for(const auto role:{q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound}){
      model.prepare_family(pair,false,1,role);model.plan_replay_uses(2,1,32,role);
    }
    for(const auto role:{q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle}){
      model.prepare_family(one,false,1,role);model.plan_replay_uses(1,1,8,role);
    }
  }
  model.prepare_family(pair,false,1,q::GraphRole::kMtpContinuation);model.plan_replay_uses(2,1,32,q::GraphRole::kMtpContinuation);
  model.freeze_shape_plan();
  if(execute){
    model.prepare_execution(prefill,1,q::GraphRole::kMtpPrefill);
    for(const auto role:{q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound})model.prepare_execution(pair,1,role);
    for(const auto role:{q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle})model.prepare_execution(one,1,role);
  }
  model.prepare_relay_execution(1,0,q::GraphRole::kMtpContinuation);
  if(!model.device_mtp_continuous_enabled())
    throw std::runtime_error("native MTP preparation changed its original finite inventory");
  const auto found=std::ranges::find_if(model.phases,[](const auto &entry){
    return entry.second->family->role==q::GraphRole::kMtpContinuation && entry.second->imported_sources;
  });
  if(found==model.phases.end())throw std::runtime_error("native MTP successor has no actual imported phase");
  auto &phase=*found->second;
  if(phase.slots.size()!=q::execution_slot_count || phase.append_nodes.empty())
    throw std::runtime_error("native MTP successor lacks its original independent execution slots");
  std::size_t total=0;
  for(auto &slot:phase.slots.slots()) {
    auto *invocation=q::take(slot.carrier.borrow());
    std::size_t selected=0;
    q::check(hyle::graph::detail::RegionAccess::visit(*invocation,
        [](void *opaque,const hyle::graph::detail::RegionAccessView &view) noexcept -> hyle::Status {
          if(view.prepared_vtable && view.prepared_vtable->query_extension &&
              view.prepared_vtable->query_extension(view.prepared,
                  hyle::graph::batch::detail::kNativeDeviceAppendRegionExtensionId))
            ++*static_cast<std::size_t *>(opaque);
          return hyle::Status::Ok();
        },&selected));
    if(selected!=phase.append_nodes.size())
      throw std::runtime_error("native MTP successor did not select every actual device-controlled append");
    total+=selected;
  }
  std::cout<<"PASS native_mtp_cold_plan backend=apple drafts=1 hidden="<<geometry.hidden
      <<" vocabulary="<<geometry.vocabulary<<" slots="<<phase.slots.size()
      <<" device_append_regions="<<total<<" model_submissions=0 backend_qualified=1 service_qualified=0\n"<<std::flush;
  if(execute){
    namespace d=qwen_device_continuation_case;
    constexpr std::array<std::int32_t,3> prompt{1,2,3};
    std::vector<qwen_mtp_batch_state_case::StateImage> serial_states,candidate_states;
    const auto serial=d::journey(model,prompt,false,inspect?&serial_states:nullptr),
        candidate=d::journey(model,prompt,true,inspect?&candidate_states:nullptr);
    d::compare_records(serial,candidate);
    if(inspect){
    const auto &first=serial_states.front();
    const auto one=projection_rows(model,1,serial.records.at(2).pending_input,first.payloads.at(first.payloads.size()-2));
    const auto two=projection_rows(model,2,serial.records.at(2).pending_input,first.payloads.at(first.payloads.size()-2));
    constexpr std::array names{"embedding_norm","hidden_norm","concat","fc","input_norm","key","value"};
    for(std::size_t index=0;index<one.size();++index){
      q::cache::HostPayloadDesc layout{};layout.role=q::cache::HostPayloadRole::kCarryHidden;layout.bytes=one[index].size();
      const auto difference=qwen_packed_prefill_case::compare_payload(model,layout,one[index],two[index]);
      std::cerr<<"NATIVE_MTP_PROJECT_ROWS stage="<<names[index]<<" different_bytes="<<difference.different_bytes
          <<" first_byte="<<difference.first_byte<<" max_absolute="<<difference.maximum_absolute<<'\n';
    }
    d::require(serial_states.size()==d::episodes && candidate_states.size()==serial_states.size(),
        "native K1 observation lost an original committed episode");
    for(std::size_t index=0;index<serial_states.size();++index){
      std::cerr<<"NATIVE_MTP_STATE_COMPARE episode="<<index<<" length="<<serial_states[index].length<<'\n';
      d::compare_state(model,serial_states[index],candidate_states[index]);
    }
    }
    d::compare_state(model,*serial.final_state,*candidate.final_state);
    d::require(candidate.native_prepared==d::episodes-1 && !candidate.recorded &&
        serial.submitted==candidate.submitted && serial.draft_rows==candidate.draft_rows,
        "native K1 chain changed work or fell back from its original prepared device successor");
    std::cout<<"PASS native_mtp_chain backend=apple drafts=1 ids="<<candidate.ids.size()
        <<" native_prepared="<<candidate.native_prepared<<" submissions="<<candidate.submitted
        <<" full_state=equal rng=equal\n"<<std::flush;
  }
  return 0;
}
} // namespace qwen_native_mtp_plan_case
