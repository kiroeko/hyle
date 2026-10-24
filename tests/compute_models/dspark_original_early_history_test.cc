// Original GA Flash ordinary target40/41/42. No MTP draft or acceptance path.
// The layer39 input and the separately labelled position126 prefix are finite
// upstream fixtures; every subsequent target state is actual native feedback.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>

#include "dspark_native_consumer.h"
#include "dspark_original_target_program.h"
#include "dspark_original_target_reference.h"
#include "dspark_original_target_weights.h"
#include "glm_shared_vocabulary_reference.h"
#include "hyle/compute/state_versions.h"

namespace {
namespace n=dspark_native_consumer;
namespace t=dspark_original_target_fragment;
namespace p=dspark_original_target_program;
namespace r=dspark_original_target_roles;
namespace ref=dspark_original_target_reference;
namespace w=dspark_original_target_weights;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using n::take;using n::check;using n::read;using n::upload;using n::bytes;using n::values;
using Value=sem::ValueId;
constexpr std::int64_t kCapacity=132,kVocabulary=129280;
constexpr std::array<std::size_t,3> kFirst{0,8,13},kCursor{7,12,20};

std::string environment(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t count=0;
  if(_dupenv_s(&value,&count,name))throw std::runtime_error("ordinary environment lookup failed");
  const std::unique_ptr<char,decltype(&std::free)> cleanup(value,&std::free);
  return value?value:"";
#else
  const auto value=std::getenv(name);return value?value:"";
#endif
}
void verified() {
  if(::testing::Test::HasFailure())throw std::runtime_error("ordinary source reference failed before publication");
}
c::WorkloadContext workload(const sem::Fragment &f) {
  if(!f.profile)throw std::runtime_error("ordinary phase has no original artifact profile");
  return {take(c::ModelArtifactProfile::create(*f.profile),"ordinary profile"),f.phase,
      {0x48594c454d4f444cULL,f.region,1}};
}
template<class Recipe>
std::vector<g::ConstantBinding> constants(n::Backend &backend,const Recipe &recipe,const sem::GraphCase &built) {
  std::vector<g::ConstantBinding> result;
  for(std::size_t i=0;i<recipe.fragment.constants.size();++i) {
    const auto value=recipe.fragment.constants[i];
    const auto found=std::find_if(recipe.constants.begin(),recipe.constants.end(),
        [&](const auto &entry){return entry.value.ordinal()==value.ordinal();});
    if(found!=recipe.constants.end())result.push_back(take(found->role.binding(built.constants[i]),"original immutable role"));
    else {
      auto lease=upload(backend.owner,recipe.fragment.constant_bytes.at(i));
      result.push_back({.value=built.constants[i],.tensor=take(lease.tensor(recipe.fragment.description(value)),"cold literal"),
          .owner=backend.owner,.lease=lease});
    }
  }
  return result;
}
using Routed=n::RoutedRun<r::Data,w::CpuSupply,t::Boundary,r::Selection
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    ,w::CudaSupply
#endif
    >;
struct Step final {p::Recipe recipe;Routed run;};
struct HeadRecipe final {
  sem::Fragment fragment{.profile=&cpu_c4_model_branches::flash_profile,.region=0x4541524c594c4d01,.quantized=true};
  std::vector<p::Constant> constants;
};
struct Programs final {
  r::Bound bound;
  m::FileMapping head_mapping;
  r::ReadonlyRole head_role;
  std::array<r::ReadonlyRole,3> output_a;
  std::shared_ptr<w::CpuSupply> cpu;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  std::shared_ptr<w::CudaSupply> cuda;
#endif
  w::Layout layout;
  HeadRecipe head_recipe;
  n::Run head;
  std::map<std::int64_t,Step> steps;
  std::size_t state_budget=0;
};

Programs prepare(n::Backend &backend,std::shared_ptr<const r::Data> source,const std::filesystem::path &vocabulary) {
  Programs result;result.bound=r::bind(std::move(source),backend.owner);
  {
    auto recipe=p::prepare_output_a(result.bound);const auto graph=sem::make_graph(recipe.fragment);
    auto run=backend.prepare(graph.graph,constants(backend,recipe,graph),1,workload(recipe.fragment));
    const auto output=backend.execute(run,{});
    if(output.size()!=3)throw std::runtime_error("ordinary output-A preparation count");
    for(std::size_t stage=0;stage<3;++stage) {
      result.output_a[stage]={"original prepared output-A",sem::dense({8,4096,1024},c::DataType::kBFloat16),backend.owner,output[stage]};
      result.bound.roles[stage][r::index(r::Role::kOutputA)].reset();
    }
  }
  result.layout=take(w::Layout::create({result.bound.source->expert_description(0,r::Leg::kGate),
      result.bound.source->expert_description(0,r::Leg::kUp),result.bound.source->expert_description(0,r::Leg::kDown)},6),"original six-expert working set");
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  if(backend.cuda) {
    if(!backend.resident_device)throw std::runtime_error("ordinary supply has no actual CUDA resource");
    result.cuda=std::make_shared<w::CudaSupply>(take(w::CudaSupply::create(backend.context,backend.owner,*backend.resident_device,result.layout),"ordinary original CUDA working set"));
  } else
#endif
  result.cpu=std::make_shared<w::CpuSupply>(take(w::CpuSupply::create(backend.owner,result.layout),"ordinary original CPU working set"));
  result.head_mapping=take(m::FileMapping::open((vocabulary/"head.weight.bf16").string().c_str()),"original independent head");
  const auto head_bytes=take(result.head_mapping.bytes(),"original head bytes");
  if(head_bytes.size()!=std::size_t(kVocabulary*r::kHidden*2)||
      original_role_input::sha256(head_bytes)!="029e3c5293b29cc426e21d87795e15efa4d363f27b2bc4a9e3aef7d79f047919")
    throw std::runtime_error("ordinary original BF16 head identity differs");
  result.head_role=r::carrier::bind("head.weight",sem::dense({kVocabulary,r::kHidden},c::DataType::kBFloat16),head_bytes,backend.owner);
  auto &hf=result.head_recipe.fragment;
  const auto input=hf.input(sem::dense({1,r::kHidden},c::DataType::kBFloat16));
  const auto bank=p::immutable(hf,result.head_recipe.constants,result.head_role,result.head_role.description);
  const auto logits=hf.one(c::model_ops::dense_linear_f32(),{input,bank});
  const auto selected=hf.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
  for(auto value:{t::reshape(hf,selected[1],{1}),logits,selected[0]})hf.outputs.push_back(t::copy(hf,value));
  const auto hg=sem::make_graph(hf);
  result.head=backend.prepare(hg.graph,constants(backend,result.head_recipe,hg),2,workload(hf));
  result.state_budget=result.head.pool.stats().capacity_bytes;
  // These are the only requested cold positions. No graph construction in an
  // executing token loop, nor a new dynamic-branch/phase execution mechanism.
  for(const std::int64_t position:{0,1,2,3,4,126,127,128}) {
    Step step;step.recipe=p::build(result.bound,result.output_a,t::Plan::make(position,kCapacity),
        0x4541524c59000000ULL+std::uint64_t(position)*8);
    auto &f=step.recipe.fragment;const auto built=sem::make_graph(f);const auto bound=constants(backend,step.recipe,built);
    auto &run=step.run;run.source=result.bound.source;run.cpu_supply=result.cpu;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    run.cuda_supply=result.cuda;
#endif
    run.select=r::select;run.semantic_input_count=p::kSemanticInputs;run.router_value_count=6;run.resident_capacity=6;
    run.resident_slot_bytes=result.layout.slot_bytes;run.bank_ids=w::kBankIds;
    run.full_inputs=f.inputs;run.value_count=f.descriptions.size();
    run.plan=model_resident_fragment::split(f,step.recipe.routers,f.region);run.outputs=run.plan->full_outputs;
    for(const auto &phase:run.plan->phases) {
      const auto graph=sem::make_graph(phase.fragment);std::vector<g::ConstantBinding> shared;
      for(std::size_t i=0;i<phase.original_constants.size();++i) {
        const auto old=phase.original_constants[i];
        const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](Value v){return v.ordinal()==old.ordinal();});
        if(found==f.constants.end())throw std::runtime_error("ordinary split constant lost source identity");
        auto binding=bound.at(std::size_t(found-f.constants.begin()));binding.value=graph.constants[i];shared.push_back(std::move(binding));
      }
      run.phases.push_back(backend.prepare(graph.graph,shared,2,workload(phase.fragment)));
      result.state_budget+=run.phases.back().pool.stats().capacity_bytes;
    }
    result.steps.emplace(position,std::move(step));
  }
  return result;
}

struct Frontier final {std::int64_t position;std::uint64_t epoch;std::int32_t pending;};
struct State final {
  c::StateSchema schema;c::StateStore store;
  std::array<c::StateEntryId,23> entries;
  std::vector<c::TensorDesc> types;
  Frontier frontier(const c::StateVersion &version) const {
    const auto raw=read(take(version.member(entries[22]),"ordinary committed frontier"));
    if(raw.size()!=sizeof(Frontier))throw std::runtime_error("ordinary frontier size");
    Frontier result;std::memcpy(&result,raw.data(),sizeof(result));return result;
  }
  std::vector<c::ReadLease> target(const c::StateVersion &version) const {
    std::vector<c::ReadLease> result;for(std::size_t i=0;i<21;++i)result.push_back(take(version.member(entries[i]),"ordinary target state"));return result;
  }
  static State create(const Programs &programs) {
    State result;std::vector<c::StateEntrySpec> specs;std::size_t initial=0;
    for(std::size_t stage=0;stage<3;++stage)for(const auto &desc:t::descriptions(r::compression_ratio(stage),kCapacity)) {
      result.types.push_back(desc);specs.push_back({.description=desc});
      initial+=(take(desc.required_bytes(),"initial ordinary state bytes")+255u)&~std::size_t{255};
    }
    specs.push_back({.description=sem::dense({1},c::DataType::kInt32)});
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Frontier)});
    result.schema=take(c::StateSchema::create(specs),"21 ordinary caches + pending/frontier schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=6,.max_drafts=2,.max_retained_blocks=256,
        .max_backing_bytes=programs.state_budget+3*initial+4096,.max_host_snapshot_bytes=4096}),"finite ordinary state store");
    for(std::size_t i=0;i<23;++i)result.entries[i]=take(result.schema.entry(i),"ordinary typed state ID");
    return result;
  }
  c::StateVersion initial(n::Backend &backend,std::int64_t position) {
    std::vector<c::StateMember> members;std::size_t ordinal=0;
    for(std::size_t stage=0;stage<3;++stage) {
      const auto ratio=r::compression_ratio(stage);
      const auto descriptions=t::descriptions(ratio,kCapacity);
      for(std::size_t local=0;local<descriptions.size();++local,++ordinal) {
        n::Bytes payload(take(descriptions[local].required_bytes(),"initial ordinary member"));
        if(local==0||local==3||(ratio==4&&local==6)) {
          const auto encoding=local==6?t::kIndexEncoding:t::kMainEncoding;
          const auto rows=std::size_t(local==0?128:kCapacity/ratio),depth=std::size_t(encoding.depth());
          std::vector<float> values(rows*depth);
          // Valid distinct poison in inactive rows makes an accidental read
          // numerically observable. These are explicit upstream fixture bytes.
          for(std::size_t row=0;row<rows;++row)for(std::size_t col=0;col<depth;++col)
            values[row*depth+col]=float(int((row*7+col+stage)%17)-8)/16.F;
          const auto encoded=ref::encoded(encoding,values);payload.assign(reinterpret_cast<const std::byte*>(encoded.bytes.data()),
              reinterpret_cast<const std::byte*>(encoded.bytes.data()+encoded.bytes.size()));
        } else if(local+1==descriptions.size())payload=bytes<std::int32_t>(std::array{std::int32_t(position)});
        else {
          std::vector<float> value(payload.size()/sizeof(float),0.F);
          const bool score=local==2||local==5;
          if(position==0&&score)std::fill(value.begin(),value.end(),-std::numeric_limits<float>::infinity());
          else if(position)for(std::size_t i=0;i<value.size();++i)value[i]=float(int((i*3+stage)%13)-6)/32.F;
          payload=bytes<float>(value);
        }
        members.push_back({entries[ordinal],upload(backend.owner,payload)});
      }
    }
    const std::array<std::int32_t,1> prompt{3};members.push_back({entries[21],upload(backend.owner,bytes<std::int32_t>(prompt))});
    const Frontier first{position,0,prompt[0]};members.push_back({entries[22],{},std::as_bytes(std::span(&first,1))});
    return take(store.import(members),"explicit complete ordinary initial prefix");
  }
};

c::ReadLease actual(const Step &step,std::span<const c::ReadLease> outputs,std::span<const c::ReadLease> inputs,Value value) {
  const auto &f=step.recipe.fragment;
  const auto out=std::find_if(f.outputs.begin(),f.outputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
  if(out!=f.outputs.end())return outputs[std::size_t(out-f.outputs.begin())];
  const auto in=std::find_if(f.inputs.begin(),f.inputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
  if(in==f.inputs.end()||std::size_t(in-f.inputs.begin())>=inputs.size())throw std::runtime_error("ordinary observer lost its real producer");
  return inputs[std::size_t(in-f.inputs.begin())];
}
std::vector<float> floating(const c::ReadLease &lease,c::DataType type) {
  if(type==c::DataType::kFloat32)return values<float>(lease);
  if(type!=c::DataType::kBFloat16)throw std::runtime_error("ordinary observer dtype");
  const auto raw=values<std::uint16_t>(lease);std::vector<float> result;
  for(const auto value:raw)result.push_back(std::bit_cast<float>(std::uint32_t(value)<<16U));return result;
}
void advance(n::Backend &backend,Programs &programs,State &state,c::StateHead &head,std::int64_t position) {
  const auto before=take(head.current(),"ordinary expected base");const auto old=state.frontier(before);
  if(old.position!=position||values<std::int32_t>(take(before.member(state.entries[21]),"actual current input"))!=std::vector<std::int32_t>{old.pending})
    throw std::runtime_error("ordinary plan/input differs from complete committed frontier");
  auto previous=state.target(before);
  std::vector<n::Bytes> old_bytes;for(const auto &lease:previous)old_bytes.push_back(read(lease));
  for(auto cursor:kCursor)if(values<std::int32_t>(previous[cursor])!=std::vector<std::int32_t>{std::int32_t(position)})
    throw std::runtime_error("ordinary plan differs from actual layer cursor");
  auto &step=programs.steps.at(position);
  std::vector<std::uint16_t> upstream(std::size_t(4*r::kHidden));
  for(std::size_t i=0;i<upstream.size();++i)upstream[i]=glm_original_indexer_reference::detail::bfloat(
      float(int((i*3+std::size_t(old.pending))%31)-15)/32.F);
  std::vector<c::ReadLease> inputs{upload(backend.owner,bytes<std::uint16_t>(upstream))};
  inputs.insert(inputs.end(),previous.begin(),previous.end());
  auto outputs=n::execute_draft(backend,step.run,inputs); // ordinary shared native DAG runner, no draft graph
  const auto get=[&](Value v){return floating(actual(step,outputs,inputs,v),step.recipe.fragment.description(v).data_type());};
  const auto raw=[&](Value v){return read(actual(step,outputs,inputs,v));};
  const auto plan=t::Plan::make(position,kCapacity);
  for(std::size_t stage=0;stage<3;++stage) {
    ref::observe_layer(*programs.bound.source,stage,step.recipe.layers[stage],step.recipe.state_inputs[stage],plan,get,raw,false);
    const auto &a=step.recipe.layers[stage].ports.attention;
    EXPECT_EQ(a.compressed.has_value(),plan.visible(r::compression_ratio(stage))!=0);
    if(a.indexer)EXPECT_EQ(a.indexer->selection.has_value(),plan.visible(4)!=0);
    EXPECT_EQ(get(a.window_decoded).size(),std::size_t(std::min(position+1,std::int64_t{128})*512));
  }
  ref::observe_head(*programs.bound.source,step.recipe.head,get);verified();
  const std::array normalized{outputs.at(p::kNormalizedPort)};
  const auto prediction=backend.execute(programs.head,normalized);
  const auto ids=values<std::int32_t>(prediction[0]);
  const auto logits=values<float>(prediction[1]),scores=values<float>(prediction[2]);
  EXPECT_TRUE(glm_shared_vocabulary_reference::expect_logits(logits,values<std::uint16_t>(normalized[0]),
      take(programs.head_mapping.bytes(),"actual original head reference"),std::size_t(kVocabulary),std::size_t(r::kHidden),false,ids));
  EXPECT_TRUE(glm_shared_vocabulary_reference::expect_greedy(logits,ids,std::size_t(kVocabulary),scores));verified();
  auto candidate=take(state.store.begin(before),"ordinary unpublished complete state");
  for(std::size_t i=0;i<21;++i)check(candidate.replace(state.entries[i],outputs.at(i)),"ordinary target next member");
  check(candidate.replace(state.entries[21],prediction[0]),"actual pending next ID lease");
  const Frontier next{position+1,old.epoch+1,ids.at(0)};
  check(candidate.replace_metadata(state.entries[22],std::as_bytes(std::span(&next,1))),"ordinary frontier witness");
  auto version=take(candidate.finalize(),"complete ordinary candidate");
  for(auto cursor:kCursor)EXPECT_EQ(values<std::int32_t>(take(version.member(state.entries[cursor]),"candidate actual cursor")),std::vector<std::int32_t>{std::int32_t(position+1)});
  check(head.publish(before.identity(),version),"atomic ordinary state/pending/frontier publication");
  EXPECT_TRUE(head.publish(before.identity(),version).IsUnavailable());
  EXPECT_EQ(state.frontier(before).position,position);
  for(std::size_t i=0;i<21;++i)EXPECT_EQ(read(take(before.member(state.entries[i]),"old reader")),old_bytes[i]);
  EXPECT_LE(state.store.stats().versions,6U);EXPECT_LE(state.store.stats().retained_blocks,256U);
  std::printf("ordinary early Flash position=%lld next=%d window=%lld compressed4=%lld compressed128=%lld target_state_members=21\n",
      static_cast<long long>(position),ids.at(0),static_cast<long long>(std::min(position+1,std::int64_t{128})),
      static_cast<long long>(plan.visible(4)),static_cast<long long>(plan.visible(128)));
}

TEST(DSparkEarlyHistoryPlan,RejectsInvalidFrontiersAndOmitsAbsentReaders) {
  EXPECT_THROW((void)t::Plan::make(-1,kCapacity),std::runtime_error);
  EXPECT_THROW((void)t::Plan::make(0,127),std::runtime_error);
  EXPECT_THROW((void)t::Plan::make(kCapacity,kCapacity),std::runtime_error);
  for(const std::int64_t position:{0,2,3,127,128})for(const std::size_t stage:{0u,1u}) {
    sem::Fragment f;const auto param=[&](r::Role role){return f.constant(r::description(stage,role),{});};
    const auto compressor=[&](bool indexer){return t::CompressorParameters{
      param(indexer?r::Role::kIndexerKey:r::Role::kCompressorKey),param(indexer?r::Role::kIndexerGate:r::Role::kCompressorGate),
      param(indexer?r::Role::kIndexerApe:r::Role::kCompressorApe),param(indexer?r::Role::kIndexerNorm:r::Role::kCompressorNorm)};};
    t::AttentionParameters parameters{param(r::Role::kQueryA),param(r::Role::kQueryNorm),param(r::Role::kQueryB),
      param(r::Role::kKv),param(r::Role::kKvNorm),f.constant(sem::dense({8,4096,1024},c::DataType::kBFloat16),{}),
      param(r::Role::kOutputB),param(r::Role::kAttentionSink),compressor(false),{}};
    if(stage==0)parameters.indexer=t::IndexerParameters{param(r::Role::kIndexerQuery),param(r::Role::kIndexerWeights),compressor(true)};
    const auto ratio=r::compression_ratio(stage);const auto plan=t::Plan::make(position,kCapacity);
    const auto result=t::append_attention(f,f.input(sem::dense({1,r::kHidden},c::DataType::kBFloat16)),t::state_inputs(f,ratio,kCapacity),parameters,ratio,plan);
    EXPECT_EQ(result.ports.compressed.has_value(),plan.visible(ratio)!=0);
    std::size_t selections=0,readers=0;
    for(const auto &node:f.nodes) {
      if(node.operation.get_if<c::TopK>())++selections;
      if(node.operation.custom_as<c::model_ops::RowCacheGatherDecode>())++readers;
    }
    EXPECT_EQ(selections,std::size_t(ratio==4&&plan.visible(ratio)!=0));
    EXPECT_EQ(readers,std::size_t{1}+std::size_t(plan.visible(ratio)!=0));
  }
}

#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
struct SharedDevice final {std::shared_ptr<c::cuda::MemoryResource> resource;m::MemoryResourceRef ref() noexcept{return resource->ref();}};
TEST(DSparkOriginalEarlyHistory,CudaOrdinaryStartFirstCompressionAndWindowWrap) {
  const auto directory=environment("HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR"),vocabulary=environment("HYLE_DSPARK_SHARED_VOCAB_DIR");
  if(directory.empty()||vocabulary.empty())GTEST_SKIP()<<"Set existing authenticated Flash target and vocabulary directories";
  auto source=std::make_shared<r::Data>(r::Data::load(directory));
  auto context=take(g::cuda::create_context(),"ordinary early CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"ordinary early CUDA resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"ordinary early native owner");
  auto queue=take(context.create_queue(),"ordinary early Queue");n::Backend backend{context,owner,queue,true,device.get()};
  auto programs=prepare(backend,source,vocabulary);auto state=State::create(programs);
  {
    auto initial=state.initial(backend,0);auto head=take(c::StateHead::create(state.store,initial),"ordinary empty-history Head");
    for(const std::int64_t position:{0,1,2,3,4})advance(backend,programs,state,head,position);
    EXPECT_EQ(state.frontier(initial).position,0);
    EXPECT_EQ(state.frontier(take(head.current(),"ordinary startup committed head")).position,5);
  }
  check(state.store.collect(),"ordinary startup state collection");
  {
    // Explicit upstream accepted126-prefix fixture, not126 generated steps.
    auto prefix=state.initial(backend,126);auto head=take(c::StateHead::create(state.store,prefix),"ordinary given126-prefix Head");
    for(const std::int64_t position:{126,127,128})advance(backend,programs,state,head,position);
    EXPECT_EQ(state.frontier(prefix).position,126);
    EXPECT_EQ(state.frontier(take(head.current(),"ordinary wrapped committed head")).position,129);
  }
  check(state.store.collect(),"ordinary final collection");
  EXPECT_EQ(state.store.stats().versions,0U);EXPECT_EQ(state.store.stats().backing_bytes,0U);
}
#endif
} // namespace
