#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dspark_native_consumer.h"
#include "dspark_original_state_consumer.h"
#include "dspark_original_draft_program.h"
#include "dspark_original_draft_reference.h"
#include "dspark_original_target_program.h"
#include "dspark_original_target_reference.h"
#include "dspark_original_target_weights.h"
#include "glm_shared_vocabulary_reference.h"
#include "hyle/compute/state_versions.h"

namespace {
namespace native=dspark_native_consumer;
namespace own=dspark_original_formal_roles;
namespace draft=dspark_original_draft_fragment;
namespace draft_program=dspark_original_draft_program;
namespace draft_reference=dspark_original_draft_reference;
namespace target=dspark_original_target_fragment;
namespace target_program=dspark_original_target_program;
namespace target_reference=dspark_original_target_reference;
namespace weights=dspark_original_target_weights;
namespace policy=dspark_original_state_consumer;
namespace lm=glm_shared_vocabulary_reference;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using native::take;using native::check;using native::upload;using native::read;
using native::values;using native::bytes;using native::dense;
using native::Backend;using native::Run;using native::Bytes;
using Value=sem::ValueId;
constexpr std::size_t kMembers=30,kPending=29,kTargetFirst=8;
struct Fixture final {
  std::int64_t position,capacity;
  const c::ModelArtifactProfileDescription *profile;
  const char *expert_directory,*frontier_file,*wait_flag,*scope;
};
inline constexpr Fixture kProFixture{4351,4368,&cpu_c4_model_branches::pro_profile,
    "HYLE_DSPARK_PRO_EXPERT_DIR","HYLE_DSPARK_PRO_FRONTIER_FILE","HYLE_DSPARK_PRO_WAIT_FOR_EXPERTS",
    "Original Pro0813 target58/59/60+distinctfinalHC/norm and formal mtp.0/1/2+independent BF16 vocabulary, all384 global expert IDs with actual selected E2M1/UE8 experts, explicit PowerOfTwo/floor1e-4 A128, fullV K5 and atomic30member state. Natural once and labelled controlled0/2/5 prefix policies; actual pending continuation, immutable old readers and bounded backing. Layer57 input and initial valid cacheprefix are finite synthetic fixtures; no complete61-layer target, quality, natural acceptance-rate or TPS claim. Missing original experts fail; external acquisition wait is explicit fixture mode."};
inline constexpr Fixture kFlashFixture{2173,2184,&cpu_c4_model_branches::flash_profile,
    "HYLE_DSPARK_FLASH_EXPERT_DIR","HYLE_DSPARK_FLASH_FRONTIER_FILE","HYLE_DSPARK_FLASH_WAIT_FOR_EXPERTS",
    "Original Flash0731 target40/41/42 and distinct finalHC/norm borrowed from their authenticated original source, formal mtp.0/1/2 from the independent97-role E4/BF16 bundle, and the two independent original BF16 vocabulary mappings. Full256 global expert domain with actual selected original E2M1/UE8 experts; target experts borrowed without reacquisition, draft missing experts fail or await explicit external supply. Same fullV K5/atomic30member state and actual pending continuation. Natural once and labelled controlled0/2/5 prefix policies. Layer39 input and initial valid cacheprefix are finite synthetic fixtures; no complete43-layer target, quality, acceptance-rate or TPS claim."};


std::string environment(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t count=0;
  if(_dupenv_s(&value,&count,name))throw std::runtime_error("Pro environment lookup failed");
  const std::unique_ptr<char,decltype(&std::free)> cleanup(value,&std::free);return value?value:"";
#else
  const auto value=std::getenv(name);return value?value:"";
#endif
}
void verified() {
  if(::testing::Test::HasFailure())throw std::runtime_error("original Pro local reference failed before StateHead publication");
}
c::WorkloadContext workload(const sem::Fragment &f) {
  if(!f.profile)throw std::runtime_error("original Pro cold Fragment has no model profile");
  return {take(c::ModelArtifactProfile::create(*f.profile),"actual Pro model profile"),f.phase,
      {0x48594c454d4f444cULL,f.region,1}};
}
std::string file_text(const std::filesystem::path &path) {
  // The ordinary Hyle file mapping shares deletion on Windows, so the
  // external writer can atomically replace a receipt while it is polled.
  auto mapped=take(hyle::mem::FileMapping::open(path.string().c_str()),"map original Pro expert receipt");
  if(mapped.size()>1024*1024)throw std::runtime_error("original Pro expert receipt exceeds metadata bound");
  const auto raw=take(mapped.bytes(),"original Pro expert receipt bytes");
  return {reinterpret_cast<const char*>(raw.data()),raw.size()};
}
std::optional<std::string> catalogue_pin(const std::filesystem::path &directory) {
  const auto text=file_text(directory/"acquisition.json");
  auto complete=text.find("\"complete\"");
  if(complete==std::string::npos||(complete=text.find(':',complete))==std::string::npos)
    throw std::runtime_error("original Pro expert receipt lacks completion");
  complete=text.find_first_not_of(" \r\n\t",complete+1);
  if(complete!=std::string::npos&&text.compare(complete,5,"false")==0)return std::nullopt;
  if(complete==std::string::npos||text.compare(complete,4,"true")!=0)
    throw std::runtime_error("original Pro expert acquisition is incomplete");
  auto begin=text.find("\"typed_catalog\"");
  if(begin==std::string::npos||(begin=text.find("\"sha256\"",begin))==std::string::npos||
      (begin=text.find(':',begin))==std::string::npos||(begin=text.find('"',begin))==std::string::npos)
    throw std::runtime_error("original Pro expert receipt lacks its explicit typed catalogue pin");
  const auto end=text.find('"',++begin);const auto pin=text.substr(begin,end-begin);
  if(pin.size()!=64||std::any_of(pin.begin(),pin.end(),[](char ch){return !(ch>='0'&&ch<='9')&&!(ch>='a'&&ch<='f');}))
    throw std::runtime_error("original Pro expert typed catalogue pin differs");
  return pin;
}
struct FrontierSupply final {
  std::shared_ptr<own::Data> data;
  std::filesystem::path directory,request;
  bool wait=false;
  std::size_t requests=0,attachments=0;
  void ensure(own::Section section,std::size_t stage,std::span<const std::int32_t> globals) {
    const char *label=section==own::Section::kTarget?"target":"draft";
    const auto started=std::chrono::steady_clock::now();bool announced=false;
    for(;;) {
      std::vector<std::int32_t> missing;
      for(const auto id:globals) {
        if(data->has_expert(section,stage,id))continue;
        std::ostringstream leaf;leaf<<"expert"<<std::setfill('0')<<std::setw(3)<<id;
        const auto path=directory/label/("stage"+std::to_string(stage))/leaf.str();
        if(!std::filesystem::exists(path/"acquisition.json")) {missing.push_back(id);continue;}
        // This explicit step follows a completed/reset router phase. Source
        // lookup and upload never download or manufacture an original role.
        const auto pin=catalogue_pin(path);
        if(!pin) {missing.push_back(id);continue;}
        const std::array only{id};
        auto acquired=std::make_shared<own::ExpertSet>(own::ExpertSet::load(path,*pin,section,stage,only,data->model()));
        data->attach_experts(std::move(acquired));++attachments;
      }
      if(missing.empty())return;
      if(!announced) {
        if(!request.empty()) {
          if(!request.parent_path().empty())std::filesystem::create_directories(request.parent_path());
          auto temporary=request;temporary+=".tmp";
          std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
          if(!out)throw std::runtime_error("cannot write actual Pro expert frontier request");
          const auto list=[&](std::span<const std::int32_t> ids) {out<<'[';for(std::size_t i=0;i<ids.size();++i){if(i)out<<',';out<<ids[i];}out<<']';};
          out<<"{\"section\":\""<<label<<"\",\"stage\":"<<stage<<",\"global_ids\":";list(globals);
          out<<",\"missing_ids\":";list(missing);out<<"}\n";out.close();
          if(!out)throw std::runtime_error("cannot finish actual Pro expert frontier request");
#if defined(_WIN32)
          if(!MoveFileExW(temporary.c_str(),request.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot atomically publish actual Pro expert frontier request");
#else
          std::filesystem::rename(temporary,request);
#endif
        }
        ++requests;announced=true;
        std::printf("DSpark Pro actual expert frontier section=%s stage=%zu missing=%zu; %s\n",label,stage,missing.size(),wait?"waiting for explicit external acquisition":"missing acquisition fails");std::fflush(stdout);
      }
      if(!wait)throw std::runtime_error("original Pro selected expert bytes are not acquired; actual frontier request recorded");
      if(std::chrono::steady_clock::now()-started>std::chrono::minutes(20))
        throw std::runtime_error("original Pro external expert acquisition exceeded bounded frontier wait");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
};
using Routed=native::RoutedRun<own::SourceView,weights::CpuSupply,dspark_formal_fragment::Boundary,own::Selection
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    ,weights::CudaSupply
#endif
    >;
struct Supply final {
  std::shared_ptr<const own::SourceView> source;
  std::shared_ptr<weights::CpuSupply> cpu;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  std::shared_ptr<weights::CudaSupply> cuda;
#endif
  weights::Layout layout;
};
Supply supply(Backend &backend,std::shared_ptr<own::Data> data,own::Section section,std::size_t capacity) {
  Supply result;result.source=std::make_shared<own::SourceView>(own::source_view(data,section));
  result.layout=take(weights::Layout::create({result.source->expert_description(0,own::Leg::kGate),
      result.source->expert_description(0,own::Leg::kUp),result.source->expert_description(0,own::Leg::kDown)},capacity),"original Pro actual FP4 residency layout");
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  if(backend.cuda) {
    if(!backend.resident_device)throw std::runtime_error("original Pro supply needs the same actual compiled IO CUDA resource");
    result.cuda=std::make_shared<weights::CudaSupply>(take(weights::CudaSupply::create(backend.context,backend.owner,
        *backend.resident_device,result.layout),"same actual Pro device expert slots"));
  } else
#endif
  result.cpu=std::make_shared<weights::CpuSupply>(take(weights::CpuSupply::create(backend.owner,result.layout),"original Pro bounded CPU expert pool"));
  return result;
}
template<class Recipe> std::vector<g::ConstantBinding> constants(Backend &backend,const Recipe &recipe,const sem::GraphCase &built) {
  const auto &f=recipe.fragment;std::vector<g::ConstantBinding> result;
  for(std::size_t i=0;i<f.constants.size();++i) {
    const auto value=f.constants[i];const auto found=std::find_if(recipe.constants.begin(),recipe.constants.end(),[&](const auto &entry){return entry.value.ordinal()==value.ordinal();});
    if(found!=recipe.constants.end())result.push_back(take(found->role.binding(built.constants[i]),"same actual Pro immutable owner/lease"));
    else {
      auto lease=upload(backend.owner,f.constant_bytes.at(i));
      result.push_back({.value=built.constants[i],.tensor=take(lease.tensor(f.description(value)),"small Pro plan constant"),.owner=backend.owner,.lease=lease});
    }
  }
  return result;
}
template<class Recipe> struct Plain final {Recipe recipe;Run run;};
template<class Recipe> Plain<Recipe> prepare_plain(Backend &backend,Recipe recipe,std::uint32_t generations=2) {
  Plain<Recipe> result{.recipe=std::move(recipe)};const auto built=sem::make_graph(result.recipe.fragment);
  const auto bound=constants(backend,result.recipe,built);
  result.run=backend.prepare(built.graph,bound,generations,workload(result.recipe.fragment));return result;
}
template<class Recipe> struct Phased final {Recipe recipe;Routed run;};
template<class Recipe> Phased<Recipe> prepare_phased(Backend &backend,Recipe recipe,Supply &s,FrontierSupply &frontier,
    own::Section section,std::size_t semantic,std::size_t router_count) {
  Phased<Recipe> result{.recipe=std::move(recipe)};auto &f=result.recipe.fragment;auto &run=result.run;
  const auto built=sem::make_graph(f);const auto bound=constants(backend,result.recipe,built);
  run.source=s.source;run.cpu_supply=s.cpu;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  run.cuda_supply=s.cuda;
#endif
  run.semantic_input_count=semantic;run.router_value_count=router_count;run.resident_capacity=s.layout.capacity;
  run.resident_slot_bytes=s.layout.slot_bytes;run.bank_ids=weights::kBankIds;
  run.select=[model=s.source->source->geometry(),capacity=s.layout.capacity](std::span<const std::int32_t> ids) {
    return own::select(ids,capacity,model);
  };
  run.before_upload=[&frontier,section](std::size_t stage,std::span<const std::int32_t> ids){frontier.ensure(section,stage,ids);};
  run.full_inputs=f.inputs;run.value_count=f.descriptions.size();
  run.plan=model_resident_fragment::split(f,result.recipe.routers,f.region);run.outputs=run.plan->full_outputs;
  for(const auto &phase:run.plan->phases) {
    const auto graph=sem::make_graph(phase.fragment);std::vector<g::ConstantBinding> shared;
    for(std::size_t i=0;i<phase.original_constants.size();++i) {
      const auto source=phase.original_constants[i];const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](Value v){return v.ordinal()==source.ordinal();});
      if(found==f.constants.end())throw std::runtime_error("Pro phase immutable Value has no complete-Fragment identity");
      auto binding=bound.at(std::size_t(found-f.constants.begin()));binding.value=graph.constants[i];shared.push_back(std::move(binding));
    }
    run.phases.push_back(backend.prepare(graph.graph,shared,2,workload(phase.fragment)));
  }
  return result;
}
struct DraftRecipe final {
  sem::Fragment fragment;
  std::shared_ptr<const own::Data> source;
  std::vector<draft_program::Constant> constants;
  draft::Inputs inputs;
  draft::Result result;
  std::array<dspark_formal_fragment::Boundary,3> routers;
};
DraftRecipe complete_draft(draft_program::Recipe recipe) {
  const auto routers=recipe.result.routers;
  return {std::move(recipe.fragment),std::move(recipe.source),std::move(recipe.constants),recipe.inputs,recipe.result,routers};
}
template<class Recipe> c::ReadLease actual(const Recipe &recipe,std::span<const c::ReadLease> output,
    std::span<const c::ReadLease> input,Value value) {
  const auto &f=recipe.fragment;
  const auto found=std::find_if(f.outputs.begin(),f.outputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
  if(found!=f.outputs.end())return output[std::size_t(found-f.outputs.begin())];
  const auto incoming=std::find_if(f.inputs.begin(),f.inputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
  if(incoming==f.inputs.end()||std::size_t(incoming-f.inputs.begin())>=input.size())
    throw std::runtime_error("Pro observer Value has no actual semantic producer lease");
  return input[std::size_t(incoming-f.inputs.begin())];
}
template<class Recipe> auto floats(const Recipe &recipe,std::span<const c::ReadLease> output,std::span<const c::ReadLease> input) {
  return [&recipe,output,input](Value value) {
    const auto lease=actual(recipe,output,input,value);
    const auto type=recipe.fragment.description(value).data_type();
    if(type==c::DataType::kFloat32)return values<float>(lease);
    if(type!=c::DataType::kBFloat16)throw std::runtime_error("Pro actual floating observation dtype differs");
    const auto source=values<std::uint16_t>(lease);std::vector<float> result;result.reserve(source.size());
    for(const auto bits:source)result.push_back(std::bit_cast<float>(std::uint32_t(bits)<<16U));return result;
  };
}
struct Verifier final {
  sem::Fragment fragment{.profile=&cpu_c4_model_branches::pro_profile,.region=0x0ca10400,.quantized=true};
  std::vector<draft_program::Constant> constants;
  Value logits,scores;
  std::vector<Value> ids;
};
Verifier verifier(const own::Bound &bound,std::int64_t rows,const c::ModelArtifactProfileDescription *profile) {
  if(rows!=1&&rows!=6)throw std::runtime_error("Pro verifier needs one or six actual normalized target rows");
  Verifier result;result.fragment.region+=std::uint64_t(rows);
  result.fragment.profile=draft_program::checked_profile(*bound.source,profile);
  auto &f=result.fragment;const auto &model=bound.source->geometry();Value joined;
  for(std::int64_t i=0;i<rows;++i) {
    const auto input=f.input(sem::dense({1,model.hidden},c::DataType::kBFloat16));
    joined=i?draft::ds::concat(f,joined,input,0):input;
  }
  const auto head=draft_program::immutable(f,result.constants,bound.at("head.weight"),bound.at("head.weight").description);
  result.logits=f.one(c::model_ops::bfloat_linear_f32(),{joined,head});
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{result.logits},2);result.scores=top[0];
  for(std::int64_t i=0;i<rows;++i) {
    auto id=draft::math::reshape(f,draft::ds::slice(f,top[1],0,i,1),{1});id=draft::math::copy(f,id);f.outputs.push_back(id);result.ids.push_back(id);
  }
  for(auto *value:{&result.logits,&result.scores}) {*value=draft::math::copy(f,*value);f.outputs.push_back(*value);}
  return result;
}
void report_head(const Run &run,std::size_t row_count) {
  const auto copied=take(run.graph.producer(take(run.graph.output_id(row_count),"Pro fullV diagnostic")),"Pro fullV Copy producer");
  if(!take(run.graph.operation(copied),"Pro fullV Copy operation").get_if<c::Copy>())throw std::runtime_error("Pro fullV diagnostic must own its Copy");
  const auto projected=take(run.graph.producer(take(run.graph.node_input(copied,0),"Pro fullV Copy predecessor")),"Pro actual fullV operation");
  if(!take(run.graph.operation(projected),"Pro original fullV operation").custom_as<c::model_ops::BFloatLinearF32>())
    throw std::runtime_error("Pro actual fullV must retain BFloatLinearF32");
  std::size_t selected=0;for(std::size_t i=0;i<run.compiled.report.step_count();++i) {
    const auto step=take(run.compiled.report.step(i),"Pro fullV selected provider");
    for(std::size_t j=0;j<step.node_count;++j)if(take(run.compiled.report.step_node(i,j),"Pro fullV actual node")==projected) {
      ++selected;::testing::Test::RecordProperty("target_fullV"+std::to_string(row_count)+"_provider",std::string(step.provider.name));
    }
  }
  if(selected!=1)throw std::runtime_error("Pro fullV head must execute once in a selected native step");
}
struct CaptureJoin final {
  sem::Fragment fragment{.profile=&cpu_c4_model_branches::pro_profile,.region=0x0ca22500,.quantized=true};
  std::vector<draft_program::Constant> constants;
};
CaptureJoin capture_join(const dspark_target_geometry::Geometry &model,const c::ModelArtifactProfileDescription *profile) {
  CaptureJoin result;result.fragment.profile=profile;auto &f=result.fragment;
  for(std::size_t stage=0;stage<3;++stage) {
    Value joined;for(std::size_t row=0;row<6;++row) {
      const auto next=f.input(sem::dense({1,4,model.hidden}));joined=row?draft::ds::concat(f,joined,next,0):next;
    }
    f.outputs.push_back(draft::math::copy(f,joined));
  }
  return result;
}
struct State final {
  c::StateSchema schema;c::StateStore store;std::array<c::StateEntryId,kMembers> entries;
  std::vector<c::TensorDesc> descriptions;
  static State create(std::span<const c::TensorDesc> target_descriptions,std::size_t budget,
      const dspark_target_geometry::Geometry &model) {
    if(target_descriptions.size()!=21)throw std::runtime_error("Pro target state must retain all21 actual cache resources");
    State result;std::vector<c::StateEntrySpec> specs;
    for(std::size_t i=0;i<3;++i)result.descriptions.push_back(take(c::row_cache_carrier_description(draft::kEncoding,128),"Pro main history schema"));
    result.descriptions.push_back(dense(c::DataType::kFloat32,{1,model.hidden}));
    for(std::size_t i=0;i<3;++i)result.descriptions.push_back(dense(c::DataType::kFloat32,{1,4,model.hidden}));
    for(const auto &description:result.descriptions)specs.push_back({.description=description});
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(policy::Boundary)});
    for(const auto &description:target_descriptions)specs.push_back({.description=description});
    specs.push_back({.description=dense(c::DataType::kInt32,{1})});
    result.schema=take(c::StateSchema::create(specs),"Pro single30member schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=16,.max_drafts=4,.max_retained_blocks=600,
        .max_backing_bytes=budget}),"Pro bounded StateStore");
    for(std::size_t i=0;i<kMembers;++i)result.entries[i]=take(result.schema.entry(i),"Pro typed state member");return result;
  }
  policy::Boundary frontier(const c::StateVersion &version) const {
    const auto data=read(take(version.member(entries[7]),"Pro actual protocol metadata"));
    if(data.size()!=sizeof(policy::Boundary))throw std::runtime_error("Pro metadata extent");
    policy::Boundary result;std::memcpy(&result,data.data(),data.size());return result;
  }
  std::vector<c::ReadLease> members(const c::StateVersion &version,std::size_t first,std::size_t count) const {
    std::vector<c::ReadLease> result;for(std::size_t i=0;i<count;++i)result.push_back(take(version.member(entries.at(first+i)),"Pro retained state lease"));return result;
  }
  c::StateVersion selected(const c::StateVersion &from,std::span<const c::ReadLease> history,
      std::span<const c::ReadLease> captures,std::span<const c::ReadLease> target_state,const c::ReadLease &pending,
      std::uint32_t consumed,std::uint64_t epoch) {
    auto facts=frontier(from);const auto ids=values<std::int32_t>(pending);
    if(history.size()!=4||captures.size()!=3||target_state.size()!=21||!consumed||consumed>6||epoch<=facts.epoch||
        ids.size()!=1||ids[0]<0||ids[0]>=draft::kVocabulary)throw std::runtime_error("Pro selected prefix/epoch/pending contract");
    for(const auto cursor:{7u,12u,20u})if(values<std::int32_t>(target_state[cursor])!=std::vector<std::int32_t>{std::int32_t(facts.next_position+consumed)})
      throw std::runtime_error("Pro target selected cursor differs from mandatory-current plus accepted prefix");
    auto staged=take(store.begin(from),"Pro complete unpublished StateDraft");
    for(std::size_t i=0;i<4;++i)check(staged.replace(entries[i],history[i]),"Pro accepted main state");
    for(std::size_t i=0;i<3;++i)check(staged.replace(entries[4+i],captures[i]),"Pro accepted actual captures");
    for(std::size_t i=0;i<21;++i)check(staged.replace(entries[8+i],target_state[i]),"Pro accepted complete target state");
    check(staged.replace(entries[kPending],pending),"Pro pending is actual verifier row lease");
    facts.epoch=epoch;facts.next_position+=consumed;facts.consumed+=consumed;facts.pending=ids[0];
    check(staged.replace_metadata(entries[7],std::as_bytes(std::span(&facts,1))),"Pro accepted epoch/frontier");
    return take(staged.finalize(),"Pro complete atomic candidate");
  }
};

void run(std::shared_ptr<own::Data> source,c::Context &context,const c::StorageOwner &owner,bool cuda,const Fixture &setup
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ) {
  const auto &model=source->geometry();model.validate();
  (void)draft_program::checked_profile(*source,setup.profile);
  auto queue=take(context.create_queue(),"Pro single native Queue");Backend backend{context,owner,queue,cuda};
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  backend.resident_device=device;
#endif
  FrontierSupply acquisition{source,environment(setup.expert_directory),environment(setup.frontier_file),environment(setup.wait_flag)=="1"};
  auto bound=own::bind(source,owner);auto target_bound=own::target_bindings(bound);
  std::array<own::ReadonlyRole,3> target_oa,draft_oa;std::vector<c::LeasePool> cold_pools;
  const auto prepare_oa=[&](auto cold,auto &ready) {
    auto prepared=prepare_plain(backend,std::move(cold),1);auto output=backend.execute(prepared.run,{});
    if(output.size()!=3)throw std::runtime_error("Pro cold OA preparation must return three actual BF16 banks");
    for(std::size_t stage=0;stage<3;++stage)ready[stage]={"Pro original prepared OA",dense(c::DataType::kBFloat16,
        {model.groups,model.group_width(),1024}),owner,output[stage]};
    cold_pools.push_back(std::move(prepared.run.pool));
  };
  prepare_oa(target_program::prepare_output_a(target_bound,0x0ca20000,model,setup.profile),target_oa);
  prepare_oa(draft_program::prepare_output_a(bound,0x0ca10001,setup.profile),draft_oa);
  for(std::size_t stage=0;stage<3;++stage) {
    bound.roles.at(own::layer_name(source->model(),own::Section::kTarget,stage,"attn.wo_a.weight")).lease={};
    bound.roles.at(own::layer_name(source->model(),own::Section::kDraft,stage,"attn.wo_a.weight")).lease={};
    target_bound.roles[stage][dspark_original_target_roles::index(dspark_original_target_roles::Role::kOutputA)].reset();
  }
  auto target_supply=supply(backend,source,own::Section::kTarget,6),draft_supply=supply(backend,source,own::Section::kDraft,30);
  auto proposal=prepare_phased(backend,complete_draft(draft_program::build(bound,draft_oa,0x0ca10100,setup.profile)),draft_supply,acquisition,own::Section::kDraft,6,30);
  std::vector<Phased<target_program::Recipe>> targets;
  for(std::int64_t row=0;row<=6;++row)targets.push_back(prepare_phased(backend,target_program::build(target_bound,target_oa,
      target::Plan::make(setup.position+row,setup.capacity,model),0x0ca21000+std::uint64_t(row)*8,setup.profile),
      target_supply,acquisition,own::Section::kTarget,22,6));
  auto main6=prepare_plain(backend,draft_program::build_main_projection(bound,6,0x0ca22000,setup.profile));
  auto history6=prepare_plain(backend,draft_program::build_main_history(bound,6,0x0ca22100,setup.profile));
  auto main1=prepare_plain(backend,draft_program::build_main_projection(bound,1,0x0ca22200,setup.profile));
  auto history1=prepare_plain(backend,draft_program::build_main_history(bound,1,0x0ca22300,setup.profile));
  auto joined_captures=prepare_plain(backend,capture_join(model,setup.profile),1);
  auto verify6=prepare_plain(backend,verifier(bound,6,setup.profile));auto verify1=prepare_plain(backend,verifier(bound,1,setup.profile));
  report_head(verify6.run,6);report_head(verify1.run,1);
  std::vector<c::TensorDesc> target_descriptions;
  for(std::size_t stage=0;stage<3;++stage) {
    const auto descriptions=target::descriptions(dspark_original_target_roles::compression_ratio(stage),setup.capacity,model);
    target_descriptions.insert(target_descriptions.end(),descriptions.begin(),descriptions.end());
  }
  std::size_t budget=0;const auto add=[&](std::size_t amount) {
    if(amount>std::numeric_limits<std::size_t>::max()-budget)throw std::runtime_error("Pro StateStore actual backing budget overflow");budget+=amount;
  };
  for(const auto &part:proposal.run.phases)add(part.pool.stats().capacity_bytes);
  for(const auto &plan:targets)for(const auto &part:plan.run.phases)add(part.pool.stats().capacity_bytes);
  for(const auto *part:{&main6.run,&history6.run,&main1.run,&history1.run,&joined_captures.run,&verify6.run,&verify1.run})add(part->pool.stats().capacity_bytes);
  const auto initial_budget=[&](std::size_t amount){if(amount>std::numeric_limits<std::size_t>::max()-255)throw std::runtime_error("Pro initial state alignment overflow");add((amount+255)&~std::size_t{255});};
  for(const auto &description:target_descriptions)initial_budget(take(description.required_bytes(),"Pro actual target initial backing"));
  for(std::size_t i=0;i<3;++i)initial_budget(128*std::size_t(draft::kEncoding.row_bytes()));
  initial_budget(std::size_t(model.hidden)*sizeof(float));for(std::size_t i=0;i<3;++i)initial_budget(std::size_t(4*model.hidden)*sizeof(float));
  initial_budget(sizeof(std::int32_t));add(22*((sizeof(policy::Boundary)+63)&~std::size_t{63}));
  auto state=State::create(target_descriptions,budget,model);std::size_t peak=0;
  const auto account=[&]{const auto stats=state.store.stats();peak=std::max(peak,stats.backing_bytes);EXPECT_LE(stats.backing_bytes,budget);EXPECT_LE(stats.versions,16U);EXPECT_LE(stats.drafts,4U);};
  std::vector<c::StateMember> members;
  for(std::size_t i=0;i<7;++i)members.push_back({state.entries[i],upload(owner,Bytes(take(state.descriptions[i].required_bytes(),"Pro finite draft initial fixture")))});
  const policy::Boundary initial{19,1,1,setup.position,3,0};members.push_back({state.entries[7],{},std::as_bytes(std::span(&initial,1))});
  std::size_t ordinal=0;
  for(std::size_t stage=0;stage<3;++stage) {
    const auto descriptions=target::descriptions(dspark_original_target_roles::compression_ratio(stage),setup.capacity,model);
    for(std::size_t i=0;i<descriptions.size();++i,++ordinal) {
      Bytes fixture(take(descriptions[i].required_bytes(),"Pro finite target initial fixture"));
      if(i+1==descriptions.size())fixture=bytes<std::int32_t>(std::array{std::int32_t(setup.position)});
      members.push_back({state.entries[8+ordinal],upload(owner,fixture)});
    }
  }
  members.push_back({state.entries[kPending],upload(owner,bytes<std::int32_t>(std::array<std::int32_t,1>{3}))});
  auto base=take(state.store.import(members),"Pro complete finite initial30member state");members.clear();account();
  const auto initial_ring=read(take(base.member(state.entries[0]),"Pro retained old ring"));
  const auto initial_pending=take(base.member(state.entries[kPending]),"Pro retained pending");
  std::size_t draft_episodes=0,target_rows=0,main_calls=0,history_calls=0,head_calls=0,full_values=0,feedback=0;
  const auto draft_from=[&](const c::StateVersion &version) {
    const auto facts=state.frontier(version);const auto pending=take(version.member(state.entries[kPending]),"Pro actual pending feedback lease");
    if(values<std::int32_t>(pending)!=std::vector<std::int32_t>{facts.pending})throw std::runtime_error("Pro metadata token differs from actual pending lease");
    std::array<std::int32_t,5> positions;std::iota(positions.begin(),positions.end(),std::int32_t(facts.next_position));
    std::vector<float> mask(std::size_t(5*(128+5)),0.F);
    std::vector<c::ReadLease> input{pending,upload(owner,bytes<std::int32_t>(positions)),upload(owner,bytes<float>(mask))};
    auto rings=state.members(version,0,3);input.insert(input.end(),rings.begin(),rings.end());
    auto output=native::execute_draft(backend,proposal.run,input);
    const auto get=floats(proposal.recipe,output,input);const auto raw=[&](Value value){return read(actual(proposal.recipe,output,input,value));};
    draft_reference::observe(*source,proposal.recipe.result,get,raw,draft_episodes==0);verified();++draft_episodes;++feedback;return output;
  };
  struct Row final {std::array<c::ReadLease,21> state;std::array<c::ReadLease,3> captures;c::ReadLease normalized;};
  const auto target_row=[&](std::int64_t position,std::int32_t token,std::span<const c::ReadLease> previous) {
    if(position<setup.position||position>setup.position+6||previous.size()!=21||token<0||token>=draft::kVocabulary)throw std::runtime_error("Pro target row lies outside its cold frontier");
    for(const auto cursor:{7u,12u,20u})if(values<std::int32_t>(previous[cursor])!=std::vector<std::int32_t>{std::int32_t(position)})
      throw std::runtime_error("Pro actual target cursor differs from the cold Plan");
    std::vector<std::uint16_t> fixture(std::size_t(4*model.hidden));
    for(std::size_t i=0;i<fixture.size();++i)fixture[i]=row_cache_test::bfloat(float(std::int64_t((i*13+std::size_t(token)*7+std::size_t(position)*3)%127)-63)/256.F);
    std::vector<c::ReadLease> input{upload(owner,bytes<std::uint16_t>(fixture))};input.insert(input.end(),previous.begin(),previous.end());
    auto &plan=targets.at(std::size_t(position-setup.position));auto output=native::execute_draft(backend,plan.run,input);
    const auto get=floats(plan.recipe,output,input);const auto raw=[&](Value value){return read(actual(plan.recipe,output,input,value));};
    const auto current=target::Plan::make(position,setup.capacity,model);
    for(std::size_t stage=0;stage<3;++stage)target_reference::observe_layer(*target_supply.source,stage,plan.recipe.layers[stage],plan.recipe.state_inputs[stage],current,get,raw,target_rows==0);
    target_reference::observe_head(*target_supply.source,plan.recipe.head,get);verified();
    Row row;for(std::size_t i=0;i<21;++i)row.state[i]=output[i];for(std::size_t i=0;i<3;++i)row.captures[i]=output[21+i];row.normalized=output[24];++target_rows;return row;
  };
  const auto catchup=[&](std::span<const Row> rows,std::span<const c::ReadLease> old,std::int64_t position) {
    if(rows.empty()||(rows.size()!=1&&rows.size()!=6)||old.size()!=3)throw std::runtime_error("Pro accepted main-history row contract");
    auto &projection=rows.size()==6?main6:main1;auto &history=rows.size()==6?history6:history1;
    std::vector<c::ReadLease> captures;
    if(rows.size()==1)captures.assign(rows[0].captures.begin(),rows[0].captures.end());
    else {
      std::vector<c::ReadLease> input;
      for(std::size_t stage=0;stage<3;++stage)for(const auto &row:rows)input.push_back(row.captures[stage]);
      captures=backend.execute(joined_captures.run,input);
    }
    auto projected=backend.execute(projection.run,captures);
    draft_reference::observe_main(*source,projection.recipe.ports,floats(projection.recipe,projected,captures),main_calls==0);verified();++main_calls;
    std::vector<std::int32_t> positions(rows.size()),slots(rows.size());std::iota(positions.begin(),positions.end(),std::int32_t(position));
    for(std::size_t i=0;i<slots.size();++i)slots[i]=positions[i]%128;
    std::vector<c::ReadLease> input{projected[0],upload(owner,bytes<std::int32_t>(positions)),upload(owner,bytes<std::int32_t>(slots))};input.insert(input.end(),old.begin(),old.end());
    auto output=backend.execute(history.run,input);
    const auto raw=[&](Value value){return read(actual(history.recipe,output,input,value));};
    draft_reference::observe_history(*source,history.recipe.ports,floats(history.recipe,output,input),raw,history_calls==0);verified();++history_calls;return output;
  };
  const auto verify_rows=[&](std::span<const Row> rows) {
    auto &plan=rows.size()==6?verify6:verify1;std::vector<c::ReadLease> input;std::vector<std::uint16_t> normalized;
    for(const auto &row:rows) {input.push_back(row.normalized);auto bits=values<std::uint16_t>(row.normalized);normalized.insert(normalized.end(),bits.begin(),bits.end());}
    auto output=backend.execute(plan.run,input);const auto logits=values<float>(output.at(rows.size())),scores=values<float>(output.at(rows.size()+1));
    std::vector<std::int32_t> ids;for(std::size_t i=0;i<rows.size();++i) {auto current=values<std::int32_t>(output[i]);if(current.size()!=1)throw std::runtime_error("Pro verifier row needs its own I32 lease");ids.push_back(current[0]);}
    EXPECT_TRUE(lm::expect_logits(logits,normalized,source->raw("head.weight"),std::size_t(draft::kVocabulary),std::size_t(model.hidden),head_calls==0,ids));
    EXPECT_TRUE(lm::expect_greedy(logits,ids,std::size_t(draft::kVocabulary),scores));verified();
    if(!head_calls)full_values=logits.size();++head_calls;return output;
  };
  {
    auto proposal_result=draft_from(base);const auto proposal_ids=values<std::int32_t>(proposal_result[3]);const auto retained_proposal=read(proposal_result[3]);
    if(proposal_ids.size()!=6)throw std::runtime_error("formal Pro draft must return current plus five actual IDs");
    std::array<Row,6> rows;auto prior=state.members(base,kTargetFirst,21);
    for(std::size_t i=0;i<rows.size();++i) {rows[i]=target_row(setup.position+std::int64_t(i),proposal_ids[i],prior);prior.assign(rows[i].state.begin(),rows[i].state.end());}
    auto rings=state.members(base,0,3);auto histories=catchup(rows,rings,setup.position);auto verified_rows=verify_rows(rows);
    std::vector<std::int32_t> target_ids;for(std::size_t i=0;i<6;++i)target_ids.push_back(values<std::int32_t>(verified_rows[i])[0]);
    const auto decision=take(policy::decide_block5(proposal_ids,target_ids),"Pro actual fullV K5 decision");
    auto head=take(c::StateHead::create(state.store,base),"Pro one complete StateHead");
    const auto select=[&](std::uint32_t accepted,std::uint64_t epoch) {
      const auto &row=rows.at(accepted);return state.selected(base,std::span(histories).subspan(std::size_t(accepted)*4,4),row.captures,row.state,verified_rows[accepted],accepted+1,epoch);
    };
    auto accepted=select(decision.accepted_drafts,2);check(head.publish(base.identity(),accepted),"Pro natural complete30member publication");
    auto current=take(head.current(),"Pro actual accepted prefix");account();
    const auto pending=take(current.member(state.entries[kPending]),"Pro selected actual target correction/bonus");
    EXPECT_EQ(read(pending),read(verified_rows[decision.pending_row]));EXPECT_EQ(state.frontier(current).next_position,setup.position+decision.consumed);
    auto next_proposal=draft_from(current);EXPECT_EQ(values<std::int32_t>(next_proposal[3])[0],values<std::int32_t>(pending)[0]);
    auto next=target_row(state.frontier(current).next_position,values<std::int32_t>(pending)[0],state.members(current,kTargetFirst,21));
    const std::array one{next};auto next_history=catchup(one,state.members(current,0,3),state.frontier(current).next_position);auto next_ids=verify_rows(one);
    auto continued=state.selected(current,std::span(next_history).first(4),next.captures,next.state,next_ids[0],1,3);
    check(head.publish(current.identity(),continued),"Pro actual pending continuation atomic publication");account();
    EXPECT_TRUE(head.publish(base.identity(),base).IsUnavailable());
    auto abandoned=take(state.store.begin(continued),"Pro cancelled speculative state");check(abandoned.replace(state.entries[0],histories[0]),"Pro stage cancelled actual lease");check(abandoned.cancel(),"Pro cancel whole unpublished tuple");
    EXPECT_EQ(take(head.current(),"Pro unchanged Head after cancellation").identity(),continued.identity());
    for(const std::uint32_t controlled:{0U,2U,5U}) {
      auto isolated=take(c::StateHead::create(state.store,base),"Pro controlled prefix policy Head");auto candidate=select(controlled,10+controlled);
      check(isolated.publish(base.identity(),candidate),"Pro explicit controlled prefix publication");
      const auto version=take(isolated.current(),"Pro controlled complete prefix");const auto facts=state.frontier(version);
      EXPECT_EQ(facts.next_position,setup.position+1+controlled);EXPECT_EQ(facts.pending,target_ids[controlled]);
      EXPECT_EQ(read(take(version.member(state.entries[kPending]),"Pro controlled verifier pending")),read(verified_rows[controlled]));account();
    }
    EXPECT_EQ(read(proposal_result[3]),retained_proposal);EXPECT_EQ(read(take(base.member(state.entries[0]),"Pro old published ring reader")),initial_ring);
    EXPECT_EQ(read(initial_pending),bytes<std::int32_t>(std::array<std::int32_t,1>{3}));
    ::testing::Test::RecordProperty("natural_accepted_drafts",int(decision.accepted_drafts));
    ::testing::Test::RecordProperty("controlled_prefixes","0,2,5; prefix policy only, not natural acceptance");
    ::testing::Test::RecordProperty("natural_proposal1",proposal_ids[1]);::testing::Test::RecordProperty("natural_target0",target_ids[0]);
  }
  base={};check(state.store.collect(),"Pro collect old/controlled complete versions");account();
  EXPECT_EQ(state.store.stats().versions,0U);EXPECT_EQ(state.store.stats().drafts,0U);
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  if(cuda) {check(draft_supply.cuda->close(),"Pro final draft upload drain after all readers");check(target_supply.cuda->close(),"Pro final target upload drain after all readers");}
#endif
  ::testing::Test::RecordProperty("actual_submissions",backend.submissions);
  ::testing::Test::RecordProperty("dense_original_roles",int(source->specs().size()));
  ::testing::Test::RecordProperty("dense_original_source_bytes",std::to_string(source->dense_source_bytes()));
  ::testing::Test::RecordProperty("actual_draft_episodes",int(draft_episodes));::testing::Test::RecordProperty("actual_target_rows",int(target_rows));
  ::testing::Test::RecordProperty("actual_main_projection_submissions",int(main_calls));::testing::Test::RecordProperty("actual_main_history_submissions",int(history_calls));
  ::testing::Test::RecordProperty("actual_verifier_submissions",int(head_calls));::testing::Test::RecordProperty("first_target_fullV_values",std::to_string(full_values));
  ::testing::Test::RecordProperty("first_draft_fullV_values",std::to_string(5*std::size_t(draft::kVocabulary)));
  ::testing::Test::RecordProperty("actual_pending_feedback",int(feedback));::testing::Test::RecordProperty("state_members",int(kMembers));
  ::testing::Test::RecordProperty("state_backing_budget",std::to_string(budget));::testing::Test::RecordProperty("state_backing_peak",std::to_string(peak));
  ::testing::Test::RecordProperty("target_resident_slot_bytes",std::to_string(target_supply.layout.slot_bytes));::testing::Test::RecordProperty("draft_resident_slot_bytes",std::to_string(draft_supply.layout.slot_bytes));
  ::testing::Test::RecordProperty("actual_expert_attachments",int(acquisition.attachments));::testing::Test::RecordProperty("actual_missing_frontiers",int(acquisition.requests));
  ::testing::Test::RecordProperty("scope",setup.scope);
}
struct DSparkProOriginalState: ::testing::Test {
  static inline std::shared_ptr<own::Data> source;
  static void SetUpTestSuite() {
    const auto directory=environment("HYLE_DSPARK_PRO_DIR");if(directory.empty())return;
    source=std::make_shared<own::Data>(own::Data::load(directory));
  }
  static void TearDownTestSuite(){source.reset();}
};
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
struct System final {m::MemoryResourceRef ref() noexcept{return m::system_resource();}};
TEST_F(DSparkProOriginalState,CpuOriginalTargetTailAndFormalDraftPublishesBlock5Prefix) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_PRO_DIR";
  auto context=take(g::cpu::create_context(),"Pro ordinary CPU Context");auto owner=take(c::StorageOwner::adopt(context,System{}),"Pro actual CPU owner");run(source,context,owner,false,kProFixture);
}
#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
struct SharedDevice final {std::shared_ptr<c::cuda::MemoryResource> resource;m::MemoryResourceRef ref() noexcept{return resource->ref();}};
TEST_F(DSparkProOriginalState,CudaOriginalTargetTailAndFormalDraftPublishesBlock5Prefix) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_PRO_DIR";
  auto context=take(g::cuda::create_context(),"Pro ordinary CUDA Context");auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Pro same actual CUDA resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Pro shared typed CUDA owner");run(source,context,owner,true,kProFixture,device.get());
}
#endif
struct DSparkFlashFormalState: ::testing::Test {
  static inline std::shared_ptr<own::Data> source;
  static void SetUpTestSuite() {
    const auto draft_directory=environment("HYLE_DSPARK_FORMAL_FLASH_DIR");if(draft_directory.empty())return;
    const auto target_directory=environment("HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR");
    const auto vocabulary_directory=environment("HYLE_DSPARK_SHARED_VOCAB_DIR");
    if(target_directory.empty()||vocabulary_directory.empty())
      throw std::runtime_error("formal Flash composition needs existing target and vocabulary caches");
    auto target_source=std::make_shared<dspark_original_target_roles::Data>(dspark_original_target_roles::Data::load(target_directory));
    auto vocabulary=std::make_shared<own::FlashVocabulary>(own::FlashVocabulary::load(vocabulary_directory));
    source=std::make_shared<own::Data>(own::Data::compose_flash(draft_directory,std::move(target_source),std::move(vocabulary)));
  }
  static void TearDownTestSuite(){source.reset();}
};
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
TEST_F(DSparkFlashFormalState,CpuOriginalTargetTailAndFormalDraftPublishesBlock5Prefix) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_FORMAL_FLASH_DIR";
  auto context=take(g::cpu::create_context(),"Flash ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"Flash actual CPU owner");run(source,context,owner,false,kFlashFixture);
}
#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
TEST_F(DSparkFlashFormalState,CudaOriginalTargetTailAndFormalDraftPublishesBlock5Prefix) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_FORMAL_FLASH_DIR";
  auto context=take(g::cuda::create_context(),"Flash ordinary CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Flash same actual CUDA resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Flash shared typed CUDA owner");run(source,context,owner,true,kFlashFixture,device.get());
}
#endif
} // namespace
