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
#include "dspark_stochastic_head.h"
#include "sampling_fixture.h"
#include "../../engine/inference/commit_observation.h"
#include "../../engine/inference/sampling_cursor.h"

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
  bool short_verifier_tails=false;
  bool stochastic_mtp=false;
  bool stochastic_branch_witness=true;
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
        std::printf("DSpark Pro actual expert frontier section=%s stage=%zu missing=%zu ids=",label,stage,missing.size());
        for(std::size_t i=0;i<missing.size();++i)std::printf("%s%d",i?",":"",int(missing[i]));
        std::printf("; %s\n",wait?"waiting for explicit external acquisition":"missing acquisition fails");std::fflush(stdout);
      }
      if(!wait)throw std::runtime_error(request.empty()?
          "original Pro selected expert bytes are not acquired; actual missing IDs printed (no frontier file configured)":
          "original Pro selected expert bytes are not acquired; actual frontier request recorded");
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
  if(rows<1||rows>6)throw std::runtime_error("Pro verifier needs one through six actual normalized target rows");
  Verifier result;result.fragment.region+=std::uint64_t(rows);
  result.fragment.profile=draft_program::checked_profile(*bound.source,profile);
  auto &f=result.fragment;const auto &model=bound.source->geometry();Value joined;
  for(std::int64_t i=0;i<rows;++i) {
    const auto input=f.input(sem::dense({1,model.hidden},c::DataType::kBFloat16));
    joined=i?draft::ds::concat(f,joined,input,0):input;
  }
  const auto head=draft_program::immutable(f,result.constants,bound.at("head.weight"),bound.at("head.weight").description);
  result.logits=f.one(c::model_ops::dense_linear_f32(),{joined,head});
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
  if(!take(run.graph.operation(projected),"Pro original fullV operation").custom_as<c::model_ops::DenseLinearF32>())
    throw std::runtime_error("Pro actual fullV must retain DenseLinearF32");
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
  std::optional<c::StateEntryId> random_entry;
  static State create(std::span<const c::TensorDesc> target_descriptions,std::size_t budget,
      const dspark_target_geometry::Geometry &model,bool stochastic=false) {
    if(target_descriptions.size()!=21)throw std::runtime_error("Pro target state must retain all21 actual cache resources");
    State result;std::vector<c::StateEntrySpec> specs;
    for(std::size_t i=0;i<3;++i)result.descriptions.push_back(take(c::row_cache_carrier_description(draft::kEncoding,128),"Pro main history schema"));
    result.descriptions.push_back(dense(c::DataType::kFloat32,{1,model.hidden}));
    for(std::size_t i=0;i<3;++i)result.descriptions.push_back(dense(c::DataType::kFloat32,{1,4,model.hidden}));
    for(const auto &description:result.descriptions)specs.push_back({.description=description});
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(policy::Boundary)});
    for(const auto &description:target_descriptions)specs.push_back({.description=description});
    specs.push_back({.description=dense(c::DataType::kInt32,{1})});
    if(stochastic)specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(hyle::inference::SamplingCursor)});
    result.schema=take(c::StateSchema::create(specs),"Pro complete model and optional RNG schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=16,.max_drafts=4,.max_retained_blocks=600,
        .max_backing_bytes=budget}),"Pro bounded StateStore");
    for(std::size_t i=0;i<kMembers;++i)result.entries[i]=take(result.schema.entry(i),"Pro typed state member");
    if(stochastic)result.random_entry=take(result.schema.entry(kMembers),"Pro transactional RNG member");
    return result;
  }
  policy::Boundary frontier(const c::StateVersion &version) const {
    const auto data=read(take(version.member(entries[7]),"Pro actual protocol metadata"));
    if(data.size()!=sizeof(policy::Boundary))throw std::runtime_error("Pro metadata extent");
    policy::Boundary result;std::memcpy(&result,data.data(),data.size());return result;
  }
  hyle::inference::SamplingCursor random(const c::StateVersion &version) const {
    if(!random_entry)throw std::runtime_error("stochastic state has no RNG metadata");
    const auto raw=read(take(version.member(*random_entry),"DSpark actual RNG metadata"));
    if(raw.size()!=sizeof(hyle::inference::SamplingCursor))throw std::runtime_error("DSpark RNG metadata extent");
    hyle::inference::SamplingCursor result;std::memcpy(&result,raw.data(),raw.size());return result;
  }
  std::vector<c::ReadLease> members(const c::StateVersion &version,std::size_t first,std::size_t count) const {
    std::vector<c::ReadLease> result;for(std::size_t i=0;i<count;++i)result.push_back(take(version.member(entries.at(first+i)),"Pro retained state lease"));return result;
  }
  c::StateVersion selected(const c::StateVersion &from,std::span<const c::ReadLease> history,
      std::span<const c::ReadLease> captures,std::span<const c::ReadLease> target_state,const c::ReadLease &pending,
      std::uint32_t consumed,std::uint64_t epoch,const hyle::inference::SamplingCursor *next_random=nullptr) {
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
    if(bool(random_entry)!=bool(next_random))throw std::runtime_error("DSpark random cursor must publish with its complete model state");
    if(next_random)check(staged.replace_metadata(*random_entry,std::as_bytes(std::span(next_random,1))),"DSpark accepted transactional RNG cursor");
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
  // Shortened verifier blocks retain at most five real one-row checkpoints
  // plus a following ordinary step. Their rings must not reuse live storage.
  auto history1=prepare_plain(backend,draft_program::build_main_history(bound,1,0x0ca22300,setup.profile),7);
  auto joined_captures=prepare_plain(backend,capture_join(model,setup.profile),1);
  std::array<std::optional<Plain<Verifier>>,6> verifiers;
  for(std::size_t rows=1;rows<=verifiers.size();++rows) {
    verifiers[rows-1].emplace(prepare_plain(backend,verifier(bound,std::int64_t(rows),setup.profile)));
    report_head(verifiers[rows-1]->run,rows);
  }
  namespace stochastic=dspark_stochastic_head;
  const stochastic::s::Policy sample_policy{.temperature=1,.top_k=0,.top_p=1};
  std::array<std::optional<Plain<stochastic::Proposal>>,6> proposals;
  std::array<std::optional<Plain<stochastic::Verification>>,6> decisions;
  std::optional<Plain<stochastic::Verification>> bonus_ordinary,bonus_decision;
  std::optional<Plain<stochastic::Proposal>> bonus_draft;
  if(setup.stochastic_mtp) {
    for(std::size_t count=0;count<=5;++count) {
      if(count)proposals[count].emplace(prepare_plain(backend,stochastic::proposal(bound,count,sample_policy,setup.profile),3));
      decisions[count].emplace(prepare_plain(backend,stochastic::verification(count,sample_policy,setup.profile),4));
    }
    if(setup.stochastic_branch_witness) {
      auto bonus_policy=sample_policy;bonus_policy.temperature=8;
      bonus_ordinary.emplace(prepare_plain(backend,stochastic::verification(0,bonus_policy,setup.profile),4));
      bonus_draft.emplace(prepare_plain(backend,stochastic::proposal(bound,1,bonus_policy,setup.profile),3));
      bonus_decision.emplace(prepare_plain(backend,stochastic::verification(1,bonus_policy,setup.profile),4));
    }
  }
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
  for(const auto *part:{&main6.run,&history6.run,&main1.run,&history1.run,&joined_captures.run})add(part->pool.stats().capacity_bytes);
  for(const auto &plan:verifiers)add(plan->run.pool.stats().capacity_bytes);
  for(const auto &plan:proposals)if(plan)add(plan->run.pool.stats().capacity_bytes);
  for(const auto &plan:decisions)if(plan)add(plan->run.pool.stats().capacity_bytes);
  if(bonus_ordinary)add(bonus_ordinary->run.pool.stats().capacity_bytes);
  if(bonus_decision)add(bonus_decision->run.pool.stats().capacity_bytes);
  if(bonus_draft)add(bonus_draft->run.pool.stats().capacity_bytes);
  const auto initial_budget=[&](std::size_t amount){if(amount>std::numeric_limits<std::size_t>::max()-255)throw std::runtime_error("Pro initial state alignment overflow");add((amount+255)&~std::size_t{255});};
  for(const auto &description:target_descriptions)initial_budget(take(description.required_bytes(),"Pro actual target initial backing"));
  for(std::size_t i=0;i<3;++i)initial_budget(128*std::size_t(draft::kEncoding.row_bytes()));
  initial_budget(std::size_t(model.hidden)*sizeof(float));for(std::size_t i=0;i<3;++i)initial_budget(std::size_t(4*model.hidden)*sizeof(float));
  initial_budget(sizeof(std::int32_t));add(22*((sizeof(policy::Boundary)+63)&~std::size_t{63}));
  if(setup.stochastic_mtp)add(22*((sizeof(hyle::inference::SamplingCursor)+63)&~std::size_t{63}));
  auto state=State::create(target_descriptions,budget,model,setup.stochastic_mtp);std::size_t peak=0;
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
  const hyle::inference::SamplingCursor initial_random{123,7,0};
  if(state.random_entry)members.push_back({*state.random_entry,{},std::as_bytes(std::span(&initial_random,1))});
  auto base=take(state.store.import(members),"Pro complete finite initial model/RNG state");members.clear();account();
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
    draft_reference::observe(*source,proposal.recipe.result,get,raw,draft_episodes==0&&!setup.short_verifier_tails&&!setup.stochastic_mtp);verified();++draft_episodes;++feedback;return output;
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
    for(std::size_t stage=0;stage<3;++stage)target_reference::observe_layer(*target_supply.source,stage,plan.recipe.layers[stage],plan.recipe.state_inputs[stage],current,get,raw,target_rows==0&&!setup.short_verifier_tails&&!setup.stochastic_mtp);
    target_reference::observe_head(*target_supply.source,plan.recipe.head,get);verified();
    Row row;for(std::size_t i=0;i<21;++i)row.state[i]=output[i];for(std::size_t i=0;i<3;++i)row.captures[i]=output[21+i];row.normalized=output[24];++target_rows;return row;
  };
  const auto catchup_one_or_six=[&](std::span<const Row> rows,std::span<const c::ReadLease> old,std::int64_t position) {
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
    draft_reference::observe_main(*source,projection.recipe.ports,floats(projection.recipe,projected,captures),main_calls==0&&!setup.short_verifier_tails&&!setup.stochastic_mtp);verified();++main_calls;
    std::vector<std::int32_t> positions(rows.size()),slots(rows.size());std::iota(positions.begin(),positions.end(),std::int32_t(position));
    for(std::size_t i=0;i<slots.size();++i)slots[i]=positions[i]%128;
    std::vector<c::ReadLease> input{projected[0],upload(owner,bytes<std::int32_t>(positions)),upload(owner,bytes<std::int32_t>(slots))};input.insert(input.end(),old.begin(),old.end());
    auto output=backend.execute(history.run,input);
    const auto raw=[&](Value value){return read(actual(history.recipe,output,input,value));};
    draft_reference::observe_history(*source,history.recipe.ports,floats(history.recipe,output,input),raw,history_calls==0&&!setup.short_verifier_tails&&!setup.stochastic_mtp);verified();++history_calls;return output;
  };
  const auto catchup=[&](std::span<const Row> rows,std::span<const c::ReadLease> old,std::int64_t position) {
    if(rows.empty()||rows.size()>6||old.size()!=3)throw std::runtime_error("Pro bounded main-history row contract");
    if(rows.size()==1||rows.size()==6)return catchup_one_or_six(rows,old,position);
    // The trained draft still has five rows. Only target/main-history work is
    // shortened, using already-prepared one-row math for this finite fixture.
    std::vector<c::ReadLease> output,prior(old.begin(),old.end());
    for(std::size_t i=0;i<rows.size();++i) {
      auto current=catchup_one_or_six(rows.subspan(i,1),prior,position+std::int64_t(i));
      prior.assign(current.begin(),current.begin()+3);
      output.insert(output.end(),current.begin(),current.begin()+4);
    }
    return output;
  };
  const auto verify_rows=[&](std::span<const Row> rows) {
    if(rows.empty()||rows.size()>6)throw std::runtime_error("Pro verifier needs one through six actual target rows");
    auto &plan=*verifiers.at(rows.size()-1);std::vector<c::ReadLease> input;std::vector<std::uint16_t> normalized;
    for(const auto &row:rows) {input.push_back(row.normalized);auto bits=values<std::uint16_t>(row.normalized);normalized.insert(normalized.end(),bits.begin(),bits.end());}
    auto output=backend.execute(plan.run,input);const auto logits=values<float>(output.at(rows.size())),scores=values<float>(output.at(rows.size()+1));
    std::vector<std::int32_t> ids;for(std::size_t i=0;i<rows.size();++i) {auto current=values<std::int32_t>(output[i]);if(current.size()!=1)throw std::runtime_error("Pro verifier row needs its own I32 lease");ids.push_back(current[0]);}
    EXPECT_TRUE(lm::expect_logits(logits,normalized,source->raw("head.weight"),std::size_t(draft::kVocabulary),std::size_t(model.hidden),head_calls==0&&!setup.stochastic_mtp,ids));
    EXPECT_TRUE(lm::expect_greedy(logits,ids,std::size_t(draft::kVocabulary),scores));verified();
    if(!head_calls)full_values=logits.size();++head_calls;return output;
  };
  if(!setup.short_verifier_tails && !setup.stochastic_mtp) {
    auto head=take(c::StateHead::create(state.store,base),"Pro bounded strategy Head");
    const auto round=[&](const c::StateVersion &from,policy::DraftControl control,std::uint64_t epoch) {
      const auto facts=state.frontier(from);
      policy::Block5Limits bounds{.output_quota=7,.context_rows=std::size_t(setup.capacity-facts.next_position),
          .vocabulary=draft::kVocabulary,.confidence_domain=policy::ConfidenceDomain::kRawLogit,.draft=control};
      const auto maximum=take(policy::block5_verification_capacity(bounds),"Pro bounded draft capacity");
      if(!maximum)throw std::runtime_error("Pro finite strategy has no target capacity");
      std::vector<std::int32_t> ids{facts.pending};std::vector<float> confidence;
      if(maximum>1) {
        auto proposed=draft_from(from);ids=values<std::int32_t>(proposed[3]);
        if(control.minimum_confidence && *control.minimum_confidence>0.F)
          confidence=values<float>(actual(proposal.recipe,proposed,{},proposal.recipe.result.confidence));
      }
      const auto count=take(policy::block5_verification_rows(bounds,confidence),"Pro raw confidence bounded prefix");
      std::vector<Row> rows;auto prior=state.members(from,kTargetFirst,21);
      for(std::size_t i=0;i<count;++i) {
        rows.push_back(target_row(facts.next_position+std::int64_t(i),ids.at(i),prior));
        prior.assign(rows.back().state.begin(),rows.back().state.end());
      }
      auto histories=catchup(rows,state.members(from,0,3),facts.next_position);
      auto verified_rows=verify_rows(rows);std::vector<std::int32_t> target_ids;
      for(std::size_t i=0;i<count;++i)target_ids.push_back(values<std::int32_t>(verified_rows[i])[0]);
      const auto decision=take(policy::decide_block5(ids,target_ids,bounds,confidence),"Pro configured exact verifier decision");
      const auto selected=decision.pending_row;
      auto candidate=state.selected(from,std::span(histories).subspan(selected*4,4),rows[selected].captures,
          rows[selected].state,verified_rows[selected],decision.consumed,epoch);
      check(head.publish(from.identity(),candidate),"Pro configured complete publication");account();
      EXPECT_EQ(read(take(candidate.member(state.entries[kPending]),"Pro bounded pending")),read(verified_rows[selected]));
      return candidate;
    };
    auto shortened=round(base,{.max_drafts=2,.minimum_confidence=.5F},2);
    const auto before_disabled=draft_episodes;
    auto ordinary=round(shortened,{.max_drafts=0,.minimum_confidence=.9F},3);
    EXPECT_EQ(draft_episodes,before_disabled); // no draft, confidence or temporary proposal cache
    EXPECT_EQ(state.frontier(ordinary).next_position,state.frontier(shortened).next_position+1);
    // Ordinary target work already caught up all three main-history rings.
    // Re-enable from that complete state, with its actual pending ID lease.
    auto resumed=draft_from(ordinary);
    EXPECT_EQ(values<std::int32_t>(resumed[3])[0],state.frontier(ordinary).pending);
    EXPECT_EQ(read(take(base.member(state.entries[0]),"Pro unchanged base ring")),initial_ring);
    ::testing::Test::RecordProperty("bounded_draft_controls","max2/raw-confidence>=0.5; disabled0 target-only; reenabled5 from complete main-history");
  }
  check(state.store.collect(),"Pro retire bounded strategy versions");

  if(setup.stochastic_mtp) {
    using Cursor=hyle::inference::SamplingCursor;
    // This finite consumer reserves its real result slots before execution.
    // Records retain only host facts, not a historical StateVersion/Identity.
    struct Commit final {
      bool committed=false,stopped=false;
      std::array<std::int32_t,6> emitted{};
      std::size_t emitted_count=0;
      policy::Boundary frontier{};
      Cursor sampling{};
    };
    struct Request final {
      std::array<Commit,2> records{};
      std::size_t commits=0,projections=0,family_observations=0;
      hyle::inference::RequestProgress progress;
      const std::int32_t *output_storage;
      std::size_t output_capacity;
      explicit Request(std::size_t maximum_outputs,Cursor initial)
          :progress(maximum_outputs),output_storage(progress.generated.data()),
           output_capacity(progress.generated.capacity()) {
        progress.initial_sampling=initial;(void)progress.begin_turn();
      }
    };
    const auto observe=[&](Request &request,std::size_t ordinal,const c::StateVersion &actual) {
      return hyle::inference::observe_record_commit(std::span<const Commit>(request.records),request.commits,
          ordinal,request.progress,
          [&](const Commit &record) {
            ++request.projections;
            return hyle::inference::CommitObservation{
                std::span(record.emitted).first(record.emitted_count),true,true,record.stopped};
          },
          [&](const Commit &record,const hyle::inference::CommitObservation &) {
            const auto current=state.frontier(actual);const auto &expected=record.frontier;
            if(current.session!=expected.session||current.branch!=expected.branch||current.epoch!=expected.epoch||
                current.next_position!=expected.next_position||current.pending!=expected.pending||
                current.consumed!=expected.consumed||state.random(actual)!=record.sampling||
                !record.emitted_count||record.emitted[record.emitted_count-1]!=current.pending)
              throw std::runtime_error("DSpark observation requires its actual committed pending/RNG result");
          },
          [&](const Commit &,const hyle::inference::CommitObservation &) noexcept {++request.family_observations;});
    };
    const auto identities=[&](const Cursor &cursor,std::size_t count,bool proposal) {
      std::vector<std::uint32_t> keys;
      for(std::size_t row=0;row<(proposal?count:count+1);++row) {
        const auto address=proposal?2*row:(row==count?2*count:2*row+1);
        const auto words=cursor.identity(address);keys.insert(keys.end(),words.begin(),words.end());
      }
      return upload(owner,bytes<std::uint32_t>(keys));
    };
    const auto same_lease=[&](const c::ReadLease &a,const c::ReadLease &b) {
      const auto av=take(a.view(),"random actual state backing"),bv=take(b.view(),"random selected producer backing");
      EXPECT_EQ(av.resource(),bv.resource());EXPECT_EQ(av.native_handle(),bv.native_handle());
      EXPECT_EQ(av.offset(),bv.offset());EXPECT_EQ(av.size(),bv.size());EXPECT_EQ(a.generation(),b.generation());
    };
    std::size_t publications=0,rejections=0,bonuses=0,target_only=0,markov_rows=0,cancelled=0,invalid=0;
    std::size_t bounded_tails=0,conditional_stops=0;
    std::size_t observer_refusals=0,observer_retries=0,duplicate_observations=0;
    std::vector<std::int32_t> emitted;emitted.reserve(6);
    const auto work_counts=[&] {
      return std::array<std::size_t,9>{std::size_t(backend.submissions),draft_episodes,target_rows,main_calls,
          history_calls,head_calls,feedback,markov_rows,publications};
    };
    std::optional<Cursor> bonus_cursor;std::int32_t bonus_proposal=-1;
    const auto round=[&](c::StateHead &head,Request &request,std::size_t maximum,std::uint64_t epoch,bool fail_and_retry,
        bool high_temperature=false,policy::Block5Limits bounds={.output_quota=6,.context_rows=6,.vocabulary=draft::kVocabulary},
        bool fail_observation=false) {
      const auto from=take(head.current(),"random original complete Head");const auto facts=state.frontier(from);
      if(request.commits!=request.progress.observed_commits())
        throw std::runtime_error("DSpark must observe its original committed result before executing another round");
      if(maximum>5)throw std::runtime_error("random DSpark maximum exceeds five trained draft rows");
      if(bounds.draft.max_drafts>5)throw std::runtime_error("random DSpark policy exceeds five trained draft rows");
      if(bounds.vocabulary!=draft::kVocabulary||facts.next_position<0||facts.next_position>setup.capacity)
        throw std::runtime_error("random DSpark limits differ from its actual source vocabulary/context");
      bounds.draft.max_drafts=static_cast<std::uint32_t>((std::min)(maximum,std::size_t(bounds.draft.max_drafts)));
      bounds.context_rows=(std::min)(bounds.context_rows,std::size_t(setup.capacity-facts.next_position));
      const auto capacity=take(policy::block5_verification_capacity(bounds),"random DSpark bounded row capacity");
      if(!capacity||std::ranges::find(bounds.eos_tokens,facts.pending)!=bounds.eos_tokens.end())
        throw std::runtime_error("random DSpark round already reached its output/context/EOS boundary");
      if(request.commits>=request.records.size())
        throw std::runtime_error("DSpark result exceeds its cold request record reservation");
      auto count=capacity-1;
      const auto cursor=state.random(from);
      std::vector<std::int32_t> ids{facts.pending};std::vector<c::ReadLease> sampled;
      auto *proposal_program=count?(high_temperature?&*bonus_draft:&*proposals[count]):nullptr;
      if(count) {
        auto raw=draft_from(from);auto &program=*proposal_program;
        const std::array inputs{actual(proposal.recipe,raw,{},proposal.recipe.result.head.base_logits),
            take(from.member(state.entries[kPending]),"random actual previous pending"),identities(cursor,count,true),
            actual(proposal.recipe,raw,{},proposal.recipe.result.hidden)};
        sampled=backend.execute(program.run,inputs);ids=values<std::int32_t>(sampled[program.recipe.ids]);
        if(ids.size()!=count+1||ids[0]!=facts.pending)throw std::runtime_error("random Markov proposal sequence mismatches pending");
        const auto &markov_role=bound.at(own::Section::kDraft,2,"markov_head.markov_w1.weight");
        const auto raw_markov=source->raw(markov_role.name);const auto rank=std::size_t(markov_role.description.dimensions().back());
        for(std::size_t row=0;row<count;++row) {
          if(values<std::uint8_t>(sampled[program.recipe.valid[row]])!=std::vector<std::uint8_t>{1})
            throw std::runtime_error("DSpark proposal q is invalid; no target row or state may use it");
          const auto expected=raw_markov.subspan(std::size_t(ids[row])*rank*2,rank*2);
          EXPECT_EQ(read(sampled[program.recipe.markov[row]]),Bytes(expected.begin(),expected.end()));++markov_rows;
        }
        std::vector<float> confidence;
        for(const auto port:program.recipe.confidence) {
          const auto value=values<float>(sampled[port]);
          if(value.size()!=1)throw std::runtime_error("random DSpark confidence is not a scalar");
          confidence.push_back(value[0]);
        }
        const auto clipped=take(policy::block5_sampled_verification_rows(bounds,confidence),"actual sampled-predecessor confidence prefix")-1;
        if(clipped<count)++conditional_stops;
        count=clipped;ids.resize(count+1);
        std::printf("DSpark random proposal pending=%d rows=%zu ids=",facts.pending,count);
        for(const auto id:ids)std::printf("%d,",id);std::printf(" draw=%llu\n",static_cast<unsigned long long>(cursor.draw));std::fflush(stdout);
      }
      const auto next_cursor=cursor.advanced(2*count+1);
      std::vector<Row> rows;auto prior=state.members(from,kTargetFirst,21);
      for(std::size_t row=0;row<=count;++row) {
        rows.push_back(target_row(facts.next_position+std::int64_t(row),ids.at(row),prior));
        prior.assign(rows.back().state.begin(),rows.back().state.end());
      }
      auto histories=catchup(rows,state.members(from,0,3),facts.next_position);

      const auto target_result=verify_rows(rows);
      auto &decision=high_temperature?(count?*bonus_decision:*bonus_ordinary):*decisions[count];
      if(setup.stochastic_branch_witness && count==2 && cursor.draw==0 && !cuda) {
        // Controlled branch coverage selects a seed from actual source q0/p0,
        // without modifying either support or a sampled result. The Graph
        // below must independently reproduce this proposal and acceptance.
        const auto q_logits=values<float>(sampled[proposal_program->recipe.adjusted[0]]);
        const auto target_logits=values<float>(target_result[count+1]);
        bonus_proposal=values<std::int32_t>(target_result[0])[0];
        const auto target_distribution_row=std::span(target_logits).first(std::size_t(draft::kVocabulary));
        const auto qmax=*std::max_element(q_logits.begin(),q_logits.end()),pmax=*std::max_element(target_distribution_row.begin(),target_distribution_row.end());
        double qmass=0,pmass=0,left=0,mass=0;
        for(std::size_t id=0;id<q_logits.size();++id) {
          const auto weight=std::exp((double(q_logits[id])-double(qmax))/8.);qmass+=weight;
          if(id<std::size_t(bonus_proposal))left+=weight;
          if(id==std::size_t(bonus_proposal))mass=weight;
          pmass+=std::exp((double(target_distribution_row[id])-double(pmax))/8.);
        }
        const auto probability=mass/qmass;
        const auto alpha=std::min(1.,std::exp((double(target_distribution_row[std::size_t(bonus_proposal)])-double(pmax))/8.)/pmass/probability);
        const auto uniform=[](const auto &identity) {
          const auto words=sampling_test::philox({0,0,identity[2],identity[3]},{identity[0],identity[1]});
          return (double(words[0])*0x1p20+double(words[1]>>12)+.5)*0x1p-52;
        };
        for(std::uint64_t seed=0;seed<4000000;++seed) {
          const Cursor trial{seed,7,0};const auto u=uniform(trial.identity());
          if(u>(left+.25*mass)/qmass && u<(left+.75*mass)/qmass && uniform(trial.identity(1))<alpha*.25) {bonus_cursor=trial;break;}
        }
        if(!bonus_cursor)throw std::runtime_error("bounded controlled CPU seed search found no original-role bonus witness");
        std::printf("DSpark controlled seed=%llu targetProposal=%d q8=%.9g alpha8=%.9g (branch witness,not natural acceptance rate)\n",
            static_cast<unsigned long long>(bonus_cursor->seed),bonus_proposal,probability,alpha);std::fflush(stdout);
      }

      std::vector<c::ReadLease> inputs{target_result[count+1],identities(cursor,count,false)};
      if(count) {
        inputs.push_back(upload(owner,bytes<std::int32_t>(std::span(ids).subspan(1))));
        for(std::size_t row=0;row<count;++row)inputs.push_back(sampled[proposal_program->recipe.q[row]]);
      }
      auto checked=backend.execute(decision.run,inputs);
      const auto validate=[&](const auto &output) {
        for(const auto port:decision.recipe.valid)
          if(values<std::uint8_t>(output[port])!=std::vector<std::uint8_t>{1})throw std::runtime_error("random speculative row invalid; state not published");
      };
      if(fail_and_retry && count) {
        const auto before=state.random(take(head.current(),"Head before invalid q"));auto bad=inputs;
        std::vector<float> nan(std::size_t(draft::kVocabulary),0.F);nan.back()=std::numeric_limits<float>::quiet_NaN();
        bad[3]=upload(owner,bytes<float>(nan));
        EXPECT_THROW(validate(backend.execute(decision.run,bad)),std::runtime_error);++invalid;
        EXPECT_EQ(take(head.current(),"Head after invalid q").identity(),from.identity());EXPECT_EQ(state.random(from),before);
        checked=backend.execute(decision.run,inputs); // exact cursor and q, no history resampling
      }
      validate(checked);std::size_t accepted=0;
      std::vector<std::uint8_t> accepts;
      for(const auto port:decision.recipe.accepted) {
        const auto value=values<std::uint8_t>(checked[port]);if(value.size()!=1||value[0]>1)throw std::runtime_error("random acceptance is not a scalar Bool");
        accepts.push_back(value[0]);
      }
      while(accepted<accepts.size()&&accepts[accepted])++accepted;
      std::vector<std::int32_t> final_ids;
      for(const auto port:decision.recipe.final_ids) {
        const auto id=values<std::int32_t>(checked[port]);if(id.size()!=1)throw std::runtime_error("random final token is not I32 scalar");
        final_ids.push_back(id[0]);
      }
      const auto select=[&](std::size_t quota,std::span<const std::int32_t> eos) {
        return hyle::inference::mtp::select_sampled_prefix(ids,accepts,final_ids,quota,eos,draft::kVocabulary,emitted);
      };
      const auto pending_for=[&](const hyle::inference::mtp::AcceptedPrefix &selection) {
        const auto row=selection.consumed-1;
        if(row<count&&accepts[row])
          return take(sampled[proposal_program->recipe.ids].subview((row+1)*sizeof(std::int32_t),sizeof(std::int32_t)),"accepted terminal proposal stays actual pending");
        return checked[decision.recipe.final_ids.at(count?(std::min)(row,count-1):0)];
      };
      const auto choice=select(bounds.output_quota,bounds.eos_tokens);
      const auto selected=choice.consumed-1;const auto consumed=static_cast<std::uint32_t>(choice.consumed);
      const auto pending=pending_for(choice);
      EXPECT_EQ(values<std::int32_t>(pending),std::vector<std::int32_t>{choice.next_id});
      auto candidate=state.selected(from,std::span(histories).subspan(selected*4,4),rows[selected].captures,
          rows[selected].state,pending,consumed,epoch,&next_cursor);
      if(fail_and_retry) {
        auto abandoned=take(state.store.begin(from),"random cancelled complete candidate");
        check(abandoned.replace(state.entries[kPending],pending),"random staged pending before cancellation");
        check(abandoned.replace_metadata(*state.random_entry,std::as_bytes(std::span(&next_cursor,1))),"random staged cursor before cancellation");
        check(abandoned.cancel(),"random pending/cursor cancellation");++cancelled;
        EXPECT_EQ(take(head.current(),"Head after random cancellation").identity(),from.identity());EXPECT_EQ(state.random(from),cursor);
      }
      if(request.commits>=request.records.size()||emitted.empty()||emitted.size()!=choice.consumed||
          emitted.size()>request.records.front().emitted.size()||emitted.size()>request.progress.remaining_outputs())
        throw std::runtime_error("DSpark actual emitted prefix exceeds its cold request reservation");
      const auto ordinal=request.commits;auto &record=request.records[ordinal];
      record.emitted_count=emitted.size();std::copy(emitted.begin(),emitted.end(),record.emitted.begin());
      record.stopped=choice.stopped;record.frontier=state.frontier(candidate);record.sampling=state.random(candidate);
      if(!request.progress.preflight_commit(ordinal,std::span(record.emitted).first(record.emitted_count),
          true,true,record.stopped))
        throw std::runtime_error("new DSpark result ordinal was already observed before Head publication");
      auto commit=take(hyle::inference::prepare_record_commit(std::span<Commit>(request.records),request.commits,record),
          "prepare original DSpark result commit witness");
      check(commit.publish([&]() noexcept {return head.publish(from.identity(),candidate);}),
          "random full model/pending/RNG atomic publication and original record witness");++publications;
      EXPECT_TRUE(record.committed);EXPECT_EQ(request.commits,ordinal+1);EXPECT_FALSE(commit.valid());
      EXPECT_EQ(state.random(candidate),next_cursor);EXPECT_EQ(state.frontier(candidate).next_position,facts.next_position+consumed);
      same_lease(take(candidate.member(state.entries[kPending]),"random committed pending"),pending);
      for(std::size_t i=0;i<21;++i)same_lease(take(candidate.member(state.entries[8+i]),"random committed target prefix"),rows[selected].state[i]);
      if(count&&accepted<count)++rejections;else if(count)++bonuses;else ++target_only;
      if(high_temperature&&count==1) {
        EXPECT_EQ(ids.at(1),bonus_proposal);EXPECT_EQ(accepted,1u);
        // Reuse the already retired, actual accepted source episode. Configured
        // EOS and a smaller publication quota must select after-current state
        // and the proposal's own lease, never consume it to obtain a bonus.
        const std::array eos{ids[1]},bonus_eos{final_ids[0]};
        for(const auto &[quota,terminal]:std::array<std::pair<std::size_t,std::span<const std::int32_t>>,3>{
            std::pair<std::size_t,std::span<const std::int32_t>>{2,eos},{1,{}},{2,bonus_eos}}) {
          const auto tail=select(quota,terminal);const auto tail_row=tail.consumed-1;const auto tail_pending=pending_for(tail);
          auto tail_state=state.selected(from,std::span(histories).subspan(tail_row*4,4),rows[tail_row].captures,
              rows[tail_row].state,tail_pending,static_cast<std::uint32_t>(tail.consumed),epoch+100+bounded_tails,&next_cursor);
          auto tail_head=take(c::StateHead::create(state.store,from),"independent original sampled tail Head");
          check(tail_head.publish(from.identity(),tail_state),"sampled EOS/quota complete prefix and RNG");
          EXPECT_EQ(state.frontier(tail_state).next_position,facts.next_position+tail.consumed);
          EXPECT_EQ(state.frontier(tail_state).pending,tail.next_id);EXPECT_EQ(state.random(tail_state),next_cursor);
          same_lease(take(tail_state.member(state.entries[kPending]),"sampled terminal original pending"),tail_pending);
          for(std::size_t i=0;i<21;++i)same_lease(take(tail_state.member(state.entries[8+i]),"sampled tail target state"),rows[tail_row].state[i]);
          if(quota==1||terminal.data()==eos.data()) {EXPECT_EQ(tail.consumed,1u);EXPECT_EQ(tail.next_id,ids[1]);}
          else {EXPECT_EQ(tail.consumed,2u);EXPECT_EQ(tail.next_id,final_ids[0]);}
          ++bounded_tails;
        }
      }
      EXPECT_TRUE(head.publish(from.identity(),candidate).IsUnavailable());account();
      std::printf("DSpark random committed m=%zu accept=%zu consumed=%u pending=%d draw=%llu\n",count,accepted,consumed,
          state.frontier(candidate).pending,static_cast<unsigned long long>(next_cursor.draw));std::fflush(stdout);
      if(fail_observation) {
        const auto before=work_counts();const auto stats=state.store.stats();
        const auto outputs=request.progress.generated;const auto observations=request.progress.observed_commits();
        // The head/record is authoritative. A stale family witness must fail
        // before observing outputs, without undoing or repeating native work.
        EXPECT_THROW((void)observe(request,ordinal,from),std::runtime_error);++observer_refusals;
        EXPECT_EQ(request.progress.observed_commits(),observations);EXPECT_EQ(request.progress.generated,outputs);
        EXPECT_EQ(request.family_observations,observations);EXPECT_EQ(work_counts(),before);
        EXPECT_EQ(take(head.current(),"DSpark late observer refusal keeps actual Head").identity(),candidate.identity());
        EXPECT_EQ(state.random(candidate),next_cursor);EXPECT_EQ(state.store.stats().versions,stats.versions);
        EXPECT_EQ(state.store.stats().drafts,stats.drafts);EXPECT_EQ(state.store.stats().backing_bytes,stats.backing_bytes);
      } else {
        EXPECT_TRUE(observe(request,ordinal,candidate));
        const auto projections=request.projections;const auto outputs=request.progress.generated;
        EXPECT_FALSE(observe(request,ordinal,from));++duplicate_observations;
        EXPECT_EQ(request.projections,projections);EXPECT_EQ(request.progress.generated,outputs);
        EXPECT_EQ(request.progress.observed_commits(),request.commits);EXPECT_EQ(request.family_observations,request.commits);
      }
      EXPECT_EQ(request.progress.generated.data(),request.output_storage);
      EXPECT_EQ(request.progress.generated.capacity(),request.output_capacity);
      return candidate;
    };
    auto head=take(c::StateHead::create(state.store,base),"random complete model/RNG Head");
    Request request(4,initial_random); // at most three actual first-round outputs, then one ordinary output
    auto first=round(head,request,2,2,setup.stochastic_branch_witness,false,
        {.output_quota=6,.context_rows=6,.vocabulary=draft::kVocabulary},true);
    {
      const auto before=work_counts();const auto stats=state.store.stats();
      EXPECT_EQ(request.commits,1u);EXPECT_EQ(request.progress.observed_commits(),0u);EXPECT_TRUE(request.progress.generated.empty());
      EXPECT_THROW((void)round(head,request,2,3,false),std::runtime_error);
      EXPECT_EQ(work_counts(),before);EXPECT_EQ(take(head.current(),"unobserved DSpark result is not reexecuted").identity(),first.identity());
      EXPECT_TRUE(observe(request,0,first));++observer_retries;
      const auto projections=request.projections;const auto outputs=request.progress.generated;
      EXPECT_FALSE(observe(request,0,base));++duplicate_observations;
      EXPECT_EQ(request.projections,projections);EXPECT_EQ(request.progress.generated,outputs);
      EXPECT_EQ(request.progress.generated.size(),request.records[0].emitted_count);
      EXPECT_TRUE(std::equal(request.progress.generated.begin(),request.progress.generated.end(),request.records[0].emitted.begin()));
      EXPECT_EQ(request.progress.generated.back(),state.frontier(first).pending);
      EXPECT_EQ(request.progress.observed_commits(),1u);EXPECT_EQ(request.family_observations,1u);
      EXPECT_EQ(work_counts(),before);EXPECT_EQ(take(head.current(),"DSpark observe-only retry never republishes Head").identity(),first.identity());
      EXPECT_EQ(state.random(first),request.records[0].sampling);EXPECT_EQ(state.store.stats().versions,stats.versions);
      EXPECT_EQ(state.store.stats().drafts,stats.drafts);EXPECT_EQ(state.store.stats().backing_bytes,stats.backing_bytes);
      EXPECT_EQ(request.progress.generated.data(),request.output_storage);EXPECT_EQ(request.progress.generated.capacity(),request.output_capacity);
    }
    auto continued=round(head,request,2,3,false,false,{.output_quota=1,.context_rows=1,.vocabulary=draft::kVocabulary,
        .draft={.max_drafts=0,.minimum_confidence=.9F}});
    EXPECT_EQ(state.frontier(continued).next_position,state.frontier(first).next_position+1);
    EXPECT_EQ(state.random(continued).draw,6u);EXPECT_EQ(state.random(base),initial_random);
    EXPECT_EQ(request.commits,2u);EXPECT_EQ(request.progress.observed_commits(),2u);EXPECT_EQ(request.family_observations,2u);
    EXPECT_EQ(request.projections,3u);EXPECT_EQ(observer_refusals,1u);EXPECT_EQ(observer_retries,1u);
    EXPECT_EQ(request.progress.generated.size(),request.records[0].emitted_count+request.records[1].emitted_count);
    EXPECT_EQ(request.records[1].emitted_count,1u);EXPECT_EQ(request.progress.generated.back(),state.frontier(continued).pending);
    EXPECT_EQ(read(take(base.member(state.entries[0]),"random old history stays immutable")),initial_ring);
    {
      const auto calls=backend.submissions;const auto before=take(head.current(),"random bounded rejection Head");
      EXPECT_THROW((void)round(head,request,2,30,false,false,{.output_quota=0,.context_rows=6,.vocabulary=draft::kVocabulary}),std::runtime_error);
      EXPECT_THROW((void)round(head,request,2,30,false,false,{.output_quota=6,.context_rows=0,.vocabulary=draft::kVocabulary}),std::runtime_error);
      EXPECT_THROW((void)round(head,request,6,30,false),std::runtime_error);
      EXPECT_EQ(backend.submissions,calls);EXPECT_EQ(take(head.current(),"bounded rejection keeps actual Head").identity(),before.identity());
      EXPECT_EQ(state.random(before),state.random(take(head.current(),"bounded rejection keeps RNG")));
    }
    if(setup.stochastic_branch_witness) {
    if(!bonus_cursor)throw std::runtime_error("controlled CPU bonus cursor unavailable");
    auto seeded=take(state.store.begin(base),"controlled original-state sampler address");
    check(seeded.replace_metadata(*state.random_entry,std::as_bytes(std::span(&*bonus_cursor,1))),"explicit fixture seed, unchanged model state");
    auto bonus_base=take(seeded.finalize(),"controlled stochastic initial state");
    auto bonus_head=take(c::StateHead::create(state.store,bonus_base),"controlled bonus complete Head");
    Request bonus_request(3,*bonus_cursor);
    auto bonus=round(bonus_head,bonus_request,1,4,false,true);
    EXPECT_EQ(state.frontier(bonus).next_position,setup.position+2);
    EXPECT_EQ(state.random(bonus),bonus_cursor->advanced(3));
    EXPECT_EQ(state.random(bonus_base),*bonus_cursor);
    const auto bonus_pending=state.frontier(bonus).pending;
    auto bonus_continued=round(bonus_head,bonus_request,0,5,false,true);
    EXPECT_EQ(state.frontier(bonus_continued).next_position,setup.position+3);
    EXPECT_EQ(state.random(bonus_continued),bonus_cursor->advanced(4));
    ::testing::Test::RecordProperty("actual_bonus_pending_consumed_next",bonus_pending);
    EXPECT_EQ(read(take(base.member(state.entries[0]),"old reader after all-accepted bonus")),initial_ring);

    auto threshold_head=take(c::StateHead::create(state.store,base),"actual sampled-confidence threshold Head");
    Request threshold_request(1,initial_random);
    auto threshold=round(threshold_head,threshold_request,2,40,false,false,{.output_quota=6,.context_rows=6,.vocabulary=draft::kVocabulary,
        .draft={.max_drafts=2,.minimum_confidence=1.F}});
    EXPECT_EQ(state.frontier(threshold).next_position,setup.position+1);
    EXPECT_EQ(state.random(threshold),initial_random.advanced(1));
    EXPECT_GT(conditional_stops,0u);

    }
    ::testing::Test::RecordProperty("random_publications",int(publications));::testing::Test::RecordProperty("random_rejection_prefixes",int(rejections));
    ::testing::Test::RecordProperty("random_all_accepted_bonus",int(bonuses));::testing::Test::RecordProperty("random_target_only_continuations",int(target_only));::testing::Test::RecordProperty("sampled_markov_rows",int(markov_rows));
    ::testing::Test::RecordProperty("random_invalid_q_retries",int(invalid));::testing::Test::RecordProperty("random_cancelled_candidates",int(cancelled));
    ::testing::Test::RecordProperty("random_actual_retained_tail_publications",int(bounded_tails));
    ::testing::Test::RecordProperty("random_actual_conditional_threshold_stops",int(conditional_stops));
    ::testing::Test::RecordProperty("random_request_committed_records",int(request.commits));
    ::testing::Test::RecordProperty("random_request_observed_records",int(request.progress.observed_commits()));
    ::testing::Test::RecordProperty("random_request_family_observations",int(request.family_observations));
    ::testing::Test::RecordProperty("random_request_record_projections",int(request.projections));
    ::testing::Test::RecordProperty("random_request_first_emitted",int(request.records[0].emitted_count));
    ::testing::Test::RecordProperty("random_request_total_emitted",int(request.progress.generated.size()));
    ::testing::Test::RecordProperty("random_request_late_observer_refusals",int(observer_refusals));
    ::testing::Test::RecordProperty("random_request_observe_only_retries",int(observer_retries));
    ::testing::Test::RecordProperty("random_request_duplicate_observations",int(duplicate_observations));
  } else

  {
    auto proposal_result=draft_from(base);const auto proposal_ids=values<std::int32_t>(proposal_result[3]);const auto retained_proposal=read(proposal_result[3]);
    if(proposal_ids.size()!=6)throw std::runtime_error("formal Pro draft must return current plus five actual IDs");
    std::vector<Row> rows;auto prior=state.members(base,kTargetFirst,21);
    const auto count=setup.short_verifier_tails?5u:6u;rows.reserve(count);
    for(std::size_t i=0;i<count;++i) {rows.push_back(target_row(setup.position+std::int64_t(i),proposal_ids[i],prior));prior.assign(rows.back().state.begin(),rows.back().state.end());}
    if(setup.short_verifier_tails) {
      std::size_t publications=0,continuations=0,policy_tails=0,bonus_eos_tails=0;
      const auto same_lease=[&](const c::ReadLease &actual_lease,const c::ReadLease &original_lease) {
        const auto a=take(actual_lease.view(),"short verifier published storage");
        const auto b=take(original_lease.view(),"short verifier actual producer storage");
        EXPECT_EQ(a.resource(),b.resource());EXPECT_EQ(a.native_handle(),b.native_handle());
        EXPECT_EQ(a.offset(),b.offset());EXPECT_EQ(a.size(),b.size());
        EXPECT_EQ(actual_lease.generation(),original_lease.generation());
      };
      for(std::size_t width=2;width<=5;++width) {
        const auto prefix=std::span(rows).first(width);
        auto histories=catchup(prefix,state.members(base,0,3),setup.position);
        auto verified_rows=verify_rows(prefix);std::vector<std::int32_t> target_ids;
        for(std::size_t i=0;i<width;++i)target_ids.push_back(values<std::int32_t>(verified_rows[i])[0]);
        std::string target_tape;for(const auto id:target_ids)target_tape+=(target_tape.empty()?"":",")+std::to_string(id);
        ::testing::Test::RecordProperty("short_verifier_targets_"+std::to_string(width),target_tape);
        const auto publish=[&](const policy::VerifiedPrefix &decision,std::uint64_t epoch) {
          const auto selected=decision.pending_row;const auto &row=prefix[selected];
          auto head=take(c::StateHead::create(state.store,base),"short verifier complete state Head");
          auto candidate=state.selected(base,std::span(histories).subspan(selected*4,4),row.captures,
              row.state,verified_rows[selected],decision.consumed,epoch);
          check(head.publish(base.identity(),candidate),"short verifier atomic30member publication");++publications;
          for(std::size_t i=0;i<4;++i)same_lease(take(candidate.member(state.entries[i]),"short verifier main history"),histories[selected*4+i]);
          for(std::size_t i=0;i<3;++i)same_lease(take(candidate.member(state.entries[4+i]),"short verifier target capture"),row.captures[i]);
          for(std::size_t i=0;i<21;++i)same_lease(take(candidate.member(state.entries[8+i]),"short verifier target cache"),row.state[i]);
          same_lease(take(candidate.member(state.entries[kPending]),"short verifier pending output"),verified_rows[selected]);
          EXPECT_EQ(state.frontier(candidate).next_position,setup.position+decision.consumed);account();
          return candidate;
        };
        policy::Block5Limits bounds{.output_quota=width,.context_rows=std::size_t(setup.capacity-setup.position),.vocabulary=draft::kVocabulary};
        const auto natural=take(policy::decide_block5(proposal_ids,target_ids,bounds),"actual shortened verifier natural decision");
        {
          auto candidate=publish(natural,100+width);const auto facts=state.frontier(candidate);
          // This source fixture rejects at its first actual target row. Reuse
          // that authenticated correction frontier, never invent a new token.
          EXPECT_EQ(natural.consumed,1u);verified();
          auto next=target_row(facts.next_position,facts.pending,state.members(candidate,kTargetFirst,21));
          const std::array one{next};auto next_history=catchup(one,state.members(candidate,0,3),facts.next_position);
          auto next_ids=verify_rows(one);
          auto continued=state.selected(candidate,std::span(next_history).first(4),next.captures,next.state,next_ids[0],1,200+width);
          auto head=take(c::StateHead::create(state.store,candidate),"short rejection continuation Head");
          check(head.publish(candidate.identity(),continued),"short rejection actual pending continuation");++continuations;
          EXPECT_EQ(state.frontier(continued).next_position,facts.next_position+1);
          same_lease(take(continued.member(state.entries[kPending]),"continued actual pending"),next_ids[0]);
          EXPECT_TRUE(head.publish(base.identity(),base).IsUnavailable());account();
        }
        // Controlled policy choices use these actual verifier IDs and leases,
        // but do not claim that the trained draft naturally proposed them.
        auto matched=proposal_ids;
        for(std::size_t i=0;i+1<width;++i)matched[i+1]=target_ids[i];
        for(std::size_t scenario=0;scenario<4;++scenario) {
          auto controlled=bounds;std::array<std::int32_t,1> eos{};
          if(scenario<2) {
            // Repeated actual IDs can make the designated bonus token an
            // earlier EOS. Record only a genuine final-row EOS for this case;
            // the exhaustive small policy/state test covers all positions.
            if(scenario==1&&std::find(target_ids.begin(),target_ids.end()-1,target_ids.back())!=target_ids.end()-1)continue;
            controlled.output_quota=7;eos[0]=target_ids[scenario==0?0:width-1];controlled.eos_tokens=eos;
            controlled.draft.max_drafts=static_cast<std::uint32_t>(width-1);
          } else if(scenario==3) {controlled.output_quota=7;controlled.context_rows=width;}
          const auto decision=take(policy::decide_block5(matched,target_ids,controlled),"actual lease EOS/quota/context policy tail");
          auto selected=publish(decision,300+width*4+scenario);
          if(scenario<2)EXPECT_TRUE(decision.stopped_on_eos);
          else if(scenario==2)EXPECT_TRUE(decision.output_quota_exhausted);
          else EXPECT_TRUE(decision.context_exhausted);
          EXPECT_FALSE(decision.can_continue());++policy_tails;
          if(scenario==1){EXPECT_EQ(decision.pending_row,width-1);++bonus_eos_tails;}
          EXPECT_EQ(state.frontier(selected).pending,target_ids[decision.pending_row]);
        }
        EXPECT_EQ(read(take(base.member(state.entries[0]),"short verifier unchanged old ring")),initial_ring);
        check(state.store.collect(),"retire shortened verifier state before the next width");
      }
      EXPECT_EQ(read(proposal_result[3]),retained_proposal);
      ::testing::Test::RecordProperty("actual_short_verifier_widths","2,3,4,5");
      ::testing::Test::RecordProperty("actual_short_state_publications",int(publications));
      ::testing::Test::RecordProperty("actual_short_rejection_continuations",int(continuations));
      ::testing::Test::RecordProperty("actual_lease_policy_tails",int(policy_tails));
      ::testing::Test::RecordProperty("actual_lease_unique_bonus_eos_tails",int(bonus_eos_tails));
      ::testing::Test::RecordProperty("tail_policy_scope","accepted EOS/bonus EOS/quota/context with actual head IDs and leases; explicitly controlled decisions, not natural EOS or acceptance-rate evidence");
    } else {
    auto rings=state.members(base,0,3);auto histories=catchup(rows,rings,setup.position);auto verified_rows=verify_rows(rows);
    std::vector<std::int32_t> target_ids;for(std::size_t i=0;i<6;++i)target_ids.push_back(values<std::int32_t>(verified_rows[i])[0]);
    const auto decision=take(policy::decide_block5(proposal_ids,target_ids,
        {.output_quota=7,.context_rows=std::size_t(setup.capacity-setup.position),.vocabulary=draft::kVocabulary}),
        "Pro actual fullV K5 decision");
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
  ::testing::Test::RecordProperty("actual_pending_feedback",int(feedback));::testing::Test::RecordProperty("state_members",int(kMembers+bool(state.random_entry)));
  ::testing::Test::RecordProperty("state_backing_budget",std::to_string(budget));::testing::Test::RecordProperty("state_backing_peak",std::to_string(peak));
  ::testing::Test::RecordProperty("target_resident_slot_bytes",std::to_string(target_supply.layout.slot_bytes));::testing::Test::RecordProperty("draft_resident_slot_bytes",std::to_string(draft_supply.layout.slot_bytes));
  ::testing::Test::RecordProperty("actual_expert_attachments",int(acquisition.attachments));::testing::Test::RecordProperty("actual_missing_frontiers",int(acquisition.requests));
  ::testing::Test::RecordProperty("scope",setup.stochastic_mtp?
      "Original identified Pro/Flash three-stage/base fullV plus source Markov conditioned on actual sampled predecessor; common stochastic q/p acceptance, residual/bonus and complete31member model/RNG Head. Natural prefix and ordinary continuation use common original commit/observation with cold multi-token output storage; an actual post-publication observer refusal retries only the original result, without Graph, Head or RNG repetition. Invalid-q retry/cancel and immutable old readers retained. When enabled, controlled CPU seed/temperature8 bonus witness is explicit; natural-only mode performs no seed search. No RequestLoop, complete trained-model quality or TPS claim.":setup.short_verifier_tails?
      "Identified original three-stage draft and target-tail, finite synthetic input/cache frontier. One trained five-row draft, actual target verifier widths2/3/4/5, complete30member state, actual first-rejection continuation. EOS/bonusEOS/quota/context use actual verifier IDs/leases with explicitly controlled policies; not natural EOS, a full-model generation, acceptance-rate or TPS claim.":setup.scope);
}
TEST(DSparkStochasticHeadSchema,RejectsEmptySourceAndOutOfTrainingCountBeforeLookup) {
  const own::Bound empty;
  const auto *profile=&cpu_c4_model_branches::flash_profile;
  EXPECT_THROW((void)dspark_stochastic_head::proposal(empty,0,{},profile),std::runtime_error);
  EXPECT_THROW((void)dspark_stochastic_head::proposal(empty,6,{},profile),std::runtime_error);
  EXPECT_THROW((void)dspark_stochastic_head::proposal(empty,1,{},profile),std::runtime_error);
  EXPECT_THROW((void)dspark_stochastic_head::verification(6,{},profile),std::runtime_error);
}
TEST(DSparkStochasticHeadSchema,TargetOnlyAndFiveDraftsKeepTheActualConditionalRows) {
  const auto *profile=&cpu_c4_model_branches::flash_profile;
  for(const std::size_t count:{0u,1u,2u,5u}) {
    const auto recipe=dspark_stochastic_head::verification(count,{},profile);
    ASSERT_EQ(recipe.fragment.inputs.size(),count?count+3:2);
    EXPECT_EQ(recipe.fragment.description(recipe.fragment.inputs[0]),sem::dense({std::int64_t(count+1),draft::kVocabulary}));
    EXPECT_EQ(recipe.accepted.size(),count);EXPECT_EQ(recipe.final_ids.size(),std::max<std::size_t>(1,count));
    EXPECT_EQ(recipe.valid.size(),std::max<std::size_t>(1,count));
    for(const auto port:recipe.final_ids)EXPECT_EQ(recipe.fragment.description(recipe.fragment.outputs[port]),sem::dense({1},c::DataType::kInt32));
  }
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
TEST_F(DSparkProOriginalState,CpuNaturalRandomMarkovPrefixAndPendingContinue) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_PRO_DIR";
  auto context=take(g::cpu::create_context(),"Pro stochastic CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"Pro stochastic actual CPU owner");
  auto setup=kProFixture;setup.stochastic_mtp=true;setup.stochastic_branch_witness=false;
  run(source,context,owner,false,setup);
}

#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
struct SharedDevice final {std::shared_ptr<c::cuda::MemoryResource> resource;m::MemoryResourceRef ref() noexcept{return resource->ref();}};
TEST_F(DSparkProOriginalState,CudaOriginalTargetTailAndFormalDraftPublishesBlock5Prefix) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_PRO_DIR";
  auto context=take(g::cuda::create_context(),"Pro ordinary CUDA Context");auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Pro same actual CUDA resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Pro shared typed CUDA owner");run(source,context,owner,true,kProFixture,device.get());
}
TEST_F(DSparkProOriginalState,CudaShortVerifierTailsPreserveActualStateAndContinue) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_PRO_DIR";
  auto context=take(g::cuda::create_context(),"Pro short-verifier CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Pro actual short-verifier resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Pro short-verifier owner");
  auto setup=kProFixture;setup.short_verifier_tails=true;run(source,context,owner,true,setup,device.get());
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

TEST_F(DSparkFlashFormalState,CpuRandomMarkovPrefixCommitsActualStateAndCursor) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_FORMAL_FLASH_DIR";
  auto context=take(g::cpu::create_context(),"Flash stochastic CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"Flash stochastic actual CPU owner");
  auto setup=kFlashFixture;setup.stochastic_mtp=true;run(source,context,owner,false,setup);
}

#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
TEST_F(DSparkFlashFormalState,CudaOriginalTargetTailAndFormalDraftPublishesBlock5Prefix) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_FORMAL_FLASH_DIR";
  auto context=take(g::cuda::create_context(),"Flash ordinary CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Flash same actual CUDA resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Flash shared typed CUDA owner");run(source,context,owner,true,kFlashFixture,device.get());
}
TEST_F(DSparkFlashFormalState,CudaShortVerifierTailsPreserveActualStateAndContinue) {
  if(!source)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_FORMAL_FLASH_DIR";
  auto context=take(g::cuda::create_context(),"Flash short-verifier CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Flash actual short-verifier resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Flash short-verifier owner");
  auto setup=kFlashFixture;setup.short_verifier_tails=true;run(source,context,owner,true,setup,device.get());
}
#endif
} // namespace
