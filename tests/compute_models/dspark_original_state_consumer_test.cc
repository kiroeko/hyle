#include <gtest/gtest.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "dspark_original_state_consumer.h"
#include "dspark_stochastic_head.h"
#include "dspark_bfloat_sidecar_parameters.h"
#if __has_include("dspark_bfloat_sidecar_artifacts.h")
#include "dspark_bfloat_sidecar_artifacts.h"
#endif
#include "dspark_native_consumer.h"
#include "dspark_original_roles.h"
#include "dspark_original_lm_reference.h"
#include "dspark_gguf_vocabulary_roles.h"
#include "dspark_gguf_vocabulary_reference.h"
#include "glm_shared_vocabulary_reference.h"
#include "dspark_original_hc_reference.h"
#include "dspark_prepare_constants.h"
#include "dspark_resident_draft.h"
#include "dspark_original_resident_weights.h"
#include "dspark_original_expert_reference.h"
#include "dspark_original_target_fragment.h"
#include "dspark_original_target_roles.h"
#include "dspark_original_target_weights.h"
#include "dspark_original_target_program.h"
#include "dspark_original_target_reference.h"
#include "dspark_target_tail_checks.h"
#include "../../engine/inference/commit_observation.h"
#include "hyle/graph/weight_binding.h"
#include "cpu_c4_flash_sidecars.h"
#include "row_cache_fixture.h"
#include "../graph_route_cuda/dspark_head_checks.h"
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif

namespace {
namespace x=dspark_original_state_consumer;
namespace h=dspark_main_history;
namespace b=dspark_target_binding;
namespace d=dspark_complete_graph;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace original=dspark_original_roles;
namespace hc=x::parameters;
namespace hc_reference=dspark_original_hc_reference;
namespace shared_ffn=x::shared_ffn;
namespace shared_reference=dspark_original_shared_reference;
namespace packed_vocabulary=dspark_gguf_vocabulary_roles;
namespace packed_vocabulary_reference=dspark_gguf_vocabulary_reference;
namespace attention=x::attention;
namespace attention_reference=dspark_original_attention_reference;
namespace expert=x::expert;
namespace expert_reference=dspark_original_expert_reference;
namespace resident=dspark_resident_draft;
namespace resident_weights=dspark_original_resident_weights;
namespace bfloat_sidecar=dspark_bfloat_sidecar_roles;
namespace bfloat_parameters=dspark_bfloat_sidecar_parameters;
namespace bfloat_reference=dspark_bfloat_sidecar_reference;
namespace target_math=dspark_original_target_fragment;
namespace target_roles=dspark_original_target_roles;
namespace target_weights=dspark_original_target_weights;
namespace target_program=dspark_original_target_program;
namespace target_reference=dspark_original_target_reference;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using Bytes=std::vector<std::byte>;
using dspark_native_consumer::take;
using dspark_native_consumer::check;
using dspark_native_consumer::bytes;
using dspark_native_consumer::read;
using dspark_native_consumer::values;
using dspark_native_consumer::upload;
using dspark_native_consumer::dense;
using dspark_native_consumer::Run;
using dspark_native_consumer::Backend;
using dspark_native_consumer::RoutedRun;
using dspark_native_consumer::execute_draft;

// A preparation wrapper around the same existing native Backend; no executor
// or alternate model lifetime is introduced for the source-specific head.
template<class Recipe> struct StochasticRun final {Recipe recipe;Run run;};
template<class Recipe> StochasticRun<Recipe> prepare_stochastic(Backend &backend,Recipe recipe,std::uint32_t generations=4) {
  const auto built=sem::make_graph(recipe.fragment);std::vector<g::ConstantBinding> constants;
  for(std::size_t i=0;i<recipe.fragment.constants.size();++i) {
    const auto value=recipe.fragment.constants[i];
    const auto found=std::find_if(recipe.constants.begin(),recipe.constants.end(),[&](const auto &entry){return entry.value.ordinal()==value.ordinal();});
    if(found!=recipe.constants.end())constants.push_back(take(found->role.binding(built.constants[i]),"original sampled Markov readonly bank"));
    else {
      const auto lease=upload(backend.owner,recipe.fragment.constant_bytes.at(i));
      constants.push_back({.value=built.constants[i],.tensor=take(lease.tensor(recipe.fragment.description(value)),"sampled-head literal"),.owner=backend.owner,.lease=lease});
    }
  }
  auto run=backend.prepare(built.graph,constants,generations);
  return {std::move(recipe),std::move(run)};
}
std::string environment(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t count=0;
  if(_dupenv_s(&value,&count,name)) throw std::runtime_error("environment lookup");
  const std::unique_ptr<char,decltype(&std::free)> cleanup(value,&std::free);return value?value:"";
#else
  auto value=std::getenv(name);return value?value:"";
#endif
}
b::TargetIdentity target() {
  return {b::kModelRepository,b::kModelRevision,b::kSidecarRepository,b::kSidecarRevision,"UD-Q4_K_XL"};
}
b::SidecarIdentity sidecar() {return {b::kSidecarRepository,b::kSidecarRevision,b::kSidecarFile};}
struct Data final {
  Bytes fc;
  std::vector<float> main_norm;
  std::array<Bytes,3> wkv;
  std::array<std::vector<float>,3> norm;
  original::Roles heads;
  std::shared_ptr<bfloat_sidecar::Data> bfloat_source;
  static Data from_bfloat(std::shared_ptr<bfloat_sidecar::Data> source) {
    if(!source)throw std::runtime_error("original S03 source absent");
    Data result;result.bfloat_source=std::move(source);
    const auto assign=[](std::vector<float> &to,std::span<const float> from){to.assign(from.begin(),from.end());};
    assign(result.main_norm,result.bfloat_source->floats("enc.output_norm.weight"));
    for(std::size_t stage=0;stage<3;++stage)
      assign(result.norm[stage],result.bfloat_source->floats(bfloat_sidecar::stage_name(stage,"attn_kv_a_norm.weight")));
    // Preserve the existing head oracle's BF16 coefficients. This is a
    // bounded host BF16 copy, never a hot/device full-F32 Markov shadow.
    result.heads.w1=bfloat_sidecar::source::decode_bfloat(result.bfloat_source->raw("markov_w1.weight"));
    result.heads.w2=bfloat_sidecar::source::decode_bfloat(result.bfloat_source->raw("markov_w2.weight"));
    result.heads.confidence=bfloat_sidecar::source::decode_bfloat(result.bfloat_source->raw("conf_proj.weight"));
    return result;
  }
  static Data load(const std::filesystem::path &base,const std::filesystem::path &extra,
                   const std::filesystem::path &markov,const std::filesystem::path &confidence) {
    Data result;
    result.fc=original_role_input::read_bytes(base/"fc.weight.q8_0",53477376,
        "4516df35cd05ecd36b93e73ac2ae74c524ec6edda13b634a7895b7eabe0c8c04");
    result.main_norm=original_role_input::read_float(base/"enc.output_norm.weight.f32",b::kHidden,
        "0f99d64b4fa46264c9bd2efdffcb66d3ec0cdd0ca61109f929ed958bcb843f0b");
    constexpr std::array weight_sha{
      "a52e49dc69f4eff1c8287b3887123dd7a583bdaee7d42efe55310f035bafd35f",
      "0a707c0a1eb00673838914bb12b070fe45d70c851efdfe8931e6b914979eb09d",
      "8624943fb84a7d82a235d65d262e79ccf93823b6af67afbcc9e2ff83c1856138"};
    constexpr std::array norm_sha{
      "edbbd17729cfb3c0852c8b6cdb8c1044eb9f501d8864c7ef4444c568e56fc966",
      "733bdaab61d5b429b3a0caa6f74d082d34d7e993f1cb27d508e403eff4baf569",
      "1e49befb7879c1a6fd05311bc7c23f5436b1818238b33bcf272f4f54cc53b556"};
    for(std::size_t stage=0;stage<3;++stage) {
      const auto root=stage?extra:base;const auto prefix="blk."+std::to_string(stage);
      result.wkv[stage]=original_role_input::read_bytes(root/(prefix+".attn_kv.weight.q8_0"),2228224,weight_sha[stage]);
      result.norm[stage]=original_role_input::read_float(root/(prefix+".attn_kv_a_norm.weight.f32"),b::kKv,norm_sha[stage]);
    }
    result.heads=original::load(markov,confidence);return result;
  }
};
struct SharedVocabularyData final {
  std::vector<std::uint16_t> embedding,head;
  std::shared_ptr<const packed_vocabulary::Data> packed;
  static SharedVocabularyData load(const std::filesystem::path &directory) {
    constexpr auto elements=original::vocabulary*original::hidden;
    return {original_role_input::read_bfloat(directory/"embed.weight.bf16",elements,
        "c05019ecf9435fa47cdcbb35df74c8ade6c4ceedd6ebe75db9ea7a607f6ad2ef"),
      original_role_input::read_bfloat(directory/"head.weight.bf16",elements,
        "029e3c5293b29cc426e21d87795e15efa4d363f27b2bc4a9e3aef7d79f047919"),{}};
  }
  static SharedVocabularyData load_packed(const std::filesystem::path &directory) {
    return {{},{},std::make_shared<const packed_vocabulary::Data>(packed_vocabulary::Data::load(directory))};
  }
};
std::string bfloat_catalogue_pin() {
#if __has_include("dspark_bfloat_sidecar_artifacts.h")
  namespace pin=dspark_bfloat_sidecar_artifacts;
  if(pin::kRepository!=bfloat_sidecar::kRepository||pin::kRevision!=bfloat_sidecar::kRevision||
      pin::kFile!=bfloat_sidecar::kFile||
      pin::kOriginalHeaderSha256!=bfloat_sidecar::kOriginalHeaderSha256||
      pin::kStaticPlanSha256!=bfloat_sidecar::kStaticPlanSha256||
      pin::kExpertSourcePlanSha256!=bfloat_sidecar::kExpertSourcePlanSha256)
    throw std::runtime_error("S03 generated actual catalogue belongs to another original artifact");
  return std::string(pin::kStaticCatalogueSha256);
#else
  throw std::runtime_error("Generate dspark_bfloat_sidecar_artifacts.h only after COMPLETE real S03 static acquisition");
#endif
}
std::string bfloat_expert_plan_pin() {
#if __has_include("dspark_bfloat_sidecar_artifacts.h")
  return std::string(dspark_bfloat_sidecar_artifacts::kExpertSourcePlanSha256);
#else
  throw std::runtime_error("S03 original expert source plan has no generated actual metadata");
#endif
}
std::string metadata_text(const std::filesystem::path &path) {
  auto mapped=take(m::FileMapping::open(path.string().c_str()),"map S03 expert ready metadata");
  if(mapped.size()>1024*1024)throw std::runtime_error("S03 expert metadata exceeds bounded extent");
  const auto raw=take(mapped.bytes(),"S03 ready bytes");
  return {reinterpret_cast<const char*>(raw.data()),raw.size()};
}
std::size_t metadata_value(const std::string &text,std::string_view field,std::size_t first=0) {
  const auto key="\""+std::string(field)+"\"";auto found=text.find(key,first);
  if(found==std::string::npos||(found=text.find(':',found+key.size()))==std::string::npos)
    throw std::runtime_error("S03 expert ready missing field "+std::string(field));
  found=text.find_first_not_of(" \r\n\t",found+1);
  if(found==std::string::npos)throw std::runtime_error("S03 expert ready missing value");return found;
}
std::string metadata_string(const std::string &text,std::string_view field,std::size_t first=0) {
  const auto begin=metadata_value(text,field,first);
  if(text[begin]!='"')throw std::runtime_error("S03 metadata string expected");
  const auto end=text.find('"',begin+1);
  if(end==std::string::npos)throw std::runtime_error("S03 metadata string is unterminated");
  return text.substr(begin+1,end-begin-1);
}
std::optional<std::string> bfloat_expert_pin(const std::filesystem::path &path,std::size_t stage,std::int32_t global) {
  if(!std::filesystem::exists(path/"ready.json"))return std::nullopt;
  const auto text=metadata_text(path/"ready.json");const auto complete=metadata_value(text,"complete");
  if(text.compare(complete,5,"false")==0)return std::nullopt;
  if(text.compare(complete,4,"true")!=0||metadata_string(text,"section")!="draft"||
      metadata_string(text,"repository")!=bfloat_sidecar::kRepository||metadata_string(text,"revision")!=bfloat_sidecar::kRevision||
      metadata_string(text,"file")!=bfloat_sidecar::kFile)
    throw std::runtime_error("S03 ready artifact/section/completion differs");
  const auto number=[&](std::size_t begin) {
    std::int64_t value=0;const auto parsed=std::from_chars(text.data()+begin,text.data()+text.size(),value);
    if(parsed.ec!=std::errc{})throw std::runtime_error("S03 ready integer differs");
    return std::pair{value,std::size_t(parsed.ptr-text.data())};
  };
  if(number(metadata_value(text,"stage")).first!=std::int64_t(stage))throw std::runtime_error("S03 ready stage differs");
  const auto list=metadata_value(text,"global_ids");
  if(text[list]!='[')throw std::runtime_error("S03 ready global-ID array differs");
  const auto first=text.find_first_not_of(" \r\n\t",list+1);
  if(first==std::string::npos)throw std::runtime_error("S03 ready empty original ID array");
  const auto [id,end]=number(first);
  const auto last=text.find_first_not_of(" \r\n\t",end);
  if(id!=global||last==std::string::npos||text[last]!=']')throw std::runtime_error("S03 ready single original global ID differs");
  const auto source=text.find("\"source_plan\"");
  if(source==std::string::npos||metadata_string(text,"sha256",source)!=bfloat_expert_plan_pin())
    throw std::runtime_error("S03 ready original expert source-plan pin differs");
  const auto pin=metadata_string(text,"catalog_sha256");const auto typed=text.find("\"typed_catalog\"");
  if(typed==std::string::npos||metadata_string(text,"sha256",typed)!=pin||pin.size()!=64||
      std::any_of(pin.begin(),pin.end(),[](char value){return !(value>='0'&&value<='9')&&!(value>='a'&&value<='f');}))
    throw std::runtime_error("S03 actual typed catalogue pin differs");
  return pin;
}
struct BfloatFrontier final {
  std::shared_ptr<bfloat_sidecar::Data> source;
  std::filesystem::path directory,request;
  bool wait=false;
  std::size_t requests=0,attachments=0;
  void ensure(std::size_t stage,std::span<const std::int32_t> globals) {
    const auto started=std::chrono::steady_clock::now();bool announced=false;
    for(;;) {
      std::vector<std::int32_t> missing;
      for(const auto global:globals) {
        if(source->has_expert(stage,global))continue;
        std::ostringstream leaf;leaf<<"expert"<<std::setfill('0')<<std::setw(3)<<global;
        const auto path=directory/"draft"/("stage"+std::to_string(stage))/leaf.str();
        const auto pin=bfloat_expert_pin(path,stage,global);
        if(!pin){missing.push_back(global);continue;}
        source->attach_expert(std::make_shared<const bfloat_sidecar::ExpertSet>(
            bfloat_sidecar::ExpertSet::load(path,*pin,stage,global)));++attachments;
      }
      if(missing.empty())return;
      if(!announced) {
        if(!request.empty()) {
          if(!request.parent_path().empty())std::filesystem::create_directories(request.parent_path());
          auto temporary=request;temporary+=".tmp";std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
          if(!out)throw std::runtime_error("cannot open S03 actual frontier request");
          const auto list=[&](std::span<const std::int32_t> ids){out<<'[';for(std::size_t i=0;i<ids.size();++i){if(i)out<<',';out<<ids[i];}out<<']';};
          out<<"{\"section\":\"draft\",\"stage\":"<<stage<<",\"global_ids\":";list(globals);
          out<<",\"missing_ids\":";list(missing);out<<"}\n";out.close();
          if(!out)throw std::runtime_error("cannot finish S03 actual frontier request");
#if defined(_WIN32)
          if(!MoveFileExW(temporary.c_str(),request.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot atomically replace S03 expert request");
#else
          std::filesystem::rename(temporary,request);
#endif
        }
        ++requests;announced=true;std::printf("S03 actual draft stage=%zu missing=%zu, explicit external wait=%d\n",stage,missing.size(),int(wait));std::fflush(stdout);
      }
      if(!wait)throw std::runtime_error("S03 actual selected original experts missing; frontier recorded");
      if(std::chrono::steady_clock::now()-started>std::chrono::minutes(20))throw std::runtime_error("S03 bounded expert frontier wait expired");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
};
struct BfloatHcView final {
  const bfloat_sidecar::Data &source;
  std::span<const float> values(std::size_t index)const{return source.floats(hc::spec(index).name);}
  std::span<const float> values(std::size_t stage,hc::StageRole role)const{return values(hc::index(stage,role));}
  std::span<const float> values(hc::HeadRole role)const{return values(hc::index(role));}
};

b::ReadonlyRole role(std::string name,const c::TensorDesc &desc,const c::StorageOwner &owner,
                      std::span<const std::byte> source) {return {std::move(name),desc,owner,upload(owner,source)};}

template<class Source> using DraftRunFor=RoutedRun<Source,resident_weights::CpuSupply,
    d5_deepseek_official_fragments::RoutedBoundary,expert::Selection
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    ,resident_weights::CudaSupply
#endif
    >;
using DraftRun=DraftRunFor<expert::Data>;
using TargetRun=RoutedRun<target_roles::Data,target_weights::CpuSupply,target_math::Boundary,expert::Selection
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    ,target_weights::CudaSupply
#endif
    >;
template<class Source>
DraftRunFor<Source> prepare_draft(Backend &backend,x::Draft &logical,std::shared_ptr<const Source> source) {
  auto &program=logical.program;auto &f=program.fragment;
  std::vector<m::Buffer> prepared;
  if(f.preparation) {
    const auto &cold_program=*f.preparation;
    const c::WorkloadContext cold_workload{
        take(c::ModelArtifactProfile::create(*cold_program.profile),"actual cold artifact profile"),
        cold_program.phase,{0x48594c454d4f444cULL,cold_program.region,1}};
    prepared=dspark_prepare_constants::prepare(program,backend.owner,backend.queue,cold_workload,
        [&](const g::Graph &graph,g::CompileOptions options){return backend.compile(graph,options);});
    ++backend.submissions;f.preparation.reset();
  } else if(!f.prepared_constants.empty())throw std::runtime_error("prepared constants lack their actual cold producer");
  const auto graph=sem::make_graph(f);
  std::vector<c::ReadLease> parameters;
  std::vector<g::ConstantBinding> bindings;
  for(std::size_t i=0;i<f.constants.size();++i) {
    const auto external=std::find_if(logical.constants.begin(),logical.constants.end(),
        [&](const auto &item){return item.value.ordinal()==f.constants[i].ordinal();});
    if(external!=logical.constants.end()) {
      bindings.push_back(take(b::constant(graph.constants[i],external->role),"share actual original KV/norm/frequency"));
      continue;
    }
    const auto cold=std::find(f.prepared_constants.begin(),f.prepared_constants.end(),i);
    if(cold!=f.prepared_constants.end()) parameters.push_back(take(backend.owner.adopt_buffer(
        std::move(prepared[std::size_t(cold-f.prepared_constants.begin())])),"retain actual cold output"));
    else parameters.push_back(upload(backend.owner,f.constant_bytes[i]));
    bindings.push_back({.value=graph.constants[i],.tensor=take(parameters.back().tensor(f.description(f.constants[i])),"immutable parameter"),
        .owner=backend.owner,.lease=parameters.back()});
  }
  DraftRunFor<Source> result;result.resident_slot_bytes=expert::kSlotBytes;result.bank_ids=resident_weights::kBankIds;result.select=expert::select;result.full_inputs=f.inputs;result.value_count=f.descriptions.size();
  // Remember original semantic provenance before phase-local ValueId remaps.
  // A shared immutable operand can be observed once in an earlier phase and
  // still be referenced directly as the same constant by a later consumer.
  std::vector<std::optional<sem::ValueId>> copied_sources(f.descriptions.size());
  for(const auto &node:f.nodes)
    if(node.operation.template get_if<c::Copy>()&&node.inputs.size()==1&&node.outputs.size()==1)
      copied_sources.at(node.outputs[0].ordinal())=node.inputs[0];
  for(auto value:f.outputs)result.observed_sources.push_back(copied_sources.at(value.ordinal()).value_or(value));
  result.source=std::move(source);
  if(result.source) {
    if(!program.observations||!program.observations->residency)throw std::runtime_error("original expert router frontiers missing");
    result.plan=resident::split(f,*program.observations->residency);
    result.outputs=result.plan->full_outputs;
    for(const auto &phase:result.plan->phases) {
      const auto built=sem::make_graph(phase.fragment);
      std::vector<g::ConstantBinding> shared_bindings;
      for(std::size_t i=0;i<phase.original_constants.size();++i) {
        const auto full_id=phase.original_constants[i].ordinal();
        const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](auto id){return id.ordinal()==full_id;});
        if(found==f.constants.end())throw std::runtime_error("resident phase constant absent from complete model");
        auto shared=bindings.at(std::size_t(found-f.constants.begin()));shared.value=built.constants[i];shared_bindings.push_back(std::move(shared));
      }
      result.phases.push_back(backend.prepare(built.graph,shared_bindings,2));
    }
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    if(backend.cuda) {
      if(!backend.resident_device)throw std::runtime_error("original CUDA expert residency needs the actual typed compiled IO device resource");
      result.cuda_supply=std::make_unique<resident_weights::CudaSupply>(take(resident_weights::CudaSupply::create(
          backend.context,backend.owner,*backend.resident_device),"bounded original CUDA expert slots on the compiled IO resource"));
    } else
#endif
    result.cpu_supply=std::make_unique<resident_weights::CpuSupply>(take(resident_weights::CpuSupply::create(backend.owner),"bounded original CPU expert storage"));
  } else {
    result.phases.push_back(backend.prepare(graph.graph,bindings,2));
    for(std::size_t port=0;port<f.outputs.size();++port)result.outputs.push_back({0,port});
  }
  f.constant_bytes.clear();f.constant_bytes.shrink_to_fit();return result;
}


struct TargetStep final {
  target_program::Recipe logical;
  TargetRun run;
  std::vector<std::size_t> ports;
};
struct TargetPrograms final {
  static constexpr std::int64_t first_position=2173,capacity=2184;
  // Original and prepared parameters, native resources and mappings outlive
  // all ordinary phase Invocations, including exceptional partial execution.
  target_roles::Bound original;
  std::array<target_roles::ReadonlyRole,3> output_a;
  std::vector<TargetStep> steps;
  std::vector<c::TensorDesc> state_descriptions;
  std::size_t completed_rows=0;
};
c::WorkloadContext model_workload(const sem::Fragment &f) {
  if(!f.profile)throw std::runtime_error("actual model Fragment has no original artifact profile");
  return {take(c::ModelArtifactProfile::create(*f.profile),"actual formal target artifact profile"),
      f.phase,{0x48594c454d4f444cULL,f.region,1}};
}
std::vector<g::ConstantBinding> bind_target_constants(Backend &backend,
    const sem::Fragment &f,const std::vector<target_program::Constant> &original,
    const sem::GraphCase &built) {
  std::vector<g::ConstantBinding> result;
  for(std::size_t i=0;i<f.constants.size();++i) {
    const auto value=f.constants[i];
    const auto found=std::find_if(original.begin(),original.end(),
        [&](const auto &item){return item.value.ordinal()==value.ordinal();});
    if(found!=original.end())result.push_back(take(found->role.binding(built.constants[i]),"same original target bank binding"));
    else {
      const auto lease=upload(backend.owner,f.constant_bytes.at(i));
      result.push_back({.value=built.constants[i],.tensor=take(lease.tensor(f.description(value)),"small model plan constant"),
          .owner=backend.owner,.lease=lease});
    }
  }
  return result;
}
TargetPrograms prepare_target_programs(Backend &backend,std::shared_ptr<const target_roles::Data> source) {
  TargetPrograms result;result.original=target_roles::bind(std::move(source),backend.owner);
  {
    auto cold=target_program::prepare_output_a(result.original);
    const auto built=sem::make_graph(cold.fragment);
    const auto constants=bind_target_constants(backend,cold.fragment,cold.constants,built);
    auto run=backend.prepare(built.graph,constants,1,model_workload(cold.fragment));
    auto output=backend.execute(run,{});
    if(output.size()!=3)throw std::runtime_error("formal target output-A preparation must publish exactly three BF16 banks");
    for(std::size_t stage=0;stage<3;++stage) {
      result.output_a[stage]={"original target output-A prepared BF16",dense(c::DataType::kBFloat16,{8,4096,1024}),backend.owner,output[stage]};
      // The encoded original OA is no longer a hot-device constant. Data
      // retains its original mapped bytes for the independent local oracle.
      result.original.roles[stage][target_roles::index(target_roles::Role::kOutputA)].reset();
    }
  }
  const auto layout=take(target_weights::Layout::create({
      result.original.source->expert_description(0,target_roles::Leg::kGate),
      result.original.source->expert_description(0,target_roles::Leg::kUp),
      result.original.source->expert_description(0,target_roles::Leg::kDown)},6),"actual original FP4 E6 layout");
  auto cpu_supply=std::shared_ptr<target_weights::CpuSupply>{};
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  auto cuda_supply=std::shared_ptr<target_weights::CudaSupply>{};
  if(backend.cuda) {
    if(!backend.resident_device)throw std::runtime_error("formal target weight slots need the actual compiled IO device");
    cuda_supply=std::make_shared<target_weights::CudaSupply>(take(target_weights::CudaSupply::create(
        backend.context,backend.owner,*backend.resident_device,layout),"same-device original FP4 target supply"));
  } else
#endif
  cpu_supply=std::make_shared<target_weights::CpuSupply>(take(target_weights::CpuSupply::create(backend.owner,layout),"bounded original FP4 CPU supply"));
  for(std::size_t stage=0;stage<3;++stage) {
    auto types=target_math::descriptions(target_roles::compression_ratio(stage),TargetPrograms::capacity);
    result.state_descriptions.insert(result.state_descriptions.end(),types.begin(),types.end());
  }
  for(std::int64_t offset=0;offset<=6;++offset) {
    TargetStep step;
    step.logical=target_program::build(result.original,result.output_a,
        target_math::Plan::make(TargetPrograms::first_position+offset,TargetPrograms::capacity),
        0x4453544152470000ULL+std::uint64_t(offset)*8);
    auto &f=step.logical.fragment;const auto full=sem::make_graph(f);
    const auto bindings=bind_target_constants(backend,f,step.logical.constants,full);
    auto &run=step.run;run.source=result.original.source;run.cpu_supply=cpu_supply;run.select=expert::select;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    run.cuda_supply=cuda_supply;
#endif
    run.semantic_input_count=22;run.router_value_count=6;run.resident_capacity=6;
    run.resident_slot_bytes=layout.slot_bytes;run.bank_ids=target_weights::kBankIds;
    run.full_inputs=f.inputs;run.value_count=f.descriptions.size();
    run.plan=model_resident_fragment::split(f,step.logical.routers,f.region);
    run.outputs=run.plan->full_outputs;
    std::vector<std::optional<sem::ValueId>> copied(f.descriptions.size());
    for(const auto &node:f.nodes)if(node.operation.template get_if<c::Copy>()&&node.inputs.size()==1&&node.outputs.size()==1)
      copied.at(node.outputs[0].ordinal())=node.inputs[0];
    for(const auto value:f.outputs)run.observed_sources.push_back(copied.at(value.ordinal()).value_or(value));
    step.ports.assign(f.descriptions.size(),std::numeric_limits<std::size_t>::max());
    for(std::size_t i=0;i<f.outputs.size();++i)step.ports.at(f.outputs[i].ordinal())=i;
    for(const auto &phase:run.plan->phases) {
      const auto built=sem::make_graph(phase.fragment);
      std::vector<g::ConstantBinding> constants;
      for(std::size_t i=0;i<phase.original_constants.size();++i) {
        const auto value=phase.original_constants[i];
        const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](auto item){return item.ordinal()==value.ordinal();});
        if(found==f.constants.end())throw std::runtime_error("actual target phase constant has no original producer");
        auto binding=bindings.at(std::size_t(found-f.constants.begin()));binding.value=built.constants[i];constants.push_back(std::move(binding));
      }
      run.phases.push_back(backend.prepare(built.graph,constants,2,model_workload(phase.fragment)));
    }
    // Every cold plan shares original immutable parameters and one bounded
    // E6 upload supply. Only small position constants differ between plans.
    f.constant_bytes.clear();f.constant_bytes.shrink_to_fit();result.steps.push_back(std::move(step));
  }
  return result;
}

g::Graph target_captures(std::int64_t rows) {
  auto builder=take(g::GraphBuilder::create(),"projected target capture fixture");
  std::array<g::ValueId,3> previous;
  for(auto &input:previous) input=take(builder.input(dense(c::DataType::kFloat32,{1,4,b::kHidden})),"prior target capture");
  const auto ids=take(builder.input(dense(c::DataType::kInt32,{rows})),"actual proposal input IDs");
  const auto positions=take(builder.input(dense(c::DataType::kInt32,{rows})),"actual verification positions");
  const auto scaled=[&](g::ValueId value) {
    auto wide=take(builder.add(c::Cast{c::DataType::kFloat32},value),"ID/position conversion");
    auto factor=take(builder.constant(dense(c::DataType::kFloat32,{})),"explicit feature factor");
    return std::pair{wide,factor};
  };
  // Both factors are supplied as immutable constants below by their order.
  auto [token,token_scale]=scaled(ids);
  auto [position,position_scale]=scaled(positions);
  auto tf=take(builder.add(c::Binary{c::BinaryKind::kMultiply},token,token_scale),"token feature");
  auto pf=take(builder.add(c::Binary{c::BinaryKind::kMultiply},position,position_scale),"position feature");
  auto features=take(builder.add(c::Binary{c::BinaryKind::kAdd},tf,pf),"actual-ID target feature");
  auto shape=take(c::View::reshape(take(c::TensorShape::create(std::array<std::int64_t,3>{rows,1,1}),"capture feature shape")),"feature reshape");
  auto expanded=take(builder.add(shape,features),"broadcast target features");
  for(auto input:previous) check(builder.publish(take(builder.add(c::Binary{c::BinaryKind::kAdd},input,expanded),"target capture continuation")),"capture output");
  return take(std::move(builder).finish(),"capture Graph");
}
Run prepare_target(Backend &backend,std::int64_t rows) {
  auto graph=target_captures(rows);
  const std::array<float,2> factors{1.F/131072.F,1.F/512.F};
  std::array<g::ConstantBinding,2> bindings;
  for(std::size_t i=0;i<2;++i) {
    const auto lease=upload(backend.owner,bytes<float>(std::span(&factors[i],1)));
    bindings[i]={.value=take(graph.constant_id(i),"target factor ID"),
      .tensor=take(lease.tensor(dense(c::DataType::kFloat32,{})),"target factor"),.owner=backend.owner,.lease=lease};
  }
  return backend.prepare(std::move(graph),bindings);
}
Run prepare_target_head(Backend &backend,std::int64_t rows) {
  auto builder=take(g::GraphBuilder::create(),"finite target verifier head");
  auto input=take(builder.input(dense(c::DataType::kFloat32,{rows,b::kHidden})),"actual main input");
  auto weight=take(builder.constant(dense(c::DataType::kFloat32,{b::kHidden,17})),"finite target score weights");
  auto scores=take(builder.add(c::MatMul{},input,weight),"finite verifier scores");
  auto node=take(builder.add_node(c::Operation(c::TopK{.count=1,.stable=true}),std::array{scores}),"finite verifier TopK");
  auto selected=take(builder.result(node,1),"actual target ID result");
  auto flat=take(builder.add(take(c::View::reshape(take(c::TensorShape::create(std::array{rows}),"target ID shape")),"target ID view"),selected),"target ID reshape");
  auto materialized=take(builder.add(c::Copy{},flat),"independent target ID generation");
  check(builder.publish(materialized),"target ID output");auto graph=take(std::move(builder).finish(),"finite verifier Graph");
  std::vector<float> weights(std::size_t(b::kHidden*17));
  for(int id=0;id<17;++id) {weights[std::size_t(id)]=float(id-8)/8;weights[std::size_t(17+id)]=float((id*7)%13-6)/8;}
  const auto lease=upload(backend.owner,bytes<float>(weights));
  const std::array bindings{g::ConstantBinding{.value=weight,.tensor=take(lease.tensor(dense(c::DataType::kFloat32,{b::kHidden,17})),"finite verifier weight"),
      .owner=backend.owner,.lease=lease}};
  return backend.prepare(std::move(graph),bindings);
}
Bytes encoded_reference(std::span<const std::uint16_t> input) {
  const auto rows=input.size()/std::size_t(b::kKv);
  const auto width=std::size_t(h::kEncoding.row_bytes());Bytes result(rows*width);
  std::vector<std::uint16_t> chunk(row_cache_test::kRows*std::size_t(b::kKv));
  for(std::size_t first=0;first<rows;first+=row_cache_test::kRows) {
    const auto count=(std::min)(row_cache_test::kRows,rows-first);std::fill(chunk.begin(),chunk.end(),0);
    std::copy_n(input.begin()+std::ptrdiff_t(first*b::kKv),count*std::size_t(b::kKv),chunk.begin());
    const auto expected=row_cache_test::reference(h::kEncoding,chunk);
    std::memcpy(result.data()+first*width,expected.bytes.data(),count*width);
  }return result;
}
// Same bounded original-Q8/F64 compensated-dot oracle as the verified main
// history test, now on all five ACTUAL proposal activations. No provider dot or
// dense decoded weight shadow is used; one decoded 4096-element row suffices.
std::vector<float> proposal_norm_reference(const Data &data,std::size_t stage,
                                           std::span<const float> activation) {
  if(activation.size()!=std::size_t(x::base::block*b::kHidden))
    throw std::runtime_error("proposal reference activation extent");
  const auto stride=std::size_t(ggml_row_size(GGML_TYPE_Q8_0,b::kHidden));
  const auto *codec=ggml_get_type_traits(GGML_TYPE_Q8_0);
  std::vector<float> row(std::size_t(b::kHidden),0.F),projected(std::size_t(x::base::block*b::kKv),0.F);
  for(std::size_t output=0;output<std::size_t(b::kKv);++output) {
    codec->to_float(data.wkv[stage].data()+output*stride,row.data(),b::kHidden);
    for(std::int64_t token=0;token<x::base::block;++token) {
      double sum=0,correction=0;
      for(std::size_t column=0;column<row.size();++column) {
        const double term=double(activation[std::size_t(token*b::kHidden)+column])*row[column];
        const double next=sum+term;
        correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;
      }
      projected[std::size_t(token*b::kKv)+output]=float(sum+correction);
    }
  }
  return cpu_c4_flash_sidecars::norm_ref(projected,data.norm[stage]);
}
// Independent reader used by the existing encoded-history attention check.
// Decode the actual stored code/scale bytes, including BF16 materialization.
std::vector<float> decode_reference(std::span<const std::byte> carrier) {
  if(carrier.size()%h::kEncoding.row_bytes()) throw std::runtime_error("proposal reader reference row extent");
  const auto rows=carrier.size()/h::kEncoding.row_bytes();
  std::vector<float> result(rows*std::size_t(b::kKv),0.F);
  for(std::size_t row=0;row<rows;++row) {
    const auto entry=carrier.subspan(row*h::kEncoding.row_bytes(),h::kEncoding.row_bytes());
    for(std::size_t column=0;column<448;++column) {
      const auto code=std::to_integer<std::uint8_t>(entry[column]);
      EXPECT_NE(code&127,127);
      const int exponent=int(std::to_integer<std::uint8_t>(entry[576+column/64]))-127;
      const double decoded=std::ldexp(row_cache_test::value(code&127,false),exponent)*(code&128?-1:1);
      result[row*std::size_t(b::kKv)+column]=float(row_cache_test::decode(row_cache_test::bfloat(float(decoded))));
    }
    for(std::size_t column=448;column<std::size_t(b::kKv);++column) {
      const auto offset=448+(column-448)*2;
      const auto bits=std::uint16_t(std::uint16_t(std::to_integer<std::uint8_t>(entry[offset]))|
          std::uint16_t(std::to_integer<std::uint8_t>(entry[offset+1]))<<8);
      result[row*std::size_t(b::kKv)+column]=float(row_cache_test::decode(bits));
    }
    EXPECT_EQ(entry[583],std::byte{0});
  }
  return result;
}
using Image=std::vector<Bytes>;
Image image(const x::State &state,const c::StateVersion &version) {
  Image result;
  for(const auto entry:state.entries)result.push_back(read(take(version.member(entry),"actual complete state member")));
  for(const auto entry:state.target_entries)result.push_back(read(take(version.member(entry),"actual target cache member")));
  if(state.pending_entry)result.push_back(read(take(version.member(*state.pending_entry),"actual target pending member")));
  return result;
}

// Check each local arithmetic boundary against its ACTUAL producer. In
// particular, no F64 HC result is fed through a later sigmoid, Sinkhorn or
// BF16 decision, and the old generated-coefficient references stay intact.
template<class Source,class Get> void check_original_parameters(const Source &source,
    const x::base::DraftObservations &ports,Get &&get) {
  const auto close=[](std::span<const float> actual,std::span<const float> expected) {
    sem::oracle::expect_close(actual,expected,8.0e-5F,4.0e-5F);
  };
  const auto norm=[&](const x::base::ds::NormPorts &p,std::span<const float> coefficients) {
    const auto input=get(p.input),raw=get(p.raw),stored=get(p.stored);
    close(raw,hc_reference::rms(input,coefficients));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(stored,raw,"original RMS store"));
  };
  const auto normalized_mix=[&](std::span<const float> projected,std::span<const float> inverse,
                                std::span<const float> actual,std::size_t width) {
    if(projected.size()!=inverse.size()*width||actual.size()!=projected.size())
      throw std::runtime_error("original HC normalized mix extent");
    std::vector<float> expected(projected.size());
    for(std::size_t i=0;i<expected.size();++i)expected[i]=projected[i]*inverse[i/width];
    close(actual,expected);
  };
  const auto pre=[&](const x::base::ds::HyperPrePorts &p,std::span<const float> fn,
                     std::span<const float> scale,std::span<const float> base,const std::string &label) {
    SCOPED_TRACE(label);
    const auto state=get(p.state),flat=get(p.flat),projected=get(p.projection),inverse=get(p.inverse),mixes=get(p.mixes);
    EXPECT_EQ(flat,state);
    EXPECT_TRUE(hc_reference::expect_linear(projected,hc_reference::linear(flat,fn,24),label));
    close(inverse,hc_reference::inverse_rms(flat,std::size_t(4*b::kHidden)));
    normalized_mix(projected,inverse,mixes,24);
    const auto split=hc_reference::split(mixes,scale,base);
    const auto actual_pre=get(p.pre),actual_post=get(p.post),actual_combination=get(p.combination);
    close(actual_pre,split.pre);close(actual_post,split.post);close(actual_combination,split.comb);
    const auto collapsed=get(p.collapsed_raw);
    close(collapsed,hc_reference::collapse(state,actual_pre));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(get(p.collapsed),collapsed,label+" collapse"));
  };
  const auto post=[&](const x::base::ds::HyperPostPorts &p,const x::base::ds::HyperPrePorts &before,
                      const std::string &label) {
    SCOPED_TRACE(label);
    const auto branch=get(p.branch),residual=get(p.residual),expanded=get(p.expanded);
    close(expanded,hc_reference::expand(branch,residual,get(before.post),get(before.combination)));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(get(p.stored),expanded,label+" expand"));
  };
  using S=hc::StageRole;
  for(std::size_t stage=0;stage<3;++stage) {
    SCOPED_TRACE(::testing::Message()<<"actual original HC stage="<<stage);
    const auto &p=ports.stages[stage];const auto prefix="blk."+std::to_string(stage);
    pre(p.attention_hc,source.values(stage,S::kAttentionFn),source.values(stage,S::kAttentionScale),
        source.values(stage,S::kAttentionBase),prefix+".hc_attn");
    norm(p.attention_norm,source.values(stage,S::kAttentionNorm));
    norm(p.attention.query_norm,source.values(stage,S::kQueryNorm));
    const auto &sink=p.attention.sink;
    const auto query=get(sink.query),keys=get(sink.keys),mask=get(sink.mask),raw=get(sink.raw);
    // The existing independent full512-depth reference consumes the actual
    // physical-ring/proposal reader and actual visibility mask, including
    // the source's learned extra softmax denominator for every head.
    close(raw,x::base::ds::old::sink_reference(query,keys,keys,mask,source.values(stage,S::kSinks),
        x::base::block,h::kWindow+x::base::block,1.0F/std::sqrt(512.0F)));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(get(sink.stored),raw,prefix+" sink store"));
    post(p.attention_post,p.attention_hc,prefix+" attention post");
    pre(p.ffn_hc,source.values(stage,S::kFfnFn),source.values(stage,S::kFfnScale),
        source.values(stage,S::kFfnBase),prefix+".hc_ffn");
    norm(p.ffn_norm,source.values(stage,S::kFfnNorm));
    post(p.ffn_post,p.ffn_hc,prefix+" FFN post");
  }
  SCOPED_TRACE("actual original output HC");
  const auto &p=ports.head;
  const auto state=get(p.state),flat=get(p.flat),projected=get(p.projection),inverse=get(p.inverse),mixes=get(p.mixes);
  EXPECT_EQ(flat,state);
  EXPECT_TRUE(hc_reference::expect_linear(projected,
      hc_reference::linear(flat,source.values(hc::HeadRole::kFn),4),"output_hc_fn"));
  close(inverse,hc_reference::inverse_rms(flat,std::size_t(4*b::kHidden)));
  normalized_mix(projected,inverse,mixes,4);
  const auto gate=get(p.gate),collapsed=get(p.collapsed_raw);
  close(gate,hc_reference::head_gate(mixes,source.values(hc::HeadRole::kScale),source.values(hc::HeadRole::kBase)));
  close(collapsed,hc_reference::collapse(state,gate));
  EXPECT_TRUE(hc_reference::expect_bfloat_store(get(p.collapsed),collapsed,"original head collapse"));
  norm(p.output_norm,source.values(hc::HeadRole::kOutputNorm));
}

void report_original_projection(const Run &run,std::size_t port,const std::string &label) {
  const auto observed=take(run.graph.output_id(port),"original HC projection observer");
  const auto copied=take(run.graph.producer(observed),"original HC independent observer producer");
  if(take(run.graph.operation_kind(copied),"original HC observer kind")!=c::OperationKind::kCopy)
    throw std::runtime_error("original HC projection must have independent Copy output");
  const auto projected=take(run.graph.node_input(copied,0),"original HC actual projection");
  const auto node=take(run.graph.producer(projected),"original HC actual MatMul");
  const auto operation=take(run.graph.operation(node),"original HC actual operation");
  const auto *matmul=operation.template get_if<c::MatMul>();
  if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32)
    throw std::runtime_error("original HC projection must preserve explicit F32 MatMul accumulation");
  for(std::size_t input=0;input<2;++input)
    if(take(run.graph.value_description(take(run.graph.node_input(node,input),"original HC operand")),
        "original HC operand description").data_type()!=c::DataType::kFloat32)
      throw std::runtime_error("original HC projection must preserve original F32 coefficients and widened operands");
  std::size_t steps=0;
  for(std::size_t index=0;index<run.compiled.report.step_count();++index) {
    const auto selected=take(run.compiled.report.step(index),"original HC selected step");
    for(std::size_t logical=0;logical<selected.node_count;++logical)
      if(take(run.compiled.report.step_node(index,logical),"original HC selected logical node")==node) {
        ++steps;::testing::Test::RecordProperty(label+"_provider",std::string(selected.provider.name));
      }
  }
  if(steps!=1)throw std::runtime_error("original HC MatMul must occur in exactly one selected plan step");
}

std::string report_shared_projection(const Run &run,std::size_t port,std::size_t parameter) {
  const auto observed=take(run.graph.output_id(port),"original shared projection output");
  const auto copied=take(run.graph.producer(observed),"original shared projection observer");
  if(take(run.graph.operation_kind(copied),"shared observer kind")!=c::OperationKind::kCopy)
    throw std::runtime_error("original shared projection needs an independent Copy output");
  const auto actual=take(run.graph.node_input(copied,0),"actual original shared projection");
  const auto node=take(run.graph.producer(actual),"original shared MatMul");
  const auto operation=take(run.graph.operation(node),"shared MatMul operation");
  const auto *matmul=operation.template get_if<c::MatMul>();
  const auto weight=take(run.graph.node_input(node,1),"actual original shared packed bank");
  if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32||
      take(run.graph.value_description(weight),"actual shared RHS description")!=shared_ffn::description(parameter)||
      take(run.graph.value_description(take(run.graph.node_input(node,0),"actual shared input")),
          "actual shared input type").data_type()!=c::DataType::kFloat32||
      take(run.graph.value_description(actual),"actual shared result type").data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("original shared projection must retain its complete canonical Q8_0 RHS and F32 accumulation/result");
  std::size_t steps=0;std::string provider;
  for(std::size_t index=0;index<run.compiled.report.step_count();++index) {
    const auto selected=take(run.compiled.report.step(index),"original shared selected step");
    for(std::size_t logical=0;logical<selected.node_count;++logical)
      if(take(run.compiled.report.step_node(index,logical),"original shared logical node")==node) {
        ++steps;provider=selected.provider.name;
      }
  }
  if(steps!=1)throw std::runtime_error("original shared MatMul must occur in exactly one selected step");
  ::testing::Test::RecordProperty("shared_projection"+std::to_string(parameter)+"_provider",provider);
  return provider;
}

std::string report_packed_vocabulary_head(const Run &run,std::size_t port,g::ValueId original_weight) {
  const auto copied=take(run.graph.producer(take(run.graph.output_id(port),"packed head output")),"packed head Copy");
  if(!take(run.graph.operation(copied),"packed head Copy operation").template get_if<c::Copy>())
    throw std::runtime_error("packed vocabulary logits must retain an actual Copy output");
  const auto actual=take(run.graph.node_input(copied,0),"packed head actual result");
  const auto node=take(run.graph.producer(actual),"packed head actual MatMul");
  const auto operation=take(run.graph.operation(node),"packed head MatMul operation");
  const auto *matmul=operation.template get_if<c::MatMul>();
  const auto activation=take(run.graph.node_input(node,0),"packed head actual activation");
  const auto cast_node=take(run.graph.producer(activation),"packed head exact BF16 widening");
  const auto cast_operation=take(run.graph.operation(cast_node),"packed head widening operation");
  const auto *cast=cast_operation.template get_if<c::Cast>();
  const auto source=take(run.graph.node_input(cast_node,0),"packed head original BF16 normalized producer");
  if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32||!cast||cast->output_type!=c::DataType::kFloat32||
      take(run.graph.node_input(node,1),"packed head exact immutable bank")!=original_weight||
      take(run.graph.value_description(original_weight),"packed head original RHS")!=packed_vocabulary::description(packed_vocabulary::Role::kHead)||
      take(run.graph.value_description(source),"packed head BF16 producer type").data_type()!=c::DataType::kBFloat16||
      take(run.graph.value_description(actual),"packed head result type").data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("packed head must preserve one complete original Q8 RHS and exact BF16-to-F32 activation");
  std::size_t steps=0;std::string provider;
  for(std::size_t i=0;i<run.compiled.report.step_count();++i) {
    const auto selected=take(run.compiled.report.step(i),"packed head selected step");
    for(std::size_t j=0;j<selected.node_count;++j)
      if(take(run.compiled.report.step_node(i,j),"packed head selected node")==node){++steps;provider=selected.provider.name;}
  }
  if(steps!=1)throw std::runtime_error("packed full-V head must execute once in one selected native step");
  return provider;
}

// Bound only additions of their actual F32 predecessors. The explicit
// quantum/overflow preconditions exclude FTZ and overflow assumptions; this
// is independent of the much larger original Q8 projection error analysis.
template<class Term> void check_shared_sum(std::span<const float> actual,std::size_t terms,
                                           Term &&term,const std::string &label) {
  constexpr double u32=0x1p-24,u64=0x1p-53;
  const auto upper=[](double value){return std::nextafter(value,std::numeric_limits<double>::infinity());};
  const double g32=upper(double(terms)*u32/(1.0-double(terms)*u32)),g64=upper(double(terms)*u64/(1.0-double(terms)*u64));
  double maximum=0;
  for(std::size_t i=0;i<actual.size();++i) {
    double center=0,sumabs=0;
    for(std::size_t slot=0;slot<terms;++slot) {
      const float value=term(i,slot);
      if(!std::isfinite(value)||(value!=0.0F&&std::abs(double(value))<0x1p-103))
        throw std::runtime_error("shared local addition needs separate nonfinite/underflow analysis");
      center+=double(value);sumabs+=std::abs(double(value));
    }
    if(!std::isfinite(sumabs)||sumabs>double(std::numeric_limits<float>::max())/4.0)
      throw std::runtime_error("shared local addition needs separate overflow analysis");
    const double upper_sum=sumabs?upper(sumabs/(1-g64)):0;
    const double bound=sumabs?upper(upper(g32+g64)*upper_sum):0;
    const double error=double(actual[i])==center?0:upper(std::abs(double(actual[i])-center));
    if(!std::isfinite(actual[i])||error>bound) {
      ADD_FAILURE()<<label<<'['<<i<<"] actual="<<actual[i]<<" F64="<<center<<" sumabs="<<sumabs<<" F32 sum bound="<<bound;
      return;
    }
    maximum=std::max(maximum,bound?error/bound:0.0);
  }
  std::printf("DSpark original shared %s terms=%zu maxBoundRatio=%.17g\n",label.c_str(),terms,maximum);
}

template<class Source,class Get> void check_original_shared(const Source &source,
    const x::base::DraftObservations &observations,Get &&get,bool projections,
    shared_reference::ActivationCodec codec) {
  using R=shared_ffn::StageRole;
  for(std::size_t stage=0;stage<3;++stage) {
    SCOPED_TRACE(::testing::Message()<<"actual original shared FFN stage="<<stage);
    if(!observations.stages[stage].shared_ffn)throw std::runtime_error("original shared FFN observations are absent");
    const auto &p=*observations.stages[stage].shared_ffn;
    const auto input=get(p.input),gate_raw=get(p.gate_raw),up_raw=get(p.up_raw),down_raw=get(p.down_raw);
    const auto gate=get(p.gate_stored),up=get(p.up_stored),down=get(p.down_stored);
    const auto activated=get(p.activated_raw),stored=get(p.activated_stored);
    if(projections) {
      for(const auto role:{R::kGate,R::kUp,R::kDown}) {
        const auto parameter=shared_ffn::index(stage,role);const auto &specification=shared_ffn::spec(parameter);
        const auto &actual=role==R::kGate?gate_raw:(role==R::kUp?up_raw:down_raw);
        if constexpr(std::is_same_v<Source,bfloat_sidecar::Data>)
          EXPECT_TRUE(bfloat_reference::expect_projection(actual,role==R::kDown?stored:input,source.raw(specification.name),
              std::size_t(specification.outputs),std::size_t(specification.reduction),specification.name,true));
        else shared_reference::expect_projection(actual,source.bytes(parameter),std::size_t(specification.outputs),
            std::size_t(specification.reduction),role==R::kDown?stored:input,codec,specification.name);
      }
    }
    EXPECT_TRUE(hc_reference::expect_bfloat_store(gate,gate_raw,"original shared gate store"));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(up,up_raw,"original shared up store"));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(down,down_raw,"original shared down store"));
    const auto gate_clipped=get(p.gate_clipped),up_clipped=get(p.up_clipped);
    if(gate.size()!=gate_clipped.size()||up.size()!=up_clipped.size()||gate.size()!=up.size())
      throw std::runtime_error("original shared bounded activation extent differs");
    for(std::size_t i=0;i<gate.size();++i) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(gate_clipped[i]),std::bit_cast<std::uint32_t>(std::min(gate[i],10.0F)))<<i;
      EXPECT_EQ(std::bit_cast<std::uint32_t>(up_clipped[i]),std::bit_cast<std::uint32_t>(std::clamp(up[i],-10.0F,10.0F)))<<i;
    }
    // Reuse the established local F32 nonlinear gate at its actual bounded
    // operands, then check the discontinuous BF16 store independently.
    sem::oracle::expect_close(activated,x::base::ds::moe::gate_reference(x::base::models[1].main_case(),
        gate_clipped,up_clipped),8.0e-5F,4.0e-5F);
    EXPECT_TRUE(hc_reference::expect_bfloat_store(stored,activated,"original shared SwiGLU store"));
    const auto routed=get(p.routed),combined=get(p.combined),full=get(p.full_raw);
    if(combined.size()!=std::size_t(x::base::block*b::kHidden)||routed.size()!=6*combined.size()||
        down.size()!=combined.size()||full.size()!=combined.size())
      throw std::runtime_error("original shared/routed merge geometry differs");
    check_shared_sum(combined,6,[&](std::size_t i,std::size_t slot) {
      return routed[(i/std::size_t(b::kHidden)*6+slot)*std::size_t(b::kHidden)+i%std::size_t(b::kHidden)];
    },"routed six-slot reduction");
    check_shared_sum(full,2,[&](std::size_t i,std::size_t slot){return slot?down[i]:combined[i];},"routed plus original shared");
    EXPECT_TRUE(hc_reference::expect_bfloat_store(get(p.full_stored),full,"original shared final add store"));
  }
}

std::string report_attention_projection(const Run &run,std::size_t port,std::size_t parameter,
    g::ValueId expected_input,std::optional<g::ValueId> expected_ids={}) {
  const auto observed_source=[&](std::size_t output_port) {
    const auto observed=take(run.graph.output_id(output_port),"attention observer value");
    const auto copy=take(run.graph.producer(observed),"attention independent observer");
    if(take(run.graph.operation_kind(copy),"attention observer kind")!=c::OperationKind::kCopy)
      throw std::runtime_error("attention projection observations need independent Copy outputs");
    return take(run.graph.node_input(copy,0),"attention actual observed producer");
  };
  const auto actual=observed_source(port);const auto node=take(run.graph.producer(actual),"original attention projection");
  const auto operation=take(run.graph.operation(node),"original attention operation");
  const auto &specification=attention::spec(parameter);const bool grouped=specification.groups!=1;
  const auto *matmul=operation.template get_if<c::MatMul>();
  if(grouped?!operation.template custom_as<c::model_ops::GroupedMatMul>():
      (!matmul||matmul->accumulation_type!=c::DataType::kFloat32))
    throw std::runtime_error("attention output-a requires natural GroupedMatMul; other projections require F32 MatMul");
  const auto weight=take(run.graph.node_input(node,grouped?0:1),"original attention weight");
  const auto activation=take(run.graph.node_input(node,grouped?1:0),"original attention activation");
  const auto expected=grouped?dense(c::DataType::kFloat32,{x::base::block,specification.groups,specification.outputs}):
      dense(c::DataType::kFloat32,{x::base::block,specification.outputs});
  if(take(run.graph.value_description(weight),"original attention packed bank")!=attention::description(parameter)||
      take(run.graph.value_description(activation),"original attention input type").data_type()!=c::DataType::kFloat32||
      take(run.graph.value_description(actual),"original attention result geometry")!=expected||
      activation!=expected_input)
    throw std::runtime_error("original attention must retain its canonical packed bank, actual named input and natural output geometry");
  if(grouped&&(!expected_ids||
      take(run.graph.node_input(node,2),"actual attention group IDs")!=*expected_ids))
    throw std::runtime_error("original output-a must consume its actual observed group IDs");
  std::size_t steps=0;std::string provider;
  for(std::size_t index=0;index<run.compiled.report.step_count();++index) {
    const auto selected=take(run.compiled.report.step(index),"attention selected step");
    for(std::size_t logical=0;logical<selected.node_count;++logical)
      if(take(run.compiled.report.step_node(index,logical),"attention selected logical node")==node) {
        ++steps;provider=selected.provider.name;
      }
  }
  if(steps!=1)throw std::runtime_error("original attention projection must occur in exactly one selected step");
  ::testing::Test::RecordProperty("attention_projection"+std::to_string(parameter)+"_provider",provider);
  return provider;
}

template<class Source,class Get,class GetIds> void check_original_attention(const Source &source,
    const x::base::DraftObservations &observations,Get &&get,GetIds &&get_ids,
    std::span<const std::int32_t> positions,bool projections,shared_reference::ActivationCodec codec) {
  const auto exact=[](std::span<const float> actual,std::span<const float> expected,const char *label)->void {
    ASSERT_EQ(actual.size(),expected.size())<<label;
    for(std::size_t i=0;i<actual.size();++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i]))<<label<<'['<<i<<']';
  };
  const auto stored=[&](sem::ValueId value,std::span<const float> raw,const char *label) {
    EXPECT_TRUE(hc_reference::expect_bfloat_store(get(value),raw,label));
  };
  const auto rotary=[&](const x::base::ds::RopePorts &p,bool inverse) {
    const auto input=get(p.input),raw=get(p.raw);
    if(input.size()!=raw.size()||input.size()!=positions.size()*64*512)
      throw std::runtime_error("original attention rotary geometry differs");
    // This existing sidecar reference returns raw F32, unlike the separate
    // official whole-chain helper that already rounds its final result.
    const auto reference=cpu_c4_flash_sidecars::rope_ref(input,64,positions,inverse);
    for(std::size_t i=0;i<raw.size();++i) {
      if(i%512<448)EXPECT_EQ(std::bit_cast<std::uint32_t>(raw[i]),std::bit_cast<std::uint32_t>(input[i]))<<i;
      else EXPECT_NEAR(raw[i],reference[i],3.e-6+3.e-6*std::abs(reference[i]))<<i;
    }
    stored(p.stored,raw,inverse?"inverse attention RoPE store":"query attention RoPE store");
  };
  using R=attention::StageRole;
  for(std::size_t stage=0;stage<3;++stage) {
    SCOPED_TRACE(::testing::Message()<<"actual original attention stage="<<stage);
    const auto &stage_ports=observations.stages[stage];
    if(!stage_ports.attention.projections)throw std::runtime_error("original attention observations are absent");
    const auto &p=*stage_ports.attention.projections;
    const auto input=get(p.input),qa_raw=get(p.qa_raw),qa=get(p.qa_stored),qb_raw=get(p.qb_raw),qb=get(p.qb_stored);
    const auto qr=get(stage_ports.attention.query_norm.stored),grouped=get(p.grouped_input),ob_input=get(p.ob_input);
    const auto oa_raw=get(p.oa_raw),oa=get(p.oa_stored),ob_raw=get(p.ob_raw),ob=get(p.ob_stored);
    const auto ids=get_ids(p.group_ids);
    if(ids.size()!=std::size_t(x::base::block*8))throw std::runtime_error("original output-a group ID geometry differs");
    for(std::size_t i=0;i<ids.size();++i)EXPECT_EQ(ids[i],std::int32_t(i%8))<<i;
    exact(input,get(stage_ports.attention_norm.stored),"attention projection input");
    exact(qa,get(stage_ports.attention.query_norm.input),"learned q_a norm actual input");
    exact(qb,get(p.query_norm.input),"query512 norm actual input");
    exact(grouped,get(p.inverse.stored),"eight-group actual inverse-RoPE input");
    exact(ob_input,oa,"output-b actual grouped result");
    exact(ob,get(stage_ports.attention_post.branch),"actual attention HC branch");
    if(projections) {
      for(const auto role:{R::kQueryA,R::kQueryB,R::kOutputA,R::kOutputB}) {
        const auto parameter=attention::index(stage,role);const auto &specification=attention::spec(parameter);
        if constexpr(std::is_same_v<Source,bfloat_sidecar::Data>) {
          if(role==R::kOutputA) {
            const auto actual=bfloat_reference::stored_input(oa_raw);
            EXPECT_TRUE(bfloat_reference::expect_output_a(actual,grouped,source.raw(specification.name),
                std::size_t(specification.groups),std::size_t(specification.outputs),std::size_t(specification.reduction),specification.name,true));
          } else EXPECT_TRUE(bfloat_reference::expect_projection(role==R::kQueryA?qa_raw:(role==R::kQueryB?qb_raw:ob_raw),
              role==R::kQueryA?input:(role==R::kQueryB?qr:ob_input),source.raw(specification.name),
              std::size_t(specification.outputs),std::size_t(specification.reduction),specification.name,true));
        } else {
          if(role==R::kOutputA)attention_reference::expect_grouped_projection(oa_raw,source.bytes(parameter),
              std::size_t(specification.groups),std::size_t(specification.outputs),std::size_t(specification.reduction),
              grouped,ids,codec,specification.name);
          else shared_reference::expect_projection(role==R::kQueryA?qa_raw:(role==R::kQueryB?qb_raw:ob_raw),
              source.bytes(parameter),std::size_t(specification.outputs),std::size_t(specification.reduction),
              role==R::kQueryA?input:(role==R::kQueryB?qr:ob_input),codec,specification.name);
        }
      }
    }
    stored(p.qa_stored,qa_raw,"original query-a store");stored(p.qb_stored,qb_raw,"original query-b store");
    stored(p.oa_stored,oa_raw,"original grouped output-a store");stored(p.ob_stored,ob_raw,"original output-b store");
    const auto &q=p.query_norm;
    const auto q_input=get(q.input),square_raw=get(q.square_raw),square=get(q.square),mean=get(q.mean_raw),rounded_mean=get(q.mean);
    const auto variance_raw=get(q.variance_raw),variance=get(q.variance),inverse_raw=get(q.inverse_raw),inverse=get(q.inverse);
    const auto output_raw=get(q.output_raw),output=get(q.output);
    if(q_input.size()%512||square_raw.size()!=q_input.size()||square.size()!=q_input.size()||
        mean.size()!=q_input.size()/512||rounded_mean.size()!=mean.size()||variance_raw.size()!=mean.size()||
        variance.size()!=mean.size()||inverse_raw.size()!=mean.size()||inverse.size()!=mean.size()||
        output_raw.size()!=q_input.size()||output.size()!=q_input.size())
      throw std::runtime_error("original attention query norm boundary geometry differs");
    std::vector<float> expected_square(q_input.size()),expected_variance(mean.size()),expected_inverse(mean.size()),expected_output(q_input.size());
    for(std::size_t i=0;i<q_input.size();++i) {
      expected_square[i]=float(double(q_input[i])*double(q_input[i]));
      if(!std::isfinite(square[i])||square[i]<0.0F||
          (square[i]!=0.0F&&double(square[i])<0x1p-94)||
          double(square[i])>double(std::numeric_limits<float>::max())/(4.0*512.0))
        throw std::runtime_error("query norm mean needs separate nonfinite/underflow/overflow analysis");
      expected_output[i]=float(double(q_input[i])*double(inverse[i/512]));
    }
    exact(square_raw,expected_square,"actual BF16 query square");stored(q.square,square_raw,"query squared BF16 store");
    check_shared_sum(mean,512,[&](std::size_t row,std::size_t column){return square[row*512+column]*(1.0F/512.0F);},"query512 actual square mean");
    stored(q.mean,mean,"query mean BF16 store");
    for(std::size_t i=0;i<mean.size();++i) {
      expected_variance[i]=float(double(rounded_mean[i])+double(1.0e-6F));
      if(!(variance[i]>0.0F)||!std::isfinite(variance[i]))throw std::runtime_error("query norm actual variance is invalid");
      expected_inverse[i]=float(1.0/std::sqrt(double(variance[i])));
    }
    exact(variance_raw,expected_variance,"actual mean plus epsilon");stored(q.variance,variance_raw,"query variance BF16 store");
    sem::oracle::expect_close(inverse_raw,expected_inverse,8.0e-5F,4.0e-5F);
    stored(q.inverse,inverse_raw,"query inverse BF16 store");
    exact(output_raw,expected_output,"actual BF16 query and inverse product");stored(q.output,output_raw,"query norm BF16 store");
    exact(output,get(p.rotary.input),"query rotary actual input");
    exact(get(stage_ports.attention.sink.stored),get(p.inverse.input),"inverse rotary actual sink input");
    rotary(p.rotary,false);rotary(p.inverse,true);
  }
}

template<class Routed> void report_original_projection(const Routed &draft,std::size_t port,const std::string &label) {
  report_original_projection(draft.run(port),draft.port(port),label);
}
template<class Routed> std::string report_shared_projection(const Routed &draft,std::size_t port,std::size_t parameter) {
  return report_shared_projection(draft.run(port),draft.port(port),parameter);
}
template<class Routed> std::string report_attention_projection(const Routed &draft,std::size_t port,std::size_t parameter,
    std::size_t input_port,std::size_t ids_port=std::numeric_limits<std::size_t>::max()) {
  const auto consumer_phase=draft.outputs.at(port).phase;
  const auto &run=draft.run(port);
  const auto operand=[&](std::size_t full_port)->g::ValueId {
    const auto location=draft.outputs.at(full_port);
    const auto &observed_run=draft.run(full_port);
    const auto copy=take(observed_run.graph.producer(take(observed_run.graph.output_id(location.port),"attention source observation")),"attention source observer producer");
    if(take(observed_run.graph.operation_kind(copy),"attention source observer kind")!=c::OperationKind::kCopy)
      throw std::runtime_error("attention operand must retain its independent actual Copy observation");
    if(location.phase==consumer_phase)return take(run.graph.node_input(copy,0),"same-phase actual attention operand");
    if(!draft.plan)throw std::runtime_error("cross-phase attention provenance requires the actual cold phase map");
    const auto original=draft.observed_sources.at(full_port);
    const auto &phase=draft.plan->phases.at(consumer_phase);
    const auto resolve=[&](const std::vector<sem::ValueId> &originals,bool constant)->std::optional<g::ValueId> {
      const auto found=std::find_if(originals.begin(),originals.end(),[&](auto value){return value.ordinal()==original.ordinal();});
      if(found==originals.end())return std::nullopt;
      const auto index=std::size_t(found-originals.begin());
      return constant?take(run.graph.constant_id(index),"same original immutable attention operand"):
          take(run.graph.input_id(index),"same original cross-phase attention input");
    };
    auto actual=resolve(phase.original_constants,true);
    if(!actual)actual=resolve(phase.original_inputs,false);
    if(!actual || take(run.graph.value_description(*actual),"resolved attention operand type")!=observed_run.outputs.at(location.port))
      throw std::runtime_error("attention operand observation has no exact original-ID binding in its actual consumer phase");
    return *actual;
  };
  return report_attention_projection(run,draft.port(port),parameter,operand(input_port),
      ids_port==std::numeric_limits<std::size_t>::max()?std::nullopt:std::optional<g::ValueId>(operand(ids_port)));
}
template<class Routed> std::string report_expert_projection(const Routed &draft,std::size_t port,std::size_t stage,expert::Leg leg,std::size_t ids_port) {
  const auto &run=draft.run(port);const auto location=draft.outputs.at(port);
  if(location.phase!=stage+1||!draft.plan)throw std::runtime_error("actual expert projection phase");
  const auto copy=take(run.graph.producer(take(run.graph.output_id(location.port),"expert observer")),"expert Copy producer");
  if(take(run.graph.operation_kind(copy),"expert observer operation")!=c::OperationKind::kCopy)throw std::runtime_error("expert projection needs owned Copy observation");
  const auto raw=take(run.graph.node_input(copy,0),"expert actual raw result");
  const auto node=take(run.graph.producer(raw),"expert actual grouped projection");
  if(!take(run.graph.operation(node),"actual expert operation").template custom_as<c::model_ops::GroupedMatMul>())
    throw std::runtime_error("original MXFP4 experts require ordinary GroupedMatMul");
  const auto bank=take(run.graph.node_input(node,0),"actual original MXFP4 bank");
  const auto &phase=draft.plan->phases[location.phase];
  const auto original=draft.plan->routers[stage].banks[std::size_t(leg)];
  const auto found=std::find_if(phase.original_inputs.begin(),phase.original_inputs.end(),[&](auto id){return id.ordinal()==original.ordinal();});
  if(found==phase.original_inputs.end()||bank!=take(run.graph.input_id(std::size_t(found-phase.original_inputs.begin())),"original resident bank input")||
      take(run.graph.value_description(bank),"actual original expert description")!=expert::resident_description(leg)||
      take(run.graph.value_description(raw),"original expert raw type").data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("expert projection must consume actual same-generation compressed resident input and produce F32");
  if(draft.outputs.at(ids_port).phase!=location.phase)throw std::runtime_error("original expert local IDs observation phase");
  const auto ids_copy=take(run.graph.producer(take(run.graph.output_id(draft.port(ids_port)),"actual local ID observer")),"local ID observer Copy");
  if(take(run.graph.operation_kind(ids_copy),"local ID observer operation")!=c::OperationKind::kCopy ||
      take(run.graph.node_input(node,2),"actual selected local expert IDs")!=take(run.graph.node_input(ids_copy,0),"actual mapped local ID result"))
    throw std::runtime_error("expert GroupedMatMul must consume the exact observed global-to-local result");
  std::size_t steps=0;std::string provider;
  for(std::size_t index=0;index<run.compiled.report.step_count();++index) {
    const auto selected=take(run.compiled.report.step(index),"expert selected step");
    for(std::size_t logical=0;logical<selected.node_count;++logical)
      if(take(run.compiled.report.step_node(index,logical),"expert selected node")==node){++steps;provider=selected.provider.name;}
  }
  if(steps!=1)throw std::runtime_error("original expert projection must have exactly one selected implementation");
  ::testing::Test::RecordProperty("expert_stage"+std::to_string(stage)+"_leg"+std::to_string(std::size_t(leg))+"_provider",provider);
  return provider;
}

template<class Routed>
void report_bfloat_projection(const Routed &draft,const x::Draft &logical,std::size_t port,
    std::string_view name,bool grouped=false) {
  const auto location=draft.outputs.at(port);const auto &run=draft.run(port);
  const auto copied=take(run.graph.producer(take(run.graph.output_id(location.port),"S03 projection observer")),"S03 owned Copy");
  if(!take(run.graph.operation(copied),"S03 observer operation").template get_if<c::Copy>())
    throw std::runtime_error("S03 projection needs an independent actual Copy observation");
  auto value=take(run.graph.node_input(copied,0),"S03 real projection result");
  const auto observed_type=take(run.graph.value_description(value),"S03 projection result dtype").data_type();
  auto node=take(run.graph.producer(value),"S03 real projection producer");
  // OA's two layout Copies are real native operations. Follow only those
  // declared layout nodes to the ordinary BF16 BMM, never synthesize a dot.
  if(grouped)for(;;) {
    const auto operation=take(run.graph.operation(node),"S03 OA layout operation");
    if(!operation.template get_if<c::Copy>()&&!operation.template get_if<c::View>())break;
    value=take(run.graph.node_input(node,0),"S03 OA actual layout source");
    node=take(run.graph.producer(value),"S03 OA actual producer");
  }
  const auto operation=take(run.graph.operation(node),"S03 actual projection operation");
  if(grouped) {
    const auto *matmul=operation.template get_if<c::MatMul>();
    if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32||observed_type!=c::DataType::kBFloat16)
      throw std::runtime_error("S03 OA requires real BF16 MatMul output with F32 accumulation");
  } else if(!operation.template custom_as<c::model_ops::DenseLinearF32>()||observed_type!=c::DataType::kFloat32)
    throw std::runtime_error("S03 stored projection requires DenseLinearF32 raw output");
  const auto bank=take(run.graph.node_input(node,1),"S03 actual original BF16 bank");
  if(take(run.graph.value_description(bank),"S03 original bank dtype").data_type()!=c::DataType::kBFloat16||
      take(run.graph.value_description(take(run.graph.node_input(node,0),"S03 actual BF16 operand")),"S03 activation type").data_type()!=c::DataType::kBFloat16)
    throw std::runtime_error("S03 BF16 projection must retain real BF16 operands");
  const auto &constants=draft.plan?draft.plan->phases.at(location.phase).original_constants:logical.program.fragment.constants;
  bool same_original=false;
  for(std::size_t i=0;i<constants.size();++i)if(take(run.graph.constant_id(i),"S03 actual constant Value")==bank) {
    const auto found=std::find_if(logical.constants.begin(),logical.constants.end(),[&](const auto &entry){return entry.value.ordinal()==constants[i].ordinal();});
    if(found!=logical.constants.end()&&found->role.name==name&&found->role.description==take(run.graph.value_description(bank),"S03 same actual original bank layout")&&
        found->role.owner&&found->role.lease) {
      const auto tensor=take(found->role.lease.tensor(found->role.description),"S03 original readonly tensor");
      same_original=tensor.storage().resource()==found->role.owner.ref();
    }
  }
  if(!same_original)throw std::runtime_error("S03 projection bank lost original immutable ordinal/owner identity");
  std::size_t selected=0;
  for(std::size_t i=0;i<run.compiled.report.step_count();++i) {
    const auto step=take(run.compiled.report.step(i),"S03 actual projection provider");
    for(std::size_t j=0;j<step.node_count;++j)if(take(run.compiled.report.step_node(i,j),"S03 actual projection node")==node) {
      ++selected;::testing::Test::RecordProperty(std::string(name)+"_provider",std::string(step.provider.name));
    }
  }
  if(selected!=1)throw std::runtime_error("S03 projection must have one actual selected implementation");
}

template<class Source,class Get,class GetIds> void check_original_experts(const Source &source,
    const x::base::DraftObservations &observations,const std::array<expert::Selection,3> &selections,
    Get &&get,GetIds &&get_ids,bool projections,shared_reference::ActivationCodec codec) {
  const auto exact=[](std::span<const float> actual,std::span<const float> expected,const char *role) {
    ASSERT_EQ(actual.size(),expected.size())<<role;
    for(std::size_t i=0;i<actual.size();++i)EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i]))<<role<<i;
  };
  for(std::size_t stage=0;stage<3;++stage) {
    SCOPED_TRACE(::testing::Message()<<"actual original routed experts stage="<<stage);
    if(!observations.stages[stage].routed_experts)throw std::runtime_error("original expert observations missing");
    const auto &p=*observations.stages[stage].routed_experts;
    const auto input=get(p.input),logits=get(p.logits),softplus=get(p.softplus),scores=get(p.scores),selection=get(p.selection);
    const auto selected=get(p.selected_scores),sum=get(p.sum),normalized=get(p.normalized),weights=get(p.weights);
    const auto ids=get_ids(p.ids),local_ids=get_ids(p.local_ids),map=get_ids(p.map);
    if constexpr(std::is_same_v<Source,bfloat_sidecar::Data>) {
      EXPECT_TRUE(bfloat_reference::expect_projection(logits,input,source.raw(bfloat_sidecar::stage_name(stage,"ffn_gate_inp.weight")),
          256,4096,"S03 original BF16 router",true));
      expert_reference::expect_routing({logits,softplus,scores,selection,selected,sum,normalized,weights,ids},
          source.floats(bfloat_sidecar::stage_name(stage,"exp_probs_b.bias")),x::base::models[1].main_case().scaling,"actual S03 original router");
    } else {
      expert_reference::expect_router(logits,input,source.routers[stage],"original BF16 router");
      expert_reference::expect_routing({logits,softplus,scores,selection,selected,sum,normalized,weights,ids},
          source.biases[stage],x::base::models[1].main_case().scaling,"actual 256-domain original router");
    }
    const auto &selected_generation=selections[stage];
    expert_reference::expect_mapping(ids,local_ids,map,selected_generation.globals);
    ASSERT_EQ(map.size(),selected_generation.map.size());
    for(std::size_t global=0;global<map.size();++global)EXPECT_EQ(map[global],selected_generation.map[global]);
    const auto gate_raw=get(p.gate_raw),gate=get(p.gate),up_raw=get(p.up_raw),up=get(p.up);
    const auto weighted_raw=get(p.weighted_raw),weighted=get(p.weighted),down_raw=get(p.down_raw),down=get(p.down);
    if(projections)for(const auto leg:{expert::Leg::kGate,expert::Leg::kUp,expert::Leg::kDown}) {
      const bool is_down=leg==expert::Leg::kDown;
      if constexpr(std::is_same_v<Source,bfloat_sidecar::Data>)
        bfloat_reference::expect_selected_projection(is_down?down_raw:(leg==expert::Leg::kGate?gate_raw:up_raw),
            is_down?weighted:input,ids,source,stage,leg,is_down?4096u:2048u,is_down?2048u:4096u,
            codec,"S03 selected MXFP4",true);
      else expert_reference::expect_selected_projection(is_down?down_raw:(leg==expert::Leg::kGate?gate_raw:up_raw),
          source.bank_bytes(stage,leg),256,is_down?4096u:2048u,is_down?2048u:4096u,
          is_down?weighted:input,ids,codec,"original selected MXFP4 stage"+std::to_string(stage)+" leg"+std::to_string(std::size_t(leg)));
    }
    EXPECT_TRUE(hc_reference::expect_bfloat_store(gate,gate_raw,"original expert gate store"));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(up,up_raw,"original expert up store"));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(weighted,weighted_raw,"original route-weighted down input store"));
    EXPECT_TRUE(hc_reference::expect_bfloat_store(down,down_raw,"original expert down store"));
    const auto gate_clipped=get(p.gate_clipped),up_clipped=get(p.up_clipped),activated=get(p.activated);
    ASSERT_EQ(gate.size(),gate_clipped.size());ASSERT_EQ(up.size(),up_clipped.size());
    std::vector<float> expected_gate(gate.size()),expected_up(up.size());
    for(std::size_t i=0;i<gate.size();++i){expected_gate[i]=std::min(gate[i],10.0F);expected_up[i]=std::clamp(up[i],-10.0F,10.0F);}
    exact(gate_clipped,expected_gate,"original expert gate clip");exact(up_clipped,expected_up,"original expert up clip");
    sem::oracle::expect_close(activated,x::base::ds::moe::gate_reference(x::base::models[1].main_case(),gate_clipped,up_clipped),8.0e-5F,4.0e-5F);
    ASSERT_EQ(activated.size(),weights.size()*2048);ASSERT_EQ(weighted_raw.size(),activated.size());
    std::vector<float> expected_weighted(activated.size());
    for(std::size_t i=0;i<activated.size();++i)expected_weighted[i]=float(double(activated[i])*double(weights[i/2048]));
    exact(weighted_raw,expected_weighted,"actual route weight before down quantization");
    const auto combined=get(p.combined);
    ASSERT_EQ(down.size(),6*combined.size());
    check_shared_sum(combined,6,[&](std::size_t i,std::size_t slot){return down[(i/4096*6+slot)*4096+i%4096];},"original MXFP4 six-slot reduction");
  }
}

template<class ExpertSource=expert::Data>
void exercise(const Data &data,c::Context &context,const c::StorageOwner &owner,bool cuda,
                const SharedVocabularyData *vocabulary=nullptr,
                std::shared_ptr<const hc::Data> parameter_source={},
                std::shared_ptr<const shared_ffn::Data> shared_source={},
                std::shared_ptr<const attention::Data> attention_source={},
                std::shared_ptr<const ExpertSource> expert_source={}
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
                ,const c::cuda::MemoryResource *expert_device=nullptr
#endif
                ,std::shared_ptr<const target_roles::Data> target_tail={},bool stochastic=false
                ) {
  constexpr bool use_bfloat=std::is_same_v<ExpertSource,bfloat_sidecar::Data>;
  if constexpr(use_bfloat) {
    if(!data.bfloat_source||data.bfloat_source.get()!=expert_source.get()||!vocabulary||!vocabulary->packed||!target_tail)
      throw std::runtime_error("S03 needs its distinct original source, actual Q8 vocabulary and formal target tail");
  }
  if(stochastic&&(!target_tail||!vocabulary))throw std::runtime_error("stochastic sidecar requires its actual complete target and vocabulary");
  if(parameter_source&&!vocabulary)throw std::runtime_error("original HC journey requires its actual full shared vocabulary");
  if(shared_source&&!parameter_source)throw std::runtime_error("original shared FFN journey requires complete original HC roles");
  if(attention_source&&!shared_source)throw std::runtime_error("original attention journey requires the original shared FFN boundary");
  if(!use_bfloat&&expert_source&&!attention_source)throw std::runtime_error("original expert journey requires complete original attention boundary");
  if(target_tail&&(!expert_source||!vocabulary))throw std::runtime_error("original target tail requires its complete original draft and shared vocabulary");
  auto queue=take(context.create_queue(),"ordinary shared Queue");Backend backend{context,owner,queue,cuda};
  backend.default_workload=c::WorkloadContext{take(c::ModelArtifactProfile::create(use_bfloat?cpu_c4_flash_sidecars::bf16_profile:cpu_c4_flash_sidecars::q8_profile),"selected Q8 sidecar profile"),
      c::WorkloadPhase::kDecode,{0x48594c454d4f444cULL,0x4453505354415445ULL,1}};
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  backend.resident_device=expert_device;
#endif
  const auto actual_target=vocabulary&&vocabulary->packed?b::TargetIdentity{b::kModelRepository,b::kModelRevision,
      std::string(packed_vocabulary::kRepository),std::string(packed_vocabulary::kRevision),std::string(packed_vocabulary::kVariant)}:
      (vocabulary?b::TargetIdentity{b::kModelRepository,b::kModelRevision,
      b::kModelRepository,b::kModelRevision,expert_source?"BF16-vocabulary+Q8_0-MXFP4-full-original-DSpark":(attention_source?"BF16-vocabulary+Q8_0-originalHC-sharedFFN-attention-DSpark":
      (shared_source?"BF16-vocabulary+Q8_0-originalHC-sharedFFN-DSpark":
      (parameter_source?"BF16-vocabulary+Q8_0-originalHC-DSpark":"BF16-shared-vocabulary+Q8_0-DSpark")))}:target());
  // The prepared pool outlives its readonly leases, including unwind.
  c::LeasePool bfloat_oa_pool;
  std::shared_ptr<bfloat_sidecar::Bound> bfloat_bound;
  std::optional<bfloat_parameters::Bindings> bfloat_bindings;
  BfloatFrontier bfloat_frontier{data.bfloat_source,environment("HYLE_DSPARK_BFLOAT_EXPERT_DIR"),
      environment("HYLE_DSPARK_BFLOAT_FRONTIER_FILE"),environment("HYLE_DSPARK_BFLOAT_WAIT_FOR_EXPERTS")=="1"};
  if constexpr(use_bfloat) {
    bfloat_bound=std::make_shared<bfloat_sidecar::Bound>(bfloat_sidecar::bind(data.bfloat_source,owner));
    std::array<b::ReadonlyRole,3> prepared_oa;
    {
      sem::Fragment cold{.profile=&cpu_c4_flash_sidecars::bf16_profile,.region=0x0c0401b0,.quantized=true};
      std::array<sem::ValueId,3> weights;
      for(std::size_t stage=0;stage<3;++stage) {
        const auto &bank=bfloat_bound->at(bfloat_sidecar::stage_name(stage,"attn_output_a.weight"));
        weights[stage]=cold.constant(bank.description,{});
        cold.outputs.push_back(dspark_stored_projection_fragment::prepare_output_a(cold,weights[stage],8,1024,4096));
      }
      const auto built=sem::make_graph(cold);std::vector<g::ConstantBinding> constants;
      for(std::size_t stage=0;stage<3;++stage)constants.push_back(take(b::constant(built.constants[stage],
          bfloat_bound->at(bfloat_sidecar::stage_name(stage,"attn_output_a.weight"))),"same original BF16 OA source"));
      auto run=backend.prepare(built.graph,constants,1,model_workload(cold));
      const auto result=backend.execute(run,{});
      if(result.size()!=3)throw std::runtime_error("S03 cold OA preparation requires three actual BF16 outputs");
      for(std::size_t stage=0;stage<3;++stage)prepared_oa[stage]={bfloat_sidecar::stage_name(stage,"attn_output_a.weight"),
          dense(c::DataType::kBFloat16,{8,4096,1024}),owner,result[stage]};
      bfloat_oa_pool=std::move(run.pool);
    }
    // All cold readers are retired. The original mapped source is retained
    // for identity/oracle; only its redundant ready-device OA lease is freed.
    for(std::size_t stage=0;stage<3;++stage)
      bfloat_bound->roles.at(bfloat_bound->source->ordinal(bfloat_sidecar::stage_name(stage,"attn_output_a.weight"))).lease={};
    bfloat_bindings.emplace(bfloat_parameters::make(bfloat_bound,std::move(prepared_oa)));
  }
  b::ProjectionRoles projection;
  if constexpr(use_bfloat)projection=bfloat_sidecar::main_roles(*bfloat_bound,actual_target);
  else {
    const auto fc_desc=take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,3*b::kHidden,b::kHidden),"original fc layout");
    projection={actual_target,sidecar(),role("fc.weight",fc_desc,owner,data.fc),
        role("enc.output_norm.weight",dense(c::DataType::kFloat32,{b::kHidden}),owner,bytes<float>(data.main_norm))};
  }
  std::array<b::StageKvRoles,3> roles;
  for(std::size_t stage=0;stage<3;++stage) {
    if constexpr(use_bfloat)roles[stage]=bfloat_sidecar::stage_roles(*bfloat_bound,actual_target,stage);
    else {
      const auto kv_desc=take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,b::kHidden,b::kKv),"original wkv layout");
      const auto prefix="blk."+std::to_string(stage);
      roles[stage]={actual_target,sidecar(),std::int32_t(stage),role(prefix+".attn_kv.weight",kv_desc,owner,data.wkv[stage]),
          role(prefix+".attn_kv_a_norm.weight",dense(c::DataType::kFloat32,{b::kKv}),owner,bytes<float>(data.norm[stage]))};
    }
  }
  const auto frequencies=x::base::ds::rope_frequencies(false,false);
  auto frequency=role("dflash.rope.frequencies",dense(c::DataType::kFloat32,{32}),owner,bytes<float>(frequencies));
  x::vocabulary::Roles shared;
  std::optional<packed_vocabulary::Bound> packed_bound;
  if(vocabulary&&vocabulary->packed) {
    packed_bound.emplace(packed_vocabulary::bind(vocabulary->packed,owner));shared=packed_bound->roles;
    for(const auto *bank:{&shared.embedding,&shared.head}) {
      auto lease=bank->lease;EXPECT_FALSE(lease.try_write());EXPECT_EQ(lease.size(),packed_vocabulary::kBankBytes);
      EXPECT_EQ(take(lease.tensor(bank->description),"actual packed vocabulary readonly tensor").storage().resource(),owner.ref());
    }
  } else if(vocabulary) {
    const auto desc=dense(c::DataType::kBFloat16,{d::vocabulary,d::hidden});
    shared={role("embed.weight",desc,owner,std::as_bytes(std::span(vocabulary->embedding))),
        role("head.weight",desc,owner,std::as_bytes(std::span(vocabulary->head)))};
  }
  hc::Roles original_parameters;
  if(parameter_source)original_parameters=hc::bind(parameter_source,owner);
  shared_ffn::Roles original_shared;
  if(shared_source)original_shared=shared_ffn::bind(shared_source,owner);
  attention::Roles original_attention;
  if(attention_source)original_attention=attention::bind(attention_source,owner);
  expert::Roles original_experts;
  if constexpr(!use_bfloat)if(expert_source)original_experts=expert::bind(expert_source,owner);
  auto logical=take([&]()->hyle::Result<x::Draft> {
    if constexpr(use_bfloat)return x::make_draft(roles,frequency,bfloat_bindings->head,true,&shared,
        nullptr,nullptr,nullptr,nullptr,&bfloat_bindings->parameters,&bfloat_bindings->state->constants);
    else return x::make_draft(roles,frequency,{data.heads.w1,data.heads.w2,data.heads.confidence},
        true,vocabulary?&shared:nullptr,parameter_source?&original_parameters:nullptr,
        shared_source?&original_shared:nullptr,attention_source?&original_attention:nullptr,
        expert_source?&original_experts:nullptr);
  }(),"full encoded draft assembly");
  ASSERT_TRUE(logical.program.encoded);
  if constexpr(use_bfloat) {ASSERT_EQ(logical.constants.size(),75u);logical.program.fragment.profile=&cpu_c4_flash_sidecars::bf16_profile;}
  else ASSERT_EQ(logical.constants.size(),expert_source?72u:(attention_source?66u:(shared_source?54u:(parameter_source?45u:(vocabulary?11u:9u)))));
  auto &fragment=logical.program.fragment;const auto named=*logical.program.encoded;
  std::size_t lm_port=std::numeric_limits<std::size_t>::max(),embedding_port=lm_port,embedding_stored_port=lm_port;
  if(vocabulary) {
    // Observe actual arithmetic results, not a replacement model input.
    // Both outputs have independent caller storage through later episodes.
    const auto table=std::find_if(logical.constants.begin(),logical.constants.end(),
        [](const auto &entry){return entry.role.name=="embed.weight";});
    const auto lm_weight=std::find_if(logical.constants.begin(),logical.constants.end(),
        [](const auto &entry){return entry.role.name=="head.weight";});
    ASSERT_NE(table,logical.constants.end());
    ASSERT_NE(lm_weight,logical.constants.end());
    bool found_embedding=false,found_lm=false;
    const auto count=fragment.nodes.size();
    for(std::size_t i=0;i<count;++i) {
      const auto &node=fragment.nodes[i];
      if(node.operation.template get_if<c::Gather>() && node.inputs.size()==2 &&
          node.inputs[0].ordinal()==table->value.ordinal()) {
        const auto value=node.outputs[0];auto stored=value;
        if(fragment.description(value).data_type()==c::DataType::kFloat32) {
          const auto found=std::find_if(fragment.nodes.begin(),fragment.nodes.begin()+std::ptrdiff_t(count),[&](const auto &candidate) {
            const auto *cast=candidate.operation.template get_if<c::Cast>();
            return cast&&cast->output_type==c::DataType::kBFloat16&&candidate.inputs.size()==1&&candidate.inputs[0].ordinal()==value.ordinal();
          });
          if(found==fragment.nodes.begin()+std::ptrdiff_t(count))throw std::runtime_error("packed embedding lacks its explicit BF16 entry store");
          stored=found->outputs[0];
        }
        embedding_port=fragment.outputs.size();
        fragment.outputs.push_back(fragment.one(c::Operation(c::Copy{}),{value}));
        embedding_stored_port=embedding_port;
        if(stored.ordinal()!=value.ordinal()) {
          embedding_stored_port=fragment.outputs.size();
          fragment.outputs.push_back(fragment.one(c::Operation(c::Copy{}),{stored}));
        }
        found_embedding=true;
      } else if((node.operation.template custom_as<c::model_ops::DenseLinearF32>()||node.operation.template get_if<c::MatMul>())&&
          node.inputs.size()==2&&node.inputs[1].ordinal()==lm_weight->value.ordinal()&&
          fragment.description(node.outputs[0])==dense(c::DataType::kFloat32,{d::rows,d::vocabulary})) {
        const auto value=node.outputs[0];
        lm_port=fragment.outputs.size();
        fragment.outputs.push_back(fragment.one(c::Operation(c::Copy{}),{value}));
        found_lm=true;
      }
    }
    ASSERT_TRUE(found_embedding);ASSERT_TRUE(found_lm);
  }
  const auto output=[&](sem::ValueId id) {const auto found=std::find_if(fragment.outputs.begin(),fragment.outputs.end(),
        [&](auto value){return value.ordinal()==id.ordinal();});
    if(found==fragment.outputs.end()) throw std::runtime_error("named encoded draft output absent");return std::size_t(found-fragment.outputs.begin());};
  const auto hidden_port=output(named.hidden),norm_port=output(named.normalized),ids_port=output(named.ids),
    markov_port=output(named.markov),logits_port=output(named.logits),confidence_port=output(named.confidence);
  auto proposal=prepare_draft(backend,logical,expert_source);
  if constexpr(use_bfloat)proposal.before_upload=[&](std::size_t stage,std::span<const std::int32_t> ids){bfloat_frontier.ensure(stage,ids);};
  // This reference's CPU arithmetic is tied to the actual fixed worker's
  // AVX nearest-even quantizer (BK/review/cpu-q8-quantizer.json/object
  // disassembly), rather than a capability query from the test executable's
  // unrelated static ggml library. CUDA covers the fixed M5 MMVQ carrier.
  const auto shared_codec=cuda?shared_reference::ActivationCodec::kCudaMmvqQ8_1Fast:
      shared_reference::ActivationCodec::kCpuAvxQ8_0;
  if(vocabulary&&vocabulary->packed) {
    const auto head=std::find_if(logical.constants.begin(),logical.constants.end(),[](const auto &item){return item.role.name=="head.weight";});
    if(head==logical.constants.end())throw std::runtime_error("packed draft head lost its original readonly constant");
    const auto location=proposal.outputs.at(lm_port);const auto &run=proposal.run(lm_port);
    const auto &original_constants=proposal.plan?proposal.plan->phases.at(location.phase).original_constants:fragment.constants;
    const auto found=std::find_if(original_constants.begin(),original_constants.end(),[&](auto value){return value.ordinal()==head->value.ordinal();});
    if(found==original_constants.end())throw std::runtime_error("packed draft head has no exact original constant in its consumer phase");
    const auto provider=report_packed_vocabulary_head(run,location.port,
        take(run.graph.constant_id(std::size_t(found-original_constants.begin())),"actual original draft vocabulary constant"));
    if(provider!=(cuda?"ggml-cuda":"ggml-cpu"))throw std::runtime_error("packed vocabulary activation certificate does not cover selected provider");
    ::testing::Test::RecordProperty("draft_vocabulary_provider",provider);
  }
  if(shared_source) {
    ASSERT_TRUE(logical.program.observations);
    for(std::size_t stage=0;stage<3;++stage) {
      const auto &p=logical.program.observations->stages[stage].shared_ffn;
      ASSERT_TRUE(p);
      const std::array raw{p->gate_raw,p->up_raw,p->down_raw};
      for(std::size_t leg=0;leg<raw.size();++leg) {
        const auto parameter=shared_ffn::index(stage,static_cast<shared_ffn::StageRole>(leg));
        const auto provider=report_shared_projection(proposal,output(raw[leg]),parameter);
        if(provider!=(cuda?"ggml-cuda":"ggml-cpu"))
          throw std::runtime_error("original shared Q8 activation reference does not cover the selected provider");
      }
    }
    ::testing::Test::RecordProperty("shared_q8_activation_codec",cuda?
        "fixed M5 native CUDA MMVQ Q8_1 strict/fast division envelope: source-derived code/halfscale intervals under NVIDIA documented approximate intrinsic accuracy, conditional rather than formal guarantee":
        "actual fixed ggml CPU worker AVX Q8_0 nearest-even: exact arch/x86/quants.c.obj disassembly in BK/review/cpu-q8-quantizer.json; no main-executable capability inference");
  }
  if(attention_source) {
    ASSERT_TRUE(logical.program.observations);
    for(std::size_t stage=0;stage<3;++stage) {
      const auto &p=logical.program.observations->stages[stage].attention.projections;
      ASSERT_TRUE(p);
      const std::array raw{p->qa_raw,p->qb_raw,p->oa_raw,p->ob_raw};
      const std::array inputs{p->input,logical.program.observations->stages[stage].attention.query_norm.stored,
          p->grouped_input,p->ob_input};
      for(std::size_t leg=0;leg<raw.size();++leg) {
        const auto parameter=attention::index(stage,static_cast<attention::StageRole>(leg));
        const auto provider=report_attention_projection(proposal,output(raw[leg]),parameter,output(inputs[leg]),
            leg==2?output(p->group_ids):std::numeric_limits<std::size_t>::max());
        if(provider!=(cuda?"ggml-cuda":"ggml-cpu"))
          throw std::runtime_error("original attention Q8 activation reference does not cover the selected provider");
      }
    }
    ::testing::Test::RecordProperty("attention_output_a_layout","original single Q8_0 role canonical[8,1024,4096],actualF32[5,8,4096],I32ids[5,8]0..7,GroupedMatMul->F32[5,8,1024]; native CUDA tokenbatch5 uses sameMMVQ codec, not40-row dense/MMQ math");
  }
  if(parameter_source) {
    ASSERT_TRUE(logical.program.observations);
    const auto &p=*logical.program.observations;
    for(std::size_t stage=0;stage<3;++stage) {
      report_original_projection(proposal,output(p.stages[stage].attention_hc.projection),"stage"+std::to_string(stage)+"_attention_fn");
      report_original_projection(proposal,output(p.stages[stage].ffn_hc.projection),"stage"+std::to_string(stage)+"_ffn_fn");
    }
    report_original_projection(proposal,output(p.head.projection),"output_hc_fn");
  }
  if(expert_source) {
    ASSERT_TRUE(proposal.plan);ASSERT_EQ(proposal.phases.size(),4u);
    for(std::size_t stage=0;stage<3;++stage) {
      const auto &ports=*logical.program.observations->stages[stage].routed_experts;
      const std::array projected{ports.gate_raw,ports.up_raw,ports.down_raw};
      for(std::size_t leg=0;leg<3;++leg)
        if(report_expert_projection(proposal,output(projected[leg]),stage,static_cast<expert::Leg>(leg),output(ports.local_ids))!=(cuda?"ggml-cuda":"ggml-cpu"))
          throw std::runtime_error("original MXFP4 activation oracle does not cover the selected provider");
    }
  }
  if constexpr(use_bfloat) {
    if(!logical.program.observations)throw std::runtime_error("S03 complete boundary observations missing");
    for(std::size_t stage=0;stage<3;++stage) {
      const auto &ports=logical.program.observations->stages[stage];
      report_original_projection(proposal,output(ports.attention_hc.projection),"S03_stage"+std::to_string(stage)+"_attention_fn");
      report_original_projection(proposal,output(ports.ffn_hc.projection),"S03_stage"+std::to_string(stage)+"_ffn_fn");
      const auto &ffn=*ports.shared_ffn;const std::array ff{ffn.gate_raw,ffn.up_raw,ffn.down_raw};
      constexpr std::array<std::string_view,3> shared_names{"ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
      for(std::size_t leg=0;leg<ff.size();++leg)
        report_bfloat_projection(proposal,logical,output(ff[leg]),bfloat_sidecar::stage_name(stage,shared_names[leg]));
      const auto &attn=*ports.attention.projections;const std::array ap{attn.qa_raw,attn.qb_raw,attn.oa_raw,attn.ob_raw};
      constexpr std::array<std::string_view,4> attention_names{"attn_q_a.weight","attn_q_b.weight","attn_output_a.weight","attn_output_b.weight"};
      for(std::size_t leg=0;leg<ap.size();++leg)
        report_bfloat_projection(proposal,logical,output(ap[leg]),bfloat_sidecar::stage_name(stage,attention_names[leg]),leg==2);
    }
    report_original_projection(proposal,output(logical.program.observations->head.projection),"S03_output_hc_fn");
  }
  auto history1=take(h::make_history(roles,frequency,1),"verified one-row catchup"),history2=take(h::make_history(roles,frequency,2),"verified two-row catchup");
  auto projection1=take(b::make_projection(projection,1,target_tail?c::DataType::kBFloat16:c::DataType::kFloat32),"original single main projection"),
    projection2=take(b::make_projection(projection,2),"original verify2 main projection");
  auto hc1=take(history1.constants(),"single stage constants"),hc2=take(history2.constants(),"verify stage constants");
  auto pc1=take(projection1.constants(),"single fc constants"),pc2=take(projection2.constants(),"verify fc constants");
  auto history_run1=backend.prepare(history1.graph,hc1,target_tail?8:5),history_run2=backend.prepare(history2.graph,hc2);
  auto main_run1=backend.prepare(projection1.graph,pc1,5),main_run2=backend.prepare(projection2.graph,pc2);
  std::optional<Run> captures1,captures2,head1,head2;
  if(!target_tail) {
    captures1.emplace(prepare_target(backend,1));captures2.emplace(prepare_target(backend,2));
    head1.emplace(prepare_target_head(backend,1));head2.emplace(prepare_target_head(backend,2));
  }
  std::optional<TargetPrograms> trained_target;
  std::array<std::optional<x::TargetVerifier>,6> verifiers;
  std::array<std::optional<Run>,6> verifier_runs;

  namespace stochastic_head=dspark_stochastic_head;
  std::array<std::optional<StochasticRun<stochastic_head::Proposal>>,6> sampled_heads;
  std::array<std::optional<StochasticRun<stochastic_head::Verification>>,6> sampled_verification;
  if(stochastic) {
    stochastic_head::HeadSource source{d::hidden,d::vocabulary,fragment.description(named.hidden).data_type(),
        use_bfloat?c::DataType::kBFloat16:c::DataType::kFloat32,{}};
    constexpr std::array<std::string_view,3> names{"markov_w1.weight","markov_w2.weight","conf_proj.weight"};
    if constexpr(use_bfloat) {
      for(std::size_t i=0;i<names.size();++i) {
        const auto &bank=bfloat_bound->at(names[i]);source.roles[i]={names[i],bank.description,bank.owner,bank.lease};
      }
    } else {
      const std::array<std::span<const std::uint16_t>,3> head_values{data.heads.w1,data.heads.w2,data.heads.confidence};
      for(std::size_t i=0;i<names.size();++i) {
        const auto description=i<2?dense(c::DataType::kBFloat16,{d::vocabulary,d::rank}):dense(c::DataType::kBFloat16,{1,d::hidden+d::rank});
        const auto bank=role(std::string(names[i]),description,owner,std::as_bytes(head_values[i]));
        source.roles[i]={names[i],bank.description,bank.owner,bank.lease};
      }
    }
    const auto *profile=fragment.profile;
    for(std::size_t count=0;count<=5;++count) {
      if(count)sampled_heads[count].emplace(prepare_stochastic(backend,stochastic_head::proposal(source,count,{},profile)));
      sampled_verification[count].emplace(prepare_stochastic(backend,stochastic_head::verification(count,{},profile)));
    }
  }
  std::size_t declared_state_budget=64u<<20;
  if(target_tail) {
    trained_target.emplace(prepare_target_programs(backend,target_tail));
    const auto prepare_verifier=[&](x::TargetVerifier &verifier,std::uint32_t generations) {
      auto &f=verifier.fragment;f.profile=&x::base::flash_profile;f.phase=c::WorkloadPhase::kDecode;
      const auto built=sem::make_graph(f);
      const auto binding=take(b::constant(built.constants[0],shared.head),"same original untied target head");
      auto run=backend.prepare(built.graph,std::span(&binding,1),generations,model_workload(f));
      const auto copied=take(run.graph.producer(take(run.graph.output_id(verifier.inputs.size()),"actual target full-V logits output")),"target full-V logits observer");
      if(!take(run.graph.operation(copied),"target logits observer operation").template get_if<c::Copy>())
        throw std::runtime_error("target full-V logits need an independent Copy output");
      const auto result=take(run.graph.node_input(copied,0),"actual single full-V result");
      const auto projected=take(run.graph.producer(result),"actual single full-V head node");
      const auto head_operation=take(run.graph.operation(projected),"actual target head operation");
      if(!(vocabulary->packed?head_operation.template get_if<c::MatMul>()!=nullptr:
           bool(head_operation.template custom_as<c::model_ops::DenseLinearF32>()))||
          take(run.graph.node_input(projected,1),"actual target head bank")!=built.constants[0]||
          take(run.graph.value_description(result),"actual target full-V result description")!=
              dense(c::DataType::kFloat32,{std::int64_t(verifier.inputs.size()),d::vocabulary}))
        throw std::runtime_error("target verifier must preserve one full-V operation and the actual original head bank");
      std::size_t selected_steps=0;
      for(std::size_t i=0;i<run.compiled.report.step_count();++i) {
        const auto selected=take(run.compiled.report.step(i),"actual target head selected step");
        for(std::size_t node=0;node<selected.node_count;++node)
          if(take(run.compiled.report.step_node(i,node),"actual target head selected node")==projected) {
            ++selected_steps;::testing::Test::RecordProperty("target_verifier"+std::to_string(verifier.inputs.size())+"_provider",std::string(selected.provider.name));
          }
      }
      if(selected_steps!=1)throw std::runtime_error("actual target full-V head must occur in one selected plan step");
      if(vocabulary->packed&&report_packed_vocabulary_head(run,verifier.inputs.size(),built.constants[0])!=(cuda?"ggml-cuda":"ggml-cpu"))
        throw std::runtime_error("packed target vocabulary activation certificate does not cover selected provider");
      return run;
    };
    for(std::size_t rows=1;rows<=verifiers.size();++rows) {
      verifiers[rows-1]=take(x::target_verifier(shared.head,std::int64_t(rows)),"actual bounded full-V target verifier");
      verifier_runs[rows-1].emplace(prepare_verifier(*verifiers[rows-1],rows==1?4:1));
    }
    const auto add_budget=[&](std::size_t amount) {
      if(amount>std::numeric_limits<std::size_t>::max()-declared_state_budget)
        throw std::runtime_error("complete target/draft backing budget overflows size_t");
      declared_state_budget+=amount;
    };
    declared_state_budget=0;
    for(const auto &run:sampled_heads)if(run)add_budget(run->run.pool.stats().capacity_bytes);
    for(const auto &run:sampled_verification)if(run)add_budget(run->run.pool.stats().capacity_bytes);
    // Each actual LeasePool root is charged once, including unused diagnostic
    // capacity. StateStore must not charge a small logical view as its root.
    for(const auto &part:proposal.phases)add_budget(part.pool.stats().capacity_bytes);
    for(const auto *run:{&history_run1,&history_run2,&main_run1,&main_run2})
      add_budget(run->pool.stats().capacity_bytes);
    for(const auto &run:verifier_runs)add_budget(run->pool.stats().capacity_bytes);
    for(const auto &step:trained_target->steps)for(const auto &part:step.run.phases)
      add_budget(part.pool.stats().capacity_bytes);
    const auto initial=[&](std::size_t amount) {
      if(amount>std::numeric_limits<std::size_t>::max()-255)throw std::runtime_error("initial state alignment overflow");
      add_budget((amount+255)&~std::size_t{255});
    };
    for(const auto &desc:trained_target->state_descriptions)initial(take(desc.required_bytes(),"original target initial bytes"));
    initial(sizeof(std::int32_t));
    initial(3*h::kWindow*h::kEncoding.row_bytes());initial(std::size_t(b::kHidden)*sizeof(float));
    for(std::size_t i=0;i<3;++i)initial(std::size_t(4*b::kHidden)*sizeof(float));
    add_budget(256*((sizeof(x::Boundary)+63)&~std::size_t{63}));
  }
  const auto target_state=trained_target?std::span<const c::TensorDesc>(trained_target->state_descriptions):std::span<const c::TensorDesc>{};
  if(stochastic)declared_state_budget+=32*64;
  auto state=take(x::State::create(actual_target,target_state,declared_state_budget,stochastic),"whole encoded target/draft state schema");
  struct Verified {std::vector<c::ReadLease> captures,history;std::vector<std::int32_t> ids;};
  const auto verify=[&](const c::StateVersion &from,const b::CaptureFrame &frame,const c::ReadLease &actual_ids) {
    const auto placement=take(h::placement(frame),"actual verification positions");
    const auto positions=upload(owner,bytes<std::int32_t>(placement.positions));
    std::vector<c::ReadLease> input;
    for(std::size_t stage=0;stage<3;++stage) input.push_back(take(from.member(state.entries[4+stage]),"prior real capture member"));
    input.push_back(take(actual_ids.subview(0,std::size_t(frame.rows)*sizeof(std::int32_t)),"actual draft ID prefix"));input.push_back(positions);
    auto produced=backend.execute(frame.rows==1?*captures1:*captures2,input);
    auto &p=frame.rows==1?projection1:projection2;
    const std::array taps{b::TargetTap{frame,b::LayerConvention::kOfficialBlockZeroBased,40,p.capture,produced[0]},
      b::TargetTap{frame,b::LayerConvention::kOfficialBlockZeroBased,41,p.capture,produced[1]},
      b::TargetTap{frame,b::LayerConvention::kOfficialBlockZeroBased,42,p.capture,produced[2]}};
    const auto qualified=take(p.qualify(frame,taps),"qualified actual three-layer capture tuple");
    auto main=backend.execute(frame.rows==1?main_run1:main_run2,qualified);
    auto &hist=frame.rows==1?history1:history2;
    auto lease=take(hist.qualify(frame,frame,main[0]),"actual fc result lease");
    std::vector<c::ReadLease> hi{lease,positions,upload(owner,bytes<std::int32_t>(placement.slots))};
    for(std::size_t stage=0;stage<3;++stage) hi.push_back(take(from.member(state.entries[stage]),"actual encoded history source"));
    auto caught=backend.execute(frame.rows==1?history_run1:history_run2,hi);
    auto target_ids=backend.execute(frame.rows==1?*head1:*head2,std::array{lease});
    for(std::size_t stage=0;stage<3;++stage) {
      const auto bf16=values<std::uint16_t>(caught[std::size_t(frame.rows*4)+stage*3+2]);
      const auto expected=encoded_reference(bf16);const auto old=read(hi[3+stage]);
      for(std::int64_t prefix=1;prefix<=frame.rows;++prefix) {
        auto all=old;
        for(std::int64_t row=0;row<prefix;++row) std::memcpy(all.data()+std::size_t(placement.slots[std::size_t(row)])*h::kEncoding.row_bytes(),
            expected.data()+std::size_t(row)*h::kEncoding.row_bytes(),h::kEncoding.row_bytes());
        EXPECT_EQ(read(caught[std::size_t((prefix-1)*4)+stage]),all);
      }
    }
    caught.resize(std::size_t(frame.rows*4));return Verified{std::move(produced),std::move(caught),values<std::int32_t>(target_ids[0])};
  };
  bool checked_proposal_boundaries=false,checked_original_lm=false,checked_original_shared=false,checked_original_attention=false,checked_original_experts=false;
  std::size_t actual_pending_feedback=0;
  const auto draft=[&](const c::StateVersion &version) {
    const auto facts=take(state.boundary(version),"complete proposal frontier");
    std::vector<std::int32_t> positions(std::size_t(x::base::block),0);std::iota(positions.begin(),positions.end(),std::int32_t(facts.next_position));
    const std::array<std::int32_t,1> pending{facts.pending};
    std::vector<float> mask(std::size_t(x::base::block*(h::kWindow+x::base::block)),0);
    // Main slots are physical ring positions. Before filling the first
    // window, only 0..frontier-1 are visible; all five proposal columns remain
    // visible to every query in the source's parallel draft episode.
    const auto visible=(std::min)(std::int64_t(h::kWindow),facts.next_position);
    for(std::int64_t row=0;row<x::base::block;++row)
      for(std::int64_t column=visible;column<h::kWindow;++column)
        mask[std::size_t(row*(h::kWindow+x::base::block)+column)]=-std::numeric_limits<float>::infinity();
    const auto pending_input=state.pending_entry?
        take(version.member(*state.pending_entry),"direct actual verifier pending producer"):
        upload(owner,bytes<std::int32_t>(pending));
    std::vector<c::ReadLease> input{pending_input,upload(owner,bytes<std::int32_t>(positions)),upload(owner,bytes<float>(mask))};
    for(std::size_t stage=0;stage<3;++stage) input.push_back(take(version.member(state.entries[stage]),"actual committed encoded draft input"));
    const auto old=image(state,version);auto result=execute_draft(backend,proposal,input);EXPECT_EQ(image(state,version),old);
    const auto floating=[&](sem::ValueId value) {
      if(fragment.description(value).data_type()==c::DataType::kFloat32)return values<float>(result[output(value)]);
      if(fragment.description(value).data_type()!=c::DataType::kBFloat16)throw std::runtime_error("draft floating observer dtype differs");
      const auto stored=values<std::uint16_t>(result[output(value)]);std::vector<float> actual;actual.reserve(stored.size());
      for(const auto bits:stored)actual.push_back(std::bit_cast<float>(std::uint32_t(bits)<<16U));return actual;
    };
    if constexpr(use_bfloat) {
      check_original_parameters(BfloatHcView{*expert_source},*logical.program.observations,floating);
      check_original_shared(*expert_source,*logical.program.observations,floating,!checked_original_shared,shared_codec);
      check_original_attention(*expert_source,*logical.program.observations,floating,
          [&](sem::ValueId value){return values<std::int32_t>(result[output(value)]);},positions,!checked_original_attention,shared_codec);
      checked_original_shared=true;checked_original_attention=true;
    }
    if(parameter_source)check_original_parameters(*parameter_source,*logical.program.observations,floating);
    if(shared_source) {
      check_original_shared(*shared_source,*logical.program.observations,
          [&](sem::ValueId value){return values<float>(result[output(value)]);},!checked_original_shared,shared_codec);
      checked_original_shared=true;
    }
    if(attention_source) {
      check_original_attention(*attention_source,*logical.program.observations,
          [&](sem::ValueId value){return values<float>(result[output(value)]);},
          [&](sem::ValueId value){return values<std::int32_t>(result[output(value)]);},
          positions,!checked_original_attention,shared_codec);
      checked_original_attention=true;
    }
    if(expert_source) {
      check_original_experts(*expert_source,*logical.program.observations,proposal.selections,
          [&](sem::ValueId value){return values<float>(result[output(value)]);},
          [&](sem::ValueId value){return values<std::int32_t>(result[output(value)]);},!checked_original_experts,shared_codec);
      checked_original_experts=true;
    }
    if(use_bfloat||!checked_proposal_boundaries) {
      for(std::size_t stage=0;stage<3;++stage) {
        SCOPED_TRACE(::testing::Message()<<"original proposal stage="<<stage);
        const auto &ports=logical.checkpoints[stage];
        const auto activation=values<float>(result[output(ports.activation)]);
        const auto normalized=values<float>(result[output(ports.normalized)]);
        const auto rotated=values<float>(result[output(ports.rotated)]);
        const auto entry=values<std::uint16_t>(result[output(ports.entry)]);
        const auto encoded=read(result[output(ports.encoded)]);
        if constexpr(use_bfloat) {
          if(!ports.raw_projection)throw std::runtime_error("S03 wkv lacks actual raw F32 observation");
          const auto raw=floating(*ports.raw_projection);
          EXPECT_TRUE(bfloat_reference::expect_projection(raw,activation,expert_source->raw(bfloat_sidecar::stage_name(stage,"attn_kv.weight")),
              512,4096,"S03 proposal wkv",!checked_proposal_boundaries));
          sem::oracle::expect_close(normalized,sem::oracle::rms_norm_rows(raw,data.norm[stage],512,1.0e-6F),8.e-5F,4.e-5F);
        } else {
          const auto expected_norm=proposal_norm_reference(data,stage,activation);
          d5_qwen38_24t_moe_profiles::expect_output(&cpu_c4_flash_sidecars::q8_profile,normalized,expected_norm);
        }
        std::vector<float> rounded(normalized.size(),0.F);
        for(std::size_t i=0;i<rounded.size();++i)
          rounded[i]=float(row_cache_test::decode(row_cache_test::bfloat(normalized[i])));
        const auto expected_rotation=cpu_c4_flash_sidecars::rope_ref(rounded,1,positions);
        if(rotated.size()!=rounded.size() || entry.size()!=rotated.size())
          throw std::runtime_error("proposal arithmetic boundary dimensions differ");
        for(std::size_t i=0;i<rotated.size();++i) {
          EXPECT_TRUE(std::isfinite(rotated[i]));
          if(i%std::size_t(b::kKv)<448) EXPECT_EQ(rotated[i],rounded[i])<<i;
          else EXPECT_NEAR(rotated[i],expected_rotation[i],3.e-6+3.e-6*std::abs(expected_rotation[i]))<<i;
          // Test the discontinuous Cast against its ACTUAL F32 predecessor,
          // then the codec against those actual BF16 bits, not a F32 tolerance
          // propagated through several independent rounding decisions.
          EXPECT_EQ(entry[i],row_cache_test::bfloat(rotated[i]))<<i;
        }
        EXPECT_EQ(encoded,encoded_reference(entry));
        auto actual_carriers=old[stage];actual_carriers.insert(actual_carriers.end(),encoded.begin(),encoded.end());
        const auto expected_reader=decode_reference(actual_carriers);
        EXPECT_EQ(read(result[output(ports.decoded)]),bytes<float>(expected_reader));
      }
      checked_proposal_boundaries=true;
    }
    auto ids=values<std::int32_t>(result[ids_port]);EXPECT_EQ(ids.front(),facts.pending);
    const auto hidden=values<float>(result[hidden_port]),normalized=values<float>(result[norm_port]);
    const auto base_logits=vocabulary?values<float>(result[lm_port]):d::lm_reference(normalized);
    const auto expected=original::reference(data.heads,base_logits,hidden,facts.pending);
    EXPECT_EQ(ids,expected.tokens);EXPECT_EQ(values<std::uint16_t>(result[markov_port]),expected.markov);
    EXPECT_TRUE(dspark_head_checks::adjusted_logits_equal(values<float>(result[logits_port]),base_logits,expected));
    if(vocabulary) EXPECT_TRUE(dspark_original_lm_reference::confidence_equal(
        values<float>(result[confidence_port]),hidden,values<std::uint16_t>(result[markov_port]),data.heads.confidence));
    else EXPECT_TRUE(dspark_head_checks::projection_equal(values<float>(result[confidence_port]),expected.confidence,"original encoded-draft confidence"));
    if(vocabulary) {
      const auto embedded=values<std::uint16_t>(result[embedding_stored_port]);
      if(embedded.size()!=std::size_t(d::rows*d::hidden))
        throw std::runtime_error("actual shared embedding observation extent differs");
      if(vocabulary->packed) {
        std::vector<float> stored;stored.reserve(embedded.size());
        for(const auto value:embedded)stored.push_back(std::bit_cast<float>(std::uint32_t(value)<<16U));
        std::vector<std::int32_t> actual_ids(std::size_t(d::rows),x::base::noise_token);actual_ids[0]=facts.pending;
        EXPECT_TRUE(packed_vocabulary_reference::expect_embedding(values<float>(result[embedding_port]),stored,
            vocabulary->packed->raw(packed_vocabulary::Role::kEmbedding),actual_ids,std::size_t(d::vocabulary),std::size_t(d::hidden)));
        EXPECT_TRUE(packed_vocabulary_reference::expect_logits(base_logits,normalized,
            vocabulary->packed->raw(packed_vocabulary::Role::kHead),std::size_t(d::vocabulary),std::size_t(d::hidden),
            shared_codec,!checked_original_lm,std::span<const std::int32_t>(ids).subspan(1)));
        checked_original_lm=true;
      } else {
        for(std::size_t row=0;row<std::size_t(d::rows);++row) {
          const auto token=row?x::base::noise_token:std::int64_t(facts.pending);
          EXPECT_TRUE(std::equal(embedded.begin()+std::ptrdiff_t(row*d::hidden),
              embedded.begin()+std::ptrdiff_t((row+1)*d::hidden),
              vocabulary->embedding.begin()+std::ptrdiff_t(token*d::hidden)));
        }
      }
      if(!vocabulary->packed&&!checked_original_lm) {
        const auto reference=dspark_original_lm_reference::evaluate(vocabulary->head,normalized);
        EXPECT_TRUE(dspark_original_lm_reference::logits_equal(base_logits,reference));
        EXPECT_TRUE(dspark_original_lm_reference::composed_logits_equal(
            values<float>(result[logits_port]),reference,expected));
        checked_original_lm=true;
      }
    }
    if constexpr(use_bfloat)if(::testing::Test::HasFailure())
      throw std::runtime_error("S03 actual original local observer failed before State publication");
    if(state.pending_entry)++actual_pending_feedback;
    return result;
  };
  if(trained_target) {
    struct Row final {
      std::array<c::ReadLease,21> target;
      std::array<c::ReadLease,3> captures;
      std::array<c::ReadLease,4> history;
      c::ReadLease normalized;
    };
    std::size_t observed_rows=0,full_target_lm_values=0,head_calls=0,publications=0,
      continued_branches=0,cancelled_drafts=0,stale_rejections=0,peak_backing=0,
      main_submissions=0,history_submissions=0;
    const auto account=[&] {
      const auto stats=state.store.stats();peak_backing=(std::max)(peak_backing,stats.backing_bytes);
      EXPECT_LE(stats.backing_bytes,declared_state_budget);EXPECT_LE(stats.versions,32u);
      EXPECT_LE(stats.retained_blocks,256u);
    };
    const auto run_row=[&](std::int64_t position,std::int32_t token,
        std::span<const c::ReadLease> previous_target,std::span<const c::ReadLease> previous_rings,
        std::uint64_t epoch) {
      if(previous_target.size()!=21||previous_rings.size()!=3||position<TargetPrograms::first_position||
          position>TargetPrograms::first_position+6||token<0||token>=d::vocabulary)
        throw std::runtime_error("actual target row is outside its complete typed frontier");
      for(const auto cursor:{7u,12u,20u}) {
        const auto actual=values<std::int32_t>(previous_target[cursor]);
        if(actual!=std::vector<std::int32_t>{std::int32_t(position)})
          throw std::runtime_error("actual target cursor differs from the cold decode plan");
      }
      auto &step=trained_target->steps.at(std::size_t(position-TargetPrograms::first_position));
      // The external layer39 boundary is explicitly synthetic. Token identity
      // selects this fixture input; no fake layer40/41/42 capture is injected.
      std::vector<std::uint16_t> upstream(std::size_t(4*b::kHidden));
      for(std::size_t i=0;i<upstream.size();++i) {
        const auto value=std::int64_t((i*13+std::size_t(token)*7+std::size_t(position)*3)%127)-63;
        upstream[i]=row_cache_test::bfloat(float(value)/256.F);
      }
      std::vector<c::ReadLease> inputs{upload(owner,bytes<std::uint16_t>(upstream))};
      inputs.insert(inputs.end(),previous_target.begin(),previous_target.end());
      auto result=execute_draft(backend,step.run,inputs);
      const auto actual=[&](sem::ValueId value)->c::ReadLease {
        const auto port=step.ports.at(value.ordinal());
        if(port!=std::numeric_limits<std::size_t>::max())return result.at(port);
        const auto found=std::find_if(step.logical.fragment.inputs.begin(),step.logical.fragment.inputs.end(),
            [&](auto input){return input.ordinal()==value.ordinal();});
        if(found==step.logical.fragment.inputs.end())throw std::runtime_error("target observer Value has no actual output/input lease");
        const auto ordinal=std::size_t(found-step.logical.fragment.inputs.begin());
        if(ordinal>=inputs.size())throw std::runtime_error("target observer requested a retired resident weight generation");
        return inputs[ordinal];
      };
      const auto get_float=[&](sem::ValueId value) {
        const auto type=step.logical.fragment.description(value).data_type();
        if(type==c::DataType::kFloat32)return values<float>(actual(value));
        if(type!=c::DataType::kBFloat16)throw std::runtime_error("target floating observer has a different semantic dtype");
        const auto source=values<std::uint16_t>(actual(value));std::vector<float> output(source.size());
        for(std::size_t i=0;i<source.size();++i)output[i]=std::bit_cast<float>(std::uint32_t(source[i])<<16u);
        return output;
      };
      const auto get_bytes=[&](sem::ValueId value){return read(actual(value));};
      const auto plan=target_math::Plan::make(position,TargetPrograms::capacity);
      for(std::size_t stage=0;stage<3;++stage)
        target_reference::observe_layer(*target_tail,stage,step.logical.layers[stage],
            step.logical.state_inputs[stage],plan,get_float,get_bytes,observed_rows==0&&!stochastic);
      target_reference::observe_head(*target_tail,step.logical.head,get_float);
      if(::testing::Test::HasFailure())throw std::runtime_error("actual target arithmetic observer failed before any State publication");
      Row row;
      for(std::size_t i=0;i<21;++i)row.target[i]=result.at(i);
      for(std::size_t stage=0;stage<3;++stage)row.captures[stage]=result.at(21+stage);
      row.normalized=result.at(24);
      const b::CaptureFrame frame{actual_target,5,7,epoch,position,1};
      const std::array taps{b::TargetTap{frame,b::LayerConvention::kOfficialBlockZeroBased,40,projection1.capture,row.captures[0]},
        b::TargetTap{frame,b::LayerConvention::kOfficialBlockZeroBased,41,projection1.capture,row.captures[1]},
        b::TargetTap{frame,b::LayerConvention::kOfficialBlockZeroBased,42,projection1.capture,row.captures[2]}};
      const auto qualified=take(projection1.qualify(frame,taps),"actual trained target capture provenance");
      auto main=backend.execute(main_run1,qualified);
      ++main_submissions;
      for(std::size_t stage=0;stage<3;++stage)
        dspark_target_tail_checks::capture_mean(values<float>(row.captures[stage]),
            values<float>(main.at(1+2*stage)),values<std::uint16_t>(main.at(2+2*stage)));
      if constexpr(use_bfloat) {
        if(!projection1.raw_projection_port)throw std::runtime_error("S03 fc raw result was not published");
        std::vector<float> joined;
        for(std::size_t stage=0;stage<3;++stage) {
          const auto mean=values<std::uint16_t>(main.at(2+2*stage));
          for(const auto bits:mean)joined.push_back(std::bit_cast<float>(std::uint32_t(bits)<<16U));
        }
        const auto projected=values<float>(main.at(*projection1.raw_projection_port));
        EXPECT_TRUE(bfloat_reference::expect_projection(projected,joined,expert_source->raw("fc.weight"),4096,12288,"S03 original fc",main_submissions==1&&!stochastic));
        sem::oracle::expect_close(values<float>(main.at(0)),sem::oracle::rms_norm_rows(projected,data.main_norm,4096,1.0e-6F),8.e-5F,4.e-5F);
      }
      const auto placement=take(h::placement(frame),"actual accepted main ring placement");
      std::vector<c::ReadLease> history_inputs{take(history1.qualify(frame,frame,main[0]),"actual original fc/mainnorm producer"),
        upload(owner,bytes<std::int32_t>(placement.positions)),upload(owner,bytes<std::int32_t>(placement.slots))};
      history_inputs.insert(history_inputs.end(),previous_rings.begin(),previous_rings.end());
      auto history=backend.execute(history_run1,history_inputs);
      ++history_submissions;
      for(std::size_t stage=0;stage<3;++stage) {
        if constexpr(use_bfloat) {
          if(!history1.raw_projection_ports[stage])throw std::runtime_error("S03 accepted wkv raw result missing");
          const auto projected=values<float>(history.at(*history1.raw_projection_ports[stage]));
          EXPECT_TRUE(bfloat_reference::expect_projection(projected,values<float>(main.at(0)),
              expert_source->raw(bfloat_sidecar::stage_name(stage,"attn_kv.weight")),512,4096,"S03 accepted main wkv",history_submissions==1&&!stochastic));
          sem::oracle::expect_close(values<float>(history.at(4+stage*3)),sem::oracle::rms_norm_rows(projected,data.norm[stage],512,1.0e-6F),8.e-5F,4.e-5F);
        }
        const auto entry=values<std::uint16_t>(history.at(4+stage*3+2));
        const auto encoded=encoded_reference(entry);auto expected=read(previous_rings[stage]);
        std::memcpy(expected.data()+std::size_t(placement.slots[0])*h::kEncoding.row_bytes(),encoded.data(),encoded.size());
        EXPECT_EQ(read(history.at(stage)),expected);
      }
      if(::testing::Test::HasFailure())throw std::runtime_error("actual target capture/history boundary failed before publication");
      for(std::size_t i=0;i<4;++i)row.history[i]=history.at(i);
      ++observed_rows;++trained_target->completed_rows;return row;
    };
    const auto verify_head=[&](std::span<const Row> rows) {
      if(rows.empty()||rows.size()>6)throw std::runtime_error("actual block5 verifier needs one through six target rows");
      std::vector<c::ReadLease> normalized;std::vector<float> actual;
      for(const auto &row:rows) {
        normalized.push_back(row.normalized);const auto bf16=values<std::uint16_t>(row.normalized);
        for(const auto value:bf16)actual.push_back(std::bit_cast<float>(std::uint32_t(value)<<16u));
      }
      auto output=backend.execute(*verifier_runs.at(rows.size()-1),normalized);
      std::vector<std::int32_t> ids;
      for(std::size_t i=0;i<rows.size();++i) {
        const auto value=values<std::int32_t>(output[i]);
        if(value.size()!=1)throw std::runtime_error("actual target verifier row ID is not an independent I32 scalar");
        ids.push_back(value[0]);auto permission=output[i];EXPECT_FALSE(permission.try_write());
      }
      if(vocabulary->packed) {
        const auto logits=values<float>(output[rows.size()]);
        EXPECT_TRUE(packed_vocabulary_reference::expect_logits(logits,actual,
            vocabulary->packed->raw(packed_vocabulary::Role::kHead),std::size_t(d::vocabulary),std::size_t(d::hidden),shared_codec,head_calls==0&&!stochastic,ids));
        EXPECT_TRUE(glm_shared_vocabulary_reference::expect_greedy(logits,ids,std::size_t(d::vocabulary),values<float>(output[rows.size()+1])));
      } else dspark_target_tail_checks::head(vocabulary->head,actual,values<float>(output[rows.size()]),ids,
          values<float>(output[rows.size()+1]),head_calls==0&&!stochastic);
      if(head_calls==0)full_target_lm_values=rows.size()*std::size_t(d::vocabulary);
      if(::testing::Test::HasFailure())throw std::runtime_error("actual target full-V verifier failed before publication");
      ++head_calls;return output;
    };
    {
      std::vector<c::StateMember> members;
      const std::vector<std::uint16_t> zero_row(std::size_t(b::kKv),0);
      const auto encoded_zero=encoded_reference(zero_row);
      const auto zero_carrier=[&](std::size_t extent) {
        if(extent%encoded_zero.size())throw std::runtime_error("initial main carrier byte extent differs");
        Bytes result(extent);for(std::size_t offset=0;offset<extent;offset+=encoded_zero.size())
          std::copy(encoded_zero.begin(),encoded_zero.end(),result.begin()+std::ptrdiff_t(offset));
        return result;
      };
      for(std::size_t i=0;i<3;++i)members.push_back({state.entries[i],upload(owner,zero_carrier(std::size_t(h::kWindow)*h::kEncoding.row_bytes()))});
      members.push_back({state.entries[3],upload(owner,Bytes(std::size_t(b::kHidden)*sizeof(float)))});
      for(std::size_t i=0;i<3;++i)members.push_back({state.entries[4+i],upload(owner,Bytes(std::size_t(4*b::kHidden)*sizeof(float)))});
      const x::Boundary initial{5,7,1,TargetPrograms::first_position,3,0};
      members.push_back({state.entries[7],{},std::as_bytes(std::span(&initial,1))});
      std::size_t entry=0;
      for(std::size_t stage=0;stage<3;++stage) {
        const auto descriptions=target_math::descriptions(target_roles::compression_ratio(stage),TargetPrograms::capacity);
        for(std::size_t i=0;i<descriptions.size();++i,++entry) {
          auto source=Bytes(take(descriptions[i].required_bytes(),"finite initial target cache extent"));
          if(i==0||i==3)source=zero_carrier(source.size());
          if(i+1==descriptions.size())source=bytes<std::int32_t>(std::array{std::int32_t(TargetPrograms::first_position)});
          members.push_back({state.target_entries.at(entry),upload(owner,source)});
        }
      }
      members.push_back({*state.pending_entry,upload(owner,bytes<std::int32_t>(std::array<std::int32_t,1>{3}))});
      const hyle::inference::SamplingCursor initial_random{123,7,0};
      if(stochastic)members.push_back({*state.random_entry,{},std::as_bytes(std::span(&initial_random,1))});
      auto base=take(state.store.import(members),"complete finite target/draft frontier");members.clear();account();
      const auto old=image(state,base);
      if(stochastic) {
        using Cursor=hyle::inference::SamplingCursor;
        struct Commit final {
          bool committed=false,stopped=false;
          std::array<std::int32_t,6> emitted{};
          std::size_t emitted_count=0;
          x::Boundary frontier{};
          Cursor sampling{};
        };
        std::array<Commit,2> records{};
        std::size_t committed_records=0,record_projections=0,family_observations=0,sampled_markov_rows=0,
          observer_refusals=0,observer_retries=0,duplicate_observations=0;
        // The first m<=2 round can emit three real tokens, followed by the
        // existing ordinary token. No record retains a historical native head.
        hyle::inference::RequestProgress progress(4);progress.initial_sampling=initial_random;
        (void)progress.begin_turn();
        const auto *output_storage=progress.generated.data();const auto output_capacity=progress.generated.capacity();
        std::vector<std::int32_t> emitted;emitted.reserve(6);
        const auto work_counts=[&] {
          return std::array<std::size_t,11>{std::size_t(backend.submissions),proposal.completed_phases,
              proposal.completed_episodes,trained_target->completed_rows,observed_rows,main_submissions,
              history_submissions,head_calls,actual_pending_feedback,sampled_markov_rows,publications};
        };
        const auto observe=[&](std::size_t ordinal,const c::StateVersion &actual) {
          return hyle::inference::observe_record_commit(std::span<const Commit>(records),committed_records,
              ordinal,progress,
              [&](const Commit &record) {
                ++record_projections;
                return hyle::inference::CommitObservation{
                    std::span(record.emitted).first(record.emitted_count),true,true,record.stopped};
              },
              [&](const Commit &record,const hyle::inference::CommitObservation &) {
                const auto current=take(state.boundary(actual),"sidecar observed actual frontier");
                const auto &expected=record.frontier;
                if(current.session!=expected.session||current.branch!=expected.branch||current.epoch!=expected.epoch||
                    current.next_position!=expected.next_position||current.pending!=expected.pending||
                    current.consumed!=expected.consumed||take(state.random(actual),"sidecar observed actual RNG")!=record.sampling||
                    !record.emitted_count||record.emitted[record.emitted_count-1]!=current.pending)
                  throw std::runtime_error("sidecar observation requires its actual committed pending/RNG result");
              },
              [&](const Commit &,const hyle::inference::CommitObservation &) noexcept {++family_observations;});
        };
        const auto key_lease=[&](const Cursor &cursor,std::size_t count,bool proposal_keys) {
          std::vector<std::uint32_t> words;
          for(std::size_t row=0;row<(proposal_keys?count:count+1);++row) {
            const auto address=proposal_keys?2*row:(row==count?2*count:2*row+1);
            const auto key=cursor.identity(address);words.insert(words.end(),key.begin(),key.end());
          }
          return upload(owner,bytes<std::uint32_t>(words));
        };
        {
          auto head=take(c::StateHead::create(state.store,base),"sidecar stochastic complete Head");
          const auto round=[&](std::size_t maximum,std::uint64_t epoch,
              x::Block5Limits bounds={.output_quota=6,.context_rows=6,.vocabulary=d::vocabulary},
              bool fail_observation=false) {
            const auto from=take(head.current(),"sidecar stochastic actual Head");
            const auto facts=take(state.boundary(from),"sidecar stochastic frontier");
            if(committed_records!=progress.observed_commits())
              throw std::runtime_error("sidecar must observe its original committed result before another round");
            if(maximum>5)throw std::runtime_error("sidecar sampled maximum exceeds five trained draft rows");
            if(bounds.draft.max_drafts>5)throw std::runtime_error("sampled sidecar policy exceeds five trained draft rows");
            if(bounds.vocabulary!=d::vocabulary||facts.next_position<0||facts.next_position>TargetPrograms::capacity)
              throw std::runtime_error("sampled sidecar limits differ from source vocabulary/context");
            bounds.draft.max_drafts=static_cast<std::uint32_t>((std::min)(maximum,std::size_t(bounds.draft.max_drafts)));
            bounds.context_rows=(std::min)(bounds.context_rows,std::size_t(TargetPrograms::capacity-facts.next_position));
            const auto capacity=take(x::block5_verification_capacity(bounds),"sampled sidecar bounded row capacity");
            if(!capacity||std::ranges::find(bounds.eos_tokens,facts.pending)!=bounds.eos_tokens.end())
              throw std::runtime_error("sampled sidecar reached output/context/EOS boundary");
            if(committed_records>=records.size())throw std::runtime_error("sidecar cold result record capacity exhausted");
            auto count=capacity-1;
            const auto cursor=take(state.random(from),"sidecar committed random cursor");
            std::vector<std::int32_t> ids{facts.pending};std::vector<c::ReadLease> sampled;
            auto *sampled_head=count?&*sampled_heads[count]:nullptr;
            if(count) {
              auto original=draft(from);
              const std::array input{original[lm_port],take(from.member(*state.pending_entry),"sidecar stochastic true pending"),
                  key_lease(cursor,count,true),original[hidden_port]};
              sampled=backend.execute(sampled_head->run,input);ids=values<std::int32_t>(sampled[sampled_head->recipe.ids]);
              if(ids.size()!=count+1||ids[0]!=facts.pending)throw std::runtime_error("sidecar sampled prefix differs from actual pending");
              for(std::size_t row=0;row<count;++row) {
                if(values<std::uint8_t>(sampled[sampled_head->recipe.valid[row]])!=std::vector<std::uint8_t>{1})
                  throw std::runtime_error("sidecar proposal q invalid before target consumption");
                const auto expected=std::span(data.heads.w1).subspan(std::size_t(ids[row])*std::size_t(d::rank),std::size_t(d::rank));
                EXPECT_EQ(read(sampled[sampled_head->recipe.markov[row]]),bytes<std::uint16_t>(expected));++sampled_markov_rows;
              }
              std::vector<float> confidence;
              for(const auto port:sampled_head->recipe.confidence) {
                const auto value=values<float>(sampled[port]);
                if(value.size()!=1)throw std::runtime_error("sidecar sampled confidence is not scalar");
                confidence.push_back(value[0]);
              }
              count=take(x::block5_sampled_verification_rows(bounds,confidence),"sidecar actual conditional-confidence prefix")-1;
              ids.resize(count+1);
              std::printf("DSpark sidecar random sampled pending=%d ids=",facts.pending);
              for(const auto id:ids)std::printf("%d,",id);std::printf(" draw=%llu\n",static_cast<unsigned long long>(cursor.draw));std::fflush(stdout);
            }
            const auto next=cursor.advanced(2*count+1);
            std::array<c::ReadLease,21> target_input;std::array<c::ReadLease,3> rings;
            for(std::size_t i=0;i<target_input.size();++i)target_input[i]=take(from.member(state.target_entries[i]),"sidecar sampled target state");
            for(std::size_t i=0;i<rings.size();++i)rings[i]=take(from.member(state.entries[i]),"sidecar sampled original rings");
            std::vector<Row> rows;
            for(std::size_t row=0;row<=count;++row) {
              rows.push_back(run_row(facts.next_position+std::int64_t(row),ids.at(row),target_input,rings,epoch));
              target_input=rows.back().target;
              for(std::size_t stage=0;stage<rings.size();++stage)rings[stage]=rows.back().history[stage];
            }
            const auto verified=verify_head(rows);auto &program=*sampled_verification[count];
            std::vector<c::ReadLease> input{verified[count+1],key_lease(cursor,count,false)};
            if(count) {
              input.push_back(upload(owner,bytes<std::int32_t>(std::span(ids).subspan(1))));
              for(std::size_t row=0;row<count;++row)input.push_back(sampled[sampled_head->recipe.q[row]]);
            }
            const auto decisions=backend.execute(program.run,input);
            for(const auto port:program.recipe.valid)if(values<std::uint8_t>(decisions[port])!=std::vector<std::uint8_t>{1})
              throw std::runtime_error("sidecar speculative conditional invalid; Head and RNG unchanged");
            std::size_t accepted=0;std::vector<std::uint8_t> accepts;
            for(const auto port:program.recipe.accepted) {
              const auto valid=values<std::uint8_t>(decisions[port]);if(valid.size()!=1||valid[0]>1)throw std::runtime_error("sidecar random accept not scalar Bool");
              accepts.push_back(valid[0]);
            }
            while(accepted<accepts.size()&&accepts[accepted])++accepted;
            std::vector<std::int32_t> final_ids;
            for(const auto port:program.recipe.final_ids) {
              const auto id=values<std::int32_t>(decisions[port]);if(id.size()!=1)throw std::runtime_error("sidecar final token is not I32 scalar");
              final_ids.push_back(id[0]);
            }
            const auto choice=hyle::inference::mtp::select_sampled_prefix(ids,accepts,final_ids,bounds.output_quota,
                bounds.eos_tokens,d::vocabulary,emitted);
            const auto consumed=static_cast<std::uint32_t>(choice.consumed);const auto selected=choice.consumed-1;
            const auto pending=selected<count&&accepts[selected]?
                take(sampled[sampled_head->recipe.ids].subview((selected+1)*sizeof(std::int32_t),sizeof(std::int32_t)),"sidecar accepted terminal actual pending"):
                decisions[program.recipe.final_ids[count?(std::min)(selected,count-1):0]];
            EXPECT_EQ(values<std::int32_t>(pending),std::vector<std::int32_t>{choice.next_id});
            const b::CaptureFrame frame{actual_target,5,7,epoch,facts.next_position,std::int64_t(consumed)};
            auto candidate=take(state.select_verified(from,rows[selected].history,rows[selected].captures,rows[selected].target,
                pending,frame,consumed,&next),"sidecar sampled complete prefix and RNG");
            if(emitted.empty()||emitted.size()!=choice.consumed||emitted.size()>records.front().emitted.size())
              throw std::runtime_error("sidecar emitted prefix exceeds its cold result extent");
            const auto ordinal=committed_records;auto &record=records[ordinal];
            record.emitted_count=emitted.size();std::copy(emitted.begin(),emitted.end(),record.emitted.begin());
            record.stopped=choice.stopped;record.frontier=take(state.boundary(candidate),"sidecar actual commit frontier");
            record.sampling=take(state.random(candidate),"sidecar actual commit RNG");
            if(!progress.preflight_commit(ordinal,std::span(record.emitted).first(record.emitted_count),true,true,record.stopped))
              throw std::runtime_error("new sidecar result was already observed before Head publication");
            auto commit=take(hyle::inference::prepare_record_commit(std::span<Commit>(records),committed_records,record),
                "prepare original sidecar result commit witness");
            check(commit.publish([&]() noexcept {return head.publish(from.identity(),candidate);}),
                "sidecar model/pending/RNG atomic commit and original record witness");++publications;account();
            EXPECT_TRUE(record.committed);EXPECT_EQ(committed_records,ordinal+1);EXPECT_FALSE(commit.valid());
            EXPECT_EQ(take(state.random(candidate),"sidecar committed advanced cursor"),next);
            EXPECT_EQ(take(state.random(from),"sidecar immutable old cursor"),cursor);
            EXPECT_EQ(take(state.boundary(candidate),"sidecar consumed frontier").next_position,facts.next_position+consumed);
            EXPECT_EQ(read(take(candidate.member(*state.pending_entry),"sidecar actual sampled pending lease")),read(pending));
            EXPECT_TRUE(head.publish(from.identity(),candidate).IsUnavailable());
            std::printf("DSpark sidecar random commit m=%zu accepted=%zu consumed=%u pending=%d draw=%llu\n",count,accepted,consumed,
                take(state.boundary(candidate),"sidecar sampled committed token").pending,static_cast<unsigned long long>(next.draw));std::fflush(stdout);
            if(fail_observation) {
              const auto before=work_counts();const auto stats=state.store.stats();
              const auto output_count=progress.generated.size(),observations=progress.observed_commits();
              EXPECT_THROW((void)observe(ordinal,from),std::runtime_error);++observer_refusals;
              EXPECT_EQ(progress.generated.size(),output_count);EXPECT_EQ(progress.observed_commits(),observations);
              EXPECT_EQ(family_observations,observations);EXPECT_EQ(work_counts(),before);
              EXPECT_EQ(take(head.current(),"sidecar late observer refusal keeps original Head").identity(),candidate.identity());
              EXPECT_EQ(take(state.random(candidate),"sidecar late observer refusal keeps actual RNG"),next);
              EXPECT_EQ(state.store.stats().versions,stats.versions);EXPECT_EQ(state.store.stats().drafts,stats.drafts);
              EXPECT_EQ(state.store.stats().backing_bytes,stats.backing_bytes);
            } else {
              EXPECT_TRUE(observe(ordinal,candidate));
              const auto projections=record_projections,output_count=progress.generated.size();
              EXPECT_FALSE(observe(ordinal,from));++duplicate_observations;
              EXPECT_EQ(record_projections,projections);EXPECT_EQ(progress.generated.size(),output_count);
              EXPECT_EQ(progress.observed_commits(),committed_records);EXPECT_EQ(family_observations,committed_records);
            }
            EXPECT_EQ(progress.generated.data(),output_storage);EXPECT_EQ(progress.generated.capacity(),output_capacity);
            return candidate;
          };
          auto selected=round(2,2,{.output_quota=6,.context_rows=6,.vocabulary=d::vocabulary},true);
          const auto pending=take(state.boundary(selected),"sidecar first random pending").pending;
          {
            const auto before=work_counts();const auto stats=state.store.stats();const auto committed_image=image(state,selected);
            EXPECT_EQ(committed_records,1u);EXPECT_EQ(progress.observed_commits(),0u);EXPECT_TRUE(progress.generated.empty());
            EXPECT_THROW((void)round(0,3),std::runtime_error);
            EXPECT_EQ(work_counts(),before);EXPECT_EQ(take(head.current(),"unobserved sidecar result is not reexecuted").identity(),selected.identity());
            EXPECT_TRUE(observe(0,selected));++observer_retries;
            const auto projections=record_projections;
            EXPECT_FALSE(observe(0,base));++duplicate_observations;
            EXPECT_EQ(record_projections,projections);EXPECT_EQ(progress.observed_commits(),1u);EXPECT_EQ(family_observations,1u);
            EXPECT_EQ(progress.generated.size(),records[0].emitted_count);
            EXPECT_TRUE(std::equal(progress.generated.begin(),progress.generated.end(),records[0].emitted.begin()));
            EXPECT_EQ(progress.generated.back(),pending);EXPECT_EQ(work_counts(),before);
            EXPECT_EQ(take(head.current(),"sidecar observe-only retry never republishes Head").identity(),selected.identity());
            EXPECT_EQ(image(state,selected),committed_image);
            EXPECT_EQ(take(state.random(selected),"sidecar observe-only retry preserves RNG"),records[0].sampling);
            EXPECT_EQ(state.store.stats().versions,stats.versions);EXPECT_EQ(state.store.stats().drafts,stats.drafts);
            EXPECT_EQ(state.store.stats().backing_bytes,stats.backing_bytes);
            EXPECT_EQ(progress.generated.data(),output_storage);EXPECT_EQ(progress.generated.capacity(),output_capacity);
          }
          auto continued=round(0,3);++continued_branches;
          EXPECT_EQ(take(state.random(continued),"sidecar complete two-round RNG").draw,6u);
          EXPECT_EQ(take(state.boundary(continued),"sidecar continuation consumed pending once").next_position,
              take(state.boundary(selected),"sidecar selected position").next_position+1);
          EXPECT_EQ(image(state,base),old);
          EXPECT_EQ(committed_records,2u);EXPECT_EQ(progress.observed_commits(),2u);EXPECT_EQ(family_observations,2u);
          EXPECT_EQ(record_projections,3u);EXPECT_EQ(observer_refusals,1u);EXPECT_EQ(observer_retries,1u);
          EXPECT_EQ(duplicate_observations,2u);EXPECT_EQ(records[1].emitted_count,1u);
          EXPECT_EQ(progress.generated.size(),records[0].emitted_count+records[1].emitted_count);
          EXPECT_EQ(progress.generated.back(),take(state.boundary(continued),"sidecar observed continuation pending").pending);
          ::testing::Test::RecordProperty("actual_random_pending_consumed_next",pending);
        }
        base={};check(state.store.collect(),"sidecar sampled old/candidate retirement");
        EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
        ::testing::Test::RecordProperty("random_publications",int(publications));
        ::testing::Test::RecordProperty("random_pending_continuations",int(continued_branches));
        ::testing::Test::RecordProperty("sampled_markov_rows",int(sampled_markov_rows));
        ::testing::Test::RecordProperty("random_request_committed_records",int(committed_records));
        ::testing::Test::RecordProperty("random_request_observed_records",int(progress.observed_commits()));
        ::testing::Test::RecordProperty("random_request_family_observations",int(family_observations));
        ::testing::Test::RecordProperty("random_request_record_projections",int(record_projections));
        ::testing::Test::RecordProperty("random_request_first_emitted",int(records[0].emitted_count));
        ::testing::Test::RecordProperty("random_request_total_emitted",int(progress.generated.size()));
        ::testing::Test::RecordProperty("random_request_late_observer_refusals",int(observer_refusals));
        ::testing::Test::RecordProperty("random_request_observe_only_retries",int(observer_retries));
        ::testing::Test::RecordProperty("random_request_duplicate_observations",int(duplicate_observations));
        ::testing::Test::RecordProperty("state_members",int(state.schema.size()));
        ::testing::Test::RecordProperty("actual_submissions",backend.submissions);
        ::testing::Test::RecordProperty("state_backing_peak",std::to_string(peak_backing));
        ::testing::Test::RecordProperty("state_backing_budget",std::to_string(declared_state_budget));
        ::testing::Test::RecordProperty("head_storage",use_bfloat?"F32 hidden/Markov visible -> BF16 projection input; original BF16 W":"F32 hidden/Markov projection; original BF16 W, F32 product/accum");
        ::testing::Test::RecordProperty("scope","Actual inventoried sidecar stem/base logits + sampled-predecessor Markov q -> original target40/41/42/head p -> shared stochastic prefix -> complete model/pending/RNG Head and one real pending continuation. Common original record commit/observation with cold multi-token output storage; actual post-publication observer refusal retries only its committed result, without repeating Graph, Head or RNG. Layer39 input/cacheprefix finite fixtures; no RequestLoop, full model, natural acceptance-rate, quality or TPS claim.");
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
        if(proposal.cuda_supply)check(proposal.cuda_supply->close(),"sampled sidecar actual expert upload drain");
        if(cuda)check(trained_target->steps[0].run.cuda_supply->close(),"sampled target actual expert upload drain");
#endif
        return;
      }

      {
        auto head=take(c::StateHead::create(state.store,base),"sidecar bounded strategy Head");
        const auto round=[&](const c::StateVersion &from,x::DraftControl control,std::uint64_t epoch) {
          const auto facts=take(state.boundary(from),"sidecar configured frontier");
          x::Block5Limits bounds{.output_quota=7,.context_rows=std::size_t(TargetPrograms::capacity-facts.next_position),
              .vocabulary=d::vocabulary,.confidence_domain=x::ConfidenceDomain::kRawLogit,.draft=control};
          const auto maximum=take(x::block5_verification_capacity(bounds),"sidecar configured capacity");
          if(!maximum)throw std::runtime_error("sidecar finite strategy has no target capacity");
          auto ids=values<std::int32_t>(take(from.member(*state.pending_entry),"sidecar configured actual pending lease"));
          if(ids!=std::vector<std::int32_t>{facts.pending})throw std::runtime_error("sidecar configured pending metadata/lease mismatch");
          std::vector<float> confidence;
          if(maximum>1) {
            auto proposed=draft(from);ids=values<std::int32_t>(proposed[ids_port]);
            // This original GGUF sidecar fragment publishes raw F32, not an
            // implicitly sigmoided value selected from its file extension.
            if(control.minimum_confidence && *control.minimum_confidence>0.F)
              confidence=values<float>(proposed[confidence_port]);
          }
          const auto count=take(x::block5_verification_rows(bounds,confidence),"sidecar raw confidence prefix");
          EXPECT_LE(count,std::size_t(control.max_drafts)+1);
          std::array<c::ReadLease,21> target_input;std::array<c::ReadLease,3> rings;
          for(std::size_t i=0;i<target_input.size();++i)target_input[i]=take(from.member(state.target_entries[i]),"sidecar configured target tuple");
          for(std::size_t i=0;i<rings.size();++i)rings[i]=take(from.member(state.entries[i]),"sidecar configured main rings");
          std::vector<Row> rows;
          for(std::size_t i=0;i<count;++i) {
            rows.push_back(run_row(facts.next_position+std::int64_t(i),ids.at(i),target_input,rings,epoch));
            target_input=rows.back().target;
            for(std::size_t stage=0;stage<rings.size();++stage)rings[stage]=rows.back().history[stage];
          }
          auto verified_rows=verify_head(rows);std::vector<std::int32_t> target_ids;
          for(std::size_t i=0;i<count;++i)target_ids.push_back(values<std::int32_t>(verified_rows[i])[0]);
          const auto decision=take(x::decide_block5(ids,target_ids,bounds,confidence),"sidecar configured exact verifier decision");
          const auto selected=decision.pending_row;
          const b::CaptureFrame frame{actual_target,5,7,epoch,facts.next_position,std::int64_t(decision.consumed)};
          auto candidate=take(state.select_verified(from,rows[selected].history,rows[selected].captures,rows[selected].target,
              verified_rows[selected],frame,decision.consumed),"sidecar configured complete state");
          check(head.publish(from.identity(),candidate),"sidecar configured atomic publication");++publications;account();
          EXPECT_EQ(read(take(candidate.member(*state.pending_entry),"sidecar configured pending")),read(verified_rows[selected]));
          return candidate;
        };
        auto shortened=round(base,{.max_drafts=2,.minimum_confidence=.5F},2);
        const auto before_disabled=proposal.completed_episodes;
        auto ordinary=round(shortened,{.max_drafts=0,.minimum_confidence=.9F},3);
        EXPECT_EQ(proposal.completed_episodes,before_disabled);
        const auto facts=take(state.boundary(ordinary),"sidecar ordinary complete frontier");
        EXPECT_EQ(facts.next_position,take(state.boundary(shortened),"sidecar shortened frontier").next_position+1);
        auto resumed=draft(ordinary);
        EXPECT_EQ(values<std::int32_t>(resumed[ids_port])[0],facts.pending);
        EXPECT_EQ(image(state,base),old);
        ::testing::Test::RecordProperty("bounded_draft_controls","max2/raw-confidence>=0.5; disabled0 target-only; reenabled5 from complete main-history");
      }
      check(state.store.collect(),"retire sidecar bounded strategy versions");
      auto proposed=draft(base);const auto proposal_image=read(proposed[ids_port]);
      const auto proposal_ids=values<std::int32_t>(proposed[ids_port]);
      if(proposal_ids.size()!=6)throw std::runtime_error("actual complete DSpark output must retain current plus five proposals");
      std::array<c::ReadLease,21> target_input;
      std::array<c::ReadLease,3> rings;
      for(std::size_t i=0;i<21;++i)target_input[i]=take(base.member(state.target_entries[i]),"actual target initial tuple");
      for(std::size_t i=0;i<3;++i)rings[i]=take(base.member(state.entries[i]),"actual draft initial main ring");
      std::array<Row,6> rows;
      for(std::size_t i=0;i<rows.size();++i) {
        rows[i]=run_row(TargetPrograms::first_position+std::int64_t(i),proposal_ids[i],target_input,rings,2);
        target_input=rows[i].target;for(std::size_t stage=0;stage<3;++stage)rings[stage]=rows[i].history[stage];
      }
      auto verified=verify_head(rows);std::vector<std::int32_t> target_ids;
      for(std::size_t i=0;i<6;++i)target_ids.push_back(values<std::int32_t>(verified[i])[0]);
      const auto natural=take(x::decide_block5(proposal_ids,target_ids,
          {.output_quota=7,.context_rows=std::size_t(TargetPrograms::capacity-TargetPrograms::first_position),.vocabulary=d::vocabulary}),
          "natural full block5 greedy acceptance");
      const std::array<std::uint32_t,4> prefixes{natural.accepted_drafts,0,2,5};
      for(std::size_t scenario=0;scenario<prefixes.size();++scenario) {
        check(state.store.collect(),"retire the previous controlled prefix before the next branch");
        const auto accepted=prefixes[scenario];const auto consumed=accepted+1;const auto &selected=rows[accepted];
        auto head=take(c::StateHead::create(state.store,base),"independent natural/controlled prefix Head");
        const b::CaptureFrame frame{actual_target,5,7,2,TargetPrograms::first_position,std::int64_t(consumed)};
        auto candidate=take(state.select_verified(base,selected.history,selected.captures,selected.target,verified[accepted],frame,consumed),
            "one complete successful actual target/draft/pending prefix");account();
        check(head.publish(base.identity(),candidate),"single complete target/draft/pending Head publication");++publications;
        auto current=take(head.current(),"published complete block5 prefix");
        EXPECT_EQ(take(state.boundary(current),"accepted complete protocol").next_position,TargetPrograms::first_position+consumed);
        const auto pending=take(current.member(*state.pending_entry),"published actual correction/bonus lease");
        const auto a=take(pending.view(),"state pending actual storage"),v=take(verified[accepted].view(),"verifier pending producer storage");
        EXPECT_EQ(a.resource(),v.resource());EXPECT_EQ(a.native_handle(),v.native_handle());EXPECT_EQ(a.offset(),v.offset());
        EXPECT_EQ(pending.generation(),verified[accepted].generation());EXPECT_EQ(image(state,base),old);
        const auto current_image=image(state,current);auto next=draft(current);(void)next;
        const auto facts=take(state.boundary(current),"continued own target/draft frontier");
        auto continuation=run_row(facts.next_position,facts.pending,selected.target,
            std::span<const c::ReadLease>(selected.history).first(3),3);
        auto continued_pending=verify_head(std::span(&continuation,1));
        const b::CaptureFrame continued_frame{actual_target,5,7,3,facts.next_position,1};
        auto continued=take(state.select_verified(current,continuation.history,continuation.captures,continuation.target,
            continued_pending[0],continued_frame,1),"continue selected target/draft checkpoint");account();
        check(head.publish(current.identity(),continued),"publish continued actual full state");++publications;++continued_branches;
        EXPECT_TRUE(head.publish(base.identity(),base).IsUnavailable());++stale_rejections;
        EXPECT_EQ(image(state,current),current_image);EXPECT_EQ(image(state,base),old);EXPECT_EQ(read(proposed[ids_port]),proposal_image);
        auto cancelled=take(state.store.begin(continued),"partially staged target failure");
        check(cancelled.replace(state.target_entries[0],selected.target[0]),"stage one target cache without publication");
        check(cancelled.cancel(),"cancel unpublished complete target draft");++cancelled_drafts;
        EXPECT_FALSE(cancelled.finalize());EXPECT_EQ(take(head.current(),"head after cancelled writer").identity(),continued.identity());
      }
      ::testing::Test::RecordProperty("natural_block5_trials",1);
      ::testing::Test::RecordProperty("natural_accepted_drafts",int(natural.accepted_drafts));
      ::testing::Test::RecordProperty("controlled_prefix_publications","0,2,5; explicit StateHead prefix tests, not natural greedy acceptance");
    }
    check(state.store.collect(),"collect complete actual target/draft state");
    EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    if(cuda)check(trained_target->steps[0].run.cuda_supply->close(),"close shared original target FP4 slots after all readers");
    else
#endif
    {
      const auto stats=trained_target->steps[0].run.cpu_supply->stats();
      EXPECT_EQ(stats.live_blocks,0u);EXPECT_EQ(stats.active_uses,0u);EXPECT_LE(stats.peak_bytes,stats.capacity_bytes);
      ::testing::Test::RecordProperty("target_cpu_expert_pool_capacity",std::to_string(stats.capacity_bytes));
      ::testing::Test::RecordProperty("target_cpu_expert_pool_peak",std::to_string(stats.peak_bytes));
    }
    ::testing::Test::RecordProperty("original_target_roles",int(target_roles::kOriginalRoles));
    ::testing::Test::RecordProperty("original_target_source_bytes",std::to_string(target_roles::kOriginalBytes));
    ::testing::Test::RecordProperty("target_state_resources",21);::testing::Test::RecordProperty("complete_state_members",30);
    ::testing::Test::RecordProperty("actual_target_rows",int(observed_rows));
    std::size_t actual_target_phases=0;
    for(const auto &step:trained_target->steps)actual_target_phases+=step.run.completed_phases;
    ::testing::Test::RecordProperty("actual_target_phase_submissions",int(actual_target_phases));
    ::testing::Test::RecordProperty("actual_target_verifier_submissions",int(head_calls));
    ::testing::Test::RecordProperty("actual_main_projection_submissions",int(main_submissions));
    ::testing::Test::RecordProperty("actual_main_history_catchup_submissions",int(history_submissions));
    ::testing::Test::RecordProperty("actual_pending_embedding_feedback_checks",int(actual_pending_feedback));
    ::testing::Test::RecordProperty("target_full_vocabulary_values_checked",int(full_target_lm_values));
    ::testing::Test::RecordProperty("target_expert_domain",256);::testing::Test::RecordProperty("target_resident_experts",6);
    ::testing::Test::RecordProperty("target_compression_ratios","4,128,4");
    ::testing::Test::RecordProperty("target_cold_plan_count",int(trained_target->steps.size()));
    ::testing::Test::RecordProperty("whole_state_publications",int(publications));
    ::testing::Test::RecordProperty("continued_prefix_branches",int(continued_branches));
    ::testing::Test::RecordProperty("cancelled_unpublished_drafts",int(cancelled_drafts));
    ::testing::Test::RecordProperty("rejected_stale_publishes",int(stale_rejections));
    ::testing::Test::RecordProperty("state_backing_budget",std::to_string(declared_state_budget));
    ::testing::Test::RecordProperty("state_backing_peak",std::to_string(peak_backing));
    ::testing::Test::RecordProperty("target_scope","formal Flash0731 original40->41->42, distinct final HC/norm and original fullV target verifier; real block5 draft IDs, actual main-history catchup, all21 target caches plus draft/pending in one Head. Layer39/initial prefix finite fixtures; controlled prefixes separate from natural trial; no full43-layer task quality, acceptance rate or TPS claim.");
  } else {
    std::vector<c::StateMember> members;
    for(std::size_t stage=0;stage<3;++stage) {
      std::vector<std::uint16_t> initial(std::size_t(h::kWindow*b::kKv));
      for(std::size_t i=0;i<initial.size();++i) initial[i]=row_cache_test::bfloat(double(stage+1)+double(i%67)/64);
      members.push_back({state.entries[stage],upload(owner,encoded_reference(initial))});
    }
    std::vector<float> carry(std::size_t(b::kHidden),0);members.push_back({state.entries[3],upload(owner,bytes<float>(carry))});
    for(std::size_t layer=0;layer<3;++layer) {
      std::vector<float> initial(std::size_t(4*b::kHidden));
      for(std::size_t i=0;i<initial.size();++i) initial[i]=float(int((i*17+layer*23)%101)-50)/64;
      members.push_back({state.entries[4+layer],upload(owner,bytes<float>(initial))});
    }
    const x::Boundary initial{5,7,1,126,3,0};members.push_back({state.entries[7],{},std::as_bytes(std::span(&initial,1))});
    auto imported=take(state.store.import(members),"finite imported target boundary");members.clear();
    const std::array<std::int32_t,1> initial_token{3};
    const b::CaptureFrame warm{actual_target,5,7,2,126,1};
    auto bootstrap=verify(imported,warm,upload(owner,bytes<std::int32_t>(initial_token)));
    auto base=take(state.select(imported,bootstrap.history,bootstrap.captures,warm,1,bootstrap.ids[0]),"actual caught-up bootstrap state");
    bootstrap={};const auto old=image(state,base);
    auto proposed=draft(base);const auto proposal_snapshot=read(proposed[ids_port]);
    const b::CaptureFrame frame{actual_target,5,7,3,127,2};auto verified=verify(base,frame,proposed[ids_port]);
    std::array<c::StateHead,3> heads;
    for(std::uint32_t accepted=0;accepted<=2;++accepted) {
      heads[accepted]=take(c::StateHead::create(state.store,base),"accepted/rejected branch head");
      auto candidate=take(state.select(base,verified.history,verified.captures,frame,accepted,verified.ids[accepted?accepted-1:0]),"select one whole accepted prefix");
      if(accepted) check(heads[accepted].publish(base.identity(),candidate),"single captures/history/carry/protocol publication");
      else EXPECT_EQ(candidate.identity(),base.identity());
      auto prior=take(heads[accepted].current(),"nominated branch state");const auto snapshot=image(state,prior);
      if(accepted) {
        for(std::size_t i=0;i<4;++i) EXPECT_EQ(snapshot[i],read(verified.history[std::size_t(accepted-1)*4+i]));
        for(std::size_t i=0;i<3;++i) EXPECT_EQ(snapshot[4+i],read(take(verified.captures[i].subview(std::size_t(accepted-1)*4*b::kHidden*sizeof(float),std::size_t(4*b::kHidden*sizeof(float))),"accepted actual capture slice")));
        EXPECT_TRUE(heads[accepted].publish(base.identity(),base).IsUnavailable());
      } else EXPECT_EQ(snapshot,old);
      auto next=draft(prior);const auto facts=take(state.boundary(prior),"actual resumed frontier");
      const b::CaptureFrame continuation{actual_target,5,7,4+accepted,facts.next_position,1};
      auto continued=verify(prior,continuation,next[ids_port]);
      auto candidate2=take(state.select(prior,continued.history,continued.captures,continuation,1,continued.ids[0]),"continue actual accepted/old branch");
      check(heads[accepted].publish(prior.identity(),candidate2),"atomic continued complete state");
      EXPECT_EQ(take(state.boundary(candidate2),"continued protocol").next_position,facts.next_position+1);
      EXPECT_EQ(take(state.boundary(candidate2),"continued consumed count").consumed,facts.consumed+1);
      EXPECT_EQ(image(state,prior),snapshot);EXPECT_EQ(image(state,base),old);EXPECT_EQ(read(proposed[ids_port]),proposal_snapshot);
    }
    auto cancelled=take(state.store.begin(base),"partial staged failure");
    check(cancelled.replace(state.entries[0],verified.history[0]),"stage one ring without publishing");
    EXPECT_TRUE(cancelled.replace(state.entries[3],verified.history[0]).IsInvalidArgument());
    check(cancelled.cancel(),"cancel whole unpublished draft");EXPECT_FALSE(cancelled.finalize());EXPECT_EQ(image(state,base),old);
  }
  check(state.store.collect(),"collect retired complete state");EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
  if constexpr(use_bfloat) {
    for(const auto &bank:bfloat_bound->roles)if(bank.lease) {
      auto lease=bank.lease;EXPECT_FALSE(lease.try_write());
      EXPECT_EQ(lease.size(),take(bank.description.required_bytes(),"S03 immutable original role extent"));
      EXPECT_EQ(take(lease.tensor(bank.description),"S03 typed immutable bank").storage().resource(),owner.ref());
    }
  } else {
    EXPECT_EQ(read(projection.projection.lease),data.fc);
    for(std::size_t stage=0;stage<3;++stage)EXPECT_EQ(read(roles[stage].projection.lease),data.wkv[stage]);
  }
  if(parameter_source)for(std::size_t i=0;i<hc::kRoles;++i) {
    auto lease=original_parameters.at(i).lease;EXPECT_FALSE(lease.try_write());
    EXPECT_EQ(read(lease),bytes<float>(parameter_source->values(i)))<<hc::spec(i).name;
  }
  if(shared_source)for(std::size_t i=0;i<shared_ffn::kRoles;++i) {
    auto lease=original_shared.at(i).lease;EXPECT_FALSE(lease.try_write());
    const auto expected=shared_source->bytes(i);const auto actual=read(lease);
    EXPECT_TRUE(std::equal(actual.begin(),actual.end(),expected.begin(),expected.end()))<<shared_ffn::spec(i).name;
  }
  if(attention_source)for(std::size_t i=0;i<attention::kRoles;++i) {
    auto lease=original_attention.at(i).lease;EXPECT_FALSE(lease.try_write());
    const auto expected=attention_source->bytes(i);const auto actual=read(lease);
    EXPECT_TRUE(std::equal(actual.begin(),actual.end(),expected.begin(),expected.end()))<<attention::spec(i).name;
  }
  if constexpr(!use_bfloat)if(expert_source)for(std::size_t stage=0;stage<3;++stage) {
    auto router=original_experts.routers[stage].lease,bias=original_experts.biases[stage].lease;
    EXPECT_FALSE(router.try_write());EXPECT_FALSE(bias.try_write());
    EXPECT_EQ(read(router),bytes<std::uint16_t>(expert_source->routers[stage]));
    EXPECT_EQ(read(bias),bytes<float>(expert_source->biases[stage]));
  }
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  if(proposal.cuda_supply)check(proposal.cuda_supply->close(),"retire both original expert upload slots");
#endif
  if(expert_source) {
    EXPECT_TRUE(checked_original_experts);
    ::testing::Test::RecordProperty("resident_draft_phases",int(proposal.phases.size()));
    ::testing::Test::RecordProperty("actual_draft_phase_submissions",int(proposal.completed_phases));
    ::testing::Test::RecordProperty("actual_draft_episodes",int(proposal.completed_episodes));
    for(std::size_t stage=0;stage<3;++stage) {
      std::string sizes;
      for(const auto count:proposal.union_sizes[stage]){if(!sizes.empty())sizes+=",";sizes+=std::to_string(count);}
      ::testing::Test::RecordProperty("actual_expert_unions_stage"+std::to_string(stage),sizes);
    }
    ::testing::Test::RecordProperty("expert_upload_accounting_scope",cuda?
        "per-phase retired WeightBatch report in stdout; logical/staged=401081344,completed=submitted,unknown=false; API accounting,not measured VRAM or physicalbus traffic":
        "ordinary ready StorageOwner CPU compressed operands from one bounded two-generation LeasePool; not CUDA WeightSlots");
    if(proposal.cpu_supply) {
      const auto stats=proposal.cpu_supply->stats();EXPECT_EQ(stats.live_blocks,0u);EXPECT_EQ(stats.active_uses,0u);
      ::testing::Test::RecordProperty("cpu_expert_pool_capacity",std::to_string(stats.capacity_bytes));
      ::testing::Test::RecordProperty("cpu_expert_pool_peak",std::to_string(stats.peak_bytes));
    }
    ::testing::Test::RecordProperty("resident_expert_slot_bytes","401081344");
    ::testing::Test::RecordProperty("expert_projection_reference_episodes",1);
    ::testing::Test::RecordProperty("expert_projection_matrices",9);
    ::testing::Test::RecordProperty("original_expert_numerics","firstepisode nine actual selected MXFP4 projections against original global-ID mmap byte ranges/native AQ bound; everyepisode actual BF16 router F64 dot,original F32 bias,actual256-domain routing/TopK,exactglobal/localmap,clip/SwiGLU/routeweight-before-down and local BF16 stores/six-slot reduction. FourordinaryGraph phases; onlyactual30I32IDs readback,activation/weights/state directleases,compressed3bank+map samegeneration. Whole State tuple published onlyafter completeproposal+verification.");
  }
  ::testing::Test::RecordProperty("actual_submissions",backend.submissions);
  ::testing::Test::RecordProperty("original_role_count",expert_source?83:(attention_source?68:(shared_source?56:(parameter_source?47:(vocabulary?13:11)))));
  const std::uint64_t original_role_bytes=expert_source?13008927644ULL:(attention_source?2734976924ULL:
      (shared_source?2400743324ULL:(parameter_source?2320527260ULL:(vocabulary?2310699520ULL:192576000ULL))));
  ::testing::Test::RecordProperty("original_role_bytes",std::to_string(original_role_bytes-
      (vocabulary&&vocabulary->packed?2*1059061760ULL-packed_vocabulary::kBytes:0)));
  ::testing::Test::RecordProperty("encoded_draft_inputs",expert_source?18:6);
  ::testing::Test::RecordProperty("semantic_draft_outputs",7);
  ::testing::Test::RecordProperty("test_boundary_outputs",int(proposal.outputs.size()-7));
  ::testing::Test::RecordProperty("proposal_numerics","all3 original M5 wkv/norm independent Q8 decode + compensated F64 dot/norm gate on actual activation; actual postnorm BF16 then independent RoPE gate; exact actual-F32->BF16 bits; exact actual-BF16->584B codes/scales; independent actual committed/proposal carrier reader equals prepared attention F32 keys byte-for-byte");
  if(vocabulary) {
    EXPECT_TRUE(checked_original_lm);
    ::testing::Test::RecordProperty("full_vocabulary_lm_values_checked",int(d::rows*d::vocabulary));
    ::testing::Test::RecordProperty("lm_reference_episodes",1);
    ::testing::Test::RecordProperty("confidence_numerics","every-episode independent original BF16 coefficients and actual widened-BF16 rawhidden/actualBF16 Markov dot; exact-F32-product/nonFTZ/overflow preconditions; analytic gamma4352(F32)+F64 absolute-product bound; old relative-gate exceedances/error/sumabs/maxratio in stdout, generated gate unchanged");
    ::testing::Test::RecordProperty("shared_vocabulary",vocabulary->packed?
        "matching main UD-Q4_K_XL independent original Q8_0 embedding/head; native packed Gather/exactBF16entry and one packed fullVMatMul; original readonly bytes, no full-weight F32 shadow":
        "formal Flash0731 BF16 full embedding/head, target-owned immutable banks; no dense F32 LM weight shadow");
    if(parameter_source) {
      ::testing::Test::RecordProperty("original_hc_numerics","every-episode seven actual widened-BF16 x original-F32 dense HC projections with independent F64/product+accum bound; actual inverse RMS, original scale/base Sinkhorn20 and head gates, actual 4-stream collapse/expansion; actual learned RMS/sink inputs and exact local BF16 stores. Seven fn providers recorded from actual CompileReport.");
      if(shared_source) {
        EXPECT_TRUE(checked_original_shared);
        ::testing::Test::RecordProperty("shared_projection_reference_episodes",1);
        ::testing::Test::RecordProperty("shared_projection_matrices",9);
        ::testing::Test::RecordProperty("original_shared_numerics","first-episode all9 original Q8_0 projections over actual5row BF16 predecessors with independent packed-byte decoding/native activation codec arithmetic and derived F32 error bounds; all episodes exact actual bounded operands/BF16 stores, existing local SwiGLU gate, derived F32 six-routed-slot reduction and routed+shared addition bounds. Actual Q8 MatMul providers and canonical packed RHS recorded; no dense F32 weight shadow or official FP8 equivalence.");
        if(attention_source) {
          EXPECT_TRUE(checked_original_attention);
          ::testing::Test::RecordProperty("attention_projection_reference_episodes",1);
          ::testing::Test::RecordProperty("attention_projection_matrices",12);
          ::testing::Test::RecordProperty("attention_projection_values_checked",691200);
          ::testing::Test::RecordProperty("original_attention_numerics","first-episode12 originalQ8_0 projections on actual5row BF16 operands,including3natural8group output-A projections with actualI32 IDs; derived nativeAQ/F32 bounds. Everyepisode exact QA/QB/OA/OB stores,actual learnedqnorm input,11-step query512norm boundaries with gamma512 mean/localrsqrt and exact BF16 stores,actual forward/inverse last64RoPE raw gates and exact stores,group/input topology and unchanged learned-sink/HC/FFN/LM/Markov/confidence/state checks.");
          if(target_tail)::testing::Test::RecordProperty("scope","83 original GGUF Q8/MXFP4 draft roles plus4725 distinct formal Flash0731 target40/41/42/finalHC/norm roles; original shared129280-token BF16 vocabulary, one6row fullV verifier and one30-member StateHead. Actual block5 natural result and separately labelled controlled0/2/5 prefix publications,sourceBF16 capturemean,acceptedmainhistorycatchup,continuation/cancel/stale/oldresults/boundedroots. Layer39 and initial cacheprefix are finite fixtures; not full43-layer inference,task quality,naturalacceptance rate or TPS. Formal target E4/UE8-A128/FP4group32 is distinct from selected GGUF draft math.");
          else if(expert_source)::testing::Test::RecordProperty("scope","83 original roles: all81 fixed Q8_0 sidecar DSpark roles including original BF16router/F32bias and9MXFP4experts,plusformalBF16 fullVembedding/head. Complete3stage draft/encodedstate,accept0/1/2/wrap/continuation/retainedresults. CompactE30 bank residency preserves all256globalIDs,actualrouter outcomes and original17B/32codec. Generated trained-target captures/verifier remain explicit; no official FP8/FP4 arithmetic equivalence,trainedtarget quality,naturalacceptance or throughput claim.");
          else ::testing::Test::RecordProperty("scope","68 original roles: previous56 formalBF16 fullVvocabulary plus originalQ8DSpark roles,now12 originalQ8 attention projections including natural8group output-A. Same complete3stage encodedstate/allVhead journey. Remaining15 sidecarroles(router3,bias3,MXFP4experts9) and actualtrainedtargetcapture/verifier unsupplied/generated. CPUAQ reference fixedactualAVXworker,CUDA MMVQ strict/fastdivisioninterval conditional; no officialFP8 arithmetic equivalence,fulltrainedtarget,naturalacceptance or throughput claim.");
        } else ::testing::Test::RecordProperty("scope","56 original roles: previous47 formal BF16 full-V vocabulary plus original Q8 DSpark fc/mainnorm,3wkv/kvnorm,Markov/confidence,13norm/sink,21HC,now9original shared-FFN Q8_0 banks. Complete three-stage encoded state and all-V head; remaining attention projections,routed experts/router and target-capture/verifier boundary generated. Selected Q8 arithmetic differs from official FP8; CPU reference tied to actual fixed AVX worker, CUDA MMVQ strict/fast division interval accuracy conditional. Same strict all-V greedy/embedding/Markov/LM/confidence and prefix0/1/2/wrap/continuation/lifetime/collection gates. Not full trained target/stages,natural acceptance or throughput.");
      } else ::testing::Test::RecordProperty("scope","47 original roles: formal Flash0731 BF16 full-V shared embedding/head plus Q8 DSpark original fc/mainnorm, 3 wkv/kvnorm, Markov/confidence, 13 norm/sink and21 HC parameters. Same complete three-stage encoded state and all-V head journey; remaining attention/MoE coefficients and target-capture/verifier boundary generated. All-V greedy, exact embedding, independent full-V LM and actual4352-term confidence bounds, old Markov cancellation/IDs gates; prefix0/1/2, wrap, continuation, retained results and collection. Not main UD-Q4 vocabulary, full original target/stages, natural acceptance or throughput.");
    } else ::testing::Test::RecordProperty("scope","original shared BF16 full-V embedding/head plus original Q8 DSpark eleven roles, complete three-stage generated remainder and generated target-capture/verifier boundary; all-V greedy and exact actual embedding rows, first-episode independent full-V F64 LM/error propagation, unchanged Markov cancellation/IDs gates, independent actual-input4352-term confidence error bound; prefix0/1/2, wrap, own-head continuation, old results and collection. Not main UD-Q4 vocabulary, full official BF16 target/stages, natural acceptance or throughput.");
  } else ::testing::Test::RecordProperty("scope","full three-stage draft HC/attention/MoE plus natural original Markov/confidence; original fc/norm and three original wkv/norm share readonly roles with accepted encoded history; prefix0/1/2, wrap, rejection, own-head continuation and retained results. Generated shared LM/embedding/other stage parameters and generated target-capture/verifier boundary, not trained target/full model or measured natural acceptance.");
  if(vocabulary&&vocabulary->packed) {
    ::testing::Test::RecordProperty("vocabulary_original_bytes",std::to_string(packed_vocabulary::kBytes));
    ::testing::Test::RecordProperty("vocabulary_artifact_repository",std::string(packed_vocabulary::kRepository));
    ::testing::Test::RecordProperty("vocabulary_artifact_revision",std::string(packed_vocabulary::kRevision));
    ::testing::Test::RecordProperty("vocabulary_codec","GGUF_Q8_0 independent embedding/head, Float32 codec carrier, original34B/32blocks");
    ::testing::Test::RecordProperty("vocabulary_activation_reference",cuda?
        "fixed Ada Q8_0 M5/M6 MMVQ Q8_1/32 strict/fast division envelope; source-derived code/halfscale ambiguity and F32 bounds":
        "actual fixed x86 worker AVX Q8_0/32 nearest-even activation; original packed-code/F32 accumulation bound");
    ::testing::Test::RecordProperty("scope","Matching Flash0731 mainUD-Q4_K_XL Q8_0 embedding/head and complete81-role Q8_0/MXFP4 DSpark sidecar, actual formal40/41/42 target-tail captures and distinctfinalHC/norm, one6row originalQ8 fullV verifier and same30memberStateHead K5/pending continuation. PackedGather F32 decode then explicitBF16 model entry; head uses actualBF16normalized exactwiden plus native activation quantization. All originalHC/attention/shared/router/selectedexperts/Markov/confidence/state gates retained. Layer39 and initialcacheprefix finite fixtures; this does not certify the whole mainUD-Q4 target body, officialBF16/E4 equivalence, quality, naturalacceptance or TPS.");
  }
  if constexpr(use_bfloat) {
    EXPECT_TRUE(checked_original_shared);EXPECT_TRUE(checked_original_attention);EXPECT_TRUE(checked_original_experts);
    ::testing::Test::RecordProperty("original_role_count",83);
    ::testing::Test::RecordProperty("original_role_bytes","12434832284");
    ::testing::Test::RecordProperty("draft_static_roles",72);
    ::testing::Test::RecordProperty("draft_static_source_bytes",std::to_string(bfloat_sidecar::kStaticBytes));
    ::testing::Test::RecordProperty("draft_expert_source_roles",9);
    ::testing::Test::RecordProperty("actual_expert_attachments",int(bfloat_frontier.attachments));
    ::testing::Test::RecordProperty("actual_missing_frontiers",int(bfloat_frontier.requests));
    ::testing::Test::RecordProperty("actual_new_expert_attachment_bytes",std::to_string(bfloat_frontier.attachments*13369344ULL));
    std::size_t retained_experts=0;
    for(std::size_t stage=0;stage<3;++stage)for(std::int32_t global=0;global<256;++global)
      retained_experts+=data.bfloat_source->has_expert(stage,global);
    ::testing::Test::RecordProperty("retained_expert_source_bytes",std::to_string(retained_experts*13369344ULL));
    ::testing::Test::RecordProperty("draft_activation_policy","explicit BF16 activation/storage; existing DenseLinearF32 raw F32 result, original BF16 OA ordinary BMM result; actual BF16 predecessors before native MXFP4 activation AQ; no peer F32-A path equivalence claim");
    ::testing::Test::RecordProperty("proposal_numerics","actual raw BF16-wkv/F32acc and original F32 learned norm at each episode; derived BF16-dot bound, exact BF16 stores/RoPE/row-cache bytes and reader; no Q8 reference certifies BF16 weights");
    ::testing::Test::RecordProperty("scope","Original S03 independent81-role BF16/F32/MXFP4 DSpark draft, matching S02 mainGGUF Q8 embedding/head, formal original Flash target40/41/42/finalHC/norm; same K5/full6row verifier/atomic30-member StateHead and real pending continuation. All72static original bytes plus actual selected expert slices are authenticated;9full expert-bank capacities are metadata, not all expert execution/acquisition. BF16-activation projection policy is explicit, no full-F32 weight shadow or equivalence to all peer GGUF dispatches. Layer39 and initial cacheprefix finite fixtures; no complete main model,quality,acceptance-rate or TPS claim.");
  }

}
class DSparkOriginalState : public ::testing::Test {
protected:
  static inline std::unique_ptr<const Data> data;
  static inline std::unique_ptr<const SharedVocabularyData> vocabulary;
  static inline std::unique_ptr<const SharedVocabularyData> gguf_vocabulary;
  static inline std::shared_ptr<const hc::Data> parameter_data;
  static inline std::shared_ptr<const shared_ffn::Data> shared_data;
  static inline std::shared_ptr<const attention::Data> attention_data;
  static inline std::shared_ptr<const expert::Data> expert_data;
  static inline std::shared_ptr<const target_roles::Data> target_data;
  static bool require_target_tail() {
    if(target_data)return true;
    const auto directory=environment("HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR");
    if(directory.empty())return false;
    target_data=std::make_shared<const target_roles::Data>(target_roles::Data::load(directory));return true;
  }
  static bool require_experts() {
    if(expert_data)return true;
    const auto directory=environment("HYLE_DSPARK_EXPERT_ROLES_DIR");
    if(directory.empty())return false;
    expert_data=std::make_shared<const expert::Data>(expert::Data::load(directory));return true;
  }
  static bool require_attention() {
    if(attention_data)return true;
    const auto directory=environment("HYLE_DSPARK_ATTENTION_ROLES_DIR");
    if(directory.empty())return false;
    attention_data=std::make_shared<const attention::Data>(attention::Data::load(directory));return true;
  }
  static bool require_shared_ffn() {
    if(shared_data)return true;
    const auto directory=environment("HYLE_DSPARK_SHARED_FFN_DIR");
    if(directory.empty())return false;
    shared_data=std::make_shared<const shared_ffn::Data>(shared_ffn::Data::load(directory));return true;
  }
  static bool require_parameters() {
    if(parameter_data)return true;
    const auto directory=environment("HYLE_DSPARK_PARAMETER_ROLES_DIR");
    if(directory.empty())return false;
    parameter_data=std::make_shared<const hc::Data>(hc::Data::load(directory));return true;
  }
  static bool require_vocabulary() {
    if(vocabulary) return true;
    const auto directory=environment("HYLE_DSPARK_SHARED_VOCAB_DIR");
    if(directory.empty()) return false;
    vocabulary=std::make_unique<const SharedVocabularyData>(SharedVocabularyData::load(directory));
    return true;
  }
  static bool require_gguf_vocabulary() {
    if(gguf_vocabulary)return true;
    const auto directory=environment("HYLE_DSPARK_GGUF_VOCAB_DIR");if(directory.empty())return false;
    gguf_vocabulary=std::make_unique<const SharedVocabularyData>(SharedVocabularyData::load_packed(directory));return true;
  }
  static void SetUpTestSuite() {
    const auto base=environment("HYLE_DSPARK_TARGET_ROLES_DIR"),extra=environment("HYLE_DSPARK_HISTORY_ROLES_DIR"),
      markov=environment("HYLE_DSPARK_MARKOV_DIR"),confidence=environment("HYLE_DSPARK_CONFIDENCE_BF16");
    if(!base.empty()&&!extra.empty()&&!markov.empty()&&!confidence.empty())
      data=std::make_unique<const Data>(Data::load(base,extra,markov,confidence));
  }
  static void TearDownTestSuite(){target_data.reset();expert_data.reset();attention_data.reset();shared_data.reset();parameter_data.reset();gguf_vocabulary.reset();vocabulary.reset();data.reset();}
  void SetUp() override {if(!data) GTEST_SKIP()<<"Set authenticated DSpark target/history/Markov/confidence role paths";}
};
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
struct System final {m::MemoryResourceRef ref() noexcept{return m::system_resource();}};
TEST_F(DSparkOriginalState,CpuEncodedMainHistoryFeedsFullDraftAndAtomicAcceptedState) {
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");exercise(*data,context,owner,false);
}
TEST_F(DSparkOriginalState,CpuOriginalSharedVocabularyFeedsFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()) GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_SHARED_VOCAB_DIR";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,vocabulary.get());
}
TEST_F(DSparkOriginalState,CpuOriginalHyperConnectionsFeedFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()||!require_parameters())GTEST_SKIP()<<"Set authenticated shared-vocabulary and HYLE_DSPARK_PARAMETER_ROLES_DIR inputs";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,vocabulary.get(),parameter_data);
}
TEST_F(DSparkOriginalState,CpuOriginalSharedFfnFeedsFullDraftAndAtomicAcceptedState) {
#if !defined(_M_X64) && !defined(__x86_64__)
  GTEST_SKIP()<<"The original shared-Q8 activation oracle currently covers the fixed x86 AVX/scalar worker source";
#endif
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn())GTEST_SKIP()<<"Set authenticated shared vocabulary, HC and HYLE_DSPARK_SHARED_FFN_DIR inputs";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,vocabulary.get(),parameter_data,shared_data);
}
TEST_F(DSparkOriginalState,CpuOriginalAttentionFeedsFullDraftAndAtomicAcceptedState) {
#if !defined(_M_X64) && !defined(__x86_64__)
  GTEST_SKIP()<<"The original attention Q8 activation oracle covers the fixed x86 AVX/scalar worker source";
#endif
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention())
    GTEST_SKIP()<<"Set authenticated vocabulary, HC, shared FFN and HYLE_DSPARK_ATTENTION_ROLES_DIR inputs";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,vocabulary.get(),parameter_data,shared_data,attention_data);
}
TEST_F(DSparkOriginalState,CpuOriginalExpertsFeedFullDraftAndAtomicAcceptedState) {
#if !defined(_M_X64) && !defined(__x86_64__)
  GTEST_SKIP()<<"The original expert activation oracle covers the fixed x86 AVX worker source";
#endif
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts())
    GTEST_SKIP()<<"Set authenticated original vocabulary/HC/shared/attention and HYLE_DSPARK_EXPERT_ROLES_DIR inputs";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,vocabulary.get(),parameter_data,shared_data,attention_data,expert_data);
}
TEST_F(DSparkOriginalState,CpuOriginalTargetTailAndFullVocabularyVerifierPublishesBlock5Prefix) {
#if !defined(_M_X64) && !defined(__x86_64__)
  GTEST_SKIP()<<"The original selected draft AQ oracle covers the fixed x86 AVX worker source";
#endif
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts()||!require_target_tail())
    GTEST_SKIP()<<"Set authenticated complete draft roles and HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,vocabulary.get(),parameter_data,shared_data,attention_data,expert_data
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      ,nullptr
#endif
      ,target_data);
}
TEST_F(DSparkOriginalState,CpuMainGgufQ8VocabularyAndOriginalTargetTailPublishesBlock5Prefix) {
#if !defined(_M_X64) && !defined(__x86_64__)
  GTEST_SKIP()<<"The original packed activation oracle covers the fixed x86 AVX worker source";
#endif
  if(!require_gguf_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts()||!require_target_tail())
    GTEST_SKIP()<<"Set authenticated complete draft/target roles and HYLE_DSPARK_GGUF_VOCAB_DIR";
  auto context=take(g::cpu::create_context(),"ordinary CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"CPU allocation owner");
  exercise(*data,context,owner,false,gguf_vocabulary.get(),parameter_data,shared_data,attention_data,expert_data
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      ,nullptr
#endif
      ,target_data);
}
TEST_F(DSparkOriginalState,CpuRandomQ8MarkovPrefixAndPendingContinue) {
  if(!require_gguf_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts()||!require_target_tail())
    GTEST_SKIP()<<"Set authenticated complete original Q8 draft/target and vocabulary roles";
  auto context=take(g::cpu::create_context(),"original Q8 sampled CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"original Q8 sampled allocation owner");
  exercise(*data,context,owner,false,gguf_vocabulary.get(),parameter_data,shared_data,attention_data,expert_data
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      ,nullptr
#endif
      ,target_data,true);
}
#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
// Own the actual typed allocation domain once. Both the compiled ordinary IO
// and WeightSlots borrow this same resource; this adapter only retains it.
struct SharedCudaDevice final {
  std::shared_ptr<c::cuda::MemoryResource> resource;
  m::MemoryResourceRef ref() noexcept {return resource->ref();}
};
TEST_F(DSparkOriginalState,CudaEncodedMainHistoryFeedsFullDraftAndAtomicAcceptedState) {
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"CUDA memory")),"CUDA allocation owner");exercise(*data,context,owner,true);
}
TEST_F(DSparkOriginalState,CudaOriginalSharedVocabularyFeedsFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()) GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_SHARED_VOCAB_DIR";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"CUDA memory")),"CUDA allocation owner");
  exercise(*data,context,owner,true,vocabulary.get());
}
TEST_F(DSparkOriginalState,CudaOriginalHyperConnectionsFeedFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()||!require_parameters())GTEST_SKIP()<<"Set authenticated shared-vocabulary and HYLE_DSPARK_PARAMETER_ROLES_DIR inputs";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"CUDA memory")),"CUDA allocation owner");
  exercise(*data,context,owner,true,vocabulary.get(),parameter_data);
}
TEST_F(DSparkOriginalState,CudaOriginalSharedFfnFeedsFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn())GTEST_SKIP()<<"Set authenticated shared vocabulary, HC and HYLE_DSPARK_SHARED_FFN_DIR inputs";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"CUDA memory")),"CUDA allocation owner");
  exercise(*data,context,owner,true,vocabulary.get(),parameter_data,shared_data);
}
TEST_F(DSparkOriginalState,CudaOriginalAttentionFeedsFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention())
    GTEST_SKIP()<<"Set authenticated vocabulary, HC, shared FFN and HYLE_DSPARK_ATTENTION_ROLES_DIR inputs";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"CUDA memory")),"CUDA allocation owner");
  exercise(*data,context,owner,true,vocabulary.get(),parameter_data,shared_data,attention_data);
}
TEST_F(DSparkOriginalState,CudaOriginalExpertsFeedFullDraftAndAtomicAcceptedState) {
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts())
    GTEST_SKIP()<<"Set authenticated original vocabulary/HC/shared/attention and HYLE_DSPARK_EXPERT_ROLES_DIR inputs";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"shared actual CUDA device allocation domain"));
  auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared CUDA allocation owner");
  exercise(*data,context,owner,true,vocabulary.get(),parameter_data,shared_data,attention_data,expert_data,device.get());
}
TEST_F(DSparkOriginalState,CudaOriginalTargetTailAndFullVocabularyVerifierPublishesBlock5Prefix) {
  if(!require_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts()||!require_target_tail())
    GTEST_SKIP()<<"Set authenticated complete draft roles and HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"shared actual CUDA target/draft IO device"));
  auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared CUDA allocation owner");
  exercise(*data,context,owner,true,vocabulary.get(),parameter_data,shared_data,attention_data,expert_data,device.get(),target_data);
}
TEST_F(DSparkOriginalState,CudaMainGgufQ8VocabularyAndOriginalTargetTailPublishesBlock5Prefix) {
  if(!require_gguf_vocabulary()||!require_parameters()||!require_shared_ffn()||!require_attention()||!require_experts()||!require_target_tail())
    GTEST_SKIP()<<"Set authenticated complete draft/target roles and HYLE_DSPARK_GGUF_VOCAB_DIR";
  auto context=take(g::cuda::create_context(),"ordinary CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"same actual CUDA target/draft IO resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared CUDA allocation owner");
  exercise(*data,context,owner,true,gguf_vocabulary.get(),parameter_data,shared_data,attention_data,expert_data,device.get(),target_data);
}
#endif
class DSparkBfloatSidecarState : public ::testing::Test {
 protected:
  static inline std::shared_ptr<bfloat_sidecar::Data> source;
  static inline std::unique_ptr<const Data> data;
  static inline std::unique_ptr<const SharedVocabularyData> vocabulary;
  static inline std::shared_ptr<const target_roles::Data> target_data;
  static void SetUpTestSuite() {
    const auto side=environment("HYLE_DSPARK_BFLOAT_SIDECAR_DIR"),
        vocab=environment("HYLE_DSPARK_GGUF_VOCAB_DIR"),tail=environment("HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR");
    if(side.empty()||vocab.empty()||tail.empty())return;
    source=std::make_shared<bfloat_sidecar::Data>(bfloat_sidecar::Data::load(side,bfloat_catalogue_pin()));
    data=std::make_unique<const Data>(Data::from_bfloat(source));
    vocabulary=std::make_unique<const SharedVocabularyData>(SharedVocabularyData::load_packed(vocab));
    target_data=std::make_shared<const target_roles::Data>(target_roles::Data::load(tail));
  }
  static void TearDownTestSuite(){target_data.reset();vocabulary.reset();data.reset();source.reset();}
  void SetUp() override {
    if(!data)GTEST_SKIP()<<"Set authenticated HYLE_DSPARK_BFLOAT_SIDECAR_DIR, HYLE_DSPARK_GGUF_VOCAB_DIR and HYLE_DSPARK_ORIGINAL_TARGET_TAIL_DIR";
  }
};
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
TEST_F(DSparkBfloatSidecarState,CpuOriginalBfloatSidecarAndTargetTailPublishesBlock5Prefix) {
#if !defined(_M_X64) && !defined(__x86_64__)
  GTEST_SKIP()<<"The original selected MXFP4 activation certificate covers the fixed x86 AVX worker source";
#endif
  auto context=take(g::cpu::create_context(),"ordinary S03 CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"S03 CPU allocation owner");
  exercise<bfloat_sidecar::Data>(*data,context,owner,false,vocabulary.get(),{},{},{},source
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      ,nullptr
#endif
      ,target_data);
}
TEST_F(DSparkBfloatSidecarState,CpuRandomBfloatMarkovPrefixAndPendingContinue) {
  auto context=take(g::cpu::create_context(),"S03 sampled CPU Context");
  auto owner=take(c::StorageOwner::adopt(context,System{}),"S03 sampled original allocation owner");
  exercise<bfloat_sidecar::Data>(*data,context,owner,false,vocabulary.get(),{},{},{},source
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      ,nullptr
#endif
      ,target_data,true);
}

#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
TEST_F(DSparkBfloatSidecarState,CudaOriginalBfloatSidecarAndTargetTailPublishesBlock5Prefix) {
  auto context=take(g::cuda::create_context(),"ordinary S03 CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"S03 same real CUDA IO/expert resource"));
  auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"S03 shared CUDA allocation owner");
  exercise<bfloat_sidecar::Data>(*data,context,owner,true,vocabulary.get(),{},{},{},source,device.get(),target_data);
}
#endif
} // namespace
