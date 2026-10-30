#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/state_versions.h"
#if defined(HYLE_TEST_GLM_STATE_CPU)
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_TEST_GLM_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#include "glm_pro_mtp_fragment.h"
#include "glm_pro_original_roles.h"
#include "glm_pro_mtp_reference.h"
#include "glm_pro_target_roles.h"
#include "glm_pro_target_fragment.h"
#include "glm_pro_target_reference.h"
#include "glm_original_moe_roles.h"
#include "glm_original_moe_reference.h"
#include "glm_original_moe_resident_weights.h"
#include "glm_shared_vocabulary.h"
#include "glm_target_verifier_fragment.h"
#include "../../engine/inference/mtp_acceptance.h"
#include "glm_stochastic_prefix_fragment.h"
#include "../../engine/inference/sampling_cursor.h"
#include "../../engine/inference/prefix_directory.h"
#include "../../engine/inference/request_window.h"
#include "../../engine/inference/request_progress.h"
#include "../../engine/inference/request_loop.h"
#include "../../engine/inference/request_wave_execution.h"
#include "../../engine/inference/step_execution.h"
#include "../../engine/inference/commit_observation.h"
#include "model_graph_run.h"
#include "model_resident_fragment.h"
#include "glm_gguf_mtp_fragment.h"
#include "glm_gguf_runtime_source.h"
#include "glm_gguf_expert_frontier.h"
#include "glm_gguf_mtp_reference.h"
#if __has_include("glm_gguf_b09_artifacts.h")
#include "glm_gguf_b09_artifacts.h"
#define HYLE_TEST_GLM_B09_ACTUAL_PINS 1
#endif
#if __has_include("glm_gguf_b10_artifacts.h")
#include "glm_gguf_b10_artifacts.h"
#define HYLE_TEST_GLM_B10_ACTUAL_PINS 1
#endif

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace model=glm_pro_mtp_fragment;
namespace own=glm_pro_original_roles;
namespace target_own=glm_pro_target_roles;
namespace target_model=glm_pro_target_fragment;
namespace vocabulary=glm_shared_vocabulary;
namespace expert=glm_original_moe_roles;
namespace supply=glm_original_moe_resident_weights;
namespace cut=model_resident_fragment;
using Value=sem::ValueId;
using Bytes=std::vector<std::byte>;
using Readonly=vocabulary::ReadonlyRole;
inline constexpr auto kMoe=glm_moe_geometry::kPro;
inline constexpr std::int64_t kCapacity=2064,kBasePosition=2053,kSelected=2048,kJournalRows=3;

template<class T> T take(hyle::Result<T> result,const char *where) {
  if(!result)throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status,const char *where) {
  if(!status.ok())throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
Bytes read(const c::ReadLease &lease) {
  Bytes bytes(lease.size());check(take(lease.view(),"read actual Pro lease").read(bytes),"read Pro bytes");return bytes;
}
template<class T> std::vector<T> values(const c::ReadLease &lease) {
  const auto raw=read(lease);
  if(raw.size()%sizeof(T))throw std::runtime_error("Pro typed lease byte extent");
  std::vector<T> result(raw.size()/sizeof(T));std::memcpy(result.data(),raw.data(),raw.size());return result;
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> bytes) {
  auto buffer=take(m::Buffer::create(owner.ref(),bytes.size(),256),"initial/immutable Pro backing");
  check(buffer.view().write(bytes),"initial/immutable Pro upload");
  return take(owner.adopt_buffer(std::move(buffer)),"initial/immutable Pro permission");
}
std::string directory(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t count=0;
  if(_dupenv_s(&value,&count,name))throw std::runtime_error("original Pro directory environment");
  const std::unique_ptr<char,decltype(&std::free)> cleanup(value,&std::free);return value?value:"";
#else
  const auto *value=std::getenv(name);return value?value:"";
#endif
}

struct External final {std::size_t slot;Readonly role;};
struct Recipe final {
  sem::Fragment fragment;
  std::vector<External> external;
};
Value external(Recipe &recipe,const Readonly &role) {
  const auto slot=recipe.fragment.constants.size();
  const auto value=recipe.fragment.constant(role.description,{});
  recipe.external.push_back({slot,role});return value;
}

// This holder compiles ordinary public Graphs. The shared model_graph_run
// function owns their bind/prepare/submit/retire/reset procedure; model state,
// original roles and expert frontiers are not part of an alternate executor.
struct Native final {
  Recipe recipe;
  sem::GraphCase logical;
  std::vector<c::ReadLease> parameters;
  std::unique_ptr<model_graph_run::ColdProgram> program;
  g::Invocation invocation;
  c::LeasePool outputs;
  std::optional<model_graph_run::InputPlan> resident_inputs;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  std::array<g::WeightBinding,4> resident_bindings{};
#endif
  std::size_t submissions=0;
  static Native create(Recipe recipe,c::Context &context,const c::StorageOwner &owner,
                       bool cuda,std::uint32_t generations=3) {
    Native result;result.recipe=std::move(recipe);result.logical=sem::make_graph(result.recipe.fragment);
    std::vector<g::ConstantBinding> constants;
    for(std::size_t slot=0;slot<result.logical.constants.size();++slot) {
      const auto found=std::find_if(result.recipe.external.begin(),result.recipe.external.end(),
          [&](const External &entry){return entry.slot==slot;});
      const auto &desc=result.recipe.fragment.description(result.recipe.fragment.constants[slot]);
      if(found!=result.recipe.external.end()) {
        if(found->role.description!=desc||found->role.owner.ref()!=owner.ref())
          throw std::runtime_error("Pro immutable constant changed description or actual owner");
        constants.push_back(take(found->role.binding(result.logical.constants[slot]),"actual ready Pro constant owner"));
      } else {
        result.parameters.push_back(upload(owner,result.recipe.fragment.constant_bytes[slot]));
        constants.push_back({result.logical.constants[slot],take(result.parameters.back().tensor(desc),"actual generated Pro constant")});
      }
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*result.recipe.fragment.profile),"actual Pro artifact profile"),
        c::WorkloadPhase::kDecode,{0x48594c454d4f444cULL,result.recipe.fragment.region,1}};
    const g::CompileOptions options{.constants=constants,.workload=workload};
    const auto compile=[&]()->hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_GLM_STATE_CPU)
      if(!cuda)return g::cpu::compile(result.logical.graph,context,options);
#endif
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      if(cuda)return g::cuda::compile(result.logical.graph,context,owner,options);
#endif
      throw std::runtime_error("original Pro native backend is unavailable");
    };
    auto compiled=take(compile(),"ordinary original Pro Graph compile");
    // AQ references qualify the actual packed backend, not a model name.
    // Inspect once at cold compilation, never route providers during replay.
    for(std::size_t step=0;step<compiled.report.step_count();++step) {
      const auto record=take(compiled.report.step(step),"actual packed projection certificate");
      for(std::size_t i=0;i<record.node_count;++i) {
        const auto node=take(compiled.report.step_node(step,i),"actual packed projection node");
        const auto op=take(result.logical.graph.operation(node),"actual packed projection operation");
        std::optional<std::size_t> weight;
        if(op.get_if<c::MatMul>())weight=1;
        else if(op.custom_as<c::model_ops::GroupedMatMul>())weight=0;
        if(!weight)continue;
        const auto value=take(result.logical.graph.node_input(node,*weight),"actual packed original weight");
        const auto desc=take(result.logical.graph.value_description(value),"actual packed weight descriptor");
        if(c::gguf_quantized_type(desc.storage_format())&&record.provider.name!=(cuda?"ggml-cuda":"ggml-cpu"))
          throw std::runtime_error("actual packed projection provider does not match its AQ certificate");
      }
    }
    std::vector<model_graph_run::Port> input_ports,output_ports;
    for(std::size_t slot=0;slot<result.logical.inputs.size();++slot)
      input_ports.push_back({result.logical.inputs[slot],result.recipe.fragment.description(result.recipe.fragment.inputs[slot])});
    std::size_t capacity=0;
    for(std::size_t slot=0;slot<result.logical.outputs.size();++slot) {
      const auto &desc=result.recipe.fragment.description(result.recipe.fragment.outputs[slot]);
      const auto bytes=take(desc.required_bytes(),"Pro output extent");
      if(bytes>std::numeric_limits<std::size_t>::max()-255||capacity>std::numeric_limits<std::size_t>::max()-(bytes+255)/256*256)
        throw std::runtime_error("Pro output backing capacity overflow");
      capacity+=(bytes+255)/256*256;
      output_ports.push_back({result.logical.outputs[slot],desc});
    }
    if(!generations||capacity>std::numeric_limits<std::size_t>::max()/generations||
        output_ports.size()>std::numeric_limits<std::uint32_t>::max()/generations)
      throw std::runtime_error("Pro result generations exceed representable capacity");
    result.program=model_graph_run::ColdProgram::create(std::move(compiled),input_ports,output_ports);
    result.invocation=take(result.program->compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"ordinary Pro Invocation");
    result.outputs=take(c::LeasePool::create(owner.ref(),capacity*generations,
        {.max_blocks=std::uint32_t(result.program->output_count())*generations,.backing_alignment=256}),"bounded Pro result generations");
    return result;
  }
  template<model_graph_run::LeaseInputRange Inputs=std::span<const c::ReadLease>>
  std::vector<c::ReadLease> execute(c::Queue &queue,const Inputs &inputs,
      const model_graph_run::InputPlan *input_plan=nullptr,const std::function<void(g::Invocation&)> &binder={}) {
    return model_graph_run::execute(*program,invocation,outputs,queue,inputs,submissions,input_plan,binder);
  }
};

// Parameter identity remains model-local. Only the actual bounded expert
// generation, router selection and submission counters are shared by the
// existing ordinary two-phase runner.
struct ResidentSource {
  static constexpr bool kPacked=false;
  using CpuBatch=supply::CpuBatch;
  expert::Roles experts;
  std::unique_ptr<supply::CpuSupply> cpu_supply;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  std::unique_ptr<supply::CudaSupply> cuda_supply;
#endif
  bool moe_full_checked=false;
  std::size_t episodes=0;
  std::vector<expert::Selection> selections;
  struct UploadObservation final {
    bool native=false,copied_bytes_unknown=false;
    std::size_t expected=0,logical=0,staged=0,completed=0,submitted=0;
  };
  UploadObservation last_upload;
};
struct Source final:ResidentSource {
  own::Bound parameters;
  vocabulary::Vocabulary words;
  std::array<Readonly,2> prepared;
  c::LeasePool prepared_pool;
  bool full_checked=false;
  std::size_t cold_submissions=0,checked_steps=0;
};
struct TargetSource final:ResidentSource {
  target_own::Bound parameters;
  std::array<Readonly,2> prepared;
  c::LeasePool prepared_pool;
  bool full_checked=false;
  std::size_t cold_submissions=0,checked_steps=0;
};
void observe_resident_upload(const ResidentSource::UploadObservation &actual) {
  if(!actual.native)return;
  EXPECT_EQ(actual.logical,actual.expected);EXPECT_EQ(actual.staged,actual.expected);
  EXPECT_EQ(actual.completed,actual.submitted);EXPECT_FALSE(actual.copied_bytes_unknown);
}

// Scenario observations use the exact resident generation of the completed
// execution, even when a later repair has already supplied another generation.
template<class Owner,class ReadValue>
bool observe_moe(Owner &source,const glm_original_moe_fragment::Result &result,
                 ReadValue actual,std::size_t selection_index) {
  namespace ref=glm_original_moe_reference;
  namespace boundary=glm_original_indexer_reference;
  namespace lm=glm_shared_vocabulary_reference;
  namespace bytes=glm_pro_mtp_reference::detail;
  constexpr auto geometry=glm_moe_geometry::kPro;
  if(!source.experts.source||source.experts.source->model()!=geometry||source.selections.empty()) {
    ADD_FAILURE()<<"original Pro MoE observer requires the actual Pro source and resident selection";
    return false;
  }
  const auto &data=*source.experts.source;
  const auto &ports=result.ports;
  const auto fp=[&](Value value){return bytes::floats(actual(value));};
  const auto bf=[&](Value value){return bytes::bfloats(actual(value));};
  const auto integer=[&](Value value){return bytes::integers(actual(value));};
  const auto input=bf(ports.input);
  const auto global=integer(ports.ids),local=integer(ports.local_ids),map=integer(ports.map);
  const auto logits=fp(ports.logits),scores=fp(ports.scores),selection=fp(ports.selection);
  const auto selected=fp(ports.selected_scores),sum=fp(ports.sum),denominator=fp(ports.denominator);
  const auto normalized=fp(ports.normalized),weights=fp(ports.weights);
  if(!ref::expect_router(logits,input,data.router_bytes(),geometry)||
      !ref::expect_routing({logits,scores,selection,selected,sum,denominator,normalized,weights,global},
          data.bias_bytes(),geometry))return false;
  const auto &resident=source.selections.at(selection_index);
  if(map!=resident.map) {
    ADD_FAILURE()<<"original Pro MoE actual map differs from the same resident generation";
    return false;
  }
  if(!ref::expect_mapping(global,map,local,resident.globals,geometry))return false;
  const bool full=!source.moe_full_checked;
  const auto matrix=[&](expert::Leg leg,std::optional<std::int32_t> selected_expert=std::nullopt) {
    const auto [outputs,width]=expert::geometry(leg,geometry);
    return ref::Fp8Matrix{
        selected_expert?data.expert_bytes(leg,*selected_expert,false):data.shared_bytes(leg,false),
        selected_expert?data.expert_bytes(leg,*selected_expert,true):data.shared_bytes(leg,true),
        std::size_t(outputs),std::size_t(width)};
  };
  const auto branch=[&](const glm_original_moe_fragment::ProjectionPorts &projection,bool routed) {
    if(!projection.gate_raw||!projection.up_raw||!projection.down_raw||projection.down_stored)
      throw std::runtime_error("original Pro E4 observer requires real F32 projection outputs");
    const auto gate=fp(*projection.gate_raw),up=fp(*projection.up_raw),down=fp(*projection.down_raw);
    const auto stored_gate=bf(projection.gate_stored),stored_up=bf(projection.up_stored);
    const auto activated=bf(projection.activation_stored);
    if(routed) {
      if(!ref::expect_selected_projection(gate,input,global,
              [&](std::int32_t id){return matrix(expert::Leg::kGate,id);},
              "original Pro selected gate",full,geometry)||
          !ref::expect_selected_projection(up,input,global,
              [&](std::int32_t id){return matrix(expert::Leg::kUp,id);},
              "original Pro selected up",full,geometry)||
          !ref::expect_selected_projection(down,activated,global,
              [&](std::int32_t id){return matrix(expert::Leg::kDown,id);},
              "original Pro selected down",full,geometry))return false;
    } else {
      if(!ref::expect_fp8_projection(gate,input,matrix(expert::Leg::kGate),"original Pro shared gate",full)||
          !ref::expect_fp8_projection(up,input,matrix(expert::Leg::kUp),"original Pro shared up",full)||
          !ref::expect_fp8_projection(down,activated,matrix(expert::Leg::kDown),"original Pro shared down",full))return false;
    }
    const auto activation=fp(projection.activation_raw);
    // kPro has no SwiGLU limit: these ports must retain the un-clipped actual
    // BF16 predecessors. The same helper explicitly checks that profile.
    return boundary::expect_bfloat_store(stored_gate,gate,"Pro actual gate BF16 store")&&
        boundary::expect_bfloat_store(stored_up,up,"Pro actual up BF16 store")&&
        ref::expect_activation(fp(projection.gate_clipped),fp(projection.up_clipped),
            activation,stored_gate,stored_up,geometry)&&
        boundary::expect_bfloat_store(activated,activation,"Pro actual SwiGLU BF16 store");
  };
  if(!branch(ports.routed,true)||!branch(ports.shared,false))return false;
  const auto weighted=fp(ports.weighted_raw);
  const auto weighted_stored=bf(ports.weighted_stored);
  const auto routed_sum=fp(ports.routed_sum_raw);
  const auto routed_stored=bf(ports.routed_sum_stored);
  const auto shared_stored=bf(ports.shared_stored);
  const auto merged=fp(ports.merge_raw);
  const auto merged_stored=bf(ports.merge_stored);
  if(!ref::expect_weighted_down(weighted,fp(ports.routed.down_raw.value()),weights,geometry)||
      !boundary::expect_bfloat_store(weighted_stored,weighted,"Pro actual weighted down BF16 store")||
      !ref::expect_routed_sum(routed_sum,weighted_stored,geometry)||
      !boundary::expect_bfloat_store(routed_stored,routed_sum,"Pro actual routed sum BF16 store")||
      !boundary::expect_bfloat_store(shared_stored,fp(ports.shared.down_raw.value()),"Pro actual shared down BF16 store")||
      !lm::expect_sum(merged,bytes::widen(routed_stored),bytes::widen(shared_stored))||
      !boundary::expect_bfloat_store(merged_stored,merged,"Pro actual shared/routed merge BF16 store")||
      !bytes::same(bf(result.output),merged_stored,"Pro actual MoE returned BF16 merge"))return false;
  source.moe_full_checked=true;
  return true;
}

model::Parameters parameters(Recipe &recipe,const Source &source) {
  const auto bank=[&](own::Role role){return external(recipe,source.parameters.at(role));};
  return {.eh=bank(own::Role::kEh),.embedding_norm=bank(own::Role::kEmbeddingNorm),
      .hidden_norm=bank(own::Role::kHiddenNorm),.input_norm=bank(own::Role::kInputNorm),
      .post_norm=bank(own::Role::kPostNorm),.index_bias=bank(own::Role::kIndexBias),
      .index_norm=bank(own::Role::kIndexNorm),.index_weights=bank(own::Role::kIndexWeights),
      .index_key=bank(own::Role::kIndexKey),.index_query=bank(own::Role::kIndexQuery),
      .kv_norm=bank(own::Role::kKvNorm),.kv_a=bank(own::Role::kKvA),.output=bank(own::Role::kOutput),
      .query_norm=bank(own::Role::kQueryNorm),.query_a=bank(own::Role::kQueryA),.query_b=bank(own::Role::kQueryB),
      .shared_norm=external(recipe,source.words.at(vocabulary::Role::kNorm)),
      .head=external(recipe,source.words.at(vocabulary::Role::kHead)),
      .prepared_key=external(recipe,source.prepared[0]),.prepared_value=external(recipe,source.prepared[1])};
}

// One router frontier cuts a pure model Fragment into the same two ordinary
// native Graphs used by Flash. Only its actual8 global IDs reach the bounded
// supplier. Activations and state cross the cut as original producer leases.
template<class SourceType=ResidentSource>
struct ResidentRun final {
  std::shared_ptr<SourceType> source;
  Recipe recipe;
  cut::ResidentDraft<glm_original_moe_fragment::Boundary,1> plan;
  std::vector<Native> phases;
  static ResidentRun create(Recipe recipe,const glm_original_moe_fragment::Boundary &boundary,
      std::shared_ptr<SourceType> source,c::Context &context,const c::StorageOwner &owner,bool cuda,
      std::uint32_t generations=3) {
    ResidentRun result;result.source=std::move(source);result.recipe=std::move(recipe);
    result.plan=cut::split(result.recipe.fragment,std::array{boundary},0x474c4d50524f5048ULL);
    std::vector<Readonly> ready;
    for(std::size_t slot=0;slot<result.recipe.fragment.constants.size();++slot) {
      const auto found=std::find_if(result.recipe.external.begin(),result.recipe.external.end(),
          [&](const External &entry){return entry.slot==slot;});
      if(found!=result.recipe.external.end())ready.push_back(found->role);
      else ready.push_back({"Pro cold component constant",
          result.recipe.fragment.description(result.recipe.fragment.constants[slot]),owner,
          upload(owner,result.recipe.fragment.constant_bytes[slot])});
    }
    for(const auto &phase:result.plan.phases) {
      Recipe local;local.fragment=phase.fragment;
      for(std::size_t slot=0;slot<phase.original_constants.size();++slot) {
        const auto original=phase.original_constants[slot];
        const auto found=std::find_if(result.recipe.fragment.constants.begin(),result.recipe.fragment.constants.end(),
            [&](Value value){return value.ordinal()==original.ordinal();});
        if(found==result.recipe.fragment.constants.end())throw std::runtime_error("Pro cold cut lost immutable constant");
        const auto index=std::size_t(found-result.recipe.fragment.constants.begin());
        local.external.push_back({slot,ready.at(index)});
      }
      result.phases.push_back(Native::create(std::move(local),context,owner,cuda,generations));
      if(phase.resident_stage) {
        auto &native=result.phases.back();
        const auto &frontier=result.plan.routers.at(*phase.resident_stage);
        const std::array banks{frontier.banks[0],frontier.banks[1],frontier.banks[2],frontier.map};
        std::array<std::size_t,4> dynamic{};
        for(std::size_t bank=0;bank<banks.size();++bank) {
          const auto found=std::find_if(phase.original_inputs.begin(),phase.original_inputs.end(),
              [&](Value value){return value.ordinal()==banks[bank].ordinal();});
          if(found==phase.original_inputs.end())throw std::runtime_error("Pro cut lost actual compressed bank or map");
          dynamic[bank]=std::size_t(found-phase.original_inputs.begin());
#if defined(HYLE_TEST_GLM_STATE_CUDA)
          native.resident_bindings[bank]={native.program->input(dynamic[bank]).value,supply::kBankIds[bank]};
#endif
        }
        native.resident_inputs=native.program->input_plan(dynamic);
      }
    }
    return result;
  }
  std::vector<c::ReadLease> execute(c::Queue &queue,std::span<const c::ReadLease> inputs,bool cuda) {
    (void)cuda;
    if(inputs.size()+4!=recipe.fragment.inputs.size())throw std::runtime_error("Pro ordinary/resident input count");
    std::vector<c::ReadLease> live(recipe.fragment.descriptions.size());
    for(std::size_t i=0;i<inputs.size();++i)live.at(recipe.fragment.inputs[i].ordinal())=inputs[i];
    std::array<std::vector<c::ReadLease>,2> completed;
    for(std::size_t index=0;index<phases.size();++index) {
      const auto &phase=plan.phases[index];
      std::optional<typename SourceType::CpuBatch> cpu_batch;
      std::optional<expert::Selection> selection;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      std::optional<c::cuda::WeightBatch> cuda_batch;
#endif
      const model_graph_run::InputPlan *input_plan=nullptr;
      std::function<void(g::Invocation&)> binder;
      if(phase.resident_stage) {
        const auto &frontier=plan.routers.at(*phase.resident_stage);
        selection=expert::select(values<std::int32_t>(live.at(frontier.ids.ordinal())),kMoe);
        if constexpr(SourceType::kPacked) {
          if(!source->before_upload)throw std::runtime_error("packed source requires its explicit artifact frontier");
          source->before_upload(selection->globals);
        }
        const std::array banks{frontier.banks[0],frontier.banks[1],frontier.banks[2],frontier.map};
#if defined(HYLE_TEST_GLM_STATE_CUDA)
        if constexpr(!SourceType::kPacked) {
        if(cuda) {
          if(!source->cuda_supply)throw std::runtime_error("Pro needs its actual shared CUDA supply");
          cuda_batch.emplace(take(source->cuda_supply->upload(source->episodes%2,*selection,*source->experts.source,queue),
              "same-generation original Pro experts/map"));
          input_plan=&*phases[index].resident_inputs;
          binder=[&](g::Invocation &active){check(g::cuda::bind(active,phases[index].resident_bindings,*cuda_batch),"bind actual Pro expert generation");};
        }
        }
#endif
        if(!cuda||SourceType::kPacked) {
          if(!source->cpu_supply)throw std::runtime_error("Pro needs its actual shared CPU supply");
          cpu_batch.emplace(take(source->cpu_supply->upload(*selection,*source->experts.source),"actual Pro CPU expert generation"));
          for(std::size_t bank=0;bank<banks.size();++bank)live.at(banks[bank].ordinal())=cpu_batch->banks[bank];
        }
      }
      const auto arguments=model_graph_run::project_inputs(phase.original_inputs,live);
      completed[index]=phases[index].execute(queue,arguments,input_plan,binder);
      for(std::size_t slot=0;slot<phase.original_outputs.size();++slot)
        live.at(phase.original_outputs[slot].ordinal())=completed[index][slot];
      if(selection) {
        source->selections.push_back(std::move(*selection));
        if constexpr(!SourceType::kPacked) {
          ResidentSource::UploadObservation observation;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
        if(cuda_batch) {
          const auto accounting=take(cuda_batch->report(),"actual Pro upload accounting");
          const auto layout=source->cuda_supply->layout();
          observation={true,accounting.copied_bytes_unknown,layout.logical_bytes,accounting.logical_bytes,
              accounting.staged_bytes,accounting.completed_bytes,accounting.submitted_bytes};
        }
#endif
          source->last_upload=observation;
        }
      }
      // The shared ordinary run helper has waited/reset before the compact
      // generation can be dropped and either of the two slots reused.
    }
    std::vector<c::ReadLease> outputs;
    for(const auto location:plan.full_outputs)outputs.push_back(completed.at(location.phase).at(location.port));
    ++source->episodes;return outputs;
  }
  std::size_t native_submissions() const {
    std::size_t result=0;for(const auto &phase:phases)result+=phase.submissions;return result;
  }
  std::size_t backing_capacity() const {
    std::size_t result=0;for(const auto &phase:phases)result+=phase.outputs.stats().capacity_bytes;return result;
  }
};

enum Entry:std::size_t {
  kMain,kCodes,kScales,kHidden,kPending,kCursor,kIndexCursor,kSelection,
  kJournal,kJournalPositions,kLag,kResources,kEpoch=kResources,kEntries
};
inline constexpr auto kTargetMain=static_cast<Entry>(kEntries);
inline constexpr auto kTargetCursor=static_cast<Entry>(kEntries+1);
inline constexpr auto kTargetTap=static_cast<Entry>(kEntries+2);
std::array<c::TensorDesc,3> target_descriptions() {
  return {sem::dense({kCapacity,model::kCacheDepth}),sem::dense({1},c::DataType::kInt32),
      sem::dense({1,model::kHidden},c::DataType::kBFloat16)};
}
std::array<c::TensorDesc,kResources> descriptions() {
  return {sem::dense({kCapacity,model::kCacheDepth}),
      sem::dense({kCapacity,model::kIndexDepth},c::DataType::kFloat8E4M3FN),sem::dense({kCapacity,1}),
      sem::dense({1,model::kHidden},c::DataType::kBFloat16),sem::dense({1},c::DataType::kInt32),
      sem::dense({1},c::DataType::kInt32),sem::dense({1},c::DataType::kInt32),
      sem::dense({kSelected},c::DataType::kInt32),
      sem::dense({kJournalRows,model::kHidden},c::DataType::kBFloat16),
      sem::dense({kJournalRows},c::DataType::kInt32),sem::dense({1},c::DataType::kInt32)};
}
struct StateLayout final {
  std::vector<c::TensorDesc> resources;
  std::array<std::optional<std::size_t>,kResources> slots;
  std::array<c::TensorDesc,3> target;
  std::size_t epoch()const noexcept{return resources.size();}
  std::size_t first_target()const noexcept{return epoch()+1;}
  std::size_t slot(Entry semantic)const {
    if(semantic<kResources) {
      if(!slots[semantic])throw std::runtime_error("this actual cache encoding has no requested state member");
      return *slots[semantic];
    }
    if(semantic==kEpoch)return epoch();
    if(semantic>=kTargetMain&&semantic<=kTargetTap)return first_target()+semantic-kTargetMain;
    throw std::runtime_error("unknown GLM continuation semantic member");
  }
};
StateLayout official_layout() {
  StateLayout r;const auto d=descriptions();r.resources.assign(d.begin(),d.end());
  for(std::size_t i=0;i<kResources;++i)r.slots[i]=i;r.target=target_descriptions();return r;
}
StateLayout converted_layout() {
  namespace cache=glm_gguf_f32_cache_fragment;const auto &g=glm_gguf_named_parameters::kGeometry;
  const auto actual=cache::unpooled_layout(g,kCapacity,kSelected,kJournalRows,
      cache::IndexDescription::from(sem::dense({kCapacity,g.index_depth})));
  StateLayout r;r.resources=actual.resources;
  r.slots[kMain]=actual.main;r.slots[kCodes]=actual.index_keys;r.slots[kScales]=actual.index_scales;
  r.slots[kHidden]=actual.hidden;r.slots[kPending]=actual.pending;r.slots[kCursor]=actual.cursor;
  r.slots[kIndexCursor]=actual.index_cursor;r.slots[kSelection]=actual.selection;
  r.slots[kJournal]=actual.journal;r.slots[kJournalPositions]=actual.journal_positions;r.slots[kLag]=actual.lag;
  r.target=cache::target_descriptions(g,kCapacity);return r;
}

Value increment(sem::Fragment &f,Value value,std::int32_t delta=1) {
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{delta}));
  const auto scan=f.one(c::Operation(c::Scan{.axis=0}),
      {f.one(c::Operation(c::Concat{.axis=0}),{value,one})});
  const auto last=delta==1?one:f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  return f.one(c::Operation(c::Gather{}),{scan,last});
}
constexpr auto kAbsent=std::numeric_limits<std::size_t>::max();
struct Step final {
  Recipe recipe;
  std::optional<model::Result> model;
  std::optional<model::IndexHistory> catchup;
  std::optional<glm_gguf_mtp_fragment::Result> converted_model;
  std::optional<glm_gguf_f32_cache_fragment::IndexHistory> converted_catchup;
  std::vector<std::pair<Value,std::size_t>> observations;
  std::vector<std::size_t> state_ports;
  std::size_t ordinary_inputs=0,target_hidden=kAbsent;
  std::int32_t position=0,journal_slot=0;
  bool fresh=false;
  Step():state_ports(kResources,kAbsent){}
  bool has_model()const noexcept{return model.has_value()||converted_model.has_value();}
  bool has_catchup()const noexcept{return catchup.has_value()||converted_catchup.has_value();}
};
std::size_t expose(Step &step,Value value) {
  const auto found=std::find_if(step.observations.begin(),step.observations.end(),
      [&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
  if(found!=step.observations.end())return found->second;
  auto &f=step.recipe.fragment;const auto port=f.outputs.size();
  f.outputs.push_back(f.one(c::Operation(c::Copy{}),{value}));
  step.observations.push_back({value,port});return port;
}
void observe_port(Step &step,Value &value) {
  const auto &inputs=step.recipe.fragment.inputs;
  const auto found=std::find_if(inputs.begin(),inputs.end(),[&](Value input){return input.ordinal()==value.ordinal();});
  if(found!=inputs.end()&&std::size_t(found-inputs.begin())<step.ordinary_inputs)return;
  (void)expose(step,value);
}
Step build_step(const Source &source,bool fresh,std::int32_t position,bool target_conditioned=false) {
  if(position<kBasePosition||position>=kCapacity)throw std::runtime_error("Pro step is outside the finite prepared domain");
  Step result;result.fresh=fresh;result.position=position;
  auto &r=result.recipe;auto &f=r.fragment;
  f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;
  f.region=0x474c4d50524f5354ULL+std::uint64_t(position-kBasePosition)*2+std::uint64_t(fresh);f.quantized=true;
  std::array<Value,kResources> state;
  const auto desc=descriptions();
  for(std::size_t i=0;i<state.size();++i)state[i]=f.input(desc[i]);
  Value hidden=state[kHidden];
  if(fresh||target_conditioned){result.target_hidden=f.inputs.size();hidden=f.input(sem::dense({1,model::kHidden},c::DataType::kBFloat16));}
  result.ordinary_inputs=f.inputs.size();
  const auto p=parameters(r,source);
  const auto embedding=f.one(c::Operation(c::Gather{}),
      {external(r,source.words.at(vocabulary::Role::kEmbedding)),state[kPending]});
  const auto prefix=model::append_prefix(f,hidden,embedding,p);
  glm_original_moe_fragment::Parameters moe;
  moe.router=external(r,source.experts.router);moe.bias=external(r,source.experts.bias);
  for(std::size_t leg=0;leg<3;++leg) {
    moe.shared[leg]=external(r,source.experts.shared[leg]);
    moe.banks[leg]=f.input(expert::resident_description(static_cast<expert::Leg>(leg),kMoe));
  }
  moe.map=f.input(sem::dense({kMoe.experts},c::DataType::kInt32));
  result.model=model::append_step(f,prefix,
      {.main=state[kMain],.codes=state[kCodes],.scales=state[kScales],.selection=state[kSelection],.positions=state[kCursor]},
      p,moe,{.visible=std::int64_t(position)+1,.selected=kSelected,.fresh=fresh});
  const auto change=[&](Entry entry,Value value){result.state_ports[entry]=expose(result,value);};
  change(kMain,result.model->state.main);change(kHidden,result.model->hidden);change(kPending,result.model->pending);
  const auto next=increment(f,state[kCursor]);change(kCursor,next);
  if(fresh) {
    change(kCodes,result.model->state.codes);change(kScales,result.model->state.scales);
    change(kSelection,result.model->state.selection);change(kIndexCursor,next);
    change(kLag,f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{0})));
  } else {
    change(kJournal,f.one(c::Operation(c::Scatter{}),{state[kJournal],state[kLag],prefix.input_norm.stored}));
    change(kJournalPositions,f.one(c::Operation(c::Scatter{}),{state[kJournalPositions],state[kLag],state[kCursor]}));
    change(kLag,increment(f,state[kLag]));
  }
  const auto probe=[&](Value &value){observe_port(result,value);};
  result.model->ports.visit_values(probe);
  result.model->feed_forward.ports.visit_values(probe);
  return result;
}
Step build_catchup(const Source &source,std::int32_t journal_slot=0) {
  if(journal_slot<0||journal_slot>=kJournalRows)throw std::runtime_error("Pro catchup journal slot outside finite schema");
  Step result;result.journal_slot=journal_slot;
  auto &r=result.recipe;auto &f=r.fragment;
  f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;f.region=0x474c4d50524f4348ULL;f.quantized=true;
  const auto desc=descriptions();
  std::array<Value,kResources> state;
  for(std::size_t i=0;i<state.size();++i)state[i]=f.input(desc[i]);
  result.ordinary_inputs=f.inputs.size();
  const auto zero=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{journal_slot}));
  const auto input=f.one(c::Operation(c::Gather{}),{state[kJournal],zero});
  const auto position=f.one(c::Operation(c::Gather{}),{state[kJournalPositions],zero});
  result.catchup=model::append_index_history(f,input,position,state[kCodes],state[kScales],parameters(r,source));
  result.state_ports[kCodes]=expose(result,result.catchup->codes);
  result.state_ports[kScales]=expose(result,result.catchup->scales);
  result.state_ports[kIndexCursor]=expose(result,increment(f,state[kIndexCursor]));
  result.state_ports[kLag]=expose(result,increment(f,state[kLag],-1));
  const auto probe=[&](Value &value){observe_port(result,value);};result.catchup->visit_values(probe);
  return result;
}

struct Epoch final {std::uint64_t epoch,selection_position,reuses;};
static_assert(sizeof(Epoch)==24);
// A complete boundary in this consumer's fixed upstream/initial-state domain.
// The directory owns the original StateVersion, including all target/NextN
// resources, pending and RNG; the fresh continuation choice is model-local.
struct GlmCheckpoint final {
  c::StateVersion version;
  std::size_t consumed=0;
  std::int32_t cursor=0;
};
struct State final {
  c::StateSchema schema;
  c::StateStore store;
  std::vector<c::StateEntryId> entries;
  std::size_t peak_backing=0;
  StateLayout layout;
  std::optional<std::size_t> sampling_entry;
  c::StateEntryId entry(Entry semantic)const{return entries.at(layout.slot(semantic));}
  static State create(const c::StorageOwner &owner,std::size_t budget,const StateLayout &layout,bool with_target=false,bool with_sampling=false) {
    State result;result.layout=layout;const auto &desc=layout.resources;
    std::vector<c::StateEntrySpec> specs(layout.first_target()+(with_target?3:0));
    result.entries.resize(specs.size());
    for(std::size_t i=0;i<desc.size();++i)specs[i]={.description=desc[i],.resource=owner.ref()};
    specs[layout.epoch()]={.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Epoch)};
    if(with_target)for(std::size_t i=0;i<3;++i)specs[layout.first_target()+i]={.description=layout.target[i],.resource=owner.ref()};
    if(with_sampling) {
      result.sampling_entry=specs.size();
      specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(hyle::inference::SamplingCursor)});
      result.entries.resize(specs.size());
    }
    result.schema=take(c::StateSchema::create(specs),"complete original Pro state schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=20,.max_drafts=2,.max_retained_blocks=192,
        .max_backing_bytes=budget}),"bounded original Pro state store");
    for(std::size_t i=0;i<result.entries.size();++i)result.entries[i]=take(result.schema.entry(i),"Pro state entry");
    return result;
  }
  c::ReadLease member(const c::StateVersion &version,Entry entry) const {
    return take(version.member(this->entry(entry)),"actual Pro continuation member");
  }
  c::ReadLease physical_member(const c::StateVersion &version,std::size_t slot)const {
    return take(version.member(entries.at(slot)),"actual physical continuation member");
  }
  std::int32_t scalar(const c::StateVersion &version,Entry entry) const {return values<std::int32_t>(member(version,entry)).at(0);}
  Epoch epoch(const c::StateVersion &version) const {
    const auto raw=read(member(version,kEpoch));Epoch result{};
    if(raw.size()!=sizeof(result))throw std::runtime_error("Pro state metadata extent");
    std::memcpy(&result,raw.data(),raw.size());return result;
  }
  c::StateVersion stage(const c::StateVersion &from,const Step &step,std::span<const c::ReadLease> outputs,std::uint64_t iteration) {
    auto draft=take(store.begin(from),"complete Pro candidate state");
    for(std::size_t i=0;i<layout.resources.size();++i)
      if(step.state_ports.at(i)!=kAbsent)check(draft.replace(entries[i],outputs[step.state_ports[i]]),"stage actual Pro result");
    auto facts=epoch(from);
    if(step.fresh)facts={iteration,std::uint64_t(scalar(from,kCursor)),0};
    else if(step.has_model())++facts.reuses;
    check(draft.replace_metadata(entry(kEpoch),std::as_bytes(std::span(&facts,1))),"stage Pro iteration identity");
    auto result=take(draft.finalize(),"finalize complete Pro candidate");
    peak_backing=std::max(peak_backing,store.stats().backing_bytes);return result;
  }
};

void require_observed(bool passed,const char *where) {
  if(!passed)throw std::runtime_error(std::string("original Pro observation: ")+where);
}
void same_storage(const c::ReadLease &first,const c::ReadLease &second) {
  const auto a=take(first.view(),"first actual Pro storage"),b=take(second.view(),"second actual Pro storage");
  EXPECT_EQ(a.resource(),b.resource());EXPECT_EQ(a.native_handle(),b.native_handle());
  EXPECT_EQ(a.offset(),b.offset());EXPECT_EQ(a.size(),b.size());EXPECT_EQ(first.generation(),second.generation());
}
using Image=std::vector<Bytes>;
Image image(const State &state,const c::StateVersion &version) {
  Image result(state.entries.size());for(std::size_t i=0;i<result.size();++i)result[i]=read(state.physical_member(version,i));
  return result;
}
std::vector<std::uint16_t> fixture_bfloat(std::size_t count,std::size_t salt) {
  std::vector<std::uint16_t> result(count);
  for(std::size_t i=0;i<count;++i) {
    const auto bits=std::bit_cast<std::uint32_t>(float(int((i*37+salt*17)%257)-128)/128.0F);
    result[i]=std::uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
  }
  return result;
}
void add_bound(std::size_t &total,std::size_t bytes) {
  if(bytes>std::numeric_limits<std::size_t>::max()-total)throw std::runtime_error("Pro bounded backing sum overflow");
  total+=bytes;
}
std::size_t aligned_bound(std::size_t bytes) {
  if(bytes>std::numeric_limits<std::size_t>::max()-255)throw std::runtime_error("Pro initial backing alignment overflow");
  return (bytes+255)/256*256;
}

std::shared_ptr<Source> prepare_source(c::Context &context,const c::StorageOwner &owner,c::Queue &queue,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device
#endif
    ) {
  const auto static_directory=directory("HYLE_GLM_PRO_DIR");
  const auto moe_directory=directory("HYLE_GLM_PRO_MOE_DIR");
  if(static_directory.empty()||moe_directory.empty())
    throw std::runtime_error("set HYLE_GLM_PRO_DIR and HYLE_GLM_PRO_MOE_DIR to authenticated original Pro parameters");
  auto result=std::make_shared<Source>();
  auto original=std::make_shared<const own::Data>(own::Data::load(static_directory));
  result->parameters=own::bind(original,owner);
  result->words=vocabulary::Vocabulary::bind(
      std::shared_ptr<const vocabulary::Data>(original,&original->vocabulary),owner);
  auto experts=std::make_shared<const expert::Data>(expert::Data::load(moe_directory,own::kMoeIndexSha256,kMoe));
  result->experts=expert::bind(std::move(experts),owner);
  {
    Recipe recipe;auto &f=recipe.fragment;
    f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;f.region=0x474c4d50524f4b56ULL;f.quantized=true;
    const auto raw=external(recipe,result->parameters.at(own::Role::kKvB));
    const auto prepared=model::append_prepare_kv(f,raw);
    f.outputs={prepared.key,prepared.value};
    auto cold=Native::create(std::move(recipe),context,owner,cuda,1);
    const auto output=cold.execute(queue,{});
    require_observed(glm_pro_mtp_reference::expect_prepared_kv(*original,
        values<std::uint16_t>(output[0]),values<std::uint16_t>(output[1])),"global-macroblock KV decode and BF16 preparation");
    result->prepared[0]={"original Pro prepared BF16 key",sem::dense({model::kHeads,model::kLatent,model::kNope},c::DataType::kBFloat16),owner,output[0]};
    result->prepared[1]={"original Pro prepared BF16 value",sem::dense({model::kHeads,model::kValue,model::kLatent},c::DataType::kBFloat16),owner,output[1]};
    result->cold_submissions=cold.submissions;result->prepared_pool=std::move(cold.outputs);
  }
  // All cold readers have retired; retain only the two compact BF16 layouts
  // on device. Original mapped E4/SF bytes remain for identity and the oracle.
  result->parameters.roles[own::index(own::Role::kKvB)].lease={};
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda) {
    if(!device)throw std::runtime_error("original Pro experts need the same actual typed CUDA resource");
    result->cuda_supply=std::make_unique<supply::CudaSupply>(take(
        supply::CudaSupply::create(context,owner,*device,kMoe),"bounded original Pro CUDA expert supply"));
  } else
#endif
  result->cpu_supply=std::make_unique<supply::CpuSupply>(take(supply::CpuSupply::create(owner,kMoe),
      "bounded original Pro CPU expert supply"));
  return result;
}

std::shared_ptr<TargetSource> prepare_target_source(c::Context &context,const c::StorageOwner &owner,
    c::Queue &queue,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device
#endif
    ) {
  const auto static_directory=directory("HYLE_GLM_PRO_TARGET_DIR");
  const auto moe_directory=directory("HYLE_GLM_PRO_TARGET_MOE_DIR");
  if(static_directory.empty()||moe_directory.empty())
    throw std::runtime_error("set HYLE_GLM_PRO_TARGET_DIR/HYLE_GLM_PRO_TARGET_MOE_DIR to distinct original77 roles");
  // CPU and CUDA borrow the same authenticated mappings, never rehashing the
  // complete target expert source or substituting the distinct layer78 data.
  static const auto original=std::make_shared<const target_own::Data>(target_own::Data::load(static_directory));
  static const auto experts=std::make_shared<const expert::Data>(expert::Data::load(moe_directory,target_own::kMoeIndexSha256,kMoe));
  auto result=std::make_shared<TargetSource>();
  result->parameters=target_own::bind(original,owner);result->experts=expert::bind(experts,owner);
  {
    Recipe recipe;auto &f=recipe.fragment;
    f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;f.region=0x474c4d5037374b56ULL;f.quantized=true;
    const auto prepared=model::append_prepare_kv(f,external(recipe,result->parameters.at(target_own::Role::kKvB)));
    f.outputs={prepared.key,prepared.value};
    auto cold=Native::create(std::move(recipe),context,owner,cuda,1);
    const auto output=cold.execute(queue,{});
    require_observed(glm_pro_mtp_reference::expect_prepared_kv_matrix(original->matrix(target_own::Role::kKvB),
        values<std::uint16_t>(output[0]),values<std::uint16_t>(output[1])),"original77 global-grid KV preparation");
    result->prepared[0]={"original77 prepared key",sem::dense({64,512,192},c::DataType::kBFloat16),owner,output[0]};
    result->prepared[1]={"original77 prepared value",sem::dense({64,256,512},c::DataType::kBFloat16),owner,output[1]};
    result->cold_submissions=cold.submissions;result->prepared_pool=std::move(cold.outputs);
  }
  result->parameters.roles[target_own::index(target_own::Role::kKvB)].lease={};
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda) {
    if(!device)throw std::runtime_error("original77 requires the same typed CUDA resource");
    result->cuda_supply=std::make_unique<supply::CudaSupply>(take(supply::CudaSupply::create(context,owner,*device,kMoe),
        "bounded original77 CUDA supply"));
  } else
#endif
  result->cpu_supply=std::make_unique<supply::CpuSupply>(take(supply::CpuSupply::create(owner,kMoe),"original77 CPU supply"));
  return result;
}
template<class Outputs>
Bytes observed(const Step&,std::span<const c::ReadLease>,const Outputs&,Value);
struct TargetStep final {
  Step step;
  target_model::Result model;
  std::optional<glm_gguf_mtp_fragment::TargetResult> converted;
  std::array<std::size_t,3> state_ports{};
};
TargetStep build_target(const TargetSource &source,const Source &nextn) {
  TargetStep result;auto &step=result.step;auto &r=step.recipe;auto &f=r.fragment;
  f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;f.region=0x474c4d5037375354ULL;f.quantized=true;
  const auto token=f.input(sem::dense({1},c::DataType::kInt32));
  const auto residual=f.input(sem::dense({1,model::kHidden},c::DataType::kBFloat16));
  const auto main=f.input(target_descriptions()[0]),cursor=f.input(target_descriptions()[1]);
  const auto ids=f.input(sem::dense({2048},c::DataType::kInt32));
  step.ordinary_inputs=f.inputs.size();
  // Explicit synthetic layer76 adapter: the REAL token lease selects its
  // original embedding row. This is not a claim of computing layers0..76.
  const auto hidden=f.one(c::Operation(c::Gather{}),{external(r,nextn.words.at(vocabulary::Role::kEmbedding)),token});
  const auto bank=[&](target_own::Role role){return external(r,source.parameters.at(role));};
  target_model::Parameters p;
  p.attention={bank(target_own::Role::kKvNorm),bank(target_own::Role::kKvA),bank(target_own::Role::kOutput),
      bank(target_own::Role::kPostNorm),bank(target_own::Role::kQueryB),external(r,source.prepared[0]),external(r,source.prepared[1])};
  p.input_norm=bank(target_own::Role::kInputNorm);p.query_a=bank(target_own::Role::kQueryA);
  p.query_norm=bank(target_own::Role::kQueryNorm);p.target_norm=bank(target_own::Role::kTargetNorm);
  glm_original_moe_fragment::Parameters moe;
  moe.router=external(r,source.experts.router);moe.bias=external(r,source.experts.bias);
  for(std::size_t leg=0;leg<3;++leg) {
    moe.shared[leg]=external(r,source.experts.shared[leg]);
    moe.banks[leg]=f.input(expert::resident_description(static_cast<expert::Leg>(leg),kMoe));
  }
  moe.map=f.input(sem::dense({256},c::DataType::kInt32));
  result.model=target_model::append(f,hidden,residual,main,ids,cursor,p,moe);
  const std::array outputs{result.model.main,result.model.cursor,result.model.normalized_tap};
  for(std::size_t i=0;i<outputs.size();++i)result.state_ports[i]=expose(step,outputs[i]);
  const auto observe=[&](Value &value){observe_port(step,value);};
  result.model.ports.visit_values(observe);result.model.feed_forward.ports.visit_values(observe);
  return result;
}
struct TargetFrame final {
  std::array<c::ReadLease,5> input;
  std::vector<c::ReadLease> output;
  std::array<c::ReadLease,3> state;
  std::size_t selection=SIZE_MAX;
  ResidentSource::UploadObservation upload;
};
TargetFrame execute_target(const TargetStep &step,ResidentRun<> &program,TargetSource &source,
    const c::StorageOwner &owner,c::Queue &queue,bool cuda,
    const c::ReadLease &token,const c::ReadLease &main,const c::ReadLease &cursor,
    const c::ReadLease &upstream_ids) {
  const auto position=values<std::int32_t>(cursor).at(0);
  if(position<0||position>=kCapacity)throw std::runtime_error("actual target77 cursor exceeds declared finite history");
  TargetFrame result;
  result.input={token,upload(owner,sem::bytes(fixture_bfloat(std::size_t(model::kHidden),std::size_t(position)+41))),
      main,cursor,upstream_ids};
  result.selection=source.selections.size();
  result.output=program.execute(queue,result.input,cuda);
  result.upload=source.last_upload;
  for(std::size_t i=0;i<result.state.size();++i)result.state[i]=result.output.at(step.state_ports[i]);
  return result;
}
void observe_target(const TargetStep &step,TargetSource &source,const Source &nextn,
    const TargetFrame &result) {
  observe_resident_upload(result.upload);
  const auto actual=[&](Value value){return observed(step.step,result.input,result.output,value);};
  require_observed(glm_pro_target_reference::observe(*source.parameters.source,step.model,actual,!source.full_checked),
      "original77 all actual MLA/residual/finalnorm predecessors");
  require_observed(observe_moe(source,step.model.feed_forward,actual,result.selection),"original77 actual selected expert mathematics");
  const std::array input_token{values<std::int32_t>(result.input[0]).at(0)};
  require_observed(glm_shared_vocabulary_reference::expect_embedding(
      glm_pro_mtp_reference::detail::bfloats(actual(step.model.ports.hidden)),
      nextn.words.source->bytes(vocabulary::Role::kEmbedding),input_token,154880,6144),"actual token enters declared layer76 adapter");
  source.full_checked=true;++source.checked_steps;
}
c::StateVersion attach_target(State &state,const c::StateVersion &candidate,const TargetFrame &target) {
  auto draft=take(state.store.begin(candidate),"joint original77/78 candidate");
  const std::array entries{kTargetMain,kTargetCursor,kTargetTap};
  for(std::size_t i=0;i<entries.size();++i)
    check(draft.replace(state.entry(entries[i]),target.state[i]),"same-head original77 state output lease");
  auto result=take(draft.finalize(),"all original77/78 state finalized");
  state.peak_backing=std::max(state.peak_backing,state.store.stats().backing_bytes);return result;
}

struct Initial final {
  Step step;
  glm_official_index_cache::Quantized encoded;
  std::size_t codes=0,scales=0;
};
Initial build_initial() {
  Initial result;auto &f=result.step.recipe.fragment;
  f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;f.region=0x474c4d50524f494eULL;
  const auto input=f.input(sem::dense({kBasePosition,model::kIndexDepth},c::DataType::kBFloat16));
  result.step.ordinary_inputs=1;
  result.encoded=glm_official_index_cache::quantize(f,model::cast(f,input,c::DataType::kFloat32),kBasePosition);
  const auto old_codes=f.constant(sem::dense({kCapacity,model::kIndexDepth},c::DataType::kFloat8E4M3FN),
      sem::bytes(std::vector<std::uint8_t>(std::size_t(kCapacity*model::kIndexDepth),0x7f)));
  const auto old_scales=f.constant(sem::dense({kCapacity,1}),
      sem::bytes(std::vector<float>(std::size_t(kCapacity),std::numeric_limits<float>::quiet_NaN())));
  std::vector<std::int32_t> prefix(std::size_t(kBasePosition),0);std::iota(prefix.begin(),prefix.end(),0);
  const auto indices=f.constant(sem::dense({kBasePosition},c::DataType::kInt32),sem::bytes(prefix));
  result.codes=expose(result.step,f.one(c::Operation(c::Scatter{}),{old_codes,indices,result.encoded.codes}));
  result.scales=expose(result.step,f.one(c::Operation(c::Scatter{}),{old_scales,indices,result.encoded.scale}));
  for(const auto value:{result.encoded.codes,result.encoded.scale,result.encoded.rotated,
      result.encoded.transformed,result.encoded.input})(void)expose(result.step,value);
  return result;
}
std::array<c::ReadLease,kResources> initial_members(const c::StorageOwner &owner,
    const Initial &initial,std::span<const c::ReadLease> output) {
  std::array<c::ReadLease,kResources> result;
  const auto desc=descriptions();
  std::vector<float> main(std::size_t(kCapacity*model::kCacheDepth),std::numeric_limits<float>::quiet_NaN());
  const auto finite=fixture_bfloat(std::size_t(kBasePosition*model::kCacheDepth),3);
  for(std::size_t i=0;i<finite.size();++i)main[i]=glm_shared_vocabulary_reference::detail::finite_bfloat(finite[i]);
  result[kMain]=upload(owner,sem::bytes(main));result[kCodes]=output[initial.codes];result[kScales]=output[initial.scales];
  result[kHidden]=upload(owner,sem::bytes(fixture_bfloat(std::size_t(model::kHidden),5)));
  const auto scalar=[&](Entry entry,std::int32_t value){result[entry]=upload(owner,sem::bytes(std::vector<std::int32_t>{value}));};
  scalar(kPending,1);scalar(kCursor,std::int32_t(kBasePosition));scalar(kIndexCursor,std::int32_t(kBasePosition));
  std::vector<std::int32_t> selected(std::size_t(kSelected),0);std::iota(selected.begin(),selected.end(),0);
  result[kSelection]=upload(owner,sem::bytes(selected));
  result[kJournal]=upload(owner,sem::bytes(std::vector<std::uint16_t>(std::size_t(kJournalRows*model::kHidden),0)));
  result[kJournalPositions]=upload(owner,sem::bytes(std::vector<std::int32_t>(std::size_t(kJournalRows),-1)));
  scalar(kLag,0);
  for(std::size_t i=0;i<result.size();++i)EXPECT_EQ(result[i].size(),take(desc[i].required_bytes(),"initial Pro resource extent"));
  return result;
}

template<class Outputs>
Bytes observed(const Step &step,std::span<const c::ReadLease> input,const Outputs &output,Value value) {
  const auto found=std::find_if(step.observations.begin(),step.observations.end(),
      [&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
  if(found!=step.observations.end())return read(output.at(found->second));
  const auto argument=std::find_if(step.recipe.fragment.inputs.begin(),step.recipe.fragment.inputs.end(),
      [&](Value entry){return entry.ordinal()==value.ordinal();});
  if(argument==step.recipe.fragment.inputs.end()||std::size_t(argument-step.recipe.fragment.inputs.begin())>=input.size())
    throw std::runtime_error("Pro observation lost its actual ordinary input or independent Copy");
  return read(input[std::size_t(argument-step.recipe.fragment.inputs.begin())]);
}
void exact_row(State &state,const c::StateVersion &before,const c::StateVersion &after,
    Entry entry,std::size_t row,std::size_t row_bytes,std::span<const std::byte> actual_update) {
  auto expected=read(state.member(before,entry));
  if(row>=expected.size()/row_bytes||actual_update.size()!=row_bytes)throw std::runtime_error("Pro actual row update extent");
  std::copy(actual_update.begin(),actual_update.end(),expected.begin()+std::ptrdiff_t(row*row_bytes));
  EXPECT_EQ(read(state.member(after,entry)),expected)<<"only the actual produced absolute cache row may change";
}
std::vector<c::ReadLease> arguments(State &state,const c::StateVersion &version,const Step &step,
    const c::ReadLease &target={}) {
  std::vector<c::ReadLease> result;
  for(std::size_t i=0;i<state.layout.resources.size();++i)result.push_back(state.physical_member(version,i));
  if(step.target_hidden!=kAbsent) {
    if(!target)throw std::runtime_error("Pro conditioned prefix requires its explicit previous target tap");
    result.push_back(target);
  }
  return result;
}
void validate_step(State &state,const c::StateVersion &version,const Step &step) {
  const auto cursor=state.scalar(version,kCursor),index=state.scalar(version,kIndexCursor),lag=state.scalar(version,kLag);
  if(step.has_catchup()) {
    if(lag<1||lag>kJournalRows||index+lag!=cursor||state.epoch(version).reuses!=std::uint64_t(lag+step.journal_slot)||
        values<std::int32_t>(state.member(version,kJournalPositions)).at(std::size_t(step.journal_slot))!=index)
      throw std::runtime_error("finite Pro catchup requires the next actual accepted journal row");
  } else {
    if(cursor!=step.position||index+lag!=cursor||lag<0||lag>=kJournalRows)
      throw std::runtime_error("actual Pro cursor/index frontier differs from this cold step");
    if(step.fresh&&lag!=0)throw std::runtime_error("fresh Pro must catch up accepted journal rows first");
  }
}
struct StepUpdates final {Bytes main,index,scales,journal;};
template<class Actual> StepUpdates observe_step(glm_gguf_runtime_source::Source &,const Step &,Actual,std::int32_t,std::size_t);
template<class Actual>
StepUpdates observe_step(Source &source,const Step &step,Actual actual,std::int32_t token,std::size_t selection) {
  if(step.model) {
    require_observed(glm_pro_mtp_reference::observe(*source.parameters.source,*step.model,actual,!source.full_checked,token),
        "original complete Pro NextN actual predecessors");
    require_observed(observe_moe(source,step.model->feed_forward,actual,selection),"original complete Pro MoE");
    source.full_checked=true;++source.checked_steps;
  } else if(step.catchup) {
    require_observed(glm_pro_mtp_reference::observe_index_history(*source.parameters.source,*step.catchup,actual,false),
        "accepted actual Pro journal catchup");
  }
  StepUpdates r;
  if(step.model) {
    const auto row=glm_pro_mtp_reference::detail::widen(glm_pro_mtp_reference::detail::bfloats(actual(step.model->ports.attention.row_stored)));
    const auto span=sem::bytes(row);r.main.assign(span.begin(),span.end());
    if(step.fresh) {r.index=actual(step.model->ports.index->history.encoded.codes);r.scales=actual(step.model->ports.index->history.encoded.scale);}
    else r.journal=actual(step.model->ports.prefix.input_norm.stored);
  } else {r.index=actual(step.catchup->encoded.codes);r.scales=actual(step.catchup->encoded.scale);}
  return r;
}
// Only original completed leases and versions, retained until the scenario's
// next explicit observation. No reference result or second native owner.
struct StepExecution final {
  c::StateVersion before,version;
  std::vector<c::ReadLease> input,output;
  std::size_t selection=SIZE_MAX;
  std::uint64_t iteration=0;
  ResidentSource::UploadObservation upload;
};
template<class SourceType,class Execute>
StepExecution execute_step(State &state,const c::StateVersion &before,const Step &step,
    SourceType &source,std::uint64_t iteration,Execute execute,const c::ReadLease &target={},c::ReadLease *retained_logits=nullptr) {
  validate_step(state,before,step);
  if(!step.fresh&&state.epoch(before).epoch!=iteration)
    throw std::runtime_error("Pro skip/catchup must retain its incoming iteration epoch");
  StepExecution result;result.before=before;result.iteration=iteration;result.input=arguments(state,before,step,target);
  result.selection=step.has_model()?source.selections.size():SIZE_MAX;
  result.output=execute(result.input);
  if constexpr(!SourceType::kPacked)if(step.has_model())result.upload=source.last_upload;
  if(retained_logits) {
    if(!step.has_model())throw std::runtime_error("random Pro proposal requires actual NextN logits");
    const auto logits=step.model?step.model->ports.logits:step.converted_model->ports.logits;
    const auto found=std::find_if(step.observations.begin(),step.observations.end(),
        [&](const auto &entry){return entry.first.ordinal()==logits.ordinal();});
    if(found==step.observations.end())throw std::runtime_error("actual Pro logits were not retained by the ordinary model Graph");
    *retained_logits=result.output.at(found->second);
  }
  result.version=state.stage(before,step,result.output,iteration);
  return result;
}
template<class SourceType>
void observe_step_execution(State &state,const Step &step,SourceType &source,const StepExecution &result) {
  const auto &before=result.before;const auto &after=result.version;const auto &output=result.output;
  const auto iteration=result.iteration;
  if(step.has_model())observe_resident_upload(result.upload);
  const auto actual=[&](Value value){return observed(step,result.input,output,value);};
  const auto updates=observe_step(source,step,actual,state.scalar(before,kPending),result.selection);
  for(std::size_t i=0;i<state.layout.resources.size();++i) {
    if(step.state_ports.at(i)==kAbsent)same_storage(state.physical_member(after,i),state.physical_member(before,i));
    else same_storage(state.physical_member(after,i),output.at(step.state_ports[i]));
  }
  if(step.has_model()) {
    exact_row(state,before,after,kMain,std::size_t(step.position),updates.main.size(),updates.main);
    EXPECT_EQ(state.scalar(after,kCursor),step.position+1);
    if(step.fresh) {
      exact_row(state,before,after,kCodes,std::size_t(step.position),updates.index.size(),updates.index);
      if(state.layout.slots[kScales])exact_row(state,before,after,kScales,std::size_t(step.position),4,updates.scales);
      EXPECT_EQ(state.scalar(after,kIndexCursor),step.position+1);EXPECT_EQ(state.scalar(after,kLag),0);
      EXPECT_EQ(state.epoch(after).epoch,iteration);EXPECT_EQ(state.epoch(after).reuses,0u);
    } else {
      const auto slot=std::size_t(state.scalar(before,kLag));
      exact_row(state,before,after,kJournal,slot,updates.journal.size(),updates.journal);
      const std::array<std::int32_t,1> position{step.position};
      exact_row(state,before,after,kJournalPositions,slot,4,std::as_bytes(std::span(position)));
      EXPECT_EQ(state.scalar(after,kLag),state.scalar(before,kLag)+1);
      EXPECT_EQ(state.epoch(after).reuses,state.epoch(before).reuses+1);
    }
    const auto selected=values<std::int32_t>(state.member(after,kSelection));
    EXPECT_EQ(selected.size(),std::size_t(kSelected));
    for(const auto id:selected)if(id<0||id>step.position)throw std::runtime_error("Pro selected a future/unwritten absolute row");
  } else {
    const auto position=std::size_t(state.scalar(before,kIndexCursor));
    exact_row(state,before,after,kCodes,position,updates.index.size(),updates.index);
    if(state.layout.slots[kScales])exact_row(state,before,after,kScales,position,4,updates.scales);
    EXPECT_EQ(state.scalar(after,kIndexCursor),state.scalar(before,kIndexCursor)+1);EXPECT_EQ(state.scalar(after,kLag),state.scalar(before,kLag)-1);
    same_storage(state.member(after,kMain),state.member(before,kMain));
    same_storage(state.member(after,kSelection),state.member(before,kSelection));
  }
}
// Scenario-only composition. Model steps call execute_step directly and return
// its original result leases to their explicit observation point instead.
template<class SourceType,class Execute>
c::StateVersion execute_observed_step(State &state,const c::StateVersion &before,const Step &step,
    SourceType &source,std::uint64_t iteration,Execute execute,const c::ReadLease &target={},c::ReadLease *retained_logits=nullptr) {
  auto result=execute_step(state,before,step,source,iteration,std::move(execute),target,retained_logits);
  observe_step_execution(state,step,source,result);
  return std::move(result.version);
}

struct Verifier final {
  Recipe recipe;
  std::array<std::size_t,2> pending{};
  std::size_t logits=0,scores=0,ids=0,reject=0;
};
Verifier build_verifier(const Source &source,bool direct_rows=false,bool single_row=false) {
  Verifier result;auto &f=result.recipe.fragment;
  f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;f.region=0x474c4d50524f5646ULL;
  Value taps;
  if(direct_rows) {
    const auto first=f.input(sem::dense({1,model::kHidden},c::DataType::kBFloat16));
    if(single_row)taps=first;
    else taps=f.one(c::Operation(c::Concat{.axis=0}),{first,f.input(sem::dense({1,model::kHidden},c::DataType::kBFloat16))});
  } else taps=f.input(sem::dense({2,model::kHidden},c::DataType::kBFloat16));
  Value logits,scores,ids;std::array<Value,2> row_ids{};
  const auto head=external(result.recipe,source.words.at(vocabulary::Role::kHead));
  if(single_row) {
    logits=f.one(c::model_ops::dense_linear_f32(),{taps,head});
    const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
    scores=top[0];ids=top[1];row_ids[0]=model::reshape(f,ids,{1});
  } else {
    const auto p=glm_target_verifier_fragment::append(f,taps,head,model::kHidden,model::kVocabulary);
    logits=p.logits;scores=p.scores;ids=p.ids;row_ids=p.row_ids;
  }
  const auto publish=[&](Value value) {
    const auto port=f.outputs.size();f.outputs.push_back(f.one(c::Operation(c::Copy{}),{value}));return port;
  };
  result.logits=publish(logits);result.scores=publish(scores);result.ids=publish(ids);
  for(std::size_t row=0;row<(single_row?1u:2u);++row)result.pending[row]=publish(row_ids[row]);
  // The second ranked REAL target row supplies a distinct actual I32 proposal
  // only to the explicit rejection control. It is never a hand-uploaded ID.
  const auto first=model::slice(f,logits,0,0,1);
  const auto top2=f.append(c::Operation(c::TopK{.count=2,.stable=true}),{first},2);
  result.reject=publish(model::reshape(f,model::slice(f,top2[1],1,1,1),{1}));
  return result;
}

void report_head(const Native &program,const Source &source,bool cuda) {
  const auto &graph=program.logical.graph;
  std::size_t heads=0;
  for(const auto &constant:program.recipe.external) {
    const auto &original_head=source.words.at(vocabulary::Role::kHead);
    // Embedding is untied but has the SAME [V,H] description. Authenticate
    // the explicit original role before checking its actual backing identity.
    if(constant.role.name!=original_head.name)continue;
    if(constant.role.description!=original_head.description)
      throw std::runtime_error("original Pro head role changed its full vocabulary description");
    same_storage(constant.role.lease,source.words.at(vocabulary::Role::kHead).lease);
    const auto head=program.logical.constants.at(constant.slot);
    for(std::size_t step=0;step<program.program->compiled.report.step_count();++step) {
      const auto selected=take(program.program->compiled.report.step(step),"original Pro head selected provider");
      for(std::size_t slot=0;slot<selected.node_count;++slot) {
        const auto node=take(program.program->compiled.report.step_node(step,slot),"actual Pro head implementation node");
        const auto operation=take(graph.operation(node),"actual Pro head operation");
        if(!operation.custom_as<c::model_ops::DenseLinearF32>())continue;
        if(take(graph.node_input(node,1),"actual Pro head original readonly bank")!=head)continue;
        const auto activation=take(graph.node_input(node,0),"actual Pro normalized BF16 head input");
        const auto description=take(graph.value_description(activation),"actual Pro head BF16 input description");
        if(description.data_type()!=c::DataType::kBFloat16||description.rank()!=2||description.dimensions()[1]!=model::kHidden||
            (cuda&&selected.provider.name!="cublasLt"))
          throw std::runtime_error("original Pro head must retain BF16 operands and its native full-V provider");
        ++heads;::testing::Test::RecordProperty("head_provider",std::string(selected.provider.name));
      }
    }
  }
  if(heads!=1)throw std::runtime_error("each Pro prediction/verifier needs one shared full-V BF16 head implementation");
}

namespace converted=glm_gguf_mtp_fragment;
namespace converted_source=glm_gguf_runtime_source;
namespace converted_data=glm_gguf_named_parameters;
namespace converted_ref=glm_gguf_mtp_reference;

// Each explicit fixture owns its authenticated host source cache. No common
// static Data can accidentally retain B09 while a B10 case changes its path.
struct ConvertedFixture final {
  const c::ModelArtifactProfileDescription *profile;
  original_named_role_source::Identity identity;
  std::span<const converted_data::Spec> statics,experts;
  std::span<const original_named_role_source::PinnedFile> pins;
  std::uint64_t static_bytes;
  std::string_view expert_plan;
  const char *role_env,*expert_env,*frontier_env,*wait_env;
  mutable std::mutex mutex;
  mutable std::filesystem::path loaded_directory;
  mutable std::shared_ptr<const original_named_role_source::Data> loaded;
  std::shared_ptr<const original_named_role_source::Data> data()const {
    if(!profile||profile->artifact_repository!=identity.repository||profile->artifact_revision!=identity.revision||
        profile->variant!=identity.variant||profile->format_family!="gguf"||statics.size()!=47||experts.size()!=6||
        pins.size()!=statics.size()||!converted_data::actual_pin(expert_plan))
      throw std::runtime_error("converted fixture profile/source/schema/pins differ");
    const auto path=directory(role_env);if(path.empty())throw std::runtime_error("converted actual role directory is absent");
    const auto canonical=std::filesystem::weakly_canonical(path);std::lock_guard lock(mutex);
    if(loaded) {
      if(canonical!=loaded_directory)throw std::runtime_error("converted fixture path changed after its authenticated source was cached");
      return loaded;
    }
    loaded=original_named_role_source::Data::load(canonical,identity,converted_data::manifest(statics,pins,static_bytes));
    loaded_directory=canonical;return loaded;
  }
};
const ConvertedFixture &converted_b09_fixture() {
#if !defined(HYLE_TEST_GLM_B09_ACTUAL_PINS)
  throw std::runtime_error("B09 complete actual static47 payload pins are absent; metadata is not executable supply");
#else
  namespace artifact=glm_gguf_b09_artifacts;
  static const ConvertedFixture fixture{&d5_moe_profile_cases::glm53_primary_gguf,
      {converted_data::kRepository,converted_data::kRevision,converted_data::kVariant,converted_data::kFirstFile},
      converted_data::kStaticSpecs,converted_data::kExpertSpecs,artifact::kStaticPins,converted_data::kStaticBytes,
      artifact::kExpertSourcePlanSha256,"HYLE_GLM_GGUF_B09_DIR","HYLE_GLM_GGUF_B09_EXPERT_DIR",
      "HYLE_GLM_GGUF_FRONTIER_FILE","HYLE_GLM_GGUF_WAIT_FOR_EXPERTS"};
  return fixture;
#endif
}
const ConvertedFixture &converted_b10_fixture() {
#if !defined(HYLE_TEST_GLM_B10_ACTUAL_PINS)
  throw std::runtime_error("B10 complete actual static47 payload pins are absent; metadata is not executable supply");
#else
  namespace artifact=glm_gguf_b10_artifacts;
  static const ConvertedFixture fixture{&d5_moe_profile_cases::glm53_fidelity_gguf,
      {artifact::kRepository,artifact::kRevision,artifact::kVariant,artifact::kFirstFile},
      artifact::kStaticSpecs47,artifact::kExpertSpecs6,artifact::kStaticPins47,artifact::kStaticBytes,
      artifact::kExpertSourcePlanSha256,"HYLE_GLM_GGUF_B10_DIR","HYLE_GLM_GGUF_B10_EXPERT_DIR",
      "HYLE_GLM_GGUF_FRONTIER_FILE","HYLE_GLM_GGUF_WAIT_FOR_EXPERTS"};
  return fixture;
#endif
}
std::function<void(std::span<const std::int32_t>)> converted_frontier(std::shared_ptr<converted_source::Experts> selected,
    const std::filesystem::path &expert_root,const std::filesystem::path &request,bool wait) {
  if(expert_root.empty())throw std::runtime_error("converted explicit selected-expert cache directory is absent");
  auto frontier=std::make_shared<glm_gguf_expert_frontier::Frontier>(glm_gguf_expert_frontier::Frontier{
      std::move(selected),expert_root,request,wait});
  return [frontier](std::span<const std::int32_t> actual_ids){frontier->ensure(actual_ids);};
}
const c::ModelArtifactProfileDescription *converted_profile(const converted_source::ResidentSource &source) {
  if(!source.profile)throw std::runtime_error("converted cold Graph needs its bound artifact profile");return source.profile;
}
std::vector<float> converted_fixture(std::size_t count,std::size_t salt) {
  std::vector<float> r(count);for(std::size_t i=0;i<count;++i)r[i]=float(int((i*37+salt*17)%257)-128)/127.0F;return r;
}
template<class SourceType>
auto converted_parameters_for_reference(const SourceType &,const Step &step) {
  return converted_data::parameter_values([&](std::string_view name) {
    const auto found=std::find_if(step.recipe.external.begin(),step.recipe.external.end(),[&](const External &e){return e.role.name==name;});
    if(found==step.recipe.external.end())throw std::runtime_error("converted original reference role is absent");
    return step.recipe.fragment.constants.at(found->slot);
  },converted_data::kTargetLayer,converted_data::kNextnLayer);
}
glm_gguf_named_parameters::Parameters converted_parameters(Recipe &r,const original_named_role_source::Roles &bound) {
  auto connected=converted_data::connect(r.fragment,bound,converted_data::kTargetLayer,converted_data::kNextnLayer);
  for(const auto &entry:connected.constants) {
    const auto found=std::find_if(r.fragment.constants.begin(),r.fragment.constants.end(),
        [&](Value v){return v.ordinal()==entry.value.ordinal();});
    if(found==r.fragment.constants.end())throw std::runtime_error("converted constant disappeared from cold recipe");
    r.external.push_back({std::size_t(found-r.fragment.constants.begin()),converted_source::readonly(entry.role)});
  }
  return connected.parameters;
}
Value converted_frequency(sem::Fragment &f) {
  const auto &g=converted_data::kGeometry;std::vector<float> table(std::size_t(g.rope/2));
  for(std::size_t i=0;i<table.size();++i)table[i]=float(std::pow(8000000.,-2*double(i)/double(g.rope)));
  return f.constant(sem::dense({g.rope/2}),sem::bytes(table));
}
std::array<Value,3> converted_banks(sem::Fragment &f,const converted_source::ResidentSource &source) {
  std::array<Value,3> banks;
  for(std::size_t leg=0;leg<3;++leg)banks[leg]=f.input(source.cpu_supply->layout().resident(leg));return banks;
}
Step build_step(const converted_source::Source &source,bool fresh,std::int32_t position,bool target_conditioned=false) {
  if(position<kBasePosition||position>=kCapacity)throw std::runtime_error("converted cold step position outside prepared history");
  const auto layout=converted_layout();const auto &g=converted_data::kGeometry;
  Step result;result.fresh=fresh;result.position=position;result.state_ports.assign(layout.resources.size(),kAbsent);
  auto &r=result.recipe;auto &f=r.fragment;f.profile=converted_profile(source);
  f.region=0x474c4d4747555354ULL+std::uint64_t(position-kBasePosition)*2+std::uint64_t(fresh);f.quantized=true;
  std::array<Value,kResources> state{};
  for(std::size_t e=0;e<kResources;++e)if(layout.slots[e])state[e]=f.input(layout.resources[*layout.slots[e]]);
  auto hidden=state[kHidden];
  if(fresh||target_conditioned){result.target_hidden=f.inputs.size();hidden=f.input(sem::dense({1,g.hidden}));}
  result.ordinary_inputs=f.inputs.size();const auto p=converted_parameters(r,source.parameters).nextn();
  const auto embedding=converted::compute::embedding_q8(f,state[kPending],p.embedding,g);
  const auto prefix=converted::compute::append_f32_prefix(f,hidden,embedding,p,g,converted::leaf::PackedProjections{});
  const auto banks=converted_banks(f,source);const auto map=f.input(sem::dense({kMoe.experts},c::DataType::kInt32));
  result.converted_model=converted::append_step(f,prefix,{state[kMain],state[kCodes],state[kSelection],state[kCursor]},
      p,g,kMoe,banks,map,converted_frequency(f),position+1,kSelected,fresh,converted::leaf::PackedProjections{});
  const auto change=[&](Entry entry,Value value){result.state_ports.at(layout.slot(entry))=expose(result,value);};
  change(kMain,result.converted_model->state.main);change(kHidden,result.converted_model->hidden);
  change(kPending,result.converted_model->pending);const auto next=increment(f,state[kCursor]);change(kCursor,next);
  if(fresh) {
    change(kCodes,result.converted_model->state.keys);change(kSelection,result.converted_model->state.selection);
    change(kIndexCursor,next);change(kLag,f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{0})));
  } else {
    change(kJournal,f.one(c::Operation(c::Scatter{}),{state[kJournal],state[kLag],prefix.input_norm.output}));
    change(kJournalPositions,f.one(c::Operation(c::Scatter{}),{state[kJournalPositions],state[kLag],state[kCursor]}));
    change(kLag,increment(f,state[kLag]));
  }
  const auto probe=[&](Value &value){observe_port(result,value);};
  result.converted_model->ports.visit_values(probe);result.converted_model->feed_forward.visit_values(probe);
  return result;
}
Step build_catchup(const converted_source::Source &source,std::int32_t journal_slot=0) {
  if(journal_slot<0||journal_slot>=kJournalRows)throw std::runtime_error("converted Pro catchup journal slot outside finite schema");
  const auto layout=converted_layout();Step result;result.journal_slot=journal_slot;result.state_ports.assign(layout.resources.size(),kAbsent);
  auto &r=result.recipe;auto &f=r.fragment;f.profile=converted_profile(source);
  f.region=0x474c4d4747554348ULL;f.quantized=true;std::array<Value,kResources> state{};
  for(std::size_t e=0;e<kResources;++e)if(layout.slots[e])state[e]=f.input(layout.resources[*layout.slots[e]]);
  result.ordinary_inputs=f.inputs.size();const auto p=converted_parameters(r,source.parameters);
  const auto zero=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{journal_slot}));
  const auto input=f.one(c::Operation(c::Gather{}),{state[kJournal],zero});
  const auto position=f.one(c::Operation(c::Gather{}),{state[kJournalPositions],zero});
  result.converted_catchup=converted::cache::append_index_history(f,input,position,state[kCodes],p.block[1],
      converted_data::kGeometry,converted_frequency(f),converted::leaf::PackedProjections{});
  result.state_ports[layout.slot(kCodes)]=expose(result,result.converted_catchup->keys);
  result.state_ports[layout.slot(kIndexCursor)]=expose(result,increment(f,state[kIndexCursor]));
  result.state_ports[layout.slot(kLag)]=expose(result,increment(f,state[kLag],-1));
  const auto probe=[&](Value &value){observe_port(result,value);};result.converted_catchup->visit_values(probe);return result;
}

template<class SourceType> auto converted_original(const SourceType &source,const Step &step) {
  return [&](Value weight)->std::pair<const original_named_role_source::Spec&,std::span<const std::byte>> {
    const auto &r=step.recipe;
    const auto found=std::find_if(r.fragment.constants.begin(),r.fragment.constants.end(),[&](Value v){return v.ordinal()==weight.ordinal();});
    if(found==r.fragment.constants.end())throw std::runtime_error("original converted projection weight is not a constant");
    const auto slot=std::size_t(found-r.fragment.constants.begin());
    const auto external=std::find_if(r.external.begin(),r.external.end(),[&](const External &e){return e.slot==slot;});
    if(external==r.external.end())throw std::runtime_error("original converted role has no immutable source lease");
    const auto &data=*source.parameters.source;
    return {data.spec(external->role.name),data.bytes(external->role.name)};
  };
}
template<class Actual>
bool observe_converted_model(converted_source::ResidentSource &source,const converted::Moe &moe,const sem::Fragment &f,
    const glm_gguf_model_types::Block &p,Actual actual,const auto &original,bool full,bool cuda) {
  const auto codec=cuda?converted_ref::base::ActivationCodec::kCudaMmvqQ8_1:converted_ref::base::ActivationCodec::kCpuAvx2;
  const auto experts=[&](std::size_t leg,std::int32_t id) {
    const auto &data=*source.experts.source;
    return std::pair{data.native(expert::Leg(leg)),data.expert_bytes(expert::Leg(leg),id)};
  };
  return converted_ref::moe(moe,f,p,kMoe,actual,original,experts,full,codec);
}
template<class Actual>
StepUpdates observe_step(converted_source::Source &source,const Step &step,Actual actual,std::int32_t token,std::size_t) {
  const auto &f=step.recipe.fragment;const auto p=converted_parameters_for_reference(source,step);
  const auto original=converted_original(source,step);
  const auto codec=source.cuda?converted_ref::base::ActivationCodec::kCudaMmvqQ8_1:converted_ref::base::ActivationCodec::kCpuAvx2;
  const auto &g=converted_data::kGeometry;StepUpdates updates;
  if(step.converted_model) {
    const auto &r=*step.converted_model;const auto &prefix=r.ports.prefix;
    require_observed(qwen27_original_nextn_reference::expect_embedding(converted_ref::floats(actual(prefix.embedding_norm.input)),
        source.parameters.source->bytes("token_embd.weight"),std::size_t(g.hidden),std::size_t(g.vocabulary),token,GGML_TYPE_Q8_0),"B09 exact original embedding feedback");
    require_observed(converted_ref::norm(prefix.hidden_norm,p.hidden_norm,g,actual,original)&&
        converted_ref::norm(prefix.embedding_norm,p.embedding_norm,g,actual,original)&&
        converted_ref::projection(prefix.eh,f,actual,original,!source.full_checked,codec)&&
        converted_ref::norm(prefix.input_norm,p.block[1].input_norm,g,actual,original)&&
        converted_ref::projection(prefix.query,f,actual,original,!source.full_checked,codec)&&
        converted_ref::norm(prefix.query_norm,p.block[1].query_norm,g,actual,original),"converted original EH/QA actual F32 predecessors");
    require_observed(converted_ref::attention(r.ports.attention,r.ports.positions,r.state.main,r.state.selection,f,p.block[1],g,actual,original,!source.full_checked,codec)&&
        observe_converted_model(source,r.feed_forward,f,p.block[1],actual,original,!source.moe_full_checked,source.cuda)&&
        converted_ref::norm(r.ports.shared_norm,p.shared_norm,g,actual,original),"converted MLA/MoE/shared norm actual predecessors");
    const auto hidden=converted_ref::floats(actual(r.hidden)),logits=converted_ref::floats(actual(r.ports.logits));
    require_observed(qwen24_original_nextn_reference::expect_projection(logits,source.parameters.source->bytes("output.weight"),
        std::size_t(g.vocabulary),std::size_t(g.hidden),GGML_TYPE_Q8_0,hidden,codec,"converted original NextN full head",!source.full_checked),"converted raw-byte fullV head");
    const auto ids=converted_ref::integers(actual(r.pending));const auto scores=converted_ref::floats(actual(r.ports.top_values));
    require_observed(qwen27_original_nextn_reference::expect_greedy(logits,scores.at(0),ids.at(0)),"converted strict fullV greedy");
    if(r.ports.history) {
      require_observed(converted_ref::history(*r.ports.history,f,p.block[1],g,actual,original,!source.full_checked,codec),"converted fresh key producer");
      updates.index=actual(r.ports.history->transformed);
      require_observed(converted_ref::projection(*r.ports.index_query,f,actual,original,!source.full_checked,codec)&&
          converted_ref::projection(*r.ports.index_weights,f,actual,original,true,codec)&&
          converted_ref::close(converted_ref::floats(actual(*r.ports.index_rotated)),converted_ref::rotate(
              converted_ref::floats(actual(r.ports.index_query->output)),converted_ref::integers(actual(r.ports.positions)),
              std::size_t(g.index_heads),std::size_t(g.index_depth),g),"B09 query index RoPE")&&
          converted_ref::selection(*r.ports.index,*r.ports.index_rotated,r.state.keys,r.ports.index_weights->output,f,g,actual),
          "B09 actual fresh index queries/scores/Top2048");
    }
    require_observed(converted_ref::add(converted_ref::floats(actual(r.ports.attention.residual_sum)),
        converted_ref::floats(actual(prefix.eh.output)),converted_ref::floats(actual(r.ports.attention.output.output.output)),"B09 EH/O residual")&&
        converted_ref::add(converted_ref::floats(actual(r.ports.combined)),converted_ref::floats(actual(r.ports.attention.residual_sum)),
            converted_ref::floats(actual(r.feed_forward.output)),"B09 residual/FFN sum"),"B09 actual F32 residual sums");
    updates.main=actual(r.ports.attention.row_stored);updates.journal=actual(prefix.input_norm.output);
    source.full_checked=true;source.moe_full_checked=true;++source.checked_steps;
  } else {
    require_observed(converted_ref::history(*step.converted_catchup,f,p.block[1],g,actual,original,false,codec),"converted accepted actual journal catchup");
    updates.index=actual(step.converted_catchup->transformed);
  }
  return updates;
}

TargetStep build_target(const converted_source::TargetSource &source,const converted_source::Source &nextn) {
  const auto layout=converted_layout();const auto &g=converted_data::kGeometry;TargetStep result;
  auto &step=result.step;auto &r=step.recipe;auto &f=r.fragment;f.profile=converted_profile(source);
  f.region=0x474c4d4747373754ULL;f.quantized=true;
  const auto token=f.input(sem::dense({1},c::DataType::kInt32)),residual=f.input(sem::dense({1,g.hidden}));
  const auto main=f.input(layout.target[0]),cursor=f.input(layout.target[1]);
  const auto ids=f.input(sem::dense({kSelected},c::DataType::kInt32));step.ordinary_inputs=f.inputs.size();
  const auto p=converted_parameters(r,source.parameters);
  const auto hidden=converted::compute::embedding_q8(f,token,p.embedding,g);
  const auto banks=converted_banks(f,source);const auto map=f.input(sem::dense({kMoe.experts},c::DataType::kInt32));
  result.converted=converted::append_target(f,hidden,residual,main,ids,cursor,p.block[0],p.target_norm,g,kMoe,
      banks,map,converted_frequency(f),converted::leaf::PackedProjections{});
  const std::array output{result.converted->main,result.converted->cursor,result.converted->normalized_tap};
  for(std::size_t i=0;i<3;++i)result.state_ports[i]=expose(step,output[i]);
  const auto probe=[&](Value &v){observe_port(step,v);};result.converted->visit_values(probe);
  (void)nextn;return result;
}
TargetFrame execute_target(const TargetStep &step,ResidentRun<converted_source::ResidentSource> &program,
    converted_source::TargetSource &source,const c::StorageOwner &owner,
    c::Queue &queue,bool cuda,const c::ReadLease &token,const c::ReadLease &main,const c::ReadLease &cursor,const c::ReadLease &ids) {
  const auto position=values<std::int32_t>(cursor).at(0);
  if(position<0||position>=kCapacity)throw std::runtime_error("converted target cursor outside actual main history");
  const auto fixture=converted_fixture(std::size_t(converted_data::kGeometry.hidden),std::size_t(position)+41);
  TargetFrame frame;frame.input={token,upload(owner,sem::bytes(fixture)),main,cursor,ids};
  frame.selection=source.selections.size();frame.output=program.execute(queue,frame.input,cuda);
  for(std::size_t i=0;i<3;++i)frame.state[i]=frame.output.at(step.state_ports[i]);
  return frame;
}
void observe_target(const TargetStep &step,converted_source::TargetSource &source,
    const converted_source::Source &nextn,const TargetFrame &frame) {
  const auto actual=[&](Value v){return observed(step.step,frame.input,frame.output,v);};
  const auto p=converted_parameters_for_reference(source,step.step);const auto original=converted_original(source,step.step);
  const auto &r=*step.converted;const auto &g=converted_data::kGeometry;
  const auto codec=source.cuda?converted_ref::base::ActivationCodec::kCudaMmvqQ8_1:converted_ref::base::ActivationCodec::kCpuAvx2;
  require_observed(converted_ref::norm(r.input_norm,p.block[0].input_norm,g,actual,original)&&
      converted_ref::projection(r.query,step.step.recipe.fragment,actual,original,!source.full_checked,codec)&&
      converted_ref::norm(r.query_norm,p.block[0].query_norm,g,actual,original)&&
      converted_ref::attention(r.attention,r.position,r.main,r.selection,step.step.recipe.fragment,p.block[0],g,actual,original,!source.full_checked,codec)&&
      observe_converted_model(source,r.feed_forward,step.step.recipe.fragment,p.block[0],actual,original,!source.moe_full_checked,source.cuda)&&
      converted_ref::norm(r.final_norm,p.target_norm,g,actual,original),"converted distinct77 MLA/MoE/finalnorm");
  require_observed(qwen27_original_nextn_reference::expect_embedding(converted_ref::floats(actual(r.hidden)),
      nextn.parameters.source->bytes("token_embd.weight"),std::size_t(g.hidden),std::size_t(g.vocabulary),values<std::int32_t>(frame.input[0]).at(0),GGML_TYPE_Q8_0)&&
      converted_ref::add(converted_ref::floats(actual(r.input_sum)),converted_ref::floats(actual(r.hidden)),
          converted_ref::floats(actual(r.residual)),"B09 target upstream rawsum")&&
      converted_ref::add(converted_ref::floats(actual(r.attention.residual_sum)),converted_ref::floats(actual(r.input_sum)),
          converted_ref::floats(actual(r.attention.output.output.output)),"B09 target O residual")&&
      converted_ref::add(converted_ref::floats(actual(r.post_ffn)),converted_ref::floats(actual(r.attention.residual_sum)),
          converted_ref::floats(actual(r.feed_forward.output)),"B09 target final FFN sum"),"B09 target original row/raw residual sums");
  source.full_checked=true;source.moe_full_checked=true;++source.checked_steps;
}
Verifier build_verifier(const converted_source::Source &source,bool direct_rows=false,bool single_row=false) {
  Verifier r;auto &f=r.recipe.fragment;const auto &g=converted_data::kGeometry;
  f.profile=converted_profile(source);f.region=0x474c4d4747564552ULL;
  auto taps=f.input(sem::dense({direct_rows?1:2,g.hidden}));
  if(direct_rows&&!single_row)taps=f.one(c::Operation(c::Concat{.axis=0}),{taps,f.input(sem::dense({1,g.hidden}))});
  const auto head=external(r.recipe,source.words.at(vocabulary::Role::kHead));
  const auto logits=converted::leaf::linear(f,taps,head).output;
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
  const auto publish=[&](Value v){const auto port=f.outputs.size();f.outputs.push_back(converted::compute::copy(f,v));return port;};
  r.logits=publish(logits);r.scores=publish(top[0]);r.ids=publish(top[1]);
  for(std::size_t row=0;row<(single_row?1u:2u);++row)
    r.pending[row]=publish(converted::compute::reshape(f,converted::compute::slice(f,top[1],0,std::int64_t(row),1),{1}));
  const auto top2=f.append(c::Operation(c::TopK{.count=2,.stable=true}),{converted::compute::slice(f,logits,0,0,1)},2);
  r.reject=publish(converted::compute::reshape(f,converted::compute::slice(f,top2[1],1,1,1),{1}));return r;
}
void report_head(const Native &program,const converted_source::Source &source,bool cuda) {
  const auto &head=source.words.at(vocabulary::Role::kHead);std::size_t count=0;
  for(const auto &entry:program.recipe.external)if(entry.role.name==head.name) {
    same_storage(entry.role.lease,head.lease);const auto w=program.logical.constants.at(entry.slot);
    for(std::size_t i=0;i<program.program->compiled.report.step_count();++i) {
      const auto record=take(program.program->compiled.report.step(i),"converted original head provider");
      for(std::size_t j=0;j<record.node_count;++j) {
        const auto node=take(program.program->compiled.report.step_node(i,j),"converted head original node");
        const auto op=take(program.logical.graph.operation(node),"converted head operation");
        if(!op.get_if<c::MatMul>()||take(program.logical.graph.node_input(node,1),"converted head actual W").ordinal()!=w.ordinal())continue;
        if(record.provider.name!=(cuda?"ggml-cuda":"ggml-cpu"))throw std::runtime_error("converted packed head lost its certified backend");
        ++count;
      }
    }
  }
  if(count!=1)throw std::runtime_error("converted full vocabulary requires one original single MatMul");
}

template<class SourceType>
Recipe build_prefix_verifier(const SourceType &source) {
  Recipe result;auto &f=result.fragment;
  constexpr bool packed=std::is_same_v<SourceType,converted_source::Source>;
  if constexpr(packed)f.profile=converted_profile(source);
  else f.profile=&d5_moe_semantic_fragments::profiles::glm53_official;
  f.region=0x474c4d5052565233ULL;
  const auto type=packed?c::DataType::kFloat32:c::DataType::kBFloat16;
  auto taps=f.input(sem::dense({1,model::kHidden},type));
  for(std::size_t row=1;row<3;++row)taps=f.one(c::Operation(c::Concat{.axis=0}),{taps,f.input(sem::dense({1,model::kHidden},type))});
  const auto head=external(result,source.words.at(vocabulary::Role::kHead));
  const auto logits=[&] {
    if constexpr(packed)return converted::leaf::linear(f,taps,head).output;
    else return f.one(c::model_ops::dense_linear_f32(),{taps,head});
  }();
  f.outputs={f.one(c::Operation(c::Copy{}),{logits})};return result;
}

struct OfficialRuntime final {
  static constexpr bool kPacked=false;using Tap=std::uint16_t;using ResidentSource=::ResidentSource;using TargetSource=::TargetSource;
  static auto layout(){return official_layout();}
  static auto fixture(std::size_t n,std::size_t salt){return fixture_bfloat(n,salt);}
  static auto build_initial(){return ::build_initial();}
  static auto initial_members(const c::StorageOwner &owner,const Initial &i,std::span<const c::ReadLease> out){return ::initial_members(owner,i,out);}
  static const auto &boundary(const Step &s){return s.model->feed_forward.boundary;}
  static const auto &target_boundary(const TargetStep &s){return s.model.feed_forward.boundary;}
  static auto prepare_source(auto &&...args){return ::prepare_source(args...);}
  static auto prepare_target_source(const Source &,auto &&...args){return ::prepare_target_source(args...);}
  static void check_initial(const Initial &i,std::span<const Tap> keys,const auto &actual) {
    require_observed(glm_pro_mtp_reference::detail::quantized(i.encoded,keys,std::size_t(kBasePosition),actual),"actual initial encoded prefix");
  }
  static bool check_logits(const Source &source,std::span<const float> logits,std::span<const Tap> taps,bool full,
      std::span<const std::int32_t> ids,bool) {
    return glm_shared_vocabulary_reference::expect_logits(logits,taps,source.words.source->bytes(vocabulary::Role::kHead),
        std::size_t(model::kVocabulary),std::size_t(model::kHidden),full,ids);
  }
};
struct ConvertedRuntime final {
  static constexpr bool kPacked=true;using Tap=float;
  using ResidentSource=converted_source::ResidentSource;using TargetSource=converted_source::TargetSource;
  static auto layout(){return converted_layout();}
  static auto fixture(std::size_t n,std::size_t salt){return converted_fixture(n,salt);}
  static const auto &boundary(const Step &s){return s.converted_model->feed_forward.boundary;}
  static const auto &target_boundary(const TargetStep &s){return s.converted->feed_forward.boundary;}
  static std::shared_ptr<converted_source::Source> prepare_source(c::Context &,const c::StorageOwner &owner,c::Queue &,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,const c::cuda::MemoryResource *
#endif
      ,const ConvertedFixture *selected_fixture=nullptr) {
    const auto &fixture=selected_fixture?*selected_fixture:converted_b09_fixture();
    const auto expert_directory=directory(fixture.expert_env);
    if(expert_directory.empty())throw std::runtime_error("converted selected-expert cache directory is absent");
    const auto data=fixture.data();
    auto source=std::make_shared<converted_source::Source>();source->parameters=original_named_role_source::bind(data,owner);
    source->cuda=cuda;source->profile=fixture.profile;source->expert_directory=expert_directory;
    source->frontier_file=directory(fixture.frontier_env);source->wait_for_experts=directory(fixture.wait_env)=="1";
    source->words.source=data;
    source->words.roles={converted_source::readonly(source->parameters.at("token_embd.weight")),
        converted_source::readonly(source->parameters.at("output.weight")),converted_source::readonly(source->parameters.at("blk.78.nextn.shared_head_norm.weight"))};
    source->target_contract=converted_source::contract(converted_source::Section::kTarget,
        std::string(fixture.expert_plan),data->identity(),fixture.experts,converted_data::kTargetLayer);
    auto selected=std::make_shared<converted_source::Experts>(converted_source::contract(converted_source::Section::kDraft,
        std::string(fixture.expert_plan),data->identity(),fixture.experts,converted_data::kNextnLayer),kMoe);
    source->experts.source=selected;
    source->before_upload=converted_frontier(selected,source->expert_directory,source->frontier_file,source->wait_for_experts);
    source->cpu_supply=std::make_unique<converted_source::Supply>(take(converted_source::Supply::create(owner,*selected),"converted packed selected native pool"));return source;
  }
  static std::shared_ptr<TargetSource> prepare_target_source(const converted_source::Source &nextn,c::Context &,
      const c::StorageOwner &owner,c::Queue &,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,const c::cuda::MemoryResource *
#endif
      ) {
    if(!nextn.target_contract)throw std::runtime_error("converted distinct target source metadata was not prepared");
    auto source=std::make_shared<TargetSource>();source->parameters=nextn.parameters;source->cuda=cuda;source->profile=nextn.profile;
    auto selected=std::make_shared<converted_source::Experts>(*nextn.target_contract,kMoe);
    source->experts.source=selected;
    source->before_upload=converted_frontier(selected,nextn.expert_directory,nextn.frontier_file,nextn.wait_for_experts);
    source->cpu_supply=std::make_unique<converted_source::Supply>(take(converted_source::Supply::create(owner,*selected),"converted distinct77 selected packed pool"));return source;
  }
  static Initial build_initial(const converted_source::Source &source) {
    Initial i;auto &f=i.step.recipe.fragment;f.profile=converted_profile(source);f.region=0x474c4d4747494e49ULL;
    const auto keys=f.input(sem::dense({kBasePosition,model::kIndexDepth}));i.step.ordinary_inputs=1;
    const auto old=f.constant(sem::dense({kCapacity,model::kIndexDepth}),sem::bytes(std::vector<float>(std::size_t(kCapacity*model::kIndexDepth),std::numeric_limits<float>::quiet_NaN())));
    std::vector<std::int32_t> prefix(std::size_t(kBasePosition),0);std::iota(prefix.begin(),prefix.end(),0);
    const auto ids=f.constant(sem::dense({kBasePosition},c::DataType::kInt32),sem::bytes(prefix));
    i.codes=expose(i.step,f.one(c::Operation(c::Scatter{}),{old,ids,keys}));i.scales=kAbsent;return i;
  }
  static void check_initial(const Initial &i,std::span<const float> keys,const auto &actual) {
    const auto value=std::find_if(i.step.observations.begin(),i.step.observations.end(),[&](const auto &e){return e.second==i.codes;});
    if(value==i.step.observations.end())throw std::runtime_error("B09 initial actual source output missing");
    const auto rows=converted_ref::floats(actual(value->first));
    if(rows.size()!=std::size_t(kCapacity*model::kIndexDepth)||!std::equal(keys.begin(),keys.end(),rows.begin()))
      throw std::runtime_error("B09 finite external prefix changed during initial F32 storage");
    for(std::size_t j=keys.size();j<rows.size();++j)if(!std::isnan(rows[j]))throw std::runtime_error("B09 future index poison changed");
  }
  static std::vector<c::ReadLease> initial_members(const c::StorageOwner &owner,const Initial &i,std::span<const c::ReadLease> outputs) {
    const auto l=layout();std::vector<c::ReadLease> r(l.resources.size());
    const auto put=[&](Entry e,const auto &v){r[l.slot(e)]=upload(owner,sem::bytes(v));};
    std::vector<float> main(std::size_t(kCapacity*model::kCacheDepth),std::numeric_limits<float>::quiet_NaN());
    const auto valid=fixture(std::size_t(kBasePosition*model::kCacheDepth),3);std::copy(valid.begin(),valid.end(),main.begin());
    put(kMain,main);r[l.slot(kCodes)]=outputs[i.codes];put(kHidden,fixture(std::size_t(model::kHidden),5));
    for(const auto [e,v]:std::array{std::pair{kPending,1},std::pair{kCursor,int(kBasePosition)},std::pair{kIndexCursor,int(kBasePosition)},std::pair{kLag,0}})
      put(e,std::vector<std::int32_t>{v});
    std::vector<std::int32_t> ids(std::size_t(kSelected),0);std::iota(ids.begin(),ids.end(),0);put(kSelection,ids);
    put(kJournal,std::vector<float>(std::size_t(kJournalRows*model::kHidden),0));put(kJournalPositions,std::vector<std::int32_t>(std::size_t(kJournalRows),-1));
    for(std::size_t slot=0;slot<r.size();++slot)if(r[slot].size()!=l.resources[slot].required_bytes().value())throw std::runtime_error("B09 initial actual resource extent differs");return r;
  }
  static bool check_logits(const converted_source::Source &source,std::span<const float> logits,std::span<const float> taps,bool full,
      std::span<const std::int32_t> ids,bool cuda) {
    const auto &g=converted_data::kGeometry;const auto codec=cuda?converted_ref::base::ActivationCodec::kCudaMmvqQ8_1:converted_ref::base::ActivationCodec::kCpuAvx2;
    if(taps.size()!=ids.size()*std::size_t(g.hidden)||logits.size()!=ids.size()*std::size_t(g.vocabulary))return false;
    for(std::size_t row=0;row<ids.size();++row)if(!qwen24_original_nextn_reference::expect_projection(logits.subspan(row*std::size_t(g.vocabulary),std::size_t(g.vocabulary)),
        source.parameters.source->bytes("output.weight"),std::size_t(g.vocabulary),std::size_t(g.hidden),GGML_TYPE_Q8_0,
        taps.subspan(row*std::size_t(g.hidden),std::size_t(g.hidden)),codec,"B09 actual fullV verifier",full))return false;return true;
  }
  static std::size_t ready(const converted_source::Source &source,const TargetSource &target,const c::StorageOwner &owner) {
    std::size_t bytes=0;
    for(const auto &[name,role]:source.parameters.roles) {
      (void)name;if(role.owner.ref()!=owner.ref())throw std::runtime_error("B09 actual static source moved to another owner");
      add_bound(bytes,role.lease.size());auto lease=role.lease;EXPECT_FALSE(lease.try_write());
      same_storage(role.lease,target.parameters.at(role.name).lease);
    }
    return bytes;
  }
};

// The finite original GLM consumer owns its actual StateHeads and programs.
// RequestLoop owns only FIFO, turns and credits; old StateVersions retained by
// the fixture/directory do not count as resident execution owners.
// Borrowed finite GLM mathematical steps over the original programs and
// StateStore. Publication and accepted-journal catchup stay with their owner.
template<class Runtime> struct GlmStepRuntime final {
  using Resident=ResidentRun<typename Runtime::ResidentSource>;
  using ModelSource=std::conditional_t<Runtime::kPacked,converted_source::Source,::Source>;
  using Cursor=hyle::inference::SamplingCursor;
  using Grant=hyle::inference::RequestCreditGrant;
  using Limits=hyle::inference::mtp::SamplingLimits;
  struct Stage final {const Step &step;Resident &program;};
  struct Target final {
    typename Runtime::TargetSource &source;
    const TargetStep &step;
    Resident &program;
    const c::ReadLease &upstream;
    const Verifier &policy;
    Native &policy_program;
    Stage repair;
  };
  struct Sampler final {Native &proposal,&decision;};
  struct Counters final {std::size_t &feedback,&skips,&repairs,&repair_submissions;};
  struct Proposal final {c::StateVersion checkpoint;std::vector<c::ReadLease> random;};
  struct Selection final {
    hyle::inference::mtp::AcceptedPrefix prefix;
    c::ReadLease pending;
    bool sampled_accept=false;
  };
  struct Advance final {
    c::StateVersion version;
    const Step *step=nullptr;
    StepExecution execution;
    std::optional<TargetFrame> target;
  };
  struct Candidate final {
    c::StateVersion second;
    std::vector<c::ReadLease> checked;
    Selection selected;
    c::StateVersion checkpoint;
    c::StateVersion version;
    std::optional<Advance> advancement;
    std::optional<StepExecution> repair;
  };

  State &state;
  const c::StorageOwner &owner;
  c::Queue &queue;
  ModelSource &source;
  bool cuda;
  Target *target;
  Sampler *sampler;
  Stage second;
  const Verifier &verifier;
  Native &verifier_program;
  Counters counts;

  Advance advance(const c::StateVersion &from,Stage stage,std::uint64_t iteration,
      c::ReadLease target_input={},c::ReadLease *first_logits=nullptr) {
    return advance_impl(from,stage,iteration,target_input,first_logits,false);
  }

  Proposal propose(const c::StateVersion &from,const c::ReadLease &logits,Cursor cursor) {
    if(!sampler)return {from,{}};
    check_cursor(from,cursor);
    (void)cursor.advanced(3);
    auto random=sample(logits,cursor);
    auto draft=take(state.store.begin(from),"actual original Pro random proposal");
    check(draft.replace(state.entry(kPending),random[1]),"real NextN categorical I32 feeds target row1");
    auto checkpoint=take(draft.finalize(),"preserve complete original Pro proposal checkpoint");
    return {std::move(checkpoint),std::move(random)};
  }

  Cursor prepare_ordinary(const c::StateVersion &from,Cursor cursor) const {
    if(sampler&&!target)throw std::runtime_error("random Pro ordinary continuation requires its target program");
    if(sampler)check_cursor(from,cursor);
    return sampler?cursor.advanced(1):cursor;
  }

  Advance ordinary(const c::StateVersion &from,Stage stage,std::uint64_t iteration,
      c::ReadLease target_input,Cursor cursor,Cursor next_cursor,
      std::vector<c::ReadLease> *observed_policy=nullptr) {
    auto continued=advance(from,stage,iteration,std::move(target_input));
    if(!sampler)return continued;
    const std::array input{state.member(continued.version,kTargetTap)};
    auto policy=target->policy_program.execute(queue,input);
    const auto sampled=sample(policy.at(target->policy.logits),cursor);
    auto draft=take(state.store.begin(continued.version),"real Pro target-only continuation");
    check(draft.replace(state.entry(kPending),sampled[1]),
        "actual target categorical pending after residual/bonus consumption");
    check(draft.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&next_cursor,1))),
        "target-only draw belongs to its successful full Head");
    continued.version=take(draft.finalize(),"original complete Pro next target pending and RNG");
    if(observed_policy)*observed_policy=std::move(policy);
    return continued;
  }

  Candidate verify_and_prepare(const c::StateVersion &base,const c::StateVersion &first_checkpoint,
      std::span<const c::ReadLease> q,Cursor cursor,Grant grant,Limits limits,
      std::span<const c::ReadLease> external_checked,std::vector<std::int32_t> &emitted,
      std::uint64_t iteration) {
    preflight(base,first_checkpoint,q,cursor,grant,limits);
    auto verified=advance_impl(first_checkpoint,second,iteration,{},nullptr,true);
    Candidate candidate;
    if(!target)candidate=prepare_verified_impl(base,first_checkpoint,verified.version,external_checked,q,cursor,grant,limits,emitted,iteration);
    else {
      const std::array input{state.member(first_checkpoint,kTargetTap),state.member(verified.version,kTargetTap)};
      auto checked=verifier_program.execute(queue,input);
      candidate=prepare_verified_impl(base,first_checkpoint,verified.version,checked,q,cursor,grant,limits,emitted,iteration);
      candidate.checked=std::move(checked);
    }
    candidate.advancement=std::move(verified);
    return candidate;
  }

  Candidate prepare_verified(const c::StateVersion &base,const c::StateVersion &first_checkpoint,
      const c::StateVersion &verified,std::span<const c::ReadLease> checked,
      std::span<const c::ReadLease> q,Cursor cursor,Grant grant,Limits limits,
      std::vector<std::int32_t> &emitted,std::uint64_t iteration) {
    preflight(base,first_checkpoint,q,cursor,grant,limits);
    return prepare_verified_impl(base,first_checkpoint,verified,checked,q,cursor,grant,limits,emitted,iteration);
  }

 private:
  Candidate prepare_verified_impl(const c::StateVersion &base,const c::StateVersion &first_checkpoint,
      const c::StateVersion &verified,std::span<const c::ReadLease> checked,
      std::span<const c::ReadLease> q,Cursor cursor,Grant grant,Limits limits,
      std::vector<std::int32_t> &emitted,std::uint64_t iteration) {
    if(sampler)check_cursor(verified,cursor);
    const auto quota=(std::min)(limits.quota,grant.additional);
    const std::array inputs{state.scalar(base,kPending),state.scalar(first_checkpoint,kPending)};
    Selection selected;
    if(sampler) {
      if(verifier.logits>=checked.size())throw std::runtime_error("actual Pro verifier logits are absent");
      const std::array input{checked[verifier.logits],q[0],q[1],identity(cursor,1),identity(cursor,2)};
      const auto output=sampler->decision.execute(queue,input);
      if(output.size()!=3||!flag(output[2]))throw std::runtime_error("actual original Pro p/q/bonus decision is invalid");
      selected.sampled_accept=flag(output[0]);
      const auto id=values<std::int32_t>(output[1]);
      if(id.size()!=1||id[0]<0||id[0]>=model::kVocabulary)
        throw std::runtime_error("original Pro random result outside vocabulary");
      const std::array flags{std::uint8_t(selected.sampled_accept)};
      selected.prefix=hyle::inference::mtp::select_sampled_prefix(inputs,flags,id,quota,limits.eos,
          std::int32_t(model::kVocabulary),emitted);
      selected.pending=selected.sampled_accept&&selected.prefix.consumed==1?q[1]:output[1];
    } else {
      if(verifier.ids>=checked.size())throw std::runtime_error("actual Pro verifier IDs are absent");
      const auto ids=values<std::int32_t>(checked[verifier.ids]);
      selected.prefix=hyle::inference::mtp::select_greedy_prefix(inputs,ids,quota,limits.eos,
          std::int32_t(model::kVocabulary),emitted);
      const auto port=verifier.pending[selected.prefix.consumed_drafts];
      if(port>=checked.size())throw std::runtime_error("actual Pro verifier pending lease is absent");
      selected.pending=checked[port];
    }
    auto checkpoint=selected.prefix.consumed_drafts?verified:first_checkpoint;
    std::optional<StepExecution> repair;
    if(target&&selected.prefix.consumed_drafts) {
      const auto before=target->repair.program.native_submissions();
      repair=execute_step(state,first_checkpoint,target->repair.step,source,iteration,
          [&](std::span<const c::ReadLease> input){return target->repair.program.execute(queue,input,cuda);},
          state.member(first_checkpoint,kTargetTap));
      ++counts.feedback;++counts.skips;++counts.repairs;
      add_bound(counts.repair_submissions,target->repair.program.native_submissions()-before);
      TargetFrame frame;
      const std::array entries{kTargetMain,kTargetCursor,kTargetTap};
      for(std::size_t index=0;index<entries.size();++index)frame.state[index]=state.member(verified,entries[index]);
      checkpoint=attach_target(state,repair->version,frame);
    }
    auto draft=take(state.store.begin(checkpoint),"all-resource K1 Pro acceptance");
    check(draft.replace(state.entry(kPending),selected.pending),"direct actual target policy result lease");
    if(sampler) {
      const auto next=cursor.advanced(3);
      check(draft.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&next,1))),
          "same complete Pro transaction advances RNG");
    }
    auto version=take(draft.finalize(),"complete K1 Pro accepted checkpoint");
    return {verified,{},std::move(selected),std::move(checkpoint),std::move(version),{},std::move(repair)};
  }

  Advance advance_impl(const c::StateVersion &from,Stage stage,std::uint64_t iteration,
      const c::ReadLease &target_input,c::ReadLease *first_logits,bool skipped) {
    Advance result;result.step=&stage.step;
    result.execution=execute_step(state,from,stage.step,source,iteration,
        [&](std::span<const c::ReadLease> input){return stage.program.execute(queue,input,cuda);},
        target_input,first_logits);
    ++counts.feedback;
    if(skipped)++counts.skips;
    result.version=result.execution.version;
    if(target) {
      result.target=execute_target(target->step,target->program,target->source,owner,queue,cuda,
          state.member(from,kPending),state.member(from,kTargetMain),state.member(from,kTargetCursor),target->upstream);
      result.version=attach_target(state,result.version,*result.target);
    }
    return result;
  }

  void check_cursor(const c::StateVersion &from,Cursor cursor) const {
    if(!state.sampling_entry)throw std::runtime_error("random Pro step has no complete sampling state");
    const auto bytes=read(state.physical_member(from,*state.sampling_entry));
    Cursor actual;
    if(bytes.size()!=sizeof(actual))throw std::runtime_error("Pro random cursor metadata extent");
    std::memcpy(&actual,bytes.data(),sizeof(actual));
    if(actual!=cursor)throw std::runtime_error("Pro step cursor differs from its actual complete state");
  }

  static bool flag(const c::ReadLease &lease) {
    const auto value=values<std::uint8_t>(lease);
    if(value.size()!=1||value[0]>1)throw std::runtime_error("Pro sampled flag must be actual Bool[1]");
    return value[0]!=0;
  }

  c::ReadLease identity(Cursor cursor,std::uint64_t offset) const {
    const auto key=cursor.identity(offset);
    return upload(owner,std::as_bytes(std::span(key)));
  }

  std::vector<c::ReadLease> sample(const c::ReadLease &logits,Cursor cursor) {
    const std::array input{logits,identity(cursor,0)};
    auto result=sampler->proposal.execute(queue,input);
    if(result.size()!=3||!flag(result[2]))throw std::runtime_error("actual Pro sampling distribution is invalid");
    return result;
  }

  void preflight(const c::StateVersion &base,const c::StateVersion &first_checkpoint,
      std::span<const c::ReadLease> q,Cursor cursor,Grant grant,Limits limits) const {
    if(grant.cancelled||!grant.additional||grant.reserved)
      throw std::runtime_error("GLM K1 step requires new uncancelled output credit");
    limits.validate(model::kVocabulary);
    if(limits.remaining_context<2)throw std::runtime_error("GLM K1 verifier requires its two actual context rows");
    if(sampler) {
      check_cursor(base,cursor);check_cursor(first_checkpoint,cursor);
      (void)cursor.advanced(3);
      if(q.size()!=3)throw std::runtime_error("GLM K1 proposal needs its actual q outputs");
    }
  }
};

// A synchronous mathematical consumer of the common complete-step ordering.
// All resources remain in the original Runtime/programs; only completed result
// leases live here until the scenario explicitly observes them. There is no
// pending successor, deferred cancellation or model Head publication.
template<class Runtime,class ObserveAdvance,class ObservePolicy>
struct GlmOrdinaryStepDriver final {
  using Steps=GlmStepRuntime<Runtime>;
  using Position=hyle::inference::StepPosition;
  using Cursor=hyle::inference::SamplingCursor;
  static constexpr bool kSupportsSuccessor=false;

  Steps &steps;
  const c::StateVersion &from;
  typename Steps::Stage stage;
  const c::ReadLease &target_input;
  Cursor cursor;
  ObserveAdvance &observe_advance;
  ObservePolicy &observe_policy;
  Cursor next_cursor{};
  std::optional<typename Steps::Advance> actual;
  std::vector<c::ReadLease> policy;
  c::StateVersion result;

  void prepare(Position) {next_cursor=steps.prepare_ordinary(from,cursor);}
  void submit(Position) {actual=steps.ordinary(from,stage,8,target_input,cursor,next_cursor,&policy);}
  bool cancelled() const noexcept {return false;}
  void cancel(bool) {throw std::logic_error("synchronous GLM ordinary steps have no deferred cancellation");}
  void finish(Position) {
    observe_advance(*actual);
    observe_policy(actual->version,policy);
    result=std::move(actual->version);
    actual.reset();policy.clear();
  }
  hyle::inference::RetirementObservation retire_failure(hyle::inference::EvaluationFailure *failure) noexcept {
    hyle::inference::RetirementObservation observed{true,{}};
    if(failure&&failure->retirement_observed())observed.status=failure->retirement().status;
    const auto retire=[&](Native &program) noexcept {
      // Match the original finite GLM retirement: only unfinished original
      // native work is drained; successful idle Invocations remain reusable.
      if(program.invocation.in_flight())
        hyle::inference::record_retirement_status(observed.status,program.invocation.drain());
      observed.retired=observed.retired&&!program.invocation.in_flight();
    };
    for(auto &phase:stage.program.phases)retire(phase);
    if(steps.target)for(auto &phase:steps.target->program.phases)retire(phase);
    if(steps.sampler) {
      if(steps.target)retire(steps.target->policy_program);
      retire(steps.sampler->proposal);
    }
    return observed;
  }
  void recover_records(bool retired) noexcept {
    // These are unpublished mathematical results, not request commit records.
    // The original caller still owns every Head and authoritative record.
    if(retired) {actual.reset();policy.clear();}
  }
};


struct GlmRequestDirectory final {
  using Progress=hyle::inference::RequestProgress;
  using Key=hyle::inference::RequestKey;
  using Cursor=hyle::inference::SamplingCursor;
  struct Commit final {
    bool committed=false,stopped=false;
    std::array<std::int32_t,1> emitted{};
    std::int32_t cursor=0,index_cursor=0,lag=0;
    std::uint64_t epoch=0;
    Cursor sampling;
  };
  State &state;
  const c::StateVersion &base;
  std::span<g::Invocation *const> originals;
  hyle::inference::RequestLoop loop;
  std::array<Key,2> keys;
  std::array<std::optional<Progress>,2> progress;
  std::array<hyle::inference::GenerationTurn,2> turns;
  std::array<std::array<std::int32_t,1>,2> inputs{};
  std::array<c::StateHead,2> heads;
  std::array<GlmCheckpoint,2> parked;
  std::array<Commit,2> records0;
  std::array<Commit,1> records1;
  std::array<std::size_t,2> committed{},observed{};
  std::array<std::size_t,4> order{};
  std::size_t admissions=0,publications=0,projections=0;
  hyle::inference::PrefixDirectory<GlmCheckpoint> directory{1};
  hyle::inference::CacheDomain domain;
  hyle::inference::TokenPlanRef prefix_input;
  hyle::inference::PrefixDirectory<GlmCheckpoint>::Hit hit;
  std::array<std::int32_t,2> consumed_inputs{};
  std::size_t evictions=0;
  hyle::Status native_status;

  GlmRequestDirectory(State &actual,const c::StateVersion &initial,
      std::span<g::Invocation *const> native,hyle::inference::CacheDomain identity,Cursor rng)
      :state(actual),base(initial),originals(native),domain(identity) {
    turns[0]={{state.scalar(base,kPending)},1};turns[1]={{},1};
    parked[0]={base,0,state.scalar(base,kCursor)};
    for(std::size_t at=0;at<progress.size();++at) {
      progress[at].emplace(2);progress[at]->bind_turns(turns);
      progress[at]->initial_sampling={rng.seed,rng.stream+at,0};
      inputs[at][0]=state.scalar(base,kPending);
    }
    loop.prepare(2,1);
    for(std::size_t at=0;at<keys.size();++at)keys[at]=loop.enqueue(at);
  }
  std::span<Commit> records(std::size_t index) {
    return index?std::span<Commit>(records1):std::span<Commit>(records0);
  }
  hyle::inference::RetirementObservation native_retirement(bool abort) noexcept {
    bool retired=true;
    for(auto *original:originals) {
      if(abort&&original->in_flight())
        hyle::inference::record_retirement_status(native_status,original->drain());
      retired=retired&&!original->in_flight();
    }
    return {retired,native_status};
  }
  void admit(Key key) {
    const auto index=key.slot;
    require_observed(index<2&&admissions<order.size(),"finite GLM admission identity/capacity");
    auto &p=*progress[index];
    require_observed(!heads[1-index].valid(),"N1 GLM admits one real resident Head");
    if(!heads[index].valid()) {
      if(index==1&&!p.turn_index) {
        require_observed(bool(prefix_input),"new GLM request waits for the real complete prefix");
        const auto length=prefix_input->ids.size();
        hit=directory.lookup(domain,*prefix_input,length,length);
        require_observed(bool(hit),"new GLM request really hits the original complete boundary");
        parked[index]=hit.payload;
      }
      heads[index]=take(c::StateHead::create(state.store,parked[index].version),"admit actual GLM complete Head");
    }
    if(!p.turn_started()) {
      const auto pending=state.scalar(take(heads[index].current(),"actual admitted GLM pending"),kPending);
      const auto input=p.begin_turn(inputs[index],p.turn_index?std::optional<std::int32_t>(pending):std::nullopt);
      require_observed(input.size()==1&&(!p.turn_index?input[0]==state.scalar(base,kPending):input[0]==pending),
          "GLM first input is the matched suffix; later input is its own actual pending");
      if(p.turn_index)require_observed(p.generated.back()==pending&&p.turn_output_begin==1,
          "later GLM turn retains its exact pending-only input boundary");
    }
    order[admissions++]=index;
  }
  void publish_turn(Key key) {
    const auto version=take(heads[key.slot].current(),"complete GLM turn checkpoint");
    const auto cursor=state.scalar(version,kCursor);
    parked[key.slot]={version,std::size_t(cursor-kBasePosition),cursor};
    if(!key.slot&&!progress[0]->turn_index) {
      const auto length=parked[0].consumed;
      require_observed(length&&length<=consumed_inputs.size(),"GLM prefix has its actual consumed suffix extent");
      prefix_input=hyle::inference::token_plan(std::span(consumed_inputs).first(length),std::size_t(kCapacity-kBasePosition));
      require_observed(directory.publish(domain,prefix_input,length,parked[0]),"publish real complete GLM prefix");
      require_observed(directory.retains([&](const GlmCheckpoint &entry){return entry.version.identity()==version.identity();}),
          "typed GLM directory retains the original complete boundary");
    }
  }
  hyle::inference::RetirementObservation retire(Key key,hyle::inference::RequestRetirement reason) {
    auto result=native_retirement(false);
    if(!result.retired)return result;
    const bool resident=heads[key.slot].valid();heads[key.slot]={};
    if(reason==hyle::inference::RequestRetirement::kEvicted)evictions+=resident;
    else parked[key.slot]={};
    hyle::inference::record_retirement_status(result.status,state.store.collect());
    return result;
  }
  template<class Execute>
  hyle::inference::RequestPumpResult pump(Execute &execute) {
    struct Adapter final {
      GlmRequestDirectory &owner;Execute &execute;
      Progress &progress(Key key)const noexcept{return *owner.progress[key.slot];}
      bool resident(Key key)const noexcept{return owner.heads[key.slot].valid();}
      void admission_preflight(Key key)const {
        require_observed(key.slot<2&&owner.native_retirement(false).retired,"GLM admission needs retired original programs");
      }
      void admit(Key key,std::span<const Key>)const{owner.admit(key);}
      void publish_turn(Key key)const{owner.publish_turn(key);}
      hyle::inference::RetirementObservation retire(Key key,hyle::inference::RequestRetirement reason)const{return owner.retire(key,reason);}
      hyle::inference::RequestWaveResult execute_wave(hyle::inference::RequestWave wave,std::span<std::size_t> remaining)const {
        struct Stage final {
          GlmRequestDirectory &owner;Execute &execute;
          std::size_t pending_count()const noexcept{return 0;}
          hyle::inference::RequestWavePendingMember pending_member(std::size_t)const noexcept{return {};}
          hyle::inference::RequestWaveAdvance advance(hyle::inference::RequestWaveAction action,hyle::inference::RequestWave actual)const {
            if(action==hyle::inference::RequestWaveAction::kDrain)return {.remaining=owner.native_retirement(false)};
            require_observed(action==hyle::inference::RequestWaveAction::kStart&&actual.keys.size()==1,
                "finite GLM synchronous wave retains its real N1 cohort");
            require_observed(actual.credits[0].additional==1&&!actual.credits[0].reserved,
                "finite GLM turn has one actual output credit");
            execute(actual.keys[0],actual.credits[0]);
            return {.advanced=true,.remaining=owner.native_retirement(false)};
          }
          hyle::inference::RetirementObservation retire(hyle::inference::RequestWaveRetirement)const noexcept {
            // An observation after a failed drain still drives the same
            // original owners. Successful idle programs are never closed.
            return owner.native_retirement(true);
          }
        } stage{owner,execute};
        return hyle::inference::advance_request_wave(wave,remaining,stage);
      }
    } adapter{*this,execute};
    return loop.pump_once(adapter);
  }
};

template<class Runtime=OfficialRuntime>
void run(c::Context &context,const c::StorageOwner &owner,bool cuda,bool use_target=false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ,const ConvertedFixture *packed_fixture=nullptr,bool stochastic=false,
    hyle::inference::mtp::SamplingLimits limits={},std::size_t maximum_drafts=1,bool request_directory_demo=false) {
  if(maximum_drafts<1||maximum_drafts>2||(!stochastic&&maximum_drafts!=1))
    throw std::runtime_error("finite Pro random component supports one or two configured drafts");
  if(stochastic)limits.validate(model::kVocabulary);
  if(stochastic&&!use_target)throw std::runtime_error("random Pro K1 requires its actual target verifier");
  if(request_directory_demo&&(Runtime::kPacked||!stochastic||!use_target||maximum_drafts!=1))
    throw std::runtime_error("request directory demonstration requires the original stochastic K1 fixture");
  if(request_directory_demo&&(limits.quota!=std::numeric_limits<std::size_t>::max()||
      limits.remaining_context!=std::numeric_limits<std::size_t>::max()||!limits.eos.empty()))
    throw std::runtime_error("request directory demonstration uses the original unrestricted K1 fixture limits");
  auto queue=take(context.create_queue(),"ordinary original Pro queue");
  using Resident=ResidentRun<typename Runtime::ResidentSource>;
  using Tap=typename Runtime::Tap;
  const auto layout=Runtime::layout();
  const auto resource_count=layout.resources.size();
  auto source=[&] {
    if constexpr(Runtime::kPacked)return Runtime::prepare_source(context,owner,queue,cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
        ,device
#endif
        ,packed_fixture);
    else {
      if(packed_fixture)throw std::runtime_error("a converted fixture cannot replace an official source");
      return Runtime::prepare_source(context,owner,queue,cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
          ,device
#endif
          );
    }
  }();
  std::shared_ptr<typename Runtime::TargetSource> target_source;
  if(use_target)target_source=Runtime::prepare_target_source(*source,context,owner,queue,cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  // Hold every ordinary Invocation inside this scope, before closing the
  // shared bounded supplier. No completed Invocation retains a compact slot.
  [&] {
    auto initializer=[&] {
      if constexpr(Runtime::kPacked)return Runtime::build_initial(*source);
      else return Runtime::build_initial();
    }();
    auto initial_program=Native::create(initializer.step.recipe,context,owner,cuda,1);
    const auto keys=Runtime::fixture(std::size_t(kBasePosition*model::kIndexDepth),7);
    const auto key_lease=upload(owner,sem::bytes(keys));
    const std::array initial_input{key_lease};
    auto initial_output=initial_program.execute(queue,initial_input);
    const auto initial_actual=[&](Value value){return observed(initializer.step,initial_input,initial_output,value);};
    Runtime::check_initial(initializer,keys,initial_actual);
    auto members=Runtime::initial_members(owner,initializer,initial_output);

    auto first_step=build_step(*source,true,std::int32_t(kBasePosition));
    auto second_step=build_step(*source,false,std::int32_t(kBasePosition+1));
    auto continuation_zero=build_step(*source,true,std::int32_t(kBasePosition+1));
    auto continuation_one=build_step(*source,true,std::int32_t(kBasePosition+2));
    auto skip_zero=build_step(*source,false,std::int32_t(kBasePosition+2));
    auto skip_one=build_step(*source,false,std::int32_t(kBasePosition+3));
    const auto resident=[&](const Step &step) {
      return Resident::create(step.recipe,Runtime::boundary(step),source,context,owner,cuda);
    };
    auto first_program=resident(first_step),second_program=resident(second_step);
    auto continuation_zero_program=resident(continuation_zero),continuation_one_program=resident(continuation_one);
    auto skip_zero_program=resident(skip_zero),skip_one_program=resident(skip_one);
    std::array<Resident*,6> model_programs{&first_program,&second_program,&continuation_zero_program,
        &continuation_one_program,&skip_zero_program,&skip_one_program};
    for(const auto *program:model_programs) {
      std::size_t heads=0;
      for(const auto &phase:program->phases) {
        const auto contains=std::any_of(phase.recipe.external.begin(),phase.recipe.external.end(),[&](const External &entry) {
          return entry.role.name==source->words.at(vocabulary::Role::kHead).name;
        });
        if(contains){report_head(phase,*source,cuda);++heads;}
      }
      if(heads!=1)throw std::runtime_error("Pro cold cut changed full-V prediction ownership");
    }
    auto catchup_step=build_catchup(*source);
    auto catchup_program=Native::create(catchup_step.recipe,context,owner,cuda);
    auto verifier=build_verifier(*source,use_target);
    auto verifier_program=Native::create(verifier.recipe,context,owner,cuda,use_target?2u:1u);
    std::optional<Step> repair_step;
    std::optional<Resident> repair_program,target_program;
    std::optional<TargetStep> target_step;
    std::optional<Verifier> policy;
    std::optional<Native> policy_program;
    c::ReadLease upstream_ids;
    std::optional<TargetFrame> warm;
    if(use_target) {
      repair_step=build_step(*source,false,std::int32_t(kBasePosition+1),true);
      repair_program=resident(*repair_step);
      target_step=build_target(*target_source,*source);
      target_program=Resident::create(target_step->step.recipe,Runtime::target_boundary(*target_step),
          target_source,context,owner,cuda,5);
      policy=build_verifier(*source,true,true);
      policy_program=Native::create(policy->recipe,context,owner,cuda,1);
      report_head(*policy_program,*source,cuda);
      std::vector<std::int32_t> selected(std::size_t(kSelected),0);std::iota(selected.begin(),selected.end(),0);
      upstream_ids=upload(owner,sem::bytes(selected));
      std::vector<float> history(std::size_t(kCapacity*model::kCacheDepth),std::numeric_limits<float>::quiet_NaN());
      const auto valid=fixture_bfloat(std::size_t((kBasePosition-1)*model::kCacheDepth),29);
      for(std::size_t i=0;i<valid.size();++i)history[i]=glm_shared_vocabulary_reference::detail::finite_bfloat(valid[i]);
      warm=execute_target(*target_step,*target_program,*target_source,owner,queue,cuda,
          upload(owner,sem::bytes(std::vector<std::int32_t>{0})),upload(owner,sem::bytes(history)),
          upload(owner,sem::bytes(std::vector<std::int32_t>{std::int32_t(kBasePosition-1)})),upstream_ids);
      observe_target(*target_step,*target_source,*source,*warm);
      warm->input={};warm->output.clear();
      members[layout.slot(kHidden)]=warm->state[2];
    }
    report_head(verifier_program,*source,cuda);

    std::optional<Native> random_q,random_decision;
    if(stochastic) {
      Recipe recipe;auto &f=recipe.fragment;f.profile=first_step.recipe.fragment.profile;f.region=0x474c4d5052515151ULL;
      const auto logits=f.input(sem::dense({1,model::kVocabulary}));
      const auto key=f.input(sem::dense({1,4},c::DataType::kUInt32));
      const auto q=model_sampling_fragment::append_proposal(f,logits,key);
      f.outputs={q.log_probabilities,q.proposal,q.valid};
      random_q=Native::create(std::move(recipe),context,owner,cuda,3);
      Recipe selected;auto &d=selected.fragment;d.profile=first_step.recipe.fragment.profile;d.region=0x474c4d5052444543ULL;
      const auto p=d.input(sem::dense({2,model::kVocabulary})),q_log=d.input(sem::dense({1,model::kVocabulary}));
      const auto proposal=d.input(sem::dense({1},c::DataType::kInt32));
      const auto accept=d.input(sem::dense({1,4},c::DataType::kUInt32)),final=d.input(sem::dense({1,4},c::DataType::kUInt32));
      const auto result=model_sampling_fragment::append_decision(d,p,q_log,proposal,accept,final);
      d.outputs={result.accepted,result.pending,result.valid};
      random_decision=Native::create(std::move(selected),context,owner,cuda,1);
    }
    std::optional<Step> repair_second_step,recursive_catchup_step,recursive_fresh_step;
    std::optional<Resident> repair_second_program,recursive_fresh_program;
    std::optional<Native> recursive_verifier,recursive_decision,recursive_catchup;
    if(maximum_drafts==2) {
      repair_second_step=build_step(*source,false,std::int32_t(kBasePosition+2),true);
      repair_second_program=resident(*repair_second_step);
      recursive_fresh_step=build_step(*source,true,std::int32_t(kBasePosition+3));
      recursive_fresh_program=resident(*recursive_fresh_step);
      recursive_catchup_step=build_catchup(*source,1);
      recursive_catchup=Native::create(recursive_catchup_step->recipe,context,owner,cuda,2);
      recursive_verifier=Native::create(build_prefix_verifier(*source),context,owner,cuda,1);
      Recipe selected;selected.fragment.profile=first_step.recipe.fragment.profile;selected.fragment.region=0x474c4d5052524432ULL;
      glm_stochastic_prefix_fragment::append(selected.fragment,model::kVocabulary);
      recursive_decision=Native::create(std::move(selected),context,owner,cuda,2);
    }

    std::size_t pools=initial_program.outputs.stats().capacity_bytes;
    for(const auto *program:model_programs)add_bound(pools,program->backing_capacity());
    add_bound(pools,catchup_program.outputs.stats().capacity_bytes);
    add_bound(pools,verifier_program.outputs.stats().capacity_bytes);
    if(stochastic){add_bound(pools,random_q->outputs.stats().capacity_bytes);add_bound(pools,random_decision->outputs.stats().capacity_bytes);}
    if(maximum_drafts==2) {
      add_bound(pools,repair_second_program->backing_capacity());add_bound(pools,recursive_fresh_program->backing_capacity());
      for(const auto *program:{&*recursive_verifier,&*recursive_decision,&*recursive_catchup})add_bound(pools,program->outputs.stats().capacity_bytes);
    }
    if(use_target) {
      add_bound(pools,repair_program->backing_capacity());add_bound(pools,target_program->backing_capacity());
      add_bound(pools,policy_program->outputs.stats().capacity_bytes);
    }
    std::size_t initial_bound=0;
    for(std::size_t i=0;i<resource_count;++i)
      if(i!=layout.slot(kCodes)&&(!layout.slots[kScales]||i!=*layout.slots[kScales])&&!(use_target&&i==layout.slot(kHidden)))
        add_bound(initial_bound,aligned_bound(members[i].size()));
    // Charge each real Program pool root ONCE, including all generations and
    // diagnostic outputs in that root. The independent initial buffers have
    // an aligned conservative bound; two encoded initial states share the
    // initializer root already counted above. Readonly weights are not state.
    constexpr std::size_t metadata_bound=16384;
    std::size_t budget=pools;add_bound(budget,initial_bound);add_bound(budget,metadata_bound);
    auto state=State::create(owner,budget,layout,use_target,stochastic);
    using SamplingCursor=hyle::inference::SamplingCursor;
    const SamplingCursor initial_random{0x474c4d50524fULL,17,0};
    const auto random_identity=[&](SamplingCursor cursor,std::uint64_t offset) {
      const auto key=cursor.identity(offset);return upload(owner,std::as_bytes(std::span(key)));
    };
    const auto random_cursor=[&](const c::StateVersion &from) {
      if(!state.sampling_entry)throw std::runtime_error("Pro random cursor was not declared");
      const auto data=read(state.physical_member(from,*state.sampling_entry));SamplingCursor result;
      if(data.size()!=sizeof(result))throw std::runtime_error("Pro random cursor metadata extent");
      std::memcpy(&result,data.data(),sizeof(result));return result;
    };
    const auto random_valid=[&](const c::ReadLease &from) {
      const auto data=values<std::uint8_t>(from);
      if(data.size()!=1||data[0]>1)throw std::runtime_error("Pro sampled flag must be actual Bool[1]");return data[0]!=0;
    };
    const auto sample_logits=[&](const c::ReadLease &logits,SamplingCursor cursor) {
      const std::array input{logits,random_identity(cursor,0)};
      auto result=random_q->execute(queue,input);
      if(!random_valid(result[2]))throw std::runtime_error("actual Pro sampling distribution is invalid");
      return result;
    };
    const auto target_before=use_target?warm->state[2]:upload(owner,sem::bytes(Runtime::fixture(std::size_t(model::kHidden),11)));
    const std::array next_targets{
        use_target?c::ReadLease{}:upload(owner,sem::bytes(Runtime::fixture(std::size_t(model::kHidden),13))),
        use_target?c::ReadLease{}:upload(owner,sem::bytes(Runtime::fixture(std::size_t(model::kHidden),17)))};
    std::vector<c::ReadLease> checked,policy_checked;
    std::vector<std::int32_t> target_ids;
    bool full_target_checked=false;
    const auto check_verifier=[&](const Verifier &definition,std::span<const c::ReadLease> outputs,
        std::span<const Tap> taps,bool full) {
      const auto logits=values<float>(outputs[definition.logits]);
      const auto scores=values<float>(outputs[definition.scores]);
      const auto ids=values<std::int32_t>(outputs[definition.ids]);
      require_observed(Runtime::check_logits(*source,logits,taps,full,ids,cuda),"actual full-V target projection");
      require_observed(glm_shared_vocabulary_reference::expect_greedy(logits,ids,std::size_t(model::kVocabulary),scores),
          "actual full-V target stable greedy");
      for(std::size_t row=0;row<ids.size();++row)
        EXPECT_EQ(values<std::int32_t>(outputs[definition.pending[row]]).at(0),ids[row]);
      const auto top2=glm_shared_vocabulary_reference::detail::candidates(std::span<const float>(logits).first(std::size_t(model::kVocabulary)));
      EXPECT_EQ(values<std::int32_t>(outputs[definition.reject]).at(0),std::int32_t(top2.at(1)));
      return ids;
    };
    if(!use_target) {
      const auto taps=Runtime::fixture(std::size_t(2*model::kHidden),19);
      const std::array input{upload(owner,sem::bytes(taps))};
      checked=verifier_program.execute(queue,input);
      target_ids=check_verifier(verifier,checked,taps,true);full_target_checked=true;policy_checked=checked;
    }
    std::size_t repairs=0,repair_submissions=0;
    std::size_t feedback=0,catchups=0,skips=0,cancelled=0,stale_publish=0;
    std::array<std::size_t,2> accepted{};
    std::uint32_t natural_accepted=0;std::int32_t natural_proposal=0;
    std::vector<std::int32_t> selection_emitted;selection_emitted.reserve(2);
    using ModelSteps=GlmStepRuntime<Runtime>;
    std::optional<typename ModelSteps::Target> step_target;
    std::optional<typename ModelSteps::Sampler> step_sampler;
    if(use_target)step_target.emplace(typename ModelSteps::Target{*target_source,*target_step,*target_program,
        upstream_ids,*policy,*policy_program,{*repair_step,*repair_program}});
    if(stochastic)step_sampler.emplace(typename ModelSteps::Sampler{*random_q,*random_decision});
    ModelSteps model_steps{state,owner,queue,*source,cuda,step_target?&*step_target:nullptr,
        step_sampler?&*step_sampler:nullptr,{second_step,second_program},verifier,verifier_program,
        {feedback,skips,repairs,repair_submissions}};
    const auto observe_advance=[&](typename ModelSteps::Advance &actual) {
      observe_step_execution(state,*actual.step,*source,actual.execution);
      if(actual.target)observe_target(*target_step,*target_source,*source,*actual.target);
      actual.execution={};actual.target.reset();
    };
    const auto observe_candidate=[&](typename ModelSteps::Candidate &actual) {
      if(actual.advancement)observe_advance(*actual.advancement);
      if(actual.repair)observe_step_execution(state,*repair_step,*source,*actual.repair);
      actual.advancement.reset();actual.repair.reset();
    };
    {
      Epoch initial_epoch{6,std::uint64_t(kBasePosition-1),0};
      std::vector<c::StateMember> initial(state.entries.size());
      for(std::size_t i=0;i<resource_count;++i)initial[i]={state.entries[i],members[i]};
      initial[layout.epoch()]={state.entry(kEpoch),{},std::as_bytes(std::span(&initial_epoch,1))};
      if(use_target) {
        const std::array entries{kTargetMain,kTargetCursor,kTargetTap};
        for(std::size_t i=0;i<entries.size();++i)initial[layout.slot(entries[i])]={state.entry(entries[i]),warm->state[i]};
      }
      if(stochastic)initial[*state.sampling_entry]={state.entries[*state.sampling_entry],{},std::as_bytes(std::span(&initial_random,1))};
      const auto base=take(state.store.import(initial),"import complete original Pro continuation");
      const auto base_image=image(state,base);

      // Only finite numeric/control observations stay here. Both request and
      // ordinary scenarios call the same actual borrowed model steps.
      const auto check_current=[&](const c::StateVersion &first) {
        if(use_target&&!stochastic) {
          const std::array input{state.member(first,kTargetTap)};
          policy_checked=policy_program->execute(queue,input);
          (void)check_verifier(*policy,policy_checked,values<Tap>(input[0]),false);
        }
      };
      const auto check_verified=[&](const c::StateVersion &first,const c::StateVersion &second) {
        if(use_target) {
          auto taps=values<Tap>(state.member(first,kTargetTap));
          const auto tail=values<Tap>(state.member(second,kTargetTap));
          taps.insert(taps.end(),tail.begin(),tail.end());
          target_ids=check_verifier(verifier,checked,taps,!full_target_checked);full_target_checked=true;
        }
      };
      const auto execute_checked_ordinary=[&](const c::StateVersion &from,typename ModelSteps::Stage stage,
          const c::ReadLease &target_input,SamplingCursor cursor) {
        const auto observe_policy=[&](const c::StateVersion &continued,std::span<const c::ReadLease> policy_observation) {
          if(use_target)EXPECT_EQ(state.scalar(continued,kTargetCursor),state.scalar(continued,kCursor));
          if(stochastic)(void)check_verifier(*policy,policy_observation,values<Tap>(state.member(continued,kTargetTap)),false);
        };
        GlmOrdinaryStepDriver<Runtime,decltype(observe_advance),decltype(observe_policy)> driver{
            model_steps,from,stage,target_input,cursor,observe_advance,observe_policy};
        EXPECT_EQ(hyle::inference::drive_steps(driver),hyle::inference::StepDriveResult::kCompleted);
        return std::move(driver.result);
      };
      const auto native_step_counts=[&] {
        return std::array{source->checked_steps,second_program.native_submissions(),
            target_program?target_program->native_submissions():std::size_t{},
            repair_program?repair_program->native_submissions():std::size_t{},verifier_program.submissions,
            random_decision?random_decision->submissions:std::size_t{}};
      };
      const auto check_step_admission=[&](const c::StateVersion &first_checkpoint,
          std::span<const c::ReadLease> q,hyle::inference::RequestCreditGrant grant) {
        const auto before=native_step_counts();
        const auto versions=state.store.stats().versions;
        for(const auto refused:{hyle::inference::RequestCreditGrant{},
            hyle::inference::RequestCreditGrant{.additional=2,.cancelled=true},
            hyle::inference::RequestCreditGrant{.reserved=1,.additional=1}}) {
          std::vector<std::int32_t> ignored;
          EXPECT_THROW((void)model_steps.verify_and_prepare(base,first_checkpoint,q,initial_random,
              refused,limits,checked,ignored,7),std::runtime_error);
          EXPECT_TRUE(ignored.empty());EXPECT_EQ(native_step_counts(),before);
          EXPECT_EQ(state.store.stats().versions,versions);EXPECT_EQ(state.store.stats().drafts,0u);
        }
        if(stochastic) {
          std::vector<std::int32_t> ignored;
          EXPECT_THROW((void)model_steps.verify_and_prepare(base,first_checkpoint,q,initial_random.advanced(1),
              grant,limits,checked,ignored,7),std::runtime_error);
          EXPECT_TRUE(ignored.empty());EXPECT_EQ(native_step_counts(),before);
          EXPECT_EQ(state.store.stats().versions,versions);EXPECT_EQ(state.store.stats().drafts,0u);
        }
        ::testing::Test::RecordProperty("model_step_credit_preflight_rejections",3);
        ::testing::Test::RecordProperty("model_step_preflight_native_submissions",0);
        ::testing::Test::RecordProperty("model_step_stale_cursor_rejected",stochastic?1:0);
      };
      const auto single_current=[&](const c::StateVersion &first,hyle::inference::mtp::SamplingLimits boundary,bool terminal_control) {
        boundary.validate(model::kVocabulary);
        const auto current_id=state.scalar(base,kPending);
        const auto before_nextn=source->checked_steps,before_target=target_source->checked_steps;
        const auto before_second=second_program.native_submissions();
        const std::array input{state.member(first,kTargetTap)};
        const auto target=policy_program->execute(queue,input);
        (void)check_verifier(*policy,target,values<Tap>(input[0]),false);
        const auto sampled=sample_logits(target[policy->logits],initial_random);
        const auto id=values<std::int32_t>(sampled[1]).at(0);const std::array terminal{id};
        if(terminal_control)boundary.eos=terminal;
        const std::array inputs{current_id},finals{id};std::vector<std::int32_t> emitted;
        const auto selected=hyle::inference::mtp::select_sampled_prefix(inputs,{},finals,boundary.quota,boundary.eos,
            std::int32_t(model::kVocabulary),emitted);
        auto head=take(c::StateHead::create(state.store,base),"single target Pro bounded Head");
        auto draft=take(state.store.begin(first),"complete mandatory-current Pro prefix");
        check(draft.replace(state.entry(kPending),sampled[1]),"actual Pro target-only sampled pending lease");
        const auto rng=initial_random.advanced(1);
        check(draft.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&rng,1))),"one target draw with complete Pro state");
        const auto published=take(draft.finalize(),"single Pro target complete checkpoint");
        EXPECT_FALSE(head.publish(first.identity(),published).ok());EXPECT_EQ(random_cursor(take(head.current(),"single Pro stale cursor")),initial_random);
        check(head.publish(base.identity(),published),"single Pro target result publishes once");
        EXPECT_EQ(selected.consumed,1u);EXPECT_EQ(selected.consumed_drafts,0u);EXPECT_EQ(selected.stopped,terminal_control||std::ranges::find(boundary.eos,id)!=boundary.eos.end());
        same_storage(state.member(published,kPending),sampled[1]);EXPECT_EQ(random_cursor(published),rng);
        EXPECT_EQ(state.scalar(published,kCursor),std::int32_t(kBasePosition+1));EXPECT_EQ(state.scalar(published,kTargetCursor),state.scalar(published,kCursor));
        for(std::size_t slot=0;slot<resource_count;++slot)if(slot!=layout.slot(kPending))same_storage(state.physical_member(published,slot),state.physical_member(first,slot));
        for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})same_storage(state.member(published,entry),state.member(first,entry));
        EXPECT_EQ(source->checked_steps,before_nextn);EXPECT_EQ(target_source->checked_steps,before_target);EXPECT_EQ(second_program.native_submissions(),before_second);
        std::printf("GLM PRO RANDOM TAIL context=%zu quota=%zu emitted=%d stopped=%d consumed=1 draw=1 second_submissions=0\n",boundary.remaining_context,boundary.quota,id,int(selected.stopped));
      };
      const auto exercise_acceptance_cancel=[&](c::StateHead &head,const c::StateVersion &checkpoint,
          const c::StateVersion &first,const c::ReadLease &pending) {
        {
          auto discarded=take(state.store.begin(checkpoint),"cancelled Pro acceptance draft");
          check(discarded.replace(state.entry(kPending),pending),"cancelled actual pending lease");
          if(stochastic) {
            const auto next=initial_random.advanced(3);
            check(discarded.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&next,1))),"cancel uncommitted Pro random cursor");
            EXPECT_FALSE(discarded.replace(state.entry(kPending),state.member(first,kTargetTap)).ok());
          }
          check(discarded.cancel(),"cancel whole uncommitted Pro acceptance");++cancelled;
          EXPECT_EQ(image(state,take(head.current(),"cancelled Pro draft cannot change Head")),base_image);
          if(stochastic)EXPECT_EQ(random_cursor(take(head.current(),"Pro cancellation retains RNG")),initial_random);
        }
      };
      const auto exercise_prefix_policies=[&](const c::StateVersion &first_checkpoint,
          const c::StateVersion &checkpoint,std::int32_t proposal,const hyle::inference::mtp::AcceptedPrefix &decision,bool sampled_accept,
          const c::ReadLease &pending,const c::ReadLease &draft_token) {

        const std::array inputs{state.scalar(base,kPending),proposal};
        const std::array flags{std::uint8_t(sampled_accept)};
        const std::array finals{values<std::int32_t>(pending).at(0)};
        // Actual accepted-prefix resources and real ID leases are reused
        // by independent policy Heads; no logits, q or acceptance changed.
        const auto boundary=[&](std::size_t quota,std::span<const std::int32_t> eos,const char *name) {
          std::vector<std::int32_t> tokens;
          const auto policy=hyle::inference::mtp::select_sampled_prefix(inputs,flags,finals,quota,eos,std::int32_t(model::kVocabulary),tokens);
          const auto lease=sampled_accept&&policy.consumed==1?draft_token:pending;
          const auto from=policy.consumed_drafts?checkpoint:first_checkpoint;
          auto branch=take(c::StateHead::create(state.store,base),"Pro independent bounded policy Head");
          auto draft=take(state.store.begin(from),"Pro same complete selected prefix");
          check(draft.replace(state.entry(kPending),lease),"Pro bounded original pending lease");
          const auto rng=initial_random.advanced(3);
          check(draft.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&rng,1))),"Pro bounded policy preserves reserved draws");
          const auto bounded=take(draft.finalize(),"Pro bounded same-prefix checkpoint");
          check(branch.publish(base.identity(),bounded),"Pro bounded actual lease/state publication");
          same_storage(state.member(bounded,kPending),lease);EXPECT_EQ(state.scalar(bounded,kPending),policy.next_id);
          EXPECT_EQ(state.scalar(bounded,kCursor),std::int32_t(kBasePosition+policy.consumed));EXPECT_EQ(state.scalar(bounded,kTargetCursor),state.scalar(bounded,kCursor));
          EXPECT_EQ(random_cursor(bounded),rng);
          for(std::size_t slot=0;slot<resource_count;++slot)if(slot!=layout.slot(kPending))same_storage(state.physical_member(bounded,slot),state.physical_member(from,slot));
          for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})same_storage(state.member(bounded,entry),state.member(from,entry));
          std::printf("GLM PRO RANDOM BOUNDARY name=%s accept=%d emitted=%zu consumed=%zu pending=%d stopped=%d draw=3\n",name,int(sampled_accept),tokens.size(),policy.consumed,policy.next_id,int(policy.stopped));
        };
        if(limits.quota==std::numeric_limits<std::size_t>::max()&&limits.remaining_context==std::numeric_limits<std::size_t>::max()&&limits.eos.empty()) {
          boundary(1,{},"quota1");check(state.store.collect(),"retire independent Pro quota Head");
          if(sampled_accept&&proposal!=inputs[0]){const std::array eos{proposal};boundary(2,eos,"accepted_proposal_EOS");check(state.store.collect(),"retire independent Pro accepted EOS Head");}
          if(finals[0]!=inputs[0]&&(!sampled_accept||finals[0]!=proposal)){const std::array eos{finals[0]};boundary(2,eos,sampled_accept?"bonus_EOS":"residual_EOS");check(state.store.collect(),"retire independent Pro final EOS Head");}
        } else {
          ::testing::Test::RecordProperty("random_finite_emitted",int(selection_emitted.size()));::testing::Test::RecordProperty("random_finite_consumed",int(decision.consumed));
          ::testing::Test::RecordProperty("random_finite_stopped",int(decision.stopped));::testing::Test::RecordProperty("random_committed_draws",3);
          return false; // Preserve the finite original policy result.
        }
        return true;
      };
      if(request_directory_demo) {
        c::StateVersion first;
        {
          std::vector<g::Invocation*> originals{&initial_program.invocation,&catchup_program.invocation,
              &verifier_program.invocation,&policy_program->invocation,&random_q->invocation,&random_decision->invocation};
          for(auto *program:model_programs)for(auto &phase:program->phases)originals.push_back(&phase.invocation);
          for(auto *program:{&*repair_program,&*target_program})
            for(auto &phase:program->phases)originals.push_back(&phase.invocation);
          GlmRequestDirectory driver(state,base,originals,{source.get(),&base},initial_random);
          auto &request_window=driver.loop;auto &request_keys=driver.keys;
          auto &request_progress=driver.progress;auto &request_commits=driver.committed;
          auto &request0_records=driver.records0;auto &request_order=driver.order;
          auto &request_turns=driver.admissions;auto &family_observations=driver.observed;
          auto &request_publications=driver.publications;auto &record_projections=driver.projections;
          const auto request_records=[&](std::size_t index){return driver.records(index);};
          const auto request_submission_counts=[&] {
            return std::array{initial_program.submissions,first_program.native_submissions(),second_program.native_submissions(),
                continuation_zero_program.native_submissions(),continuation_one_program.native_submissions(),
                skip_zero_program.native_submissions(),skip_one_program.native_submissions(),catchup_program.submissions,
                verifier_program.submissions,repair_program->native_submissions(),target_program->native_submissions(),
                policy_program->submissions,random_q->submissions,random_decision->submissions};
          };
          const auto progress_stopped=[&](std::int32_t token){return std::ranges::find(limits.eos,token)!=limits.eos.end();};
          std::array<const std::int32_t*,2> progress_storage;
          std::array<std::size_t,2> progress_capacity;
          for(std::size_t at=0;at<2;++at) {
            progress_storage[at]=request_progress[at]->generated.data();
            progress_capacity[at]=request_progress[at]->generated.capacity();
          }
          c::StateVersion first_checkpoint,second,published,caught;
          Image first_image,first_checkpoint_image,second_image,published_image,caught_image,second_request_image;
          c::ReadLease pending;
          std::int32_t proposal=0;
          hyle::inference::mtp::AcceptedPrefix decision{};
          const SamplingCursor second_random{initial_random.seed,initial_random.stream+1,0};
          auto *checkpoints=&driver.directory;GlmCheckpoint second_request;
          std::size_t late_catchup_refusals=0,observer_refusals=0;
          std::optional<decltype(request_submission_counts())> retry_submissions,restore_submissions;
          std::size_t retry_publications=0;
          const auto publish_request_result=[&](std::size_t index,const c::StateIdentity &expected,
              const c::StateVersion &version,std::span<const std::int32_t> emitted,bool stopped) {
            auto &head=driver.heads[index];
            const auto records=request_records(index);
            require_observed(emitted.size()==1&&request_commits[index]<records.size(),
                "original GLM request reserves one real emitted pending per Graph result");
            auto &record=records[request_commits[index]];
            record.emitted[0]=emitted[0];record.stopped=stopped;
            // The ledger witnesses publication without retaining a native
            // StateIdentity, which would pin this historical GPU version.
            record.cursor=state.scalar(version,kCursor);record.index_cursor=state.scalar(version,kIndexCursor);
            record.lag=state.scalar(version,kLag);record.epoch=state.epoch(version).epoch;
            record.sampling=random_cursor(version);
            const auto ordinal=request_commits[index];
            auto commit=take(hyle::inference::prepare_record_commit(records,request_commits[index],record),
                "prepare original GLM result commit witness");
            check(commit.publish([&]() noexcept {return head.publish(expected,version);}),
                "publish original GLM Head and immediately attest its result record");
            ++request_publications;
            require_observed(record.committed&&request_commits[index]==ordinal+1&&!commit.valid(),
                "successful GLM Head publication immediately owns exactly one record ordinal");
          };
          const auto observe_request_result=[&](std::size_t index,std::size_t ordinal,
              const c::StateVersion &actual) {
            return hyle::inference::observe_record_commit(
                std::span<const GlmRequestDirectory::Commit>(request_records(index)),request_commits[index],ordinal,*request_progress[index],
                [&](const GlmRequestDirectory::Commit &record) {
                  ++record_projections;
                  return hyle::inference::CommitObservation{record.emitted,true,true,record.stopped};
                },
                [&](const GlmRequestDirectory::Commit &record,const hyle::inference::CommitObservation &) {
                  if(record.emitted[0]!=state.scalar(actual,kPending)||record.sampling!=random_cursor(actual)||
                      record.cursor!=state.scalar(actual,kCursor)||record.index_cursor!=state.scalar(actual,kIndexCursor)||
                      record.lag!=state.scalar(actual,kLag)||record.epoch!=state.epoch(actual).epoch)
                    throw std::runtime_error("GLM observation requires its actual committed pending/RNG result");
                },
                [&](const GlmRequestDirectory::Commit &,const hyle::inference::CommitObservation &) noexcept {++family_observations[index];});
          };

          const auto execute=[&](hyle::inference::RequestKey key,hyle::inference::RequestCreditGrant grant) {
            const auto index=key.slot;auto &progress=*request_progress[index];auto &head=driver.heads[index];
            constexpr int scenario=0;
            // A head/record can be authoritative while a later host observer
            // refused. The next common wave consumes that original result only.
            if(request_commits[index]>progress.observed_commits()) {
              require_observed(index==0&&progress.observed_commits()==0&&request_commits[0]==1&&observer_refusals==1,
                  "GLM retry names its one real published unobserved result");
              EXPECT_EQ(request_submission_counts(),*retry_submissions);EXPECT_EQ(request_publications,retry_publications);
              const auto before=image(state,caught);
              EXPECT_TRUE(observe_request_result(0,0,published));
              EXPECT_FALSE(observe_request_result(0,0,base));
              EXPECT_EQ(request_submission_counts(),*retry_submissions);EXPECT_EQ(request_publications,retry_publications);
              EXPECT_EQ(take(head.current(),"GLM observer retry never republishes Head").identity(),caught.identity());
              EXPECT_EQ(image(state,caught),before);EXPECT_EQ(random_cursor(caught),initial_random.advanced(3));
              EXPECT_EQ(family_observations[0],1u);EXPECT_EQ(progress.observed_commits(),1u);
              EXPECT_EQ(progress.generated.size(),1u);EXPECT_EQ(progress.generated.back(),state.scalar(caught,kPending));
              EXPECT_TRUE(progress.turn_done);EXPECT_EQ(progress.remaining_outputs(),0u);
              return;
            }
            if(index==0&&!progress.turn_index) {
              c::ReadLease first_logits;
              if(stochastic&&std::ranges::find(limits.eos,state.scalar(base,kPending))!=limits.eos.end())
                throw std::runtime_error("Pro cannot consume an EOS pending token");
              EXPECT_THROW(validate_step(state,base,catchup_step),std::runtime_error);
              auto actual=model_steps.advance(base,{first_step,first_program},7,target_before,stochastic?&first_logits:nullptr);
              observe_advance(actual);first=std::move(actual.version);
              check_current(first);
              std::vector<c::ReadLease> random_draft;
              if(stochastic) {
                const auto current_id=state.scalar(base,kPending);
                if(std::ranges::find(limits.eos,current_id)!=limits.eos.end())throw std::runtime_error("Pro cannot consume an EOS pending token");

                if(limits.single_row()) {
                  single_current(first,limits,false);
                  ::testing::Test::RecordProperty("random_tail_target_rows",1);::testing::Test::RecordProperty("random_tail_draws",1);
                  ::testing::Test::RecordProperty("scope","Finite one-row target-only emission with same-prefix mandatory-current NextN maintenance; no q sample or proposed second-row execution. Actual p0 pending and one draw publish with the complete Head. Not a request scheduler.");
                  return;
                }
                single_current(first,{1,1,{}},false);check(state.store.collect(),"retire independent Pro single Head");single_current(first,{1,1,{}},true);check(state.store.collect(),"retire independent Pro terminal Head");
                auto proposed=model_steps.propose(first,first_logits,initial_random);
                random_draft=std::move(proposed.random);first=std::move(proposed.checkpoint);
              }
              first_image=image(state,first);
              bool wrong_skip_submitted=false;
              EXPECT_THROW((void)execute_step(state,first,second_step,*source,8,
                  [&](std::span<const c::ReadLease>) {
                    wrong_skip_submitted=true;return std::vector<c::ReadLease>{};
                  }),std::runtime_error);
              EXPECT_FALSE(wrong_skip_submitted);
              first_checkpoint=first;
              if(scenario) {
                auto controlled=take(state.store.begin(first),"explicit controlled proposal policy");
                check(controlled.replace(state.entry(kPending),policy_checked[scenario==1?(use_target?policy->pending[0]:verifier.pending[0]):(use_target?policy->reject:verifier.reject)]),
                    "controlled proposal retains actual target Top1/Top2 I32 output");
                first_checkpoint=take(controlled.finalize(),"controlled actual pending checkpoint");
              }
              proposal=state.scalar(first_checkpoint,kPending);
              first_checkpoint_image=image(state,first_checkpoint);
              check_step_admission(first_checkpoint,random_draft,grant);
              auto candidate=model_steps.verify_and_prepare(base,first_checkpoint,random_draft,initial_random,
                  grant,limits,checked,selection_emitted,7);
              observe_candidate(candidate);
              second=candidate.second;if(use_target)checked=std::move(candidate.checked);check_verified(first_checkpoint,second);
              second_image=image(state,second);
              if(scenario==0) {
                bool wrong_catchup_submitted=false;
                EXPECT_THROW((void)execute_step(state,second,catchup_step,*source,8,
                    [&](std::span<const c::ReadLease>) {
                      wrong_catchup_submitted=true;return std::vector<c::ReadLease>{};
                    }),std::runtime_error);
                EXPECT_FALSE(wrong_catchup_submitted);
              }
              const auto &selected_prefix=candidate.selected;
              decision=selected_prefix.prefix;pending=selected_prefix.pending;
              const bool sampled_accept=selected_prefix.sampled_accept;
              if(scenario==0){natural_proposal=proposal;natural_accepted=std::uint32_t(decision.consumed_drafts);}
              else EXPECT_EQ(decision.consumed_drafts,scenario==1?1u:0u);
              ++accepted.at(decision.consumed_drafts);
              auto checkpoint=std::move(candidate.checkpoint);
              exercise_acceptance_cancel(head,checkpoint,first,pending);
              published=std::move(candidate.version);
              if(stochastic) {
                EXPECT_FALSE(head.publish(second.identity(),published).ok());
                EXPECT_EQ(random_cursor(take(head.current(),"stale Pro publish keeps RNG")),initial_random);
              }
              if(request_directory_demo) {
                require_observed(selection_emitted.size()==1,"actual GLM grant bounds its selected K1 prefix before publication");
                require_observed(request_progress[0]->preflight_commit(0,selection_emitted,true,true,decision.stopped),
                    "preflight actual K1 host output before its complete Head publication");
              }
              if(request_directory_demo)publish_request_result(0,base.identity(),published,selection_emitted,decision.stopped);
              else check(head.publish(base.identity(),published),"atomically publish Pro accepted prefix and pending");
              if(stochastic)EXPECT_EQ(random_cursor(take(head.current(),"successful Pro publish advances RNG once")),initial_random.advanced(3));
              EXPECT_FALSE(head.publish(base.identity(),second).ok());++stale_publish;
              published_image=image(state,published);
              EXPECT_EQ(state.scalar(published,kCursor),std::int32_t(kBasePosition+1+decision.consumed_drafts));
              EXPECT_EQ(state.scalar(published,kIndexCursor),std::int32_t(kBasePosition+1));
              EXPECT_EQ(state.scalar(published,kLag),std::int32_t(decision.consumed_drafts));
              EXPECT_EQ(state.scalar(published,kPending),decision.next_id);
              same_storage(state.member(published,kPending),pending);
              for(std::size_t i=0;i<resource_count;++i)if(i!=layout.slot(kPending))
                same_storage(state.physical_member(published,i),state.physical_member(checkpoint,i));
              if(use_target) {
                for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})
                  same_storage(state.member(published,entry),state.member(checkpoint,entry));
                EXPECT_EQ(state.scalar(published,kTargetCursor),state.scalar(published,kCursor));
              }
              if(stochastic&&!exercise_prefix_policies(first_checkpoint,checkpoint,proposal,decision,sampled_accept,pending,random_draft[1]))return;
              caught=published;
              if(decision.consumed_drafts) {
                caught=execute_observed_step(state,published,catchup_step,*source,7,
                    [&](std::span<const c::ReadLease> input){return catchup_program.execute(queue,input);});
                ++catchups;check(head.publish(published.identity(),caught),"same accepted Head catches up its one journal row");
              } else if(request_directory_demo) {
                const auto submissions=request_submission_counts();
                const auto publications=request_publications;
                bool refused=false;
                try {validate_step(state,published,catchup_step);}
                catch(const std::runtime_error &error) {
                  require_observed(std::string(error.what())=="finite Pro catchup requires the next actual accepted journal row",
                      "late GLM catchup preserves its original journal precondition");
                  refused=true;
                }
                require_observed(refused,"no accepted GLM journal row may launch catchup after publication");
                ++late_catchup_refusals;
                EXPECT_TRUE(request0_records[0].committed);EXPECT_EQ(request_commits[0],1u);
                EXPECT_EQ(request_progress[0]->observed_commits(),0u);EXPECT_TRUE(request_progress[0]->generated.empty());
                EXPECT_EQ(take(head.current(),"late catchup refusal retains actual committed Head").identity(),published.identity());
                EXPECT_EQ(image(state,published),published_image);EXPECT_EQ(random_cursor(published),initial_random.advanced(3));
                EXPECT_EQ(request_submission_counts(),submissions);EXPECT_EQ(request_publications,publications);
              } else EXPECT_THROW(validate_step(state,published,catchup_step),std::runtime_error);
              EXPECT_EQ(state.scalar(caught,kLag),0);EXPECT_EQ(state.scalar(caught,kCursor),state.scalar(caught,kIndexCursor));
              if(use_target)for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})
                same_storage(state.member(caught,entry),state.member(published,entry));
              require_observed(glm_shared_vocabulary_reference::expect_feedback(values<std::int32_t>(state.member(caught,kPending)),
                  values<std::int32_t>(pending)),"actual target pending enters next embedding");
              if(stochastic)EXPECT_EQ(random_cursor(caught),initial_random.advanced(3));

              driver.consumed_inputs={state.scalar(base,kPending),proposal};
              caught_image=image(state,caught);
              retry_submissions=request_submission_counts();retry_publications=request_publications;
              // The original stale-family witness is rejected after commit.
              // Let RequestLoop see this late failure; a later pump observes
              // the same original record without resubmitting any math.
              ++observer_refusals;
              (void)observe_request_result(0,0,base);
              throw std::runtime_error("stale GLM observer unexpectedly accepted its old witness");
            }
            if(index==1&&!progress.turn_index) {
              const auto &input=driver.prefix_input;
              const auto length=input->ids.size();
              EXPECT_FALSE(checkpoints->evict_one());
              EXPECT_FALSE(checkpoints->lookup({source.get(),&first},*input,length,length));
              auto &hit=driver.hit;
              require_observed(bool(hit),"second GLM request really hits the complete suffix boundary");
              EXPECT_EQ(hit.complete_length(),length);EXPECT_EQ(hit.payload.consumed,length);
              EXPECT_EQ(hit.payload.cursor,state.scalar(caught,kCursor));
              EXPECT_EQ(hit.payload.version.identity(),caught.identity());
              EXPECT_FALSE(checkpoints->evict_one());
              EXPECT_FALSE(checkpoints->lookup({source.get(),&first},*input,length,length));
              const auto hit_image=image(state,hit.payload.version);
              const std::array tap{state.member(hit.payload.version,kTargetTap)};
              const auto target=policy_program->execute(queue,tap);
              (void)check_verifier(*policy,target,values<Tap>(tap[0]),false);
              const auto sampled=sample_logits(target[policy->logits],second_random);
              const auto rng=second_random.advanced(1);
              {
                auto cancelled_hit=take(state.store.begin(hit.payload.version),"cancel new GLM request cache fork");
                check(cancelled_hit.replace(state.entry(kPending),sampled[1]),"cancel actual cache-hit sampled pending");
                check(cancelled_hit.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&rng,1))),
                    "cancel cache-hit sampling cursor");
                check(cancelled_hit.cancel(),"discard whole cache-hit fork");
                EXPECT_EQ(image(state,take(head.current(),"cancelled cache fork Head")),hit_image);
                EXPECT_EQ(request_window.active().size(),1u);EXPECT_EQ(request_window.active()[0].slot,1u);
                EXPECT_EQ(request_window.active()[0],request_keys[1]);
                EXPECT_FALSE(request_window.waiting_front());EXPECT_EQ(request_turns,2u);
                EXPECT_TRUE(request_progress[1]->generated.empty());EXPECT_EQ(request_progress[1]->observed_commits(),0u);
              }
              auto fork=take(state.store.begin(hit.payload.version),"new request full-prefix sampling fork");
              check(fork.replace(state.entry(kPending),sampled[1]),"new request directly retains its actual sampled token lease");
              check(fork.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&rng,1))),
                  "new request has an independent sampling stream");
              auto version=take(fork.finalize(),"complete new GLM request cache fork");
              EXPECT_FALSE(head.publish(base.identity(),version).ok());
              EXPECT_EQ(image(state,take(head.current(),"stale cache fork leaves Head unchanged")),hit_image);
              EXPECT_EQ(request_window.active().size(),1u);EXPECT_EQ(request_window.active()[0].slot,1u);
              EXPECT_EQ(request_window.active()[0],request_keys[1]);
              EXPECT_FALSE(request_window.waiting_front());EXPECT_EQ(request_turns,2u);
              EXPECT_TRUE(request_progress[1]->generated.empty());EXPECT_EQ(request_progress[1]->observed_commits(),0u);
              const std::array fork_output{state.scalar(version,kPending)};
              request_progress[1]->preflight_outputs(fork_output,true,true,progress_stopped(fork_output[0]));
              check(head.publish(hit.payload.version.identity(),version),"atomically publish new request pending and RNG");
              request_progress[1]->apply_outputs(fork_output,true,true,progress_stopped(fork_output[0]));
              EXPECT_EQ(request_progress[1]->observed_commits(),0u); // A cache fork is not another Graph commit.
              EXPECT_EQ(request_progress[1]->generated.size(),1u);EXPECT_EQ(request_progress[1]->generated.back(),fork_output[0]);
              EXPECT_TRUE(request_progress[1]->turn_done);EXPECT_EQ(request_progress[1]->remaining_outputs(),0u);
              EXPECT_EQ(random_cursor(version),rng);EXPECT_EQ(random_cursor(caught),initial_random.advanced(3));
              EXPECT_EQ(image(state,hit.payload.version),hit_image);
              same_storage(state.member(version,kPending),sampled[1]);
              for(std::size_t i=0;i<resource_count;++i)if(i!=layout.slot(kPending))
                same_storage(state.physical_member(version,i),state.physical_member(caught,i));
              for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})
                same_storage(state.member(version,entry),state.member(caught,entry));
              second_request={std::move(version),hit.payload.consumed,hit.payload.cursor};
              second_request_image=image(state,second_request.version);
              hit.reset();

              EXPECT_TRUE(request_progress[1]->more_turns());
              restore_submissions=request_submission_counts();
              return;
            }
            if(index==0&&progress.turn_index==1) {
              EXPECT_EQ(request_submission_counts(),*restore_submissions);
              const auto restored=take(head.current(),"same-request GLM restore");
              EXPECT_EQ(restored.identity(),caught.identity());EXPECT_EQ(image(state,restored),caught_image);
              EXPECT_EQ(random_cursor(restored),initial_random.advanced(3));same_storage(state.member(restored,kPending),pending);
              {
              auto &next_step=decision.consumed_drafts?continuation_one:continuation_zero;
              auto &next_program=decision.consumed_drafts?continuation_one_program:continuation_zero_program;
              auto continued=execute_checked_ordinary(caught,{next_step,next_program},
                  use_target?state.member(caught,kTargetTap):next_targets[decision.consumed_drafts],initial_random.advanced(3));
              if(request_directory_demo) {
                const std::array emitted{state.scalar(continued,kPending)};
                require_observed(request_progress[0]->preflight_commit(1,emitted,true,true,progress_stopped(emitted[0])),
                    "preflight original request's actual ordinary continuation");
              }
              if(request_directory_demo) {
                const std::array emitted{state.scalar(continued,kPending)};
                publish_request_result(0,caught.identity(),continued,emitted,progress_stopped(emitted[0]));
              } else check(head.publish(caught.identity(),continued),"publish actual accepted-frontier fresh seed in the same Head");
              if(request_directory_demo) {
                const std::array emitted{state.scalar(continued,kPending)};
                EXPECT_TRUE(observe_request_result(0,1,continued));
                EXPECT_FALSE(observe_request_result(0,1,base));
                EXPECT_EQ(request_progress[0]->observed_commits(),2u);EXPECT_EQ(request_progress[0]->generated.size(),2u);
                EXPECT_EQ(request_progress[0]->generated.back(),emitted[0]);EXPECT_TRUE(request_progress[0]->turn_done);
                EXPECT_EQ(request_progress[0]->remaining_outputs(),0u);EXPECT_FALSE(request_progress[0]->more_turns());
              }
              if(stochastic)EXPECT_EQ(random_cursor(take(head.current(),"next target result and RNG published together")),initial_random.advanced(4));
              const auto continued_image=image(state,continued);
              const auto seed=take(head.current(),"actual accepted fresh Pro seed");
              EXPECT_EQ(seed.identity(),continued.identity());EXPECT_EQ(state.epoch(seed).epoch,8u);
              EXPECT_EQ(state.epoch(seed).selection_position,std::uint64_t(next_step.position));
              EXPECT_EQ(state.epoch(seed).reuses,0u);
              auto &step_zero=decision.consumed_drafts?skip_one:skip_zero;
              auto &zero_program=decision.consumed_drafts?skip_one_program:skip_zero_program;
              const auto skipped=execute_observed_step(state,seed,step_zero,*source,8,
                  [&](std::span<const c::ReadLease> input){return zero_program.execute(queue,input,cuda);});
              ++feedback;++skips;
              same_storage(state.member(skipped,kSelection),state.member(seed,kSelection));
              same_storage(state.member(skipped,kCodes),state.member(seed,kCodes));
              EXPECT_EQ(state.epoch(skipped).epoch,8u);EXPECT_EQ(state.epoch(skipped).reuses,1u);
              EXPECT_EQ(state.scalar(skipped,kLag),1);
              EXPECT_EQ(image(state,take(head.current(),"unverified step0 stays unpublished")),continued_image);
              EXPECT_EQ(image(state,published),published_image);EXPECT_EQ(image(state,first_checkpoint),first_checkpoint_image);
              EXPECT_EQ(image(state,second),second_image);EXPECT_EQ(image(state,base),base_image);EXPECT_EQ(image(state,first),first_image);
              EXPECT_FALSE(state.member(published,kPending).try_write());EXPECT_FALSE(state.member(continued,kMain).try_write());
              EXPECT_LE(state.peak_backing,budget);
              std::printf("GLM Pro K1 stochastic=%d scenario=%d proposal=%d target0=%d target1=%d accepted=%u frontier=%d pending=%d continued_pending=%d\n",
                  int(stochastic),scenario,proposal,target_ids[0],target_ids[1],std::uint32_t(decision.consumed_drafts),state.scalar(published,kCursor),decision.next_id,state.scalar(continued,kPending));
              if(request_directory_demo) {
                ::testing::Test::RecordProperty("request_directory_r0_pending",state.scalar(continued,kPending));
                ::testing::Test::RecordProperty("request_directory_r0_committed_draws",int(random_cursor(continued).draw));
              }

              }
              return;
            }
            require_observed(index==1&&progress.turn_index==1,"finite GLM wave keeps its original request/turn");
            EXPECT_EQ(image(state,take(head.current(),"second request restores pending and RNG")),second_request_image);
            EXPECT_EQ(random_cursor(second_request.version),second_random.advanced(1));
            EXPECT_EQ(second_request.cursor,state.scalar(second_request.version,kCursor));
              auto &step=second_request.consumed==2?continuation_one:continuation_zero;
              auto &program=second_request.consumed==2?continuation_one_program:continuation_zero_program;
              auto continued=execute_checked_ordinary(second_request.version,{step,program},state.member(second_request.version,kTargetTap),second_random.advanced(1));
              const auto rng=second_random.advanced(2);
              const std::array emitted{state.scalar(continued,kPending)};
              require_observed(request_progress[1]->preflight_commit(0,emitted,true,true,progress_stopped(emitted[0])),
                  "preflight second request's actual ordinary continuation");
              publish_request_result(1,second_request.version.identity(),continued,emitted,progress_stopped(emitted[0]));
              EXPECT_TRUE(observe_request_result(1,0,continued));
              EXPECT_FALSE(observe_request_result(1,0,base));
              EXPECT_EQ(request_progress[1]->observed_commits(),1u);EXPECT_EQ(request_progress[1]->generated.size(),2u);
              EXPECT_EQ(request_progress[1]->generated.back(),emitted[0]);EXPECT_TRUE(request_progress[1]->turn_done);
              EXPECT_EQ(request_progress[1]->remaining_outputs(),0u);EXPECT_FALSE(request_progress[1]->more_turns());
              EXPECT_EQ(random_cursor(take(head.current(),"second request final Head")),rng);
              EXPECT_EQ(state.scalar(continued,kCursor),second_request.cursor+1);
              EXPECT_EQ(image(state,second_request.version),second_request_image);
              EXPECT_EQ(image(state,caught),caught_image);

          };
          std::size_t pumps=0,late_failures=0;
          while(driver.loop.matches(request_keys[0])||driver.loop.matches(request_keys[1])) {
            require_observed(++pumps<=8,"finite GLM request loop did not finish its four real turns");
            try {(void)driver.pump(execute);}
            catch(const std::runtime_error &error) {
              if(late_failures||std::string(error.what())!="GLM observation requires its actual committed pending/RNG result")throw;
              ++late_failures;
              EXPECT_TRUE(request0_records[0].committed);EXPECT_EQ(request_commits[0],1u);
              EXPECT_EQ(request_progress[0]->observed_commits(),0u);EXPECT_TRUE(request_progress[0]->generated.empty());
              EXPECT_EQ(family_observations[0],0u);
              EXPECT_EQ(take(driver.heads[0].current(),"late observer failure keeps original complete Head").identity(),caught.identity());
              EXPECT_EQ(image(state,caught),caught_image);EXPECT_EQ(random_cursor(caught),initial_random.advanced(3));
              EXPECT_EQ(request_submission_counts(),*retry_submissions);EXPECT_EQ(request_publications,retry_publications);
              require_observed(driver.native_retirement(false).retired,"late GLM observer failure really retired original native work");
            }
          }
          EXPECT_EQ(late_failures,1u);EXPECT_EQ(pumps,5u);
          EXPECT_EQ(request_order,(std::array<std::size_t,4>{0,1,0,1}));EXPECT_EQ(request_turns,4u);
          EXPECT_TRUE(request_window.active().empty());EXPECT_FALSE(request_window.waiting_front());
          EXPECT_TRUE(checkpoints->evict_one());EXPECT_EQ(checkpoints->size(),0u);EXPECT_EQ(checkpoints->evictions,1u);
          EXPECT_EQ(image(state,second_request.version),second_request_image);
          EXPECT_LE(state.peak_backing,budget);
          EXPECT_LE(state.store.stats().backing_bytes,budget);
          for(std::size_t index=0;index<request_progress.size();++index) {
            const auto &progress=*request_progress[index];
            EXPECT_EQ(progress.generated.data(),progress_storage[index]);EXPECT_EQ(progress.generated.capacity(),progress_capacity[index]);
            EXPECT_EQ(progress.turn_index,1u);EXPECT_EQ(progress.output_limit(),2u);EXPECT_TRUE(progress.prefilled);
          }
          ::testing::Test::RecordProperty("request_progress_output_counts","2,2");
          ::testing::Test::RecordProperty("request_progress_committed_observations","2,1");
          ::testing::Test::RecordProperty("request_progress_duplicate_commit_noops",3);
          ::testing::Test::RecordProperty("request_progress_full_hit_outputs",1);
          ::testing::Test::RecordProperty("request_progress_pending_only_turn_inputs",2);
          EXPECT_EQ(request_commits,(std::array<std::size_t,2>{2,1}));
          EXPECT_EQ(family_observations,request_commits);EXPECT_EQ(request_publications,3u);
          EXPECT_EQ(record_projections,4u);EXPECT_EQ(late_catchup_refusals,1u);EXPECT_EQ(observer_refusals,1u);
          ::testing::Test::RecordProperty("request_record_committed_prefixes","2,1");
          ::testing::Test::RecordProperty("request_record_head_publications",int(request_publications));
          ::testing::Test::RecordProperty("request_record_late_catchup_refusals",int(late_catchup_refusals));
          ::testing::Test::RecordProperty("request_record_observer_refusals",int(observer_refusals));
          ::testing::Test::RecordProperty("request_record_projected_observations",int(record_projections));
          ::testing::Test::RecordProperty("request_record_observer_retry_graphs",0);
          ::testing::Test::RecordProperty("request_record_observer_retry_head_publications",0);
          ::testing::Test::RecordProperty("request_record_observer_retry_rng_draws",0);
          ::testing::Test::RecordProperty("request_progress_scope","Shared RequestProgress with the actual grant applied to the selected K1 prefix and new-request cache fork, then the two original ordinary continuations. Each finite turn has quota1; actual model selector/EOS facts are preserved. A later empty new-user suffix consumes only its actual historical pending. No extra graph, model publication, RNG address, complete GLM prompt prefill or request server is introduced.");
          ::testing::Test::RecordProperty("request_directory_requests",2);
          ::testing::Test::RecordProperty("request_directory_turns",int(request_turns));
          ::testing::Test::RecordProperty("request_directory_fifo","0,1,0,1");
          ::testing::Test::RecordProperty("request_directory_full_hits",1);
          ::testing::Test::RecordProperty("request_directory_evictions",int(checkpoints->evictions));
          ::testing::Test::RecordProperty("request_directory_same_request_restore_graphs",0);
          ::testing::Test::RecordProperty("request_directory_new_request_draws",2);
          ::testing::Test::RecordProperty("request_directory_cancelled_forks",1);
          ::testing::Test::RecordProperty("request_directory_stale_forks",1);
          ::testing::Test::RecordProperty("request_directory_retained_state_versions",int(state.store.stats().versions));
          ::testing::Test::RecordProperty("request_directory_current_backing_bytes",std::to_string(state.store.stats().backing_bytes));
          ::testing::Test::RecordProperty("request_directory_scope","Original target77/NextN78 finite suffix driven by shared RequestLoop and advance_request_wave after this exact initial StateVersion and upstream fixture; N1 FIFO limits resident Heads, not all retained StateStore versions. Directory holds real complete lag0 state. New request resamples full-V pending with stream18; same request restores stream17 draw3 without sampling. Each request then consumes its own pending through actual target+NextN ordinary continuation. No arbitrary GLM prompt prefill or complete-model/server claim.");

          EXPECT_EQ(driver.evictions,2u);EXPECT_FALSE(driver.heads[0].valid());EXPECT_FALSE(driver.heads[1].valid());
          ::testing::Test::RecordProperty("request_loop_pumps",int(pumps));
          ::testing::Test::RecordProperty("request_loop_resident_evictions",int(driver.evictions));
          ::testing::Test::RecordProperty("request_wave_late_observer_retry",1);
        }
        check(state.store.collect(),"retire finite GLM request loop while retaining base/first readers");
        EXPECT_EQ(state.store.stats().versions,2u);EXPECT_EQ(state.store.stats().drafts,0u);
      } else {

        c::ReadLease first_logits;
        if(stochastic&&std::ranges::find(limits.eos,state.scalar(base,kPending))!=limits.eos.end())
          throw std::runtime_error("Pro cannot consume an EOS pending token");
        EXPECT_THROW(validate_step(state,base,catchup_step),std::runtime_error);
        auto actual=model_steps.advance(base,{first_step,first_program},7,target_before,stochastic?&first_logits:nullptr);
        observe_advance(actual);auto first=std::move(actual.version);
        check_current(first);
        if(stochastic&&maximum_drafts==2&&!limits.single_row()) {
          const auto q=sample_logits(first_logits,initial_random);
          const auto q0=values<std::int32_t>(q[1]).at(0);
          const bool recurse=limits.quota>2&&limits.remaining_context>2&&std::ranges::find(limits.eos,q0)==limits.eos.end();
          if(recurse) {
            const auto with_pending=[&](const c::StateVersion &from,const c::ReadLease &id) {
              auto draft=take(state.store.begin(from),"Pro recursive actual sampled proposal");
              check(draft.replace(state.entry(kPending),id),"Pro recursive I32 proposal lease");
              return take(draft.finalize(),"Pro complete recursive input state");
            };
            const auto joint=[&](const c::StateVersion &from,const Step &step,Resident &program,
                std::uint64_t epoch,c::ReadLease *logits=nullptr) {
              auto result=execute_observed_step(state,from,step,*source,epoch,
                  [&](std::span<const c::ReadLease> input){return program.execute(queue,input,cuda);},
                  step.fresh?state.member(from,kTargetTap):c::ReadLease{},logits);
              const auto target=execute_target(*target_step,*target_program,*target_source,owner,queue,cuda,
                  state.member(from,kPending),state.member(from,kTargetMain),state.member(from,kTargetCursor),upstream_ids);
              observe_target(*target_step,*target_source,*source,target);
              result=attach_target(state,result,target);
              EXPECT_EQ(state.scalar(result,kTargetCursor),state.scalar(result,kCursor));return result;
            };
            const auto proposed0=with_pending(first,q[1]);c::ReadLease q1_logits;
            const auto second=joint(proposed0,second_step,second_program,7,&q1_logits);
            const auto q_next=sample_logits(q1_logits,initial_random.advanced(1));
            const auto q1=values<std::int32_t>(q_next[1]).at(0);
            const auto proposed1=with_pending(second,q_next[1]);
            const auto third=joint(proposed1,skip_zero,skip_zero_program,7);
            const std::array taps{state.member(first,kTargetTap),state.member(second,kTargetTap),state.member(third,kTargetTap)};
            const auto projected=recursive_verifier->execute(queue,taps);
            const auto p=values<float>(projected[0]);const auto vocabulary=std::size_t(model::kVocabulary);
            EXPECT_EQ(p.size(),3*vocabulary);
            std::vector<Tap> joined;for(const auto &tap:taps){const auto data=values<Tap>(tap);joined.insert(joined.end(),data.begin(),data.end());}
            const std::array required{q0,q1,q1};
            require_observed(Runtime::check_logits(*source,p,joined,false,required,cuda),"actual Pro T3 full-vocabulary target projection");
            const std::array ids{q0,q1};const auto id_lease=upload(owner,sem::bytes(std::vector(ids.begin(),ids.end())));
            const auto decide_prefix=[&](std::array<std::uint64_t,3> offsets) {
              std::vector<std::uint32_t> words;for(const auto offset:offsets){const auto key=initial_random.identity(offset);words.insert(words.end(),key.begin(),key.end());}
              const std::array input{projected[0],upload(owner,sem::bytes(words)),id_lease,q[0],q_next[0]};
              const auto output=recursive_decision->execute(queue,input);glm_stochastic_prefix_fragment::Decision result;
              for(std::size_t row=0;row<2;++row) {
                if(!random_valid(output[row*3+2]))throw std::runtime_error("actual Pro K2 p/q decision invalid");
                result.accepts[row]=std::uint8_t(random_valid(output[row*3]));result.final_leases[row]=output[row*3+1];
                result.final_ids[row]=values<std::int32_t>(result.final_leases[row]).at(0);
              }return result;
            };
            std::array<c::StateVersion,3> repaired{first,{},{}};
            const auto replay=[&](std::size_t count) {
              for(std::size_t row=1;row<=count;++row)if(!repaired[row].valid()) {
                const auto from=with_pending(repaired[row-1],row==1?q[1]:q_next[1]);
                auto &step=row==1?*repair_step:*repair_second_step;
                auto &program=row==1?*repair_program:*repair_second_program;
                auto fixed=execute_observed_step(state,from,step,*source,7,
                    [&](std::span<const c::ReadLease> input){return program.execute(queue,input,cuda);},state.member(row==1?first:second,kTargetTap));
                const auto &target=row==1?second:third;TargetFrame frame;
                const std::array entries{kTargetMain,kTargetCursor,kTargetTap};
                for(std::size_t i=0;i<entries.size();++i)frame.state[i]=state.member(target,entries[i]);
                repaired[row]=attach_target(state,fixed,frame);
              }return repaired[count];
            };
            const auto observe=[&](const c::StateVersion &selected,const c::StateVersion &checkpoint,
                const hyle::inference::mtp::AcceptedPrefix &selection,SamplingCursor rng) {
              EXPECT_EQ(random_cursor(selected),rng);EXPECT_EQ(state.scalar(selected,kPending),selection.next_id);
              EXPECT_EQ(state.scalar(selected,kCursor),std::int32_t(kBasePosition+selection.consumed));
              EXPECT_EQ(state.scalar(selected,kTargetCursor),state.scalar(selected,kCursor));
              EXPECT_EQ(state.scalar(selected,kLag),std::int32_t(selection.consumed_drafts));
              for(std::size_t i=0;i<resource_count;++i)if(i!=layout.slot(kPending))same_storage(state.physical_member(selected,i),state.physical_member(checkpoint,i));
              for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})same_storage(state.member(selected,entry),state.member(checkpoint,entry));
            };
            const auto continued=[&](c::StateHead &head,const c::StateVersion &selected,std::size_t consumed,SamplingCursor rng) {
              auto caught=selected;
              for(std::size_t slot=0;slot<consumed;++slot) {
                auto &step=slot==0?catchup_step:*recursive_catchup_step;auto &program=slot==0?catchup_program:*recursive_catchup;
                const auto next=execute_observed_step(state,caught,step,*source,7,
                    [&](std::span<const c::ReadLease> input){return program.execute(queue,input);});
                check(head.publish(caught.identity(),next),"Pro K2 actual accepted journal row publication");caught=next;
              }
              EXPECT_EQ(state.scalar(caught,kLag),0);EXPECT_EQ(state.scalar(caught,kCursor),state.scalar(caught,kIndexCursor));
              auto &step=consumed==0?continuation_zero:consumed==1?continuation_one:*recursive_fresh_step;
              auto &program=consumed==0?continuation_zero_program:consumed==1?continuation_one_program:*recursive_fresh_program;
              const auto next=joint(caught,step,program,8);
              const std::array input{state.member(next,kTargetTap)};const auto target=policy_program->execute(queue,input);
              const auto sampled=sample_logits(target[policy->logits],rng);auto draft=take(state.store.begin(next),"Pro K2 target-only real pending continuation");
              check(draft.replace(state.entry(kPending),sampled[1]),"Pro K2 next target categorical result");const auto advanced=rng.advanced(1);
              check(draft.replace_metadata(state.entries[*state.sampling_entry],std::as_bytes(std::span(&advanced,1))),"Pro K2 target-only continuation RNG");
              const auto version=take(draft.finalize(),"Pro K2 complete real continuation");check(head.publish(caught.identity(),version),"Pro K2 next epoch consumes actual pending");
              EXPECT_EQ(random_cursor(version),advanced);EXPECT_EQ(state.scalar(version,kCursor),std::int32_t(kBasePosition+2+consumed));
              std::printf("GLM PRO K2 CONTINUED consumed_drafts=%zu consumed_pending=%d next_pending=%d cursor=%llu\n",consumed,state.scalar(caught,kPending),state.scalar(version,kPending),static_cast<unsigned long long>(advanced.draw));
            };
            const auto q0_log=values<float>(q[0]),q1_log=values<float>(q_next[0]);
            const std::array alpha{glm_stochastic_prefix_fragment::log_alpha(std::span(p).first(vocabulary),q0_log,q0),
                glm_stochastic_prefix_fragment::log_alpha(std::span(p).subspan(vocabulary,vocabulary),q1_log,q1)};
            glm_stochastic_prefix_fragment::exercise(state.store,base,state.entry(kPending),state.entries[*state.sampling_entry],state.scalar(base,kPending),ids,
                std::array{q[1],q_next[1]},state.member(first,kTargetTap),initial_random,std::int32_t(model::kVocabulary),cuda,alpha,limits,
                decide_prefix,replay,observe,continued);
            EXPECT_EQ(image(state,base),base_image);return;
          }
        }
        std::vector<c::ReadLease> random_draft;
        if(stochastic) {
          const auto current_id=state.scalar(base,kPending);
          if(std::ranges::find(limits.eos,current_id)!=limits.eos.end())throw std::runtime_error("Pro cannot consume an EOS pending token");

          if(limits.single_row()) {
            single_current(first,limits,false);
            ::testing::Test::RecordProperty("random_tail_target_rows",1);::testing::Test::RecordProperty("random_tail_draws",1);
            ::testing::Test::RecordProperty("scope","Finite one-row target-only emission with same-prefix mandatory-current NextN maintenance; no q sample or proposed second-row execution. Actual p0 pending and one draw publish with the complete Head. Not a request scheduler.");
            return;
          }
          single_current(first,{1,1,{}},false);check(state.store.collect(),"retire independent Pro single Head");single_current(first,{1,1,{}},true);check(state.store.collect(),"retire independent Pro terminal Head");
          auto proposed=model_steps.propose(first,first_logits,initial_random);
          random_draft=std::move(proposed.random);first=std::move(proposed.checkpoint);
        }
        const auto first_image=image(state,first);
        bool wrong_skip_submitted=false;
        EXPECT_THROW((void)execute_step(state,first,second_step,*source,8,
            [&](std::span<const c::ReadLease>) {
              wrong_skip_submitted=true;return std::vector<c::ReadLease>{};
            }),std::runtime_error);
        EXPECT_FALSE(wrong_skip_submitted);
        // One natural trial and two separately labelled controls. Matching uses
        // the actual full-V Top1 lease; rejection uses the actual distinct Top2
        // lease. The second real Graph consumes precisely that proposal lease.
        for(int scenario=0;scenario<(stochastic?1:3);++scenario) {
          SCOPED_TRACE(::testing::Message()<<"GLM Pro K1 scenario="<<scenario);
          {
            auto first_checkpoint=first;
            if(scenario) {
              auto controlled=take(state.store.begin(first),"explicit controlled proposal policy");
              check(controlled.replace(state.entry(kPending),policy_checked[scenario==1?(use_target?policy->pending[0]:verifier.pending[0]):(use_target?policy->reject:verifier.reject)]),
                  "controlled proposal retains actual target Top1/Top2 I32 output");
              first_checkpoint=take(controlled.finalize(),"controlled actual pending checkpoint");
            }
            const auto proposal=state.scalar(first_checkpoint,kPending);
            const auto first_checkpoint_image=image(state,first_checkpoint);
            if(!scenario)check_step_admission(first_checkpoint,random_draft,{.additional=2});
            auto candidate=model_steps.verify_and_prepare(base,first_checkpoint,random_draft,initial_random,
                {.additional=2},limits,checked,selection_emitted,7);
            observe_candidate(candidate);
            auto second=candidate.second;if(use_target)checked=std::move(candidate.checked);check_verified(first_checkpoint,second);
            if(!stochastic&&scenario==1) {
              const auto before=native_step_counts();
              std::vector<std::int32_t> tokens;
              auto bounded=model_steps.prepare_verified(base,first_checkpoint,second,checked,random_draft,
                  initial_random,{.additional=1},limits,tokens,7);
              observe_candidate(bounded);
              EXPECT_TRUE(bounded.selected.prefix.first_agrees);
              EXPECT_EQ(bounded.selected.prefix.consumed,1u);EXPECT_EQ(bounded.selected.prefix.consumed_drafts,0u);
              EXPECT_EQ(tokens,(std::vector<std::int32_t>{proposal}));
              EXPECT_EQ(state.scalar(bounded.version,kPending),proposal);
              EXPECT_EQ(state.scalar(bounded.version,kCursor),state.scalar(first_checkpoint,kCursor));
              EXPECT_EQ(state.scalar(bounded.version,kLag),0);
              for(std::size_t slot=0;slot<resource_count;++slot)if(slot!=layout.slot(kPending))
                same_storage(state.physical_member(bounded.version,slot),state.physical_member(first_checkpoint,slot));
              if(use_target)for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})
                same_storage(state.member(bounded.version,entry),state.member(first_checkpoint,entry));
              EXPECT_EQ(native_step_counts(),before);EXPECT_EQ(image(state,base),base_image);
              ::testing::Test::RecordProperty("model_step_grant1_matched_proposal_pending",proposal);
              ::testing::Test::RecordProperty("model_step_grant1_matched_consumed",1);
              ::testing::Test::RecordProperty("model_step_grant1_matched_consumed_drafts",0);
              ::testing::Test::RecordProperty("model_step_grant1_extra_graphs",0);
            }
            const auto second_image=image(state,second);
            if(scenario==0) {
              bool wrong_catchup_submitted=false;
              EXPECT_THROW((void)execute_step(state,second,catchup_step,*source,8,
                  [&](std::span<const c::ReadLease>) {
                    wrong_catchup_submitted=true;return std::vector<c::ReadLease>{};
                  }),std::runtime_error);
              EXPECT_FALSE(wrong_catchup_submitted);
            }
            const auto &selected_prefix=candidate.selected;
            const auto &decision=selected_prefix.prefix;const auto &pending=selected_prefix.pending;
            const bool sampled_accept=selected_prefix.sampled_accept;
            if(scenario==0){natural_proposal=proposal;natural_accepted=std::uint32_t(decision.consumed_drafts);}
            else EXPECT_EQ(decision.consumed_drafts,scenario==1?1u:0u);
            ++accepted.at(decision.consumed_drafts);
            auto checkpoint=std::move(candidate.checkpoint);
            auto head=take(c::StateHead::create(state.store,base),"one atomic Pro accepted Head");
            exercise_acceptance_cancel(head,checkpoint,first,pending);
            const auto published=std::move(candidate.version);
            if(stochastic) {
              EXPECT_FALSE(head.publish(second.identity(),published).ok());
              EXPECT_EQ(random_cursor(take(head.current(),"stale Pro publish keeps RNG")),initial_random);
            }

            check(head.publish(base.identity(),published),"atomically publish Pro accepted prefix and pending");
            if(stochastic)EXPECT_EQ(random_cursor(take(head.current(),"successful Pro publish advances RNG once")),initial_random.advanced(3));
            EXPECT_FALSE(head.publish(base.identity(),second).ok());++stale_publish;
            const auto published_image=image(state,published);
            EXPECT_EQ(state.scalar(published,kCursor),std::int32_t(kBasePosition+1+decision.consumed_drafts));
            EXPECT_EQ(state.scalar(published,kIndexCursor),std::int32_t(kBasePosition+1));
            EXPECT_EQ(state.scalar(published,kLag),std::int32_t(decision.consumed_drafts));
            EXPECT_EQ(state.scalar(published,kPending),decision.next_id);
            same_storage(state.member(published,kPending),pending);
            for(std::size_t i=0;i<resource_count;++i)if(i!=layout.slot(kPending))
              same_storage(state.physical_member(published,i),state.physical_member(checkpoint,i));
            if(use_target) {
              for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})
                same_storage(state.member(published,entry),state.member(checkpoint,entry));
              EXPECT_EQ(state.scalar(published,kTargetCursor),state.scalar(published,kCursor));
            }
            if(stochastic&&!exercise_prefix_policies(first_checkpoint,checkpoint,proposal,decision,sampled_accept,pending,random_draft[1]))return;
            auto caught=published;
            if(decision.consumed_drafts) {
              caught=execute_observed_step(state,published,catchup_step,*source,7,
                  [&](std::span<const c::ReadLease> input){return catchup_program.execute(queue,input);});
              ++catchups;check(head.publish(published.identity(),caught),"same accepted Head catches up its one journal row");
            } else EXPECT_THROW(validate_step(state,published,catchup_step),std::runtime_error);
            EXPECT_EQ(state.scalar(caught,kLag),0);EXPECT_EQ(state.scalar(caught,kCursor),state.scalar(caught,kIndexCursor));
            if(use_target)for(const auto entry:{kTargetMain,kTargetCursor,kTargetTap})
              same_storage(state.member(caught,entry),state.member(published,entry));
            require_observed(glm_shared_vocabulary_reference::expect_feedback(values<std::int32_t>(state.member(caught,kPending)),
                values<std::int32_t>(pending)),"actual target pending enters next embedding");
            if(stochastic)EXPECT_EQ(random_cursor(caught),initial_random.advanced(3));

            {
            auto &next_step=decision.consumed_drafts?continuation_one:continuation_zero;
            auto &next_program=decision.consumed_drafts?continuation_one_program:continuation_zero_program;
            auto continued=execute_checked_ordinary(caught,{next_step,next_program},
                use_target?state.member(caught,kTargetTap):next_targets[decision.consumed_drafts],initial_random.advanced(3));

            check(head.publish(caught.identity(),continued),"publish actual accepted-frontier fresh seed in the same Head");

            if(stochastic)EXPECT_EQ(random_cursor(take(head.current(),"next target result and RNG published together")),initial_random.advanced(4));
            const auto continued_image=image(state,continued);
            const auto seed=take(head.current(),"actual accepted fresh Pro seed");
            EXPECT_EQ(seed.identity(),continued.identity());EXPECT_EQ(state.epoch(seed).epoch,8u);
            EXPECT_EQ(state.epoch(seed).selection_position,std::uint64_t(next_step.position));
            EXPECT_EQ(state.epoch(seed).reuses,0u);
            auto &step_zero=decision.consumed_drafts?skip_one:skip_zero;
            auto &zero_program=decision.consumed_drafts?skip_one_program:skip_zero_program;
            const auto skipped=execute_observed_step(state,seed,step_zero,*source,8,
                [&](std::span<const c::ReadLease> input){return zero_program.execute(queue,input,cuda);});
            ++feedback;++skips;
            same_storage(state.member(skipped,kSelection),state.member(seed,kSelection));
            same_storage(state.member(skipped,kCodes),state.member(seed,kCodes));
            EXPECT_EQ(state.epoch(skipped).epoch,8u);EXPECT_EQ(state.epoch(skipped).reuses,1u);
            EXPECT_EQ(state.scalar(skipped,kLag),1);
            EXPECT_EQ(image(state,take(head.current(),"unverified step0 stays unpublished")),continued_image);
            EXPECT_EQ(image(state,published),published_image);EXPECT_EQ(image(state,first_checkpoint),first_checkpoint_image);
            EXPECT_EQ(image(state,second),second_image);EXPECT_EQ(image(state,base),base_image);EXPECT_EQ(image(state,first),first_image);
            EXPECT_FALSE(state.member(published,kPending).try_write());EXPECT_FALSE(state.member(continued,kMain).try_write());
            EXPECT_LE(state.peak_backing,budget);
            std::printf("GLM Pro K1 stochastic=%d scenario=%d proposal=%d target0=%d target1=%d accepted=%u frontier=%d pending=%d continued_pending=%d\n",
                int(stochastic),scenario,proposal,target_ids[0],target_ids[1],std::uint32_t(decision.consumed_drafts),state.scalar(published,kCursor),decision.next_id,state.scalar(continued,kPending));

            } // Retire request0 continuation/skip readers before reusing its pools.

          }
          check(state.store.collect(),"retire finite Pro scenario while retaining base/first readers");

          EXPECT_EQ(state.store.stats().versions,2u);EXPECT_EQ(state.store.stats().drafts,0u);
        }
      }
    }
    members={};initial_output.clear();
    check(state.store.collect(),"retire complete Pro state journey");
    EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
    if(!stochastic){EXPECT_GT(accepted[0],0u);EXPECT_GT(accepted[1],0u);}
    EXPECT_EQ(source->checked_steps,source->episodes);EXPECT_EQ(feedback,source->checked_steps);
    EXPECT_TRUE(source->full_checked);EXPECT_TRUE(source->moe_full_checked);
    std::size_t submissions=source->cold_submissions;
    add_bound(submissions,initial_program.submissions);add_bound(submissions,catchup_program.submissions);
    add_bound(submissions,verifier_program.submissions);
    for(const auto *program:model_programs)add_bound(submissions,program->native_submissions());
    if(use_target) {
      add_bound(submissions,target_source->cold_submissions);add_bound(submissions,target_program->native_submissions());
      add_bound(submissions,repair_program->native_submissions());add_bound(submissions,policy_program->submissions);
      EXPECT_TRUE(target_source->full_checked);EXPECT_TRUE(target_source->moe_full_checked);
      EXPECT_EQ(target_source->episodes,target_source->checked_steps);if(!stochastic)EXPECT_GT(repairs,0u);
      if constexpr(!Runtime::kPacked) {
      ::testing::Test::RecordProperty("original_target_roles",1559);
      ::testing::Test::RecordProperty("original_target_source_bytes","9872042432");
      ::testing::Test::RecordProperty("actual_target_calls",std::to_string(target_source->checked_steps));
      ::testing::Test::RecordProperty("actual_target_submissions",std::to_string(target_program->native_submissions()));
      ::testing::Test::RecordProperty("actual_target_policy_submissions",std::to_string(policy_program->submissions));
      ::testing::Test::RecordProperty("accepted_target_conditioned_repairs",std::to_string(repairs));
      ::testing::Test::RecordProperty("actual_accepted_kv_repair_submissions",std::to_string(repair_submissions));
      EXPECT_EQ(repair_submissions,repairs*2);
      }
    }
    if(stochastic) {
      EXPECT_EQ(accepted[0]+accepted[1],1u);EXPECT_EQ(cancelled,1u);
      ::testing::Test::RecordProperty("stochastic_policy","Actual NextN q samples proposal at draw0; actual target p0/q/p1 consumes accept1/final2. Same complete StateHead publishes pending and Cursor. Cancellation/stale publish advances0, successful K1 advances3; real target-only continuation consumes actual residual/bonus, samples its target pending and atomically advances one additional draw.");
      ::testing::Test::RecordProperty("random_accepted",int(accepted[1]));
      ::testing::Test::RecordProperty("random_rejected",int(accepted[0]));
      ::testing::Test::RecordProperty("random_actual_proposal",natural_proposal);
      ::testing::Test::RecordProperty("random_committed_draws",4);
      ::testing::Test::RecordProperty("actual_nextn_steps",std::to_string(source->checked_steps));
      ::testing::Test::RecordProperty("actual_target_calls",std::to_string(target_source->checked_steps));
      ::testing::Test::RecordProperty("actual_catchups",std::to_string(catchups));
      ::testing::Test::RecordProperty("actual_target_conditioned_repairs",std::to_string(repairs));
      ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(budget));
      ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(state.peak_backing));
      ::testing::Test::RecordProperty("scope",request_directory_demo?
          "Finite original target77+NextN78 natural K1 plus two-request/two-turn typed directory and FIFO adapter; original request tokens/pending/draw4 unchanged, second request independently resamples and continues actual target+NextN from the complete cached boundary. Previous layer input/residual/upstream74 index selection and initial history remain explicit fixtures. Not arbitrary prompt prefill, full-model generation, task quality, acceptance-rate measurement, TPS or other quantized source coverage.":
          "Finite original target77+NextN78 stochastic consumer and actual untied full vocabulary, ordinary existing two-phase Graphs/StateHead/supplier. Natural single trial only; no forced proposal, logits, p/q or branch. Previous layer input/residual/upstream74 index selection and initial history remain explicit fixtures. Not full-model generation, task quality, natural acceptance-rate measurement or other quantized source coverage.");
      return;
    }
    std::size_t ready=0;
    if constexpr(Runtime::kPacked) {ready=Runtime::ready(*source,*target_source,owner);}
    else {

    for(const auto &role:source->parameters.roles)if(role.lease) {
      add_bound(ready,role.lease.size());EXPECT_EQ(role.owner.ref(),owner.ref());
      auto permission=role.lease;EXPECT_FALSE(permission.try_write());
    }
    for(const auto &role:source->words.roles) {
      add_bound(ready,role.lease.size());EXPECT_EQ(role.owner.ref(),owner.ref());
      auto permission=role.lease;EXPECT_FALSE(permission.try_write());
    }
    for(const auto &role:source->prepared){add_bound(ready,role.lease.size());auto permission=role.lease;EXPECT_FALSE(permission.try_write());}
    EXPECT_FALSE(source->parameters.at(own::Role::kKvB).lease);
    if(use_target) {
      EXPECT_FALSE(target_source->parameters.at(target_own::Role::kKvB).lease);
      for(const auto &role:target_source->parameters.roles)if(role.lease) {
        EXPECT_EQ(role.owner.ref(),owner.ref());auto permission=role.lease;EXPECT_FALSE(permission.try_write());
      }
      ::testing::Test::RecordProperty("first_full_target_projection_reference_values",74304);
      ::testing::Test::RecordProperty("original_target_checked_calls",std::to_string(target_source->checked_steps));
      ::testing::Test::RecordProperty("original_target_expert_domain",256);
      ::testing::Test::RecordProperty("original_target_resident_capacity",8);
      ::testing::Test::RecordProperty("target_cache_members",3);
      ::testing::Test::RecordProperty("source_target_index_policy","Original shared-index layer77 has no indexer roles. Explicit synthetic upstream74 Top2048 IDs; no appended self/draft-tail or cloned78 indexer.");
    }
    }
    ::testing::Test::RecordProperty("actual_submissions",std::to_string(submissions));
    if constexpr(Runtime::kPacked) {
      const auto &artifact=source->parameters.source->identity();
      ::testing::Test::RecordProperty("artifact_identity",std::string(artifact.repository)+"@"+std::string(artifact.revision)+" "+std::string(artifact.variant));
      ::testing::Test::RecordProperty("artifact_repository",std::string(artifact.repository));
      ::testing::Test::RecordProperty("artifact_revision",std::string(artifact.revision));
      ::testing::Test::RecordProperty("artifact_variant",std::string(artifact.variant));
      ::testing::Test::RecordProperty("artifact_first_file",std::string(artifact.file));
      ::testing::Test::RecordProperty("expert_source_plan_sha256",source->experts.source->source_contract().plan_sha256);
      ::testing::Test::RecordProperty("static_source_roles",std::to_string(source->parameters.source->size()));
      std::size_t static_bytes=0;
      for(std::size_t i=0;i<source->parameters.source->size();++i)
        add_bound(static_bytes,source->parameters.source->spec(source->parameters.source->name(i)).bytes);
      ::testing::Test::RecordProperty("static_source_bytes",std::to_string(static_bytes));
      ::testing::Test::RecordProperty("actual_nextn_steps",std::to_string(source->checked_steps));
      ::testing::Test::RecordProperty("actual_target_calls",std::to_string(target_source->checked_steps));
      ::testing::Test::RecordProperty("actual_pending_embedding_feedback_checks",std::to_string(feedback));
      ::testing::Test::RecordProperty("actual_index_catchup_steps",std::to_string(catchups));
      ::testing::Test::RecordProperty("actual_skip_steps",std::to_string(skips));
      ::testing::Test::RecordProperty("accepted_zero_draft_checkpoints",std::to_string(accepted[0]));
      ::testing::Test::RecordProperty("accepted_one_draft_checkpoints",std::to_string(accepted[1]));
      ::testing::Test::RecordProperty("accepted_target_conditioned_repairs",std::to_string(repairs));
      ::testing::Test::RecordProperty("actual_accepted_kv_repair_submissions",std::to_string(repair_submissions));
      ::testing::Test::RecordProperty("natural_k1_trials",1);
      ::testing::Test::RecordProperty("natural_k1_accepted_drafts",int(natural_accepted));
      ::testing::Test::RecordProperty("controlled_matching_trials",1);
      ::testing::Test::RecordProperty("controlled_rejection_trials",1);
      ::testing::Test::RecordProperty("cancelled_unpublished_drafts",std::to_string(cancelled));
      ::testing::Test::RecordProperty("rejected_stale_publishes",std::to_string(stale_publish));
      ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(budget));
      ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(state.peak_backing));
      ::testing::Test::RecordProperty("target_first_full_reference_values",std::to_string(2*model::kVocabulary));
      ::testing::Test::RecordProperty("first_full_nextn_head_reference_values",std::to_string(model::kVocabulary));
      ::testing::Test::RecordProperty("ready_non_moe_bank_capacity_bytes",std::to_string(ready));
      ::testing::Test::RecordProperty("component_schema","13 resources + Epoch24: 10 F32/plain-index NextN resources and distinct target main/cursor/F32tap; no index scales member");
      ::testing::Test::RecordProperty("scope","Original converted target77+NextN78 packed roles with actual full-V Q8 verifier; explicit F32 compute/hidden/journal/main/index runtime, complete indexed fresh/skip/accepted-lag catchup. Upstream74 IDs/layer76 input+residual/initial histories are finite fixtures; not complete target-engine quality or TPS.");
    } else {
    ::testing::Test::RecordProperty("full_original_nextn_roles",1571);
    ::testing::Test::RecordProperty("full_original_nextn_source_bytes","13838963840");
    ::testing::Test::RecordProperty("actual_nextn_steps",std::to_string(source->checked_steps));
    ::testing::Test::RecordProperty("actual_pending_embedding_feedback_checks",std::to_string(feedback));
    ::testing::Test::RecordProperty("actual_index_catchup_steps",std::to_string(catchups));
    ::testing::Test::RecordProperty("actual_skip_steps",std::to_string(skips));
    ::testing::Test::RecordProperty("accepted_zero_draft_checkpoints",std::to_string(accepted[0]));
    ::testing::Test::RecordProperty("accepted_one_draft_checkpoints",std::to_string(accepted[1]));
    ::testing::Test::RecordProperty("natural_k1_trials",1);
    ::testing::Test::RecordProperty("natural_k1_proposal",natural_proposal);
    ::testing::Test::RecordProperty("natural_k1_accepted_drafts",int(natural_accepted));
    ::testing::Test::RecordProperty("controlled_matching_trials",1);
    ::testing::Test::RecordProperty("controlled_rejection_trials",1);
    ::testing::Test::RecordProperty("cancelled_unpublished_drafts",std::to_string(cancelled));
    ::testing::Test::RecordProperty("rejected_stale_publishes",std::to_string(stale_publish));
    ::testing::Test::RecordProperty("first_full_nextn_head_reference_values",std::to_string(model::kVocabulary));
    ::testing::Test::RecordProperty("target_first_full_reference_values",std::to_string(2*model::kVocabulary));
    ::testing::Test::RecordProperty("target_verifier_submissions",std::to_string(verifier_program.submissions));
    ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(budget));
    ::testing::Test::RecordProperty("state_pool_backing_bound",std::to_string(pools));
    ::testing::Test::RecordProperty("state_initial_backing_bound",std::to_string(initial_bound));
    ::testing::Test::RecordProperty("state_metadata_backing_bound",std::to_string(metadata_bound));
    ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(state.peak_backing));
    ::testing::Test::RecordProperty("ready_non_moe_bank_capacity_bytes",std::to_string(ready));
    ::testing::Test::RecordProperty("prepared_kv_capacity_bytes",std::to_string(source->prepared[0].lease.size()+source->prepared[1].lease.size()));
    ::testing::Test::RecordProperty("resident_expert_domain",int(kMoe.experts));
    ::testing::Test::RecordProperty("resident_expert_capacity",int(kMoe.slots));
    ::testing::Test::RecordProperty("resident_slot_capacity_bytes",std::to_string(glm_moe_geometry::layout(kMoe).slot_bytes));
    std::string unions;
    for(const auto &selected:source->selections){if(!unions.empty())unions+=",";unions+=std::to_string(selected.globals.size());}
    ::testing::Test::RecordProperty("actual_resident_union_sizes",unions);
    ::testing::Test::RecordProperty("component_schema",use_target?
        "14 resources plus Epoch24: existing11 NextN plus original77 mainF32[2064,576] exactBF16widen, targetcursorI32 and normalizedtapBF16[1,6144]; one StateStore/Head":
        "11 resources plus Epoch24 metadata: mainF32[2064,576] BF16widen CKV512+KPE64, unpooled E4[2064,128]+F32scale, BF16hidden6144, I32pending/cursors/2048absoluteIDs, BF16journal3 and positions/lag; one StateHead");
    ::testing::Test::RecordProperty("pending_policy",use_target?
        "Actual77 target-current/proposal taps enter one R2 full-V projection. Pending is direct stable row[accepted] lease; accept0 retains current; accept1 repairs proposal78 KV/journal from target-current tap and retains actual second target main/cursor/tap. Accepted lag is caught up; current bonus then runs real77+78 with targettap seed; unverified skip0 remains unpublished.":
        "Natural K1 uses actual NextN proposal and full-V R2 target-tap head. Controlled match/reject consume real target Top1/Top2 proposal leases in the actual second Graph, separate from natural acceptance. accept0 retains after-current first; accept1 after-proposal second; pending directly retains target row[accepted]. Only accepted lag1 is caught up; next fresh epoch8 publishes before unverified skip0 remains unpublished.");
    ::testing::Test::RecordProperty("scope",use_target?
        "Distinct original GLM-5.3 aca966 target77 1559 roles including its model.norm, plus original78 complete1571 roles and the same untied154880 BF16 head/embedding. Shared-index77 uses explicit upstream74 Top2048; target main576/cursor/finalnormalizedtap and draft11 resources share one expected-base StateHead. Actual token leases select original embeddings only as a labelled synthetic layer76 hidden adapter; residual, upstream74 selection and finite initial caches remain fixtures. Warm/current/proposal/bonus real77 MLA and256-domain un-clipped MoE produce true model.norm taps. Natural K1 and real Top1/Top2 controls are separate. Accept0 keeps after-current; accept1 recomputes accepted proposal78 KV/journal with target-current tap before atomic publication and accepted index catchup. Same ordinary two-phase execution and bounded supplies. Not full78-layer target, task quality, vendor fusion-bit equivalence, TPS or measured physical peak.":
        "Official original GLM-5.3 aca966 Pro layer78 complete1571 roles including untied full154880 BF16 vocabulary. Original H6144 Qrank2048 NoPE192+RoPE64/latent512/value256, unpooled indexTop2048 and non-fused Hadamard/BF16/UE8M0, BF16weights_proj F32acc, original256-domain Top8 MoE with actual selected E4/SF experts, scale2.5 and no Flash clip10. Ordinary two-phase Graphs, same native resource and bounded double-slot supply, direct activation/state leases, all actual store/route/merge/fullV gates. Initial prepared history and previous/final target taps are explicit synthetic external fixtures; not a complete78-layer trained target, vendor physical-cache ABI, quality/natural-acceptance benchmark or TPS. Capacity/upload accounting is not measured physical peak.");
    }
  }();
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if constexpr(!Runtime::kPacked) if(source->cuda_supply)check(source->cuda_supply->close(),"close original Pro uploads after all ordinary readers retire");
#endif
  if(source->cpu_supply) {
    EXPECT_EQ(source->cpu_supply->stats().live_blocks,0u);EXPECT_EQ(source->cpu_supply->stats().active_uses,0u);
  }
  if(target_source) {
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    if constexpr(!Runtime::kPacked) if(target_source->cuda_supply)check(target_source->cuda_supply->close(),"retire original77 bounded expert supply");
#endif
    if(target_source->cpu_supply) {
      EXPECT_EQ(target_source->cpu_supply->stats().live_blocks,0u);EXPECT_EQ(target_source->cpu_supply->stats().active_uses,0u);
    }
  }
}

#if defined(HYLE_TEST_GLM_STATE_CUDA)
struct SharedCudaDevice final {
  std::shared_ptr<c::cuda::MemoryResource> resource;
  m::MemoryResourceRef ref() noexcept{return resource->ref();}
};
TEST(GlmProMtpStateConsumer,CudaOriginalTarget77AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"original77/78 shared CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original77/78 same CUDA allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"original77/78 same CUDA owner");
  run(context,owner,true,true,device.get());
}
TEST(GlmProMtpStateConsumer,CudaOriginalTarget77AndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cuda::create_context(),"original random77/78 shared CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original random77/78 allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"original random77/78 shared owner");
  run(context,owner,true,true,device.get(),nullptr,true,{},1,true);
}
TEST(GlmProMtpStateConsumer,CudaOriginalTarget77AndNextNRandomK2PublishesCompletePrefixes) {
  auto context=take(g::cuda::create_context(),"original K2 77/78 CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original K2 Pro allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"original K2 Pro shared owner");
  run(context,owner,true,true,device.get(),nullptr,true,{},2);
}
TEST(GlmProMtpStateConsumer,CudaOriginalNextNFullVocabularyPublishesAndCatchesUp) {
  auto context=take(g::cuda::create_context(),"original Pro CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"same actual original Pro CUDA allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared original Pro CUDA owner");
  run(context,owner,true,false,device.get());
}
TEST(GlmProMtpStateConsumer,CudaConvertedB09TargetAndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"B09 CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"B09 actual device"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"B09 actual shared owner");
  run<ConvertedRuntime>(context,owner,true,true,device.get());
}
TEST(GlmProMtpStateConsumer,CudaConvertedB09TargetAndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cuda::create_context(),"B09 CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"B09 actual device"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"B09 actual shared owner");
  run<ConvertedRuntime>(context,owner,true,true,device.get(),nullptr,true);
}
TEST(GlmProMtpStateConsumer,CudaConvertedB10TargetAndNextNFullVocabularyVerifier) {
  const auto &fixture=converted_b10_fixture();
  auto context=take(g::cuda::create_context(),"B10 CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"B10 actual device"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"B10 actual shared owner");
  run<ConvertedRuntime>(context,owner,true,true,device.get(),&fixture);
}

#endif
#if defined(HYLE_TEST_GLM_STATE_CPU)
struct SystemResource final {m::MemoryResourceRef ref() noexcept{return m::system_resource();}};
TEST(GlmProMtpStateConsumer,CpuOriginalTarget77AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"original77/78 CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"original77/78 CPU owner");
  run(context,owner,false,true);
}
TEST(GlmProMtpStateConsumer,CpuOriginalTarget77AndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cpu::create_context(),"original random77/78 CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"original random77/78 CPU owner");
  run(context,owner,false,true
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,nullptr,true);
}
TEST(GlmProMtpStateConsumer,CpuOriginalNextNFullVocabularyPublishesAndCatchesUp) {
  auto context=take(g::cpu::create_context(),"original Pro CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"shared original Pro CPU owner");
  run(context,owner,false);
}
TEST(GlmProMtpStateConsumer,CpuConvertedB09TargetAndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"B09 CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"B09 actual CPU owner");
  run<ConvertedRuntime>(context,owner,false,true);
}
TEST(GlmProMtpStateConsumer,CpuConvertedB09TargetAndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cpu::create_context(),"B09 CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"B09 actual CPU owner");
  run<ConvertedRuntime>(context,owner,false,true
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,nullptr,true);
}
TEST(GlmProMtpStateConsumer,CpuConvertedB10TargetAndNextNFullVocabularyVerifier) {
  const auto &fixture=converted_b10_fixture();
  auto context=take(g::cpu::create_context(),"B10 CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"B10 actual CPU owner");
  run<ConvertedRuntime>(context,owner,false,true
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,&fixture);
}

#endif

} // namespace
