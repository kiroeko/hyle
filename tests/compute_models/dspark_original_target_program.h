#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "cpu_c4_model_branches.h"
#include "dspark_original_target_fragment.h"
#include "dspark_original_target_roles.h"

// Cold wiring of the original three-layer target tail. The caller executes the cold
// OA preparation once, binds its three actual leases, then cuts/runs this
// ordinary Fragment at the three actual router frontiers. No execution,
// resource allocation, acceptance policy or host model math is hidden here.
namespace dspark_original_target_program {
namespace roles=dspark_original_target_roles;
namespace math=dspark_original_target_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace base=cpu_c4_model_branches;
namespace c=hyle::compute;
using Value=sem::ValueId;
using Role=roles::Role;
using FinalRole=roles::FinalRole;

struct Constant final {Value value;roles::ReadonlyRole role;};
struct Preparation final {
  sem::Fragment fragment{.profile=&base::flash_profile,.region=0x0b5a0001,.quantized=true};
  std::shared_ptr<const void> source;
  std::vector<Constant> constants;
};

inline Value immutable(sem::Fragment &fragment,std::vector<Constant> &constants,
    const roles::ReadonlyRole &role,const c::TensorDesc &expected) {
  if(!role.owner||!role.lease||role.description!=expected)
    throw std::runtime_error("original target immutable role has a different layout or no ready lease");
  const auto value=fragment.constant(expected,{});
  constants.push_back({value,role});
  return value;
}

template<class Source>
Preparation prepare_output_a(const roles::TargetBindings<Source> &bound,
    std::uint64_t region=0x0b5a0001,
    const dspark_target_geometry::Geometry &geometry=dspark_target_geometry::kFlash,
    const c::ModelArtifactProfileDescription *profile=&base::flash_profile) {
  if(!bound.source)throw std::runtime_error("original target OA preparation needs its raw source owner");
  geometry.validate();
  if(!profile)throw std::runtime_error("original target OA preparation needs its actual model profile");
  Preparation result;result.source=bound.source;result.fragment.region=region;
  result.fragment.profile=profile;
  auto &f=result.fragment;
  for(std::size_t stage=0;stage<roles::kStages;++stage) {
    const auto encoded=immutable(f,result.constants,bound.at(stage,Role::kOutputA),
        c::float8_weight_description(geometry.group_width(),geometry.groups*math::kOutputRank,
            c::Float8ScaleMode::kFloat32Block128).value());
    const auto decoded=f.one(c::model_ops::float8_weight_decode(c::DataType::kBFloat16),{encoded});
    const auto grouped=math::reshape(f,decoded,{geometry.groups,math::kOutputRank,geometry.group_width()});
    const auto transposed=f.one(c::Operation(c::View::permute(std::array<std::int32_t,3>{0,2,1}).value()),{grouped});
    const auto ready=math::copy(f,transposed);
    math::require(f,ready,sem::dense({geometry.groups,geometry.group_width(),math::kOutputRank},c::DataType::kBFloat16));
    f.outputs.push_back(ready);
  }
  return result;
}

struct Recipe final {
  sem::Fragment fragment{.profile=&base::flash_profile,.region=0x0b5a0100,.quantized=true};
  // Preserve the original mapped source and reference lifetime; cold graph
  // construction consumes only Values and their immutable ready leases.
  std::shared_ptr<const void> source;
  Value inputBF16;
  std::array<math::LayerState,3> state_inputs;
  std::array<math::LayerResult,3> layers;
  math::HeadResult head;
  std::vector<Constant> constants;
  std::array<math::Boundary,3> routers;
};
inline constexpr std::size_t kStateInputs=21,kSemanticInputs=22,kInputs=34;
inline constexpr std::size_t kStateOutputs=21,kCaptureFirst=21,kNormalizedPort=24,kSemanticOutputs=25;

template<class Source>
Recipe build(const roles::TargetBindings<Source> &bound,
    const std::array<roles::ReadonlyRole,3> &prepared_oa,math::Plan plan,
    std::uint64_t region=0x0b5a0100,
    const c::ModelArtifactProfileDescription *profile=&base::flash_profile) {
  if(!bound.source)throw std::runtime_error("original target tail needs its original parameter owner");
  const auto &geometry=plan.geometry;
  (void)math::Plan::make(plan.position,plan.capacity,geometry);
  if(!profile)throw std::runtime_error("original target tail needs its actual model profile");
  Recipe result;result.source=bound.source;result.fragment.region=region;
  result.fragment.profile=profile;
  auto &f=result.fragment;
  f.phase=c::WorkloadPhase::kDecode;
  result.inputBF16=f.input(sem::dense({1,4,geometry.hidden},c::DataType::kBFloat16));
  for(std::size_t stage=0;stage<roles::kStages;++stage)
    result.state_inputs[stage]=math::state_inputs(f,roles::compression_ratio(stage),plan.capacity,geometry);
  if(f.inputs.size()!=kSemanticInputs)throw std::runtime_error("original target input state schema differs");
  std::array<std::array<Value,3>,3> banks;
  std::array<Value,3> maps;
  for(std::size_t stage=0;stage<roles::kStages;++stage) {
    for(std::size_t leg=0;leg<3;++leg) {
      const auto outputs=leg==2?geometry.hidden:geometry.intermediate;
      const auto reduction=leg==2?geometry.intermediate:geometry.hidden;
      banks[stage][leg]=f.input(c::float4_grouped_weight_description(6,outputs,reduction).value());
    }
    maps[stage]=f.input(sem::dense({geometry.experts},c::DataType::kInt32));
  }
  if(f.inputs.size()!=kInputs)throw std::runtime_error("original target resident input schema differs");

  std::array<std::array<std::optional<Value>,roles::kRoles>,3> values;
  const auto parameter=[&](std::size_t stage,Role role) {
    auto &value=values[roles::stage_index(stage)][roles::index(role)];
    if(!value)value=immutable(f,result.constants,bound.at(stage,role),bound.at(stage,role).description);
    return *value;
  };
  const auto compressor=[&](std::size_t stage,bool indexer) {
    return math::CompressorParameters{
        parameter(stage,indexer?Role::kIndexerKey:Role::kCompressorKey),
        parameter(stage,indexer?Role::kIndexerGate:Role::kCompressorGate),
        parameter(stage,indexer?Role::kIndexerApe:Role::kCompressorApe),
        parameter(stage,indexer?Role::kIndexerNorm:Role::kCompressorNorm)};
  };
  auto actual=result.inputBF16;
  for(std::size_t stage=0;stage<roles::kStages;++stage) {
    math::LayerParameters supplied;
    supplied.attention_hc={
        {parameter(stage,Role::kAttentionHcFn),parameter(stage,Role::kAttentionHcScale),
         parameter(stage,Role::kAttentionHcBase)},parameter(stage,Role::kAttentionNorm)};
    supplied.ffn_hc={
        {parameter(stage,Role::kFfnHcFn),parameter(stage,Role::kFfnHcScale),
         parameter(stage,Role::kFfnHcBase)},parameter(stage,Role::kFfnNorm)};
    const auto output_a=immutable(f,result.constants,prepared_oa[stage],
        sem::dense({geometry.groups,geometry.group_width(),math::kOutputRank},c::DataType::kBFloat16));
    supplied.attention={parameter(stage,Role::kQueryA),parameter(stage,Role::kQueryNorm),
        parameter(stage,Role::kQueryB),parameter(stage,Role::kKv),parameter(stage,Role::kKvNorm),
        output_a,parameter(stage,Role::kOutputB),parameter(stage,Role::kAttentionSink),
        compressor(stage,false),{}};
    if(roles::compression_ratio(stage)==4)
      supplied.attention.indexer=math::IndexerParameters{
          parameter(stage,Role::kIndexerQuery),parameter(stage,Role::kIndexerWeights),compressor(stage,true)};
    supplied.moe={parameter(stage,Role::kRouter),parameter(stage,Role::kRouterBias),
        {parameter(stage,Role::kSharedGate),parameter(stage,Role::kSharedUp),parameter(stage,Role::kSharedDown)},
        banks[stage],maps[stage]};
    result.layers[stage]=math::append_layer(f,actual,result.state_inputs[stage],supplied,
        roles::compression_ratio(stage),plan);
    result.routers[stage]=result.layers[stage].boundary;
    actual=result.layers[stage].output;
  }
  const auto final_parameter=[&](FinalRole role) {
    return immutable(f,result.constants,bound.at(role),bound.at(role).description);
  };
  result.head=math::append_head(f,actual,{final_parameter(FinalRole::kFn),final_parameter(FinalRole::kScale),
      final_parameter(FinalRole::kBase),final_parameter(FinalRole::kNorm)});

  // Preserve actual router frontiers before materializing observations. The
  // cut algorithm must see real input/IDs/weights producers at each phase.
  const auto actual_values=f.descriptions.size();
  std::vector<std::optional<Value>> copies(actual_values);
  const auto own=[&](Value &value) {
    const auto original=value.ordinal();
    if(original>=actual_values)return; // already a materialized observation
    auto &ready=copies[original];
    if(!ready) {
      ready=math::copy(f,value);
      f.outputs.push_back(*ready);
    }
    value=*ready;
  };
  // First25 outputs are the stable state/capture/tap ABI used by the caller.
  for(auto &layer:result.layers)layer.state.visit_values(own);
  if(f.outputs.size()!=kStateOutputs)throw std::runtime_error("original target output state schema differs");
  for(auto &layer:result.layers)own(layer.capture);
  own(result.head.normalized);
  if(f.outputs.size()!=kSemanticOutputs)throw std::runtime_error("original target semantic output schema differs");
  for(auto &layer:result.layers) {
    own(layer.output);
    layer.ports.visit_values(own);
  }
  result.head.ports.visit_values(own);
  return result;
}
} // namespace dspark_original_target_program
