#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "cpu_c4_model_branches.h"
#include "dspark_original_draft_fragment.h"
#include "dspark_original_formal_roles.h"

// Named formal parameter wiring into one ordinary Fragment. Source ownership
// and ready device leases are shared by target/draft/vocabulary; construction
// neither uploads weights nor executes model operations. The caller retains
// the existing four-phase router cut, native runner and StateHead.
namespace dspark_original_draft_program {
namespace roles=dspark_original_formal_roles;
namespace math=dspark_original_draft_fragment;
namespace common=dspark_formal_fragment;
namespace sem=math::sem;
namespace c=math::c;
namespace base=cpu_c4_model_branches;
using Value=math::Value;
struct Constant final {Value value;roles::ReadonlyRole role;};
template<class Visit> void materialize(sem::Fragment &f,Visit visit) {
  const auto actual_values=f.descriptions.size();
  std::vector<std::optional<Value>> copies(actual_values);
  for(auto &value:f.outputs) {
    auto &ready=copies.at(value.ordinal());
    if(!ready)ready=common::copy(f,value);
    value=*ready;
  }
  const auto observe=[&](Value &value) {
    const auto original=value.ordinal();if(original>=actual_values)return;
    auto &ready=copies[original];
    if(!ready) {ready=common::copy(f,value);f.outputs.push_back(*ready);}
    value=*ready;
  };
  visit(observe);
}
inline Value immutable(sem::Fragment &f,std::vector<Constant> &constants,const roles::ReadonlyRole &role,
    const c::TensorDesc &expected) {
  if(!role.owner||!role.lease||role.description!=expected||role.lease.size()!=expected.required_bytes().value())
    throw std::runtime_error("formal DSpark immutable role has a different typed layout or ready extent");
  const auto permission=role.lease.tensor(expected);
  if(!permission||permission->storage().resource()!=role.owner.ref())
    throw std::runtime_error("formal DSpark constant owner differs from its actual resource");
  const auto value=f.constant(expected,{});constants.push_back({value,role});return value;
}
inline const c::ModelArtifactProfileDescription *checked_profile(const roles::Data &data,
    const c::ModelArtifactProfileDescription *profile) {
  if(!profile||profile->model_repository!=data.model().repository||profile->model_revision!=data.model().revision||
      profile->artifact_repository!=data.model().repository||profile->artifact_revision!=data.model().revision)
    throw std::runtime_error("formal source and actual Graph model profile differ");
  return profile;
}
struct Preparation final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10001,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
};
inline Preparation prepare_output_a(const roles::Bound &bound,std::uint64_t region=0x0ca10001,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile) {
  if(!bound.source)throw std::runtime_error("formal DSpark outputA preparation needs its actual source");
  Preparation result;result.source=bound.source;result.fragment.region=region;result.fragment.profile=checked_profile(*bound.source,profile);auto &f=result.fragment;
  const auto &model=bound.source->geometry();
  for(std::size_t stage=0;stage<3;++stage) {
    const auto &role=bound.at(roles::Section::kDraft,stage,"attn.wo_a.weight");
    const auto weight=immutable(f,result.constants,role,c::float8_weight_description(
        model.group_width(),model.groups*common::geometry::kOutputRank,c::Float8ScaleMode::kFloat32Block128).value());
    const auto decoded=f.one(c::model_ops::float8_weight_decode(c::DataType::kBFloat16),{weight});
    const auto grouped=common::reshape(f,decoded,{model.groups,common::geometry::kOutputRank,model.group_width()});
    const auto rhs=f.one(c::Operation(c::View::permute(std::array<std::int32_t,3>{0,2,1}).value()),{grouped});
    const auto ready=common::copy(f,rhs);
    common::require(f,ready,sem::dense({model.groups,model.group_width(),common::geometry::kOutputRank},c::DataType::kBFloat16));
    f.outputs.push_back(ready);
  }
  return result;
}
struct Recipe final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10100,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
  math::Inputs inputs;
  math::Result result;
};
inline constexpr std::size_t kSemanticInputs=6,kInputs=18,kSemanticOutputs=7;
// Exactly the older complete-draft semantic port order, with original storage
// boundaries visible in each output descriptor rather than silently widened.
inline Recipe build(const roles::Bound &bound,const std::array<roles::ReadonlyRole,3> &prepared_oa,
    std::uint64_t region=0x0ca10100,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile) {
  if(!bound.source)throw std::runtime_error("formal DSpark draft needs its actual parameter source");
  Recipe recipe;recipe.source=bound.source;recipe.fragment.region=region;recipe.fragment.profile=checked_profile(*bound.source,profile);auto &f=recipe.fragment;
  f.phase=c::WorkloadPhase::kDecode;const auto &model=bound.source->geometry();model.validate();
  recipe.inputs.pending=f.input(sem::dense({1},c::DataType::kInt32));
  recipe.inputs.positions=f.input(sem::dense({math::kRows},c::DataType::kInt32));
  recipe.inputs.mask=f.input(sem::dense({math::kRows,math::kWindow+math::kRows}));
  for(auto &history:recipe.inputs.histories)history=f.input(c::row_cache_carrier_description(math::kEncoding,math::kWindow).value());
  if(f.inputs.size()!=kSemanticInputs)throw std::runtime_error("formal DSpark semantic input schema differs");
  math::Parameters parameters;
  for(std::size_t stage=0;stage<3;++stage) {
    for(std::size_t leg=0;leg<3;++leg) {
      const auto n=leg==2?model.hidden:model.intermediate,k=leg==2?model.intermediate:model.hidden;
      parameters.stages[stage].moe.banks[leg]=f.input(c::float4_grouped_weight_description(
          math::kRows*common::geometry::kSlots,n,k).value());
    }
    parameters.stages[stage].moe.map=f.input(sem::dense({model.experts},c::DataType::kInt32));
  }
  if(f.inputs.size()!=kInputs)throw std::runtime_error("formal DSpark resident input schema differs");
  std::unordered_map<std::string_view,Value> values;
  const auto parameter=[&](const roles::ReadonlyRole &role) {
    const auto found=values.find(role.name);if(found!=values.end())return found->second;
    const auto value=immutable(f,recipe.constants,role,role.description);values.emplace(role.name,value);return value;
  };
  const auto stage_parameter=[&](std::size_t stage,std::string_view suffix) {
    return parameter(bound.at(roles::Section::kDraft,stage,suffix));
  };
  parameters.embedding=parameter(bound.at("embed.weight"));parameters.head=parameter(bound.at("head.weight"));
  parameters.markov_w1=stage_parameter(2,"markov_head.markov_w1.weight");
  parameters.markov_w2=stage_parameter(2,"markov_head.markov_w2.weight");
  parameters.markov_rank=f.description(parameters.markov_w1).dimensions().back();
  parameters.confidence=stage_parameter(2,"confidence_head.proj.weight");
  parameters.head_hc={stage_parameter(2,"hc_head_fn"),stage_parameter(2,"hc_head_scale"),
      stage_parameter(2,"hc_head_base"),stage_parameter(2,"norm.weight")};
  for(std::size_t stage=0;stage<3;++stage) {
    auto &p=parameters.stages[stage];
    p.attention_hc={{stage_parameter(stage,"hc_attn_fn"),stage_parameter(stage,"hc_attn_scale"),stage_parameter(stage,"hc_attn_base")},
        stage_parameter(stage,"attn_norm.weight")};
    p.ffn_hc={{stage_parameter(stage,"hc_ffn_fn"),stage_parameter(stage,"hc_ffn_scale"),stage_parameter(stage,"hc_ffn_base")},
        stage_parameter(stage,"ffn_norm.weight")};
    const auto oa=immutable(f,recipe.constants,prepared_oa[stage],sem::dense(
        {model.groups,model.group_width(),common::geometry::kOutputRank},c::DataType::kBFloat16));
    p.attention={stage_parameter(stage,"attn.wq_a.weight"),stage_parameter(stage,"attn.q_norm.weight"),
        stage_parameter(stage,"attn.wq_b.weight"),stage_parameter(stage,"attn.wkv.weight"),stage_parameter(stage,"attn.kv_norm.weight"),
        oa,stage_parameter(stage,"attn.wo_b.weight"),stage_parameter(stage,"attn.attn_sink")};
    p.moe.router=stage_parameter(stage,"ffn.gate.weight");p.moe.bias=stage_parameter(stage,"ffn.gate.bias");
    p.moe.shared={stage_parameter(stage,"ffn.shared_experts.w1.weight"),stage_parameter(stage,"ffn.shared_experts.w3.weight"),
        stage_parameter(stage,"ffn.shared_experts.w2.weight")};
  }
  recipe.result=math::append(f,recipe.inputs,parameters,model);
  auto &p=recipe.result;
  f.outputs={p.state,p.hidden,p.normalized,p.ids,p.markov,p.logits,p.confidence};
  // Frontier identities stay original; every semantic/diagnostic value gets
  // an independently owned ordinary Copy for retention after reset.
  if(f.outputs.size()!=kSemanticOutputs)throw std::runtime_error("formal DSpark semantic output schema differs");
  materialize(f,[&](auto observe){p.visit_values(observe);});
  return recipe;
}

struct MainRecipe final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10200,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
  std::array<Value,3> captures;
  math::MainPorts ports;
};
inline MainRecipe build_main_projection(const roles::Bound &bound,std::int64_t rows,std::uint64_t region=0x0ca10200,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile) {
  if(!bound.source||rows<=0)throw std::runtime_error("formal DSpark capture projection needs its source and rows");
  MainRecipe result;result.source=bound.source;result.fragment.region=region;result.fragment.profile=checked_profile(*bound.source,profile);auto &f=result.fragment;const auto &model=bound.source->geometry();
  for(auto &capture:result.captures)capture=f.input(sem::dense({rows,4,model.hidden}));
  const auto &weight=bound.at(roles::Section::kDraft,0,"main_proj.weight");
  const auto &norm=bound.at(roles::Section::kDraft,0,"main_norm.weight");
  result.ports=math::main_projection(f,result.captures,{immutable(f,result.constants,weight,weight.description),
      immutable(f,result.constants,norm,norm.description)},model);
  f.outputs={result.ports.norm.stored};
  materialize(f,[&](auto observe){result.ports.visit_values(observe);});
  return result;
}
struct HistoryRecipe final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10300,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
  Value main,positions,write_rows;
  std::array<Value,3> histories;
  math::MainHistoryPorts ports;
};
inline HistoryRecipe build_main_history(const roles::Bound &bound,std::int64_t rows,std::uint64_t region=0x0ca10300,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile) {
  if(!bound.source||rows<=0)throw std::runtime_error("formal DSpark main history needs its source and rows");
  HistoryRecipe result;result.source=bound.source;result.fragment.region=region;result.fragment.profile=checked_profile(*bound.source,profile);auto &f=result.fragment;const auto &model=bound.source->geometry();
  result.main=f.input(sem::dense({rows,model.hidden},c::DataType::kBFloat16));
  result.positions=f.input(sem::dense({rows},c::DataType::kInt32));result.write_rows=f.input(sem::dense({rows},c::DataType::kInt32));
  for(auto &history:result.histories)history=f.input(c::row_cache_carrier_description(math::kEncoding,math::kWindow).value());
  std::array<math::MainHistoryParameters,3> parameters;
  for(std::size_t stage=0;stage<3;++stage) {
    const auto &weight=bound.at(roles::Section::kDraft,stage,"attn.wkv.weight");
    const auto &norm=bound.at(roles::Section::kDraft,stage,"attn.kv_norm.weight");
    parameters[stage]={immutable(f,result.constants,weight,weight.description),immutable(f,result.constants,norm,norm.description)};
  }
  result.ports=math::main_history(f,result.main,result.positions,result.write_rows,result.histories,parameters,model);
  for(const auto &snapshot:result.ports.checkpoints)for(const auto value:snapshot)f.outputs.push_back(value);
  materialize(f,[&](auto observe){result.ports.visit_values(observe);});
  return result;
}
} // namespace dspark_original_draft_program
