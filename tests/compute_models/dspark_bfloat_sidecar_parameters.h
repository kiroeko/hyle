#pragma once

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "dspark_bfloat_sidecar_roles.h"
#include "dspark_bfloat_sidecar_reference.h"
#include "dspark_original_state_consumer.h"

// Cold parameter assembly for the existing complete draft. No scheduling,
// provider selection or state controller is added. Source/ready lease owners
// are retained independently of the Graph-building callback objects.
namespace dspark_bfloat_sidecar_parameters {
namespace roles=dspark_bfloat_sidecar_roles;
namespace ref=dspark_bfloat_sidecar_reference;
namespace consumer=dspark_original_state_consumer;
namespace base=consumer::base;
namespace sem=consumer::sem;
namespace c=hyle::compute;
namespace b=dspark_target_binding;
struct State final {
  std::shared_ptr<const roles::Bound> bound;
  std::array<b::ReadonlyRole,3> output_a;
  std::vector<consumer::ExternalConstant> constants;
  sem::ValueId parameter(sem::Fragment &f,const b::ReadonlyRole &role) {
    if(!role.owner||!role.lease||role.lease.size()!=role.description.required_bytes().value())
      throw std::runtime_error("S03 cold parameter has no complete original readonly lease");
    const auto tensor=roles::source::take(role.lease.tensor(role.description),"S03 readonly parameter tensor");
    if(tensor.storage().resource()!=role.owner.ref())
      throw std::runtime_error("S03 readonly parameter owner differs from its actual resource");
    const auto value=f.constant(role.description,{});constants.push_back({value,role});return value;
  }
  sem::ValueId parameter(sem::Fragment &f,std::string_view name) {return parameter(f,bound->at(name));}
};
struct Bindings final {
  std::shared_ptr<State> state;
  base::DraftParameters parameters;
  consumer::d::HeadParameters head;
};
inline Bindings make(std::shared_ptr<const roles::Bound> bound,std::array<b::ReadonlyRole,3> prepared_output_a) {
  if(!bound||!bound->source)throw std::runtime_error("S03 complete original parameter source missing");
  const auto prepared=c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t,3>{8,4096,1024}).value();
  for(std::size_t stage=0;stage<3;++stage) {
    const auto &role=prepared_output_a[stage];
    if(role.name!=roles::stage_name(stage,"attn_output_a.weight")||role.description!=prepared||
        !role.owner||!role.lease||role.lease.size()!=prepared.required_bytes().value())
      throw std::runtime_error("S03 output-a needs the actual once-prepared BF16 RHS lease");
  }
  auto state=std::make_shared<State>();state->bound=std::move(bound);state->output_a=std::move(prepared_output_a);
  Bindings result{.state=state};
  result.parameters.parameter=[state](sem::Fragment &f,std::size_t stage,base::DraftParameter parameter) {
    const auto index=consumer::parameter_index(stage,parameter);
    const auto name=consumer::parameters::spec(index).name;
    return base::ds::ImmutableF32Parameter{state->parameter(f,name),state->bound->source->floats(name)};
  };
  result.parameters.shared_ffn=[state](sem::Fragment &f,std::size_t stage,base::ds::official::SharedFfnRole role) {
    constexpr std::array<std::string_view,3> suffix{
        "ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
    if(std::size_t(role)>=suffix.size())throw std::runtime_error("S03 shared role outside original source");
    const auto name=roles::stage_name(stage,suffix[std::size_t(role)]);
    const auto &bank=state->bound->at(name);const auto description=bank.description;
    const auto raw=state->bound->source->raw(name);
    return base::ds::official::PackedProjectionParameter{state->parameter(f,bank),raw,
        [source=state->bound->source,name](std::span<const float> input) {
          const auto description=source->description(name);const auto shape=description.dimensions();
          return ref::decoded_reference(source->raw(name),std::size_t(shape[0]),std::size_t(shape[1]),input);
        },description};
  };
  result.parameters.attention_projection=[state](sem::Fragment &f,std::size_t stage,base::AttentionProjectionRole role) {
    constexpr std::array<std::string_view,4> suffix{
        "attn_q_a.weight","attn_q_b.weight","attn_output_a.weight","attn_output_b.weight"};
    if(std::size_t(role)>=suffix.size())throw std::runtime_error("S03 attention role outside original source");
    const auto name=roles::stage_name(stage,suffix[std::size_t(role)]);
    const bool grouped=role==base::AttentionProjectionRole::kOutputA;
    const auto &bank=grouped?state->output_a.at(stage):state->bound->at(name);
    return base::ds::official::PackedProjectionParameter{state->parameter(f,bank),state->bound->source->raw(name),
        [source=state->bound->source,name,grouped](std::span<const float> input) {
          if(grouped)return ref::decoded_grouped_reference(source->raw(name),8,1024,4096,input);
          const auto description=source->description(name);const auto shape=description.dimensions();
          return ref::decoded_reference(source->raw(name),std::size_t(shape[0]),std::size_t(shape[1]),input);
        },bank.description};
  };
  result.parameters.routed_experts=[state](sem::Fragment &f,std::size_t stage) {
    const auto router=state->parameter(f,roles::stage_name(stage,"ffn_gate_inp.weight"));
    const auto bias=state->parameter(f,roles::stage_name(stage,"exp_probs_b.bias"));
    std::array<sem::ValueId,3> weights;
    for(std::size_t leg=0;leg<weights.size();++leg)
      weights[leg]=f.input(dspark_original_expert_roles::resident_description(roles::Leg(leg)));
    const auto map=f.input(sem::dense({256},c::DataType::kInt32));
    return base::ds::official::RoutedExpertParameters{router,bias,weights,map,30};
  };
  result.head.readonly_parameter=[state](sem::Fragment &f,std::string_view name) {
    if(name!="markov_w1.weight"&&name!="markov_w2.weight"&&name!="conf_proj.weight")
      throw std::runtime_error("S03 unknown original head role");
    return state->parameter(f,name);
  };
  return result;
}
// Caller uses the existing make_draft arguments with all old role pointers
// null, &bindings.parameters and &bindings.state->constants appended. Create
// one Bindings per cold recipe; share its immutable Bound/prepared OA leases.
} // namespace dspark_bfloat_sidecar_parameters
