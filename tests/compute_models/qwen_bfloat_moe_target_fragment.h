#pragma once

#include "qwen_bfloat_original_roles.h"

// The original BF16 MoE target uses the same attention and selected-expert
// math as NextN. Only its independently acquired static roles and final norm
// differ. Runtime expert inputs remain ordinary caller-supplied Values.
namespace qwen_bfloat_moe_target_fragment {
namespace source=qwen_bfloat_original_roles;
namespace attention=qwen_full_attention_fragment;
namespace moe=qwen24_original_moe_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
enum class Role:std::size_t {
  kInputNorm,kQuery,kKey,kValue,kQueryNorm,kKeyNorm,kOutput,kPostNorm,
  kRouter,kSharedGate,kSharedProjectionGate,kSharedUp,kSharedDown,kFinalNorm,kCount
};
inline constexpr std::size_t kRoles=std::size_t(Role::kCount);
using Names=std::array<std::string_view,kRoles>;
// Files/SHA/name are all caller metadata from the actual original header.
// The two large fused expert source files are deliberately not static roles.
inline std::array<source::Spec,kRoles> specs(attention::Geometry geometry,
    std::int64_t shared_intermediate,std::int64_t experts,const Names &names,
    const std::array<source::PinnedFile,kRoles> &files) {
  const auto schema=source::moe_static_specs(geometry,shared_intermediate,experts,1,{},
      {source::kMoeStaticRoleNames[17],source::kMoeStaticRoleNames[18]});
  constexpr std::array<std::size_t,kRoles> slots{3,4,5,6,8,9,7,10,12,13,14,15,16,11};
  std::array<source::Spec,kRoles> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={names[slot],files[slot].file,files[slot].sha256,schema[slots[slot]].description};
  return result;
}
struct Parameters final {
  attention::TargetAttentionParameters attention;
  moe::Parameters feed_forward;
  Value final_gamma; // Effective F32 gamma, not the original raw BF16 delta.
};
struct Result final {
  std::shared_ptr<const source::Data> source;
  std::vector<source::Constant> constants;
  attention::TargetAttentionResult attention;
  moe::BFloatResult feed_forward;
  Value post_ffn,normalized_tap;
  attention::LocalTrace tail_trace;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    attention.ports.visit_values(visitor);
    if(attention.trace)attention.trace->visit_values(visitor);
    feed_forward.ports.visit_values(visitor);
    visitor(post_ffn);visitor(normalized_tap);tail_trace.visit_values(visitor);
    // feed_forward.boundary deliberately keeps the actual original frontier
    // producer ordinals; the normal Graph splitter owns its live-through Copy.
  }
};
inline Result append(sem::Fragment &fragment,Value input,Value key,Value value,
    Value cursor,Value position,Value frequency,const Parameters &parameters,
    attention::Geometry geometry) {
  if(geometry.storage!=c::DataType::kBFloat16)
    throw std::runtime_error("original BF16 MoE target requires its BF16 storage geometry");
  attention::require_geometry(geometry);
  Result result;
  result.attention=attention::append_target_attention(fragment,input,key,value,cursor,position,
      frequency,parameters.attention,geometry);
  result.feed_forward=moe::append_bfloat(fragment,result.attention.ports.ffn_input,parameters.feed_forward);
  result.post_ffn=fragment.one(c::Operation(c::Binary{.kind=c::BinaryKind::kAdd}),
      {result.attention.ports.residual,result.feed_forward.output});
  result.normalized_tap=attention::normalize(fragment,result.post_ffn,parameters.final_gamma,
      geometry.hidden,geometry,&result.tail_trace);
  return result;
}
inline Result append(sem::Fragment &fragment,Value input,Value key,Value value,
    Value cursor,Value position,Value frequency,const source::Bound &bound,const Names &names,
    const source::DenseNormConventions &rules,attention::Geometry geometry,
    const std::array<Value,3> &banks,Value map) {
  if(!bound.source)throw std::runtime_error("original BF16 MoE target needs its mapped static source owner");
  std::vector<source::Constant> constants;
  const auto parameter=[&](Role slot) {
    const auto &role=bound.at(names[std::size_t(slot)]);
    if(!role.owner||!role.lease)throw std::runtime_error("original BF16 target parameter is not ready");
    const auto permission=source::original::detail::take(role.lease.tensor(role.description),"BF16 target readonly parameter");
    if(permission.storage().resource()!=role.owner.ref())throw std::runtime_error("BF16 target parameter owner differs");
    const auto result=fragment.constant(role.description,{});constants.push_back({result,role});return result;
  };
  const auto gamma=[&](Role slot,source::NormConvention convention,std::int64_t width) {
    return source::effective_gamma(fragment,{parameter(slot),convention},width);
  };
  Parameters parameters;
  parameters.attention={gamma(Role::kInputNorm,rules.input,geometry.hidden),
      parameter(Role::kQuery),parameter(Role::kKey),parameter(Role::kValue),
      gamma(Role::kQueryNorm,rules.query,geometry.depth),gamma(Role::kKeyNorm,rules.key,geometry.depth),
      parameter(Role::kOutput),gamma(Role::kPostNorm,rules.post,geometry.hidden)};
  parameters.feed_forward={parameter(Role::kRouter),parameter(Role::kSharedGate),
      {parameter(Role::kSharedProjectionGate),parameter(Role::kSharedUp),parameter(Role::kSharedDown)},banks,map};
  parameters.final_gamma=gamma(Role::kFinalNorm,rules.head,geometry.hidden);
  auto result=append(fragment,input,key,value,cursor,position,frequency,parameters,geometry);
  result.source=bound.source;result.constants=std::move(constants);return result;
}
} // namespace qwen_bfloat_moe_target_fragment
