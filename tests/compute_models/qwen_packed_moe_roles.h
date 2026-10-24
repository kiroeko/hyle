#pragma once

#include <functional>
#include "qwen24_original_nextn_roles.h"
#include "qwen24_original_target_roles.h"
#include "original_named_role_source.h"

// Cold immutable view of actual packed Qwen MoE roles. It owns no executor,
// routing policy, acquisition or decoded weights. Primary mappings and named
// selected-range sources use the same graph/reference interface.
namespace qwen_packed_moe_roles {
namespace nextn=qwen24_original_nextn_roles;
namespace target=qwen24_original_target_roles;
namespace original=qwen_original_mtp_roles;
namespace named=original_named_role_source;
namespace c=hyle::compute;namespace m=hyle::mem;
using Leg=nextn::Leg;
struct Spec final {
  std::string_view name;
  std::optional<c::GgufQuantizedType> codec;
  ggml_type native=GGML_TYPE_F32;
  std::int64_t k=0,n=0,e=1;
  std::size_t bytes=0;
  c::TensorDesc description;
};
template<class Role,std::size_t Count> struct Source final {
  std::shared_ptr<const void> owner;
  const original::Identity *identity=nullptr;
  std::array<Spec,Count> specs;
  std::array<std::span<const std::byte>,Count> payloads;
  std::array<m::ConstBufferView,Count> views;
  std::array<std::span<const float>,Count> norms;
  std::array<Role,3> experts;
  std::function<std::span<const std::byte>(Leg,std::int32_t)> selected_bytes;
  std::function<m::ConstBufferView(Leg,std::int32_t)> selected_view;
  static std::size_t index(Role role) {
    const auto i=std::size_t(role);if(i>=Count)throw std::runtime_error("packed MoE source role outside its actual schema");return i;
  }
  const Spec &spec(Role role)const{return specs.at(index(role));}
  c::TensorDesc description(Role role)const{return spec(role).description;}
  std::span<const std::byte> bytes(Role role)const {
    const auto i=index(role);
    if(payloads[i].empty())throw std::runtime_error("packed MoE whole expert bank is not acquired; select actual IDs");
    return payloads[i];
  }
  m::ConstBufferView view(Role role)const {
    (void)bytes(role);return views.at(index(role));
  }
  std::span<const float> norm(Role role)const {
    const auto i=index(role);const auto &s=specs[i];
    if(s.k||s.native!=GGML_TYPE_F32||s.description.rank()!=1||norms[i].size()!=std::size_t(s.n))
      throw std::runtime_error("packed MoE norm requires its actual F32 vector");
    return norms[i];
  }
  c::TensorDesc expert_description(Leg leg)const{return description(experts.at(std::size_t(leg)));}
  c::TensorDesc resident_description(Leg leg)const {
    const auto &s=spec(experts.at(std::size_t(leg)));
    if(!s.codec)throw std::runtime_error("packed MoE source needs its real GGUF expert codec");
    return original::detail::take(c::gguf_grouped_weight_description(*s.codec,c::DataType::kFloat32,
        nextn::kResidentCapacity,s.n,s.k),"actual packed MoE resident description");
  }
  std::span<const std::byte> expert_bytes(Leg leg,std::int32_t global)const {
    if(selected_bytes)return selected_bytes(leg,global);
    const auto role=experts.at(std::size_t(leg));const auto &d=spec(role).description;
    if(global<0||global>=d.dimensions()[0]||d.byte_strides()[0]<=0)throw std::runtime_error("packed MoE actual expert ID/stride differs");
    const auto stride=std::size_t(d.byte_strides()[0]);return bytes(role).subspan(std::size_t(global)*stride,stride);
  }
  m::ConstBufferView expert_view(Leg leg,std::int32_t global)const {
    if(selected_view)return selected_view(leg,global);
    const auto role=experts.at(std::size_t(leg));const auto bytes_=expert_bytes(leg,global);
    return original::detail::take(view(role).subview(std::size_t(global)*std::size_t(spec(role).description.byte_strides()[0]),bytes_.size()),"actual packed MoE selected source view");
  }
};
template<class Role,std::size_t Count> struct Bound final {
  std::shared_ptr<const Source<Role,Count>> source;
  std::array<original::ReadonlyRole,Count> roles;
  const original::ReadonlyRole &at(Role role)const {
    const auto &r=roles.at(Source<Role,Count>::index(role));
    if(!r.lease.valid())throw std::runtime_error("packed MoE expert banks are source-only; bind an actual selected generation");return r;
  }
};
using NextnSource=Source<nextn::Role,nextn::kRoles>;
using TargetSource=Source<target::Role,target::kRoles>;
using NextnRoles=Bound<nextn::Role,nextn::kRoles>;
using TargetRoles=Bound<target::Role,target::kRoles>;
inline std::shared_ptr<const NextnRoles> adapt(std::shared_ptr<const nextn::Roles> actual) {
  if(!actual||!actual->source)throw std::runtime_error("packed NextN view requires its actual primary owner");
  auto source=std::make_shared<NextnSource>();source->owner=actual->source;source->identity=&nextn::kIdentity;
  source->experts={nextn::Role::kExpertGate,nextn::Role::kExpertUp,nextn::Role::kExpertDown};
  auto result=std::make_shared<NextnRoles>();result->source=source;
  for(std::size_t i=0;i<nextn::kRoles;++i) {
    const auto role=nextn::Role(i);const auto &s=nextn::spec(role);
    source->specs[i]={s.name,s.codec,s.native,s.k,s.n,s.e,s.bytes,nextn::description(role)};
    source->payloads[i]=actual->source->bytes(role);source->views[i]=actual->source->view(role);
    if(!s.k)source->norms[i]=actual->source->norm(role);
  }
  for(const auto role:nextn::kStaticRoles)result->roles[std::size_t(role)]=actual->at(role);
  return result;
}
inline std::shared_ptr<const TargetRoles> adapt(std::shared_ptr<const target::Bound> actual) {
  if(!actual||!actual->source)throw std::runtime_error("packed target view requires its actual primary owner");
  auto source=std::make_shared<TargetSource>();source->owner=actual->source;source->identity=&target::kIdentity;
  source->experts={target::Role::kExpertGate,target::Role::kExpertUp,target::Role::kExpertDown};
  auto result=std::make_shared<TargetRoles>();result->source=source;
  for(std::size_t i=0;i<target::kRoles;++i) {
    const auto role=target::Role(i);const auto &s=target::spec(role);
    source->specs[i]={s.name,s.codec,s.native,s.k,s.n,s.e,s.bytes,target::description(role)};
    source->payloads[i]=actual->source->bytes(role);source->views[i]=actual->source->view(role);
    if(!s.k)source->norms[i]=actual->source->norm(role);
  }
  for(const auto role:target::kStaticRoles)result->roles[std::size_t(role)]=actual->at(role);
  return result;
}
// Named static bindings already own their uploaded constants. This view does
// not copy/re-upload them; it supplies only real per-role metadata/references.
template<class Role,std::size_t Count,class PrimarySpecs,class StaticRoles,class Selected>
std::shared_ptr<const Bound<Role,Count>> named_view(std::shared_ptr<const named::Roles> actual,
    const PrimarySpecs &names,const StaticRoles &statics,std::array<Role,3> experts,
    std::shared_ptr<Selected> selected) {
  if(!actual||!actual->source||!selected)throw std::runtime_error("selected packed MoE view needs actual static and expert owners");
  const auto &id=actual->source->identity();const auto &contract=selected->contract();
  if(id.repository!=contract.repository||id.revision!=contract.revision||id.variant!=contract.variant)
    throw std::runtime_error("selected packed MoE static/expert artifact identity differs");
  auto source=std::make_shared<Source<Role,Count>>();source->owner=actual->source;source->identity=&id;source->experts=experts;
  auto result=std::make_shared<Bound<Role,Count>>();result->source=source;
  for(const auto role:statics) {
    const auto i=std::size_t(role);const auto &schema=names.at(i);const auto &s=actual->source->spec(schema.name);
    if(s.k!=schema.k||s.n!=schema.n||(schema.codec?(s.codec!=c::GgufQuantizedType::kQ8_0||s.native!=GGML_TYPE_Q8_0):
        (s.codec.has_value()||s.native!=GGML_TYPE_F32)))
      throw std::runtime_error("B07 static role differs from its frozen actual Q8_0/F32 census geometry");
    auto description=s.description;auto binding=actual->at(schema.name);
    // Named F32 input files keep physical dense [N,K] rows. MatMul borrows
    // their transposed descriptor and SAME ready bytes/lease, with no upload.
    if(s.k&&s.native==GGML_TYPE_F32)description=original::detail::take(description.transposed(),"actual F32 router RHS");
    binding.description=description;result->roles[i]=std::move(binding);
    source->specs[i]={s.name,s.codec,s.native,s.k,s.n,1,s.bytes,description};
    source->payloads[i]=actual->source->bytes(schema.name);source->views[i]=actual->source->view(schema.name);
    if(!s.k)source->norms[i]=actual->source->norm(schema.name);
  }
  for(std::size_t leg=0;leg<3;++leg) {
    const auto role=experts[leg];const auto &schema=names.at(std::size_t(role));
    const auto d=selected->expert_description(static_cast<Leg>(leg));const auto codec=c::gguf_quantized_type(d.storage_format());
    if(!codec||d.rank()!=3||d.dimensions()[0]!=nextn::kExpertDomain||d.dimensions()[1]!=schema.n||d.dimensions()[2]!=schema.k||
        selected->contract().banks[leg].role!=schema.name)
      throw std::runtime_error("selected packed expert role/section/geometry differs from the frozen census");
    source->specs[std::size_t(role)]={schema.name,codec,named::native_codec(*codec),schema.k,schema.n,nextn::kExpertDomain,
        original::detail::take(d.required_bytes(),"logical original expert extent"),d};
  }
  source->selected_bytes=[selected](Leg leg,std::int32_t global){return selected->expert_bytes(leg,global);};
  source->selected_view=[selected](Leg leg,std::int32_t global){return selected->expert_view(leg,global);};
  return result;
}
template<class Selected> std::shared_ptr<const NextnRoles> nextn_view(
    std::shared_ptr<const named::Roles> actual,std::shared_ptr<Selected> selected) {
  return named_view<nextn::Role,nextn::kRoles>(std::move(actual),nextn::kSpecs,nextn::kStaticRoles,
      std::array{nextn::Role::kExpertGate,nextn::Role::kExpertUp,nextn::Role::kExpertDown},std::move(selected));
}
template<class Selected> std::shared_ptr<const TargetRoles> target_view(
    std::shared_ptr<const named::Roles> actual,std::shared_ptr<Selected> selected) {
  return named_view<target::Role,target::kRoles>(std::move(actual),target::kSpecs,target::kStaticRoles,
      std::array{target::Role::kExpertGate,target::Role::kExpertUp,target::Role::kExpertDown},std::move(selected));
}
} // namespace qwen_packed_moe_roles
