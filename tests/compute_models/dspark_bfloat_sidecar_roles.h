#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dspark_bfloat_sidecar_specs.h"
#include "dspark_original_expert_roles.h"

// S03 source I/O and immutable bindings only. Catalogue pins are supplied
// explicitly after the real acquisition. Neither Q8 payload hashes nor an
// unverified receipt can qualify these different original BF16/MXFP4 bytes.
namespace dspark_bfloat_sidecar_roles {
namespace b=dspark_target_binding;
namespace c=hyle::compute;
namespace m=hyle::mem;
using Leg=dspark_original_expert_roles::Leg;
inline constexpr std::size_t kStages=3,kExpertDomain=256,kLegs=3;
inline b::SidecarIdentity identity() {return {std::string(kRepository),std::string(kRevision),std::string(kFile)};}
inline std::string stage_name(std::size_t stage,std::string_view suffix) {
  if(stage>=kStages)throw std::runtime_error("S03 stage outside original sidecar");
  return "blk."+std::to_string(stage)+"."+std::string(suffix);
}
inline std::string expert_name(std::size_t stage,Leg leg) {
  if(std::size_t(leg)>=kLegs)throw std::runtime_error("S03 expert leg outside source");
  constexpr std::array<std::string_view,3> suffix{
      "ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
  return stage_name(stage,suffix[std::size_t(leg)]);
}
inline c::TensorDesc expert_description(std::size_t stage,Leg leg) {
  (void)expert_name(stage,leg);
  return dspark_original_expert_roles::resident_description(leg,kExpertDomain);
}
inline std::size_t expert_bytes(Leg leg) {
  const auto description=expert_description(0,leg);
  const auto bytes=source::take(description.required_bytes(),"S03 full MXFP4 bank bytes");
  if(bytes%kExpertDomain)throw std::runtime_error("S03 expert byte stride differs");
  return bytes/kExpertDomain;
}
class ExpertSet final {
 public:
  ExpertSet(const ExpertSet&)=delete;
  ExpertSet& operator=(const ExpertSet&)=delete;
  ExpertSet(ExpertSet&&) noexcept=default;
  ExpertSet& operator=(ExpertSet&&)=delete;
  static ExpertSet load(const std::filesystem::path &directory,std::string_view catalogue_pin,
      std::size_t stage,std::int32_t global) {
    if(stage>=kStages||global<0||global>=std::int32_t(kExpertDomain))
      throw std::runtime_error("S03 selected expert identity outside original domain");
    std::array<std::string,3> names;
    std::array<source::Spec,3> specifications;
    std::size_t bytes=0;
    for(std::size_t leg=0;leg<kLegs;++leg) {
      names[leg]=expert_name(stage,Leg(leg));
      const auto extent=expert_bytes(Leg(leg));
      specifications[leg]={names[leg],c::DataType::kUInt8,std::int64_t(extent),0};bytes+=extent;
    }
    // U8 describes the authenticated byte slice, not the logical compute
    // tensor. Compute keeps the original grouped MXFP4 description above.
    return ExpertSet(stage,global,source::Bundle::load(directory,catalogue_pin,specifications,bytes));
  }
  std::size_t stage()const noexcept{return stage_;}
  std::int32_t global()const noexcept{return global_;}
  std::span<const std::byte> bytes(Leg leg)const{return bundle_.raw(expert_name(stage_,leg));}
  m::ConstBufferView view(Leg leg)const{return bundle_.view(expert_name(stage_,leg));}
 private:
  ExpertSet(std::size_t stage,std::int32_t global,source::Bundle bundle):
      bundle_(std::move(bundle)),stage_(stage),global_(global){}
  source::Bundle bundle_;
  std::size_t stage_;
  std::int32_t global_;
};
class Data final {
 public:
  Data(const Data&)=delete;
  Data& operator=(const Data&)=delete;
  Data(Data&&) noexcept=default;
  Data& operator=(Data&&)=delete;
  static Data load(const std::filesystem::path &directory,std::string_view actual_catalogue_sha256) {
    return Data(source::Bundle::load(directory,actual_catalogue_sha256,kStaticSpecs,kStaticBytes));
  }
  std::size_t ordinal(std::string_view name)const{return bundle_.ordinal(name);}
  const source::Spec &spec(std::string_view name)const{return bundle_.spec(name);}
  c::TensorDesc description(std::string_view name)const{return spec(name).description();}
  std::span<const std::byte> raw(std::string_view name)const{return bundle_.raw(name);}
  m::ConstBufferView view(std::string_view name)const{return bundle_.view(name);}
  std::span<const float> floats(std::string_view name)const {
    if(spec(name).type!=c::DataType::kFloat32)throw std::runtime_error("S03 role is not original F32");
    return floats_.at(ordinal(name));
  }
  void attach_expert(std::shared_ptr<const ExpertSet> selected) {
    if(!selected)throw std::runtime_error("S03 missing explicit expert attachment");
    auto &destination=experts_.at(selected->stage()).at(std::size_t(selected->global()));
    if(destination)throw std::runtime_error("S03 expert cannot be rebound after attachment");
    destination=std::move(selected);
  }
  bool has_expert(std::size_t stage,std::int32_t global)const {
    if(stage>=kStages||global<0||global>=std::int32_t(kExpertDomain))
      throw std::runtime_error("S03 expert lookup outside original domain");
    return bool(experts_[stage][std::size_t(global)]);
  }
  c::TensorDesc expert_description(std::size_t stage,Leg leg)const {
    return dspark_bfloat_sidecar_roles::expert_description(stage,leg);
  }
  std::span<const std::byte> expert_bytes(std::size_t stage,Leg leg,std::int32_t global)const {
    return selected(stage,global).bytes(leg);
  }
  m::ConstBufferView expert_view(std::size_t stage,Leg leg,std::int32_t global)const {
    return selected(stage,global).view(leg);
  }
 private:
  explicit Data(source::Bundle bundle):bundle_(std::move(bundle)) {
    for(std::size_t index=0;index<kStaticRoles;++index)
      if(kStaticSpecs[index].type==c::DataType::kFloat32)
        floats_[index]=source::decode_f32(bundle_.raw(kStaticSpecs[index].name));
  }
  const ExpertSet &selected(std::size_t stage,std::int32_t global)const {
    if(!has_expert(stage,global))throw std::runtime_error("S03 selected original expert has not been acquired and attached");
    return *experts_[stage][std::size_t(global)];
  }
  // Sources precede all views/attachments and are destroyed last.
  source::Bundle bundle_;
  std::array<std::vector<float>,kStaticRoles> floats_;
  std::array<std::array<std::shared_ptr<const ExpertSet>,kExpertDomain>,kStages> experts_;
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<b::ReadonlyRole,kStaticRoles> roles;
  const b::ReadonlyRole &at(std::string_view name)const {
    if(!source)throw std::runtime_error("S03 bound source absent");
    return roles.at(source->ordinal(name));
  }
};
inline Bound bind(std::shared_ptr<const Data> original,const c::StorageOwner &owner) {
  if(!original)throw std::runtime_error("S03 immutable source absent");
  Bound result{.source=std::move(original)};
  for(std::size_t index=0;index<kStaticRoles;++index) {
    const auto &specification=kStaticSpecs[index];
    auto ready=source::bind(specification.name,specification.description(),result.source->raw(specification.name),owner);
    result.roles[index]={std::string(ready.name),ready.description,std::move(ready.owner),std::move(ready.lease)};
  }
  return result;
}
inline b::ProjectionRoles main_roles(const Bound &bound,b::TargetIdentity target) {
  return {std::move(target),identity(),bound.at("fc.weight"),bound.at("enc.output_norm.weight"),identity()};
}
inline b::StageKvRoles stage_roles(const Bound &bound,b::TargetIdentity target,std::size_t stage) {
  return {std::move(target),identity(),std::int32_t(stage),bound.at(stage_name(stage,"attn_kv.weight")),
      bound.at(stage_name(stage,"attn_kv_a_norm.weight")),identity()};
}
} // namespace dspark_bfloat_sidecar_roles
