#include "dspark_original_formal_roles.h"

namespace dspark_original_formal_roles {
bool same_model(const Model &left,const Model &right) noexcept {
  const auto &a=left.geometry;const auto &b=right.geometry;
  return left.repository==right.repository&&left.revision==right.revision&&a.hidden==b.hidden&&a.heads==b.heads&&
      a.query_rank==b.query_rank&&a.groups==b.groups&&a.experts==b.experts&&a.intermediate==b.intermediate&&
      a.index_top==b.index_top&&a.route_scale==b.route_scale&&a.first_target_layer==b.first_target_layer;
}
}

namespace dspark_original_formal_roles {
std::string layer_name(const Model &model,Section section,std::size_t stage,std::string_view suffix) {
  if(stage>=3)throw std::runtime_error("formal source stage must be0..2");
  if(section!=Section::kTarget&&section!=Section::kDraft)throw std::runtime_error("formal source section differs");
  return (section==Section::kTarget?"layers."+std::to_string(model.geometry.first_target_layer+std::int64_t(stage)):
      "mtp."+std::to_string(stage))+"."+std::string(suffix);
}
}

namespace dspark_original_formal_roles {
Selection select(std::span<const std::int32_t> actual_ids,std::size_t capacity,
    const Geometry &geometry) {
  geometry.validate();
  if(actual_ids.empty()||!capacity||capacity>std::size_t(geometry.experts)||actual_ids.size()>capacity)
    throw std::runtime_error("formal selection exceeds its actual resident capacity");
  Selection result;result.map.assign(std::size_t(geometry.experts),-1);result.globals.reserve(actual_ids.size());
  for(const auto id:actual_ids) {
    if(id<0||id>=geometry.experts)throw std::runtime_error("formal global expert outside original domain");
    result.globals.push_back(id);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(),result.globals.end()),result.globals.end());
  for(std::size_t i=0;i<result.globals.size();++i)result.map[std::size_t(result.globals[i])]=std::int32_t(i);
  return result;
}
}

namespace dspark_original_formal_roles {
std::string expert_name(const Model &model,Section section,std::size_t stage,Leg leg,
    std::int32_t global,bool scales) {
  if(global<0||global>=model.geometry.experts)throw std::runtime_error("formal global expert outside original domain");
  const char *which=nullptr;
  switch(leg) {case Leg::kGate:which="w1";break;case Leg::kUp:which="w3";break;case Leg::kDown:which="w2";break;
    default:throw std::runtime_error("unknown formal expert leg");}
  return layer_name(model,section,stage,"ffn.experts."+std::to_string(global)+"."+which+(scales?".scale":".weight"));
}
}

namespace dspark_original_formal_roles {
SourceView source_view(std::shared_ptr<const Data> source,Section section) {
  if(!source)throw std::runtime_error("formal original source view needs its owner");
  (void)layer_name(source->model(),section,0,"");return {std::move(source),section};
}
}

namespace dspark_original_formal_roles {
Bound bind(std::shared_ptr<const Data> source_data,const c::StorageOwner &owner) {
  if(!source_data||!owner)throw std::runtime_error("formal original bind needs original data and actual Graph owner");
  Bound result;result.source=std::move(source_data);
  for(const auto &spec:result.source->specs()) {
    if(spec.type==c::DataType::kUInt8)continue;
    result.roles.emplace(spec.name,source::bind(spec.name,result.source->description(spec.name),
        result.source->payload(spec.name),owner));
  }
  return result;
}
}

namespace dspark_original_formal_roles {
dspark_original_target_roles::TargetBindings<Data> target_bindings(const Bound &named) {
  namespace target=dspark_original_target_roles;
  if(!named.source)throw std::runtime_error("formal original target bindings need their source owner");
  target::TargetBindings<Data> result;result.source=named.source;
  for(std::size_t stage=0;stage<target::kStages;++stage)
    for(std::size_t slot=0;slot<target::kRoles;++slot) {
      const auto role=static_cast<target::Role>(slot);
      if(!target::present(stage,role))continue;
      const auto suffix=target::kRawSpecs[target::index(target::raw_role(role))].suffix;
      result.roles[stage][slot]=named.at(Section::kTarget,stage,suffix);
    }
  for(std::size_t slot=0;slot<target::kFinalRoles;++slot) {
    const auto role=static_cast<target::FinalRole>(slot);
    // These top-level final-HC/norm roles are distinct from mtp.2's head.
    result.final[slot]=named.at(target::final_spec(role).suffix);
  }
  return result;
}
}
