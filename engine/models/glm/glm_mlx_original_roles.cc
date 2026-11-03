#include "glm_mlx_original_roles.h"

namespace glm_mlx_original_roles {
std::string expert_name(std::int32_t id,Leg leg,Plane plane,std::string_view prefix) {
  if(id<0||id>=std::int32_t(kDomain))throw std::runtime_error("MLX actual expert ID outside288-domain");
  if(prefix!="model.language_model.layers.43."&&prefix!="model.language_model.layers.44."&&prefix!=kPrefix)throw std::runtime_error("MLX source layer prefix differs");
  return std::string(prefix)+"mlp.experts."+std::to_string(id)+"."+std::string(kLegs.at(std::size_t(leg)))+"_proj."+std::string(kPlanes.at(std::size_t(plane)));
}
}

namespace glm_mlx_original_roles {
c::TensorDesc expert_description(Leg leg,Plane plane) {
  const bool down=leg==Leg::kDown;const auto n=down?4096:2048,k=down?2048:4096;
  const auto bits=down?5:4;
  return sem::dense({n,plane==Plane::kCodes?k/32*bits:k/64},plane==Plane::kCodes?c::DataType::kUInt32:c::DataType::kFloat16);
}
}

namespace glm_mlx_original_roles {
c::TensorDesc resident_description(Leg leg,Plane plane) {
  const auto d=expert_description(leg,plane);return sem::dense({std::int64_t(kCapacity),d.dimensions()[0],d.dimensions()[1]},d.data_type());
}
}

namespace glm_mlx_original_roles {
std::vector<raw::Spec> static_specs() {
  std::vector<raw::Spec> result;result.reserve(kRoles);
  for(const auto &s:kSpecs) {
    if(raw::take(s.description().required_bytes(),"MLX static extent")!=s.bytes)throw std::runtime_error("MLX original physical schema extent differs");
    result.push_back({s.name,s.type,s.rows,s.columns});
  }
  return result;
}
}

namespace glm_mlx_original_roles {
bool pin(std::string_view p) {
  return p.size()==64&&std::all_of(p.begin(),p.end(),[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');});
}
}

namespace glm_mlx_original_roles {
Bound bind(std::shared_ptr<Data> source,const c::StorageOwner &owner) {
  if(!source||!owner)throw std::runtime_error("MLX original binding requires its actual strong owner");
  Bound result;result.source=std::move(source);
  for(const auto &s:result.source->specs()) {
    const auto d=result.source->description(s.name);const auto bytes=result.source->bytes(s.name);
    auto buffer=raw::take(m::Buffer::create(owner.ref(),bytes.size(),256),"MLX original native allocation");
    raw::check(buffer.view().write(bytes),"MLX unchanged original plane upload");
    result.roles.push_back({s.name,d,owner,raw::take(owner.adopt_buffer(std::move(buffer)),"MLX immutable original lease")});
  }
  return result;
}
}
