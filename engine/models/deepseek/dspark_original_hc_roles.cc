#include "dspark_original_hc_roles.h"

namespace dspark_original_hc_roles {
std::size_t index(std::size_t stage,StageRole role) {
  const auto parameter=static_cast<std::size_t>(role);
  if(stage>=3 || parameter>=kStageRoles)throw std::runtime_error("original DSpark stage parameter is absent");
  return stage*kStageRoles+parameter;
}
}

namespace dspark_original_hc_roles {
std::size_t index(HeadRole role) {
  const auto parameter=static_cast<std::size_t>(role);
  if(parameter>=kHeadRoles)throw std::runtime_error("original DSpark output parameter is absent");
  return 3*kStageRoles+parameter;
}
}

namespace dspark_original_hc_roles {
const Spec &spec(std::size_t parameter) {
  if(parameter>=kSpecs.size())throw std::runtime_error("original DSpark parameter index is absent");
  return kSpecs[parameter];
}
}

namespace dspark_original_hc_roles {
c::TensorDesc description(std::size_t parameter) {
  const auto &s=spec(parameter);
  auto result=c::TensorDesc::dense(c::DataType::kFloat32,std::span(s.dimensions).first(s.rank));
  if(!result)throw std::runtime_error(std::string(result.error().message()));
  return *result;
}
}

namespace dspark_original_hc_roles {
Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source)throw std::runtime_error("original DSpark coefficient source is absent");
  Roles result{.source=std::move(source)};
  for(std::size_t i=0;i<kRoles;++i) {
    const auto raw=std::as_bytes(result.source->values(i));
    const auto desc=description(i);
    const auto extent=desc.required_bytes();
    if(!extent || raw.size()!=*extent)throw std::runtime_error("original DSpark coefficient extent differs");
    auto allocated=m::Buffer::create(owner.ref(),raw.size(),256);
    if(!allocated)throw std::runtime_error(std::string(allocated.error().message()));
    auto buffer=std::move(*allocated);const auto written=buffer.view().write(raw);
    if(!written.ok())throw std::runtime_error(std::string(written.message()));
    auto lease=owner.adopt_buffer(std::move(buffer));
    if(!lease)throw std::runtime_error(std::string(lease.error().message()));
    result.parameters[i]={std::string(spec(i).name),desc,owner,std::move(*lease)};
  }
  return result;
}
}

namespace dspark_original_hc_roles {
hyle::Status qualify(const Roles &roles,const b::TargetIdentity &target) {
  if(!roles.source || target.model_repository!=b::kModelRepository || target.model_revision!=b::kModelRevision)
    return hyle::Status::InvalidArgument("original DSpark parameters require matching formal Flash0731");
  for(std::size_t i=0;i<kRoles;++i) {
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.at(i),spec(i).name,description(i)));
  }
  return hyle::Status::Ok();
}
}
