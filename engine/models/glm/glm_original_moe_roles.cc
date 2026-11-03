#include "glm_original_moe_roles.h"

namespace glm_original_moe_roles {
std::size_t index(Leg leg) {
  const auto result = static_cast<std::size_t>(leg);
  if (result >= 3) throw std::runtime_error("unknown original GLM MoE leg");
  return result;
}
}

namespace glm_original_moe_roles {
std::pair<std::int64_t, std::int64_t> geometry(Leg leg,const Geometry &model) {
  glm_moe_geometry::require(model);
  (void)index(leg);
  return leg == Leg::kDown ? std::pair{model.hidden, model.intermediate} : std::pair{model.intermediate, model.hidden};
}
}

namespace glm_original_moe_roles {
c::TensorDesc shared_description(Leg leg,const Geometry &model,
    WeightStorage storage) {
  const auto [outputs, reduction] = geometry(leg,model);
  return glm_moe_storage::packed_description(outputs,reduction,1,storage,false);
}
}

namespace glm_original_moe_roles {
c::TensorDesc resident_description(Leg leg,const Geometry &model,
    WeightStorage storage) {
  const auto [outputs, reduction] = geometry(leg,model);
  return glm_moe_storage::packed_description(outputs,reduction,model.slots,storage,true);
}
}

namespace glm_original_moe_roles {
c::TensorDesc router_description(const Geometry &model) {
  glm_moe_geometry::require(model);
  return shared::take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{model.experts, model.hidden}), "original GLM router description");
}
}

namespace glm_original_moe_roles {
c::TensorDesc bias_description(const Geometry &model) {
  glm_moe_geometry::require(model);
  return shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{model.experts}), "original GLM bias description");
}
}

namespace glm_original_moe_roles {
Selection select(std::span<const std::int32_t> actual_ids,const Geometry &model) {
  glm_moe_geometry::require(model);
  if (actual_ids.empty() || actual_ids.size() > model.slots)
    throw std::runtime_error("original GLM selection needs actual1x8 global IDs");
  Selection result;
  result.map.assign(std::size_t(model.experts),-1);
  result.globals.reserve(actual_ids.size());
  for (const auto global : actual_ids) {
    if (global < 0 || global >= model.experts)
      throw std::runtime_error("actual GLM expert ID outside original model domain");
    result.globals.push_back(global);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(), result.globals.end()), result.globals.end());
  for (std::size_t local = 0; local < result.globals.size(); ++local)
    result.map[std::size_t(result.globals[local])] = std::int32_t(local);
  return result;
}
}

namespace glm_original_moe_roles {
void require_identity(const SourceIdentity &identity) {
  if(identity.repository.empty()||identity.revision.size()!=40||
      !std::ranges::all_of(identity.revision,[](char ch) {
        return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); })||
      identity.layer_prefix.empty()||identity.layer_prefix.back()!='.'||
      (identity.scale_member!="weight_scale"&&identity.scale_member!="weight_scale_inv"))
    throw std::runtime_error("selected GLM source needs its exact identity and scale spelling");
}
}

namespace glm_original_moe_roles {
c::TensorDesc source_description(Leg leg,bool scales,const Geometry &model,
    WeightStorage storage) {
  const auto [n,k]=geometry(leg,model);
  glm_moe_storage::require(storage,model);
  if(storage==WeightStorage::kMxfp4Group32)
    return shared::take(c::TensorDesc::dense(c::DataType::kUInt8,
        std::array{n,k/(scales?32:2)}),"original MXFP4 source plane description");
  return shared::take(c::TensorDesc::dense(scales?c::DataType::kFloat32:c::DataType::kFloat8E4M3FN,
      std::array{scales?n/128:n,scales?k/128:k}),"original GLM source plane description");
}
}

namespace glm_original_moe_roles {
std::string source_name(const SourceIdentity &identity,Leg leg,bool scales,
    std::optional<std::int32_t> global) {
  constexpr std::array<std::string_view,3> legs{"gate","up","down"};
  const auto module=global?"mlp.experts."+std::to_string(*global)+".":"mlp.shared_experts.";
  return identity.layer_prefix+module+std::string(legs[index(leg)])+"_proj."+
      (scales?identity.scale_member:"weight");
}
}

namespace glm_original_moe_roles {
const FilePin &source_file(std::span<const FilePin> files,std::string_view name) {
  const FilePin *result=nullptr;
  for(const auto &file:files)if(file.name==name) {
    if(result)throw std::runtime_error("duplicate selected GLM original role name");
    result=&file;
  }
  if(!result)throw std::runtime_error("selected GLM original role name is missing");
  return *result;
}
}

namespace glm_original_moe_roles {
m::FileMapping map_source_file(const std::filesystem::path &directory,const FilePin &pin,
    std::string_view expected_name,const c::TensorDesc &description) {
  const auto filename=std::filesystem::path(std::string(pin.file));
  if(pin.name!=expected_name||filename.empty()||filename.filename()!=filename||
      pin.description!=description||pin.sha256.size()!=64||
      !std::ranges::all_of(pin.sha256,[](char ch) {
        return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); }))
    throw std::runtime_error("selected GLM role requires its exact name/type/shape/file/SHA");
  auto result=shared::take(m::FileMapping::open((directory/filename).string().c_str()),
      "map original GLM selected role");
  if(result.size()!=shared::take(description.required_bytes(),"selected GLM role bytes")||
      shared::detail::mapped_sha256(shared::take(result.bytes(),"selected GLM mapped bytes"))!=pin.sha256)
    throw std::runtime_error("selected GLM original role length/SHA differs");
  return result;
}
}

namespace glm_original_moe_roles {
Roles bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original GLM MoE needs original data and actual storage owner");
  Roles result;
  result.source = std::move(source);
  const auto ready = [&](std::string_view name, const c::TensorDesc &description,
      std::span<const std::byte> raw) {
    const auto extent = shared::take(description.required_bytes(), "original GLM MoE immutable extent");
    if (raw.size() != extent) throw std::runtime_error("original GLM MoE immutable payload extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original GLM MoE immutable bank");
    shared::check(bank.view().write(raw), "original GLM MoE once-only upload");
    return ReadonlyRole{name, description, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original GLM MoE readonly lease")};
  };
  result.router = ready("original GLM BF16 router", router_description(result.source->model()), result.source->router_bytes());
  result.bias = ready("original GLM F32 correction bias", bias_description(result.source->model()), result.source->bias_bytes());
  constexpr std::array<std::string_view, 3> names{
      "original GLM shared gate", "original GLM shared up", "original GLM shared down"};
  for (std::size_t leg = 0; leg < 3; ++leg) {
    const auto mapped = result.source->shared_mapping(static_cast<Leg>(leg));
    result.shared[leg] = ready(names[leg], shared_description(static_cast<Leg>(leg),result.source->model(),result.source->weight_storage()),
        mapped.bytes());
  }
  return result;
}
}
