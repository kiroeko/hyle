#include "dspark_original_target_roles.h"

namespace dspark_original_target_roles {
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
}

namespace dspark_original_target_roles {
std::size_t stage_index(std::size_t stage) {
  if (stage >= kStages) throw std::runtime_error("original target stage outside40..42");
  return stage;
}
}

namespace dspark_original_target_roles {
int compression_ratio(std::size_t stage) { return stage_index(stage) == 1 ? 128 : 4; }
}

namespace dspark_original_target_roles {
std::size_t index(RawRole role) { return enum_index(role, kRawRoles); }
}

namespace dspark_original_target_roles {
std::size_t index(Role role) { return enum_index(role, kRoles); }
}

namespace dspark_original_target_roles {
std::size_t index(FinalRole role) { return enum_index(role, kFinalRoles); }
}

namespace dspark_original_target_roles {
std::size_t index(Leg leg) { return enum_index(leg, 3); }
}

namespace dspark_original_target_roles {
RawRole raw_role(Role role) { return kRoleSources[index(role)].value; }
}

namespace dspark_original_target_roles {
bool present(std::size_t stage, RawRole role) {
  (void)stage_index(stage); (void)index(role);
  if (stage != 1) return true;
  return role != RawRole::kIndexerApe && role != RawRole::kIndexerNorm &&
      role != RawRole::kIndexerGate && role != RawRole::kIndexerKey &&
      role != RawRole::kIndexerWeights && role != RawRole::kIndexerQueryCodes &&
      role != RawRole::kIndexerQueryScale;
}
}

namespace dspark_original_target_roles {
bool present(std::size_t stage, Role role) { return present(stage, raw_role(role)); }
}

namespace dspark_original_target_roles {
RawSpec raw_spec(std::size_t stage, RawRole role) {
  if (!present(stage, role)) throw std::runtime_error("ratio128 layer has no indexer parameters");
  auto result = kRawSpecs[index(role)];
  if (stage == 1) {
    if (role == RawRole::kCompressorApe) { result.rows = 128; result.columns = 512; }
    if (role == RawRole::kCompressorGate || role == RawRole::kCompressorKey) result.rows = 512;
  }
  return result;
}
}

namespace dspark_original_target_roles {
c::TensorDesc raw_description(std::size_t stage, RawRole role) {
  const auto spec = raw_spec(stage, role);
  if (spec.columns == 0) return take(c::TensorDesc::dense(spec.type, std::array{spec.rows}), "formal target raw vector");
  return take(c::TensorDesc::dense(spec.type, std::array{spec.rows,spec.columns}), "formal target raw matrix");
}
}

namespace dspark_original_target_roles {
bool is_matrix(Role role) { return kRawSpecs[index(raw_role(role))].type == c::DataType::kFloat8E4M3FN; }
}

namespace dspark_original_target_roles {
bool cold_widen(Role role) {
  return role == Role::kCompressorGate || role == Role::kCompressorKey ||
      role == Role::kIndexerGate || role == Role::kIndexerKey;
}
}

namespace dspark_original_target_roles {
RawRole scale_role(Role role) {
  if (!is_matrix(role)) throw std::runtime_error("formal target scale requested for non-FP8 role");
  return kRoleSources[index(role)].scale;
}
}

namespace dspark_original_target_roles {
c::TensorDesc description(std::size_t stage, Role role) {
  const auto spec = raw_spec(stage,raw_role(role));
  if (is_matrix(role)) return take(c::float8_weight_description(spec.columns,spec.rows,
      c::Float8ScaleMode::kFloat32Block128), "formal target canonical FP8");
  if (cold_widen(role)) return take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array{spec.rows,spec.columns}), "formal target source-required F32 compressor");
  return raw_description(stage,raw_role(role));
}
}

namespace dspark_original_target_roles {
RawSpec final_spec(FinalRole role) {
  switch (role) {
    case FinalRole::kBase: return {"hc_head_base",c::DataType::kFloat32,4,0};
    case FinalRole::kFn: return {"hc_head_fn",c::DataType::kFloat32,4,16384};
    case FinalRole::kScale: return {"hc_head_scale",c::DataType::kFloat32,1,0};
    case FinalRole::kNorm: return {"norm.weight",c::DataType::kBFloat16,4096,0};
    default: throw std::runtime_error("unknown formal target final role");
  }
}
}

namespace dspark_original_target_roles {
c::TensorDesc description(FinalRole role) {
  const auto spec = final_spec(role);
  if (!spec.columns) return take(c::TensorDesc::dense(spec.type,std::array{spec.rows}),"formal target final vector");
  return take(c::TensorDesc::dense(spec.type,std::array{spec.rows,spec.columns}),"formal target final matrix");
}
}

namespace dspark_original_target_roles {
std::pair<std::int64_t,std::int64_t> geometry(Leg leg) {
  (void)index(leg);
  return leg == Leg::kDown ? std::pair{kHidden,kIntermediate} : std::pair{kIntermediate,kHidden};
}
}

namespace dspark_original_target_roles {
c::TensorDesc resident_description(Leg leg,std::size_t capacity) {
  if (!capacity || capacity > std::size_t(kDomain)) throw std::runtime_error("invalid formal target resident expert capacity");
  const auto [outputs,reduction] = geometry(leg);
  return take(c::float4_grouped_weight_description(std::int64_t(capacity),outputs,reduction),"formal target FP4 resident bank");
}
}

namespace dspark_original_target_roles {
Selection select(std::span<const std::int32_t> actual_ids) {
  if (actual_ids.empty() || actual_ids.size() > kResidentCapacity) throw std::runtime_error("formal target selection requires actual Top6 IDs");
  Selection result; result.map.fill(-1); result.globals.reserve(actual_ids.size());
  for (const auto global : actual_ids) {
    if (global < 0 || global >= kDomain) throw std::runtime_error("formal target expert outside original256 domain");
    result.globals.push_back(global);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(),result.globals.end()),result.globals.end());
  for (std::size_t local=0;local<result.globals.size();++local) result.map[std::size_t(result.globals[local])]=std::int32_t(local);
  return result;
}
}

namespace dspark_original_target_roles {
Bound bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("formal target bind needs original data and actual Graph owner");
  Bound result; result.source=std::move(source);
  const auto ready=[&](std::string_view name,const c::TensorDesc &desc,std::span<const std::byte> payload) {
    return carrier::bind(name,desc,payload,owner);
  };
  for (std::size_t stage=0;stage<kStages;++stage)
    for (std::size_t slot=0;slot<kRoles;++slot) {
      const auto role=static_cast<Role>(slot);
      if (present(stage,role)) result.roles[stage][slot]=ready(raw_spec(stage,raw_role(role)).suffix,
          description(stage,role),result.source->payload(stage,role));
    }
  for (std::size_t slot=0;slot<kFinalRoles;++slot) {
    const auto role=static_cast<FinalRole>(slot);
    result.final[slot]=ready(final_spec(role).suffix,description(role),result.source->bytes(role));
  }
  return result;
}
}
