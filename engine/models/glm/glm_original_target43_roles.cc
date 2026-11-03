#include "glm_original_target43_roles.h"

namespace glm_original_target43_roles {
std::size_t index(RawRole role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRawRoles) throw std::runtime_error("unknown original GLM43 raw role");
  return slot;
}
}

namespace glm_original_target43_roles {
std::size_t index(Role role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRoles) throw std::runtime_error("unknown original GLM43 parameter");
  return slot;
}
}

namespace glm_original_target43_roles {
const Spec &spec(RawRole role) { return kSpecs[index(role)]; }
}

namespace glm_original_target43_roles {
RawRole raw_role(Role role) {
  switch (role) {
  case Role::kHcAttnBase: return RawRole::kHcAttnBase;
  case Role::kHcAttnFn: return RawRole::kHcAttnFn;
  case Role::kHcAttnScale: return RawRole::kHcAttnScale;
  case Role::kHcFfnBase: return RawRole::kHcFfnBase;
  case Role::kHcFfnFn: return RawRole::kHcFfnFn;
  case Role::kHcFfnScale: return RawRole::kHcFfnScale;
  case Role::kInputNorm: return RawRole::kInputNorm;
  case Role::kQueryProjection: return RawRole::kQueryCodes;
  case Role::kQueryNorm: return RawRole::kQueryNorm;
  default: throw std::runtime_error("unknown original GLM43 parameter raw role");
  }
}
}

namespace glm_original_target43_roles {
bool is_hc(Role role) noexcept { return role == Role::kHcAttnFn || role == Role::kHcFfnFn; }
}

namespace glm_original_target43_roles {
c::TensorDesc raw_description(RawRole role) {
  const auto &s = spec(role);
  if (s.columns == 0)
    return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows}), "original GLM43 vector");
  return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows, s.columns}), "original GLM43 matrix");
}
}

namespace glm_original_target43_roles {
c::TensorDesc description(Role role) {
  if (is_hc(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{24, 16384}), "original GLM43 runtime HC");
  if (role == Role::kQueryProjection)
    return shared::take(c::float8_weight_description(kHidden, kQueryRank,
        c::Float8ScaleMode::kFloat32Block128), "original GLM43 q_a canonical");
  return raw_description(raw_role(role));
}
}

namespace glm_original_target43_roles {
Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original GLM43 binding requires strong source and actual Graph owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot); const auto desc = description(role);
    const auto raw = result.source->payload(role);
    const auto extent = shared::take(desc.required_bytes(), "original GLM43 ready bytes");
    if (raw.size() != extent) throw std::runtime_error("original GLM43 prepared payload differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original GLM43 immutable bank");
    shared::check(bank.view().write(raw), "original GLM43 once-only upload");
    result.roles[slot] = {spec(raw_role(role)).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original GLM43 readonly lease")};
  }
  // MLA and indexer are deliberately bound by their existing factories. No
  // extra raw KV_b device lease is retained here after their cold K/V split.
  return result;
}
}

namespace glm_original_target43_roles {
moe::Data load_moe(const std::filesystem::path &directory) {
  return moe::Data::load(directory, kMoeIndexSha256);
}
}
