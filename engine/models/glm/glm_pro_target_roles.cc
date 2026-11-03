#include "glm_pro_target_roles.h"

namespace glm_pro_target_roles {
std::size_t index(RawRole role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRawRoles) throw std::runtime_error("unknown original GLM Pro raw role");
  return slot;
}
}

namespace glm_pro_target_roles {
std::size_t index(Role role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRoles) throw std::runtime_error("unknown original GLM Pro runtime role");
  return slot;
}
}

namespace glm_pro_target_roles {
const Spec &spec(RawRole role) { return kSpecs[index(role)]; }
}

namespace glm_pro_target_roles {
RawRole raw_role(Role role) { return kRawForRole[index(role)]; }
}

namespace glm_pro_target_roles {
bool is_matrix(Role role) {
  return spec(raw_role(role)).type == c::DataType::kFloat8E4M3FN;
}
}

namespace glm_pro_target_roles {
c::TensorDesc raw_description(RawRole role) {
  const auto &s = spec(role);
  if (s.columns == 0)
    return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows}), "original Pro target77 vector");
  return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows, s.columns}), "original Pro target77 matrix");
}
}

namespace glm_pro_target_roles {
c::TensorDesc description(Role role) {
  const auto &s = spec(raw_role(role));
  if (is_matrix(role))
    return shared::take(c::float8_weight_description(s.columns, s.rows,
        c::Float8ScaleMode::kFloat32Block128), "original Pro target77 canonical E4");
  return raw_description(raw_role(role));
}
}

namespace glm_pro_target_roles {
Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original Pro target77 banks need source and actual Graph owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot); const auto desc = description(role);
    const auto extent = shared::take(desc.required_bytes(), "original Pro target77 ready extent");
    const auto raw = result.source->payload(role);
    if (raw.size() != extent) throw std::runtime_error("original Pro target77 prepared byte extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original Pro target77 ready immutable bank");
    shared::check(bank.view().write(raw), "original Pro target77 once-only upload");
    result.roles[slot] = {spec(raw_role(role)).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original Pro target77 ready readonly lease")};
  }
  // The original KV_b ready lease may retire after the shared cold K/V split.
  return result;
}
}
