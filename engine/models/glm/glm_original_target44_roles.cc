#include "glm_original_target44_roles.h"

namespace glm_original_target44_roles {
std::size_t index(Role role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRoles) throw std::runtime_error("unknown original GLM target44 role");
  return slot;
}
}

namespace glm_original_target44_roles {
const Spec &spec(Role role) { return kSpecs[index(role)]; }
}

namespace glm_original_target44_roles {
bool is_hc(Role role) noexcept {
  return role == Role::kHcAttnFn || role == Role::kHcFfnFn;
}
}

namespace glm_original_target44_roles {
bool is_conv(Role role) noexcept {
  return role == Role::kKeyConv || role == Role::kQueryConv || role == Role::kValueConv;
}
}

namespace glm_original_target44_roles {
bool is_norm(Role role) noexcept {
  return role == Role::kInputNorm || role == Role::kPostNorm ||
      role == Role::kOutputNorm || role == Role::kModelNorm;
}
}

namespace glm_original_target44_roles {
c::TensorDesc raw_description(Role role) {
  const auto &s = spec(role);
  return shared::take(c::TensorDesc::dense(s.raw_type,
      std::span(s.shape.data(), s.rank)), "original target44 raw descriptor");
}
}

namespace glm_original_target44_roles {
c::TensorDesc description(Role role) {
  const auto &s = spec(role);
  if (is_hc(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{24, kStreams * kHidden}), "original target44 HC runtime descriptor");
  if (is_conv(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{kHeads * kDepth, 4}), "original target44 conv runtime descriptor");
  return shared::take(c::TensorDesc::dense(s.raw_type,
      std::span(s.shape.data(), s.rank)), "original target44 runtime descriptor");
}
}

namespace glm_original_target44_roles {
Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original target44 needs original data and its actual owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot);
    const auto desc = description(role);
    const auto raw = result.source->payload(role);
    const auto extent = shared::take(desc.required_bytes(), "original target44 runtime extent");
    if (raw.size() != extent) throw std::runtime_error("original target44 prepared extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original target44 immutable bank");
    shared::check(bank.view().write(raw), "original target44 once-only upload");
    result.roles[slot] = {spec(role).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original target44 readonly lease")};
  }
  return result;
}
}

namespace glm_original_target44_roles {
moe::Data load_moe(const std::filesystem::path &directory) {
  return moe::Data::load(directory, kMoeIndexSha256);
}
}
