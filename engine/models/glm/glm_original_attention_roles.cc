#include "glm_original_attention_roles.h"

namespace glm_original_attention_roles {
std::size_t index(RawRole role) {
  const auto value = static_cast<std::size_t>(role);
  if (value >= kRawRoles) throw std::runtime_error("unknown original GLM attention raw role");
  return value;
}
}

namespace glm_original_attention_roles {
std::size_t index(Parameter parameter) {
  const auto value = static_cast<std::size_t>(parameter);
  if (value >= kParameters) throw std::runtime_error("unknown original GLM attention parameter");
  return value;
}
}

namespace glm_original_attention_roles {
c::TensorDesc raw_description(RawRole role) {
  const auto &spec = kSpecs[index(role)];
  if (spec.columns == 0)
    return shared::take(c::TensorDesc::dense(spec.type, std::array{spec.rows}), "original attention norm type");
  return shared::take(c::TensorDesc::dense(spec.type, std::array{spec.rows, spec.columns}), "original attention raw type");
}
}

namespace glm_original_attention_roles {
c::TensorDesc description(Parameter parameter) {
  switch (parameter) {
  case Parameter::kQueryProjection:
    return shared::take(c::float8_weight_description(kQueryRank, kHeads * kNope,
        c::Float8ScaleMode::kFloat32Block128), "original q_b packed type");
  case Parameter::kKvProjection:
    return shared::take(c::float8_weight_description(kHidden, kLatent,
        c::Float8ScaleMode::kFloat32Block128), "original kv_a packed type");
  case Parameter::kOutput:
    return shared::take(c::float8_weight_description(kHeads * kValue, kHidden,
        c::Float8ScaleMode::kFloat32Block128), "original o packed type");
  case Parameter::kKvNorm: return raw_description(RawRole::kKvNorm);
  case Parameter::kKvB: return raw_description(RawRole::kKvB);
  case Parameter::kPostNorm: return raw_description(RawRole::kPostNorm);
  default: throw std::runtime_error("unknown original GLM attention description");
  }
}
}

namespace glm_original_attention_roles {
std::span<const std::byte> payload(const Data &data, Parameter parameter) {
  return data.payload(parameter);
}
}

namespace glm_original_attention_roles {
Roles bind(const Data &data, const c::StorageOwner &owner) {
  if (!owner) throw std::runtime_error("original attention needs its actual Graph storage owner");
  Roles result;
  for (std::size_t slot = 0; slot < kParameters; ++slot) {
    const auto parameter = static_cast<Parameter>(slot);
    const auto desc = description(parameter);
    const auto extent = shared::take(desc.required_bytes(), "original attention ready bank extent");
    const auto raw = payload(data, parameter);
    if (raw.size() != extent) throw std::runtime_error("original attention ready payload extent differs");
    auto buffer = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original attention immutable bank");
    shared::check(buffer.view().write(raw), "original attention once-only bank upload");
    result[slot] = {kNames[slot], desc, owner,
        shared::take(owner.adopt_buffer(std::move(buffer)), "original attention readonly lease")};
  }
  return result;
}
}
