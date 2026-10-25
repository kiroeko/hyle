#pragma once

#include <algorithm>
#include <stdexcept>

#include "prepared_roles.h"
#include "hyle/compute/tensor_desc.h"

namespace qwen35 {

// Source mathematics fixed before state/Graph preparation. A F16 stored weight
// by itself does not select F16 compute (the existing GPTQ reader uses BF16).
class PrecisionProfile final {
public:
  constexpr PrecisionProfile() noexcept = default;
  constexpr hyle::compute::DataType activation_type() const noexcept {
    return f16_ ? hyle::compute::DataType::kFloat16
                : hyle::compute::DataType::kBFloat16;
  }
  constexpr bool is_f16() const noexcept { return f16_; }
  friend constexpr bool operator==(PrecisionProfile, PrecisionProfile) noexcept = default;

  static PrecisionProfile from_roles(const prepared::Roles &roles) {
    const bool has_nf4 = std::ranges::any_of(roles.selected, [](const auto &role) {
      return role.projection_codec == prepared::ProjectionCodec::kNf4DoubleQuant;
    });
    if ((has_nf4 && !roles.nf4_policy.enabled) ||
        (roles.nf4_policy.enabled && (roles.float8_policy.enabled ||
            roles.ct_mixed_policy.enabled || roles.ct_asymmetric_policy.enabled || roles.mlx_policy.enabled() ||
            !roles.omitted_cache_calibration.empty())))
      throw std::runtime_error("incompatible source activation precision policies");
    PrecisionProfile result;
    result.f16_ = roles.nf4_policy.enabled;
    return result;
  }

private:
  bool f16_ = false;
};
} // namespace qwen35
