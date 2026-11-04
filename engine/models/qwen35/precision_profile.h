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
    return f32_ ? hyle::compute::DataType::kFloat32
         : f16_ ? hyle::compute::DataType::kFloat16
                : hyle::compute::DataType::kBFloat16;
  }
  constexpr hyle::compute::DataType cache_type() const noexcept {
    return f16_ ? hyle::compute::DataType::kFloat16 : hyle::compute::DataType::kBFloat16;
  }
  constexpr std::size_t activation_bytes() const noexcept { return f32_ ? 4 : 2; }
  constexpr bool is_f16() const noexcept { return f16_; }
  friend constexpr bool operator==(PrecisionProfile, PrecisionProfile) noexcept = default;

  static PrecisionProfile from_roles(const prepared::Roles &roles) {
    const bool has_nf4 = std::ranges::any_of(roles.selected, [](const auto &role) {
      return role.projection_codec == prepared::ProjectionCodec::kNf4DoubleQuant;
    });
    const bool has_static_fp8 = std::ranges::any_of(roles.selected, [](const auto &role) {
      return role.projection_codec == prepared::ProjectionCodec::kModelOptStaticFp8;
    });
    if (has_static_fp8 && !roles.static_fp8_policy.enabled)
      throw std::runtime_error("static FP8 source requires its selected stored-scale reader policy");
    if (roles.static_fp8_policy.enabled && (roles.nf4_policy.enabled || roles.float8_policy.enabled ||
        roles.ct_mixed_policy.enabled() || roles.ct_asymmetric_policy.enabled || roles.mlx_policy.enabled() ||
        !roles.cache_calibration.empty()))
      throw std::runtime_error("incompatible static FP8 source precision policies");
    if ((has_nf4 && !roles.nf4_policy.enabled) ||
        (roles.nf4_policy.enabled && (roles.float8_policy.enabled ||
            roles.ct_mixed_policy.enabled() || roles.ct_asymmetric_policy.enabled || roles.mlx_policy.enabled() ||
            !roles.cache_calibration.empty())))
      throw std::runtime_error("incompatible source activation precision policies");
    PrecisionProfile result;
    result.f16_ = roles.nf4_policy.enabled;
    // GGUF Q8_0/Q4_K_M ordinary graphs keep their quantized projections and
    // residuals in F32. NextN and the other source readers retain their actual
    // original boundaries. KV/query/output remain independently BF16.
    const auto quantized = [](const auto &role) {
      return role.dtype == "Q8_0" || role.dtype == "Q4_K" || role.dtype == "Q6_K";
    };
    result.f32_ = roles.representation == prepared::SourceRepresentation::kGgufConverted &&
        std::ranges::none_of(roles.selected, [](const auto &role) { return role.name.starts_with("mtp."); }) &&
        std::ranges::any_of(roles.selected, [](const auto &role) { return role.dtype == "Q8_0" || role.dtype == "Q4_K"; }) &&
        std::ranges::all_of(roles.selected, [&](const auto &role) {
          return (role.meaning != prepared::ParameterMeaning::kLinearWeight &&
                  role.meaning != prepared::ParameterMeaning::kEmbeddingRows) || quantized(role);
        });
    return result;
  }

private:
  bool f16_ = false;
  bool f32_ = false;
};
} // namespace qwen35
