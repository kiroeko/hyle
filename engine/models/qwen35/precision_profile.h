#pragma once

#include <algorithm>
#include <stdexcept>

#include "prepared_roles.h"
#include "gguf_capabilities.h"
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

  // Model equations such as a vision tower can prescribe F32 activations
  // independently of the target LM and the original physical weight encoding.
  static constexpr PrecisionProfile float32_activations() noexcept {
    PrecisionProfile result; result.f32_=true; return result;
  }

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
    // UNADOPTED NUMERICAL EXPERIMENT: ordinary HF BF16 target only.
    // Both loader/cache planning and Model construction already consume this
    // cold function. Do not reinterpret stored weights or widen attention/KV.
    // A selected NextN or encoded projection retains its original policy.
    const bool experimental_bf16_target =
        roles.representation == prepared::SourceRepresentation::kHfOriginal &&
        roles.cache_policy == prepared::CachePolicy::kDefault && roles.cache_calibration.empty() &&
        std::ranges::any_of(roles.selected, [](const auto &role) {
          return role.name == "model.language_model.embed_tokens.weight" && role.dtype == "BF16";
        }) &&
        std::ranges::all_of(roles.selected, [](const auto &role) {
          const bool dense_bf16 = role.dtype == "BF16" &&
              (role.projection_codec == prepared::ProjectionCodec::kUnspecified ||
               role.projection_codec == prepared::ProjectionCodec::kDense);
          const bool original_f32_auxiliary = role.dtype == "F32" &&
              role.projection_codec == prepared::ProjectionCodec::kUnspecified;
          return !role.name.starts_with("mtp.") && (dense_bf16 || original_f32_auxiliary);
        });
    if (experimental_bf16_target) return float32_activations();
    PrecisionProfile result;
    result.f16_ = roles.nf4_policy.enabled;
    // Supported GGUF graphs keep target/NextN projections and residuals in
    // F32, including original dense F32 operands. KV/query/attention output
    // remain independently BF16.
    result.f32_ = roles.representation == prepared::SourceRepresentation::kGgufConverted &&
        std::ranges::any_of(roles.selected, [](const auto &role) {
          const auto *capability = gguf_capabilities(role.dtype);
          return capability && capability->f32_operand;
        }) &&
        std::ranges::all_of(roles.selected, [&](const auto &role) {
          return (role.meaning != prepared::ParameterMeaning::kLinearWeight &&
                  role.meaning != prepared::ParameterMeaning::kEmbeddingRows) || supports_f32_gguf_operand(role.dtype);
        });
    return result;
  }

private:
  bool f16_ = false;
  bool f32_ = false;
};
} // namespace qwen35
