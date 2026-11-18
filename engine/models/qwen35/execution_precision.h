#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>

#include "precision_profile.h"

namespace qwen35 {

// Cold execution mathematics is distinct from the unchanged source reader.
// A missing/same choice preserves every existing source contract. Actual
// backend/Graph qualification and performance preference remain separate.
inline PrecisionProfile execution_precision(const prepared::Roles &roles,
    std::optional<PrecisionProfile> selected = {}) {
  const auto source = PrecisionProfile::from_roles(roles);
  if (!selected || *selected == source) return source;
  const bool dense_floating = !roles.selected.empty() &&
      std::ranges::all_of(roles.selected, [](const auto &role) {
        return (role.dtype == "F16" || role.dtype == "BF16" || role.dtype == "F32") &&
            (role.projection_codec == prepared::ProjectionCodec::kUnspecified ||
             role.projection_codec == prepared::ProjectionCodec::kDense);
      });
  const bool fixed_reader_math = roles.nf4_policy.enabled || roles.float8_policy.enabled ||
      roles.ct_mixed_policy.enabled() || roles.ct_asymmetric_policy.enabled ||
      roles.mlx_policy.enabled() || roles.static_fp8_policy.enabled;
  // KV encoding/calibration is independent. Its presence is not an activation
  // veto; the existing source/carrier/scale validation still applies in full.
  if (selected->activation_type() != hyle::compute::DataType::kFloat32 ||
      selected->cache_type() != source.cache_type() || !dense_floating || fixed_reader_math)
    throw std::invalid_argument("execution precision requires dense floating source roles and unchanged cache mathematics");
  return *selected;
}

} // namespace qwen35
