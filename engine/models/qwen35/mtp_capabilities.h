#pragma once

#include "prepared_roles.h"

namespace qwen35 {

// This describes the selected source projection, not whether a backend can
// execute a particular MTP route. That is queried from the actual backend.
inline bool supports_stored_draft_projection(const prepared::Roles &roles) {
  return std::ranges::any_of(roles.selected, [&](const auto &role) {
    return role.name == roles.geometry.head_role() &&
        (role.projection_codec == prepared::ProjectionCodec::kDense ||
         role.projection_codec == prepared::ProjectionCodec::kUnspecified) &&
        (role.dtype == "BF16" || role.dtype == "F16" || role.dtype == "F32" ||
         role.dtype == "Q8_0" || role.dtype == "Q4_K" || role.dtype == "Q6_K");
  });
}

} // namespace qwen35
