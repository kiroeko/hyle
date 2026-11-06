#pragma once

#include "prepared_roles.h"
#include "gguf_capabilities.h"

namespace qwen35 {

// The currently built bounded device draft chain has up to three proposals.
inline constexpr std::size_t kMaximumDeviceMtpDrafts = 3;

// This describes the selected source projection, not whether a backend can
// execute a particular MTP route. That is queried from the actual backend.
inline bool supports_stored_draft_projection(const prepared::Roles &roles) {
  return std::ranges::any_of(roles.selected, [&](const auto &role) {
    const auto *capability = gguf_capabilities(role.dtype);
    return role.name == roles.geometry.head_role() &&
        (role.projection_codec == prepared::ProjectionCodec::kDense ||
         role.projection_codec == prepared::ProjectionCodec::kUnspecified) &&
        (role.dtype == "BF16" || role.dtype == "F16" || role.dtype == "F32" ||
         (capability && capability->stored_draft_head));
  });
}

} // namespace qwen35
