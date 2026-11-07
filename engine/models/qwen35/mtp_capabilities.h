#pragma once

#include "prepared_roles.h"
#include "gguf_capabilities.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace qwen35 {

// The currently built bounded device draft chain has up to three proposals.
inline constexpr std::size_t kMaximumDeviceMtpDrafts = 3;

// Whether the original head can supply independent vocabulary rows. Encoded
// rows additionally require a backend with a bit-preserving gather operation.
inline bool supports_draft_vocabulary(const prepared::Roles &roles, bool encoded_rows_on_backend = true) {
  const auto head = std::ranges::find(roles.selected, roles.geometry.head_role(), &prepared::Role::name);
  if (head == roles.selected.end()) return false;
  const bool plain = head->dtype == "BF16" || head->dtype == "F16" || head->dtype == "F32";
  const auto *capability = gguf_capabilities(head->dtype);
  return (plain || (encoded_rows_on_backend && capability && capability->copy_encoded_rows)) &&
      (head->projection_codec == prepared::ProjectionCodec::kDense ||
       head->projection_codec == prepared::ProjectionCodec::kUnspecified) &&
      !head->projection_group_size &&
      head->shape == std::vector<std::int64_t>{roles.geometry.vocabulary, roles.geometry.hidden};
}

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
