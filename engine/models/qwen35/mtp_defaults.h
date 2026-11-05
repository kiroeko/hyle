#pragma once

#include "weights.h"
#include "../../inference/mtp_policy.h"

namespace qwen35 {
// Retain the installed name for existing prepared directories. It is not a
// model/row-count fingerprint or a restriction on explicit asset selection.
inline constexpr std::string_view kDefaultK3Vocabulary = "mtp-draft-vocabulary-k3.ids";

struct SourceMtpCapabilities final {
  bool nextn = false, stored_projection = false;
  hyle::inference::mtp::TargetProjection target = hyle::inference::mtp::TargetProjection::kDense;
};

inline SourceMtpCapabilities source_mtp_capabilities(const prepared::Roles &roles) {
  SourceMtpCapabilities result;
  // These readers have no NextN implementation. The original role reader
  // checks source identity, geometry and the complete parameter inventory.
  result.nextn = roles.geometry.mtp_layers == 1 && !roles.mlx_policy.enabled() &&
      roles.ct_mixed_policy.kind != prepared::CtMixedPolicy::Kind::kBlock128;
  const auto head = std::ranges::find(roles.selected, roles.geometry.head_role(), &prepared::Role::name);
  if (head == roles.selected.end() || head->projection_group_size ||
      (head->projection_codec != prepared::ProjectionCodec::kDense &&
       head->projection_codec != prepared::ProjectionCodec::kUnspecified)) return result;
  if (head->dtype == "BF16" || head->dtype == "F16" || head->dtype == "F32") result.stored_projection = true;
  else if (head->dtype == "Q8_0" || head->dtype == "Q4_K" || head->dtype == "Q6_K") {
    result.stored_projection = true;
    result.target = hyle::inference::mtp::TargetProjection::kRowBlockQuantized;
  }
  return result;
}
} // namespace qwen35
