#pragma once

#include "weights.h"

namespace qwen35 {
// A model-specific, offline-prepared default, not a Graph/Compute capacity or
// a limit on explicit vocabulary selection. Other sources keep their policy.
inline constexpr std::string_view kDefaultK3Vocabulary = "mtp-draft-vocabulary-k3.ids";
inline constexpr std::size_t kDefaultK3VocabularyRows = 114688;

inline bool has_qualified_k3_preference(const PreparedWeightSource &source) {
  const auto &g = source.roles.geometry;
  return source.draft_head && source.draft_vocabulary &&
      source.draft_vocabulary->size() == kDefaultK3VocabularyRows &&
      source.roles.representation == prepared::SourceRepresentation::kHfOriginal &&
      source.roles.cache_calibration.empty() &&
      source.roles.cache_policy != prepared::CachePolicy::kFloat8OriginalCalibration &&
      g.hidden == 2048 && g.intermediate == 6144 && g.vocabulary == 248320 &&
      g.query_heads == 8 && g.kv_heads == 2 && g.key_depth == 256 && g.value_depth == 256 &&
      g.rotary == 64 && g.gdn_key_heads == 16 && g.gdn_value_heads == 16 &&
      g.gdn_key_depth == 128 && g.gdn_value_depth == 128 && g.conv_kernel == 4 &&
      g.context_limit == 262144 && g.rope_theta == 10000000 && g.epsilon == 1e-6f &&
      g.tied_head && g.mtp_layers == 1 && g.gdn_layout == GdnHeadLayout::kGrouped &&
      g.layers == "GGGAGGGAGGGAGGGAGGGAGGGA" &&
      !source.roles.selected.empty() &&
      std::ranges::all_of(source.roles.selected, [](const auto &role) {
        return (role.dtype == "BF16" || (role.dtype == "F32" && role.shape.size() <= 1)) &&
            (role.projection_codec == prepared::ProjectionCodec::kDense ||
             role.projection_codec == prepared::ProjectionCodec::kUnspecified);
      });
}
} // namespace qwen35
