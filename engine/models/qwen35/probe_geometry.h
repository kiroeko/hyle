#pragma once

#include "model_geometry.h"
#include <filesystem>

namespace qwen35 {
ModelGeometry read_weight_geometry(const std::filesystem::path &manifest);
}

namespace qwen35::probe {
// These historical diagnostics compare fixed 2B vectors and page layouts.
// Reject other geometries before weight IO; they do not qualify new models.
inline ModelGeometry read_two_b_geometry(const std::filesystem::path &inputs) {
  auto geometry = read_weight_geometry(inputs / "weights.tsv");
  if (geometry.hidden != 2048 || geometry.intermediate != 6144 || geometry.vocabulary != 248320 ||
      geometry.layers != "GGGAGGGAGGGAGGGAGGGAGGGA" || !geometry.tied_head ||
      geometry.query_heads != 8 || geometry.kv_heads != 2 ||
      geometry.key_depth != 256 || geometry.value_depth != 256 ||
      geometry.gdn_key_heads != 16 || geometry.gdn_value_heads != 16 ||
      geometry.gdn_key_depth != 128 || geometry.gdn_value_depth != 128 || geometry.conv_kernel != 4)
    throw std::runtime_error("this diagnostic requires the fixed Qwen3.5-2B geometry");
  return geometry;
}
} // namespace qwen35::probe
