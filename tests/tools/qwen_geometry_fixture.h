#pragma once
#include "../../engine/models/qwen35/model_geometry.h"

namespace qwen_test {
inline std::string geometry_line(const qwen35::ModelGeometry &g) {
  std::ostringstream out; out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<float>::max_digits10) << "geometry "
      << g.hidden << ' ' << g.intermediate << ' ' << g.vocabulary << ' ' << g.query_heads << ' '
      << g.kv_heads << ' ' << g.key_depth << ' ' << g.value_depth << ' ' << g.rotary << ' '
      << g.gdn_key_heads << ' ' << g.gdn_value_heads << ' ' << g.gdn_key_depth << ' '
      << g.gdn_value_depth << ' ' << g.conv_kernel << ' ' << g.context_limit << ' '
      << g.rope_theta << ' ' << g.epsilon << ' ' << g.tied_head << ' ' << g.mtp_layers << ' '
      << (g.gdn_layout == qwen35::GdnHeadLayout::kGrouped ? "grouped" : "cyclic") << ' ' << std::quoted(g.layers);
  return out.str();
}
inline qwen35::ModelGeometry q35(bool converted = false) {
  auto g = qwen35::read_model_geometry("geometry 2048 6144 248320 8 2 256 256 64 16 16 128 128 4 262144 10000000 0.000001 1 1 grouped \"GGGAGGGAGGGAGGGAGGGAGGGA\"");
  if (converted) g.gdn_layout = qwen35::GdnHeadLayout::kCyclic;
  return g;
}
inline qwen35::ModelGeometry q27(bool converted = true) {
  auto g = q35(converted); g.hidden = 5120; g.intermediate = 17408;
  g.query_heads = 24; g.kv_heads = 4; g.gdn_value_heads = 48; g.tied_head = false;
  g.layers.clear(); for (int i = 0; i < 16; ++i) g.layers += "GGGA";
  g.validate(); return g;
}
} // namespace qwen_test
