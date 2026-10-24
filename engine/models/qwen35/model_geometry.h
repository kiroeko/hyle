#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qwen35 {

enum class GdnHeadLayout { kGrouped, kCyclic };

// Identified source mathematics, not a runtime/provider selection. Every axis
// is checked against the complete original role inventory before weight IO.
// The same immutable geometry builds ordinary, verifier and NextN graphs.
struct ModelGeometry final {
  std::int64_t hidden = 0, intermediate = 0, vocabulary = 0;
  std::int64_t query_heads = 0, kv_heads = 0, key_depth = 0, value_depth = 0, rotary = 0;
  std::int64_t gdn_key_heads = 0, gdn_value_heads = 0, gdn_key_depth = 0, gdn_value_depth = 0;
  std::int64_t conv_kernel = 0, context_limit = 0;
  float rope_theta = 0, epsilon = 0;
  bool tied_head = false;
  std::size_t mtp_layers = 0;
  GdnHeadLayout gdn_layout = GdnHeadLayout::kGrouped;
  friend bool operator==(const ModelGeometry &, const ModelGeometry &) = default;
  std::string layers; // G: gated delta net; A: full attention. Target only.

  static std::int64_t multiply(std::int64_t a, std::int64_t b) {
    if (a <= 0 || b <= 0 || a > INT64_MAX / b)
      throw std::runtime_error("model geometry product is invalid or overflows");
    return a * b;
  }
  static std::int64_t add(std::int64_t a, std::int64_t b) {
    if (a <= 0 || b <= 0 || a > INT64_MAX - b)
      throw std::runtime_error("model geometry sum is invalid or overflows");
    return a + b;
  }
  bool is_gdn(std::size_t layer) const { return layers.at(layer) == 'G'; }
  std::size_t attention_layers() const noexcept { return static_cast<std::size_t>(std::ranges::count(layers, 'A')); }
  std::size_t recurrent_layers() const noexcept { return static_cast<std::size_t>(std::ranges::count(layers, 'G')); }
  std::size_t attention_slot(std::size_t layer) const {
    if (layer >= layers.size() || is_gdn(layer)) throw std::runtime_error("layer has no full-attention KV slot");
    return static_cast<std::size_t>(std::count(layers.begin(), layers.begin() + static_cast<std::ptrdiff_t>(layer), 'A'));
  }
  std::int64_t gdn_key_width() const { return multiply(gdn_key_heads, gdn_key_depth); }
  std::int64_t gdn_value_width() const { return multiply(gdn_value_heads, gdn_value_depth); }
  std::int64_t gdn_qkv_width() const { return add(multiply(2, gdn_key_width()), gdn_value_width()); }
  std::int64_t attention_width() const { return multiply(query_heads, value_depth); }
  std::int64_t query_gate_width() const { return multiply(query_heads, add(key_depth, value_depth)); }
  std::size_t hidden_bytes() const { return static_cast<std::size_t>(multiply(hidden, 2)); }
  std::size_t key_row_bytes() const { return static_cast<std::size_t>(multiply(multiply(kv_heads, key_depth), 2)); }
  std::size_t value_row_bytes() const { return static_cast<std::size_t>(multiply(multiply(kv_heads, value_depth), 2)); }
  std::string_view head_role() const noexcept {
    return tied_head ? "model.language_model.embed_tokens.weight" : "lm_head.weight";
  }
  void validate() const {
    for (const auto value : {hidden, intermediate, vocabulary, query_heads, kv_heads, key_depth, value_depth,
         rotary, gdn_key_heads, gdn_value_heads, gdn_key_depth, gdn_value_depth, conv_kernel, context_limit})
      if (value <= 0) throw std::runtime_error("model geometry requires positive dimensions");
    if (vocabulary > INT32_MAX || context_limit > INT32_MAX || query_heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        key_depth > UINT32_MAX || value_depth > UINT32_MAX || gdn_key_heads > INT32_MAX ||
        gdn_value_heads > INT32_MAX || query_heads % kv_heads ||
        gdn_value_heads % gdn_key_heads || rotary % 2 || rotary > key_depth || conv_kernel < 2 ||
        !std::isfinite(rope_theta) || rope_theta <= 0 || !std::isfinite(epsilon) || epsilon <= 0 ||
        layers.empty() || layers.size() > UINT32_MAX ||
        std::ranges::any_of(layers, [](char layer) { return layer != 'G' && layer != 'A'; }) || mtp_layers > 1)
      throw std::runtime_error("unsupported or invalid source Qwen geometry");
    (void)multiply(vocabulary, hidden); (void)multiply(intermediate, hidden);
    (void)multiply(hidden, 2); (void)multiply(gdn_qkv_width(), conv_kernel);
    (void)multiply(gdn_value_heads, multiply(gdn_key_depth, gdn_value_depth));
    (void)query_gate_width(); (void)attention_width(); (void)hidden_bytes();
    (void)key_row_bytes(); (void)value_row_bytes();
  }
};

inline ModelGeometry read_model_geometry(std::string_view line) {
  std::istringstream input{std::string(line)}; input.imbue(std::locale::classic());
  std::string tag, word, layout, extra;
  if (!(input >> tag) || tag != "geometry") throw std::runtime_error("prepared V2 input requires source geometry");
  const auto unsigned_value = [&]() {
    if (!(input >> word)) throw std::runtime_error("missing prepared geometry integer");
    std::uint64_t value = 0;
    const auto result = std::from_chars(word.data(), word.data() + word.size(), value);
    if (result.ec != std::errc{} || result.ptr != word.data() + word.size() || value > INT64_MAX)
      throw std::runtime_error("invalid or overflowing prepared geometry integer");
    return static_cast<std::int64_t>(value);
  };
  ModelGeometry geometry;
  for (auto *axis : {&geometry.hidden, &geometry.intermediate, &geometry.vocabulary, &geometry.query_heads,
       &geometry.kv_heads, &geometry.key_depth, &geometry.value_depth, &geometry.rotary,
       &geometry.gdn_key_heads, &geometry.gdn_value_heads, &geometry.gdn_key_depth, &geometry.gdn_value_depth,
       &geometry.conv_kernel, &geometry.context_limit}) *axis = unsigned_value();
  if (!(input >> geometry.rope_theta >> geometry.epsilon)) throw std::runtime_error("invalid prepared geometry floating parameters");
  const auto tied = unsigned_value(), mtp = unsigned_value();
  if (tied > 1 || mtp > 1 || !(input >> layout) || (layout != "grouped" && layout != "cyclic"))
    throw std::runtime_error("invalid prepared geometry head or source layout");
  geometry.tied_head = tied != 0; geometry.mtp_layers = static_cast<std::size_t>(mtp);
  geometry.gdn_layout = layout == "grouped" ? GdnHeadLayout::kGrouped : GdnHeadLayout::kCyclic;
  input >> std::ws;
  if (input.peek() != '"' || !(input >> std::quoted(geometry.layers)) || (input >> extra))
    throw std::runtime_error("invalid prepared geometry target layer sequence");
  geometry.validate(); return geometry;
}
} // namespace qwen35
