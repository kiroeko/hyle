#pragma once

#include "hyle/compute/low_precision.h"

#include <array>
#include <string_view>

namespace qwen35 {

// Source-encoding facts for the current model implementation. Individual
// operations retain separate capabilities; a backend still qualifies its own
// actual descriptors and execution resources.
struct GgufEncodingCapabilities final {
  std::string_view dtype;
  hyle::compute::GgufQuantizedType encoding;
  bool f32_operand;
  bool stored_draft_head;
  bool copy_encoded_rows;
  bool concurrent_cold_family;
};

inline constexpr std::array kGgufEncodingCapabilities{
    GgufEncodingCapabilities{"Q4_K", hyle::compute::GgufQuantizedType::kQ4K, true, true, true, true},
    GgufEncodingCapabilities{"Q5_K", hyle::compute::GgufQuantizedType::kQ5K, true, true, true, true},
    GgufEncodingCapabilities{"Q6_K", hyle::compute::GgufQuantizedType::kQ6K, true, true, true, true},
    GgufEncodingCapabilities{"Q8_0", hyle::compute::GgufQuantizedType::kQ8_0, true, true, true, true},
};

constexpr const GgufEncodingCapabilities *gguf_capabilities(std::string_view dtype) noexcept {
  for (const auto &entry : kGgufEncodingCapabilities)
    if (entry.dtype == dtype) return &entry;
  return nullptr;
}

constexpr const GgufEncodingCapabilities *gguf_capabilities(hyle::compute::GgufQuantizedType encoding) noexcept {
  for (const auto &entry : kGgufEncodingCapabilities)
    if (entry.encoding == encoding) return &entry;
  return nullptr;
}

constexpr bool supports_f32_gguf_operand(std::string_view dtype) noexcept {
  const auto *capability = gguf_capabilities(dtype);
  return dtype == "F32" || (capability && capability->f32_operand);
}

} // namespace qwen35
