#pragma once

#include <array>
#include <cstdint>

namespace qwen35 {

// Special tokens of this consumer's Qwen tokenizer. RequestStopPolicy owns
// extra stop IDs and ignore-model-EOS policy; Compute/Graph know neither.
inline constexpr std::array<std::int32_t, 2> model_eos_tokens{248044, 248046};

constexpr bool is_end_of_generation(std::int32_t token) noexcept {
  for (const auto eos : model_eos_tokens)
    if (token == eos) return true;
  return false;
}

} // namespace qwen35
