#pragma once
#include <array>
#include <span>
#include <vector>
#include <ggml.h>
#include "../compute_models/cpu_c5_non_gguf_fixture.h"

inline std::array<float, 8> apple_gguf_dot_reference(ggml_type type, std::int64_t k,
    const std::vector<std::byte> &, const std::vector<float> &decoded,
    std::span<const float> activation, bool f32_activation) {
  std::array<float, 8> result{};
  for (std::size_t row = 0; row < result.size(); ++row) {
    double value = 0;
    for (std::int64_t col = 0; col < k; ++col) {
      float input = activation[col];
      if (!f32_activation && type == GGML_TYPE_BF16)
        input = ggml_bf16_to_fp32(ggml_fp32_to_bf16(input));
      else if (!f32_activation && type == GGML_TYPE_F16)
        input = ggml_fp16_to_fp32(ggml_fp32_to_fp16(input));
      value += double(decoded[row * k + col]) * input;
    }
    result[row] = static_cast<float>(value);
  }
  return result;
}

// Existing test-oracle hook; no backend capability macro is manufactured.
#define HYLE_C5_GGUF_DOT_REFERENCE apple_gguf_dot_reference
#include "../compute_models/cpu_c5_gguf_fixture.h"
#undef HYLE_C5_GGUF_DOT_REFERENCE
