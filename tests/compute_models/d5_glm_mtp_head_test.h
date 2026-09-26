#pragma once

#include <span>
#include <vector>

#include "d5_qwen_gguf_vocabulary_cases.h"

// Defined by each route's existing vocabulary consumer. The actual decoder
// output crosses a declared host staging boundary; the reference stays separate.
namespace d5_glm_mtp_test {
void check_head(const d5_qwen_gguf_vocabulary_cases::Case &profile,
                std::span<const std::vector<float>> observed,
                std::span<const std::vector<float>> reference);
}
