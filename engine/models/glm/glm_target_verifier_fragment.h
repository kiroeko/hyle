#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "../detail/component_definition.h"
#include "hyle/compute/model_ops.h"
#include "../detail/model_sampling_fragment.h"

// Pure model-local target-head construction. There is no
// loader, invocation, provider choice, or target model execution in this file.
namespace glm_target_verifier_fragment {
namespace c = hyle::compute;
namespace sem = hyle::inference::model_definition;
using Value = sem::ValueId;
inline constexpr std::int64_t kHidden = 4096, kVocabulary = 154880, kRows = 2;
static_assert(kVocabulary <= std::numeric_limits<std::int32_t>::max());

struct Ports final {
  Value logits; // F32[2,V], one full shared-head projection.
  Value scores; // F32[2,1], actual stable TopK1 scores.
  Value ids;    // I32[2,1], actual full-vocabulary IDs.
  // Each row is a distinct Copy I32[1], suitable for an actual output lease
  // bound directly into StateDraft pending, without host token reconstruction.
  std::array<Value, std::size_t(kRows)> row_ids;
};

Ports append(sem::Fragment &fragment, Value final_normalized_taps,
                    Value original_shared_head, std::int64_t hidden = kHidden,
                    std::int64_t vocabulary = kVocabulary);

} // namespace glm_target_verifier_fragment
