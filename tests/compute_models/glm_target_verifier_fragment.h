#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/model_ops.h"

// Pure model-local construction and greedy K1 consumer policy. There is no
// loader, invocation, provider choice, or target model execution in this file.
namespace glm_target_verifier_fragment {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
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

inline Ports append(sem::Fragment &fragment, Value final_normalized_taps,
                    Value original_shared_head, std::int64_t hidden = kHidden,
                    std::int64_t vocabulary = kVocabulary) {
  if (hidden <= 0 || vocabulary <= 0 ||
      vocabulary > std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("GLM K1 verifier needs positive hidden and an I32 vocabulary extent");
  const auto tap_description = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{kRows, hidden});
  const auto head_description = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{vocabulary, hidden});
  if (!tap_description || !head_description || !tap_description->required_bytes() ||
      !head_description->required_bytes() ||
      final_normalized_taps.ordinal() >= fragment.descriptions.size() ||
      original_shared_head.ordinal() >= fragment.descriptions.size() ||
      fragment.description(final_normalized_taps) != *tap_description ||
      fragment.description(original_shared_head) != *head_description)
    throw std::runtime_error("GLM K1 verifier needs bounded final normalized BF16[2,H] taps and original BF16[V,H] head");

  // The boundary is AFTER target model.norm. Never apply the distinct NextN
  // shared_head.norm, normalize again, or narrow a caller's F32 tap implicitly.
  // Constant ownership/authentication belongs to the caller's ordinary Graph.
  Ports result;
  result.logits = fragment.one(c::model_ops::bfloat_linear_f32(),
      {final_normalized_taps, original_shared_head});
  const auto top = fragment.append(c::Operation(c::TopK{.count = 1, .stable = true}),
      {result.logits}, 2);
  result.scores = top[0]; result.ids = top[1];
  const auto scalar_shape = c::TensorShape::create(std::array<std::int64_t, 1>{1}).value();
  for (std::size_t row = 0; row < result.row_ids.size(); ++row) {
    const auto one = fragment.one(c::Operation(c::View::slice(0, std::int64_t(row), 1).value()),
        {result.ids});
    const auto flattened = fragment.one(c::Operation(c::View::reshape(scalar_shape).value()), {one});
    result.row_ids[row] = fragment.one(c::Operation(c::Copy{}), {flattened});
  }
  return result;
}

// Mandatory current token t0 has already been consumed by BOTH checkpoints.
// These labels deliberately exclude the pre-t0 base version: accepting zero
// draft tokens retains first, not base; accepting proposal1 retains second.
enum class Checkpoint : std::uint8_t { kAfterCurrent, kAfterProposal };
struct K1Decision final {
  std::uint32_t accepted_drafts;
  Checkpoint checkpoint;
  std::size_t next_pending_row;
  std::int32_t next_pending; // Diagnostic policy value, not a new uploaded token.
};
inline K1Decision decide_k1(std::int32_t actual_proposal,
                            std::span<const std::int32_t> actual_target_ids,
                            std::int64_t vocabulary = kVocabulary) {
  if (vocabulary <= 0 || vocabulary > std::numeric_limits<std::int32_t>::max() ||
      actual_target_ids.size() != std::size_t(kRows) || actual_proposal < 0 ||
      actual_proposal >= vocabulary)
    throw std::runtime_error("GLM greedy K1 policy needs one proposal and two actual target IDs");
  for (const auto id : actual_target_ids)
    if (id < 0 || id >= vocabulary)
      throw std::runtime_error("GLM greedy K1 target ID is outside the complete vocabulary");
  const std::uint32_t accepted = actual_proposal == actual_target_ids[0] ? 1U : 0U;
  // The consumer binds Ports.row_ids[next_pending_row]'s real result lease.
  // next_pending is only for observations; this helper submits no computation.
  return {accepted, accepted ? Checkpoint::kAfterProposal : Checkpoint::kAfterCurrent,
      std::size_t(accepted), actual_target_ids[accepted]};
}
} // namespace glm_target_verifier_fragment
