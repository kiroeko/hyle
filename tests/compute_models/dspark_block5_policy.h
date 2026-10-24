#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <span>

#include "hyle/error/result.h"
#include "../../engine/inference/mtp_acceptance.h"

// Consumer policy for the formal fixed five-row DSpark draft. This does not
// change its trained shape, execute a verifier, or manufacture a pending lease.
namespace dspark_original_state_consumer {
// The authenticated original-role and current GGUF sidecar fragments both
// publish raw projection logits. A consumer of an already-sigmoided sidecar
// must declare kProbability; storage/quantization alone cannot select this.
enum class ConfidenceDomain : std::uint8_t { kRawLogit, kProbability };
struct DraftControl final {
  std::uint32_t max_drafts = 5;
  std::optional<float> minimum_confidence{}; // probability threshold in [0,1]
};
struct Block5Limits final {
  std::size_t output_quota = 0, context_rows = 0;
  std::int64_t vocabulary = 0;
  std::span<const std::int32_t> eos_tokens{};
  ConfidenceDomain confidence_domain = ConfidenceDomain::kRawLogit; // producer contract, not a draft preference
  DraftControl draft{};
};

struct VerifiedPrefix final {
  // DS control facts include quota/context exhaustion; acceptance itself uses
  // the same public selector as the other model consumers.
  std::uint32_t accepted_drafts = 0, consumed = 0;
  std::size_t pending_row = 0, emitted_count = 0;
  bool stopped_on_eos = false, output_quota_exhausted = false, context_exhausted = false;
  bool can_continue() const noexcept {
    return !stopped_on_eos && !output_quota_exhausted && !context_exhausted;
  }
};

// The current token was emitted by the preceding round but is not yet in the
// target cache. Each actual target row consumes one input and emits one output;
// the final emitted output stays pending. Zero means do not submit a round.
// Validate the complete metadata/control contract before invoking a producer.
// This capacity does not need confidence; only its later clipping needs data.
inline hyle::Result<std::size_t> block5_verification_capacity(Block5Limits limits) {
  if (limits.vocabulary <= 0 || limits.vocabulary > (std::numeric_limits<std::int32_t>::max)())
    return std::unexpected(hyle::Status::InvalidArgument("DSpark block5 needs a positive I32 vocabulary"));
  for (const auto token : limits.eos_tokens)
    if (token < 0 || token >= limits.vocabulary)
      return std::unexpected(hyle::Status::InvalidArgument("DSpark EOS token is outside the shared vocabulary"));
  if (limits.draft.max_drafts > 5)
    return std::unexpected(hyle::Status::InvalidArgument("DSpark maximum drafts must be in 0..5; trained geometry remains five rows"));
  if (limits.confidence_domain != ConfidenceDomain::kRawLogit &&
      limits.confidence_domain != ConfidenceDomain::kProbability)
    return std::unexpected(hyle::Status::InvalidArgument("DSpark confidence domain is unknown"));
  const auto threshold = limits.draft.minimum_confidence;
  if (threshold && (!std::isfinite(*threshold) || *threshold < 0.F || *threshold > 1.F))
    return std::unexpected(hyle::Status::InvalidArgument("DSpark confidence threshold must be a probability in [0,1]"));
  return (std::min)({std::size_t(limits.draft.max_drafts) + 1, limits.output_quota, limits.context_rows});
}

namespace block5_detail {
inline hyle::Result<std::size_t> confidence_rows(Block5Limits limits,
    std::size_t rows, std::span<const float> confidence, std::size_t expected) {
  const auto threshold = limits.draft.minimum_confidence;
  // Ordinary target-only steps, exhausted tails and a disabled threshold do
  // not require executing the trained draft or reading its confidence result.
  if (rows <= 1 || !threshold || *threshold == 0.F) return rows;
  if (confidence.size() != expected)
    return std::unexpected(hyle::Status::InvalidArgument("DSpark threshold policy needs its exact original conditional confidence rows"));
  for (const auto value : confidence)
    if (std::isnan(value) || (limits.confidence_domain == ConfidenceDomain::kProbability &&
        (!std::isfinite(value) || value < 0.F || value > 1.F)))
      return std::unexpected(hyle::Status::InvalidArgument("DSpark confidence value differs from its declared domain"));
  // Compare original logits directly with a once-converted threshold. Do not
  // apply sigmoid to an output whose domain already is probability.
  const auto cutoff = limits.confidence_domain == ConfidenceDomain::kProbability ? double(*threshold) :
      *threshold == 0.F ? -(std::numeric_limits<double>::infinity)() :
      *threshold == 1.F ? (std::numeric_limits<double>::infinity)() :
      std::log(double(*threshold)) - std::log1p(-double(*threshold));
  for (std::size_t draft = 0; draft + 1 < rows; ++draft)
    if (double(confidence[draft]) < cutoff) return draft + 1;
  return rows;
}
} // namespace block5_detail

inline hyle::Result<std::size_t> block5_verification_rows(Block5Limits limits,
    std::span<const float> confidence = {}) {
  HYLE_ASSIGN_OR_RETURN(const auto rows, block5_verification_capacity(limits));
  return block5_detail::confidence_rows(limits, rows, confidence, 5);
}

// A sampled Markov prefix exposes only its prepared candidates. Each confidence
// uses the preceding sampled IDs, before this candidate's own draw. Truncating
// at its first low row may therefore discard that unpublished draw and use the
// same address for the target final sample, as in Qwen recursive proposals.
inline hyle::Result<std::size_t> block5_sampled_verification_rows(Block5Limits limits,
    std::span<const float> conditional_confidence = {}) {
  HYLE_ASSIGN_OR_RETURN(const auto rows, block5_verification_capacity(limits));
  return block5_detail::confidence_rows(limits, rows, conditional_confidence, rows ? rows - 1 : 0);
}

// proposed is the original [current, proposal1, ..., proposal5], even when
// only a tail of 1..6 target rows is verified; an ordinary one-row target step
// can supply just [current] without executing the draft. A matched EOS or the last quota
// output is emitted and kept pending, not consumed merely to obtain a bonus.
// Bind target row pending_row's original I32 result lease into the selected
// after-current/after-proposal checkpoint. The helper uploads no host token.
inline hyle::Result<VerifiedPrefix> decide_block5(std::span<const std::int32_t> proposed,
    std::span<const std::int32_t> target_ids, Block5Limits limits, std::span<const float> confidence = {}) {
  HYLE_ASSIGN_OR_RETURN(const auto rows, block5_verification_rows(limits, confidence));
  if (!rows)
    return std::unexpected(hyle::Status::FailedPrecondition("DSpark block5 has no remaining output or context capacity"));
  if ((proposed.size() != 6 && !(rows == 1 && proposed.size() == 1)) || target_ids.size() != rows)
    return std::unexpected(hyle::Status::InvalidArgument("DSpark needs trained current+five proposals (or one current for a target-only step) and its exact bounded target rows"));
  for (const auto token : proposed)
    if (token < 0 || token >= limits.vocabulary)
      return std::unexpected(hyle::Status::InvalidArgument("DSpark proposal token is outside the shared vocabulary"));
  for (const auto token : target_ids)
    if (token < 0 || token >= limits.vocabulary)
      return std::unexpected(hyle::Status::InvalidArgument("DSpark target token is outside the shared vocabulary"));
  const auto eos = [&](std::int32_t token) {
    return std::find(limits.eos_tokens.begin(), limits.eos_tokens.end(), token) != limits.eos_tokens.end();
  };
  if (eos(proposed[0]))
    return std::unexpected(hyle::Status::FailedPrecondition("DSpark current pending token already ended generation"));

  // The complete trained proposal and all IDs were checked above, including
  // any unverified tail. Select only the actual bounded target rows without
  // allocating a vector or replacing the caller's original pending lease.
  std::array<std::int32_t, 6> emitted;
  const auto selected = hyle::inference::mtp::select_greedy_prefix(proposed.first(rows),
      target_ids, limits.output_quota, limits.eos_tokens, static_cast<std::int32_t>(limits.vocabulary),
      std::span(emitted).first(rows));
  return VerifiedPrefix{.accepted_drafts = static_cast<std::uint32_t>(selected.consumed_drafts),
      .consumed = static_cast<std::uint32_t>(selected.consumed), .pending_row = selected.consumed - 1,
      .emitted_count = selected.consumed, .stopped_on_eos = selected.stopped,
      .output_quota_exhausted = selected.consumed == limits.output_quota,
      .context_exhausted = selected.consumed == limits.context_rows};
}
} // namespace dspark_original_state_consumer
