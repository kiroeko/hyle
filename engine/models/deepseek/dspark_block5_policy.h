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
#include "../../inference/mtp_acceptance.h"

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
hyle::Result<std::size_t> block5_verification_capacity(Block5Limits limits);

namespace block5_detail {
hyle::Result<std::size_t> confidence_rows(Block5Limits limits,
    std::size_t rows, std::span<const float> confidence, std::size_t expected);
} // namespace block5_detail

hyle::Result<std::size_t> block5_verification_rows(Block5Limits limits,
    std::span<const float> confidence = {});

// A sampled Markov prefix exposes only its prepared candidates. Each confidence
// uses the preceding sampled IDs, before this candidate's own draw. Truncating
// at its first low row may therefore discard that unpublished draw and use the
// same address for the target final sample, as in Qwen recursive proposals.
hyle::Result<std::size_t> block5_sampled_verification_rows(Block5Limits limits,
    std::span<const float> conditional_confidence = {});

// proposed is the original [current, proposal1, ..., proposal5], even when
// only a tail of 1..6 target rows is verified; an ordinary one-row target step
// can supply just [current] without executing the draft. A matched EOS or the last quota
// output is emitted and kept pending, not consumed merely to obtain a bonus.
// Bind target row pending_row's original I32 result lease into the selected
// after-current/after-proposal checkpoint. The helper uploads no host token.
hyle::Result<VerifiedPrefix> decide_block5(std::span<const std::int32_t> proposed,
    std::span<const std::int32_t> target_ids, Block5Limits limits, std::span<const float> confidence = {});
} // namespace dspark_original_state_consumer
