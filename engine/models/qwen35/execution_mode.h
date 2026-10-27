#pragma once

#include <cstddef>

namespace qwen35 {

enum class GraphRole { kText, kVerify, kMtpPrefill, kMtpRound, kMtpCatchupRound,
                       kMtpSingle, kMtpCatchupSingle, kMtpCatchupPrefill };
constexpr bool has_mtp(GraphRole role) noexcept { return role >= GraphRole::kMtpPrefill; }
constexpr bool is_mtp_prefill(GraphRole role) noexcept {
  return role == GraphRole::kMtpPrefill || role == GraphRole::kMtpCatchupPrefill;
}
constexpr bool drafts_token(GraphRole role) noexcept {
  return role == GraphRole::kMtpRound || role == GraphRole::kMtpCatchupRound;
}
constexpr bool verifies_prefix(GraphRole role) noexcept {
  return role == GraphRole::kVerify || drafts_token(role);
}
constexpr std::size_t draft_lag(GraphRole role) noexcept {
  return role == GraphRole::kMtpCatchupRound || role == GraphRole::kMtpCatchupSingle ||
      role == GraphRole::kMtpCatchupPrefill ? 1u : 0u;
}
enum class TextCompletion { kEmitNext, kStateOnly };

// Completion changes whether this input episode emits a sampled next token.
// Prefill and teacher-forced single rows can commit state alone; speculative
// proposal/verification episodes keep their original complete-output contract.
constexpr bool supports_state_only(GraphRole role) noexcept {
  return role == GraphRole::kText || is_mtp_prefill(role) ||
      role == GraphRole::kMtpSingle || role == GraphRole::kMtpCatchupSingle;
}
constexpr bool supports_completion(GraphRole role, TextCompletion completion) noexcept {
  return completion == TextCompletion::kEmitNext ||
      (completion == TextCompletion::kStateOnly && supports_state_only(role));
}
constexpr bool supports_commit_completion(GraphRole role, TextCompletion completion,
                                         bool generation, bool prefill_complete) noexcept {
  return supports_completion(role, completion) &&
      (completion != TextCompletion::kStateOnly || (!generation && !prefill_complete));
}

} // namespace qwen35
