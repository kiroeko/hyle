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

} // namespace qwen35
