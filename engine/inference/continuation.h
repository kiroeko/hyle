#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>

#include "hyle/container/small_vector.h"
#include "request_loop.h"

namespace hyle::inference {

struct ContinuationMember final {
  RequestKey key;
  std::size_t turn = 0;
  friend bool operator==(const ContinuationMember &, const ContinuationMember &) = default;
};

// A request owner's ordered reuse cohort and drain intent only. It neither
// owns pending calls/native work nor decides that either has retired. Model
// Source/boundary checks remain necessary after this identity check.
class ContinuationCohort final {
 public:
  enum class Selection { kSame, kChanged, kRejected, kPendingConflict };
  enum class Release { kKeep, kReady, kDrain };

  void prepare(std::size_t capacity) {
    if (!members_.try_reserve(capacity)) throw std::bad_alloc{};
  }
  std::span<const ContinuationMember> members() const noexcept { return {members_.data(), members_.size()}; }
  bool drain_requested() const noexcept { return drain_requested_; }

  // A ready-only reset releases reuse hints. It deliberately does not clear
  // drain intent or any caller-owned pending metadata after an unretired error.
  void reset_ready() noexcept { members_.clear(); }
  // Acknowledge a handled control intent. This conveys no retirement or credit
  // release permission; only the original owner may clear pending metadata.
  void clear_drain() noexcept { drain_requested_ = false; }

  template<class Calls, class Identity>
  bool matches(const Calls &calls, Identity identity) const noexcept(noexcept(identity(calls[0]))) {
    if (calls.size() != members_.size()) return false;
    for (std::size_t j = 0; j < calls.size(); ++j)
      if (members_[j] != identity(calls[j])) return false;
    return true;
  }
  template<class Calls, class Identity>
  Selection select(const Calls &calls, Identity identity, bool pending) noexcept(noexcept(identity(calls[0]))) {
    if (pending) return matches(calls, identity) ? Selection::kSame : Selection::kPendingConflict;
    if (calls.empty() || calls.size() > members_.capacity()) {
      reset_ready(); return Selection::kRejected;
    }
    if (matches(calls, identity)) return Selection::kSame;
    reset_ready();
    for (const auto &call : calls) members_.push_back(identity(call));
    return Selection::kChanged;
  }
  Release admission(ContinuationMember selected, bool pending) noexcept {
    if (std::ranges::find(members_, selected) != members_.end()) return Release::kKeep;
    return release_or_drain(pending);
  }
  Release release(RequestKey selected, bool pending) noexcept {
    if (std::ranges::none_of(members_, [&](const auto &member) { return member.key == selected; }))
      return Release::kKeep;
    return release_or_drain(pending);
  }
  bool draining(bool requested, std::span<const RequestCreditGrant> grants) const noexcept {
    return requested || drain_requested_ ||
        std::ranges::any_of(grants, [](const auto &grant) { return grant.cancelled; });
  }

 private:
  Release release_or_drain(bool pending) noexcept {
    if (pending) { drain_requested_ = true; return Release::kDrain; }
    return Release::kReady;
  }
  hyle::container::SmallVector<ContinuationMember, 2> members_;
  bool drain_requested_ = false;
};

// Numeric bound on a borrowed grant, not a reservation or another quota
// ledger. The original model ledger uses this once alongside its exact pending
// record/boundary checks. Cancel/drain/turn policy remains with the request owner.
inline std::optional<std::size_t> continuation_quota(const RequestCreditGrant &grant,
    std::size_t maximum_outputs, std::size_t remaining_outputs) noexcept {
  if (grant.cancelled || !maximum_outputs || grant.reserved != maximum_outputs ||
      grant.reserved > remaining_outputs || grant.additional > remaining_outputs - grant.reserved)
    return std::nullopt;
  return grant.reserved + grant.additional;
}

// Actual post-publication request facts. No model state, native permission,
// token/RNG record or duplicated ownership is stored in this policy.
struct ContinuationProgress final {
  bool published = false, completed = false, stopping = false, failed = false;
  std::size_t remaining_outputs = 0;
};
enum class ContinuationRetention { kKeep, kReleaseReady, kResetReady, kRetirePending };

template<class Calls, class Progress>
ContinuationRetention continuation_retention(const Calls &calls, Progress progress,
    bool pending, bool native_cancelled, bool unavailable, bool draining) {
  bool reset = native_cancelled || unavailable, exhausted = false;
  for (const auto &call : calls) {
    const auto member = progress(call);
    reset |= !member.published || member.completed || member.stopping || member.failed;
    exhausted |= !member.remaining_outputs;
  }
  if (pending) return reset || exhausted ? ContinuationRetention::kRetirePending : ContinuationRetention::kKeep;
  if (reset) return ContinuationRetention::kResetReady;
  return draining ? ContinuationRetention::kReleaseReady : ContinuationRetention::kKeep;
}

} // namespace hyle::inference
