#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/graph/batch.h"

namespace hyle::inference {

struct AppendExtent final {
  std::size_t after_rows = 0, prefix_pages = 0, page_count = 0, dirty_count = 0;
  std::uint32_t tail_rows = 0, tail_append_rows = 0;
};

// Physical page and original member index. The caller retains the sorted,
// unique page table throughout return_tails; withdrawn members stay present.
using AppendTailOwner = std::pair<std::uint32_t, std::size_t>;
struct AppendTailDestination final {
  std::uint32_t initialized_rows = 0;
  compute::PagedPageAppender *tail = nullptr;
};

// Actual logical input rows only; physical query padding is not an append.
// Division/remainder avoids an overflowing ceil numerator at SIZE_MAX.
inline Result<AppendExtent> append_extent(std::size_t before_rows,
    std::size_t append_rows, std::size_t capacity_rows, std::uint32_t page_rows) noexcept {
  if (!page_rows || !append_rows)
    return std::unexpected(Status::InvalidArgument("append extent requires nonempty rows and a positive page size"));
  if (before_rows > capacity_rows || append_rows > capacity_rows - before_rows)
    return std::unexpected(Status::OutOfRange("append extent exceeds its original logical capacity"));
  AppendExtent result;
  result.after_rows = before_rows + append_rows;
  result.prefix_pages = before_rows / page_rows;
  result.page_count = result.after_rows / page_rows + (result.after_rows % page_rows != 0);
  result.dirty_count = result.page_count - result.prefix_pages;
  result.tail_rows = static_cast<std::uint32_t>(before_rows % page_rows);
  if (result.tail_rows) {
    const auto available = page_rows - result.tail_rows;
    result.tail_append_rows = append_rows < available ? static_cast<std::uint32_t>(append_rows) : available;
  }
  return result;
}

// A read-only sizing hint for the caller's original dirty/tail storage. The
// caller still supplies its epoch/relay/private-work eligibility. Matching
// metadata is not append permission: reserve_tail calls the actual SDK gate.
inline bool tail_compatible(std::span<const compute::PagedPageAppender> tails,
    std::uint32_t last_page, const AppendExtent &extent, bool allowed) noexcept {
  if (!allowed || !extent.tail_rows || !extent.tail_append_rows) return false;
  for (const auto &tail : tails)
    if (!tail.valid() || tail.reserved_rows() || tail.page() != last_page ||
        tail.initialized_rows() != extent.tail_rows) return false;
  return true;
}

// Borrow the original appender cohort after its read-only compatibility check
// and caller-owned capacity preparation. No owner, ticket or allocation is
// created here. An empty bank cohort keeps the original vacuous loop result.
// A late Unavailable permits COW only after earlier reservations roll back.
// Other failures retain their exact Status and remaining facts in the actual
// appenders; a rollback failure never becomes a successful COW decision.
inline Result<bool> reserve_tail(std::span<compute::PagedPageAppender> tails,
    const AppendExtent &extent) noexcept {
  if (!extent.tail_append_rows)
    return std::unexpected(Status::InvalidArgument("append extent has no original partial tail to reserve"));
  for (std::size_t reserved = 0; reserved != tails.size(); ++reserved) {
    const auto status = tails[reserved].reserve_rows(extent.tail_append_rows);
    if (status.ok()) continue;
    for (std::size_t i = 0; i != reserved; ++i)
      HYLE_RETURN_IF_ERROR(tails[i].rollback());
    if (status.IsUnavailable()) return false;
    return std::unexpected(status);
  }
  return true;
}

// Prepare one sequence's actual page-reference/dirty-bank cohort. PageRef is
// the original inventory's policy reference, not a new native capability.
// Acquire synchronously obtains one original page, puts its real writers in
// the already reserved dirty banks, and returns that page's original ref.
// The caller reserves one tail destination per bank before this call; no
// callback or borrowed container survives it. Source-epoch/relay eligibility
// remains the caller's fact; an empty source-tail span changes no owner.
//
// A late refusal keeps prior references, writers and moved tails in these same
// candidate containers for original episode recovery. It does not roll back a
// native acquisition or infer retirement. Only a real tail Unavailable, after
// reserve_tail's original rollback, permits the existing COW acquisition path.
template<class PageRef, class DirtyBanks, class TailBanks, class Acquire>
Result<bool> prepare_append_pages(const AppendExtent &extent,
    std::span<const PageRef> old_pages, std::vector<PageRef> &next_pages,
    std::span<compute::PagedPageAppender> reusable_tails, bool allow_tail_reuse,
    DirtyBanks &&dirty_banks, TailBanks &&returned_tails, Acquire &&acquire_page) {
  static_assert(std::is_nothrow_copy_constructible_v<PageRef> &&
      std::is_nothrow_move_constructible_v<PageRef>,
      "original page references must copy and move without throwing");
  static_assert(std::is_same_v<std::invoke_result_t<Acquire &>, Result<PageRef>>,
      "page acquisition must return the original inventory reference");
  if (!next_pages.empty() || !extent.dirty_count || extent.page_count < extent.prefix_pages ||
      extent.dirty_count != extent.page_count - extent.prefix_pages ||
      old_pages.size() < extent.prefix_pages + (extent.tail_rows ? 1u : 0u) ||
      dirty_banks.size() != returned_tails.size() ||
      (!reusable_tails.empty() && reusable_tails.size() != dirty_banks.size()))
    return std::unexpected(Status::InvalidArgument("append pages differ from their original sequence and bank cohort"));
  bool reuse = tail_compatible(reusable_tails, old_pages.empty() ? 0 : old_pages.back().page(),
      extent, allow_tail_reuse && !old_pages.empty());
  next_pages.reserve(extent.page_count);
  const auto reserve_dirty = [&](std::size_t count) -> Status {
    for (auto &bank : dirty_banks) {
      if (count > (std::numeric_limits<std::size_t>::max)() - bank.size())
        return Status::OutOfRange("append dirty-bank capacity overflows");
      bank.reserve(bank.size() + count);
    }
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(reserve_dirty(extent.dirty_count - (reuse ? 1u : 0u)));
  if (reuse) {
    HYLE_ASSIGN_OR_RETURN(const bool reserved, reserve_tail(reusable_tails, extent));
    if (!reserved) {
      reuse = false;
      HYLE_RETURN_IF_ERROR(reserve_dirty(extent.dirty_count));
    }
  }
  for (std::size_t page = 0; page < extent.prefix_pages; ++page)
    next_pages.push_back(old_pages[page]);
  if (reuse) {
    next_pages.push_back(old_pages.back());
    for (std::size_t bank = 0; bank < reusable_tails.size(); ++bank)
      returned_tails[bank].push_back(std::move(reusable_tails[bank]));
  } else for (auto &tail : reusable_tails) tail = {};
  for (std::size_t page = reuse ? 1u : 0u; page < extent.dirty_count; ++page) {
    HYLE_ASSIGN_OR_RETURN(auto reference, acquire_page());
    next_pages.push_back(std::move(reference));
  }
  return reuse;
}

// Borrow an original episode's already reserved storage. No transaction,
// Invocation, permission, completion or retirement status is owned here.
// The caller keeps these spans stable throughout each synchronous call.
class AppendTransactions final {
 public:
  AppendTransactions(std::span<const graph::NodeId> nodes,
      std::span<graph::batch::PagedAppendPages> transactions,
      std::span<std::uint8_t> bound) noexcept
      : nodes_(nodes), transactions_(transactions), bound_(bound) {}

  std::size_t size() const noexcept { return transactions_.size(); }

  // Same extent preconditions as std::span::subspan on the original arrays.
  AppendTransactions subspan(std::size_t first, std::size_t count) const noexcept {
    return {nodes_.subspan(first, count), transactions_.subspan(first, count), bound_.subspan(first, count)};
  }

  // Each original bind checks its actual reader/effect authority. A later
  // refusal leaves earlier successes marked for retired recovery; the failing
  // transaction stays caller-owned. This is not an atomic bind or rollback.
  // An invalid cache input selects the original external-write-only overload.
  Status bind(graph::Invocation &run, const graph::Graph &logical,
              std::span<const graph::ValueId> cache_inputs) const noexcept {
    HYLE_RETURN_IF_ERROR(validate());
    if (cache_inputs.size() != transactions_.size())
      return Status::InvalidArgument("append cache inputs differ from the original cohort extent");
    for (const auto value : bound_)
      if (value) return Status::FailedPrecondition("original append cohort is already partly bound");
    for (std::size_t i = 0; i < transactions_.size(); ++i) {
      const auto status = cache_inputs[i].valid()
          ? graph::batch::bind_append(run, logical, nodes_[i], cache_inputs[i], std::move(transactions_[i]))
          : graph::batch::bind_append(run, logical, nodes_[i], std::move(transactions_[i]));
      HYLE_RETURN_IF_ERROR(status);
      bound_[i] = 1;
    }
    return Status::Ok();
  }

  // Atomic original Graph withdrawal, with no wait and no publication. A
  // successful return attests actual retirement only; the caller still owns
  // the whole outcome and must prove success before publishing any payload.
  Status take(graph::Invocation &run) const noexcept {
    HYLE_RETURN_IF_ERROR(validate());
    for (const auto value : bound_)
      if (!value) return Status::FailedPrecondition("original append cohort is not fully bound");
    HYLE_RETURN_IF_ERROR(graph::batch::take_appends(run, nodes_, transactions_));
    for (auto &value : bound_) value = 0;
    return Status::Ok();
  }

  // Return only capabilities from the original caller-owned Append cohort.
  // Its whole Graph success and actual take remain the caller's witnesses;
  // this operation neither observes native work nor publishes a frame/head.
  // Resolve is a synchronous borrow: bank is relative to this subspan and
  // member comes from the original page table. A null destination discards
  // that tail. Earlier moves/discards survive a later refusal; repair the
  // remaining facts and retry without retaking or recreating the cohort.
  template<class Resolve>
  Status return_tails(std::span<const AppendTailOwner> owners, Resolve &&resolve) const noexcept {
    static_assert(std::is_nothrow_invocable_v<Resolve &, std::size_t, std::size_t> &&
        std::is_same_v<std::invoke_result_t<Resolve &, std::size_t, std::size_t>, AppendTailDestination>,
        "tail destinations must resolve directly without throwing");
    HYLE_RETURN_IF_ERROR(validate());
    for (std::size_t bank = 0; bank < transactions_.size(); ++bank)
      if (bound_[bank] || !transactions_[bank].valid())
        return Status::FailedPrecondition("original append cohort has not all returned to its caller");
    for (std::size_t at = 1; at < owners.size(); ++at)
      if (owners[at - 1].first >= owners[at].first)
        return Status::InvalidArgument("append tail owners must have strictly increasing unique pages");
    for (std::size_t bank = 0; bank < transactions_.size(); ++bank) {
      auto &transaction = transactions_[bank];
      for (std::size_t at = 0; at < transaction.appender_count(); ++at) {
        HYLE_ASSIGN_OR_RETURN(auto &tail, transaction.appender(at));
        if (!tail.valid()) continue;
        auto owner = owners.begin();
        if (owners.size() != 1)
          owner = std::lower_bound(owners.begin(), owners.end(), tail.page(),
              [](const AppendTailOwner &candidate, std::uint32_t page) noexcept { return candidate.first < page; });
        if (owner == owners.end() || owner->first != tail.page())
          return Status::FailedPrecondition("returned append tail has no original final-page owner");
        const auto destination = resolve(bank, owner->second);
        if (!destination.tail) { tail = {}; continue; }
        if (tail.reserved_rows() || destination.tail->valid() || tail.initialized_rows() != destination.initialized_rows)
          return Status::FailedPrecondition("returned append tail differs from its empty destination or committed frontier");
        *destination.tail = std::move(tail);
      }
    }
    return Status::Ok();
  }

  // Partial bind/take recovery uses the same original Graph gate. An earlier
  // returned bank stays remembered if a later take is unavailable or fails.
  // Never replace the episode's original mathematical/retirement status here.
  Result<bool> recover(graph::Invocation &run) const noexcept {
    HYLE_RETURN_IF_ERROR(validate());
    for (std::size_t i = 0; i < transactions_.size(); ++i) if (bound_[i]) {
      if (transactions_[i].valid())
        return std::unexpected(Status::FailedPrecondition("bound append still has a caller transaction"));
      auto returned = graph::batch::take_append(run, nodes_[i]);
      if (!returned) {
        if (returned.error().IsUnavailable()) return false;
        return std::unexpected(returned.error());
      }
      transactions_[i] = std::move(*returned);
      bound_[i] = 0;
    }
    return true;
  }

  static Status rollback_tail(compute::PagedPageAppender &tail) noexcept {
    if (!tail.valid() || !tail.reserved_rows()) return Status::Ok();
    return tail.rollback();
  }

  static Status discard_page(compute::PagedPageWriter &page) noexcept {
    if (!page.valid()) return Status::Ok(); // Already successfully published/discarded.
    page.lease().reset();
    HYLE_RETURN_IF_ERROR(page.discard()); // Unavailable preserves the reservation.
    page = {};
    return Status::Ok();
  }

  // Only caller-owned original transactions may be discarded. Successful
  // rollback/discard is remembered by each actual tail/page; no cleanup ledger
  // is added. Unavailable retains the remaining facts for the next attempt.
  // Keep transaction metadata until the caller also recovers its staging pages.
  Result<bool> discard() const noexcept {
    HYLE_RETURN_IF_ERROR(validate());
    for (const auto value : bound_)
      if (value) return false;
    for (auto &transaction : transactions_) if (transaction.valid()) {
      for (std::size_t i = 0; i < transaction.appender_count(); ++i) {
        HYLE_ASSIGN_OR_RETURN(auto &tail, transaction.appender(i));
        const auto status = rollback_tail(tail);
        if (status.IsUnavailable()) return false;
        HYLE_RETURN_IF_ERROR(status);
      }
      for (std::size_t i = 0; i < transaction.page_count(); ++i) {
        HYLE_ASSIGN_OR_RETURN(auto &page, transaction.page(i));
        const auto status = discard_page(page);
        if (status.IsUnavailable()) return false;
        HYLE_RETURN_IF_ERROR(status);
      }
    }
    return true;
  }

 private:
  Status validate() const noexcept {
    if (nodes_.size() != transactions_.size() || bound_.size() != transactions_.size())
      return Status::InvalidArgument("append transaction spans differ from the original cohort extent");
    return Status::Ok();
  }

  std::span<const graph::NodeId> nodes_;
  std::span<graph::batch::PagedAppendPages> transactions_;
  std::span<std::uint8_t> bound_;
};

} // namespace hyle::inference
