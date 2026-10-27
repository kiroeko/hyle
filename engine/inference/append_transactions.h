#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/graph/batch.h"

namespace hyle::inference {

// Borrow an original episode's already reserved storage. No transaction,
// Invocation, permission, completion or retirement status is owned here.
// The caller keeps these spans stable throughout each synchronous call.
class AppendTransactions final {
 public:
  AppendTransactions(std::span<const graph::NodeId> nodes,
      std::span<graph::batch::PagedAppendPages> transactions,
      std::span<std::uint8_t> bound) noexcept
      : nodes_(nodes), transactions_(transactions), bound_(bound) {}

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
