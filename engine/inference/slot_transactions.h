#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "hyle/graph/slots.h"

namespace hyle::inference {

struct SlotTransactionSpec final {
  graph::NodeId store;
  std::size_t sources = 0, destinations = 0;
  graph::slots::SlotBindingsOptions options{};
};

// One original Store cohort's cold metadata and caller-owned transactions.
// bind() transfers the actual SlotBindings to Graph atomically. Only Graph's
// real retired take returns them; this object never owns an Invocation, Queue,
// Completion, pool, publication policy or permission independent of Graph.
// A pending imported reader is supplied by the original Graph input owner;
// its local transaction has zero ready sources, exactly as slots::bind needs.
class SlotTransactions final {
 public:
  SlotTransactions() = default;
  SlotTransactions(const SlotTransactions &) = delete;
  SlotTransactions &operator=(const SlotTransactions &) = delete;
  // Original BindRequest pointers and the recovery owner remain stable even
  // while Graph owns a cohort. Consumers keep this object in its cold carrier.
  SlotTransactions(SlotTransactions &&) = delete;
  SlotTransactions &operator=(SlotTransactions &&) = delete;

  // Configure once, before filling any leases. Every array and BindRequest
  // pointer is reserved here, never resized during stage/bind/take/recovery.
  Status prepare(std::span<const SlotTransactionSpec> specs) noexcept {
    if (!banks_.empty() || remaining_)
      return Status::FailedPrecondition("slot transactions are already prepared");
    for (std::size_t i = 0; i < specs.size(); ++i) {
      if (!specs[i].store.valid() || !specs[i].destinations)
        return Status::InvalidArgument("slot transaction needs an original Store and positive writer count");
      for (std::size_t j = 0; j < i; ++j)
        if (specs[i].store == specs[j].store)
          return Status::InvalidArgument("slot transaction cohort repeats a Store");
    }
    try {
      std::vector<Bank> banks(specs.size());
      std::vector<graph::slots::BindRequest> requests(specs.size());
      for (std::size_t i = 0; i < specs.size(); ++i) {
        banks[i].store = specs[i].store;
        banks[i].options = specs[i].options;
        banks[i].sources.resize(specs[i].sources);
        banks[i].writers.resize(specs[i].destinations);
        requests[i] = {specs[i].store, &banks[i].transaction};
      }
      banks_.swap(banks); requests_.swap(requests);
    } catch (...) {
      return Status::ResourceExhausted("slot transaction cold metadata allocation failed");
    }
    return Status::Ok();
  }

  std::size_t size() const noexcept { return banks_.size(); }
  bool bound() const noexcept { return remaining_ != 0; }
  // Borrowed original staging handles, valid until this object's destruction.
  // Indices are caller-checked against size(), as for a span.
  std::span<compute::ReadLease> sources(std::size_t bank) noexcept { return banks_[bank].sources; }
  std::span<compute::WriteLease> writers(std::size_t bank) noexcept { return banks_[bank].writers; }

  // Pool storage may be published after the logical graph/counts are cold
  // prepared. Set its original layout exactly once before staging any lease;
  // a refusal never replaces an already configured geometry or pool identity.
  Status set_layout(std::size_t bank, compute::LeaseSlotLayout layout) noexcept {
    if (bank >= banks_.size()) return Status::OutOfRange("slot transaction bank is out of range");
    if (!layout.valid()) return Status::InvalidArgument("slot transaction needs a valid original layout");
    auto &item = banks_[bank];
    if (remaining_ || item.layout.valid())
      return Status::FailedPrecondition("slot transaction layout is already fixed or bound");
    item.layout = std::move(layout);
    return Status::Ok();
  }

  // Layout, options and counts are immutable cold facts. create/refill checks
  // every actual lease before consuming any writer. A refusal preserves
  // caller sources/writers. Success leaves their only extra
  // ready pins inside the original transaction, not a second staging mirror.
  Status stage(std::size_t bank) noexcept {
    if (bank >= banks_.size()) return Status::OutOfRange("slot transaction bank is out of range");
    if (remaining_) return Status::FailedPrecondition("original slot transactions have not all returned");
    auto &item = banks_[bank];
    if (!item.layout.valid()) return Status::FailedPrecondition("original slot transaction layout is not configured");
    if (item.transaction.valid()) {
      HYLE_RETURN_IF_ERROR(item.transaction.refill(item.sources, item.writers));
    } else {
      HYLE_ASSIGN_OR_RETURN(item.transaction, graph::slots::SlotBindings::create(
          item.layout, item.sources, item.writers, item.options));
    }
    for (auto &source : item.sources) source = {};
    return Status::Ok();
  }

  // Original slots::bind performs all rejecting checks/rollback before its
  // final non-failing cohort move. Failure never changes this return count.
  Status bind(graph::Invocation &run, const graph::Graph &logical) noexcept {
    if (remaining_) return Status::FailedPrecondition("original slot transactions are already bound");
    HYLE_RETURN_IF_ERROR(graph::slots::bind(run, logical, requests_));
    remaining_ = banks_.size();
    return Status::Ok();
  }

  // Returned writers remain unpublished. The owner must observe its original
  // whole-Graph status/append attestation before publishing selected rows.
  // No native work or observation is fabricated by a successful take.
  Result<std::span<compute::WriteLease>> take(graph::Invocation &run, std::size_t bank) noexcept {
    if (bank >= banks_.size()) return std::unexpected(Status::OutOfRange("slot transaction bank is out of range"));
    auto &item = banks_[bank];
    if (!remaining_ || item.transaction.valid())
      return std::unexpected(Status::FailedPrecondition("slot transaction is not owned by this Graph"));
    HYLE_ASSIGN_OR_RETURN(item.transaction, graph::slots::take(run, item.store));
    --remaining_;
    return item.transaction.destinations();
  }

  // Never release a Graph-owned transaction, even after an observed error.
  // A prior partial take is still caller-owned and may safely clear its pins.
  Status clear(std::size_t bank) noexcept {
    if (bank >= banks_.size()) return Status::OutOfRange("slot transaction bank is out of range");
    auto &item = banks_[bank];
    if (remaining_ && !item.transaction.valid())
      return Status::Unavailable("original Graph still owns this slot transaction");
    item.transaction.clear();
    for (auto &source : item.sources) source = {};
    for (auto &writer : item.writers) writer = {};
    return Status::Ok();
  }

  // Original retirement gate only, no wait or publication. Earlier returned
  // banks stay remembered if a later take is unavailable or fails; retries do
  // not retake or erase them. Graph's original failure status remains owned by
  // its sole Completion/Step, separately from this cleanup observation.
  Result<bool> recover(graph::Invocation &run) noexcept {
    for (std::size_t bank = 0; bank < banks_.size(); ++bank) {
      if (remaining_ && !banks_[bank].transaction.valid()) {
        auto returned = take(run, bank);
        if (!returned) {
          if (returned.error().IsUnavailable()) return false;
          return std::unexpected(returned.error());
        }
      }
      HYLE_RETURN_IF_ERROR(clear(bank));
    }
    return true;
  }

 private:
  struct Bank final {
    graph::NodeId store;
    compute::LeaseSlotLayout layout;
    graph::slots::SlotBindingsOptions options{};
    graph::slots::SlotBindings transaction;
    std::vector<compute::ReadLease> sources;
    std::vector<compute::WriteLease> writers;
  };
  std::vector<Bank> banks_;
  std::vector<graph::slots::BindRequest> requests_;
  // Replaces the consumer's old bound flag; one decrement per real return.
  // SlotBindings.valid already remembers partial returns, without per-bank
  // booleans, a second permission ledger or a scan after every take.
  std::size_t remaining_ = 0;
};

} // namespace hyle::inference
