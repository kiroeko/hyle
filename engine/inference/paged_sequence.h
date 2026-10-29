#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/paged_frame.h"
#include "hyle/container/small_vector.h"
#include "hyle/core/assert.h"

namespace hyle::inference {

struct PagedPageInventory;

// Policy identity only. The original inventory outlives every borrowed ref;
// neither a copied ref nor a free ID grants native page permissions.
class PagedPageRef final {
 public:
  PagedPageRef() noexcept = default;
  PagedPageRef(const PagedPageRef &) noexcept;
  PagedPageRef &operator=(const PagedPageRef &) noexcept;
  PagedPageRef(PagedPageRef &&other) noexcept
      : owner_(std::exchange(other.owner_, nullptr)), page_(other.page_) {}
  PagedPageRef &operator=(PagedPageRef &&other) noexcept {
    if (this != &other) { PagedPageRef next(std::move(other)); swap(next); }
    return *this;
  }
  ~PagedPageRef();
  std::uint32_t page() const noexcept { return page_; }

 private:
  friend struct PagedPageInventory;
  PagedPageRef(PagedPageInventory *owner, std::uint32_t page) noexcept
      : owner_(owner), page_(page) {}
  void swap(PagedPageRef &other) noexcept {
    std::swap(owner_, other.owner_); std::swap(page_, other.page_);
  }
  PagedPageInventory *owner_ = nullptr;
  std::uint32_t page_ = 0;
};

struct PagedPageReservation final {
  // Destruction and replacement release native writers before policy reuse.
  PagedPageRef reference;
  container::SmallVector<compute::PagedPageWriter, 32> writers;
  PagedPageReservation() = default;
  PagedPageReservation(const PagedPageReservation &) = delete;
  PagedPageReservation &operator=(const PagedPageReservation &) = delete;
  PagedPageReservation(PagedPageReservation &&) noexcept = default;
  PagedPageReservation &operator=(PagedPageReservation &&other) noexcept {
    if (this != &other) {
      writers.clear(); reference = {};
      reference = std::move(other.reference); writers = std::move(other.writers);
    }
    return *this;
  }
};

// One policy ID spans any number of independent original arena banks. Bank
// order is the caller's physical schema, with no K/V or model interpretation.
struct PagedPageInventory final {
  std::vector<compute::PagedArena> banks;
  std::vector<std::uint32_t> owners, free_ids;

  PagedPageInventory() = default;
  PagedPageInventory(const PagedPageInventory &) = delete;
  PagedPageInventory &operator=(const PagedPageInventory &) = delete;
  PagedPageInventory(PagedPageInventory &&other) noexcept {
    HYLE_FATAL_ASSERT(!other.has_live_refs());
    banks = std::move(other.banks); owners = std::move(other.owners);
    free_ids = std::move(other.free_ids); bank_count_ = std::exchange(other.bank_count_, 0);
  }
  PagedPageInventory &operator=(PagedPageInventory &&other) noexcept {
    if (this != &other) {
      HYLE_FATAL_ASSERT(!has_live_refs() && !other.has_live_refs());
      banks = std::move(other.banks); owners = std::move(other.owners);
      free_ids = std::move(other.free_ids); bank_count_ = std::exchange(other.bank_count_, 0);
    }
    return *this;
  }
  ~PagedPageInventory() { HYLE_FATAL_ASSERT(!has_live_refs()); }

  // Cold replacement only. All allocation precedes changing the old banks or
  // policy table; existing arena handles survive when their bank is retained.
  Status prepare(std::uint32_t pages, std::size_t bank_count) noexcept {
    if (has_live_refs())
      return Status::FailedPrecondition("paged inventory still has live policy references");
    if (bank_count > banks.max_size() || pages > owners.max_size() || pages > free_ids.max_size())
      return Status::OutOfRange("paged inventory metadata extent overflows");
    try {
      std::vector<compute::PagedArena> next_banks(bank_count);
      std::vector<std::uint32_t> next_owners(pages, 0), next_free;
      next_free.reserve(pages);
      for (auto page = pages; page != 0; --page) next_free.push_back(page - 1);
      for (std::size_t bank = 0; bank < (std::min)(banks.size(), bank_count); ++bank)
        next_banks[bank] = std::move(banks[bank]);
      banks.swap(next_banks); owners.swap(next_owners); free_ids.swap(next_free);
      bank_count_ = bank_count;
      return Status::Ok();
    } catch (const std::bad_alloc &) {
      return Status::ResourceExhausted("paged inventory metadata allocation failed");
    }
  }

  Result<PagedPageReservation> acquire() noexcept {
    if (banks.size() != bank_count_)
      return std::unexpected(Status::InvalidArgument("paged inventory bank count is invalid"));
    // Equivalent to moving each unavailable last ID to the front, but with
    // one final rotation instead of quadratic metadata movement. A full scan
    // of unavailable native pages leaves the original policy order intact.
    for (std::size_t at = free_ids.size(); at != 0;) {
      const auto index = --at;
      const auto page = free_ids[index];
      HYLE_FATAL_ASSERT(page < owners.size() && !owners[page]);
      PagedPageReservation result;
      if (!result.writers.try_resize(bank_count_)) {
        rotate_attempt(index);
        return std::unexpected(Status::ResourceExhausted("paged writer metadata allocation failed"));
      }
      Status status;
      for (std::size_t bank = 0; bank < bank_count_; ++bank) {
        auto writer = banks[bank].acquire(page);
        if (!writer) { status = writer.error(); break; }
        result.writers[bank] = std::move(*writer);
      }
      if (status.ok()) {
        rotate_attempt(index); free_ids.pop_back();
        owners[page] = 1; result.reference = PagedPageRef(this, page);
        return result;
      }
      result.writers.clear(); // Real native permissions retire before ID reuse.
      if (status.IsUnavailable()) continue;
      rotate_attempt(index);
      return std::unexpected(std::move(status));
    }
    return std::unexpected(Status::ResourceExhausted(
        "paged inventory has no available original page; release retained state or enlarge its capacity"));
  }

  // Move the same acquired writers into a prepared episode cohort. A failure
  // before acquisition changes nothing; native acquisition retains its Status.
  Result<PagedPageRef> acquire_into(std::span<std::vector<compute::PagedPageWriter>> dirty) noexcept {
    if (banks.size() != bank_count_ || dirty.size() != bank_count_)
      return std::unexpected(Status::InvalidArgument("paged dirty cohort differs from its inventory banks"));
    for (const auto &bank : dirty)
      if (bank.size() == bank.capacity())
        return std::unexpected(Status::FailedPrecondition("paged dirty bank has no prepared writer capacity"));
    HYLE_ASSIGN_OR_RETURN(auto page, acquire());
    for (std::size_t bank = 0; bank < dirty.size(); ++bank)
      dirty[bank].push_back(std::move(page.writers[bank]));
    return std::move(page.reference);
  }

  void retain(std::uint32_t page) noexcept {
    HYLE_FATAL_ASSERT(page < owners.size() && owners[page] && owners[page] != UINT32_MAX);
    ++owners[page];
  }
  void release(std::uint32_t page) noexcept {
    HYLE_FATAL_ASSERT(page < owners.size() && owners[page]);
    if (--owners[page] == 0) {
      HYLE_FATAL_ASSERT(free_ids.size() < free_ids.capacity());
      free_ids.push_back(page);
    }
  }

 private:
  bool has_live_refs() const noexcept {
    return std::ranges::any_of(owners, [](auto count) noexcept { return count != 0; });
  }
  void rotate_attempt(std::size_t index) noexcept {
    std::rotate(free_ids.begin(), free_ids.begin() + static_cast<std::ptrdiff_t>(index + 1), free_ids.end());
  }
  std::size_t bank_count_ = 0;
};

inline PagedPageRef::PagedPageRef(const PagedPageRef &other) noexcept
    : owner_(other.owner_), page_(other.page_) {
  if (owner_) owner_->retain(page_);
}
inline PagedPageRef &PagedPageRef::operator=(const PagedPageRef &other) noexcept {
  if (this != &other) { PagedPageRef next(other); swap(next); }
  return *this;
}
inline PagedPageRef::~PagedPageRef() { if (owner_) owner_->release(page_); }

// Only page-policy refs and original frames. Layouts/deltas own their actual
// row boundaries; model length, cache position and frontier stay with models.
struct PagedSequenceState final {
  std::vector<PagedPageRef> pages;
  container::SmallVector<compute::PagedArenaFrame, 32> frames;

  PagedSequenceState() = default;
  PagedSequenceState(const PagedSequenceState &) = delete;
  PagedSequenceState &operator=(const PagedSequenceState &) = delete;
  PagedSequenceState(PagedSequenceState &&) noexcept = default;
  PagedSequenceState &operator=(PagedSequenceState &&other) noexcept {
    if (this != &other) {
      clear(); pages = std::move(other.pages); frames = std::move(other.frames);
    }
    return *this;
  }
  void clear() noexcept { frames.clear(); pages.clear(); }
  Status can_seal() const noexcept {
    for (const auto &frame : frames) HYLE_RETURN_IF_ERROR(frame.can_seal());
    return Status::Ok();
  }
  // The serialized owner first preflights its whole publication collection.
  void seal() noexcept {
    for (auto &frame : frames) {
      const auto status = frame.seal(); HYLE_FATAL_ASSERT(status.ok());
    }
  }
  Result<PagedSequenceState> fork() const noexcept { return copy_frames(false); }
  Result<PagedSequenceState> frozen_alias() const noexcept { return copy_frames(true); }

 private:
  Result<PagedSequenceState> copy_frames(bool frozen) const noexcept {
    PagedSequenceState result;
    try { result.pages = pages; }
    catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("paged sequence reference metadata allocation failed"));
    }
    if (!result.frames.try_resize(frames.size()))
      return std::unexpected(Status::ResourceExhausted("paged sequence frame metadata allocation failed"));
    for (std::size_t bank = 0; bank < frames.size(); ++bank) {
      if (frozen) {
        HYLE_ASSIGN_OR_RETURN(auto snapshot, frames[bank].snapshot());
        HYLE_ASSIGN_OR_RETURN(result.frames[bank], compute::PagedArenaFrame::from_frozen(snapshot));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto view, frames[bank].view());
        HYLE_ASSIGN_OR_RETURN(result.frames[bank], compute::PagedArenaFrame::fork(view));
      }
    }
    return result;
  }
};

// Synchronous borrow of one original candidate's frame cohort and tickets.
// The episode retains append/native recovery. This edit has no executor,
// completion, state version, callback or independent publication authority.
class PagedSequenceEdit final {
 public:
  PagedSequenceEdit(const PagedSequenceState &base, PagedSequenceState &candidate,
      std::span<compute::PagedArenaCommit> commits = {}) noexcept
      : base_(base), candidate_(candidate), commits_(commits) {}

  const compute::PagedArenaFrame &selected(std::size_t bank) const noexcept {
    HYLE_FATAL_ASSERT(bank < base_.frames.size() && bank < candidate_.frames.size());
    return candidate_.frames[bank].valid() ? candidate_.frames[bank] : base_.frames[bank];
  }
  Result<compute::PagedArenaFootprint> source(std::size_t bank) noexcept {
    HYLE_RETURN_IF_ERROR(validate_bank(bank));
    auto &candidate = candidate_.frames[bank];
    if (!candidate.valid() && base_.frames[bank].sealed()) {
      HYLE_ASSIGN_OR_RETURN(auto original, base_.frames[bank].view());
      HYLE_ASSIGN_OR_RETURN(candidate, compute::PagedArenaFrame::fork(original));
    }
    return selected(bank).view();
  }
  Status prepare_commit(std::size_t bank, compute::PagedArenaDelta &&delta) noexcept {
    HYLE_RETURN_IF_ERROR(validate_bank(bank));
    if (commits_.size() != base_.frames.size())
      return Status::InvalidArgument("paged commit tickets differ from their original bank cohort");
    if (commits_[bank].valid())
      return Status::FailedPrecondition("paged bank already has its original prepared commit");
    HYLE_ASSIGN_OR_RETURN(commits_[bank], selected(bank).prepare_commit(std::move(delta)));
    return Status::Ok();
  }
  Status preflight() const noexcept {
    if (&base_ == &candidate_ || candidate_.frames.size() != base_.frames.size() ||
        commits_.size() != base_.frames.size())
      return Status::InvalidArgument("paged publication requires its complete original bank cohort");
    for (std::size_t bank = 0; bank < commits_.size(); ++bank)
      HYLE_RETURN_IF_ERROR(selected(bank).validate_commit(commits_[bank]));
    return Status::Ok();
  }
  // Call immediately after successful whole-cohort preflight, with no frame
  // control interleaved. Moving a wrapper preserves the original ticket base.
  void publish(PagedSequenceState *previous = nullptr) noexcept {
    HYLE_FATAL_ASSERT(candidate_.frames.size() == base_.frames.size() && commits_.size() == base_.frames.size());
    for (std::size_t bank = 0; bank < commits_.size(); ++bank) {
      auto &destination = candidate_.frames[bank];
      if (!destination.valid()) {
        HYLE_FATAL_ASSERT(previous == &base_ && previous->frames[bank].valid() && !previous->frames[bank].sealed());
        destination = std::move(previous->frames[bank]);
      }
      const auto status = destination.commit(std::move(commits_[bank]));
      HYLE_FATAL_ASSERT(status.ok());
    }
  }

 private:
  Status validate_bank(std::size_t bank) const noexcept {
    if (&base_ == &candidate_ || candidate_.frames.size() != base_.frames.size() || bank >= base_.frames.size())
      return Status::InvalidArgument("paged edit bank differs from its original candidate cohort");
    return Status::Ok();
  }
  const PagedSequenceState &base_;
  PagedSequenceState &candidate_;
  std::span<compute::PagedArenaCommit> commits_;
};

} // namespace hyle::inference
