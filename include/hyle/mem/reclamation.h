/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>

#include "hyle/core/assert.h"
#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_MEM_RECLAIM_API
#elif defined(HYLE_MEM_RECLAIM_BUILD)
#define HYLE_MEM_RECLAIM_API HYLE_EXPORT
#else
#define HYLE_MEM_RECLAIM_API HYLE_IMPORT
#endif

namespace hyle::mem {

using ReclaimCallback = void (*)(void *) noexcept;

class ReclamationDomain;

// Caller-owned, allocation-free storage for one deferred callback. A record
// may be reused after its callback begins, but it must not be retired twice or
// destroyed while pending. Embed it in the object or owner whose allocator and
// destruction policy the callback understands.
class RetireRecord final {
public:
  constexpr RetireRecord() noexcept = default;
  ~RetireRecord() noexcept { HYLE_FATAL_ASSERT(!pending()); }

  RetireRecord(const RetireRecord &) = delete;
  RetireRecord &operator=(const RetireRecord &) = delete;
  RetireRecord(RetireRecord &&) = delete;
  RetireRecord &operator=(RetireRecord &&) = delete;

  [[nodiscard]] bool pending() const noexcept {
    return pending_.load(std::memory_order_acquire);
  }

private:
  friend class ReclamationDomain;

  RetireRecord *next_ = nullptr;
  void *state_ = nullptr;
  ReclaimCallback callback_ = nullptr;
  std::atomic<bool> pending_{false};
};

namespace detail {

inline constexpr std::size_t kReclamationReaderSlots = 64;
inline constexpr std::size_t kReclamationFallbackStripes = 8;
inline constexpr std::uint64_t kReclamationReaderEpochBit =
    std::uint64_t{1} << 63;
inline constexpr std::uint64_t kReclamationReaderDepthMask =
    ~kReclamationReaderEpochBit;

// A leased slot has one live-thread writer, so warmed read regions use ordinary
// atomic stores rather than contended read-modify-write operations. Bounded
// fallback stripes preserve correctness above the exclusive-slot capacity.
struct alignas(core::kDestructiveInterferenceSize) ReclamationReaderSlot {
  std::atomic<std::uint64_t> epoch_readers{0};
};

struct alignas(core::kDestructiveInterferenceSize)
    ReclamationFallbackStripe {
  std::atomic<std::size_t> readers[2]{};
};

// One process image leases each live thread an exclusive route and returns it
// at thread exit. Values below kReclamationReaderSlots name exclusive slots;
// higher values encode a bounded fallback stripe for unusual >64-thread use.
[[nodiscard]] HYLE_MEM_RECLAIM_API std::size_t
reclamation_thread_route() noexcept;

inline std::uint8_t reclamation_state_epoch(std::uint64_t state) noexcept {
  return static_cast<std::uint8_t>(state >> 63U);
}

inline std::uint64_t reclamation_state(std::uint8_t epoch,
                                       std::uint64_t depth) noexcept {
  return (static_cast<std::uint64_t>(epoch) << 63U) | depth;
}

} // namespace detail

// Move-only lifetime guard for one reclamation domain. Establish the guard
// before loading a published pointer; protect() performs the required acquire
// load and keeps the pointed-to storage alive until reset/destruction. Like a
// mature RCU read lock, a live guard must enter and leave on the same thread.
class ReclamationGuard final {
public:
  ReclamationGuard() noexcept = default;
  ~ReclamationGuard() noexcept { reset(); }

  ReclamationGuard(const ReclamationGuard &) = delete;
  ReclamationGuard &operator=(const ReclamationGuard &) = delete;

  ReclamationGuard(ReclamationGuard &&other) noexcept { move_from(other); }
  ReclamationGuard &operator=(ReclamationGuard &&other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  [[nodiscard]] bool active() const noexcept { return domain_ != nullptr; }

  template <class T>
  [[nodiscard]] T *protect(const std::atomic<T *> &published) const noexcept {
    HYLE_FATAL_ASSERT(active());
    return published.load(std::memory_order_acquire);
  }

  void reset() noexcept;

private:
  friend class ReclamationDomain;

  ReclamationGuard(ReclamationDomain &domain,
                   detail::ReclamationReaderSlot &slot,
                   std::uint8_t epoch) noexcept
      : domain_(&domain), slot_(&slot), epoch_(epoch) {}
  ReclamationGuard(ReclamationDomain &domain, std::size_t fallback_stripe,
                   std::uint8_t epoch) noexcept
      : domain_(&domain), fallback_stripe_(fallback_stripe), epoch_(epoch) {}

  void move_from(ReclamationGuard &other) noexcept {
    domain_ = std::exchange(other.domain_, nullptr);
    slot_ = other.slot_;
    fallback_stripe_ = other.fallback_stripe_;
    epoch_ = other.epoch_;
  }

  ReclamationDomain *domain_ = nullptr;
  detail::ReclamationReaderSlot *slot_ = nullptr;
  std::size_t fallback_stripe_ = 0;
  std::uint8_t epoch_ = 0;
};

// Two-generation read-copy-update domain. Read-side work is allocation-free,
// lock-free and normally uses an exclusive cache-line reader slot. Bounded
// fallback stripes preserve correctness after unusual thread churn. Writers
// own retire records and may collect opportunistically or wait for a grace
// period.
// Export is method-level: the inline read path must see the concrete layout,
// while exporting the whole class would incorrectly require MSVC's std::atomic
// and std::mutex implementations to carry a DLL interface (C4251).
class ReclamationDomain final {
public:
  static constexpr std::size_t kDefaultRetireThreshold = 64;

  explicit ReclamationDomain(
      std::size_t retire_threshold = kDefaultRetireThreshold) noexcept
      : retire_threshold_(retire_threshold) {}
  HYLE_MEM_RECLAIM_API ~ReclamationDomain() noexcept;

  ReclamationDomain(const ReclamationDomain &) = delete;
  ReclamationDomain &operator=(const ReclamationDomain &) = delete;
  ReclamationDomain(ReclamationDomain &&) = delete;
  ReclamationDomain &operator=(ReclamationDomain &&) = delete;

  [[nodiscard]] ReclamationGuard read() noexcept {
    const std::size_t route = detail::reclamation_thread_route();
    if (route < detail::kReclamationReaderSlots) {
      detail::ReclamationReaderSlot *slot = &reader_slots_[route];
      std::uint64_t state =
          slot->epoch_readers.load(std::memory_order_relaxed);
      const std::uint64_t depth =
          state & detail::kReclamationReaderDepthMask;
      if (depth != 0) {
        HYLE_FATAL_ASSERT(depth != detail::kReclamationReaderDepthMask);
        slot->epoch_readers.store(state + 1, std::memory_order_relaxed);
        return ReclamationGuard(
            *this, *slot, detail::reclamation_state_epoch(state));
      }

      for (;;) {
        const std::uint8_t epoch =
            active_epoch_.load(std::memory_order_seq_cst);
        slot->epoch_readers.store(detail::reclamation_state(epoch, 1),
                                  std::memory_order_seq_cst);
        if (active_epoch_.load(std::memory_order_seq_cst) == epoch) {
          return ReclamationGuard(*this, *slot, epoch);
        }
        slot->epoch_readers.store(0, std::memory_order_release);
        slot->epoch_readers.notify_all();
      }
    }

    const std::size_t stripe =
        (route - detail::kReclamationReaderSlots) &
        (detail::kReclamationFallbackStripes - 1U);
    for (;;) {
      const std::uint8_t epoch =
          active_epoch_.load(std::memory_order_seq_cst);
      fallback_stripes_[stripe].readers[epoch].fetch_add(
          1, std::memory_order_seq_cst);
      if (active_epoch_.load(std::memory_order_seq_cst) == epoch) {
        return ReclamationGuard(*this, stripe, epoch);
      }
      leave_fallback(stripe, epoch);
    }
  }

  // Enqueue one callback without allocating. The callback and state remain
  // borrowed until invocation. A threshold-triggered collection never waits.
  HYLE_MEM_RECLAIM_API void retire(RetireRecord &record, void *state,
                                   ReclaimCallback callback) noexcept;

  // Typed, allocation-free convenience for the common intrusive-owner shape.
  // Callback is a compile-time value so RetireRecord does not need to store a
  // second erased function pointer or allocate a deleter object.
  template <auto Callback, class T>
  void retire(RetireRecord &record, T *state) noexcept {
    static_assert(!std::is_const_v<T>,
                  "retired callback state must be mutable storage");
    static_assert(std::is_nothrow_invocable_r_v<void, decltype(Callback), T *>,
                  "retired callback must be callable as noexcept void(T*)");
    retire(
        record, static_cast<void *>(state),
        +[](void *raw) noexcept { Callback(static_cast<T *>(raw)); });
  }

  // Return callbacks invoked by this call. try_reclaim() never waits for a
  // reader; synchronize() waits only for readers preceding its epoch flips.
  [[nodiscard]] HYLE_MEM_RECLAIM_API std::size_t try_reclaim() noexcept;
  [[nodiscard]] HYLE_MEM_RECLAIM_API std::size_t synchronize() noexcept;

  // External quiescence is required: no new guard or retire may race drain or
  // destruction. Callbacks may enqueue follow-up records; drain loops until
  // the queue becomes empty.
  [[nodiscard]] HYLE_MEM_RECLAIM_API std::size_t drain() noexcept;

  [[nodiscard]] std::size_t pending() const noexcept {
    return pending_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::size_t reclaimed() const noexcept {
    return reclaimed_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::size_t retire_threshold() const noexcept {
    return retire_threshold_;
  }

private:
  friend class ReclamationGuard;

  void leave_slot(detail::ReclamationReaderSlot &slot,
                  std::uint8_t epoch) noexcept {
    const std::uint64_t state =
        slot.epoch_readers.load(std::memory_order_relaxed);
    const std::uint64_t depth =
        state & detail::kReclamationReaderDepthMask;
    HYLE_FATAL_ASSERT(depth != 0);
    HYLE_FATAL_ASSERT(detail::reclamation_state_epoch(state) == epoch);
    if (depth != 1) {
      slot.epoch_readers.store(state - 1, std::memory_order_relaxed);
      return;
    }
    slot.epoch_readers.store(0, std::memory_order_release);
    slot.epoch_readers.notify_all();
  }

  void leave_fallback(std::size_t stripe, std::uint8_t epoch) noexcept {
    const std::size_t prior =
        fallback_stripes_[stripe].readers[epoch].fetch_sub(
            1, std::memory_order_seq_cst);
    HYLE_FATAL_ASSERT(prior != 0);
    if (prior == 1) {
      fallback_stripes_[stripe].readers[epoch].notify_all();
    }
  }

  void start_grace_period() noexcept;
  [[nodiscard]] bool grace_period_clear() const noexcept;
  [[nodiscard]] bool readers_clear() const noexcept;
  void wait_for_grace_period() const noexcept;
  [[nodiscard]] std::size_t finish_grace_period() noexcept;

  std::array<detail::ReclamationReaderSlot,
             detail::kReclamationReaderSlots>
      reader_slots_{};
  std::array<detail::ReclamationFallbackStripe,
             detail::kReclamationFallbackStripes>
      fallback_stripes_{};
  std::atomic<std::uint8_t> active_epoch_{0};

  // Queue lock linearizes retire with an epoch flip. Collector lock permits
  // one callback runner while retire remains reentrant from a callback.
  std::mutex queue_mutex_;
  std::mutex collector_mutex_;
  RetireRecord *retired_[2]{nullptr, nullptr};
  bool grace_in_progress_ = false; // guarded by collector_mutex_
  std::uint8_t grace_epoch_ = 0;   // guarded by collector_mutex_

  std::atomic<std::size_t> pending_{0};
  std::atomic<std::size_t> reclaimed_{0};
  const std::size_t retire_threshold_;
};

// Process-lifetime convenience for small owners and shared containers. Prefer
// an explicit domain when a long reader must not delay unrelated reclamation.
HYLE_MEM_RECLAIM_API ReclamationDomain &
default_reclamation_domain() noexcept;

inline void ReclamationGuard::reset() noexcept {
  if (domain_ != nullptr) {
    if (slot_ != nullptr) {
      domain_->leave_slot(*slot_, epoch_);
    } else {
      domain_->leave_fallback(fallback_stripe_, epoch_);
    }
    domain_ = nullptr;
  }
}

} // namespace hyle::mem

#undef HYLE_MEM_RECLAIM_API
