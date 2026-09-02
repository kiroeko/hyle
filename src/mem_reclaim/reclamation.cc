#include "hyle/mem/reclamation.h"

#include <new>

namespace hyle::mem {

namespace {

thread_local ReclamationDomain *callback_domain = nullptr;

std::atomic<std::uint64_t> &leased_reader_routes() noexcept {
  static std::atomic<std::uint64_t> routes{0};
  return routes;
}

std::atomic<std::size_t> &next_fallback_route() noexcept {
  static std::atomic<std::size_t> next{0};
  return next;
}

class ThreadReaderRoute final {
public:
  ThreadReaderRoute() noexcept {
    std::atomic<std::uint64_t> &routes = leased_reader_routes();
    for (;;) {
      std::uint64_t current = routes.load(std::memory_order_relaxed);
      if (current == ~std::uint64_t{0}) {
        break;
      }
      for (std::size_t slot = 0; slot != detail::kReclamationReaderSlots;
           ++slot) {
        const std::uint64_t bit = std::uint64_t{1} << slot;
        if ((current & bit) != 0) {
          continue;
        }
        if (routes.compare_exchange_weak(current, current | bit,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
          value_ = slot;
          return;
        }
        break;
      }
    }
    value_ = detail::kReclamationReaderSlots +
             next_fallback_route().fetch_add(1, std::memory_order_relaxed) %
                 detail::kReclamationFallbackStripes;
  }

  ~ThreadReaderRoute() noexcept {
    if (value_ >= detail::kReclamationReaderSlots) {
      return;
    }
    const std::uint64_t bit = std::uint64_t{1} << value_;
    const std::uint64_t prior =
        leased_reader_routes().fetch_and(~bit, std::memory_order_release);
    HYLE_FATAL_ASSERT((prior & bit) != 0);
  }

  [[nodiscard]] std::size_t value() const noexcept { return value_; }

private:
  std::size_t value_ = detail::kReclamationReaderSlots;
};

template <class T> class NoDestructor final {
public:
  NoDestructor() noexcept { ::new (storage_) T(); }

  T &get() noexcept {
    return *std::launder(reinterpret_cast<T *>(storage_));
  }

private:
  alignas(T) unsigned char storage_[sizeof(T)];
};

} // namespace

std::size_t detail::reclamation_thread_route() noexcept {
  static thread_local ThreadReaderRoute route;
  return route.value();
}

ReclamationDomain::~ReclamationDomain() noexcept {
  static_cast<void>(drain());
  HYLE_FATAL_ASSERT(readers_clear());
}

void ReclamationDomain::retire(RetireRecord &record, void *state,
                               ReclaimCallback callback) noexcept {
  HYLE_FATAL_ASSERT(callback != nullptr);
  bool expected = false;
  HYLE_FATAL_ASSERT(record.pending_.compare_exchange_strong(
      expected, true, std::memory_order_acq_rel, std::memory_order_acquire));

  record.state_ = state;
  record.callback_ = callback;

  std::size_t count = 0;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    const std::uint8_t epoch =
        active_epoch_.load(std::memory_order_seq_cst);
    record.next_ = retired_[epoch];
    retired_[epoch] = &record;
    count = pending_.fetch_add(1, std::memory_order_release) + 1;
  }

  if (retire_threshold_ != 0 && count % retire_threshold_ == 0) {
    static_cast<void>(try_reclaim());
  }
}

void ReclamationDomain::start_grace_period() noexcept {
  HYLE_FATAL_ASSERT(!grace_in_progress_);
  std::lock_guard<std::mutex> lock(queue_mutex_);
  grace_epoch_ = active_epoch_.load(std::memory_order_seq_cst);
  active_epoch_.store(static_cast<std::uint8_t>(grace_epoch_ ^ 1U),
                      std::memory_order_seq_cst);
  grace_in_progress_ = true;
}

bool ReclamationDomain::grace_period_clear() const noexcept {
  HYLE_FATAL_ASSERT(grace_in_progress_);
  for (const detail::ReclamationReaderSlot &slot : reader_slots_) {
    const std::uint64_t state =
        slot.epoch_readers.load(std::memory_order_seq_cst);
    if ((state & detail::kReclamationReaderDepthMask) != 0 &&
        detail::reclamation_state_epoch(state) == grace_epoch_) {
      return false;
    }
  }
  for (const detail::ReclamationFallbackStripe &stripe :
       fallback_stripes_) {
    if (stripe.readers[grace_epoch_].load(std::memory_order_seq_cst) != 0) {
      return false;
    }
  }
  return true;
}

bool ReclamationDomain::readers_clear() const noexcept {
  for (const detail::ReclamationReaderSlot &slot : reader_slots_) {
    if ((slot.epoch_readers.load(std::memory_order_seq_cst) &
         detail::kReclamationReaderDepthMask) != 0) {
      return false;
    }
  }
  for (const detail::ReclamationFallbackStripe &stripe :
       fallback_stripes_) {
    for (const std::atomic<std::size_t> &readers : stripe.readers) {
      if (readers.load(std::memory_order_seq_cst) != 0) {
        return false;
      }
    }
  }
  return true;
}

void ReclamationDomain::wait_for_grace_period() const noexcept {
  HYLE_FATAL_ASSERT(grace_in_progress_);
  for (const detail::ReclamationReaderSlot &slot : reader_slots_) {
    const std::atomic<std::uint64_t> &epoch_readers = slot.epoch_readers;
    std::uint64_t state = epoch_readers.load(std::memory_order_seq_cst);
    while ((state & detail::kReclamationReaderDepthMask) != 0 &&
           detail::reclamation_state_epoch(state) == grace_epoch_) {
      epoch_readers.wait(state, std::memory_order_seq_cst);
      state = epoch_readers.load(std::memory_order_seq_cst);
    }
  }
  for (const detail::ReclamationFallbackStripe &stripe :
       fallback_stripes_) {
    const std::atomic<std::size_t> &readers = stripe.readers[grace_epoch_];
    std::size_t value = readers.load(std::memory_order_seq_cst);
    while (value != 0) {
      readers.wait(value, std::memory_order_seq_cst);
      value = readers.load(std::memory_order_seq_cst);
    }
  }
}

std::size_t ReclamationDomain::finish_grace_period() noexcept {
  HYLE_FATAL_ASSERT(grace_in_progress_);

  // A reader that sampled the old epoch before the flip may increment it
  // after the successful zero scan. It then fails read()'s validation and
  // leaves without loading a protected pointer. Zero is therefore the grace
  // decision, not a stable postcondition that can be asserted again here.

  RetireRecord *records = nullptr;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    records = std::exchange(retired_[grace_epoch_], nullptr);
  }
  grace_in_progress_ = false;

  std::size_t count = 0;
  while (records != nullptr) {
    RetireRecord *record = records;
    records = record->next_;
    void *state = record->state_;
    ReclaimCallback callback = record->callback_;
    record->next_ = nullptr;
    record->state_ = nullptr;
    record->callback_ = nullptr;
    record->pending_.store(false, std::memory_order_release);
    pending_.fetch_sub(1, std::memory_order_release);
    reclaimed_.fetch_add(1, std::memory_order_release);
    ++count;
    ReclamationDomain *previous_domain =
        std::exchange(callback_domain, this);
    callback(state);
    callback_domain = previous_domain;
  }
  return count;
}

std::size_t ReclamationDomain::try_reclaim() noexcept {
  // retire() is intentionally callback-safe. A threshold crossed by a
  // callback must not recursively acquire this domain's collector mutex.
  if (callback_domain == this) {
    return 0;
  }
  std::unique_lock<std::mutex> collector(collector_mutex_, std::try_to_lock);
  if (!collector.owns_lock()) {
    return 0;
  }
  if (!grace_in_progress_) {
    start_grace_period();
  }
  if (!grace_period_clear()) {
    return 0;
  }
  return finish_grace_period();
}

std::size_t ReclamationDomain::synchronize() noexcept {
  HYLE_FATAL_ASSERT(callback_domain != this);
  std::lock_guard<std::mutex> collector(collector_mutex_);
  std::size_t count = 0;
  if (grace_in_progress_) {
    wait_for_grace_period();
    count += finish_grace_period();
  }

  // A second flip is required when we inherited a nonblocking grace period:
  // records retired before this call may reside in the then-active epoch.
  start_grace_period();
  wait_for_grace_period();
  count += finish_grace_period();
  return count;
}

std::size_t ReclamationDomain::drain() noexcept {
  HYLE_FATAL_ASSERT(callback_domain != this);
  std::size_t count = 0;
  while (pending() != 0) {
    count += synchronize();
  }
  return count;
}

ReclamationDomain &default_reclamation_domain() noexcept {
  static NoDestructor<ReclamationDomain> domain;
  return domain.get();
}

} // namespace hyle::mem
