#pragma once

#include <atomic>
#include <cstdint>

#include "hyle/core/compiler.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hyle::container::detail {

// Compact writer-preferring reader/writer lock for a sharded container.
//
// The state layout follows oneTBB spin_rw_mutex (writer, writer-pending, reader
// count), with one extra reader-waiting bit. A waiter parks with C++20
// atomic::wait after a short bounded spin. The waiting bit is performance-
// critical: notify_all on every uncontended last-reader release enters
// libstdc++'s shared waiter pool and becomes an unrelated global bottleneck.
// That distinction matters for a flat hash shard: a grow/rehash can be longer
// than the tiny critical section for which an unbounded spin mutex is suitable.
class AdaptiveRwLock {
public:
  AdaptiveRwLock() noexcept = default;
  AdaptiveRwLock(const AdaptiveRwLock&) = delete;
  AdaptiveRwLock& operator=(const AdaptiveRwLock&) = delete;
  AdaptiveRwLock(AdaptiveRwLock&&) = delete;
  AdaptiveRwLock& operator=(AdaptiveRwLock&&) = delete;

  void lock() noexcept {
    std::uint32_t spins = 0;
    for (;;) {
      std::uintptr_t state = state_.load(std::memory_order_relaxed);
      if ((state & kBusy) == 0) {
        if (state_.compare_exchange_weak(
                state, kWriter | (state & kReaderWaiting),
                std::memory_order_acquire, std::memory_order_relaxed))
          return;
        spins = 0;
        continue;
      }
      if ((state & kWriterPending) == 0)
        state_.fetch_or(kWriterPending, std::memory_order_relaxed);
      wait_or_pause(spins, state);
    }
  }

  bool try_lock() noexcept {
    std::uintptr_t state = state_.load(std::memory_order_relaxed);
    return (state & kBusy) == 0 &&
           state_.compare_exchange_strong(
               state, kWriter | (state & kReaderWaiting),
               std::memory_order_acquire, std::memory_order_relaxed);
  }

  void unlock() noexcept {
    // A reader may have loaded an apparently free state just before this
    // writer acquired, then transiently add its count and undo it after seeing
    // kWriter. Preserve that transient count while clearing writer flags; a
    // plain store(0) could race its fetch_sub and underflow the state.
    const std::uintptr_t previous =
        state_.fetch_and(kReaders, std::memory_order_release);
    if ((previous & (kWriterPending | kReaderWaiting)) != 0)
      state_.notify_all();
  }

  void lock_shared() noexcept {
    std::uint32_t spins = 0;
    for (;;) {
      std::uintptr_t state = state_.load(std::memory_order_relaxed);
      if ((state & (kWriter | kWriterPending)) == 0) {
        const std::uintptr_t previous =
            state_.fetch_add(kOneReader, std::memory_order_acquire);
        if ((previous & kWriter) == 0)
          return;
        remove_reader();
      }
      if ((state & kReaderWaiting) == 0)
        state_.fetch_or(kReaderWaiting, std::memory_order_relaxed);
      wait_or_pause(spins, state);
    }
  }

  bool try_lock_shared() noexcept {
    const std::uintptr_t state = state_.load(std::memory_order_relaxed);
    if ((state & (kWriter | kWriterPending)) != 0)
      return false;
    const std::uintptr_t previous =
        state_.fetch_add(kOneReader, std::memory_order_acquire);
    if ((previous & kWriter) == 0)
      return true;
    remove_reader();
    return false;
  }

  void unlock_shared() noexcept { remove_reader(); }

private:
  void remove_reader() noexcept {
    const std::uintptr_t previous =
        state_.fetch_sub(kOneReader, std::memory_order_release);
    HYLE_FATAL_ASSERT((previous & kReaders) != 0);
    if ((previous & kReaders) == kOneReader && (previous & kWriterPending) != 0)
      state_.notify_all();
  }
  static HYLE_FORCE_INLINE void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("isb" ::: "memory");
#endif
  }

  void wait_or_pause(std::uint32_t &spins, std::uintptr_t observed) noexcept {
    if (spins < kSpinLimit) {
      const std::uint32_t count = std::uint32_t{1} << (spins / 8);
      for (std::uint32_t i = 0; i < count; ++i)
        cpu_relax();
      ++spins;
      return;
    }
    // Wait on the state observed by the acquisition loop, not a fresh load.
    // The owner may have released between the loop's failed check and here. A
    // fresh load could then see the now-acquirable kWriterPending-only state
    // and sleep forever after its only notification; atomic::wait(old) instead
    // returns immediately whenever that transition already happened.
    state_.wait(observed, std::memory_order_relaxed);
    spins = 0;
  }

  static constexpr std::uintptr_t kWriter = 1;
  static constexpr std::uintptr_t kWriterPending = 2;
  static constexpr std::uintptr_t kReaderWaiting = 4;
  static constexpr std::uintptr_t kOneReader = 8;
  static constexpr std::uintptr_t kReaders =
      ~(kWriter | kWriterPending | kReaderWaiting);
  static constexpr std::uintptr_t kBusy = kWriter | kReaders;
  static constexpr std::uint32_t kSpinLimit = 32;

  // Bit 0 writer; bit 1 writer pending; bit 2 reader waiter present; bits 3..
  // reader count. Waiter bits are hints, so a no-waiter hot release never needs
  // to enter the platform notification machinery.
  static_assert(std::atomic<std::uintptr_t>::is_always_lock_free);
  std::atomic<std::uintptr_t> state_{0};
};

} // namespace hyle::container::detail
