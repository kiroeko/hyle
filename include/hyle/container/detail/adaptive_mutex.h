#pragma once

#include <atomic>
#include <cstdint>

#include "hyle/core/compiler.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hyle::container::detail {

// Compact exclusive lock for sharded containers whose common critical section
// is short but whose cold grow/allocation path may block. The fast path is one
// test-and-set plus a release clear; bounded spinning absorbs brief hand-offs,
// then atomic::wait parks without making every uncontended unlock call notify.
// A waiter count, rather than a lossy bool hint, ensures successive owners wake
// every parked waiter without a thundering herd.
class AdaptiveMutex {
public:
  AdaptiveMutex() noexcept = default;
  AdaptiveMutex(const AdaptiveMutex&) = delete;
  AdaptiveMutex& operator=(const AdaptiveMutex&) = delete;
  AdaptiveMutex(AdaptiveMutex&&) = delete;
  AdaptiveMutex& operator=(AdaptiveMutex&&) = delete;

  void lock() noexcept {
    if (!locked_.test_and_set(std::memory_order_acquire))
      return;

    for (std::uint32_t spin = 0; spin < kSpinLimit; ++spin) {
      const std::uint32_t pauses = std::uint32_t{1} << (spin / 8);
      for (std::uint32_t i = 0; i < pauses; ++i)
        cpu_relax();
      if (!locked_.test(std::memory_order_relaxed) &&
          !locked_.test_and_set(std::memory_order_acquire))
        return;
    }

    waiters_.fetch_add(1, std::memory_order_relaxed);
    while (locked_.test_and_set(std::memory_order_acquire))
      locked_.wait(true, std::memory_order_relaxed);
    waiters_.fetch_sub(1, std::memory_order_relaxed);
  }

  bool try_lock() noexcept {
    return !locked_.test(std::memory_order_relaxed) &&
           !locked_.test_and_set(std::memory_order_acquire);
  }

  void unlock() noexcept {
    locked_.clear(std::memory_order_release);
    if (waiters_.load(std::memory_order_relaxed) != 0)
      locked_.notify_one();
  }

private:
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

  static constexpr std::uint32_t kSpinLimit = 32;
  std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
  std::atomic<std::uint32_t> waiters_{0};
};

} // namespace hyle::container::detail
