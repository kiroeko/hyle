#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

#include "hyle/macros.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

// Thread-level spinlock (clay spin_mutex_t). Satisfies the C++ *Lockable* named
// requirement (lock/try_lock/unlock) so it composes with std::lock_guard /
// std::scoped_lock -- no custom RAII wrapper needed (clay had spin_lock_t).
//
// For VERY short critical sections only (a few instructions); a contended or
// long hold should use a real mutex (a spinner burns a whole core). It blocks
// the OS THREAD, not a fiber -- sched's fiber-aware sync (sched-05) is separate.
//
// Improvements over clay's naive test-and-set:
//   - TTAS (test-test-and-set): spin on a plain load and only issue the atomic
//     RMW when the lock looks free. clay's RMW-every-spin bounces the cache line
//     between cores (each test_and_set is a write); a relaxed load spins in the
//     local cache until the owner's release flips it.
//   - cpu_relax() pause hint: cuts power, frees the sibling hyperthread, and
//     speeds exit from the spin loop.
//   - bounded exponential backoff: brief contention stays in CPU pause/yield
//     instructions; sustained contention yields the OS thread. This is still a
//     spin lock, not a substitute for a blocking mutex.
//
// The lock is deliberately COMPACT. Padding every lock to a cache line bloats
// ordinary owner objects (oneTBB spin_mutex and Folly MicroSpinLock are compact
// for the same reason). An owner that has measured false sharing should place
// its lock/member in an explicitly cache-line-aligned wrapper.

namespace hyle::util {

// Spin-loop hint: pause/yield so a busy-wait doesn't saturate the pipeline.
HYLE_FORCE_INLINE void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
  __yield();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  // oneTBB and Folly use an instruction-synchronization barrier on AArch64.
  // Apple's `yield` hint is too weak under a many-waiter hand-off and caused
  // the releasing core to be mobbed by repeated atomic exchanges.
  __asm__ __volatile__("isb" ::: "memory");
#else
  // unknown arch: no-op (still correct, just no pipeline hint)
#endif
}

namespace detail {

class SpinBackoff {
 public:
  void pause() noexcept {
    if (count_ <= kPausesBeforeYield) {
      for (std::uint32_t i = 0; i < count_; ++i) cpu_relax();
      count_ *= 2;
    } else {
      std::this_thread::yield();
    }
  }

 private:
  static constexpr std::uint32_t kPausesBeforeYield = 16;
  std::uint32_t count_ = 1;
};

}  // namespace detail

class SpinLock {
 public:
  SpinLock() noexcept = default;
  HYLE_DISABLE_COPY_AND_MOVE(SpinLock);

  void lock() noexcept {
    detail::SpinBackoff backoff;
    // TTAS from the first probe matters under contention: an eager RMW fast
    // path makes every arriving waiter write the already-owned cache line.
    while (flag_.load(std::memory_order_relaxed) ||
           flag_.exchange(true, std::memory_order_acq_rel)) {
      backoff.pause();
    }
  }

  bool try_lock() noexcept {
    // One shot. relaxed pre-check avoids a needless RMW when already held.
    return !flag_.load(std::memory_order_relaxed) &&
           !flag_.exchange(true, std::memory_order_acq_rel);
  }

  void unlock() noexcept { flag_.store(false, std::memory_order_release); }

  bool is_locked() const noexcept {
    return flag_.load(std::memory_order_relaxed);
  }

 private:
  static_assert(std::atomic<bool>::is_always_lock_free);
  std::atomic<bool> flag_{false};
};

}  // namespace hyle::util
