#pragma once

#include <atomic>

#include "hyle/macros.h"

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
//   - cache-line padding: the flag owns its line so it can't false-share with a
//     neighbouring hot variable.

namespace hyle::util {

// Spin-loop hint: pause/yield so a busy-wait doesn't saturate the pipeline.
HYLE_FORCE_INLINE void cpu_relax() noexcept {
#if defined(_MSC_VER)
  _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield");
#else
  // unknown arch: no-op (still correct, just no pipeline hint)
#endif
}

class HYLE_ALIGNED SpinLock {
 public:
  SpinLock() noexcept = default;
  HYLE_DISABLE_COPY_AND_MOVE(SpinLock);

  void lock() noexcept {
    for (;;) {
      // Fast path: claim it if free.
      if (!flag_.exchange(true, std::memory_order_acquire)) return;
      // Contended: spin on a RELAXED load (TTAS) until it looks free, so we
      // re-attempt the write only when the owner has released -- no cache-line
      // ping-pong while waiting.
      while (flag_.load(std::memory_order_relaxed)) cpu_relax();
    }
  }

  bool try_lock() noexcept {
    // One shot. relaxed pre-check avoids a needless RMW when already held.
    return !flag_.load(std::memory_order_relaxed) &&
           !flag_.exchange(true, std::memory_order_acquire);
  }

  void unlock() noexcept { flag_.store(false, std::memory_order_release); }

  bool is_locked() const noexcept {
    return flag_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> flag_{false};
};

}  // namespace hyle::util
