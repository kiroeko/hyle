#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/waker.h"
#include "hyle/util/spin_lock.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

// Fiber-aware synchronization primitives (sched-design §2.4 缺口补记, §3 spine).
//
// WHY they exist: a std::mutex / std::condition_variable blocks the WORKER OS
// THREAD. On an M:N scheduler that strands every OTHER fiber pinned to that core
// -- the whole point of fibers is defeated. The fix is "on contention, YIELD the
// fiber (it leaves the run loop), and on release RE-INJECT it" -- which is just
// another completion/wake, i.e. the sched-04a Waker spine reused.
//
// Convergence (marl's lesson, adopted): do NOT write a bespoke wake mechanism for
// mutex / waitgroup / channel / latch. Build ONE fiber-aware ConditionVariable
// and layer everything on it. Everything below is a thin wrapper around CV.
//
// Dual-mode (REQUIRED, not optional): a waiter may be a fiber (yield + re-inject)
// OR a bare OS thread (fall back to platform::Parker). HYLE's cross-process
// federation and external-thread submission paths hand work to the scheduler from
// threads that are NOT fibers; assuming "all waiters are fibers" would deadlock
// them. The CV picks the mode per-wait from whether it runs inside a worker fiber.
//
// TIMED WAITS (wait_for/wait_until, sched-04c): THREAD-MODE ships here. A bare
// (non-fiber) waiter parks with a bounded platform::Parker::park_timeout and
// retracts its own node on the deadline; the notify-vs-timeout race is arbitrated
// under the CV monitor lock via the node's `linked` flag (whoever acts first
// wins, the loser reports the other's outcome so no wake is lost). This covers
// external submitters and cross-process federation threads.
//
// FIBER-MODE timed wait is still deferred: a fiber waiter would arm a time-02
// wheel timer whose fire races the notify, and the loser must CANCEL the other --
// but the woken fiber may resume on a different worker than the one holding its
// timer, so that needs the per-worker cross-worker timer-cancel command queue
// (the time-02 headline divergence point,真正 load-bearing 于 io-timeout). Until
// that lands, a timed wait from a worker fiber asserts. The design mirrors folly
// fibers Baton (atomic-arbitrated timeout-vs-post, same-thread timer to dodge
// cross-core cancel).

namespace hyle::sched {

// The one fiber-aware blocking primitive; all others build on it. Mirrors
// std::condition_variable_any: wait() takes ANY BasicLockable (the monitor lock),
// releases it while blocked, and re-acquires it before returning (Mesa
// semantics -- always re-check the predicate in a loop).
//
// API macro is on the methods, not the class: the SpinLock member holds a
// std::atomic, so exporting the whole class would drag it across the DLL seam
// (MSVC C4251). Method-level export keeps the layout client-side (same pattern as
// Fiber / Parker / Scheduler).
class ConditionVariable {
 public:
  ConditionVariable() noexcept = default;
  HYLE_DISABLE_COPY_AND_MOVE(ConditionVariable);

  // Atomically release `lock` and block until notified, then re-acquire `lock`.
  // Precondition: the caller holds `lock`. Spurious wakeups are possible only via
  // the predicate overload; the bare overload returns exactly once per notify.
  template <class Lock>
  void wait(Lock& lock) {
    Waiter node;
    const bool fiber = prepare_wait_(&node);  // enqueue under internal lock
    lock.unlock();                            // release the monitor lock
    finish_wait_(fiber, node.token);          // fiber: suspend(); thread: park()
    lock.lock();                              // re-acquire before returning
  }

  // Wait until `stop()` holds, immune to spurious/lost wakeups (the standard
  // predicate loop). `stop` is evaluated with `lock` held.
  template <class Lock, class Pred>
  void wait(Lock& lock, Pred stop) {
    while (!stop()) wait(lock);
  }

  // Timed wait (sched-04c). Atomically release `lock` and block until notified
  // OR `timeout_ns` elapses, then re-acquire `lock`. Returns whether it returned
  // because of the timeout. Deadline keys on time::monotonic_ns().
  //
  // THREAD-MODE ONLY for now: a bare (non-fiber) waiter parks with a bounded
  // Parker::park_timeout -- no timer wheel, no cross-worker cancel, so it is
  // simple and correct. A FIBER waiter would need to arm a wheel timer whose
  // fire races the notify and whose loser must cancel across workers (the
  // per-worker cancel command queue -- still sched-04c-pending); until that
  // lands, calling this from a worker fiber asserts. Bare-thread timed waits
  // (external submitters, cross-process federation) ship now.
  enum class CvStatus { kNoTimeout, kTimeout };

  template <class Lock>
  CvStatus wait_for(Lock& lock, std::uint64_t timeout_ns) {
    Waiter node;
    prepare_wait_(&node);  // (asserts thread-mode; sets up token)
    lock.unlock();
    const CvStatus st = finish_wait_timed_(&node, timeout_ns);
    lock.lock();
    return st;
  }

  // Predicate form: loops re-checking `stop()` until it holds or the deadline
  // passes. Returns stop() -- true if the condition was met, false on timeout.
  // Recomputes the remaining time each iteration so spurious wakes don't extend
  // the total wait (deadline captured once from now + timeout_ns).
  template <class Lock, class Pred>
  bool wait_for(Lock& lock, std::uint64_t timeout_ns, Pred stop) {
    return wait_until(lock, deadline_from_now_(timeout_ns), stop);
  }

  // Absolute-deadline variants (monotonic ns), mirroring std::cv wait_until.
  template <class Lock>
  CvStatus wait_until(Lock& lock, std::uint64_t deadline_ns) {
    return wait_for(lock, remaining_ns_(deadline_ns));
  }

  template <class Lock, class Pred>
  bool wait_until(Lock& lock, std::uint64_t deadline_ns, Pred stop) {
    while (!stop()) {
      if (wait_for(lock, remaining_ns_(deadline_ns)) == CvStatus::kTimeout) {
        return stop();  // last check: the predicate may have become true at the wire
      }
    }
    return true;
  }

  // Wake one / all blocked waiters. Safe to call from any thread or fiber. The
  // caller should hold whatever lock guards the predicate the waiters test (so a
  // state change + notify is not split by a sleeping waiter -- classic monitor
  // discipline); notify itself takes only the CV's own internal lock. Fast-path:
  // when no waiter is parked, returns without touching the internal lock (the
  // `waiters_` count is published under the monitor lock by a parking waiter, so
  // a notifier holding that same monitor lock sees a non-zero count -- no missed
  // wake). This is what keeps an uncontended Mutex::unlock cheap.
  HYLE_SCHED_API void notify_one() noexcept;
  HYLE_SCHED_API void notify_all() noexcept;

 private:
  // One waiter, allocated on the blocking fiber/thread's own stack and linked
  // into the FIFO list while parked. `waker` names the fiber to re-inject (valid
  // iff fiber-mode); `token` points at a thread-local park token (non-null iff
  // thread-mode). Fields are read by the notifier under the internal lock BEFORE
  // the waiter is woken (after which the node may be destroyed). The list is
  // DOUBLY linked so a timed-out waiter can unlink itself in O(1) (it holds its
  // own node but not its predecessor); `linked` makes that removal idempotent
  // against a concurrent notify that already popped it.
  struct Waiter {
    Waiter* next;
    Waiter* prev;
    Waker waker;
    void* token;
    bool linked;  // true while in the list; guarded by mu_
  };

  // Capture mode (fiber vs thread), set up `node`, and link it onto the FIFO tail
  // under the internal lock. Returns true iff this is a fiber waiter. Must run
  // while the caller still holds the monitor lock (closes the lost-wakeup window).
  HYLE_SCHED_API bool prepare_wait_(Waiter* node) noexcept;

  // Block until woken: fiber -> suspend() (leaves the run loop, re-injected by
  // the Waker); thread -> park() loop on the thread-local token.
  HYLE_SCHED_API static void finish_wait_(bool fiber, void* token) noexcept;

  // Timed thread-mode block: park the OS thread with a bounded park_timeout
  // until the node's token is notified or the deadline elapses. On timeout it
  // retracts `node` from the list (remove_waiter_) so a later notify can't pop a
  // dead node. Returns whether it timed out. (Out-of-line: pulls in time/ +
  // platform futex, kept out of this header.)
  HYLE_SCHED_API CvStatus finish_wait_timed_(Waiter* node,
                                             std::uint64_t timeout_ns) noexcept;

  // Unlink `node` from the list if still linked (idempotent). Caller must hold
  // mu_. Used by a timed waiter that woke on its deadline before a notify reached
  // it -- it retracts its own node so a later notify can't pop a dead stack node.
  HYLE_SCHED_API void remove_waiter_(Waiter* node) noexcept;

  // now + timeout_ns as an absolute monotonic deadline (saturating).
  HYLE_SCHED_API static std::uint64_t deadline_from_now_(
      std::uint64_t timeout_ns) noexcept;
  // deadline - now, clamped at 0 (already passed => non-blocking).
  HYLE_SCHED_API static std::uint64_t remaining_ns_(
      std::uint64_t deadline_ns) noexcept;

  util::SpinLock mu_;        // guards the waiter list (tiny critical sections)
  Waiter* head_ = nullptr;   // FIFO: notify pops head
  Waiter* tail_ = nullptr;   // FIFO: prepare_wait appends here
  // Count of parked waiters, so notify can skip the lock when none. Mutated
  // under mu_ (++ in prepare_wait_, -= when popped by notify); read lock-free on
  // the notify fast-path. Correctness rests on the monitor-lock handoff, not on
  // this count alone (the under-lock head_ check is still authoritative).
  std::atomic<int> waiters_{0};
};

// ---- Mutex ------------------------------------------------------------------
// A futex-based mutex (Drepper's 3-state lock), built on HYLE's own
// platform::wait_on_address (futex / WaitOnAddress / __ulock) -- NOT on the
// fiber CV. Uncontended lock/unlock is a single atomic CAS, matching std::mutex
// / marl (which is std::mutex). On contention the WORKER thread parks on the
// futex word; it does NOT yield the fiber.
//
// WHY not fiber-aware (this is the SOTA call, learned from marl): a mutex guards
// SHORT critical sections. Yielding the fiber (suspend + re-inject, ~100ns of
// context switch) is far costlier than briefly parking on a futex while the
// holder -- running on another worker -- finishes its few-instruction section.
// marl deliberately keeps its mutex == std::mutex for exactly this reason; an
// earlier fiber-aware HYLE Mutex measured 5.8x SLOWER than marl on contention.
// For genuinely LONG / cross-yield waits, use ConditionVariable (fiber-aware) --
// that is where yielding pays off. Contract (same as marl/std): do not hold a
// Mutex across a blocking call or a fiber yield; use a CV for that.
//
// state_: 0 = unlocked, 1 = locked (no waiters), 2 = locked (waiters possible).
// Satisfies BasicLockable/Lockable (composes with std::lock_guard).
class Mutex {
 public:
  Mutex() noexcept = default;
  HYLE_DISABLE_COPY_AND_MOVE(Mutex);

  void lock() noexcept {
    std::uint32_t expected = 0;
    // Fast path: claim a free lock with one CAS (0 -> 1).
    if (!state_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
      lock_slow(expected);  // contended: mark waiters + futex-park
    }
  }

  bool try_lock() noexcept {
    std::uint32_t expected = 0;
    return state_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                          std::memory_order_relaxed);
  }

  void unlock() noexcept {
    // Drepper: decrement; if the previous value was not 1 (i.e. was 2, waiters),
    // there is someone to wake.
    if (state_.fetch_sub(1, std::memory_order_release) != 1) {
      unlock_slow();
    }
  }

 private:
  // Out-of-line so the header carries no futex dependency; the hot path above is
  // pure inline atomics.
  HYLE_SCHED_API void lock_slow(std::uint32_t c) noexcept;
  HYLE_SCHED_API void unlock_slow() noexcept;

  std::atomic<std::uint32_t> state_{0};
};

// ---- WaitGroup --------------------------------------------------------------
// Go-style "wait until N tasks have finished". add(n) before spawning, done()
// per finished task, wait() blocks (fiber-aware) until the count hits zero.
class WaitGroup {
 public:
  WaitGroup() noexcept = default;
  explicit WaitGroup(long initial) noexcept : count_(initial) {}
  HYLE_DISABLE_COPY_AND_MOVE(WaitGroup);

  void add(long n = 1) noexcept {
    mu_.lock();
    count_ += n;
    HYLE_FATAL_ASSERT(count_ >= 0);  // more done() than add() is a logic bug
    mu_.unlock();
  }

  void done() noexcept {
    mu_.lock();
    HYLE_FATAL_ASSERT(count_ > 0);
    const bool drained = (--count_ == 0);
    mu_.unlock();
    if (drained) cv_.notify_all();  // release everyone parked in wait()
  }

  void wait() noexcept {
    mu_.lock();
    while (count_ > 0) cv_.wait(mu_);
    mu_.unlock();
  }

 private:
  util::SpinLock mu_;
  ConditionVariable cv_;
  long count_ = 0;
};

// ---- Latch ------------------------------------------------------------------
// One-shot countdown (std::latch shape). Constructed with a count; count_down()
// decrements; wait() blocks (fiber-aware) until it reaches zero. Unlike
// WaitGroup, the count only ever goes down -- no add().
class Latch {
 public:
  explicit Latch(long count) noexcept : count_(count) {
    HYLE_FATAL_ASSERT(count >= 0);
  }
  HYLE_DISABLE_COPY_AND_MOVE(Latch);

  void count_down(long n = 1) noexcept {
    mu_.lock();
    HYLE_FATAL_ASSERT(count_ >= n);  // counting past zero is a logic bug
    const bool opened = ((count_ -= n) == 0);
    mu_.unlock();
    if (opened) cv_.notify_all();
  }

  bool try_wait() const noexcept {
    mu_.lock();
    const bool open = (count_ == 0);
    mu_.unlock();
    return open;
  }

  void wait() noexcept {
    mu_.lock();
    while (count_ > 0) cv_.wait(mu_);
    mu_.unlock();
  }

 private:
  mutable util::SpinLock mu_;
  ConditionVariable cv_;
  long count_;
};

// ---- Channel ----------------------------------------------------------------
// Bounded MPMC channel of T, fiber-aware on both ends. Producers block (yield)
// when full, consumers when empty, via two CVs over one SpinLock + a ring buffer.
// close() wakes everyone: subsequent send() fails, recv() drains then fails.
//
// T must be movable. Storage comes from the injected allocator (charter: no
// hidden heap). Built on the same CV foundation -- demonstrates it composes.
template <class T>
class Channel {
 public:
  Channel(mem::Allocator alloc, std::size_t capacity)
      : alloc_(alloc), cap_(capacity) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(capacity >= 1);
    buf_ = static_cast<T*>(alloc_.allocate(sizeof(T) * cap_, alignof(T)));
    HYLE_FATAL_ASSERT(buf_ != nullptr);
  }

  ~Channel() {
    while (size_ > 0) {  // destroy anything still buffered
      buf_[head_].~T();
      head_ = (head_ + 1) % cap_;
      --size_;
    }
    alloc_.deallocate(buf_, sizeof(T) * cap_, alignof(T));
  }

  HYLE_DISABLE_COPY_AND_MOVE(Channel);

  // Enqueue `v`, blocking the fiber while the buffer is full. Returns false if
  // the channel is closed (the value is dropped).
  bool send(T v) {
    mu_.lock();
    while (size_ == cap_ && !closed_) not_full_.wait(mu_);
    if (closed_) {
      mu_.unlock();
      return false;
    }
    new (&buf_[tail_]) T(std::move(v));
    tail_ = (tail_ + 1) % cap_;
    ++size_;
    mu_.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Dequeue into `out`, blocking the fiber while empty. Returns false once the
  // channel is closed AND drained (no value written).
  bool recv(T& out) {
    mu_.lock();
    while (size_ == 0 && !closed_) not_empty_.wait(mu_);
    if (size_ == 0) {  // closed and drained
      mu_.unlock();
      return false;
    }
    out = std::move(buf_[head_]);
    buf_[head_].~T();
    head_ = (head_ + 1) % cap_;
    --size_;
    mu_.unlock();
    not_full_.notify_one();
    return true;
  }

  // Close the channel: wakes all blocked senders/receivers. Idempotent.
  void close() {
    mu_.lock();
    closed_ = true;
    mu_.unlock();
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  mem::Allocator alloc_;
  util::SpinLock mu_;             // guards the ring + closed_ (short holds)
  ConditionVariable not_empty_;   // consumers wait here
  ConditionVariable not_full_;    // producers wait here
  T* buf_ = nullptr;
  std::size_t cap_;
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  std::size_t size_ = 0;
  bool closed_ = false;
};

}  // namespace hyle::sched
