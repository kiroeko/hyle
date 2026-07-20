#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform.h"  // HYLE_OS_WINDOWS (Mutex backend split)
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
// TIMED WAITS (wait_for/wait_until): a bare
// (non-fiber) waiter parks with a bounded platform::Parker::park_timeout and
// retracts its own node on the deadline; the notify-vs-timeout race is arbitrated
// under the CV monitor lock via the node's `linked` flag (whoever acts first
// wins, the loser reports the other's outcome so no wake is lost). This covers
// external submitters and cross-process federation threads.
//
// Fiber mode uses the per-worker time wheel plus a per-episode temporary pin:
// notify and timer arbitrate through one atomic winner byte; the winning wake
// returns to the wheel owner, where a still-armed loser timer is cancelled O(1).
// This mirrors folly fibers Baton (atomic timeout-vs-post arbitration) without a
// cross-worker timer mutation.

namespace hyle::time {
class TimerEntry;
}

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
  // Bare threads use Parker::park_timeout. Fibers arm their current worker's
  // timer wheel and are temporarily pinned to that worker for this wait episode,
  // so the notify winner resumes on the wheel owner and cancels in O(1). An
  // atomic winner state prevents timer+notify from waking the same episode twice.
  enum class CvStatus { kNoTimeout, kTimeout };

  template <class Lock>
  CvStatus wait_for(Lock& lock, std::uint64_t timeout_ns) {
    Waiter node;
    const bool fiber = prepare_wait_(&node);
    lock.unlock();
    const CvStatus st = fiber ? finish_wait_timed_fiber_(&node, timeout_ns)
                              : finish_wait_timed_(&node, timeout_ns);
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
    // 0 waiting, 1 notify won, 2 timeout won. Used only by fiber timed waits.
    std::atomic<unsigned char> timed_state{0};
    bool fiber_timed = false;
  };
  struct TimedContext {
    ConditionVariable* cv;
    Waiter* node;
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
  HYLE_SCHED_API CvStatus finish_wait_timed_fiber_(
      Waiter* node, std::uint64_t timeout_ns) noexcept;
  HYLE_SCHED_API static void timed_fiber_fire_(
      hyle::time::TimerEntry* entry) noexcept;

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
//
// PLATFORM SPLIT (2026-07-03, measured): on Windows the contended lock delegates
// to the OS SRWLOCK -- the SAME primitive std::mutex/marl use, and measured 2.66x
// FASTER than a hand-rolled Drepper futex over WaitOnAddress under contention
// (SRWLOCK has kernel-assisted adaptive spinning; WaitOnAddress park is heavier,
// and a userspace pre-park spin can't be used over an M:N fiber scheduler -- it
// burns the worker thread the lock holder's fiber may be queued on). On Linux the
// Drepper futex over the futex syscall matches glibc (measured a statistical tie
// vs marl/std::mutex, so no OS-lock win to chase there). SRWLOCK is pointer-sized
// and zero-init (== SRWLOCK_INIT), stored opaquely here as void* and cast in the
// .cc so this header pulls in no <windows.h>.
//
// macOS falls in the #else (Drepper futex over os_sync_wait_on_address, which is
// correct + works). macOS has os_unfair_lock (its SRWLOCK-equivalent fast OS lock)
// -- a likely win there too, but UNMEASURED (no mac to bench), so not added: adding
// an untested platform branch would violate "write+verify". Revisit with a mac.
#if defined(HYLE_OS_WINDOWS)
class Mutex {
 public:
  Mutex() noexcept = default;
  HYLE_DISABLE_COPY_AND_MOVE(Mutex);

  // All out-of-line: SRWLOCK's own acquire/release ARE function calls (same as
  // std::mutex, which wraps AcquireSRWLockExclusive) -- so this is apples-to-apples.
  HYLE_SCHED_API void lock() noexcept;
  HYLE_SCHED_API bool try_lock() noexcept;
  HYLE_SCHED_API void unlock() noexcept;

 private:
  void* srw_ = nullptr;  // RTL_SRWLOCK{PVOID Ptr}; nullptr == SRWLOCK_INIT
};
#else
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
    // Publish unlocked in one RMW.  The old Drepper fetch_sub(2 -> 1) followed
    // by unlock_slow's store(0) performed two writes on every contended handoff;
    // exchange keeps the same 0/1/2 protocol with one cache-line transition.
    if (state_.exchange(0, std::memory_order_release) == 2) {
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
#endif

// ---- WaitGroup --------------------------------------------------------------
// Go-style "wait until N tasks have finished". add(n) before spawning, done()
// per finished task, wait() blocks (fiber-aware) until the count hits zero.
class WaitGroup {
 public:
  WaitGroup() noexcept = default;
  explicit WaitGroup(long initial) noexcept : count_(initial) {
    HYLE_FATAL_ASSERT(initial >= 0);
  }
  HYLE_DISABLE_COPY_AND_MOVE(WaitGroup);

  bool try_add(long n = 1) noexcept {
    // add is the uncommon control-plane operation and remains serialized with
    // wait's predicate check. done() is the data-plane operation: all non-final
    // completions avoid this lock.
    mu_.lock();
    const long count = count_.load(std::memory_order_relaxed);
    if (n < 0 || count > (std::numeric_limits<long>::max)() - n) {
      mu_.unlock();
      return false;
    }
    count_.store(count + n, std::memory_order_release);
    mu_.unlock();
    return true;
  }

  void add(long n = 1) noexcept { HYLE_FATAL_ASSERT(try_add(n)); }

  bool try_done() noexcept {
    long count = count_.load(std::memory_order_acquire);
    do {
      if (count <= 0) return false;
    } while (!count_.compare_exchange_weak(
        count, count - 1, std::memory_order_acq_rel,
        std::memory_order_acquire));
    if (count == 1) {
      // Pair with wait's predicate-check + waiter publication under mu_. Taking
      // the lock only on 1 -> 0 prevents a final completion from notifying in
      // the gap between the waiter's check and ConditionVariable::wait linking
      // its node, without charging every preceding done() for that lock.
      mu_.lock();
      mu_.unlock();
      cv_.notify_all();
    }
    return true;
  }

  void done() noexcept { HYLE_FATAL_ASSERT(try_done()); }

  void wait() noexcept {
    mu_.lock();
    while (count_.load(std::memory_order_acquire) > 0) cv_.wait(mu_);
    mu_.unlock();
  }

 private:
  util::SpinLock mu_;
  ConditionVariable cv_;
  std::atomic<long> count_{0};
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

  bool try_count_down(long n = 1) noexcept {
    long count = count_.load(std::memory_order_acquire);
    do {
      if (n < 0 || count < n) return false;
    } while (!count_.compare_exchange_weak(
        count, count - n, std::memory_order_acq_rel,
        std::memory_order_acquire));
    if (count == n) {
      mu_.lock();
      mu_.unlock();
      cv_.notify_all();
    }
    return true;
  }

  void count_down(long n = 1) noexcept {
    HYLE_FATAL_ASSERT(try_count_down(n));
  }

  bool try_wait() const noexcept {
    return count_.load(std::memory_order_acquire) == 0;
  }

  void wait() noexcept {
    mu_.lock();
    while (count_.load(std::memory_order_acquire) > 0) cv_.wait(mu_);
    mu_.unlock();
  }

 private:
  mutable util::SpinLock mu_;
  ConditionVariable cv_;
  std::atomic<long> count_;
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
  enum class SendStatus { kSent, kFullOrTimeout, kClosed };
  enum class RecvStatus { kValue, kEmptyOrTimeout, kClosed };

  Channel(mem::Allocator alloc, std::size_t capacity)
      : alloc_(alloc), cap_(capacity) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(capacity >= 1);
    HYLE_FATAL_ASSERT(
        capacity <= (std::numeric_limits<std::size_t>::max)() / sizeof(T));
    buf_ = static_cast<T*>(alloc_.allocate(sizeof(T) * cap_, alignof(T)));
  }

  ~Channel() {
    if (buf_ == nullptr) return;
    while (size_ > 0) {  // destroy anything still buffered
      buf_[head_].~T();
      head_ = (head_ + 1) % cap_;
      --size_;
    }
    alloc_.deallocate(buf_, sizeof(T) * cap_, alignof(T));
  }

  HYLE_DISABLE_COPY_AND_MOVE(Channel);

  // Allocation failure leaves a closed/empty invalid channel. Check this once
  // after construction; send/recv also fail safely so FFI factories can convert
  // OOM into their normal NULL + last-error channel.
  bool valid() const noexcept { return buf_ != nullptr; }

  // Enqueue `v`, blocking the fiber while the buffer is full. Returns false if
  // the channel is closed (the value is dropped).
  bool send(T v) {
    if (buf_ == nullptr) return false;
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

  SendStatus try_send(T v) {
    if (buf_ == nullptr) return SendStatus::kClosed;
    mu_.lock();
    if (closed_) {
      mu_.unlock();
      return SendStatus::kClosed;
    }
    if (size_ == cap_) {
      mu_.unlock();
      return SendStatus::kFullOrTimeout;
    }
    new (&buf_[tail_]) T(std::move(v));
    tail_ = (tail_ + 1) % cap_;
    ++size_;
    mu_.unlock();
    not_empty_.notify_one();
    return SendStatus::kSent;
  }

  SendStatus send_for(T v, std::uint64_t timeout_ns) {
    if (buf_ == nullptr) return SendStatus::kClosed;
    mu_.lock();
    const bool ready = not_full_.wait_for(
        mu_, timeout_ns, [&] { return size_ != cap_ || closed_; });
    if (closed_) {
      mu_.unlock();
      return SendStatus::kClosed;
    }
    if (!ready) {
      mu_.unlock();
      return SendStatus::kFullOrTimeout;
    }
    new (&buf_[tail_]) T(std::move(v));
    tail_ = (tail_ + 1) % cap_;
    ++size_;
    mu_.unlock();
    not_empty_.notify_one();
    return SendStatus::kSent;
  }

  // Dequeue into `out`, blocking the fiber while empty. Returns false once the
  // channel is closed AND drained (no value written).
  bool recv(T& out) {
    if (buf_ == nullptr) return false;
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

  RecvStatus try_recv(T& out) {
    if (buf_ == nullptr) return RecvStatus::kClosed;
    mu_.lock();
    if (size_ == 0) {
      const RecvStatus status =
          closed_ ? RecvStatus::kClosed : RecvStatus::kEmptyOrTimeout;
      mu_.unlock();
      return status;
    }
    out = std::move(buf_[head_]);
    buf_[head_].~T();
    head_ = (head_ + 1) % cap_;
    --size_;
    mu_.unlock();
    not_full_.notify_one();
    return RecvStatus::kValue;
  }

  RecvStatus recv_for(T& out, std::uint64_t timeout_ns) {
    if (buf_ == nullptr) return RecvStatus::kClosed;
    mu_.lock();
    const bool ready = not_empty_.wait_for(
        mu_, timeout_ns, [&] { return size_ != 0 || closed_; });
    if (size_ == 0) {
      const RecvStatus status =
          closed_ ? RecvStatus::kClosed : RecvStatus::kEmptyOrTimeout;
      mu_.unlock();
      return status;
    }
    HYLE_FATAL_ASSERT(ready);
    out = std::move(buf_[head_]);
    buf_[head_].~T();
    head_ = (head_ + 1) % cap_;
    --size_;
    mu_.unlock();
    not_full_.notify_one();
    return RecvStatus::kValue;
  }

  // Close the channel: wakes all blocked senders/receivers. Idempotent.
  void close() {
    if (buf_ == nullptr) return;
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

// ---- Semaphore --------------------------------------------------------------
// Fiber-aware counting semaphore: `permits` tokens; acquire(n) blocks (yields the
// fiber) until n are available, release(n) hands them back. The standard async
// rate-limiter (cap in-flight connections/requests, gate a buffer pool). Built on
// the CV foundation (marl's lesson: one wake mechanism, everything layers on it).
//
// SEMANTICS (tokio batch_semaphore, adopted): STRICT FIFO with NO head-of-line
// skip -- a waiter that wants n permits sits at the queue head and accumulates
// permits across release()s; releases NEVER skip past it to satisfy a smaller
// later waiter. This is what prevents large-request starvation (and is exactly
// why RwLock's writer -- which wants ALL permits -- can't be starved by a stream
// of readers). Tradeoff: deliberate head-of-line blocking, chosen for fairness.
//
// Each waiter owns its own CV on its stack, so release() wakes EXACTLY the head
// waiter it just satisfied (no thundering herd re-checking a shared predicate).
// The waiter list is intrusive + doubly linked so a (future) timed/cancelled
// acquire can unlink in O(1); the node lives on the blocking fiber/thread stack
// and stays linked only while parked (same discipline as ConditionVariable).
class Semaphore {
 public:
  explicit Semaphore(std::size_t permits) noexcept : permits_(permits) {}
  HYLE_DISABLE_COPY_AND_MOVE(Semaphore);

  // Acquire `n` permits, blocking (fiber-aware) until available. Returns false if
  // the semaphore was closed before n could be granted (permits already handed to
  // this waiter are returned to the pool). n may exceed the initial permit count
  // (it just waits for enough release()s).
  HYLE_SCHED_API bool acquire(std::size_t n = 1) noexcept;

  // Non-blocking: take `n` permits iff all are available right now. Returns false
  // if fewer than n are available OR the semaphore is closed. Never queues.
  HYLE_SCHED_API bool try_acquire(std::size_t n = 1) noexcept;

  // Return `n` permits to the pool, waking FIFO-front waiters whose full request
  // is thereby satisfied. Wakes are delivered after the internal lock is dropped.
  HYLE_SCHED_API bool try_release(std::size_t n = 1) noexcept;
  void release(std::size_t n = 1) noexcept {
    HYLE_FATAL_ASSERT(try_release(n));
  }

  // Close: wake every waiter (their acquire() returns false). Post-close acquire/
  // try_acquire return false. Idempotent. Permits already in the pool stay.
  HYLE_SCHED_API void close() noexcept;

  // Available permits right now (racy snapshot; for diagnostics/tests).
  HYLE_SCHED_API std::size_t available() const noexcept;

  // RAII permit: holds `n` permits and returns them to the pool on destruction.
  // The ergonomic limiter usage -- `auto p = sem.acquire_guard(); ...` frees the
  // slot automatically on scope exit, even on early return. owns() is false if the
  // acquire failed (semaphore closed).
  class Permit {
   public:
    Permit() noexcept = default;
    Permit(Semaphore* s, std::size_t n) noexcept : sem_(s), n_(n) {}
    Permit(Permit&& o) noexcept
        : sem_(std::exchange(o.sem_, nullptr)), n_(o.n_) {}
    Permit& operator=(Permit&& o) noexcept {
      if (this != &o) {
        reset();
        sem_ = std::exchange(o.sem_, nullptr);
        n_ = o.n_;
      }
      return *this;
    }
    HYLE_DISABLE_COPY(Permit);
    ~Permit() { reset(); }
    bool owns() const noexcept { return sem_ != nullptr; }

    // Consume the permit WITHOUT returning it: permanently removes `n` permits from
    // the semaphore (tokio SemaphorePermit::forget) -- used to shrink capacity.
    void forget() noexcept { sem_ = nullptr; }

   private:
    void reset() noexcept {
      if (sem_ != nullptr) sem_->release(n_);
      sem_ = nullptr;
    }
    Semaphore* sem_ = nullptr;
    std::size_t n_ = 0;
  };

  // Blocking acquire returning an RAII Permit (owns()==false if closed).
  Permit acquire_guard(std::size_t n = 1) noexcept {
    return acquire(n) ? Permit(this, n) : Permit();
  }
  // Non-blocking acquire returning an RAII Permit (owns()==false if unavailable).
  Permit try_acquire_guard(std::size_t n = 1) noexcept {
    return try_acquire(n) ? Permit(this, n) : Permit();
  }

 private:
  // One queued acquirer. `need` is how many permits it still wants; `granted` +
  // its own CV are how release() hands control back to exactly this waiter.
  struct Waiter {
    Waiter* next = nullptr;
    Waiter* prev = nullptr;
    std::size_t need = 0;    // permits still required before this waiter proceeds
    bool granted = false;    // set by release(): request fully satisfied
    bool aborted = false;    // set by close(): give up, acquire() returns false
    ConditionVariable cv;    // this waiter blocks here; woken 1:1
  };

  mutable util::SpinLock mu_;
  std::size_t permits_;
  Waiter* head_ = nullptr;   // FIFO front: release satisfies this one first
  Waiter* tail_ = nullptr;
  bool closed_ = false;
};

// ---- RwLock -----------------------------------------------------------------
// Fiber-aware readers-writer lock, WRITE-PREFERRING (no writer starvation). Built
// directly on Semaphore, tokio's design: the lock IS a Semaphore initialized with
// kMaxReaders permits; a read holds 1 permit, a write holds ALL of them. Because
// Semaphore never skips its FIFO head, a queued writer (wanting all permits) makes
// subsequent readers (wanting 1) queue BEHIND it instead of stealing the lock --
// fairness for free, no hand-rolled reader/writer counters.
//
// CONTRACT (same as std::shared_mutex / tokio): do not acquire a write lock while
// holding a read lock on the same RwLock from the same fiber (self-deadlock -- the
// write waits for all permits including the one your read holds). Guards are RAII;
// unlock on scope exit.
class RwLock {
 public:
  RwLock() noexcept : sem_(kMaxReaders) {}
  HYLE_DISABLE_COPY_AND_MOVE(RwLock);

  // RAII read guard: holds 1 permit; releases it on destruction.
  class ReadGuard {
   public:
    ReadGuard() noexcept = default;
    explicit ReadGuard(RwLock* rw) noexcept : rw_(rw) {}
    ReadGuard(ReadGuard&& o) noexcept : rw_(std::exchange(o.rw_, nullptr)) {}
    ReadGuard& operator=(ReadGuard&& o) noexcept {
      if (this != &o) {
        reset();
        rw_ = std::exchange(o.rw_, nullptr);
      }
      return *this;
    }
    HYLE_DISABLE_COPY(ReadGuard);
    ~ReadGuard() { reset(); }
    bool owns_lock() const noexcept { return rw_ != nullptr; }

   private:
    void reset() noexcept {
      if (rw_ != nullptr) rw_->sem_.release(1);
      rw_ = nullptr;
    }
    RwLock* rw_ = nullptr;
  };

  // RAII write guard: holds all kMaxReaders permits; releases them on destruction.
  class WriteGuard {
   public:
    WriteGuard() noexcept = default;
    explicit WriteGuard(RwLock* rw) noexcept : rw_(rw) {}
    WriteGuard(WriteGuard&& o) noexcept : rw_(std::exchange(o.rw_, nullptr)) {}
    WriteGuard& operator=(WriteGuard&& o) noexcept {
      if (this != &o) {
        reset();
        rw_ = std::exchange(o.rw_, nullptr);
      }
      return *this;
    }
    HYLE_DISABLE_COPY(WriteGuard);
    ~WriteGuard() { reset(); }
    bool owns_lock() const noexcept { return rw_ != nullptr; }

    // Atomically downgrade to a read lock: release all but ONE permit, keeping the
    // last one so the lock is NEVER fully free during the transition -- a queued
    // writer (which needs ALL permits) therefore cannot slip in between, so no
    // other writer can observe a gap. The retained permit becomes the read guard.
    // Consumes this WriteGuard (owns_lock() becomes false). std::shared_mutex has
    // no downgrade; tokio's RwLockWriteGuard::downgrade is the model.
    ReadGuard downgrade() noexcept {
      HYLE_FATAL_ASSERT(rw_ != nullptr);
      RwLock* rw = rw_;
      rw_ = nullptr;                    // this write guard no longer owns anything
      rw->sem_.release(kMaxReaders - 1);  // drop all but one -> now a single reader
      return ReadGuard(rw);
    }

   private:
    void reset() noexcept {
      if (rw_ != nullptr) rw_->sem_.release(kMaxReaders);
      rw_ = nullptr;
    }
    RwLock* rw_ = nullptr;
  };

  // Block (fiber-aware) until the lock is held; return an RAII guard.
  ReadGuard read() noexcept {
    lock_read();
    return ReadGuard(this);
  }
  WriteGuard write() noexcept {
    lock_write();
    return WriteGuard(this);
  }

  // Non-blocking. The guard's owns_lock() is false if the lock wasn't taken.
  ReadGuard try_read() noexcept {
    return sem_.try_acquire(1) ? ReadGuard(this) : ReadGuard();
  }
  WriteGuard try_write() noexcept {
    return sem_.try_acquire(kMaxReaders) ? WriteGuard(this) : WriteGuard();
  }

  // Raw lock/unlock pairs for C ABI and foreign-language RAII equivalents.
  void lock_read() noexcept { (void)sem_.acquire(1); }
  bool try_lock_read() noexcept { return sem_.try_acquire(1); }
  void unlock_read() noexcept { sem_.release(1); }
  void lock_write() noexcept { (void)sem_.acquire(kMaxReaders); }
  bool try_lock_write() noexcept { return sem_.try_acquire(kMaxReaders); }
  void unlock_write() noexcept { sem_.release(kMaxReaders); }

 private:
  // Max concurrent readers. Matches tokio's u32::MAX>>3 headroom idea, sized to
  // std::size_t here; a write acquires all of them for exclusivity.
  static constexpr std::size_t kMaxReaders = (static_cast<std::size_t>(1) << 32);
  Semaphore sem_;
};

// ---- oneshot ----------------------------------------------------------------
// Single-value, single-producer single-consumer channel: a Sender sends ONE value
// (once), a Receiver receives it (once). The canonical "spawn a task, get its one
// result back" primitive (pairs with JoinHandle) and a request/response reply
// slot. Move-only Sender/Receiver over a refcounted shared state (charter: from an
// injected Allocator, no hidden heap). Fiber-aware: recv() yields the fiber until
// the value arrives or the Sender is dropped.
//
// SEMANTICS (tokio oneshot, adopted):
//  - send(v) consumes the Sender; recv() consumes the Receiver (single-shot).
//  - Sender dropped without send -> recv() returns Cancelled (the sender is gone).
//  - Receiver dropped -> send() returns the value back in an error (nobody home).
//  - Sender::is_closed() lets a producer learn the Receiver went away and abort
//    expensive work before computing the value (the cancellation hook).
namespace detail {

template <class T>
struct OneshotState {
  mem::Allocator alloc;
  util::SpinLock mu;
  ConditionVariable cv;              // receiver blocks here for value/close
  ConditionVariable closed_cv;       // sender blocks here in closed() (rx gone)
  std::atomic<int> refcount{2};      // Sender + Receiver
  bool value_set = false;            // a value is stored (sender did send)
  bool rx_alive = true;              // Receiver not yet dropped (set under mu)
  bool tx_alive = true;              // Sender not yet dropped (set under mu) --
                                     // the close signal recv() waits on. Must be
                                     // flipped UNDER mu (not inferred from the
                                     // lock-free refcount): a waiter re-checks the
                                     // predicate under mu, so the state change that
                                     // frees it has to serialize with that check,
                                     // or the notify races ahead of the flag and
                                     // the receiver parks forever (lost wakeup).
  alignas(T) unsigned char storage[sizeof(T)];

  void release() noexcept {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (value_set) reinterpret_cast<T*>(storage)->~T();  // unreceived value
      mem::Allocator a = alloc;
      this->~OneshotState();
      a.deallocate(this, sizeof(OneshotState), alignof(OneshotState));
    }
  }
};

}  // namespace detail

template <class T>
class OneshotReceiver;

template <class T>
class OneshotSender {
 public:
  OneshotSender() noexcept = default;
  explicit OneshotSender(detail::OneshotState<T>* s) noexcept : state_(s) {}
  OneshotSender(OneshotSender&& o) noexcept : state_(std::exchange(o.state_, nullptr)) {}
  OneshotSender& operator=(OneshotSender&& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = std::exchange(o.state_, nullptr);
    }
    return *this;
  }
  HYLE_DISABLE_COPY(OneshotSender);
  ~OneshotSender() { reset_(); }

  // Send the one value. Returns the value back (as an error payload) if the
  // Receiver has already been dropped. Consumes the send capability (state_ is
  // released after), so a second send is impossible.
  Result<void> send(T v) noexcept {
    detail::OneshotState<T>* s = state_;
    state_ = nullptr;  // consume: this Sender is spent regardless of outcome
    s->mu.lock();
    if (!s->rx_alive) {
      s->mu.unlock();
      s->release();  // our ref; receiver already gone
      return std::unexpected(Status::Cancelled("oneshot receiver dropped"));
    }
    new (s->storage) T(std::move(v));
    s->value_set = true;
    s->tx_alive = false;  // the send capability is consumed successfully
    s->mu.unlock();
    s->cv.notify_one();  // wake a blocked receiver
    s->release();
    return {};
  }

  // True once the Receiver has been dropped (cancellation hook: stop computing).
  bool is_closed() const noexcept {
    state_->mu.lock();
    const bool closed = !state_->rx_alive;
    state_->mu.unlock();
    return closed;
  }

  // Block (fiber-aware) until the Receiver is dropped -- the async cancellation
  // hook: a producer awaits this and, when it fires, abandons an expensive
  // computation whose result nobody will read. Returns immediately if the
  // Receiver is already gone. (Separate CV from recv()'s: sender waits for "rx
  // gone", receiver waits for "value/tx gone" -- tokio's two independent slots.)
  void closed() noexcept {
    state_->mu.lock();
    state_->closed_cv.wait(state_->mu, [&] { return !state_->rx_alive; });
    state_->mu.unlock();
  }

 private:
  void reset_() noexcept {
    if (state_ != nullptr) {
      // Dropped without sending: flip the close flag UNDER mu (so a receiver
      // re-checking the predicate under mu sees it -- flipping it via the
      // lock-free refcount would race the notify and lose the wakeup), THEN notify.
      state_->mu.lock();
      state_->tx_alive = false;
      state_->mu.unlock();
      state_->cv.notify_one();  // wake the receiver so recv() returns Cancelled
      state_->release();
      state_ = nullptr;
    }
  }
  detail::OneshotState<T>* state_ = nullptr;
};

template <class T>
class OneshotReceiver {
 public:
  OneshotReceiver() noexcept = default;
  explicit OneshotReceiver(detail::OneshotState<T>* s) noexcept : state_(s) {}
  OneshotReceiver(OneshotReceiver&& o) noexcept : state_(std::exchange(o.state_, nullptr)) {}
  OneshotReceiver& operator=(OneshotReceiver&& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = std::exchange(o.state_, nullptr);
    }
    return *this;
  }
  HYLE_DISABLE_COPY(OneshotReceiver);
  ~OneshotReceiver() { reset_(); }

  // Block (fiber-aware) until the value arrives or the Sender is dropped. Consumes
  // the receive capability. Returns the value, or Cancelled if the sender dropped
  // without sending.
  Result<T> recv() noexcept {
    detail::OneshotState<T>* s = state_;
    state_ = nullptr;
    s->mu.lock();
    // Wait until a value is set OR the sender dropped without sending. Both flags
    // are written under mu (tx_alive by the sender's reset_, value_set by send),
    // so the predicate re-check under mu can't miss either -- no lost wakeup.
    s->cv.wait(s->mu, [&] { return s->value_set || !s->tx_alive; });
    if (s->value_set) {
      T* p = reinterpret_cast<T*>(s->storage);
      T out = std::move(*p);
      p->~T();
      s->value_set = false;  // consumed; release() won't double-destroy
      s->mu.unlock();
      s->release();
      return out;
    }
    s->mu.unlock();
    s->release();
    return std::unexpected(Status::Cancelled("oneshot sender dropped"));
  }

  // Non-blocking: value if present, else NotFound (empty) / Cancelled (sender gone).
  Result<T> try_recv() noexcept {
    state_->mu.lock();
    if (state_->value_set) {
      T* p = reinterpret_cast<T*>(state_->storage);
      T out = std::move(*p);
      p->~T();
      state_->value_set = false;
      state_->mu.unlock();
      return out;
    }
    const bool sender_gone = !state_->tx_alive;  // flipped under mu by sender drop
    state_->mu.unlock();
    return std::unexpected(sender_gone ? Status::Cancelled("oneshot sender dropped")
                                       : Status::NotFound("oneshot empty"));
  }

 private:
  void reset_() noexcept {
    if (state_ != nullptr) {
      state_->mu.lock();
      state_->rx_alive = false;  // tell the sender (is_closed/closed) nobody's home
      state_->mu.unlock();
      state_->closed_cv.notify_all();  // wake a sender blocked in closed()
      state_->release();
      state_ = nullptr;
    }
  }
  detail::OneshotState<T>* state_ = nullptr;
};

// Create a linked oneshot Sender/Receiver pair over allocator-backed shared state.
template <class T>
Result<std::pair<OneshotSender<T>, OneshotReceiver<T>>> try_make_oneshot(
    mem::Allocator alloc) noexcept {
  using detail::OneshotState;
  if (!alloc.valid())
    return std::unexpected(
        Status::InvalidArgument("oneshot requires a complete allocator"));
  auto* s = static_cast<OneshotState<T>*>(
      alloc.allocate(sizeof(OneshotState<T>), alignof(OneshotState<T>)));
  if (s == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("oneshot state allocation failed"));
  new (s) OneshotState<T>{};
  s->alloc = alloc;
  return std::pair<OneshotSender<T>, OneshotReceiver<T>>{
      OneshotSender<T>(s), OneshotReceiver<T>(s)};
}

template <class T>
std::pair<OneshotSender<T>, OneshotReceiver<T>> make_oneshot(
    mem::Allocator alloc) noexcept {
  auto result = try_make_oneshot<T>(alloc);
  HYLE_FATAL_ASSERT(result.has_value());
  return std::move(*result);
}

// ---- watch ------------------------------------------------------------------
// Latest-value broadcast: a Sender publishes a value that every Receiver can
// observe; Receivers see only the MOST RECENT value (no accumulation), and can
// block in changed() until it moves. The canonical config-hot-reload / shutdown-
// signal primitive: a late subscriber immediately sees the current value, and
// changed() wakes every waiter on each update. Distinct from a Channel (which
// queues every item) -- watch keeps exactly one cell + a version counter.
//
// SEMANTICS (tokio watch, adopted):
//  - Receivers are COPYABLE (subscribe many); each caches the version it has seen.
//  - changed() blocks until the stored version != the receiver's seen version,
//    then updates the seen version. Uses notify-before-recheck ordering (the CV
//    predicate re-reads version under the lock) so an update can't be missed.
//  - borrow() returns a COPY of the current value under the lock (no long-held Ref
//    across a suspend point -- that would block the sender; we copy instead).
//  - close (all Senders dropped) makes changed() return Cancelled.
namespace detail {

template <class T>
struct WatchState {
  mem::Allocator alloc;
  util::SpinLock mu;
  ConditionVariable cv;
  std::atomic<int> refcount{0};   // senders + receivers; last drop frees
  std::atomic<int> senders{0};    // >0 while any Sender alive (handle bookkeeping)
  std::atomic<int> receivers{0};  // live Receiver count (send() checks for 0)
  bool closed = false;            // all senders gone; set UNDER mu (changed()'s
                                  // close predicate -- flipping via the lock-free
                                  // senders count would race the notify and lose
                                  // the wakeup, so the flag is mu-guarded)
  std::uint64_t version = 0;      // bumped on each send; receivers track it
  alignas(T) unsigned char storage[sizeof(T)];  // the single current value

  T& value() noexcept { return *reinterpret_cast<T*>(storage); }
  void add_ref() noexcept { refcount.fetch_add(1, std::memory_order_relaxed); }
  // A new Receiver handle: bump BOTH the total refcount and the receiver count.
  void add_receiver() noexcept {
    refcount.fetch_add(1, std::memory_order_relaxed);
    receivers.fetch_add(1, std::memory_order_relaxed);
  }
  void release() noexcept {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      value().~T();
      mem::Allocator a = alloc;
      this->~WatchState();
      a.deallocate(this, sizeof(WatchState), alignof(WatchState));
    }
  }
};

}  // namespace detail

template <class T>
class WatchReceiver {
 public:
  WatchReceiver() noexcept = default;
  WatchReceiver(detail::WatchState<T>* s, std::uint64_t seen) noexcept
      : state_(s), seen_(seen) {}
  WatchReceiver(const WatchReceiver& o) noexcept
      : state_(o.state_), seen_(o.seen_) {
    if (state_) state_->add_receiver();  // another subscriber
  }
  WatchReceiver& operator=(const WatchReceiver& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = o.state_;
      seen_ = o.seen_;
      if (state_) state_->add_receiver();
    }
    return *this;
  }
  WatchReceiver(WatchReceiver&& o) noexcept
      : state_(std::exchange(o.state_, nullptr)), seen_(o.seen_) {}
  WatchReceiver& operator=(WatchReceiver&& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = std::exchange(o.state_, nullptr);
      seen_ = o.seen_;
    }
    return *this;
  }
  ~WatchReceiver() { reset_(); }

  // A COPY of the current value (does not mark it seen -- use changed() to consume
  // the update signal). Cheap to call repeatedly (config read on every request).
  T borrow() const noexcept {
    state_->mu.lock();
    T v = state_->value();  // copy under the lock
    state_->mu.unlock();
    return v;
  }

  // A COPY of the current value AND mark this version seen (so the next changed()
  // waits for the NEXT update). Use this -- not borrow() -- inside a changed()
  // loop: it aligns `seen_` to the version observed under the lock in the same
  // critical section as the read, so a send racing between changed() returning and
  // this read can't make you process the same logical value twice (tokio's
  // borrow_and_update contract).
  T borrow_and_update() noexcept {
    state_->mu.lock();
    T v = state_->value();
    seen_ = state_->version;  // align seen to the version we just read
    state_->mu.unlock();
    return v;
  }

  // Block (fiber-aware) until the value changes since this receiver last saw it,
  // then mark it seen. Returns Cancelled if all Senders have dropped (no more
  // updates will ever come). A pending change already present returns at once.
  Result<void> changed() noexcept {
    state_->mu.lock();
    // version and closed are both written under mu (send bumps version, the last
    // sender's reset_ sets closed) -> the predicate re-check can't miss either.
    state_->cv.wait(state_->mu, [&] {
      return state_->version != seen_ || state_->closed;
    });
    const bool changed = (state_->version != seen_);
    seen_ = state_->version;  // mark seen (even on close, so we don't spin)
    state_->mu.unlock();
    if (changed) return {};
    return std::unexpected(Status::Cancelled("watch senders dropped"));
  }

  // True if the value has changed since last seen (non-blocking peek).
  bool has_changed() const noexcept {
    state_->mu.lock();
    const bool c = (state_->version != seen_);
    state_->mu.unlock();
    return c;
  }

 private:
  void reset_() noexcept {
    if (state_ != nullptr) {
      // Last receiver gone -> wake a sender blocked in closed() (via notify on the
      // shared cv, whose closed() predicate re-checks receivers==0).
      if (state_->receivers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        state_->cv.notify_all();
      }
      state_->release();  // (receivers already decremented above)
      state_ = nullptr;
    }
  }
  detail::WatchState<T>* state_ = nullptr;
  std::uint64_t seen_ = 0;
};

template <class T>
class WatchSender {
 public:
  WatchSender() noexcept = default;
  explicit WatchSender(detail::WatchState<T>* s) noexcept : state_(s) {}
  WatchSender(const WatchSender& o) noexcept : state_(o.state_) {
    if (state_) {
      state_->add_ref();
      state_->senders.fetch_add(1, std::memory_order_relaxed);
    }
  }
  WatchSender& operator=(const WatchSender& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = o.state_;
      if (state_) {
        state_->add_ref();
        state_->senders.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return *this;
  }
  WatchSender(WatchSender&& o) noexcept : state_(std::exchange(o.state_, nullptr)) {}
  WatchSender& operator=(WatchSender&& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = std::exchange(o.state_, nullptr);
    }
    return *this;
  }
  ~WatchSender() { reset_(); }

  // Publish a new value TO CONSUMERS: replace the cell, bump the version, wake
  // every waiter. Returns Cancelled if there are no receivers (the value is still
  // stored, but nobody is listening -- the "deliver to a consumer" semantics). Use
  // send_replace/send_modify to update the shared state regardless of listeners.
  Result<void> send(T v) noexcept {
    if (state_->receivers.load(std::memory_order_acquire) == 0)
      return std::unexpected(Status::Cancelled("watch has no receivers"));
    state_->mu.lock();
    state_->value() = std::move(v);
    ++state_->version;  // under mu_: a receiver seeing the new version reads this v
    state_->mu.unlock();
    state_->cv.notify_all();
    return {};
  }

  // Replace the value unconditionally (even with no receivers -- watch as a shared
  // "latest state" register) and RETURN THE OLD VALUE. Always bumps + notifies.
  T send_replace(T v) noexcept {
    state_->mu.lock();
    T old = std::move(state_->value());
    state_->value() = std::move(v);
    ++state_->version;
    state_->mu.unlock();
    state_->cv.notify_all();
    return old;
  }

  // Apply `f` to the value in place under the lock, then unconditionally bump the
  // version + notify. For mutate-in-place updates (avoids a copy). f: void(T&).
  template <class F>
  void send_modify(F&& f) noexcept {
    state_->mu.lock();
    f(state_->value());
    ++state_->version;
    state_->mu.unlock();
    state_->cv.notify_all();
  }

  // Apply `f` to the value in place; f returns whether it changed anything. Only
  // if it returns true do we bump the version + notify (a "silent" false edit is
  // visible via borrow() but wakes no waiter). f: bool(T&).
  template <class F>
  void send_if_modified(F&& f) noexcept {
    state_->mu.lock();
    const bool changed = f(state_->value());
    if (changed) ++state_->version;
    state_->mu.unlock();
    if (changed) state_->cv.notify_all();
  }

  // A COPY of the current value (sender-side peek).
  T borrow() const noexcept {
    state_->mu.lock();
    T v = state_->value();
    state_->mu.unlock();
    return v;
  }

  // How many Receivers currently exist (a hint: races with concurrent drops).
  std::size_t receiver_count() const noexcept {
    return static_cast<std::size_t>(
        state_->receivers.load(std::memory_order_acquire));
  }

  // Subscribe a new Receiver that has ALREADY SEEN the current value (so its next
  // changed() waits for the next send, not the current value). Matches tokio.
  WatchReceiver<T> subscribe() const noexcept {
    state_->mu.lock();
    const std::uint64_t v = state_->version;
    state_->mu.unlock();
    state_->add_receiver();
    return WatchReceiver<T>(state_, v);
  }

 private:
  void reset_() noexcept {
    if (state_ != nullptr) {
      // Decide "last sender" and flip `closed` UNDER mu, so a receiver re-checking
      // changed()'s predicate under mu observes closed before it re-parks (flipping
      // via the lock-free senders count alone would let the notify race ahead of
      // the flag -> lost wakeup -> permanent hang after all senders gone).
      state_->mu.lock();
      const bool last = (state_->senders.fetch_sub(1, std::memory_order_acq_rel) == 1);
      if (last) state_->closed = true;
      state_->mu.unlock();
      if (last) state_->cv.notify_all();  // wake receivers -> changed() returns Cancelled
      state_->release();
      state_ = nullptr;
    }
  }
  detail::WatchState<T>* state_ = nullptr;
};

// Create a watch channel seeded with `initial`. The returned Receiver has NOT yet
// seen a change (its changed() waits for the first send after construction), but
// borrow() returns `initial` immediately -- so a late subscriber / shutdown flag
// reads the current state without waiting.
template <class T>
Result<std::pair<WatchSender<T>, WatchReceiver<T>>> try_make_watch(
    mem::Allocator alloc, T initial) noexcept {
  using detail::WatchState;
  if (!alloc.valid())
    return std::unexpected(
        Status::InvalidArgument("watch requires a complete allocator"));
  auto* s = static_cast<WatchState<T>*>(
      alloc.allocate(sizeof(WatchState<T>), alignof(WatchState<T>)));
  if (s == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("watch state allocation failed"));
  new (s) WatchState<T>{};
  s->alloc = alloc;
  new (s->storage) T(std::move(initial));
  s->refcount.store(2, std::memory_order_relaxed);   // 1 sender + 1 receiver
  s->senders.store(1, std::memory_order_relaxed);
  s->receivers.store(1, std::memory_order_relaxed);  // the returned Receiver
  return std::pair<WatchSender<T>, WatchReceiver<T>>{
      WatchSender<T>(s), WatchReceiver<T>(s, /*seen=*/0)};
}

template <class T>
std::pair<WatchSender<T>, WatchReceiver<T>> make_watch(mem::Allocator alloc,
                                                       T initial) noexcept {
  auto result = try_make_watch<T>(alloc, std::move(initial));
  HYLE_FATAL_ASSERT(result.has_value());
  return std::move(*result);
}

// ---- broadcast --------------------------------------------------------------
// Multi-consumer fan-out: every Receiver observes EVERY value a Sender sends
// (unlike watch, which keeps only the latest). A bounded ring holds the last
// `capacity` values; a slow Receiver that falls more than `capacity` behind gets
// a Lagged error naming how many it missed, then resumes at the oldest retained
// value (tokio's overwrite-oldest policy -- senders NEVER block on a slow reader).
//
// SEMANTICS (tokio broadcast, adopted, simplified to one lock):
//  - Receivers COPYABLE (subscribe many); each has its own monotonic read cursor.
//  - send() overwrites the oldest slot, bumps the write position, wakes all
//    receiver waiters. Never blocks.
//  - recv() blocks (fiber-aware) until its cursor < write position; if the cursor
//    fell behind the retained window it returns Lagged and fast-forwards.
//  - New subscribers start at the current write position (future messages only).
//  - All Senders dropped + ring drained -> recv() returns Cancelled (closed).
//
// One SpinLock guards the whole ring (not tokio's per-slot mutex): correct and
// simple; the per-slot-lock throughput refinement is deferred until a real
// high-fanout consumer needs it (charter: no speculative complexity).
namespace detail {

template <class T>
struct BroadcastState {
  mem::Allocator alloc;
  util::SpinLock mu;
  ConditionVariable cv;            // receivers block here for the next value
  std::atomic<int> refcount{0};    // senders + receivers; last drop frees
  std::atomic<int> senders{0};     // >0 while any Sender alive (handle bookkeeping)
  std::atomic<int> receivers{0};   // live Receiver count (for receiver_count())
  bool closed = false;             // all senders gone; set UNDER mu (recv()'s close
                                   // predicate -- mu-guarded, not inferred from the
                                   // lock-free senders count, to avoid lost wakeup)
  std::size_t cap;                 // ring capacity (number of retained values)
  std::uint64_t write_pos = 0;     // monotonic id of the NEXT value to be written
  T* buf = nullptr;                // ring of `cap` slots (index = pos % cap)
  bool* occupied = nullptr;        // whether slot i currently holds a live T

  void add_ref() noexcept { refcount.fetch_add(1, std::memory_order_relaxed); }
  // A new Receiver handle: bump both the total refcount and the receiver count.
  void add_receiver() noexcept {
    refcount.fetch_add(1, std::memory_order_relaxed);
    receivers.fetch_add(1, std::memory_order_relaxed);
  }
  void release() noexcept {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      for (std::size_t i = 0; i < cap; ++i)
        if (occupied[i]) buf[i].~T();
      mem::Allocator a = alloc;
      const std::size_t c = cap;
      T* b = buf;
      bool* o = occupied;
      this->~BroadcastState();
      a.deallocate(b, sizeof(T) * c, alignof(T));
      a.deallocate(o, sizeof(bool) * c, alignof(bool));
      a.deallocate(this, sizeof(BroadcastState), alignof(BroadcastState));
    }
  }
};

}  // namespace detail

template <class T>
class BroadcastReceiver {
 public:
  BroadcastReceiver() noexcept = default;
  BroadcastReceiver(detail::BroadcastState<T>* s, std::uint64_t cursor) noexcept
      : state_(s), cursor_(cursor) {}
  BroadcastReceiver(const BroadcastReceiver& o) noexcept
      : state_(o.state_), cursor_(o.cursor_) {
    if (state_) state_->add_receiver();
  }
  BroadcastReceiver& operator=(const BroadcastReceiver& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = o.state_;
      cursor_ = o.cursor_;
      if (state_) state_->add_receiver();
    }
    return *this;
  }
  BroadcastReceiver(BroadcastReceiver&& o) noexcept
      : state_(std::exchange(o.state_, nullptr)), cursor_(o.cursor_) {}
  BroadcastReceiver& operator=(BroadcastReceiver&& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = std::exchange(o.state_, nullptr);
      cursor_ = o.cursor_;
    }
    return *this;
  }
  ~BroadcastReceiver() { reset_(); }

  // Block (fiber-aware) until a value at/after this receiver's cursor is available.
  // Returns:
  //  - the value (cursor advances by 1);
  //  - Unavailable("lagged N") if the cursor fell behind the retained window (the
  //    cursor fast-forwards to the oldest retained value; retry recv() to get it);
  //  - Cancelled if all senders dropped AND nothing remains to read.
  Result<T> recv() noexcept {
    auto* s = state_;
    s->mu.lock();
    // write_pos (send) and closed (last sender's reset_) are both written under mu,
    // so the predicate re-check can't miss a value or the close -> no lost wakeup.
    s->cv.wait(s->mu, [&] { return cursor_ < s->write_pos || s->closed; });
    // Closed and fully drained.
    if (cursor_ >= s->write_pos) {
      s->mu.unlock();
      return std::unexpected(Status::Cancelled("broadcast closed"));
    }
    // Lag check: the oldest retained value has pos == write_pos - cap (once the
    // ring has wrapped). If our cursor is older than that, we missed some.
    const std::uint64_t oldest =
        s->write_pos > s->cap ? s->write_pos - s->cap : 0;
    if (cursor_ < oldest) {
      cursor_ = oldest;  // fast-forward to the oldest still-retained value
      s->mu.unlock();
      return std::unexpected(Status::Unavailable("broadcast lagged"));
    }
    T out = s->buf[cursor_ % s->cap];  // copy the value for this receiver
    ++cursor_;
    s->mu.unlock();
    return out;
  }

  // Non-blocking counterpart to recv(): NotFound means no new value yet,
  // Unavailable means lagged (cursor fast-forwards), and Cancelled means closed
  // and drained.
  Result<T> try_recv() noexcept {
    auto* s = state_;
    s->mu.lock();
    if (cursor_ >= s->write_pos) {
      const bool closed = s->closed;
      s->mu.unlock();
      return std::unexpected(
          closed ? Status::Cancelled("broadcast closed")
                 : Status::NotFound("broadcast empty"));
    }
    const std::uint64_t oldest =
        s->write_pos > s->cap ? s->write_pos - s->cap : 0;
    if (cursor_ < oldest) {
      cursor_ = oldest;
      s->mu.unlock();
      return std::unexpected(Status::Unavailable("broadcast lagged"));
    }
    T out = s->buf[cursor_ % s->cap];
    ++cursor_;
    s->mu.unlock();
    return out;
  }

 private:
  void reset_() noexcept {
    if (state_ != nullptr) {
      state_->receivers.fetch_sub(1, std::memory_order_acq_rel);
      state_->release();
      state_ = nullptr;
    }
  }
  detail::BroadcastState<T>* state_ = nullptr;
  std::uint64_t cursor_ = 0;
};

template <class T>
class BroadcastSender {
 public:
  BroadcastSender() noexcept = default;
  explicit BroadcastSender(detail::BroadcastState<T>* s) noexcept : state_(s) {}
  BroadcastSender(const BroadcastSender& o) noexcept : state_(o.state_) {
    if (state_) {
      state_->add_ref();
      state_->senders.fetch_add(1, std::memory_order_relaxed);
    }
  }
  BroadcastSender& operator=(const BroadcastSender& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = o.state_;
      if (state_) {
        state_->add_ref();
        state_->senders.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return *this;
  }
  BroadcastSender(BroadcastSender&& o) noexcept
      : state_(std::exchange(o.state_, nullptr)) {}
  BroadcastSender& operator=(BroadcastSender&& o) noexcept {
    if (this != &o) {
      reset_();
      state_ = std::exchange(o.state_, nullptr);
    }
    return *this;
  }
  ~BroadcastSender() { reset_(); }

  // Publish `v` to all receivers (overwrites the oldest slot if the ring is full;
  // never blocks). Wakes every waiting receiver.
  void send(T v) noexcept {
    state_->mu.lock();
    const std::size_t idx = state_->write_pos % state_->cap;
    if (state_->occupied[idx]) {
      state_->buf[idx] = std::move(v);  // overwrite oldest
    } else {
      new (&state_->buf[idx]) T(std::move(v));
      state_->occupied[idx] = true;
    }
    ++state_->write_pos;
    state_->mu.unlock();
    state_->cv.notify_all();
  }

  // Subscribe a new Receiver starting at the current write position (future
  // messages only -- it does not replay history already in the ring).
  BroadcastReceiver<T> subscribe() const noexcept {
    state_->mu.lock();
    const std::uint64_t pos = state_->write_pos;
    state_->mu.unlock();
    state_->add_receiver();
    return BroadcastReceiver<T>(state_, pos);
  }

  // Live handle counts (hints: race with concurrent clone/drop).
  std::size_t receiver_count() const noexcept {
    return static_cast<std::size_t>(
        state_->receivers.load(std::memory_order_acquire));
  }
  std::size_t sender_count() const noexcept {
    return static_cast<std::size_t>(
        state_->senders.load(std::memory_order_acquire));
  }

 private:
  void reset_() noexcept {
    if (state_ != nullptr) {
      // Flip `closed` UNDER mu when the last sender leaves (see recv()'s predicate)
      // so a blocked receiver observes it before re-parking -- no lost wakeup.
      state_->mu.lock();
      const bool last = (state_->senders.fetch_sub(1, std::memory_order_acq_rel) == 1);
      if (last) state_->closed = true;
      state_->mu.unlock();
      if (last) state_->cv.notify_all();  // unblock receivers -> recv() returns closed
      state_->release();
      state_ = nullptr;
    }
  }
  detail::BroadcastState<T>* state_ = nullptr;
};

// Create a broadcast channel retaining the last `capacity` values. Returns a
// Sender + one Receiver (subscribe() more from the Sender). capacity >= 1.
template <class T>
Result<std::pair<BroadcastSender<T>, BroadcastReceiver<T>>> try_make_broadcast(
    mem::Allocator alloc, std::size_t capacity) noexcept {
  using detail::BroadcastState;
  if (!alloc.valid())
    return std::unexpected(
        Status::InvalidArgument("broadcast requires a complete allocator"));
  if (capacity == 0 ||
      capacity > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
    return std::unexpected(
        Status::InvalidArgument("invalid broadcast capacity"));
  auto* s = static_cast<BroadcastState<T>*>(
      alloc.allocate(sizeof(BroadcastState<T>), alignof(BroadcastState<T>)));
  if (s == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("broadcast state allocation failed"));
  new (s) BroadcastState<T>{};
  s->alloc = alloc;
  s->cap = capacity;
  s->buf = static_cast<T*>(alloc.allocate(sizeof(T) * capacity, alignof(T)));
  s->occupied = static_cast<bool*>(
      alloc.allocate(sizeof(bool) * capacity, alignof(bool)));
  if (s->buf == nullptr || s->occupied == nullptr) {
    if (s->buf != nullptr)
      alloc.deallocate(s->buf, sizeof(T) * capacity, alignof(T));
    if (s->occupied != nullptr)
      alloc.deallocate(s->occupied, sizeof(bool) * capacity, alignof(bool));
    s->~BroadcastState<T>();
    alloc.deallocate(s, sizeof(BroadcastState<T>),
                     alignof(BroadcastState<T>));
    return std::unexpected(
        Status::ResourceExhausted("broadcast ring allocation failed"));
  }
  for (std::size_t i = 0; i < capacity; ++i) s->occupied[i] = false;
  s->refcount.store(2, std::memory_order_relaxed);   // 1 sender + 1 receiver
  s->senders.store(1, std::memory_order_relaxed);
  s->receivers.store(1, std::memory_order_relaxed);  // the returned Receiver
  return std::pair<BroadcastSender<T>, BroadcastReceiver<T>>{
      BroadcastSender<T>(s), BroadcastReceiver<T>(s, /*cursor=*/0)};
}

template <class T>
std::pair<BroadcastSender<T>, BroadcastReceiver<T>> make_broadcast(
    mem::Allocator alloc, std::size_t capacity) noexcept {
  auto result = try_make_broadcast<T>(alloc, capacity);
  HYLE_FATAL_ASSERT(result.has_value());
  return std::move(*result);
}

}  // namespace hyle::sched
