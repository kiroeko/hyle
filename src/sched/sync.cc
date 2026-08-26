#include "hyle/sched/sync.h"

#include <atomic>

#include "hyle/core/compiler.h"
#include "hyle/platform.h"  // HYLE_OS_WINDOWS (Mutex backend split)
#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"
#include "hyle/time/clock.h"
#include "hyle/time/timer_entry.h"

namespace hyle::sched {
namespace {

// Park token for a THREAD-mode waiter. Lives in thread_local storage (NOT on the
// wait() stack frame) so it stays valid after the waiter returns -- the notifier
// reads the node's token pointer, drops the CV lock, then touches the token; a
// stack-local token could be destroyed out from under it. One per thread is
// enough: a thread blocks in at most one wait() at a time.
//
// `notified` distinguishes "this episode's notify fired" from a stale/spurious
// Parker token left over from a previous episode (Parker remembers an unpark even
// across episodes), so the park loop never returns early on a phantom wake.
struct ThreadToken {
  platform::Parker parker;
  std::atomic<bool> notified{false};
};

thread_local ThreadToken t_token;

}  // namespace

bool ConditionVariable::prepare_wait_(Waiter* node) noexcept {
  // current_waker() names {scheduler, this fiber} iff we run inside a worker
  // fiber; off a worker (a bare/external thread) it is invalid -> thread mode.
  const Waker w = current_waker();
  const bool fiber = w.valid();
  node->next = nullptr;
  node->prev = nullptr;
  node->linked = true;
  node->timed_state.store(0, std::memory_order_relaxed);
  node->fiber_timed = false;
  if (fiber) {
    node->waker = w;
    node->token = nullptr;
    // Mark before publication. Untimed fiber waits also use the one-shot notify
    // state harmlessly; timed waits need a notify racing finish_wait_timed_fiber_
    // to record that it won before the timer is armed.
    node->fiber_timed = true;
  } else {
    // Arm this thread's token BEFORE the node is published, so a notify that
    // pops it sees notified==false and flips it to true (under our list lock).
    t_token.notified.store(false, std::memory_order_relaxed);
    node->waker = Waker{};
    node->token = &t_token;
  }
  // Link onto the FIFO tail. Done while the caller still holds the MONITOR lock
  // (the predicate lock) -- that is what closes the lost-wakeup window: a
  // notifier must take that same monitor lock to change the predicate, so it
  // cannot squeeze its state-change + notify between our predicate check and
  // this enqueue.
  mu_.lock();
  node->prev = tail_;
  if (tail_ == nullptr) {
    head_ = node;
  } else {
    tail_->next = node;
  }
  tail_ = node;
  waiters_.fetch_add(1, std::memory_order_relaxed);  // published via mu_/monitor
  mu_.unlock();
  return fiber;
}

// Unlink `node` from the doubly-linked list if still linked (idempotent).
// Caller holds mu_. A timed waiter that hit its deadline calls this to retract
// its own (stack-resident) node so a subsequent notify can't pop a dead node.
void ConditionVariable::remove_waiter_(Waiter* node) noexcept {
  if (!node->linked) return;  // already popped by a notify
  if (node->prev != nullptr) {
    node->prev->next = node->next;
  } else {
    head_ = node->next;  // was head
  }
  if (node->next != nullptr) {
    node->next->prev = node->prev;
  } else {
    tail_ = node->prev;  // was tail
  }
  node->linked = false;
  waiters_.fetch_sub(1, std::memory_order_relaxed);
}

void ConditionVariable::finish_wait_(bool fiber, void* token) noexcept {
  if (fiber) {
    // Leave the run loop; a Waker::wake() (driven by notify_*) re-injects us.
    // The wake-before-suspend race is closed by the Fiber rendezvous inside
    // suspend()/wake() (sched-04a) -- same spine as a raw io completion.
    suspend();
    return;
  }
  // Thread mode: block the OS thread until our token is flagged. Loop guards
  // against spurious low-level wakes and stale Parker tokens.
  auto* tok = static_cast<ThreadToken*>(token);
  while (!tok->notified.load(std::memory_order_acquire)) {
    tok->parker.park();
  }
}

// Re-inject / unblock one detached waiter. The node's fields were read out under
// the list lock by the caller and the node was removed from the list, so no
// other notifier can touch it; the waiter itself cannot destroy it until it is
// woken (which is what we are about to do). For a thread waiter we must read the
// token pointer (stable thread_local) -- the node may die the instant the thread
// observes `notified`, so flag THEN unpark, and never touch the node afterwards.
static void wake_one(const Waker& waker, void* token) noexcept {
  if (token != nullptr) {
    auto* tok = static_cast<ThreadToken*>(token);
    tok->notified.store(true, std::memory_order_release);
    tok->parker.unpark();
  } else {
    waker.wake();
  }
}

void ConditionVariable::notify_one() noexcept {
  // Fast path: nobody parked. A waiter publishes waiters_ under the monitor lock
  // before suspending; a notifier holding that same lock therefore sees it, so
  // skipping here cannot drop a wake (see header contract).
  if (waiters_.load(std::memory_order_acquire) == 0) return;
  Waker waker;
  void* token = nullptr;
  bool have = false;
  {
    mu_.lock();
    Waiter* node = head_;
    if (node != nullptr) {
      head_ = node->next;
      if (head_ == nullptr) {
        tail_ = nullptr;
      } else {
        head_->prev = nullptr;
      }
      node->linked = false;  // popped: a timed waiter's remove_waiter_ is now a no-op
      waiters_.fetch_sub(1, std::memory_order_relaxed);
      bool won = true;
      if (node->fiber_timed) {
        unsigned char expected = 0;
        won = node->timed_state.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel);
      }
      // Copy out everything we need BEFORE releasing the lock / waking: the node
      // may be destroyed the instant its owner resumes.
      waker = node->waker;
      token = node->token;
      have = won;
    }
    mu_.unlock();
  }
  if (have) wake_one(waker, token);
}

void ConditionVariable::notify_all() noexcept {
  if (waiters_.load(std::memory_order_acquire) == 0) return;  // nobody parked
  // Detach the whole list under the lock, then wake each outside it (wake/unpark
  // can run scheduler code / a syscall -- not under our tiny spinlock).
  Waiter* node;
  {
    mu_.lock();
    node = head_;
    // Mark every node unlinked so a concurrent timed-out waiter's remove_waiter_
    // becomes a no-op (we own them all now). Done under mu_.
    for (Waiter* p = node; p != nullptr; p = p->next) {
      p->linked = false;
      if (p->fiber_timed) {
        unsigned char expected = 0;
        (void)p->timed_state.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel);
      }
    }
    head_ = nullptr;
    tail_ = nullptr;
    waiters_.store(0, std::memory_order_relaxed);  // detached the whole list
    mu_.unlock();
  }
  while (node != nullptr) {
    // Read next + copy fields BEFORE waking: once woken, the node can vanish.
    Waiter* next = node->next;
    const Waker waker = node->waker;
    void* token = node->token;
    if (!node->fiber_timed ||
        node->timed_state.load(std::memory_order_acquire) == 1)
      wake_one(waker, token);
    node = next;
  }
}

// ---- timed wait (sched-04c, thread-mode) ------------------------------------

std::uint64_t ConditionVariable::deadline_from_now_(
    std::uint64_t timeout_ns) noexcept {
  return time::deadline_after_ns(timeout_ns);
}

std::uint64_t ConditionVariable::remaining_ns_(
    std::uint64_t deadline_ns) noexcept {
  return time::remaining_ns(deadline_ns);
}

ConditionVariable::CvStatus ConditionVariable::finish_wait_timed_(
    Waiter* node, std::uint64_t timeout_ns) noexcept {
  // Thread-mode only. prepare_wait_ set token to this thread's ThreadToken; a
  // fiber node has token==nullptr and is routed to finish_wait_timed_fiber_.
  HYLE_FATAL_ASSERT(node->token != nullptr);
  auto* tok = static_cast<ThreadToken*>(node->token);

  // Absolute deadline so spurious/early park returns don't extend the wait.
  const std::uint64_t deadline = deadline_from_now_(timeout_ns);
  bool notified = tok->notified.load(std::memory_order_acquire);
  while (!notified) {
    std::uint64_t rem = remaining_ns_(deadline);
    if (rem == 0) break;  // deadline reached
    // park_timeout is single-shot: it may return early (spurious / real unpark).
    // The loop re-checks `notified` and recomputes the remaining time, so the
    // total wait never exceeds the deadline. A real notify unparks + sets
    // notified (wake_one), so the next load observes it.
    tok->parker.park_timeout(rem);
    notified = tok->notified.load(std::memory_order_acquire);
  }
  if (notified) return CvStatus::kNoTimeout;

  // Deadline hit. Retract our node under the monitor lock -- but a notify may
  // have popped us in the tiny gap since the last `notified` load. Resolve it
  // under mu_: if still linked, WE win the race (retract -> timeout). If already
  // unlinked, a notify won (it will/did set notified) -> report no-timeout so we
  // don't lose its wake (classic notify-vs-timeout arbitration, single-locked).
  mu_.lock();
  const bool still_linked = node->linked;
  if (still_linked) remove_waiter_(node);
  mu_.unlock();
  if (still_linked) return CvStatus::kTimeout;
  // A notify claimed us: wait for its unpark so the ThreadToken (thread_local,
  // outlives this frame) isn't left with a pending unpark that leaks into the
  // next episode. It set notified under mu_ before unparking, so this is bounded.
  while (!tok->notified.load(std::memory_order_acquire)) tok->parker.park();
  return CvStatus::kNoTimeout;
}

void ConditionVariable::timed_fiber_fire_(
    time::TimerEntry* entry) noexcept {
  auto* ctx = static_cast<TimedContext*>(entry->data());
  ConditionVariable* cv = ctx->cv;
  Waiter* node = ctx->node;
  bool wake = false;
  cv->mu_.lock();
  unsigned char expected = 0;
  if (node->linked &&
      node->timed_state.compare_exchange_strong(
          expected, 2, std::memory_order_acq_rel)) {
    cv->remove_waiter_(node);
    wake = true;
  }
  cv->mu_.unlock();
  if (wake) node->waker.wake();
}

ConditionVariable::CvStatus ConditionVariable::finish_wait_timed_fiber_(
    Waiter* node, std::uint64_t timeout_ns) noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr && node->waker.valid());
  // prepare_wait_ marked every fiber waiter before publishing its stack node.
  // Do not write that mode flag here: a notifier may already have popped and
  // inspected the node after enqueue but before this function starts.

  // An immediate deadline still has to arbitrate with a notify that may have
  // popped us after the monitor lock was released.
  if (timeout_ns == 0) {
    bool timed_out = false;
    mu_.lock();
    unsigned char expected = 0;
    if (node->linked &&
        node->timed_state.compare_exchange_strong(
            expected, 2, std::memory_order_acq_rel)) {
      remove_waiter_(node);
      timed_out = true;
    }
    mu_.unlock();
    if (timed_out) return CvStatus::kTimeout;
    suspend();  // notify owns the rendezvous; consume its pending wake
    return CvStatus::kNoTimeout;
  }

  // If notify already won before we could arm, only complete its rendezvous.
  if (node->timed_state.load(std::memory_order_acquire) == 1) {
    suspend();
    return CvStatus::kNoTimeout;
  }

  const int previous_owner = self->owner_worker();
  if (previous_owner < 0) {
    const int current = current_worker_index();
    HYLE_FATAL_ASSERT(current >= 0);
    self->set_owner_worker(current);
  }

  time::TimerEntry timer;
  TimedContext ctx{this, node};
  timer.set_fire_callback(&ConditionVariable::timed_fiber_fire_);
  timer.set_data(&ctx);  // callback context; callback path needs no Fiber payload
  // Publish the complete callback record before inserting it into the live
  // queue. A sub-tick deadline may fire as soon as arm_timeout returns.
  arm_timeout(&timer, deadline_from_now_(timeout_ns));
  suspend();

  // We resume on the temporary owner worker. If notify won before expiry, remove
  // its still-live timer from this worker's queue before the stack entry dies.
  if (timer.scheduled()) cancel_timeout(&timer);
  if (previous_owner < 0) self->set_owner_worker(-1);
  return node->timed_state.load(std::memory_order_acquire) == 2
             ? CvStatus::kTimeout
             : CvStatus::kNoTimeout;
}


// ---- Semaphore --------------------------------------------------------------

bool Semaphore::try_acquire(std::size_t n) noexcept {
  mu_.lock();
  // Only take permits if none are queued ahead of us: honoring FIFO here is what
  // lets RwLock's queued writer block later readers (a try_read must NOT steal a
  // permit the writer is waiting for). If a waiter is queued, we defer.
  if (!closed_ && head_ == nullptr && permits_ >= n) {
    permits_ -= n;
    mu_.unlock();
    return true;
  }
  mu_.unlock();
  return false;
}

bool Semaphore::acquire(std::size_t n) noexcept {
  mu_.lock();
  if (closed_) {
    mu_.unlock();
    return false;
  }
  // Fast path: enough permits AND nobody ahead of us in line (FIFO -- never jump
  // the queue, or a big head-of-line request could starve).
  if (head_ == nullptr && permits_ >= n) {
    permits_ -= n;
    mu_.unlock();
    return true;
  }
  // Slow path: queue a waiter with its own CV, block until release() grants us the
  // full request or close() aborts us. The node lives on this stack frame; it stays
  // linked only while we are parked (unlinked by release/close before waking us).
  Waiter node;
  node.need = n;
  node.prev = tail_;
  if (tail_ == nullptr) {
    head_ = &node;
  } else {
    tail_->next = &node;
  }
  tail_ = &node;
  // Mesa monitor: wait on our own CV under mu_, re-checking the predicate. release
  // sets granted (+ already debited permits_); close sets aborted. Either way the
  // node has been UNLINKED by the setter before notify, so no dangling stack node.
  node.cv.wait(mu_, [&] { return node.granted || node.aborted; });
  const bool ok = node.granted;
  mu_.unlock();
  return ok;
}

bool Semaphore::try_release(std::size_t n) noexcept {
  // Collect the CVs to notify AFTER dropping mu_ (waking can run scheduler code;
  // don't do it under the tiny spinlock). Each satisfied waiter is unlinked here.
  // We notify by pointer into each woken node's own CV -- valid because the waiter
  // cannot destroy its node until its cv.wait returns, which needs this notify.
  Waiter* woken_head = nullptr;
  Waiter* woken_tail = nullptr;  // build a temporary "to notify" list via `next`
  mu_.lock();
  if (n > std::numeric_limits<std::size_t>::max() - permits_) {
    mu_.unlock();
    return false;
  }
  permits_ += n;
  // FIFO: satisfy the front waiter fully before moving on; STOP at the first
  // waiter we cannot fully satisfy (no skip-ahead -> no large-request starvation).
  while (head_ != nullptr && permits_ >= head_->need) {
    Waiter* w = head_;
    permits_ -= w->need;
    // Unlink from the wait list.
    head_ = w->next;
    if (head_ == nullptr) {
      tail_ = nullptr;
    } else {
      head_->prev = nullptr;
    }
    w->granted = true;
    // Splice onto the local to-notify list (reuse `next`; `prev` no longer used).
    w->next = nullptr;
    if (woken_tail == nullptr) {
      woken_head = w;
    } else {
      woken_tail->next = w;
    }
    woken_tail = w;
  }
  mu_.unlock();
  // Wake each satisfied waiter on its own CV, outside the lock.
  for (Waiter* w = woken_head; w != nullptr;) {
    Waiter* next = w->next;  // read before notify: w may vanish once woken
    w->cv.notify_one();
    w = next;
  }
  return true;
}

void Semaphore::close() noexcept {
  Waiter* aborted_head = nullptr;
  mu_.lock();
  closed_ = true;
  // Detach the whole queue; each waiter's acquire() will see aborted and return
  // false. Reuse `next` to build the local list (as in release()).
  Waiter* w = head_;
  head_ = nullptr;
  tail_ = nullptr;
  aborted_head = w;
  while (w != nullptr) {
    w->aborted = true;
    Waiter* n = w->next;
    w->prev = nullptr;
    w->next = n;  // keep FIFO order for tidy waking (order is not load-bearing)
    w = n;
  }
  mu_.unlock();
  for (Waiter* p = aborted_head; p != nullptr;) {
    Waiter* next = p->next;
    p->cv.notify_one();
    p = next;
  }
}

std::size_t Semaphore::available() const noexcept {
  mu_.lock();
  const std::size_t p = permits_;
  mu_.unlock();
  return p;
}

// inline; unlock_slow runs only when there were waiters.

#if !defined(HYLE_OS_WINDOWS)
void Mutex::lock_slow(std::uint32_t c) noexcept {
  // `c` is the value the fast-path CAS observed (non-zero -> contended). Flag
  // the lock as "has waiters" (2) and futex-park while it stays 2. Each wakeup
  // re-tries the exchange: getting 0 back means we won the lock.
  //
  // NO pre-park spin (deliberately, verified 2026-07-03): a bounded TTAS spin --
  // the glibc/absl trick -- MADE IT WORSE (spread 12% -> 59%). Reason is the fiber
  // model: spinning burns the WORKER OS THREAD, but the lock holder may be another
  // FIBER queued on that SAME worker -- spinning the worker then starves the
  // holder, which can never run to release. On real OS threads the holder runs on
  // its own core so spinning pays off; over an M:N fiber scheduler it can starve.
  // So the Linux path parks immediately. (Windows uses SRWLOCK -- see below.)
  if (c != 2) c = state_.exchange(2, std::memory_order_acquire);
  while (c != 0) {
    platform::detail::wait_on_address(&state_, 2);  // sleep while state_ == 2
    c = state_.exchange(2, std::memory_order_acquire);
  }
}

void Mutex::unlock_slow() noexcept {
  // unlock() has already published state 0. The woken waiter re-exchanges to 2,
  // so a still-contended lock keeps its "waiters" flag (no lost wakeup).
  platform::detail::wake_one(&state_);
}
#endif  // !HYLE_OS_WINDOWS

}  // namespace hyle::sched

// ---- Windows Mutex = SRWLOCK (measured 2.66x faster contended than the futex) --
#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

// SRWLOCK is RTL_SRWLOCK{PVOID Ptr}; our void* srw_ IS that pointer field, so
// &srw_ reinterprets cleanly as PSRWLOCK. Zero-init (nullptr) == SRWLOCK_INIT, so
// the default-constructed Mutex is a valid unheld SRWLOCK with no explicit init.
static_assert(sizeof(void*) == sizeof(SRWLOCK), "SRWLOCK must be pointer-sized");

namespace hyle::sched {

void Mutex::lock() noexcept {
  ::AcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw_));
}
bool Mutex::try_lock() noexcept {
  return ::TryAcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw_)) != 0;
}
void Mutex::unlock() noexcept {
  ::ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw_));
}

}  // namespace hyle::sched
#endif  // HYLE_OS_WINDOWS
