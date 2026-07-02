#include "hyle/sched/sync.h"

#include <atomic>

#include "hyle/macros.h"
#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/waker.h"
#include "hyle/time/clock.h"

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
  if (fiber) {
    node->waker = w;
    node->token = nullptr;
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
      // Copy out everything we need BEFORE releasing the lock / waking: the node
      // may be destroyed the instant its owner resumes.
      waker = node->waker;
      token = node->token;
      have = true;
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
    for (Waiter* p = node; p != nullptr; p = p->next) p->linked = false;
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
    wake_one(waker, token);
    node = next;
  }
}

// ---- timed wait (sched-04c, thread-mode) ------------------------------------

std::uint64_t ConditionVariable::deadline_from_now_(
    std::uint64_t timeout_ns) noexcept {
  std::uint64_t now = time::monotonic_ns();
  std::uint64_t d = now + timeout_ns;
  return d < now ? UINT64_MAX : d;  // saturate on overflow
}

std::uint64_t ConditionVariable::remaining_ns_(
    std::uint64_t deadline_ns) noexcept {
  std::uint64_t now = time::monotonic_ns();
  return deadline_ns > now ? deadline_ns - now : 0;
}

ConditionVariable::CvStatus ConditionVariable::finish_wait_timed_(
    Waiter* node, std::uint64_t timeout_ns) noexcept {
  // Thread-mode only (fiber-mode timed wait = sched-04c-pending: needs a wheel
  // timer whose fire races notify + a cross-worker cancel). prepare_wait_ set
  // token to this thread's ThreadToken; a fiber node would have token==nullptr.
  HYLE_FATAL_ASSERT(node->token != nullptr &&
                    "fiber-mode CV timed wait not yet supported (sched-04c)");
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


// inline; unlock_slow runs only when there were waiters.

void Mutex::lock_slow(std::uint32_t c) noexcept {
  // `c` is the value the fast-path CAS observed (non-zero -> contended). Flag
  // the lock as "has waiters" (2) and futex-park while it stays 2. Each wakeup
  // re-tries the exchange: getting 0 back means we won the lock.
  if (c != 2) c = state_.exchange(2, std::memory_order_acquire);
  while (c != 0) {
    platform::detail::wait_on_address(&state_, 2);  // sleep while state_ == 2
    c = state_.exchange(2, std::memory_order_acquire);
  }
}

void Mutex::unlock_slow() noexcept {
  // We already fetch_sub'd 2->1 (release) on the hot path; there were waiters.
  // Drop to 0 and wake one. The woken waiter re-exchanges to 2, so a still-
  // contended lock keeps its "waiters" flag (no lost wakeup).
  state_.store(0, std::memory_order_release);
  platform::detail::wake_one(&state_);
}

}  // namespace hyle::sched
