#include "hyle/sched/sync.h"

#include <atomic>

#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/waker.h"

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
      if (head_ == nullptr) tail_ = nullptr;
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

// ---- Mutex (futex, Drepper 3-state) -----------------------------------------
// state_: 0=unlocked, 1=locked no waiters, 2=locked waiters. lock()'s fast CAS
// (0->1) is inline; this is the contended tail. unlock()'s fast fetch_sub is
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
