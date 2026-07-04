#pragma once

#include <cstdint>
#include <new>

#include "hyle/io/net.h"
#include "hyle/io/proactor.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"
#include "hyle/time/timer_source.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-06 timed io (capability C2): recv/accept/connect WITH a deadline. The op and a
// timer race; first to fire wins, loser is cancelled. "Almost every network
// library's survival trait" (tokio timeout / Go SetReadDeadline / asio cancel_after
// / seastar with_timeout) -- without it a slow/dead peer holds a fiber+fd+buffer
// forever.
//
// TWO HARD PARTS (verified against folly Baton / seastar with_timeout / tokio-uring
// + HYLE's own scheduler, 2026-07-05):
//
//  1. ARBITRATION needs NO atomics but MUST be symmetric. A connection fiber is
//     PINNED to its owner worker (shard-per-core): its io ring, timer wheel, and
//     resume all run on that ONE worker thread, serially (io completion reaped in
//     io_context::poll; timer fired in fire_due_timers). So a plain `io_done` /
//     `timer_fired` flag pair is race-free (seastar's non-atomic timer.cancel()
//     trick; folly needs atomics only for cross-thread post -- not us).
//     BUT: `Fiber::wake_rendezvous()` is a 2-PARTY handshake (worker-dispatch +
//     ONE waker). A timed op has TWO wakers (io + timer). If BOTH call
//     wake_rendezvous in one episode, the second flips wake_state_ back to 1 and
//     LEAKS it into the fiber's next suspend -> a lost/spurious wake. So each side
//     must WAKE ONLY IF IT WON: io side wakes iff !timer_fired; timer side wakes
//     iff !io_done. That guarantees exactly ONE wake_rendezvous per episode. The
//     timer side runs as TimerEntry::on_fire (fire_due_timers calls it instead of
//     the blind sleep wake), so it can check io_done first.
//
//  2. LATE-COMPLETION LIFETIME. Timer wins -> we submit a proactor cancel, but that
//     is ASYNC: the cancelled kernel op STILL delivers a final CQE later (io_uring
//     -ECANCELED / IOCP ERROR_OPERATION_ABORTED). If the slot were on the fiber
//     stack, the fiber has already resumed (-ETIMEDOUT) and its frame is gone when
//     that late completion lands -> UAF. So a timed op's slot is HEAP-allocated
//     with a refcount (tokio-uring Lifecycle::Ignored, same idea). Refcount = 2:
//     ONE per holder that will still touch the slot -- the KERNEL op (its
//     completion, normal or -ECANCELED, will land) and the FIBER (it will resume
//     and read the result). Each holder release()s exactly once when done. Ordinary
//     untimed ops keep their zero-alloc STACK slot; only a timed op pays one heap
//     alloc + the orphan lifetime.
//
// The completion router (io_context::poll) distinguishes a timed slot from an
// ordinary stack IoOperation by a LOW-BIT TAG on the user_data (pointers are >=8B
// aligned -> bit 0 free): ordinary = &IoOperation (bit0 clear), timed = &TimedSlot
// | 1. The cancel op's OWN completion carries kCancelToken, dropped by the router.

namespace hyle::io {

// Heap state for one timed op. Refcount {kernel op, fiber} = 2. Single-threaded per
// owner worker => refcount + flags are PLAIN (non-atomic).
struct TimedSlot {
  IoOperation io{};          // the io op's landing pad (reaper fills result)
  time::TimerEntry timer{};  // armed on the owner worker's wheel; on_fire = arbiter
  mem::Allocator alloc;      // frees the slot at refcount 0
  Proactor* p = nullptr;     // ring, for submit_cancel when the timer wins
  int refcount = 2;          // {kernel op, fiber}; non-atomic (owner-thread only)
  bool io_done = false;      // io completion arrived
  bool timer_fired = false;  // deadline fired

  void release() noexcept {
    if (--refcount == 0) {
      mem::Allocator a = alloc;
      this->~TimedSlot();
      a.deallocate(this, sizeof(TimedSlot), alignof(TimedSlot));
    }
  }
};

// Token tagging: bit 0 == timed slot (vs an ordinary IoOperation*).
inline std::uint64_t to_timed_token(TimedSlot* s) noexcept {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(s)) | 1ull;
}
inline bool is_timed_token(std::uint64_t ud) noexcept { return (ud & 1ull) != 0; }
inline TimedSlot* from_timed_token(std::uint64_t ud) noexcept {
  return reinterpret_cast<TimedSlot*>(static_cast<std::uintptr_t>(ud & ~1ull));
}

// Reserved token for a cancel op's OWN completion (io_uring emits one CQE for the
// cancel request itself). Tagged (bit0 set) so is_timed_token catches it and the
// timed router drops this exact value instead of dereferencing it as a slot. Must
// differ from Proactor::kWakeToken (~0): use ~0 with bit1 cleared, so bit0 stays
// set (tagged) but the value is distinct from the wake token.
inline constexpr std::uint64_t kCancelToken = (~std::uint64_t{0}) & ~(std::uint64_t{1} << 1);

// Router entry for a TIMED completion (io_context::poll calls this for any reaped
// completion whose token has bit 0 set). Owner-thread only. Handles the normal
// completion, the LATE -ECANCELED/ABORTED one, and the cancel op's own token.
inline void complete_timed(const Completion& c) noexcept {
  if (c.user_data == kCancelToken) return;  // cancel op's own CQE: drop
  TimedSlot* s = from_timed_token(c.user_data);
  s->io.result = c.result;
  s->io.flags = c.flags;
  s->io_done = true;
  // The kernel op just completed -> its ref is spent, release it. If the timer
  // already won, the fiber is gone (resumed with -ETIMEDOUT): DON'T wake, this is
  // the late cancellation completion, just drop the ref (may free the slot). If io
  // won the race, wake the fiber FIRST (it still holds its own ref), THEN release
  // the kernel ref -- wake before release so the slot is alive across the wake.
  if (s->timer_fired) {
    s->release();  // late completion: kernel ref, fiber ref already dropped/ing
    return;
  }
  s->io.waker.wake();  // io won -> exactly one wake_rendezvous this episode
  s->release();        // kernel ref spent
}

// Timer-side arbiter, installed as TimedSlot::timer.on_fire. Runs in fire_due_timers
// on the owner worker. Symmetric to complete_timed: mark timer_fired; wake ONLY if
// io hasn't already won (else io already did the single wake -> touching
// wake_rendezvous here would leak wake_state_). The timer holds NO ref, so it never
// release()s. `e->data` points back to the slot (set in make_timed_slot) -- a direct
// read, no offsetof (fire_due_timers doesn't touch data for on_fire entries).
inline void timed_on_fire(time::TimerEntry* e) noexcept {
  auto* s = static_cast<TimedSlot*>(e->data);
  s->timer_fired = true;
  if (s->io_done) return;  // io already won + woke the fiber: do not touch rendezvous
  s->io.waker.wake();      // timeout won -> exactly one wake_rendezvous this episode
}

namespace detail {

// Allocate + arm a timed op. Captures the fiber waker into the io slot, arms the
// timer (on_fire = arbiter, so fire_due_timers routes through timed_on_fire, NOT the
// blind sleep wake). Caller then submits the io op with to_timed_token(slot).
inline TimedSlot* make_timed_slot(mem::Allocator alloc, Proactor* p,
                                  std::uint64_t deadline_ns) noexcept {
  auto* s = static_cast<TimedSlot*>(
      alloc.allocate(sizeof(TimedSlot), alignof(TimedSlot)));
  if (s == nullptr) return nullptr;
  new (s) TimedSlot{};
  s->alloc = alloc;
  s->p = p;
  s->io.waker = sched::current_waker();
  s->timer.on_fire = &timed_on_fire;  // route timer fire through the arbiter
  sched::arm_timeout(&s->timer, deadline_ns);
  s->timer.data = s;  // arbiter reads slot back via data (arm_timeout set it to the
                      // fiber for sleep semantics; a timed entry overrides to slot)
  return s;
}

// Resume-side arbitration + cleanup. Returns the value for await_resume.
inline std::int64_t resume_timed(TimedSlot* s) noexcept {
  std::int64_t result;
  if (s->io_done) {
    // io won (or completed alongside the timer). Cancel the timer if it hasn't
    // fired -- synchronous wheel unlink, no late event. result = the io bytes/errno.
    if (!s->timer_fired) sched::cancel_timeout(&s->timer);
    result = s->io.result;
    s->release();  // fiber ref (kernel ref was released by complete_timed)
    return result;
  }
  // Timer won: io still in flight. Ask the proactor to cancel it; the late CQE will
  // drop the kernel ref via complete_timed. Drop the fiber ref now, return timeout.
  s->p->submit_cancel(to_timed_token(s), kCancelToken);
  result = -110;  // -ETIMEDOUT (Linux errno 110; a stable negative sentinel)
  s->release();   // fiber ref; kernel ref stays until the late completion lands
  return result;
}

// Called when the proactor submit failed (ring full / bad fd): the kernel never
// took the op, so no completion will drop its kernel ref -- drop it here, cancel
// the timer, mark done so await_resume returns the error without suspending.
inline void on_submit_failed(TimedSlot* s) noexcept {
  sched::cancel_timeout(&s->timer);
  s->io.result = -1;
  s->io_done = true;
  s->release();  // kernel ref (never materialized)
}

}  // namespace detail

// co_await TimedRecvOp -> bytes read (>=0), negative errno, or -ETIMEDOUT (-110).
// `deadline_ns` is absolute monotonic (time::monotonic_ns() + budget). `alloc`
// backs the heap slot (use the same allocator the connection uses).
struct TimedRecvOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  mem::Allocator alloc;
  std::uint64_t deadline_ns;

  TimedSlot* slot_ = nullptr;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    slot_ = detail::make_timed_slot(alloc, p, deadline_ns);
    if (slot_ == nullptr) return false;  // OOM -> await_resume returns error
    if (!p->submit_recv(fd, buf, len, 0, to_timed_token(slot_))) {
      detail::on_submit_failed(slot_);
      return false;  // resume immediately
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (slot_ == nullptr) return -1;  // OOM
    return detail::resume_timed(slot_);
  }
};

// co_await TimedAcceptOp -> accepted fd (>=0), negative errno, or -ETIMEDOUT.
// The listener's accept is raced against the deadline (a server that wants to stop
// accepting after a quiet period, or bound its accept latency).
struct TimedAcceptOp {
  Proactor* p;
  std::intptr_t fd;  // listening socket
  mem::Allocator alloc;
  std::uint64_t deadline_ns;

  TimedSlot* slot_ = nullptr;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    slot_ = detail::make_timed_slot(alloc, p, deadline_ns);
    if (slot_ == nullptr) return false;
    if (!p->submit_accept(fd, nullptr, nullptr, to_timed_token(slot_))) {
      detail::on_submit_failed(slot_);
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (slot_ == nullptr) return -1;
    return detail::resume_timed(slot_);
  }
};

// co_await TimedConnectOp -> 0 on success, negative errno, or -ETIMEDOUT. Bounds a
// connect to a slow/unreachable peer (the classic "connect timeout"). `addr` must
// outlive the co_await (keep it on the coroutine frame).
struct TimedConnectOp {
  Proactor* p;
  std::intptr_t fd;
  const void* addr;
  std::uint32_t addrlen;
  mem::Allocator alloc;
  std::uint64_t deadline_ns;

  TimedSlot* slot_ = nullptr;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    slot_ = detail::make_timed_slot(alloc, p, deadline_ns);
    if (slot_ == nullptr) return false;
    if (!p->submit_connect(fd, addr, addrlen, to_timed_token(slot_))) {
      detail::on_submit_failed(slot_);
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (slot_ == nullptr) return -1;
    return detail::resume_timed(slot_);
  }
};

}  // namespace hyle::io
