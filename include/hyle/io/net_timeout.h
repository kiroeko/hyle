#pragma once

#include <atomic>
#include <cstdint>
#include <new>
#include <optional>

#include "hyle/io/net.h"
#include "hyle/io/proactor.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"
#include "hyle/time/timer_entry.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
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
//     PINNED to its owner worker (shard-per-core): its io ring, timer queue, and
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
//     timer side runs as TimerEntry::fire_callback (fire_due_timers calls it instead of
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
  enum class Winner : std::uint8_t {
    kNone,
    kIo,
    kDeadline,
    kCancelled,
  };

  IoOperation io{};          // the io op's landing pad (reaper fills result)
  time::TimerEntry timer{};  // armed on owner worker's queue; callback = arbiter
  mem::AllocatorRef alloc;      // frees the slot at refcount 0
  Proactor* p = nullptr;     // ring, for submit_cancel when the timer wins
  int refcount = 2;          // {kernel op, fiber}; non-atomic (owner-thread only)
  // Cancellation callbacks may run on a different thread.  Only the optional
  // deadline/cancellation path pays for this arbitration atomic; ordinary IO
  // continues to use the stack-only IoOperation fast path.
  std::atomic<Winner> winner{Winner::kNone};
  bool timer_armed = false;
  bool timer_fired = false;  // owner-worker timer callback has run

  void release() noexcept {
    if (--refcount == 0) {
      mem::AllocatorRef a = alloc;
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
  TimedSlot::Winner expected = TimedSlot::Winner::kNone;
  if (s->winner.compare_exchange_strong(expected, TimedSlot::Winner::kIo,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
    // IO won. Wake before spending the kernel ref so the fiber-owned ref keeps
    // the slot alive even if the scheduler runs it immediately.
    s->io.waker.wake();
  }
  // Deadline/cancellation already won -> this is the required late completion.
  s->release();
}

// Timer-side arbiter, installed as TimedSlot::timer's fire callback. Runs in
// fire_due_timers
// on the owner worker. Symmetric to complete_timed: mark timer_fired; wake ONLY if
// io hasn't already won (else io already did the single wake -> touching
// wake_rendezvous here would leak wake_state_). The timer holds NO ref, so it never
// release()s. `e->data()` points back to the slot (set in make_timed_slot) -- a direct
// read, no offsetof (fire_due_timers doesn't touch data for callback entries).
inline void timed_on_fire(time::TimerEntry* e) noexcept {
  auto* s = static_cast<TimedSlot*>(e->data());
  s->timer_fired = true;
  TimedSlot::Winner expected = TimedSlot::Winner::kNone;
  if (!s->winner.compare_exchange_strong(
          expected, TimedSlot::Winner::kDeadline,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  // Cancel immediately rather than waiting for the resumed fiber to run. The
  // original completion still owns the kernel ref and is the only event that
  // spends it.
  s->p->submit_cancel(to_timed_token(s), kCancelToken);
  s->io.waker.wake();
}

namespace detail {

// Allocate + arm a timed op. Captures the fiber waker into the io slot, arms the
// timer (callback = arbiter, so fire_due_timers routes through timed_on_fire, NOT the
// blind sleep wake). Caller then submits the io op with to_timed_token(slot).
inline TimedSlot* make_timed_slot(mem::AllocatorRef alloc, Proactor* p,
                                  std::uint64_t deadline_ns) noexcept {
  auto* s = static_cast<TimedSlot*>(
      alloc.allocate(sizeof(TimedSlot), alignof(TimedSlot)));
  if (s == nullptr) return nullptr;
  new (s) TimedSlot{};
  s->alloc = alloc;
  s->p = p;
  s->io.waker = sched::current_waker();
  s->timer.set_fire_callback(
      &timed_on_fire);  // route timer fire through the arbiter
  s->timer.set_data(s);  // publish complete callback context before the live arm
  sched::arm_timeout(&s->timer, time::monotonic_time_point(deadline_ns));
  s->timer_armed = true;
  return s;
}

inline TimedSlot* make_budget_slot(
    mem::AllocatorRef alloc, Proactor* p,
    const std::optional<time::MonotonicTimePoint>& deadline) noexcept {
  auto* s = static_cast<TimedSlot*>(
      alloc.allocate(sizeof(TimedSlot), alignof(TimedSlot)));
  if (s == nullptr) return nullptr;
  new (s) TimedSlot{};
  s->alloc = alloc;
  s->p = p;
  s->io.waker = sched::current_waker();
  if (deadline.has_value()) {
    s->timer.set_fire_callback(&timed_on_fire);
    s->timer.set_data(s);
    sched::arm_timeout(&s->timer, *deadline);
    s->timer_armed = true;
  }
  return s;
}

// Resume-side arbitration + cleanup. Returns the value for await_resume.
inline std::int64_t resume_timed(TimedSlot* s) noexcept {
  const TimedSlot::Winner winner =
      s->winner.load(std::memory_order_acquire);
  HYLE_FATAL_ASSERT(winner != TimedSlot::Winner::kNone);
  if (s->timer_armed && !s->timer_fired) sched::cancel_timeout(&s->timer);

  std::int64_t result = s->io.result;
  if (winner == TimedSlot::Winner::kDeadline) result = kDeadlineExceeded;
  if (winner == TimedSlot::Winner::kCancelled) {
    // CancelToken callbacks may run on a foreign thread, while a Proactor's
    // submission side belongs exclusively to the socket's owner worker. The
    // callback only wins+wakes; submit the native cancellation here after the
    // pinned fiber resumes on its owner.
    s->p->submit_cancel(to_timed_token(s), kCancelToken);
    result = kCancelled;
  }

  // For a deadline/cancellation winner, the late original CQE drops the kernel
  // ref. For an IO winner it already did. Drop the fiber ref in every case.
  //
  // submit_cancel's return value is DELIBERATELY IGNORED, and that is correct (red-
  // team U5 investigation, 2026-07-05, cross-checked vs tokio-uring + liburing +
  // seastar): the kernel ref is dropped by the ORIGINAL op's completion, NOT by the
  // cancel. io_uring guarantees a CQE for every kernel-consumed SQE (liburing
  // io_uring_prep_cancel.3: "a CQE is always generated"); a cancel that returns
  // -ENOENT / whose submit_cancel returns false means the target op ALREADY
  // completed and its CQE is already reaped-or-queued -> complete_timed still runs
  // and still drops the kernel ref. The async-cancel is only an OPTIMIZATION to make
  // that completion arrive SOONER (a silent/dead peer would otherwise hold the op
  // until close); it is never the thing that frees the slot. tokio-uring's
  // Lifecycle::Ignored frees only on the CQE; seastar issues NO kernel cancel at all
  // and relies purely on "every SQE yields a completion". HYLE does the same, with
  // IoContext::~IoContext's orphan-drain as the teardown backstop (drains each ring
  // to in_flight()==0 so a late CQE after scheduler stop still frees the slot).
  //
  // The ONE path that WOULD leak the kernel ref -- the ORIGINAL op's submit failing
  // so its SQE was never kernel-consumed (no CQE ever comes) -- is handled elsewhere:
  // on_submit_failed() drops the kernel ref synchronously when submit_recv/accept/
  // connect returns false. So a cancel-submit failure here is genuinely harmless.
  s->release();
  return result;
}

// Cross-thread cancellation arbiter for a bounded operation. The
// callback owns no ref: CancelCallback destruction is serialized against an
// in-progress invocation, while the suspended fiber and kernel completion each
// retain one slot ref. It deliberately does not touch the owner-thread-only
// Proactor; resume_timed submits native cancellation on the pinned owner.
inline void cancel_timed_slot(void* context) noexcept {
  auto* s = static_cast<TimedSlot*>(context);
  TimedSlot::Winner expected = TimedSlot::Winner::kNone;
  if (!s->winner.compare_exchange_strong(
          expected, TimedSlot::Winner::kCancelled,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  s->io.waker.wake();
}

// Called when the proactor submit failed (ring full / bad fd): the kernel never
// took the op, so no completion will drop its kernel ref -- drop it here, cancel
// the timer, mark done so await_resume returns the error without suspending.
inline void on_submit_failed(TimedSlot* s) noexcept {
  if (s->timer_armed) sched::cancel_timeout(&s->timer);
  s->io.result = kSubmitFailed;  // never submitted (distinct from -EPERM); see net.h
  s->winner.store(TimedSlot::Winner::kIo, std::memory_order_release);
  s->release();  // kernel ref (never materialized)
}

}  // namespace detail

// co_await TimedRecvOp -> bytes read (>=0), negative errno, or -ETIMEDOUT (-110).
// `deadline_ns` is absolute monotonic (prefer
// time::deadline_after_ns(budget), which saturates). `alloc`
// backs the heap slot (use the same allocator the connection uses).
struct TimedRecvOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  mem::AllocatorRef alloc;
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
    if (slot_ == nullptr) return kSubmitFailed;  // OOM
    return detail::resume_timed(slot_);
  }
};

// Timed counterpart of SendOp. Useful for bounded backpressure when a peer stops
// draining its receive window.
struct TimedSendOp {
  Proactor* p;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  mem::AllocatorRef alloc;
  std::uint64_t deadline_ns;
  std::uint32_t flags = 0;

  TimedSlot* slot_ = nullptr;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    slot_ = detail::make_timed_slot(alloc, p, deadline_ns);
    if (slot_ == nullptr) return false;
    if (!p->submit_send(fd, buf, len, flags, to_timed_token(slot_))) {
      detail::on_submit_failed(slot_);
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept {
    return slot_ == nullptr ? kSubmitFailed : detail::resume_timed(slot_);
  }
};

// co_await TimedAcceptOp -> accepted fd (>=0), negative errno, or -ETIMEDOUT.
// The listener's accept is raced against the deadline (a server that wants to stop
// accepting after a quiet period, or bound its accept latency).
struct TimedAcceptOp {
  Proactor* p;
  std::intptr_t fd;  // listening socket
  mem::AllocatorRef alloc;
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
    if (slot_ == nullptr) return kSubmitFailed;
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
  mem::AllocatorRef alloc;
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
    if (slot_ == nullptr) return kSubmitFailed;
    return detail::resume_timed(slot_);
  }
};

struct TimedRecvFromOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  void* addr;
  std::uint32_t addr_cap;
  mem::AllocatorRef alloc;
  std::uint64_t deadline_ns;

  TimedSlot* slot_ = nullptr;
  alignas(16) unsigned char scratch_[Proactor::kDatagramScratch]{};

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    slot_ = detail::make_timed_slot(alloc, p, deadline_ns);
    if (slot_ == nullptr) return false;
    if (!p->submit_recvfrom(fd, buf, len, addr, addr_cap, scratch_,
                            to_timed_token(slot_))) {
      detail::on_submit_failed(slot_);
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept {
    return slot_ == nullptr ? kSubmitFailed : detail::resume_timed(slot_);
  }
};

struct TimedSendToOp {
  Proactor* p;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  const void* addr;
  std::uint32_t addrlen;
  mem::AllocatorRef alloc;
  std::uint64_t deadline_ns;

  TimedSlot* slot_ = nullptr;
  alignas(16) unsigned char scratch_[Proactor::kDatagramScratch]{};

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    slot_ = detail::make_timed_slot(alloc, p, deadline_ns);
    if (slot_ == nullptr) return false;
    if (!p->submit_sendto(fd, buf, len, addr, addrlen, scratch_,
                          to_timed_token(slot_))) {
      detail::on_submit_failed(slot_);
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept {
    return slot_ == nullptr ? kSubmitFailed : detail::resume_timed(slot_);
  }
};

}  // namespace hyle::io
