#pragma once

#include <cstdint>

#include "hyle/io/proactor.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/waker.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-04 net: the co_await socket layer over the proactor. This is the seam where
// io/ meets sched/ -- an async op suspends the FIBER (not just the coroutine) and
// resumes when the completion carries its result back.
//
// The mechanism (sched-design §3.2/§3.3): a submitted op needs two things back --
// a WAKE (re-inject the fiber) and a RESULT (bytes / negative errno). The Waker
// spine gives the wake; the result rides an `IoOperation` slot the awaiter parks
// on its own (frozen) fiber stack. We pass `&op` as the proactor's `user_data`,
// so when the worker reaps the completion it writes op.result and calls
// op.waker.wake(). The fiber resumes and reads op.result. One op is in flight per
// await, and the slot outlives the wait because the suspended fiber's stack is
// frozen until the wake -- exactly like the timer SleepNode (sched-04b).
//
// proactor.h stays sched-agnostic (it only moves u64 tokens); THIS header is the
// io<->sched composition, so it is the one place io/ depends on sched/'s Waker.

namespace hyle::io {

// Result sentinel for "the op was never submitted" (ring/backend rejected the
// submit, or the timed-op heap slot OOM'd) -- distinct from any real negative
// errno. The base ops previously used -1, which COLLIDES with -EPERM (errno 1):
// a caller inspecting the result couldn't tell "never submitted" from a genuine
// EPERM. Real Linux errnos are 1..~133, and -125 (ECANCELED, kCancelled) / -110
// (ETIMEDOUT) are already taken as out-of-band codes; -1000 is safely outside the
// errno range and distinct from those. Callers that only test `result < 0` (the
// common "fail/close" branch) are unaffected; this only makes the value legible.
inline constexpr std::int64_t kSubmitFailed = -1000;

// The per-await slot: a completion's landing pad. Lives on the suspending fiber's
// stack; its address is the proactor `user_data`. The reaper (worker loop, io-04
// integration) fills `result` then fires `waker`. `result` is bytes transferred
// (>= 0), a negative errno, or kSubmitFailed if the op was never submitted.
//
// STACK-SLOT LIFETIME CONTRACT (red-team L5, why the stack slot is safe): the
// reaper dereferences this IoOperation* when the completion lands, so the slot MUST
// outlive its completion. For a BASE op it does, because a base op is NEVER
// cancelled -- it only ever wakes on ITS OWN completion, and the suspended fiber
// stays counted in the scheduler's `outstanding` while it waits. So drain()/
// shutdown() block until the completion arrives (the worker blocks in a GETEVENTS
// reap while ops are in flight), waking the fiber before its frame is reclaimed;
// IoContext's teardown drain then only ever sees TIMED completions (heap slots),
// never a base op's dead stack slot. The ONE way to break this is to destroy a
// fiber with a base op still in flight WITHOUT draining to completion (e.g. giving
// up on a stuck op via drain_for(timeout) then tearing down) -- then a late
// completion writes a freed stack slot (UAF). So: an op whose completion is NOT
// guaranteed (a peer that may never send / a deadline you must bound) MUST use a
// TIMED op (net_timeout.h -- heap slot + refcount survives a late CQE) or a
// cancellable op (net_cancel.h), NOT a bare base op. Base ops assume
// drain-to-completion; that is the contract, not an accident.
struct IoOperation {
  sched::Waker waker;         // whom to wake when this op completes
  std::int64_t result = 0;    // filled by the reaper from Completion::result
  std::uint32_t flags = 0;    // Completion flags (kMore/kBuffer), if the op sets them
  // io-06: set true by complete() when the completion arrives. A plain op ignores
  // it (it only ever wakes on its own completion). A TIMED op (net_timeout.h) reads
  // it after resume to arbitrate io-vs-timer: `completed==true` => io won (use
  // result); `false` => the timer woke us first (io still in flight -> cancel it,
  // return -ETIMEDOUT). Single-threaded per worker (shard-per-core), so no atomic.
  bool completed = false;
};

// Route one reaped completion back to the fiber waiting on it. The reaper calls
// this for every Completion whose user_data is an IoOperation* (io ops encode the
// slot address there). Kept inline + header-only so both the worker loop and a
// standalone test driver use the identical routing. A post() wake (user_data is a
// raw Waker id, not a slot) is NOT routed here -- the reaper tells them apart by
// how the token was minted (io ops use to_token(&op)).
inline void complete(const Completion& c) noexcept {
  auto* op = reinterpret_cast<IoOperation*>(static_cast<std::uintptr_t>(c.user_data));
  op->result = c.result;
  op->flags = c.flags;
  op->completed = true;  // io-06: mark done BEFORE the wake (the timed-op arbiter reads it)
  op->waker.wake();  // safe from any thread; re-injects the fiber (sched-04a)
}

// Encode an IoOperation slot as the proactor user_data token (round-trips
// verbatim through the completion, per proactor.h's contract).
inline std::uint64_t to_token(IoOperation* op) noexcept {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(op));
}

// ---- co_await-able socket ops ----------------------------------------------
// Each awaitable: (1) on await_suspend, capture the running fiber's Waker into a
// stack IoOperation, submit the proactor op with to_token(&op), and fiber-yield;
// (2) the reaper wakes it and fills op.result; (3) await_resume returns result.
// await_suspend returns false-equivalent behaviour via the shared SuspendFiber
// bridge (fiber already waited; resume the coroutine, don't re-park it).
//
// These take a raw proactor + fd: the higher-level TcpStream/TcpListener wrap fd
// ownership on top. A submit failure (ring full / bad handle) surfaces as a
// negative result without ever suspending (there is no completion to wait for).
//
// PLAN A INVARIANT: `p` MUST be the ring of the worker running this fiber (get it
// from IoContext::proactor_for_current()). io_uring/IOCP don't make submit + reap
// on one ring mutually thread-safe across threads, so an op must be submitted on,
// and reaped by, the SAME worker. Passing a foreign worker's ring races the
// reaper. The awaitable is constructed inside the coroutine (on the worker), so
// resolving the ring there picks the right one.

// co_await RecvOp{p, fd, buf, len} -> bytes read (>=0) or negative errno.
// For a DIRECT (registered-file) stream set fixed_slot >= 0: the op then references
// the io_uring file-table SLOT via IOSQE_FIXED_FILE (submit_recv_fixed) instead of a
// raw fd -- the accepted-direct path (no process fd ever existed). fixed_slot < 0
// (default) = the ordinary raw-fd recv.
struct RecvOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  std::uint32_t flags = 0;
  int fixed_slot = -1;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = fixed_slot >= 0
                     ? p->submit_recv_fixed(fixed_slot, buf, len, flags, to_token(&op_))
                     : p->submit_recv(fd, buf, len, flags, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;  // submit failed: no completion coming, resume at once
      return false;
    }
    sched::suspend();  // fiber yields; returns when complete() wakes it
    return false;      // fiber is back -> continue the coroutine
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await SendOp{p, fd, buf, len} -> bytes written (>=0) or negative errno.
struct SendOp {
  Proactor* p;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  std::uint32_t flags = 0;
  int fixed_slot = -1;  // >=0 => direct (registered-file) send via IOSQE_FIXED_FILE

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = fixed_slot >= 0
                     ? p->submit_send_fixed(fixed_slot, buf, len, flags, to_token(&op_))
                     : p->submit_send(fd, buf, len, flags, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await AcceptOp{p, listen_fd} -> accepted fd (>=0) or negative errno.
struct AcceptOp {
  Proactor* p;
  std::intptr_t fd;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = p->submit_accept(fd, nullptr, nullptr, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await ConnectOp{p, fd, addr, addrlen} -> 0 on success or negative errno.
struct ConnectOp {
  Proactor* p;
  std::intptr_t fd;
  const void* addr;
  std::uint32_t addrlen;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = p->submit_connect(fd, addr, addrlen, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await RecvFromOp{p, fd, buf, len, addr, addr_cap} -> bytes received (>=0) or
// negative errno; the sender's address is written into `addr`. `addr` (a
// sockaddr_storage-sized buffer) is caller-owned and must outlive the co_await
// (keep it on the coroutine frame). scratch_ is the io_uring msghdr workspace;
// it lives HERE, on the awaitable (== the frozen fiber stack), for exactly as
// long as the op is in flight -- which is what proactor.h::kDatagramScratch needs.
struct RecvFromOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  void* addr;
  std::uint32_t addr_cap;

  IoOperation op_{};
  alignas(16) unsigned char scratch_[Proactor::kDatagramScratch];
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = p->submit_recvfrom(fd, buf, len, addr, addr_cap, scratch_,
                                    to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await SendToOp{p, fd, buf, len, addr, addrlen} -> bytes sent (>=0) or
// negative errno. `addr` (the destination sockaddr) and `buf` are caller-owned and
// must outlive the co_await. scratch_ as in RecvFromOp.
struct SendToOp {
  Proactor* p;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  const void* addr;
  std::uint32_t addrlen;

  IoOperation op_{};
  alignas(16) unsigned char scratch_[Proactor::kDatagramScratch];
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = p->submit_sendto(fd, buf, len, addr, addrlen, scratch_,
                                  to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

}  // namespace hyle::io
