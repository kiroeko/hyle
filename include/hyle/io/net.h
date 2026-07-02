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

// The per-await slot: a completion's landing pad. Lives on the suspending fiber's
// stack; its address is the proactor `user_data`. The reaper (worker loop, io-04
// integration) fills `result` then fires `waker`. `result` is bytes transferred
// (>= 0) or a negative errno, the same convention as Completion::result.
struct IoOperation {
  sched::Waker waker;         // whom to wake when this op completes
  std::int64_t result = 0;    // filled by the reaper from Completion::result
  std::uint32_t flags = 0;    // Completion flags (kMore/kBuffer), if the op sets them
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
struct RecvOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  std::uint32_t flags = 0;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = p->submit_recv(fd, buf, len, flags, to_token(&op_));
    if (!submitted_) {
      op_.result = -1;  // submit failed: no completion coming, resume at once
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

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    submitted_ = p->submit_send(fd, buf, len, flags, to_token(&op_));
    if (!submitted_) {
      op_.result = -1;
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
      op_.result = -1;
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
      op_.result = -1;
      return false;
    }
    sched::suspend();
    return false;
  }
  std::int64_t await_resume() noexcept { return op_.result; }
};

}  // namespace hyle::io
