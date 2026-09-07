#pragma once

#include <cstdint>

#include "hyle/io/detail/net.h"
#include "hyle/io/detail/net_timeout.h"  // kCancelToken (the cancel-op's own-CQE drop token)
#include "hyle/io/detail/proactor.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/waker.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// sched-07 / io: CANCELLABLE io ops -- recv/accept bound to a CancelToken. When the
// token is cancelled, an in-flight kernel op is actively cancelled (submit_cancel:
// io_uring ASYNC_CANCEL / IOCP CancelIoEx / epoll synth -ECANCELED) so a fiber
// blocked on a dead peer wakes promptly with -ECANCELED. This is the structured-
// cancellation counterpart to the TIMED ops in net_timeout.h: there a timer wins the
// race; here a token cancel (propagated from an ancestor -- e.g. the whole HTTP
// request) does.
//
// LIFETIME (simpler than net_timeout's heap slot -- no orphan). A cancellable op has
// exactly ONE waker: the op's own completion. A cancel does NOT wake the fiber; it
// only asks the proactor to cancel the in-flight op, which then completes (-ECANCELED)
// and THAT wakes the fiber (the single wake). So the slot lives on the fiber stack:
// the fiber never resumes until its op's completion (normal or -ECANCELED) lands, and
// the stack is frozen until then. No refcount, no heap.
//
// The cancel op's OWN CQE (io_uring emits one, result 0/-ENOENT/-EALREADY) is
// submitted under Proactor::kWakeToken, which the completion router (io_context::poll)
// already DROPS -- so no new routing is needed and it is never dereferenced as a slot.
//
// SHARD-PER-CORE: the fiber is pinned to its owner worker; the CancelCallback fires on
// that SAME worker (the token tree is owner-worker-only), submit + reap are on that
// worker's ring. Callback-fires-submit_cancel, the reap of -ECANCELED, and the resume
// all run single-threaded -- no atomics on the arbitration path.
//
// The CancelCallback is registered only for the duration of the await (it lives on the
// await_suspend frame, which is frozen on the fiber stack across the suspend) and is
// unregistered on resume: if the op completes NORMALLY first, its cancel hook is torn
// down before the fiber returns, so a later token cancel won't submit_cancel a
// completed op (or a reused fd).

namespace hyle::io::detail {

// CancelCallback context for a cancellable op: on token cancel, ask the proactor to
// cancel the in-flight op named by `token` (its user_data). Owner-worker-only, no
// atomics. `done` guards a double submit (cannot happen on one shard, but documents
// that a cancel is a single edge).
struct CancelOpCtx {
  Proactor* p = nullptr;
  std::uint64_t token = 0;  // the in-flight op's user_data (== to_token(&op))
  bool done = false;

  static void on_cancel(void* ctx) noexcept {
    auto* c = static_cast<CancelOpCtx*>(ctx);
    if (c->done) return;
    c->done = true;
    // Cancel the in-flight op. The cancel op's own CQE uses kCancelToken -- NOT
    // kWakeToken: the wake token is the park-merge wake-read's reserved value, and a
    // reap seeing it (a) skips the inflight decrement (treating it as permanent infra)
    // and (b) RE-ARMS the wake-read. Using it for the cancel op's CQE therefore leaked
    // inflight by 1 per cancel AND spuriously re-armed the wake-read -- benign under
    // COOP_TASKRUN but a livelock under DEFER_TASKRUN (the spurious re-arm's completion
    // loops on the GETEVENTS-only delivery). kCancelToken routes through complete_timed
    // (dropped there) and reaps as an ORDINARY completion, so inflight decrements
    // correctly. A false return (op already completed) is benign -- its normal
    // completion is already on the way and will wake the fiber.
    c->p->submit_cancel(c->token, kCancelToken);
  }
};

// co_await CancelRecvOp{p, fd, buf, len, token} -> bytes read (>=0), negative errno,
// or -ECANCELED (-125) if the token was cancelled while the recv was in flight (or
// before it started). `token` must outlive the co_await (it lives on the connection/
// request frame above this awaitable). Awaited from the fiber pinned to the fd owner.
struct CancelRecvOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  const sched::CancelToken* token;
  std::uint32_t flags = 0;

  IoOperation op_{};
  CancelOpCtx cctx_{};
  bool submitted_ = false;

  bool await_ready() noexcept {
    // Already cancelled before we submit -> skip the ring entirely, resume with
    // kCancelled (recorded in op_.result so await_resume returns it).
    if (token != nullptr && token->is_cancelled()) {
      op_.result = kCancelled;
      return true;
    }
    return false;
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    cctx_.p = p;
    cctx_.token = to_token(&op_);
    submitted_ = p->submit_recv(fd, buf, len, flags, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;  // submit failed: no completion coming, resume at once
      return false;
    }
    // Register the cancel hook (RAII: unregistered when this frame unwinds after
    // resume). If the token is cancelled between now and suspend -- or is already
    // cancelled -- the callback fires here and submit_cancel runs at once, queuing
    // the -ECANCELED that will wake us. Lives on this (frozen) await_suspend frame.
    if (token != nullptr) {
      sched::CancelCallback cb{*token, &CancelOpCtx::on_cancel, &cctx_};
      sched::suspend();  // cb remains registered across the fiber suspension
    } else {
      // A null token means "no cancellation" and behaves like the base op. The
      // await_ready/await_resume null checks already promised this; dereferencing
      // it here made that apparent contract crash.
      sched::suspend();
    }
    return false;      // fiber is back -> continue the coroutine (cb unregisters)
  }
  std::int64_t await_resume() noexcept {
    // Normalize a cancelled result: if the token was cancelled and the op came back
    // with a (backend-specific) negative code, report the canonical kCancelled so the
    // caller need not know -ECANCELED vs -ERROR_OPERATION_ABORTED.
    if (op_.result < 0 && token != nullptr && token->is_cancelled()) return kCancelled;
    return op_.result;
  }
};

struct CancelSendOp {
  Proactor* p;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  const sched::CancelToken* token;
  std::uint32_t flags = 0;

  IoOperation op_{};
  CancelOpCtx cctx_{};
  bool submitted_ = false;

  bool await_ready() noexcept {
    if (token != nullptr && token->is_cancelled()) {
      op_.result = kCancelled;
      return true;
    }
    return false;
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    cctx_.p = p;
    cctx_.token = to_token(&op_);
    submitted_ = p->submit_send(fd, buf, len, flags, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    if (token != nullptr) {
      sched::CancelCallback callback{
          *token, &CancelOpCtx::on_cancel, &cctx_};
      sched::suspend();
    } else {
      sched::suspend();
    }
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (op_.result < 0 && token != nullptr && token->is_cancelled())
      return kCancelled;
    return op_.result;
  }
};

// co_await CancelAcceptOp{p, listen_fd, token} -> accepted fd (>=0), negative errno,
// or -ECANCELED. Lets a server's accept loop be cancelled as part of a subtree (or on
// shutdown) rather than blocking forever on a quiet listener.
struct CancelAcceptOp {
  Proactor* p;
  std::intptr_t fd;
  const sched::CancelToken* token;

  IoOperation op_{};
  CancelOpCtx cctx_{};
  bool submitted_ = false;

  bool await_ready() noexcept {
    if (token != nullptr && token->is_cancelled()) {
      op_.result = kCancelled;
      return true;
    }
    return false;
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    cctx_.p = p;
    cctx_.token = to_token(&op_);
    submitted_ = p->submit_accept(fd, nullptr, nullptr, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    if (token != nullptr) {
      sched::CancelCallback cb{*token, &CancelOpCtx::on_cancel, &cctx_};
      sched::suspend();
    } else {
      sched::suspend();
    }
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (op_.result < 0 && token != nullptr && token->is_cancelled()) return kCancelled;
    return op_.result;
  }
};

struct CancelConnectOp {
  Proactor* p;
  std::intptr_t fd;
  const void* addr;
  std::uint32_t addrlen;
  const sched::CancelToken* token;

  IoOperation op_{};
  CancelOpCtx cctx_{};
  bool submitted_ = false;

  bool await_ready() noexcept {
    if (token != nullptr && token->is_cancelled()) {
      op_.result = kCancelled;
      return true;
    }
    return false;
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    cctx_.p = p;
    cctx_.token = to_token(&op_);
    submitted_ = p->submit_connect(fd, addr, addrlen, to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    if (token != nullptr) {
      sched::CancelCallback callback{
          *token, &CancelOpCtx::on_cancel, &cctx_};
      sched::suspend();
    } else {
      sched::suspend();
    }
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (op_.result < 0 && token != nullptr && token->is_cancelled())
      return kCancelled;
    return op_.result;
  }
};

struct CancelRecvFromOp {
  Proactor* p;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  void* addr;
  std::uint32_t addr_cap;
  const sched::CancelToken* token;

  IoOperation op_{};
  CancelOpCtx cctx_{};
  alignas(16) unsigned char scratch_[Proactor::kDatagramScratch]{};
  bool submitted_ = false;

  bool await_ready() noexcept {
    if (token != nullptr && token->is_cancelled()) {
      op_.result = kCancelled;
      return true;
    }
    return false;
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    cctx_.p = p;
    cctx_.token = to_token(&op_);
    submitted_ = p->submit_recvfrom(fd, buf, len, addr, addr_cap, scratch_,
                                    to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    if (token != nullptr) {
      sched::CancelCallback callback{
          *token, &CancelOpCtx::on_cancel, &cctx_};
      sched::suspend();
    } else {
      sched::suspend();
    }
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (op_.result < 0 && token != nullptr && token->is_cancelled())
      return kCancelled;
    return op_.result;
  }
};

struct CancelSendToOp {
  Proactor* p;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  const void* addr;
  std::uint32_t addrlen;
  const sched::CancelToken* token;

  IoOperation op_{};
  CancelOpCtx cctx_{};
  alignas(16) unsigned char scratch_[Proactor::kDatagramScratch]{};
  bool submitted_ = false;

  bool await_ready() noexcept {
    if (token != nullptr && token->is_cancelled()) {
      op_.result = kCancelled;
      return true;
    }
    return false;
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    op_.waker = sched::current_waker();
    cctx_.p = p;
    cctx_.token = to_token(&op_);
    submitted_ = p->submit_sendto(fd, buf, len, addr, addrlen, scratch_,
                                  to_token(&op_));
    if (!submitted_) {
      op_.result = kSubmitFailed;
      return false;
    }
    if (token != nullptr) {
      sched::CancelCallback callback{
          *token, &CancelOpCtx::on_cancel, &cctx_};
      sched::suspend();
    } else {
      sched::suspend();
    }
    return false;
  }
  std::int64_t await_resume() noexcept {
    if (op_.result < 0 && token != nullptr && token->is_cancelled())
      return kCancelled;
    return op_.result;
  }
};

}  // namespace hyle::io::detail
