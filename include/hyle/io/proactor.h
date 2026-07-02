#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// Proactor: the platform-neutral COMPLETION-reaping seam (io-design "模块形状",
// sched-design §3.2/§3.3). One proactor per worker. It is NOT the socket/file
// API (that is io-04 net/); it is the thin layer that (a) lets a backend record
// "operation X, when it completes, should wake THIS token" and (b) lets the
// worker loop reap finished operations and hand their tokens back so the caller
// wakes the fibers.
//
// The completion->fiber mapping is io-design §3.3's physical mechanism, verified
// on this machine before this interface was written: every backend carries a
// 64-bit user_data with the submitted op (io_uring SQE user_data / IOCP
// completion key+OVERLAPPED), and returns it verbatim on completion. We store a
// `Completion` token there. sched/ owns the Waker; io/ only moves tokens.
//
// Backends (io-02+): io_uring (Linux, per-core ring), IOCP (Windows, per-worker
// port), epoll/kqueue (reactor -> completion emulation). This header is what
// they all present upward; the consumer (a sched worker) is backend-agnostic.

namespace hyle::io {

// A finished operation handed back by the proactor. `user_data` is the exact
// 64-bit value the submitter attached (typically a Waker pointer / id, cast to
// u64); `result` is the op's outcome (bytes transferred, or a negative errno on
// failure -- the same convention io_uring CQE `res` uses, IOCP maps onto it).
//
// `flags` mirrors io_uring's `cqe->flags` (IOCP synthesizes it). It is NOT
// optional dressing -- the io_uring SOTA mechanisms REQUIRE it (verified against
// liburing v2.15):
//   - kMore: this op will produce MORE completions (multishot accept/recv set
//     IORING_CQE_F_MORE). One submit -> many completions sharing one user_data;
//     the caller must NOT recycle the token/Waker until a completion arrives
//     with kMore CLEAR. A "1 submit = 1 completion" abstraction would corrupt
//     multishot -- so the flag lives here from day one.
//   - kBuffer + buffer_id: the kernel picked a provided/ring buffer (zero-copy
//     recv); IORING_CQE_F_BUFFER, id in the high bits. Surfaced so net/ can
//     recover which buffer without a side table.
// Backends that lack a notion (IOCP) just leave flags 0 / buffer_id 0.
struct Completion {
  std::uint64_t user_data = 0;
  std::int64_t result = 0;
  std::uint32_t flags = 0;      // backend completion flags (see kMore/kBuffer)
  std::uint32_t buffer_id = 0;  // provided-buffer id when (flags & kBuffer)

  enum : std::uint32_t {
    kMore = 1u << 0,    // more completions coming for this user_data (multishot)
    kBuffer = 1u << 1,  // buffer_id is valid (kernel-selected zero-copy buffer)
  };
};

// Per-worker proactor. Not thread-safe across workers by design: each worker
// owns one (its own IOCP port / io_uring ring), submits and reaps on its own
// thread -- the per-core philosophy that matches the run-queue (sched-design
// §3.2). Cross-thread submission (a foreign thread arming io) is a backend
// concern handled where it arises, not by locking this object.
//
// API macro on the methods (not the class): backends hold OS handles / std
// members; method-level export keeps the layout backend-side and the seam clean
// (same pattern as Parker/Scheduler).
class Proactor {
 public:
  Proactor() noexcept = default;
  HYLE_DISABLE_COPY_AND_MOVE(Proactor);
  HYLE_IO_API ~Proactor();

  // Create the backend's completion object (IOCP port / io_uring ring). Returns
  // false if the OS refused (out of fds, unsupported kernel). Call once before use.
  HYLE_IO_API bool init(unsigned entries = 256) noexcept;

  // Reap up to `max` finished operations into `out` (caller-provided array),
  // returning how many were written. `timeout_ns` bounds the wait: 0 = poll and
  // return immediately (the worker's non-blocking loop step), UINT64_MAX = block
  // until at least one completes. This is GetQueuedCompletionStatusEx / io_uring
  // cq peek behind one signature. The caller then wakes each completion's fiber.
  HYLE_IO_API std::size_t reap(Completion* out, std::size_t max,
                               std::uint64_t timeout_ns) noexcept;

  // Manually post a completion carrying `user_data` (result defaults to 0). This
  // is the backend-independent "inject a wake into the completion stream" hook --
  // used to unblock a parked reap() from another thread, and to drive tests
  // without a real fd. (IOCP PostQueuedCompletionStatus / io_uring you'd use an
  // eventfd/nop; both present it here.)
  HYLE_IO_API bool post(std::uint64_t user_data, std::int64_t result = 0) noexcept;

  bool valid() const noexcept { return impl_ != nullptr; }

 private:
  struct Impl;
  Impl* impl_ = nullptr;  // backend state (IOCP port / io_uring ring), pimpl
};

}  // namespace hyle::io
