#pragma once

#include <cstddef>

#include "hyle/io/net.h"
#include "hyle/io/proactor.h"
#include "hyle/sched/scheduler.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-04 integration (plan B): the bridge that plugs the proactor into the sched
// worker loop. IoContext owns a Proactor and registers a poll hook on the
// Scheduler; each worker, at the top of its loop, calls the hook to reap io
// completions and route each one back to the fiber that awaited it (net::complete
// -> op.result + Waker::wake()).
//
// Plan B scope (single shared proactor, non-blocking reap from the hot loop +
// bounded park re-poll -- see scheduler kIoPollNs): correct and simple, but the
// reap is serialized on whichever worker runs it and a completion can wait up to
// one poll interval when all workers are parked. Plan A (per-worker proactor +
// post() wake into the ring so a parked worker is unparked immediately) is the
// follow-up; this interface (a poll hook taking worker_idx) already accommodates
// per-worker rings without an API change.
//
// Thread-safety note (plan B): the single Proactor is reaped from potentially
// different worker threads across loop iterations, and submit_* is called from
// whichever worker runs the awaiting fiber. io_uring/IOCP submit and reap are not
// mutually thread-safe on the same ring, so plan B is currently exercised with a
// SINGLE-worker scheduler in tests; multi-worker safe operation is plan A's
// per-worker ring. This is called out honestly rather than pretended away.

namespace hyle::io {

class IoContext {
 public:
  // Stands up the proactor and registers the poll hook on `sched`. `sched` must
  // outlive this IoContext. Returns via valid() whether the proactor backend came
  // up (false only if the OS refused -- caller should check before submitting).
  HYLE_IO_API explicit IoContext(sched::Scheduler& sched) noexcept;
  HYLE_IO_API ~IoContext();
  HYLE_DISABLE_COPY_AND_MOVE(IoContext);

  bool valid() const noexcept { return proactor_.valid(); }

  // The proactor the co_await ops submit against (net.h awaitables take this).
  Proactor& proactor() noexcept { return proactor_; }

 private:
  // The worker-loop trampoline registered as the Scheduler poll hook. Reaps up to
  // a batch of completions and routes each to its waiting fiber. Static + ctx=this
  // so it matches Scheduler::IoPollFn without naming a bound method.
  static void poll_trampoline(void* ctx, unsigned worker_idx,
                              unsigned long long timeout_ns) noexcept;
  void poll(unsigned long long timeout_ns) noexcept;

  sched::Scheduler& sched_;
  Proactor proactor_;
};

}  // namespace hyle::io
