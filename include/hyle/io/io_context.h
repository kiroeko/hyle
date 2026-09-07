#pragma once

#include <cstddef>

#include "hyle/io/net.h"
#include "hyle/io/proactor.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-04 integration (plan A): per-core proactor rings plugged into the sched
// worker loop. IoContext owns ONE Proactor PER WORKER and registers a poll hook
// on the Scheduler; each worker, at the top of its loop, reaps ITS OWN ring
// (passed its worker_idx) and routes each completion back to the fiber that
// awaited it (net::complete -> op.result + Waker::wake()).
//
// Why per-worker rings (sched-design §3.2 "每核一个 io_uring 实例"): io_uring and
// IOCP do not make submit and reap on the same ring mutually thread-safe. If one
// shared ring were reaped from whichever worker ran the loop AND submitted from
// whichever worker ran the awaiting fiber, those would race. Per-core rings make
// submit + the reap of that op's completion happen on the SAME worker thread:
// an awaitable submits on current_worker_index()'s ring, and that same worker's
// loop reaps it. A fiber may migrate between awaits, but each op is wholly owned
// (submit + reap) by one ring/thread for its lifetime. No ring lock needed.
//
// Worker parking is merged with the backend wait: the scheduler passes its
// deadline to reap(), and cross-thread work interrupts the backend through
// Proactor::wake(). There is no 1 ms polling loop.

namespace hyle::io {

struct IoContextOptions {
  // One completion queue per scheduler worker. The value is a capacity hint;
  // backends may round it to a supported queue size.
  unsigned queue_depth = 256;
  mem::AllocatorRef allocator = mem::system_allocator();
  unsigned max_blocking_threads = 512;
  time::Duration blocking_keep_alive = std::chrono::seconds(10);
};

class IoContext {
 public:
  // Stands up exactly one proactor per Scheduler worker and registers the poll
  // hook on `scheduler`. The worker count is deliberately inferred: asking a
  // caller to repeat it creates an invalid state without adding a capability.
  // `scheduler` must outlive this context and must be shut down before context
  // destruction. valid()/status() report recoverable setup failures.
  HYLE_IO_API explicit IoContext(sched::Scheduler& scheduler,
                                 IoContextOptions options = {}) noexcept;
  HYLE_IO_API ~IoContext();
  IoContext(const IoContext&) = delete;
  IoContext& operator=(const IoContext&) = delete;
  IoContext(IoContext&&) = delete;
  IoContext& operator=(IoContext&&) = delete;

  bool valid() const noexcept { return valid_; }
  const Status& status() const noexcept { return status_; }

  // The proactor for a specific worker (used by the poll hook + by awaitables via
  // proactor_for_current()). Index must be < num_workers.
  Proactor& proactor(unsigned worker_idx) noexcept {
    return rings_[worker_idx];
  }

  // The proactor of the worker running the current fiber -- the ring an awaitable
  // must submit on so submit + reap stay on one thread. Precondition: called from
  // a fiber on a worker of the matching scheduler (else the index is -1 and this
  // asserts). This is what net.h ops call to get their ring.
  HYLE_IO_API Proactor& proactor_for_current() noexcept;

  // Bind an fd to `worker_idx`'s ring (shard-per-core): registers it there
  // (needed on IOCP -- a socket associates with exactly one completion port; a
  // no-op on io_uring/epoll) and returns worker_idx so callers can pass it to
  // Scheduler::spawn_on for the connection's fiber. All of that fd's ops then run
  // on that one worker. Returns whether registration succeeded.
  HYLE_IO_API bool bind_fd(unsigned worker_idx, std::intptr_t fd) noexcept;

  unsigned num_workers() const noexcept {
    return ring_count_;
  }

  // Portable file operations use this pool whenever the selected backend cannot
  // complete a regular-file operation asynchronously. Network operations never
  // use it. The pool belongs to the context and is drained on destruction.
  sched::BlockingPool& blocking_pool() noexcept { return blocking_; }

  // Backs only bounded operations whose state must survive an early timeout or
  // cancellation return. Ordinary operations remain stack-only.
  mem::AllocatorRef operation_allocator() const noexcept { return allocator_; }

 private:
  static bool poll_trampoline(void* ctx, unsigned worker_idx,
                              unsigned long long timeout_ns) noexcept;
  static void wake_trampoline(void* ctx, unsigned worker_idx) noexcept;
  bool poll(unsigned worker_idx, unsigned long long timeout_ns) noexcept;
  void destroy_rings() noexcept;

  sched::Scheduler& sched_;
  mem::AllocatorRef allocator_;
  Status status_;
  Proactor* rings_ = nullptr;  // allocator-backed contiguous owner table
  unsigned ring_count_ = 0;
  sched::BlockingPool blocking_;
  bool valid_ = false;
};

}  // namespace hyle::io
