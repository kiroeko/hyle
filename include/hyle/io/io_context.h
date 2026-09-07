#pragma once

#include <cstddef>

#include "hyle/io/capabilities.h"
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

namespace detail {
class Proactor;
struct IoContextAccess;
}

struct IoContextOptions {
  // Maximum admitted user operations per scheduler worker. Backend storage may
  // round upward, but ordinary submissions are backpressured at this logical
  // bound. Cancellation uses bounded control-plane headroom so a full queue can
  // still retire timed/cancelled work.
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

  unsigned num_workers() const noexcept {
    return ring_count_;
  }

  // Snapshot the selected backend and optional accelerators for one worker.
  // Invalid contexts or out-of-range workers return an empty snapshot.  This is
  // observational only: portable transport operations never require branching
  // on the backend.
  HYLE_IO_API IoCapabilities capabilities(unsigned worker = 0) const noexcept;

#if HYLE_OS_LINUX
  // Acquire the explicit Linux acceleration track for one worker.  An empty
  // handle means this context selected the epoll fallback (or is invalid); no
  // io_uring-only operation exists on non-Linux builds.
  HYLE_IO_API IoUringExtension io_uring(unsigned worker = 0) noexcept;
#endif

 private:
  friend struct detail::IoContextAccess;

  static bool poll_trampoline(void* ctx, unsigned worker_idx,
                              unsigned long long timeout_ns) noexcept;
  static void wake_trampoline(void* ctx, unsigned worker_idx) noexcept;
  bool poll(unsigned worker_idx, unsigned long long timeout_ns) noexcept;
  HYLE_IO_API bool bind_handle(unsigned worker_idx,
                               std::intptr_t handle) noexcept;
  void destroy_rings() noexcept;

  sched::Scheduler& sched_;
  mem::AllocatorRef allocator_;
  Status status_;
  detail::Proactor* rings_ = nullptr;  // allocator-backed owner table
  unsigned ring_count_ = 0;
  sched::BlockingPool blocking_;
  bool valid_ = false;
};

}  // namespace hyle::io
