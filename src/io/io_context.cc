#include "hyle/io/io_context.h"

#include "hyle/io/net.h"
#include "hyle/macros.h"
#include "hyle/sched/waker.h"  // current_worker_index()

// io-04 integration (plan A). One proactor per worker; the poll hook reaps the
// caller worker's ring and routes each completion to its awaiting fiber via
// net::complete(). See io_context.h for the per-core-ring rationale.

namespace hyle::io {

IoContext::IoContext(sched::Scheduler& sched, unsigned num_workers) noexcept
    : sched_(sched) {
  HYLE_FATAL_ASSERT(num_workers >= 1);
  rings_.reserve(num_workers);
  valid_ = true;
  for (unsigned i = 0; i < num_workers; ++i) {
    auto p = std::make_unique<Proactor>();
    if (!p->init()) valid_ = false;  // a ring the OS refused -> context invalid
    rings_.push_back(std::move(p));
  }
  sched_.set_io_poll(&IoContext::poll_trampoline, this);
  sched_.set_io_wake(&IoContext::wake_trampoline, this);
}

IoContext::~IoContext() {
  // Unregister before we (and our rings) are destroyed so no worker calls a
  // dangling hook during teardown. The scheduler is expected already stopped
  // (its shutdown() drained fibers) by the time IoContext is destroyed.
  sched_.set_io_wake(nullptr, nullptr);
  sched_.set_io_poll(nullptr, nullptr);
}

Proactor& IoContext::proactor_for_current() noexcept {
  int idx = sched::current_worker_index();
  HYLE_FATAL_ASSERT(idx >= 0);  // must be called from a worker fiber
  return *rings_[static_cast<unsigned>(idx)];
}

bool IoContext::bind_fd(unsigned worker_idx, std::intptr_t fd) noexcept {
  // Associate the fd with this worker's ring. On IOCP this is the (once-only)
  // CreateIoCompletionPort association; on io_uring/epoll register_handle is a
  // no-op returning true. After this, the fd's ops must run on worker_idx (pin
  // the connection fiber there with Scheduler::spawn_on).
  return rings_[worker_idx]->register_handle(fd);
}

bool IoContext::poll_trampoline(void* ctx, unsigned worker_idx,
                                unsigned long long timeout_ns) noexcept {
  return static_cast<IoContext*>(ctx)->poll(worker_idx, timeout_ns);
}

void IoContext::wake_trampoline(void* ctx, unsigned worker_idx) noexcept {
  // Break worker_idx's blocking reap by waking ITS ring (park merge). Writes the
  // ring's wake eventfd (io_uring) / posts its port (IOCP) -- thread-safe, no SQ
  // race. The woken reap returns and the worker re-checks its run queue.
  static_cast<IoContext*>(ctx)->rings_[worker_idx]->wake();
}

bool IoContext::poll(unsigned worker_idx,
                     unsigned long long timeout_ns) noexcept {
  // Reap THIS worker's own ring only (per-core: submit + reap share this thread,
  // no lock). A batch amortizes the reap syscall. Every completion here is an io
  // op whose user_data is an IoOperation*, so net::complete() casts it back,
  // stores the result, and wakes the fiber.
  //
  // Return whether ops were in flight BEFORE we reaped: the scheduler uses this
  // at the park boundary to decide whether to loop (io outstanding -> we may have
  // blocked in the ring harvesting, so re-check the run queue) or park on its
  // futex (nothing outstanding -> a non-blocking poll only, so a cross-thread
  // wake should interrupt an ordinary park, which a reap block cannot). The
  // blocking GETEVENTS reap when timeout_ns > 0 is what actually surfaces
  // io_uring completions that a non-blocking reap(0) never would (the hang fix).
  Proactor& p = *rings_[worker_idx];
  bool had_io = p.in_flight();
  Completion batch[64];
  std::size_t n = p.reap(batch, 64, had_io ? timeout_ns : 0);
  for (std::size_t i = 0; i < n; ++i) {
    // Skip the cross-thread wake token: it carries no IoOperation* (it only broke
    // the block). io_uring/epoll drop it inside reap, but IOCP's wake posts it as a
    // completion, so filter here to defend all backends uniformly.
    if (batch[i].user_data == Proactor::kWakeToken) continue;
    complete(batch[i]);
  }
  return had_io;
}

}  // namespace hyle::io
