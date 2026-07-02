#include "hyle/io/io_context.h"

#include "hyle/io/net.h"

// io-04 integration (plan B). The poll hook reaps a batch of completions and
// routes each back to its awaiting fiber via net::complete(). See io_context.h
// for the plan-B vs plan-A scope split.

namespace hyle::io {

IoContext::IoContext(sched::Scheduler& sched) noexcept : sched_(sched) {
  proactor_.init();
  sched_.set_io_poll(&IoContext::poll_trampoline, this);
}

IoContext::~IoContext() {
  // Unregister before we (and our proactor) are destroyed, so no worker calls a
  // dangling hook during teardown. The scheduler's own shutdown() drains fibers
  // first; by the time IoContext is destroyed the scheduler is expected stopped.
  sched_.set_io_poll(nullptr, nullptr);
}

void IoContext::poll_trampoline(void* ctx, unsigned /*worker_idx*/,
                                unsigned long long timeout_ns) noexcept {
  static_cast<IoContext*>(ctx)->poll(timeout_ns);
}

void IoContext::poll(unsigned long long timeout_ns) noexcept {
  // Reap a batch and route each completion to the fiber waiting on it. A batch
  // amortizes the reap syscall (io_uring for_each_cqe / IOCP GetQueuedExStatusEx).
  // Every completion here is an io op (its user_data is an IoOperation*), so
  // net::complete() casts it back, stores the result, and wakes the fiber.
  Completion batch[64];
  std::size_t n = proactor_.reap(batch, 64, timeout_ns);
  for (std::size_t i = 0; i < n; ++i) complete(batch[i]);
}

}  // namespace hyle::io
