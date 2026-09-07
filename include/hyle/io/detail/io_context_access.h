#pragma once

#include "hyle/io/detail/proactor.h"
#include "hyle/io/io_context.h"

namespace hyle::io::detail {

// Source-only access for composed IO operations. The public IoContext surface
// deliberately does not expose Scheduler machinery merely so the implementation
// can fan out pinned connection attempts.
struct IoContextAccess {
  static sched::Scheduler& scheduler(IoContext& context) noexcept {
    return context.sched_;
  }

  static Proactor& proactor(IoContext& context, unsigned worker) noexcept {
    HYLE_FATAL_ASSERT(worker < context.ring_count_);
    return context.rings_[worker];
  }

  static Proactor& proactor_for_current(IoContext& context) noexcept {
    const int worker = sched::current_worker_index();
    HYLE_FATAL_ASSERT(worker >= 0);
    return proactor(context, static_cast<unsigned>(worker));
  }

  static bool bind_handle(IoContext& context, unsigned worker,
                          std::intptr_t handle) noexcept {
    return context.bind_handle(worker, handle);
  }

  static sched::BlockingPool& blocking_pool(IoContext& context) noexcept {
    return context.blocking_;
  }

  static mem::AllocatorRef operation_allocator(
      const IoContext& context) noexcept {
    return context.allocator_;
  }
};

}  // namespace hyle::io::detail
