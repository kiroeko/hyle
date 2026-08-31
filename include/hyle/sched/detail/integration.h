#pragma once

#include "hyle/sched/scheduler.h"

namespace hyle::sched::detail {

// Build-only bridge for HYLE modules that participate in the scheduler worker
// loop. Ordinary users compose IO through io::IoContext and must not mutate
// these single-slot hooks. HyleComponents excludes sched/detail from installed
// SDKs so this type cannot become an accidental public compatibility contract.
class SchedulerIntegration {
 public:
  using PollFn = bool (*)(void* context, unsigned worker_index,
                          unsigned long long timeout_ns) noexcept;
  using WakeFn = void (*)(void* context, unsigned worker_index) noexcept;

  static void set_io_poll(Scheduler& scheduler, PollFn fn,
                          void* context) noexcept {
    scheduler.set_io_poll_(fn, context);
  }

  static void set_io_wake(Scheduler& scheduler, WakeFn fn,
                          void* context) noexcept {
    scheduler.set_io_wake_(fn, context);
  }
};

}  // namespace hyle::sched::detail
