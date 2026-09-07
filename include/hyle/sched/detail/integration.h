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

  // A Scheduler owns exactly one IO driver slot. Claim both callbacks as one
  // identity so a second IoContext cannot silently replace half of a live
  // integration. Detach is conditional on that same identity.
  static bool attach_io(Scheduler& scheduler, PollFn poll, WakeFn wake,
                        void* context) noexcept {
    return scheduler.attach_io_(poll, wake, context);
  }

  static void detach_io(Scheduler& scheduler, void* context) noexcept {
    scheduler.detach_io_(context);
  }
};

}  // namespace hyle::sched::detail
