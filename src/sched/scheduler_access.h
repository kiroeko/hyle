#pragma once

#include "hyle/sched/scheduler.h"

namespace hyle::sched {

namespace detail {

class HYLE_HIDDEN SchedulerAccess {
 public:
  // These seams deliberately remain outside the installed product surface.
  // Bulk execution uses identity to avoid borrowing a worker from a different
  // runtime, then distributes only its bounded sibling runners so each can
  // activate a provider without changing ordinary schedule() locality.
  static bool is_current_scheduler(const Scheduler* scheduler) noexcept;
  static Status schedule_bulk_sibling(Scheduler& scheduler, unsigned ordinal,
                                      Scheduler::DetachedFn fn,
                                      void* context) noexcept;
};

}  // namespace detail
}  // namespace hyle::sched
