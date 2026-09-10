#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "hyle/mem/fiber_stack.h"

namespace hyle::sched {

// A bounded execution hint, deliberately smaller than an application request
// policy. The scheduler chooses the highest priority that is ready at each
// work boundary; it never preempts a task that is already running.
enum class ExecutionPriority : std::uint8_t {
  kLow = 0,
  kNormal = 1,
  kHigh = 2,
};

namespace detail {
constexpr bool is_valid_execution_priority(
    ExecutionPriority priority) noexcept {
  return priority == ExecutionPriority::kLow ||
         priority == ExecutionPriority::kNormal ||
         priority == ExecutionPriority::kHigh;
}
}  // namespace detail

// Normal-only keeps the existing single-lane runtime. Three-lane mode opts the
// Scheduler into high/low sidecar queues and priority-aware selection.
enum class SchedulerPriorityMode : std::uint8_t {
  kNormalOnly = 0,
  kThreeLanes = 1,
};

// An absent priority inherits the current task when submission targets the
// same priority-enabled Scheduler. From an external thread or across Scheduler
// instances it resolves to normal.
struct ExecutionOptions {
  std::optional<ExecutionPriority> priority;
};

struct FiberLaunchOptions {
  std::optional<ExecutionPriority> priority;
  std::size_t stack_size = mem::kFiberStackDefault;
};

}  // namespace hyle::sched
