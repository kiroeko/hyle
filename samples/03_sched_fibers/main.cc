// Sample 03 -- sched: run cooperative fibers on the work-stealing scheduler.
//
// The core concurrency primitive: a Fiber is a stackful coroutine the Scheduler runs on
// a pool of OS-thread workers (M:N). Fibers cooperatively yield(); the scheduler may
// resume a yielded fiber on a DIFFERENT worker (work-stealing). This sample spawns N
// fibers that each do a few yield()-separated steps, and confirms they all complete.
//
// SAFETY (usage-contract.md): a fiber body must NOT hold a thread_local-derived value
// across yield() -- it may resume on another thread. Re-read such state after yield.
//
// Platform: needs a supported HYLE fcontext backend. See CMakeLists.txt.
// Build: see ../README.md.

#include <atomic>
#include <cstdio>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;

namespace {
struct Work {
  int steps;
  std::atomic<int>* total_steps;
  std::atomic<int>* completed;
};

// A fiber entry: do `steps` units of work, yielding between each so other fibers run.
void fiber_body(void* arg) {
  auto* w = static_cast<Work*>(arg);
  for (int i = 0; i < w->steps; ++i) {
    w->total_steps->fetch_add(1, std::memory_order_relaxed);
    Fiber::yield();  // cooperative hand-off; may resume on another worker
  }
  w->completed->fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

int main() {
  constexpr int kFibers = 100;
  constexpr int kSteps = 5;
  std::atomic<int> total_steps{0};
  std::atomic<int> completed{0};

  std::vector<Work> work(kFibers, Work{kSteps, &total_steps, &completed});
  {
    // 4 workers, system allocator, run-queue capacity 256 (power of two).
    Scheduler sched(4, system_allocator(), 256);
    bool submitted_all = true;
    for (int i = 0; i < kFibers; ++i) {
      // Fire-and-forget callbacks that may yield use spawn_detached: the scheduler
      // owns the Fiber and recycles its guarded FiberStack after completion. Raw
      // Fiber construction is for callers that specifically need manual ownership.
      if (!sched
               .spawn_detached(&fiber_body, &work[i], kDefaultStackSize)
               .ok()) {
        std::printf("failed to submit fiber %d\n", i);
        submitted_all = false;
        break;
      }
    }
    sched.shutdown();  // drain: wait until every fiber finishes, then join workers
    if (!submitted_all) return 1;
  }

  std::printf("fibers spawned : %d\n", kFibers);
  std::printf("fibers done    : %d\n", completed.load());
  std::printf("total steps    : %d (expected %d)\n", total_steps.load(), kFibers * kSteps);
  return (completed.load() == kFibers && total_steps.load() == kFibers * kSteps) ? 0 : 1;
}
