#include "task_scope_p1_common.h"

#include <span>
#include <utility>

#include "hyle/sched/task_scope.h"

int main() {
  namespace task_scope = hyle::bench::sched_task_scope_p1;
  return task_scope::run_benchmark(
      [](hyle::sched::Scheduler &scheduler, hyle::mem::AllocatorRef allocator,
         std::span<task_scope::Child> children) -> hyle::Status {
        auto created = hyle::sched::TaskScope::create(scheduler, allocator);
        if (!created)
          return std::move(created).error();
        auto scope = *std::move(created);
        for (task_scope::Child &child : children) {
          // The scope owns lifecycle/error/drain after this handle is dropped.
          scope.spawn(task_scope::helper(allocator, &child)).detach();
        }
        return scope.join();
      });
}
