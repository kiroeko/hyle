#include "task_scope_p1_common.h"

#include <memory_resource>
#include <span>
#include <vector>

#include "hyle/mem/pmr.h"
#include "hyle/sched/join.h"

int main() {
  namespace task_scope = hyle::bench::sched_task_scope_p1;
  return task_scope::run_benchmark(
      [](hyle::sched::Scheduler &scheduler, hyle::mem::AllocatorRef allocator,
         std::span<task_scope::Child> children) -> hyle::Status {
        hyle::mem::PmrResource resource(allocator);
        std::pmr::vector<hyle::sched::JoinHandle<hyle::Status>> handles(
            &resource);
        handles.reserve(children.size());
        for (task_scope::Child &child : children) {
          handles.push_back(hyle::sched::spawn_task(
              scheduler, allocator, task_scope::helper(allocator, &child)));
        }

        hyle::Status first_error;
        for (auto &handle : handles) {
          auto result = handle.join();
          if (!result) {
            first_error.Update(result.error());
          } else {
            // Copying here matches TaskScope retaining the first Status while
            // the JoinState still owns the child's typed value.
            first_error.Update(*result);
          }
        }
        return first_error;
      });
}
