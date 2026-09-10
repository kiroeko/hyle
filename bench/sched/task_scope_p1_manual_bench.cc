#include "task_scope_p1_common.h"

#include <memory_resource>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "hyle/mem/pmr.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/join.h"
#include "hyle/util/spin_lock.h"

int main() {
  namespace task_scope = hyle::bench::sched_task_scope_p1;
  return task_scope::run_benchmark(
      [](hyle::sched::Scheduler &scheduler, hyle::mem::AllocatorRef allocator,
         std::span<task_scope::Child> children) -> hyle::Status {
        auto token_result = hyle::sched::CancelToken::try_create_root(allocator);
        if (!token_result) return std::move(token_result).error();
        auto group_token = *std::move(token_result);

        hyle::mem::PmrResource resource(allocator);
        std::pmr::vector<hyle::sched::JoinHandle<hyle::Status>> handles(
            &resource);
        handles.reserve(children.size());
        hyle::util::SpinLock admission_lock;
        bool closed = false;
        for (task_scope::Child &child : children) {
          // This is the public pre-v7 way to linearize a runtime-sized handle
          // set with close. The benchmark is single-producer, but the lock and
          // state are still charged because they are required by the G02
          // contract rather than being optional benchmark work.
          std::lock_guard<hyle::util::SpinLock> lock(admission_lock);
          if (closed) {
            return hyle::Status::FailedPrecondition(
                "manual task owner is closed");
          }
          handles.push_back(hyle::sched::spawn_task(
              scheduler, allocator,
              task_scope::helper(allocator, &child, &group_token)));
        }
        {
          std::lock_guard<hyle::util::SpinLock> lock(admission_lock);
          closed = true;
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
