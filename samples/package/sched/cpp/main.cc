#include <atomic>
#include <chrono>

#include <hyle/mem/allocator.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/fiber.h>
#include <hyle/sched/join.h>
#include <hyle/sched/scheduler.h>
#include <hyle/sched/sync.h>
#include <hyle/sched/timer.h>

namespace {

hyle::sched::Task<int> delayed_value(hyle::mem::Allocator,
                                     int value) noexcept {
  hyle::sched::sleep_for(std::chrono::milliseconds(1));
  co_return value;
}

}  // namespace

int main() {
  auto allocator = hyle::mem::system_allocator();
  hyle::sched::SchedulerOptions options;
  options.idle_policy = hyle::sched::SchedulerIdlePolicy::kPowerSaving;
  hyle::sched::Scheduler scheduler(2, allocator, 64, options);

  std::atomic<int> callbacks{0};
  if (!scheduler.schedule([&callbacks]() noexcept {
        callbacks.fetch_add(1, std::memory_order_relaxed);
      }).ok()) {
    return 1;
  }
  if (!scheduler.spawn_detached([&callbacks]() noexcept {
        hyle::sched::Fiber::yield();
        callbacks.fetch_add(1, std::memory_order_relaxed);
      }).ok()) {
    return 2;
  }

  auto task = hyle::sched::spawn_task(
      scheduler, allocator, delayed_value(allocator, 42));
  auto task_result = task.join();
  if (!task_result || *task_result != 42) return 3;

  hyle::sched::Channel<int> channel(allocator, 2);
  if (!channel.try_send(7)) return 4;
  auto message = channel.try_recv();
  if (!message || *message != 7) return 5;

  hyle::sched::BlockingPool blocking(allocator, 2);
  auto blocking_task = hyle::sched::spawn_blocking(
      blocking, allocator, []() noexcept { return 11; });
  auto blocking_result = blocking_task.join();
  if (!blocking_result || *blocking_result != 11) return 6;
  blocking.shutdown();

  scheduler.shutdown();
  return callbacks.load(std::memory_order_relaxed) == 2 ? 0 : 7;
}
