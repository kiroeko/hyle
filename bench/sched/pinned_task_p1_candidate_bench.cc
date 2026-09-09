#include "pinned_task_p1_common.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

namespace {

namespace common = hyle::bench::sched_p1;
namespace pinned = hyle::bench::sched_pinned_task_p1;

struct Slot {
  unsigned worker = 0;
  hyle::sched::Waker waker{};
  std::atomic<std::size_t> published{0};
  std::atomic<std::size_t> completed{0};
  std::atomic<unsigned>* failures = nullptr;
  common::WorkerHits* hits = nullptr;
};

hyle::sched::Task<std::uint64_t> pinned_task(
    hyle::mem::AllocatorRef, Slot* slot) noexcept {
  constexpr std::size_t total =
      pinned::kWarmupEpisodes + pinned::kMeasuredEpisodes;
  std::uint64_t checksum = 0;
  for (std::size_t episode = 0; episode < total; ++episode) {
    if (hyle::sched::current_worker_index() !=
        static_cast<int>(slot->worker)) {
      slot->failures->fetch_add(1, std::memory_order_relaxed);
    }
    slot->waker = co_await hyle::sched::CaptureWaker{};
    if (!slot->waker.valid())
      slot->failures->fetch_add(1, std::memory_order_relaxed);
    slot->published.store(episode + 1, std::memory_order_release);
    slot->published.notify_one();
    co_await hyle::sched::SuspendFiber{};
    if (hyle::sched::current_worker_index() !=
        static_cast<int>(slot->worker)) {
      slot->failures->fetch_add(1, std::memory_order_relaxed);
    }
    if (episode >= pinned::kWarmupEpisodes) {
      const std::size_t measured = episode - pinned::kWarmupEpisodes;
      slot->hits->hit(hyle::sched::current_worker_index());
      checksum += pinned::episode_value(slot->worker, measured);
    }
    slot->completed.store(episode + 1, std::memory_order_release);
    slot->completed.notify_one();
  }
  co_return checksum;
}

}  // namespace

int main() {
  const auto cpus = common::requested_cpus();
  common::AffinityState affinity(cpus);
  hyle::sched::SchedulerOptions options;
  options.worker_cpu_set = std::span<const unsigned>(cpus);
  options.worker_config_policy = hyle::platform::ThreadConfigPolicy::kRequired;
  options.on_worker_start = +[](void* raw, unsigned worker) noexcept {
    static_cast<common::AffinityState*>(raw)->record(worker,
                                                     common::current_cpu());
  };
  options.worker_hook_context = &affinity;

  const auto allocator = hyle::mem::system_allocator();
  hyle::Status scheduler_status;
  hyle::sched::Scheduler scheduler(
      static_cast<unsigned>(common::kWorkerCount), allocator, 256, options,
      &scheduler_status);
  if (!scheduler_status.ok()) common::fail("HYLE scheduler construction failed");
  affinity.await_ready();
  if (!affinity.exact()) common::fail("HYLE worker placement is not exact");

  std::array<Slot, common::kWorkerCount> slots{};
  std::array<hyle::sched::JoinHandle<std::uint64_t>,
             common::kWorkerCount>
      handles{};
  std::atomic<unsigned> failures{0};
  common::WorkerHits hits;
  for (std::size_t worker = 0; worker < slots.size(); ++worker) {
    slots[worker].worker = static_cast<unsigned>(worker);
    slots[worker].failures = &failures;
    slots[worker].hits = &hits;
    handles[worker] = hyle::sched::spawn_task_on(
        scheduler, static_cast<unsigned>(worker), allocator,
        pinned_task(allocator, &slots[worker]));
  }

  common::Clock::time_point start{};
  constexpr std::size_t total =
      pinned::kWarmupEpisodes + pinned::kMeasuredEpisodes;
  for (std::size_t episode = 0; episode < total; ++episode) {
    for (const Slot& slot : slots)
      common::wait_at_least(slot.published, episode + 1);
    if (episode == pinned::kWarmupEpisodes) start = common::Clock::now();
    for (Slot& slot : slots) slot.waker.wake();
    for (const Slot& slot : slots)
      common::wait_at_least(slot.completed, episode + 1);
  }

  std::uint64_t checksum = 0;
  for (auto& handle : handles) {
    auto result = handle.join();
    if (!result) common::fail("spawn_task_on failed to join");
    checksum += *result;
  }
  const auto finish = common::Clock::now();

  const std::uint64_t callbacks =
      pinned::kMeasuredEpisodes * common::kWorkerCount;
  if (failures.load(std::memory_order_relaxed) != 0 ||
      hits.total() != callbacks || checksum != pinned::expected_checksum()) {
    common::fail("spawn_task_on violated owner, completion or result contract");
  }
  const double wall_ns = common::elapsed_ns(start, finish);
  common::print_measurement(
      "pinned-structured-task", static_cast<double>(callbacks) * 1e9 / wall_ns,
      pinned::kMeasuredEpisodes, common::kWorkerCount, callbacks, checksum,
      wall_ns, hits, affinity);

  scheduler.shutdown();
  common::print_run_telemetry(cpus);
  return 0;
}
