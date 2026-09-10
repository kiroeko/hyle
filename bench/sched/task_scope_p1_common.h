#pragma once

#include "p1_common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace hyle::bench::sched_task_scope_p1 {

namespace common = hyle::bench::sched_p1;

inline constexpr std::size_t kWarmupRequests = 64;
inline constexpr std::size_t kMeasuredRequests = 1'024;
inline constexpr std::size_t kFanout = 32;
inline constexpr std::size_t kFailurePeriod = 16;

struct Child {
  std::uint64_t seed = 0;
  std::uint64_t output = 0;
  bool fail = false;
  common::WorkerHits *hits = nullptr;
};

constexpr std::uint64_t child_value(std::uint64_t seed) noexcept {
  seed ^= seed >> 30;
  seed *= 0xbf58476d1ce4e5b9ULL;
  seed ^= seed >> 27;
  seed *= 0x94d049bb133111ebULL;
  return seed ^ (seed >> 31);
}

inline hyle::sched::Task<hyle::Status> helper(hyle::mem::AllocatorRef,
                                              Child *child) noexcept {
  // A real helper is not guaranteed to finish in its first dispatch. Both
  // routes pay the same cooperative yield and scheduler reinjection.
  hyle::sched::Fiber::yield();
  child->hits->hit(hyle::sched::current_worker_index());
  child->output = child_value(child->seed);
  if (child->fail)
    co_return hyle::Status::Unavailable();
  co_return hyle::Status{};
}

template <class RunRequest> int run_benchmark(RunRequest &&run_request) {
  const auto cpus = common::requested_cpus();
  common::AffinityState affinity(cpus);
  hyle::sched::SchedulerOptions options;
  options.worker_cpu_set = std::span<const unsigned>(cpus);
  options.worker_config_policy = hyle::platform::ThreadConfigPolicy::kRequired;
  options.on_worker_start = +[](void *raw, unsigned worker) noexcept {
    static_cast<common::AffinityState *>(raw)->record(worker,
                                                      common::current_cpu());
  };
  options.worker_hook_context = &affinity;

  const hyle::mem::AllocatorRef allocator = hyle::mem::system_allocator();
  hyle::Status scheduler_status;
  hyle::sched::Scheduler scheduler(static_cast<unsigned>(common::kWorkerCount),
                                   allocator, 512, options, &scheduler_status);
  if (!scheduler_status.ok())
    common::fail("Scheduler construction failed");
  affinity.await_ready();
  if (!affinity.exact())
    common::fail("Scheduler worker placement is not exact");

  common::WorkerHits hits;
  auto run_phase = [&](std::size_t first_request,
                       std::size_t request_count) -> std::uint64_t {
    std::array<Child, kFanout> children{};
    std::uint64_t checksum = 0;
    for (std::size_t local = 0; local < request_count; ++local) {
      const std::size_t request = first_request + local;
      const bool expected_failure = request % kFailurePeriod == 7;
      for (std::size_t child = 0; child < children.size(); ++child) {
        children[child] =
            Child{(static_cast<std::uint64_t>(request) << 32) |
                      static_cast<std::uint64_t>(child + 1),
                  0, expected_failure && child == children.size() / 2, &hits};
      }

      const hyle::Status status =
          run_request(scheduler, allocator, std::span<Child>(children));
      if (expected_failure ? status.code() != hyle::StatusCode::Unavailable
                           : !status.ok()) {
        common::fail(
            "dynamic helper owner returned the wrong aggregate status");
      }
      for (const Child &child : children) {
        const std::uint64_t expected = child_value(child.seed);
        if (child.output != expected)
          common::fail("dynamic helper output was not fully drained");
        checksum += child.output;
      }
    }
    return checksum;
  };

  (void)run_phase(0, kWarmupRequests);
  hits.reset();
  const auto start = common::Clock::now();
  const std::uint64_t checksum = run_phase(kWarmupRequests, kMeasuredRequests);
  const auto finish = common::Clock::now();

  constexpr std::uint64_t callbacks = kMeasuredRequests * kFanout;
  const double wall_ns = common::elapsed_ns(start, finish);
  if (hits.total() != callbacks)
    common::fail("dynamic helper callback count is incomplete");
  common::print_measurement(
      "dynamic-task-scope",
      static_cast<double>(kMeasuredRequests) * 1e9 / wall_ns, kMeasuredRequests,
      kFanout, callbacks, checksum, wall_ns, hits, affinity);

  scheduler.shutdown();
  common::print_run_telemetry(cpus);
  return 0;
}

} // namespace hyle::bench::sched_task_scope_p1
