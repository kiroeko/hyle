#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "../common/perf_process.h"
#include "hyle/compute/provider.h"
#include "hyle/sched/bulk.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace hyle::bench::cpu_route {

// The same public, open-runtime synchronous bulk consumer as the correctness
// sample. Affinity and worker-start observations are cold benchmark setup.
class Executor final {
public:
  Executor() : cpu_list_(environment("HYLE_PERF_WORKER_CPUS")) {}
  ~Executor() {
    if (scheduler_) {
      scheduler_->begin_shutdown();
      scheduler_->shutdown();
    }
  }
  bool enabled() const noexcept { return !cpu_list_.empty(); }

  void start() {
    if (!enabled())
      fail("two-worker qualification requires a physical CPU assignment");
    const char *cursor = cpu_list_.c_str();
    for (std::size_t i = 0; i < cpus_.size(); ++i) {
      char *end = nullptr;
      const unsigned long cpu = std::strtoul(cursor, &end, 10);
      if (end == cursor || cpu >= 64 || (i == 0 ? *end != ',' : *end != '\0'))
        fail("expected two distinct supported worker CPUs");
      cpus_[i] = static_cast<unsigned>(cpu);
      cursor = end + (i == 0 ? 1 : 0);
    }
    if (cpus_[0] == cpus_[1])
      fail("worker CPU identities overlap");
    const auto caller_text = environment("HYLE_PERF_CALLER_CPU");
    char *caller_end = nullptr;
    const auto caller_cpu = std::strtoul(caller_text.c_str(), &caller_end, 10);
    if (caller_end == caller_text.c_str() || *caller_end != '\0' ||
        caller_cpu >= 64 || caller_cpu == cpus_[0] || caller_cpu == cpus_[1])
      fail("external caller requires a separate CPU from both workers");
    sched::SchedulerOptions options;
    options.idle_policy = sched::SchedulerIdlePolicy::kLowLatency;
    options.worker_hook_context = this;
    options.on_worker_start = [](void *opaque, unsigned worker) noexcept {
      auto &self = *static_cast<Executor *>(opaque);
      if (worker >= self.cpus_.size() || !pin(self.cpus_[worker]))
        fail("worker affinity could not be established");
      self.started_.fetch_add(1, std::memory_order_release);
    };
    Status status;
    scheduler_ = std::make_unique<sched::Scheduler>(2, mem::system_allocator(),
                                                    64, options, &status);
    if (!status.ok())
      fail("cannot create the caller-owned Scheduler");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (started_.load(std::memory_order_acquire) != 2) {
      if (std::chrono::steady_clock::now() >= deadline)
        fail("worker start did not complete");
      std::this_thread::yield();
    }
  }

  compute::ParallelExecutorRef ref() noexcept {
    return {this, 2, &parallel_for, &in_parallel};
  }

  void print_telemetry(bool peer_first, bool peer_allocation_first,
                       perf::ThreadQosState qos,
                       perf::ThreadAffinityState affinity) const {
    if (enabled() && started_.load(std::memory_order_acquire) != 2)
      fail("parallel measurements did not use both actual worker owners");
    const auto topology = environment("HYLE_PERF_SAMPLING_TOPOLOGY");
    const auto mask = environment("HYLE_PERF_PARENT_MASK");
    const auto priority = environment("HYLE_PERF_PRIORITY");
    std::fprintf(
        stderr,
        "# HYLE_GRAPH_ONEDNN_RUN execution_order=%s "
        "allocation_order=%s apple_qos=%s qos_ok=%s "
        "affinity_topology=%s caller_cpu=%s worker_count=%u "
        "worker_cpus=%s parent_mask=%s priority=%s affinity_ok=%s "
        "idle_policy=low-latency run_nonce=%s\n",
        peer_first ? "peer-first" : "candidate-first",
        peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
        qos.ok, enabled() ? topology.c_str() : affinity.topology, affinity.cpu,
        enabled() ? 2U : 0U, cpu_list_.c_str(), mask.c_str(), priority.c_str(),
        enabled() ? "1" : affinity.ok, perf::run_nonce());
  }

private:
  [[noreturn]] static void fail(const char *message) {
    std::fprintf(stderr, "CPU route executor benchmark: %s\n", message);
    std::abort();
  }
  static std::string environment(const char *name) {
#if defined(_WIN32)
    char *storage = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&storage, &size, name) != 0)
      fail("cannot read benchmark environment");
    std::string value = storage == nullptr ? "" : storage;
    std::free(storage);
    return value;
#else
    const char *value = std::getenv(name);
    return value == nullptr ? "" : value;
#endif
  }
  static bool pin(unsigned cpu) noexcept {
#if defined(_WIN32)
    return ::SetThreadAffinityMask(::GetCurrentThread(), DWORD_PTR{1} << cpu) !=
               0 &&
           ::GetCurrentProcessorNumber() == cpu;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0 &&
           sched_getcpu() == static_cast<int>(cpu);
#else
    (void)cpu;
    return false;
#endif
  }
  static bool in_parallel(void *) noexcept {
    return sched::current_worker_index() >= 0;
  }
  static void parallel_for(void *opaque, std::size_t count,
                           compute::ParallelTaskFn task,
                           void *context) noexcept {
    auto &self = *static_cast<Executor *>(opaque);
    const auto status =
        sched::bulk_execute(*self.scheduler_, 0, count,
                            [&](std::size_t begin, std::size_t end) noexcept {
                              for (std::size_t i = begin; i < end; ++i)
                                task(context, i, count);
                            });
    if (!status.ok())
      fail("open-runtime bulk failed; no partial prefix is replayed");
  }

  std::string cpu_list_;
  std::array<unsigned, 2> cpus_{};
  std::atomic<unsigned> started_{0};
  std::unique_ptr<sched::Scheduler> scheduler_;
};

} // namespace hyle::bench::cpu_route
