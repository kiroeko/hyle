// FiberStackPool performance certificate through its real scheduler consumer.
//
// Both variants create, schedule, run, drain, and destroy the same batch of
// guarded Fibers. The pooled variant obtains the standard stack tier from the
// Scheduler-owned FiberStackPool and returns it on Fiber destruction. The
// direct variant reserves/releases an otherwise identical guarded VM stack for
// every Fiber. Timing is at the whole-batch boundary; p50/p95/p99 therefore
// include scheduler execution and the cache/VM lifecycle without inserting a
// clock read into individual stack operations.

// This is deliberately not a raw FiberStackPool nanobenchmark: the product
// boundary says the stack cache is scheduler-owned, so only the consuming
// Scheduler + Fiber lifecycle is a valid certification workload.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace {

using hyle::mem::FiberStack;
using hyle::mem::FiberStackPool;
using hyle::mem::kFiberStackDefault;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::Scheduler;

constexpr std::size_t kFibers = 32;

struct Work {
  std::uint64_t value = 0;
};

void fiber_entry(void *arg) noexcept {
  auto *work = static_cast<Work *>(arg);
  work->value = (reinterpret_cast<std::uintptr_t>(arg) >> 4) | 1u;
}

int env_samples() noexcept {
  constexpr int fallback = 2048;
  const char *value = std::getenv("HYLE_MEM_FIBER_SAMPLES");
  if (value == nullptr || value[0] == '\0')
    return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' && parsed >= 100 && parsed <= 100000
             ? static_cast<int>(parsed)
             : fallback;
}

void configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char *requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char *name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_mem_fiber_stack_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int rc = pthread_set_qos_class_self_np(qos, 0);
  std::printf("# HYLE_MEM_FIBER_BENCH apple_main_qos=%s qos_ok=%d "
              "affinity=unsupported\n",
              name, rc == 0 ? 1 : 0);
  if (rc != 0)
    std::abort();
#endif
}

void print_metadata() {
#if defined(__clang__) && defined(_MSC_VER)
  std::printf("HYLE_MEM_FIBER_CERT meta format=1 "
              "compiler=clangcl-%d.%d.%d-msvc%d platform=windows\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__,
              _MSC_VER);
#elif defined(_MSC_VER)
  std::printf("HYLE_MEM_FIBER_CERT meta format=1 compiler=msvc-%d "
              "platform=windows\n",
              _MSC_VER);
#elif defined(__apple_build_version__)
  std::printf("HYLE_MEM_FIBER_CERT meta format=1 "
              "compiler=appleclang-%d.%d.%d-%d platform=macos\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__,
              __apple_build_version__);
#elif defined(__clang__)
  std::printf("HYLE_MEM_FIBER_CERT meta format=1 compiler=clang-%d.%d.%d "
              "platform=linux\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  std::printf("HYLE_MEM_FIBER_CERT meta format=1 compiler=gcc-%d.%d.%d "
              "platform=linux\n",
              __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  std::printf("HYLE_MEM_FIBER_CERT meta format=1 compiler=unknown "
              "platform=unknown\n");
#endif
}

void run_batch(Scheduler &scheduler, FiberStackPool *pool, bool use_pool,
               std::vector<Fiber> &fibers,
               std::array<Work, kFibers> &work) {
  fibers.clear();
  for (Work &item : work)
    item.value = 0;

  for (std::size_t i = 0; i != kFibers; ++i) {
    auto acquired =
        use_pool ? pool->acquire() : FiberStack::allocate(kFiberStackDefault);
    if (!acquired) {
      std::fputs("hyle_mem_fiber_stack_bench: stack allocation failed\n",
                 stderr);
      std::abort();
    }
    FiberStack stack = *std::move(acquired);
    fibers.emplace_back(std::move(stack), &fiber_entry, &work[i],
                        use_pool ? pool : nullptr);
    if (!scheduler.spawn(&fibers.back())) {
      std::fputs("hyle_mem_fiber_stack_bench: scheduler rejected fiber\n",
                 stderr);
      std::abort();
    }
  }
  scheduler.drain();

  std::uint64_t sink = 0;
  for (const Work &item : work)
    sink += item.value;
  if (sink == 0) {
    std::fputs("hyle_mem_fiber_stack_bench: fiber work was not observed\n",
               stderr);
    std::abort();
  }
  // Fiber destruction is part of the measured lifecycle: return the guarded
  // stack to the scheduler pool, or release its VM reservation directly.
  fibers.clear();
}

struct Percentiles {
  std::uint64_t p50;
  std::uint64_t p95;
  std::uint64_t p99;
};

Percentiles summarize(std::vector<std::uint64_t> samples) {
  std::sort(samples.begin(), samples.end());
  auto nearest_rank = [&](std::size_t percentile) {
    const std::size_t rank =
        (percentile * samples.size() + 99) / 100; // ceil(p*N)
    return samples[rank == 0 ? 0 : rank - 1];
  };
  return {nearest_rank(50), nearest_rank(95), nearest_rank(99)};
}

} // namespace

int main() {
  configure_main_benchmark_thread();
  print_metadata();

  const int samples = env_samples();
  Scheduler scheduler(4, system_allocator(), 4096);
  FiberStackPool *pool = scheduler.fiber_stack_pool(kFiberStackDefault);
  if (pool == nullptr) {
    std::fputs("hyle_mem_fiber_stack_bench: scheduler pool unavailable\n",
               stderr);
    return 2;
  }

  std::vector<Fiber> fibers;
  fibers.reserve(kFibers);
  std::array<Work, kFibers> work{};

  // Fill the scheduler-owned cache and warm both dispatch paths before timing.
  for (int i = 0; i != 16; ++i) {
    run_batch(scheduler, pool, true, fibers, work);
    run_batch(scheduler, pool, false, fibers, work);
  }
  if (pool->cached() < kFibers) {
    std::fputs("hyle_mem_fiber_stack_bench: pool did not retain warm stacks\n",
               stderr);
    return 2;
  }

  using Clock = std::chrono::steady_clock;
  auto measure = [&](bool use_pool) {
    const auto begin = Clock::now();
    run_batch(scheduler, pool, use_pool, fibers, work);
    const auto end = Clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count());
  };

  std::vector<std::uint64_t> pooled;
  std::vector<std::uint64_t> direct;
  pooled.reserve(static_cast<std::size_t>(samples));
  direct.reserve(static_cast<std::size_t>(samples));
  for (int i = 0; i != samples; ++i) {
    if ((i & 1) == 0) {
      pooled.push_back(measure(true));
      direct.push_back(measure(false));
    } else {
      direct.push_back(measure(false));
      pooled.push_back(measure(true));
    }
  }

  const Percentiles pooled_result = summarize(std::move(pooled));
  const Percentiles direct_result = summarize(std::move(direct));
  auto print = [&](const char *variant, const Percentiles &result) {
    std::printf("HYLE_MEM_FIBER_CERT latency variant=%s samples=%d fibers=%zu "
                "p50_batch_ns=%llu p95_batch_ns=%llu p99_batch_ns=%llu\n",
                variant, samples, kFibers,
                static_cast<unsigned long long>(result.p50),
                static_cast<unsigned long long>(result.p95),
                static_cast<unsigned long long>(result.p99));
  };
  print("scheduler_pool", pooled_result);
  print("direct_vm", direct_result);

  scheduler.shutdown();
  return 0;
}
