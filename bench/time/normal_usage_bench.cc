// Normal public-entry benchmark for time. The algorithm-only N=4096/rival
// certificate stays in timer_queue_bench.cc; this executable covers the shapes
// users actually write: embedded/scattered nodes, incremental event-loop work,
// schedule_after including its clock read, and sleep latency.
#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "hyle/time/timer_queue.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::time::TimerEntry;
using hyle::time::TimerQueue;

constexpr std::size_t kIncrementalTimers = 64;
constexpr std::uint64_t kTickNs = 100'000;

nb::Bench make_bench() {
  nb::Bench bench;
#if defined(__APPLE__)
  // QoS is a soft scheduling contract, not affinity. A longer epoch averages
  // over macOS millisecond scheduler bursts instead of letting them decide a
  // 3ns queue comparison. Windows/Linux keep their existing epoch behavior.
  bench.minEpochTime(std::chrono::milliseconds(8));
#endif
  return bench;
}

// Darwin has no public hard P/E-core affinity. Give SSH-launched normal-use
// samples the same explicit QoS contract as the algorithm benchmark, and emit
// the requested state instead of presenting a scheduler accident as M4 data.
void configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char* requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char* name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_time_normal_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int rc = pthread_set_qos_class_self_np(qos, 0);
  std::fprintf(stderr,
               "# HYLE_TIME_NORMAL_BENCH apple_main_qos=%s qos_ok=%d "
               "affinity=unsupported\n",
               name, rc == 0 ? 1 : 0);
  if (rc != 0) std::abort();
#endif
}

struct alignas(64) TimerOwner {
  std::array<std::byte, 192> request_state{};
  TimerEntry timer;
  std::uint64_t request_id = 0;
};

void run_incremental_epoch(TimerQueue& queue,
                           const std::vector<TimerEntry*>& entries,
                           std::uint64_t& base) noexcept {
  std::uint64_t fired = 0;
  std::uint64_t cancelled = 0;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    queue.schedule(entries[i], base + (i + 8) * kTickNs);
    if (i >= 4 && (i - 4) % 10 < 7) {
      cancelled += queue.cancel(entries[i - 4]) ? 1u : 0u;
    }
    TimerQueue::EntryList due;
    queue.advance(base + i * kTickNs, due);
    due.consume_all([&](TimerEntry*) noexcept { ++fired; });
  }
  // The last four timers have not reached their four-tick cancellation point.
  for (std::size_t i = entries.size() - 4; i < entries.size(); ++i) {
    if (i % 10 < 7) cancelled += queue.cancel(entries[i]) ? 1u : 0u;
  }
  TimerQueue::EntryList due;
  queue.advance(base + (entries.size() + 8) * kTickNs, due);
  due.consume_all([&](TimerEntry*) noexcept { ++fired; });
  base += (entries.size() + 8) * kTickNs;
  nb::doNotOptimizeAway(fired);
  nb::doNotOptimizeAway(cancelled);
}

template <class Sleep>
void report_sleep_distribution(const char* name, std::uint64_t request_ns,
                               std::size_t sample_count, Sleep&& sleep) {
  std::vector<std::uint64_t> overshoot;
  overshoot.reserve(sample_count);
  std::size_t early = 0;
  for (std::size_t i = 0; i < sample_count; ++i) {
    const std::uint64_t start = hyle::time::monotonic_ns();
    sleep(request_ns);
    const std::uint64_t elapsed = hyle::time::monotonic_ns() - start;
    if (elapsed < request_ns) ++early;
    overshoot.push_back(elapsed > request_ns ? elapsed - request_ns : 0);
  }
  std::sort(overshoot.begin(), overshoot.end());
  const auto percentile = [&](std::size_t numerator) noexcept {
    const std::size_t rank =
        (overshoot.size() * numerator + 99) / 100;
    return overshoot[rank == 0 ? 0 : rank - 1];
  };
  std::printf(
      "# HYLE_SLEEP source=%s request_ns=%llu n=%zu p50_over_ns=%llu "
      "p95_over_ns=%llu p99_over_ns=%llu max_over_ns=%llu early=%zu\n",
      name, static_cast<unsigned long long>(request_ns), sample_count,
      static_cast<unsigned long long>(percentile(50)),
      static_cast<unsigned long long>(percentile(95)),
      static_cast<unsigned long long>(percentile(99)),
      static_cast<unsigned long long>(overshoot.back()), early);
}

}  // namespace

int main() {
  configure_main_benchmark_thread();

  // Heap-embedded objects are individually allocated and traversed in a
  // coprime permutation, avoiding the vector-only locality of the old bench.
  std::array<TimerEntry, kIncrementalTimers> contiguous_entries;
  std::vector<TimerEntry*> contiguous;
  contiguous.reserve(kIncrementalTimers);
  for (TimerEntry& entry : contiguous_entries) contiguous.push_back(&entry);

  std::vector<std::unique_ptr<TimerOwner>> owners;
  owners.reserve(kIncrementalTimers);
  for (std::size_t i = 0; i < kIncrementalTimers; ++i) {
    auto owner = std::make_unique<TimerOwner>();
    owner->request_id = i;
    owner->timer.set_data(owner.get());
    owners.push_back(std::move(owner));
  }
  std::vector<TimerEntry*> scattered;
  scattered.reserve(kIncrementalTimers);
  for (std::size_t i = 0; i < kIncrementalTimers; ++i) {
    scattered.push_back(&owners[(i * 37) % kIncrementalTimers]->timer);
  }

  {
    nb::Bench bench = make_bench();
    bench.title("incremental event-loop timer lifecycle (64 steps)")
        .relative(true)
        .batch(kIncrementalTimers)
        .minEpochIterations(256);
    TimerQueue contiguous_queue(0);
    TimerQueue scattered_queue(0);
    std::uint64_t contiguous_base = 0;
    std::uint64_t scattered_base = 0;
    bench.run("HYLE contiguous TimerEntry", [&] {
      run_incremental_epoch(contiguous_queue, contiguous, contiguous_base);
    });
    bench.run("HYLE heap-embedded scattered TimerEntry", [&] {
      run_incremental_epoch(scattered_queue, scattered, scattered_base);
    });
  }

  {
    nb::Bench bench = make_bench();
    bench.title("normal schedule entry cost")
        .relative(true)
        .batch(1)
        .minEpochIterations(100'000);
    constexpr std::uint64_t kTimeoutNs = 30'000'000'000u;
    TimerQueue absolute_queue;
    TimerQueue presampled_queue;
    TimerQueue clocked_queue;
    TimerQueue chrono_presampled_queue;
    TimerQueue chrono_clocked_queue;
    TimerEntry absolute_entry;
    TimerEntry presampled_entry;
    TimerEntry clocked_entry;
    TimerEntry chrono_presampled_entry;
    TimerEntry chrono_clocked_entry;
    const std::uint64_t sampled_now = hyle::time::monotonic_ns();
    const auto sampled_time = hyle::time::monotonic_time_point(sampled_now);
    const std::uint64_t deadline =
        hyle::time::saturating_add_ns(sampled_now, kTimeoutNs);
    bench.run("schedule absolute + cancel", [&] {
      absolute_queue.schedule(&absolute_entry, deadline);
      nb::doNotOptimizeAway(absolute_queue.cancel(&absolute_entry));
    });
    bench.run("schedule_after presampled + cancel", [&] {
      presampled_queue.schedule_after(&presampled_entry, kTimeoutNs,
                                      sampled_now);
      nb::doNotOptimizeAway(presampled_queue.cancel(&presampled_entry));
    });
    bench.run("schedule_after reads monotonic + cancel", [&] {
      clocked_queue.schedule_after(&clocked_entry, kTimeoutNs);
      nb::doNotOptimizeAway(clocked_queue.cancel(&clocked_entry));
    });
    bench.run("chrono schedule_after presampled + cancel", [&] {
      chrono_presampled_queue.schedule_after(
          &chrono_presampled_entry, std::chrono::seconds(30), sampled_time);
      nb::doNotOptimizeAway(
          chrono_presampled_queue.cancel(&chrono_presampled_entry));
    });
    bench.run("chrono schedule_after reads clock + cancel", [&] {
      chrono_clocked_queue.schedule_after(&chrono_clocked_entry,
                                          std::chrono::seconds(30));
      nb::doNotOptimizeAway(
          chrono_clocked_queue.cancel(&chrono_clocked_entry));
    });
  }

  {
    nb::Bench bench = make_bench();
    bench.title("timer object creation")
        .relative(true)
        .batch(1)
        .minEpochIterations(20'000);
    bench.run("C++ new/delete TimerEntry", [] {
      auto* entry = new TimerEntry();
      nb::doNotOptimizeAway(entry);
      delete entry;
    });
  }

  // Warm the per-thread sleep primitive before distribution sampling; this
  // section measures scheduler/OS wake quality, not one-time handle creation.
  hyle::time::sleep_for_ns(1);
  constexpr std::array<std::pair<std::uint64_t, std::size_t>, 3> kSleepCases{{
      {100'000u, 128},
      {1'000'000u, 64},
      {10'000'000u, 16},
  }};
  for (const auto& [request_ns, samples] : kSleepCases) {
    report_sleep_distribution(
        "HYLE", request_ns, samples,
        [](std::uint64_t ns) { hyle::time::sleep_for_ns(ns); });
    report_sleep_distribution("std", request_ns, samples,
                              [](std::uint64_t ns) {
                                std::this_thread::sleep_for(
                                    std::chrono::nanoseconds(ns));
                              });
  }
  return 0;
}
