// Normal public-entry benchmark for time. The algorithm-only N=4096/rival
// certificate stays in timer_queue_bench.cc; this executable covers the shapes
// users actually write: embedded/scattered nodes, incremental event-loop work,
// schedule_after including its clock read, the opaque C ABI, and sleep latency.
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

#include "hyle/c/error_helpers.h"
#include "hyle/c/time.h"
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

void require_ok(hyle_status status) noexcept {
  if (!hyle_status_is_ok(status)) std::abort();
  hyle_status_free(&status);
}

void count_due(hyle_time_timer*, void* context) {
  ++*static_cast<std::uint64_t*>(context);
}

struct MutationVisitContext {
  hyle_time_queue* queue = nullptr;
  std::array<hyle_time_timer*, 4>* timers = nullptr;
  hyle_time_timer* rearmed = nullptr;
  std::uint64_t next_deadline = 0;
  std::size_t visited = 0;
  bool mutated = false;
};

void mutate_due_peers(hyle_time_timer* current, void* context) {
  auto& state = *static_cast<MutationVisitContext*>(context);
  ++state.visited;
  if (state.mutated) return;

  hyle_time_timer* cancel_target = nullptr;
  hyle_time_timer* rearm_target = nullptr;
  for (hyle_time_timer* timer : *state.timers) {
    if (timer == current) continue;
    if (cancel_target == nullptr) {
      cancel_target = timer;
    } else {
      rearm_target = timer;
      break;
    }
  }
  if (cancel_target == nullptr || rearm_target == nullptr) std::abort();

  int cancelled = 0;
  require_ok(hyle_time_queue_cancel(&cancelled, state.queue, cancel_target));
  if (cancelled != 1) std::abort();
  require_ok(hyle_time_queue_schedule(state.queue, rearm_target,
                                      state.next_deadline));
  state.rearmed = rearm_target;
  state.mutated = true;
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
    bench.title("C ABI normal timer operations")
        .relative(true)
        .batch(1)
        .minEpochIterations(50'000);
    TimerQueue cpp_queue(0);
    TimerEntry cpp_entry;
    hyle_time_queue* c_queue = nullptr;
    hyle_time_timer* c_timer = nullptr;
    require_ok(hyle_time_queue_create_at(&c_queue, 0));
    require_ok(hyle_time_timer_create(&c_timer, nullptr));
    bench.run("C++ schedule+cancel", [&] {
      cpp_queue.schedule(&cpp_entry, 1);
      nb::doNotOptimizeAway(cpp_queue.cancel(&cpp_entry));
    });
    bench.run("C ABI schedule+cancel+Status", [&] {
      require_ok(hyle_time_queue_schedule(c_queue, c_timer, 1));
      int cancelled = 0;
      require_ok(hyle_time_queue_cancel(&cancelled, c_queue, c_timer));
      nb::doNotOptimizeAway(cancelled);
    });
    bench.run("C ABI schedule_after clock+cancel+Status", [&] {
      require_ok(hyle_time_queue_schedule_after(c_queue, c_timer,
                                                30'000'000'000u));
      int cancelled = 0;
      require_ok(hyle_time_queue_cancel(&cancelled, c_queue, c_timer));
      nb::doNotOptimizeAway(cancelled);
    });
    require_ok(hyle_time_timer_destroy(c_timer));
    require_ok(hyle_time_queue_destroy(c_queue));
  }

  {
    nb::Bench bench = make_bench();
    bench.title("C ABI schedule+advance+visitor")
        .relative(false)
        .batch(1)
        .minEpochIterations(50'000);
    hyle_time_queue* queue = nullptr;
    hyle_time_timer* timer = nullptr;
    require_ok(hyle_time_queue_create_at(&queue, 0));
    require_ok(hyle_time_timer_create(&timer, nullptr));
    std::uint64_t now = 0;
    std::uint64_t visited = 0;
    bench.run("C ABI fire one timer", [&] {
      require_ok(hyle_time_queue_schedule(queue, timer, now + 1));
      std::size_t fired = 0;
      require_ok(hyle_time_queue_advance(&fired, queue, now + 1, count_due,
                                         &visited));
      ++now;
      nb::doNotOptimizeAway(fired);
    });
    nb::doNotOptimizeAway(visited);
    require_ok(hyle_time_timer_destroy(timer));
    require_ok(hyle_time_queue_destroy(queue));
  }

  {
    // Exercise the real mutation-safe delivery contract: while four timers are
    // pending in one synchronous batch, the first callback cancels one peer and
    // rearms another.  A benchmark that only visits one timer cannot detect the
    // bookkeeping cost or a regression that delivers a suppressed peer.
    nb::Bench bench = make_bench();
    bench.title("C ABI mutation-safe timer delivery")
        .relative(false)
        .batch(1)
        .minEpochIterations(20'000);
    hyle_time_queue* plain_queue = nullptr;
    hyle_time_queue* mutation_queue = nullptr;
    std::array<hyle_time_timer*, 4> plain_timers{};
    std::array<hyle_time_timer*, 4> mutation_timers{};
    require_ok(hyle_time_queue_create_at(&plain_queue, 0));
    require_ok(hyle_time_queue_create_at(&mutation_queue, 0));
    for (std::size_t i = 0; i < plain_timers.size(); ++i) {
      require_ok(hyle_time_timer_create(&plain_timers[i], nullptr));
      require_ok(hyle_time_timer_create(&mutation_timers[i], nullptr));
    }
    std::uint64_t plain_now = 0;
    std::uint64_t mutation_now = 0;
    bench.run("C ABI fire four timers", [&] {
      for (hyle_time_timer* timer : plain_timers) {
        require_ok(hyle_time_queue_schedule(plain_queue, timer, plain_now + 1));
      }
      std::size_t fired = 0;
      std::uint64_t visited = 0;
      require_ok(hyle_time_queue_advance(&fired, plain_queue, plain_now + 1,
                                         count_due, &visited));
      if (fired != plain_timers.size() || visited != plain_timers.size()) {
        std::abort();
      }
      ++plain_now;
      nb::doNotOptimizeAway(fired);
    });
    bench.run("C ABI cancel+reschedule pending peers", [&] {
      for (hyle_time_timer* timer : mutation_timers) {
        require_ok(
            hyle_time_queue_schedule(mutation_queue, timer, mutation_now + 1));
      }
      MutationVisitContext context{mutation_queue, &mutation_timers, nullptr,
                                   mutation_now + 2, 0, false};
      std::size_t fired = 0;
      require_ok(hyle_time_queue_advance(&fired, mutation_queue,
                                         mutation_now + 1, mutate_due_peers,
                                         &context));
      if (fired != 2 || context.visited != 2 || context.rearmed == nullptr) {
        std::abort();
      }
      int cancelled = 0;
      require_ok(hyle_time_queue_cancel(&cancelled, mutation_queue,
                                        context.rearmed));
      if (cancelled != 1) std::abort();
      ++mutation_now;
      nb::doNotOptimizeAway(fired);
    });
    for (std::size_t i = 0; i < plain_timers.size(); ++i) {
      require_ok(hyle_time_timer_destroy(plain_timers[i]));
      require_ok(hyle_time_timer_destroy(mutation_timers[i]));
    }
    require_ok(hyle_time_queue_destroy(plain_queue));
    require_ok(hyle_time_queue_destroy(mutation_queue));
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
    bench.run("C opaque timer create/destroy+Status", [] {
      hyle_time_timer* timer = nullptr;
      require_ok(hyle_time_timer_create(&timer, nullptr));
      nb::doNotOptimizeAway(timer);
      require_ok(hyle_time_timer_destroy(timer));
    });
    const std::size_t storage_size = hyle_time_timer_storage_size();
    const std::size_t storage_alignment =
        hyle_time_timer_storage_alignment();
    std::vector<std::byte> storage(storage_size + storage_alignment - 1);
    void* storage_ptr = storage.data();
    std::size_t storage_space = storage.size();
    if (std::align(storage_alignment, storage_size, storage_ptr,
                   storage_space) == nullptr) {
      std::abort();
    }
    bench.run("C caller-storage timer init/deinit+Status", [&] {
      hyle_time_timer* timer = nullptr;
      require_ok(hyle_time_timer_init(&timer, storage_ptr, storage_size,
                                      nullptr));
      nb::doNotOptimizeAway(timer);
      require_ok(hyle_time_timer_deinit(timer));
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
