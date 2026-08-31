#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <thread>

#include "../common/perf_process.h"

#if !defined(_WIN32)
#error "The sched P1 qualification replay is currently qualified on W1-WIN only"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

namespace hyle::bench::sched_p1 {

using Clock = std::chrono::steady_clock;

inline constexpr std::size_t kWorkerCount = 4;
inline constexpr std::size_t kRequestWarmup = 2'048;
inline constexpr std::size_t kRequestCount = 16'384;
inline constexpr std::size_t kRequestTileElements = 64;
inline constexpr std::size_t kStructuredWarmupFanout = 64;
inline constexpr std::size_t kStructuredRounds = 16;
inline constexpr std::size_t kStructuredFanout = 128;
// Graph's current automatic parallel grain has a reviewed floor of 1024
// elements. Keep the scheduler replay at that consumer boundary instead of
// inventing a smaller benchmark-only tile.
inline constexpr std::size_t kStructuredTileElements = 1'024;
inline constexpr std::size_t kWakeWarmupEpisodes = 256;
inline constexpr std::size_t kWakeMeasuredEpisodes = 4'096;
inline constexpr auto kWaitTimeout = std::chrono::seconds(30);

[[noreturn]] inline void fail(const char* message) {
  std::fprintf(stderr, "hyle_sched_p1_bench: %s\n", message);
  std::exit(2);
}

inline std::array<unsigned, kWorkerCount> requested_cpus() {
  const char* value = std::getenv("HYLE_PERF_WORKER_CPUS");
  if (value == nullptr || *value == '\0')
    fail("HYLE_PERF_WORKER_CPUS must list four logical CPUs");

  std::array<unsigned, kWorkerCount> cpus{};
  const char* cursor = value;
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(cursor, &end, 10);
    if (end == cursor || parsed >= sizeof(DWORD_PTR) * 8 ||
        parsed > (std::numeric_limits<unsigned>::max)()) {
      fail("worker CPU list contains an unsupported logical CPU");
    }
    cpus[index] = static_cast<unsigned>(parsed);
    if (index + 1 == cpus.size()) {
      if (*end != '\0')
        fail("worker CPU list must contain exactly four entries");
    } else {
      if (*end != ',')
        fail("worker CPU list must be comma-separated");
      cursor = end + 1;
    }
  }
  auto sorted = cpus;
  std::ranges::sort(sorted);
  if (std::ranges::adjacent_find(sorted) != sorted.end())
    fail("worker CPU list must not contain duplicates");
  return cpus;
}

inline unsigned current_cpu() noexcept {
  return static_cast<unsigned>(::GetCurrentProcessorNumber());
}

inline bool pin_current_thread(unsigned cpu) noexcept {
  const DWORD_PTR mask = DWORD_PTR{1} << cpu;
  return ::SetThreadAffinityMask(::GetCurrentThread(), mask) != 0 &&
         current_cpu() == cpu;
}

template <class Predicate>
inline void wait_until(Predicate&& predicate, const char* timeout_message) {
  const auto deadline = Clock::now() + kWaitTimeout;
  while (!predicate()) {
    if (Clock::now() >= deadline)
      fail(timeout_message);
    std::this_thread::yield();
  }
}

inline void wait_at_least(const std::atomic<std::size_t>& value,
                          std::size_t target) noexcept {
  std::size_t observed = value.load(std::memory_order_acquire);
  while (observed < target) {
    value.wait(observed, std::memory_order_acquire);
    observed = value.load(std::memory_order_acquire);
  }
}

struct AffinityState {
  explicit AffinityState(
      const std::array<unsigned, kWorkerCount>& requested_cpus) noexcept
      : requested(requested_cpus) {
    for (auto& cpu : observed)
      cpu.store((std::numeric_limits<unsigned>::max)(),
                std::memory_order_relaxed);
  }

  void record(std::size_t worker, unsigned cpu) noexcept {
    if (worker >= kWorkerCount)
      fail("scheduler reported an out-of-range worker index");
    const unsigned sentinel = (std::numeric_limits<unsigned>::max)();
    unsigned expected = sentinel;
    if (!observed[worker].compare_exchange_strong(
            expected, cpu, std::memory_order_release,
            std::memory_order_relaxed)) {
      fail("scheduler initialized one worker more than once");
    }
    ready.fetch_add(1, std::memory_order_release);
  }

  void await_ready() const {
    wait_until(
        [&] { return ready.load(std::memory_order_acquire) == kWorkerCount; },
        "scheduler workers did not initialize");
  }

  std::array<unsigned, kWorkerCount> observed_snapshot() const noexcept {
    std::array<unsigned, kWorkerCount> result{};
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = observed[i].load(std::memory_order_acquire);
    return result;
  }

  bool exact() const noexcept {
    return observed_snapshot() == requested;
  }

  std::array<unsigned, kWorkerCount> requested{};
  std::array<std::atomic<unsigned>, kWorkerCount> observed{};
  std::atomic<std::size_t> ready{0};
};

struct WorkerHits {
  void reset() noexcept {
    for (auto& value : values)
      value.store(0, std::memory_order_relaxed);
  }

  void hit(int worker) noexcept {
    if (worker < 0 || static_cast<std::size_t>(worker) >= kWorkerCount)
      fail("work ran outside a qualified scheduler worker");
    values[static_cast<std::size_t>(worker)].fetch_add(
        1, std::memory_order_relaxed);
  }

  std::size_t active() const noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        values, [](const auto& value) {
          return value.load(std::memory_order_relaxed) != 0;
        }));
  }

  std::uint64_t total() const noexcept {
    std::uint64_t result = 0;
    for (const auto& value : values)
      result += value.load(std::memory_order_relaxed);
    return result;
  }

  std::array<std::atomic<std::uint64_t>, kWorkerCount> values{};
};

inline std::uint64_t tile_digest(std::span<const std::uint64_t> input,
                                 std::uint64_t seed) noexcept {
  std::uint64_t digest = seed ^ 0x9e3779b97f4a7c15ULL;
  for (const std::uint64_t value : input) {
    digest ^= value + 0x9e3779b97f4a7c15ULL + (digest << 6) + (digest >> 2);
    digest = std::rotl(digest, 17) * 0xbf58476d1ce4e5b9ULL;
  }
  return digest;
}

inline void print_cpu_list(
    const std::array<unsigned, kWorkerCount>& cpus) noexcept {
  for (std::size_t i = 0; i < cpus.size(); ++i) {
    if (i != 0)
      std::putchar(',');
    std::printf("%u", cpus[i]);
  }
}

inline void print_json_cpu_list(
    const std::array<unsigned, kWorkerCount>& cpus) noexcept {
  std::putchar('[');
  for (std::size_t i = 0; i < cpus.size(); ++i) {
    if (i != 0)
      std::putchar(',');
    std::printf("%u", cpus[i]);
  }
  std::putchar(']');
}

inline void print_json_singleton_masks(
    const std::array<unsigned, kWorkerCount>& cpus) noexcept {
  std::putchar('[');
  for (std::size_t i = 0; i < cpus.size(); ++i) {
    if (i != 0)
      std::putchar(',');
    const auto mask = std::uint64_t{1} << cpus[i];
    std::printf("\"0x%llx\"", static_cast<unsigned long long>(mask));
  }
  std::putchar(']');
}

inline double elapsed_ns(Clock::time_point start,
                         Clock::time_point finish) noexcept {
  return std::chrono::duration<double, std::nano>(finish - start).count();
}

inline void print_measurement(const char* row_id, double operations_per_second,
                              std::uint64_t work_items,
                              std::uint64_t fanout,
                              std::uint64_t callback_count,
                              std::uint64_t checksum, double wall_ns,
                              const WorkerHits& hits,
                              const AffinityState& affinity) {
  if (!affinity.exact())
    fail("a scheduler worker did not remain on its requested logical CPU");
  // Placement is proved independently for every configured worker. Dynamic
  // schedulers are allowed to finish a short batch before all workers steal a
  // task; preserving the observed active-worker count is evidence, while
  // manufacturing a start barrier here would change the consumer workload.
  if (hits.active() == 0)
    fail("a qualified workload executed no scheduler work");
  if (work_items == 0 || work_items % kWorkerCount != 0 || fanout == 0 ||
      callback_count != work_items * fanout || operations_per_second <= 0.0 ||
      wall_ns <= 0.0) {
    fail("structured workload accounting is inconsistent");
  }

  const auto observed = affinity.observed_snapshot();
  std::printf(
      "{\"record\":\"hyle_perf_structured_measurement\","
      "\"schema\":\"hyle.perf.structured-measurement/v1\","
      "\"row_id\":\"%s\",\"value\":%.9f,\"worker_count\":%zu,"
      "\"work_items_per_worker\":%llu,\"work_item_count\":%llu,"
      "\"fanout\":%llu,\"callback_count\":%llu,"
      "\"expected_callback_count\":%llu,\"affinity_ok\":true,"
      "\"requested_logical_cpus\":",
      row_id, operations_per_second, kWorkerCount,
      static_cast<unsigned long long>(work_items / kWorkerCount),
      static_cast<unsigned long long>(work_items),
      static_cast<unsigned long long>(fanout),
      static_cast<unsigned long long>(callback_count),
      static_cast<unsigned long long>(work_items * fanout));
  print_json_cpu_list(affinity.requested);
  std::printf(",\"observed_logical_cpus\":");
  print_json_cpu_list(observed);
  std::printf(",\"worker_singleton_masks\":");
  print_json_singleton_masks(affinity.requested);
  std::printf(
      ",\"diagnostics\":{\"elapsed_ns\":%.3f,\"checksum\":%llu,"
      "\"active_workers\":%zu,\"worker_callbacks\":%llu}}\n",
      wall_ns, static_cast<unsigned long long>(checksum), hits.active(),
      static_cast<unsigned long long>(hits.total()));
}

inline void print_run_telemetry(
    const std::array<unsigned, kWorkerCount>& cpus) noexcept {
  const char* execution_order = std::getenv("HYLE_PERF_EXECUTION_ORDER");
  const char* topology = std::getenv("HYLE_PERF_SAMPLING_TOPOLOGY");
  const char* parent_mask = std::getenv("HYLE_PERF_PARENT_MASK");
  const char* priority = std::getenv("HYLE_PERF_PRIORITY");
  std::printf("# HYLE_PERF_STRUCTURED_RUN schema=v1 execution_order=%s "
              "affinity_topology=%s worker_count=%zu worker_cpus=",
              execution_order != nullptr ? execution_order : "unmanaged",
              topology != nullptr ? topology : "unmanaged", kWorkerCount);
  print_cpu_list(cpus);
  std::printf(" parent_mask=%s priority=%s affinity_ok=1 run_nonce=%s\n",
              parent_mask != nullptr ? parent_mask : "unmanaged",
              priority != nullptr ? priority : "unmanaged",
              hyle::bench::perf::run_nonce());
}

}  // namespace hyle::bench::sched_p1
