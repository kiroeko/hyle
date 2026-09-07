#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include "../common/perf_process.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/mem/system_inline.h"

#if defined(HYLE_CONTAINER_V6_PEER)
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/concurrent_unordered_set.h>
#else
#include "hyle/container/concurrent_flat_hash_set.h"
#include "hyle/container/unbounded_queue.h"
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;

constexpr std::size_t kWorkerCount = 8;
constexpr std::size_t kProducerCount = 4;
constexpr std::size_t kSetEntries = 32768;
constexpr std::size_t kWarmupIterations = 32768;
constexpr std::size_t kMeasuredIterations = 262144;

struct StableHash {
  std::uint64_t operator()(std::uint64_t value) const noexcept {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }
};

#if defined(HYLE_CONTAINER_V6_PEER)
using ActiveSet = oneapi::tbb::concurrent_unordered_set<
    std::uint64_t, StableHash, std::equal_to<>,
    hyle::mem::SystemStdAllocator<std::uint64_t>>;
using HandoffQueue =
    oneapi::tbb::concurrent_queue<std::uint64_t,
                                  hyle::mem::SystemStdAllocator<std::uint64_t>>;
constexpr const char *kProvider = "oneTBB";
#else
using ActiveSet = hyle::container::ConcurrentFlatHashSet<
    std::uint64_t, StableHash, std::equal_to<>, hyle::mem::SystemAllocator>;
using HandoffQueue =
    hyle::container::UnboundedQueue<std::uint64_t, hyle::mem::SystemAllocator>;
constexpr const char *kProvider = "HYLE";
#endif

struct WorkerSample {
  Clock::time_point started{};
  Clock::time_point finished{};
  std::uint64_t checksum = 0;
  unsigned requested_cpu = 0;
  unsigned observed_cpu = 0;
  bool affinity_ok = false;
};

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "hyle_container_v6_concurrent_bench: %s\n", message);
  std::exit(2);
}

std::array<unsigned, kWorkerCount> requested_cpus() {
  const char *value = std::getenv("HYLE_PERF_WORKER_CPUS");
  if (value == nullptr || *value == '\0')
    fail("HYLE_PERF_WORKER_CPUS must list eight logical CPUs");
  std::array<unsigned, kWorkerCount> cpus{};
  const char *cursor = value;
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(cursor, &end, 10);
    if (end == cursor || parsed > std::numeric_limits<unsigned>::max())
      fail("worker CPU list contains an invalid logical CPU");
    cpus[index] = static_cast<unsigned>(parsed);
    if (index + 1 == cpus.size()) {
      if (*end != '\0')
        fail("worker CPU list has too many entries");
    } else {
      if (*end != ',')
        fail("worker CPU list must be comma-separated");
      cursor = end + 1;
    }
  }
  auto sorted = cpus;
  std::ranges::sort(sorted);
  if (std::ranges::adjacent_find(sorted) != sorted.end())
    fail("worker CPU list contains duplicates");
  return cpus;
}

bool pin_current_thread(unsigned cpu, unsigned &observed) noexcept {
#if defined(_WIN32)
  if (cpu >= sizeof(DWORD_PTR) * 8)
    return false;
  if (SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) == 0)
    return false;
  observed = GetCurrentProcessorNumber();
  return observed == cpu;
#else
  (void)cpu;
  observed = 0;
  return false;
#endif
}

template <class Fn>
std::pair<double, std::array<WorkerSample, kWorkerCount>>
run_workers(const std::array<unsigned, kWorkerCount> &cpus, Fn &&fn) {
  std::array<WorkerSample, kWorkerCount> samples{};
  std::barrier start(static_cast<std::ptrdiff_t>(kWorkerCount + 1));
  std::vector<std::thread> threads;
  threads.reserve(kWorkerCount);
  for (std::size_t worker = 0; worker < kWorkerCount; ++worker) {
    samples[worker].requested_cpu = cpus[worker];
    threads.emplace_back([&, worker] {
      WorkerSample &sample = samples[worker];
      sample.affinity_ok =
          pin_current_thread(sample.requested_cpu, sample.observed_cpu);
      start.arrive_and_wait();
      sample.started = Clock::now();
      sample.checksum = std::invoke(fn, worker);
      sample.finished = Clock::now();
    });
  }
  start.arrive_and_wait();
  for (std::thread &thread : threads)
    thread.join();
  const auto first =
      std::ranges::min_element(samples, {}, &WorkerSample::started)->started;
  const auto last =
      std::ranges::max_element(samples, {}, &WorkerSample::finished)->finished;
  return {std::chrono::duration<double>(last - first).count(), samples};
}

enum class InsertResult : std::uint8_t {
  kInserted,
  kAlreadyPresent,
  kOutOfMemory,
};

InsertResult insert_set(ActiveSet &set, std::uint64_t key) {
#if defined(HYLE_CONTAINER_V6_PEER)
  try {
    return set.insert(key).second ? InsertResult::kInserted
                                  : InsertResult::kAlreadyPresent;
  } catch (const std::bad_alloc &) {
    return InsertResult::kOutOfMemory;
  }
#else
  const auto result = set.try_insert(key);
  if (result == ActiveSet::InsertResult::kInserted)
    return InsertResult::kInserted;
  if (result == ActiveSet::InsertResult::kAlreadyPresent)
    return InsertResult::kAlreadyPresent;
  return InsertResult::kOutOfMemory;
#endif
}

bool reserve_set(ActiveSet &set, std::size_t count) {
#if defined(HYLE_CONTAINER_V6_PEER)
  try {
    set.reserve(count);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
#else
  return set.try_reserve(count);
#endif
}

bool queue_push(HandoffQueue &queue, std::uint64_t value) {
#if defined(HYLE_CONTAINER_V6_PEER)
  try {
    queue.push(value);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
#else
  return queue.try_push(value);
#endif
}

bool queue_pop(HandoffQueue &queue, std::uint64_t &value) {
  return queue.try_pop(value);
}

struct RunResult {
  double seconds = 0.0;
  std::uint64_t checksum = 0;
  std::array<WorkerSample, kWorkerCount> samples{};
};

RunResult run_set(const std::array<unsigned, kWorkerCount> &cpus,
                  ActiveSet &set, std::size_t iterations) {
  auto [seconds, samples] = run_workers(cpus, [&](std::size_t worker) {
    std::uint64_t found = 0;
    for (std::size_t index = 0; index < iterations; ++index) {
      const std::uint64_t base =
          (index * 131 + worker * 977) & (kSetEntries - 1);
      const std::uint64_t key = base * 2 + (index & 1U);
      found += set.contains(key) ? 1U : 0U;
    }
    return found;
  });
  std::uint64_t checksum = 0;
  for (const WorkerSample &sample : samples)
    checksum += sample.checksum;
  return {seconds, checksum, samples};
}

RunResult run_queue(const std::array<unsigned, kWorkerCount> &cpus,
                    HandoffQueue &queue, std::size_t iterations) {
  auto [seconds, samples] = run_workers(cpus, [&](std::size_t worker) {
    std::uint64_t checksum = 0;
    if (worker < kProducerCount) {
      const std::uint64_t begin = worker * iterations + 1;
      for (std::size_t index = 0; index < iterations; ++index) {
        const std::uint64_t value = begin + index;
        while (!queue_push(queue, value))
          std::this_thread::yield();
        checksum += value;
      }
    } else {
      for (std::size_t index = 0; index < iterations; ++index) {
        std::uint64_t value = 0;
        while (!queue_pop(queue, value))
          std::this_thread::yield();
        checksum += value;
      }
    }
    return checksum;
  });
  std::uint64_t checksum = 0;
  for (const WorkerSample &sample : samples)
    checksum += sample.checksum;
  return {seconds, checksum, samples};
}

bool affinity_ok(const RunResult &result) {
  return std::ranges::all_of(result.samples, [](const WorkerSample &sample) {
    return sample.affinity_ok && sample.requested_cpu == sample.observed_cpu;
  });
}

void print_json_cpu_list(const RunResult &result, bool observed) {
  std::putchar('[');
  for (std::size_t index = 0; index < result.samples.size(); ++index) {
    if (index != 0)
      std::putchar(',');
    std::printf("%u", observed ? result.samples[index].observed_cpu
                               : result.samples[index].requested_cpu);
  }
  std::putchar(']');
}

void print_json_masks(const RunResult &result) {
  std::putchar('[');
  for (std::size_t index = 0; index < result.samples.size(); ++index) {
    if (index != 0)
      std::putchar(',');
    const std::uint64_t mask = std::uint64_t{1}
                               << result.samples[index].requested_cpu;
    std::printf("\"0x%llx\"", static_cast<unsigned long long>(mask));
  }
  std::putchar(']');
}

void print_measurement(const char *row_id, double operations_per_second,
                       const RunResult &result, std::uint64_t work_items,
                       std::uint64_t expected_checksum) {
  std::printf("{\"record\":\"hyle_perf_structured_measurement\","
              "\"schema\":\"hyle.perf.structured-measurement/v1\","
              "\"row_id\":\"%s\",\"value\":%.9f,\"worker_count\":%zu,"
              "\"work_items_per_worker\":%zu,\"work_item_count\":%llu,"
              "\"fanout\":1,\"callback_count\":%llu,"
              "\"expected_callback_count\":%llu,\"affinity_ok\":true,"
              "\"requested_logical_cpus\":",
              row_id, operations_per_second, kWorkerCount, kMeasuredIterations,
              static_cast<unsigned long long>(work_items),
              static_cast<unsigned long long>(work_items),
              static_cast<unsigned long long>(work_items));
  print_json_cpu_list(result, false);
  std::printf(",\"observed_logical_cpus\":");
  print_json_cpu_list(result, true);
  std::printf(",\"worker_singleton_masks\":");
  print_json_masks(result);
  std::printf(",\"diagnostics\":{\"seconds\":%.9f,\"checksum\":%llu,"
              "\"expected_checksum\":%llu}}\n",
              result.seconds, static_cast<unsigned long long>(result.checksum),
              static_cast<unsigned long long>(expected_checksum));
}

void print_cpu_list(const std::array<unsigned, kWorkerCount> &cpus) {
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    if (index != 0)
      std::putchar(',');
    std::printf("%u", cpus[index]);
  }
}

} // namespace

int main() {
  const auto cpus = requested_cpus();

  ActiveSet set;
#if !defined(HYLE_CONTAINER_V6_PEER)
  if (!set.valid())
    fail("HYLE concurrent set construction failed");
#endif
  if (!reserve_set(set, kSetEntries))
    fail("set reserve ran out of memory");
  for (std::uint64_t index = 0; index < kSetEntries; ++index) {
    if (insert_set(set, index * 2) != InsertResult::kInserted)
      fail("set setup insert failed");
  }
  (void)run_set(cpus, set, kWarmupIterations);
  const RunResult set_result = run_set(cpus, set, kMeasuredIterations);
  const std::uint64_t set_work = kWorkerCount * kMeasuredIterations;
  const std::uint64_t expected_set = set_work / 2;
  if (!affinity_ok(set_result) || set_result.checksum != expected_set)
    fail("concurrent membership accounting failed");
  print_measurement("read-mostly-membership-8x262144",
                    static_cast<double>(set_work) / set_result.seconds,
                    set_result, set_work, expected_set);

  HandoffQueue queue;
#if !defined(HYLE_CONTAINER_V6_PEER)
  if (!queue.valid())
    fail("HYLE unbounded queue construction failed");
#endif
  const RunResult queue_warmup = run_queue(cpus, queue, kWarmupIterations);
  const std::uint64_t warmup_items = kProducerCount * kWarmupIterations;
  const std::uint64_t expected_warmup = warmup_items * (warmup_items + 1);
  if (!affinity_ok(queue_warmup) || queue_warmup.checksum != expected_warmup)
    fail("queue warmup accounting failed");
  const RunResult queue_result = run_queue(cpus, queue, kMeasuredIterations);
  const std::uint64_t transferred = kProducerCount * kMeasuredIterations;
  const std::uint64_t queue_operations = 2 * transferred;
  const std::uint64_t expected_queue = transferred * (transferred + 1);
  if (!affinity_ok(queue_result) || queue_result.checksum != expected_queue)
    fail("queue transfer accounting failed");
  print_measurement("unbounded-mpmc-handoff-4p4c-262144",
                    static_cast<double>(queue_operations) /
                        queue_result.seconds,
                    queue_result, queue_operations, expected_queue);

  const char *execution_order = std::getenv("HYLE_PERF_EXECUTION_ORDER");
  const char *topology = std::getenv("HYLE_PERF_SAMPLING_TOPOLOGY");
  const char *parent_mask = std::getenv("HYLE_PERF_PARENT_MASK");
  const char *priority = std::getenv("HYLE_PERF_PRIORITY");
  std::printf(
      "# HYLE_PERF_STRUCTURED_RUN schema=v1 provider=%s execution_order=%s "
      "affinity_topology=%s worker_count=%zu worker_cpus=",
      kProvider, execution_order != nullptr ? execution_order : "unmanaged",
      topology != nullptr ? topology : "unmanaged", kWorkerCount);
  print_cpu_list(cpus);
  std::printf(" parent_mask=%s priority=%s affinity_ok=1 hash=stable "
              "allocator=hyle-system allocation_failure=recoverable "
              "warmup_iterations=%zu "
              "measured_iterations=%zu producer_count=%zu run_nonce=%s\n",
              parent_mask != nullptr ? parent_mask : "unmanaged",
              priority != nullptr ? priority : "unmanaged", kWarmupIterations,
              kMeasuredIterations, kProducerCount,
              hyle::bench::perf::run_nonce());
  return 0;
}
