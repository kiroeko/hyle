#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../common/perf_process.h"
#include "hyle/obs/log.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWorkerCount = 8;
constexpr std::size_t kWarmupIterations = 16'384;
constexpr std::size_t kMeasuredIterations = 262'144;
constexpr std::size_t kChunkIterations = 256;
constexpr std::array<std::size_t, 3> kSinkCounts{1, 4, 16};

struct alignas(64) Counter {
  std::atomic<std::uint64_t> writes{0};

  hyle::obs::LogSink sink() noexcept {
    return hyle::obs::LogSink{this, &write, nullptr};
  }

  static void write(void *state, const hyle::obs::LogRecord &) noexcept {
    static_cast<Counter *>(state)->writes.fetch_add(1,
                                                    std::memory_order_relaxed);
  }
};

struct WorkerSample {
  Clock::time_point started{};
  Clock::time_point finished{};
  std::vector<double> chunk_ns_per_emit{};
  unsigned requested_cpu = 0;
  unsigned observed_cpu = 0;
  bool affinity_ok = false;
};

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "hyle_obs_concurrent_bench: %s\n", message);
  std::exit(2);
}

std::array<unsigned, kWorkerCount> requested_cpus() {
  const char *value = std::getenv("HYLE_PERF_WORKER_CPUS");
  if (value == nullptr || *value == '\0') {
    fail("HYLE_PERF_WORKER_CPUS must list eight logical CPUs");
  }

  std::array<unsigned, kWorkerCount> cpus{};
  const char *cursor = value;
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(cursor, &end, 10);
    if (end == cursor || parsed > std::numeric_limits<unsigned>::max()) {
      fail("worker CPU list contains an invalid logical CPU");
    }
    cpus[index] = static_cast<unsigned>(parsed);
    if (index + 1 == cpus.size()) {
      if (*end != '\0')
        fail("worker CPU list must contain exactly eight entries");
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

bool pin_current_thread(unsigned cpu, unsigned &observed_cpu) noexcept {
#if defined(_WIN32)
  if (cpu >= sizeof(DWORD_PTR) * 8)
    return false;
  const DWORD_PTR mask = DWORD_PTR{1} << cpu;
  if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0)
    return false;
  // Crossing the affinity boundary is synchronized by the start barrier.  The
  // observed processor is evidence that Windows scheduled this worker on its
  // requested singleton mask before measurement began.
  observed_cpu = GetCurrentProcessorNumber();
  return observed_cpu == cpu;
#else
  (void)cpu;
  observed_cpu = 0;
  return false;
#endif
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty())
    fail("cannot compute a percentile from no samples");
  std::ranges::sort(values);
  const double position = quantile * static_cast<double>(values.size() - 1);
  const auto index = static_cast<std::size_t>(position);
  return values[index];
}

struct RunResult {
  double throughput_ns_per_emit = 0.0;
  double throughput_ns_per_callback = 0.0;
  double worker_p50_ns_per_emit = 0.0;
  double worker_p99_ns_per_emit = 0.0;
  double worker_max_ns_per_emit = 0.0;
  double chunk_p50_ns_per_emit = 0.0;
  double chunk_p99_ns_per_emit = 0.0;
  double chunk_max_ns_per_emit = 0.0;
  std::uint64_t writes = 0;
  bool affinity_ok = false;
  std::array<unsigned, kWorkerCount> requested_cpus{};
  std::array<unsigned, kWorkerCount> observed_cpus{};
};

RunResult run_workers(const std::array<unsigned, kWorkerCount> &cpus,
                      std::size_t iterations, std::size_t sink_count,
                      bool collect_chunks) {
  if (iterations % kChunkIterations != 0)
    fail("iteration count must be an exact number of measurement chunks");

  std::array<WorkerSample, kWorkerCount> samples{};
  std::barrier start(static_cast<std::ptrdiff_t>(kWorkerCount + 1));
  std::vector<std::thread> threads;
  threads.reserve(kWorkerCount);

  for (std::size_t worker = 0; worker < kWorkerCount; ++worker) {
    samples[worker].requested_cpu = cpus[worker];
    if (collect_chunks) {
      samples[worker].chunk_ns_per_emit.resize(iterations / kChunkIterations);
    }
    threads.emplace_back([&, worker] {
      WorkerSample &sample = samples[worker];
      sample.affinity_ok =
          pin_current_thread(sample.requested_cpu, sample.observed_cpu);
      start.arrive_and_wait();
      sample.started = Clock::now();
      for (std::size_t chunk = 0; chunk < iterations / kChunkIterations;
           ++chunk) {
        const auto chunk_start = Clock::now();
        for (std::size_t i = 0; i < kChunkIterations; ++i) {
          (void)hyle::obs::log_message(hyle::obs::LogLevel::Info, "ready");
        }
        const auto chunk_end = Clock::now();
        if (collect_chunks) {
          sample.chunk_ns_per_emit[chunk] =
              std::chrono::duration<double, std::nano>(chunk_end - chunk_start)
                  .count() /
              static_cast<double>(kChunkIterations);
        }
      }
      sample.finished = Clock::now();
    });
  }

  start.arrive_and_wait();
  for (std::thread &thread : threads)
    thread.join();

  const auto first_start = std::ranges::min_element(
      samples, {}, &WorkerSample::started)->started;
  const auto last_finish = std::ranges::max_element(
      samples, {}, &WorkerSample::finished)->finished;
  const double wall_ns =
      std::chrono::duration<double, std::nano>(last_finish - first_start)
          .count();
  const double total_emits =
      static_cast<double>(kWorkerCount) * static_cast<double>(iterations);

  std::vector<double> worker_ns;
  std::vector<double> chunk_ns;
  worker_ns.reserve(kWorkerCount);
  if (collect_chunks)
    chunk_ns.reserve(kWorkerCount * (iterations / kChunkIterations));
  bool affinity_ok = true;
  for (const WorkerSample &sample : samples) {
    affinity_ok = affinity_ok && sample.affinity_ok &&
                  sample.observed_cpu == sample.requested_cpu;
    worker_ns.push_back(
        std::chrono::duration<double, std::nano>(sample.finished -
                                                sample.started)
            .count() /
        static_cast<double>(iterations));
    chunk_ns.insert(chunk_ns.end(), sample.chunk_ns_per_emit.begin(),
                    sample.chunk_ns_per_emit.end());
  }

  RunResult result{};
  result.throughput_ns_per_emit = wall_ns / total_emits;
  result.throughput_ns_per_callback =
      result.throughput_ns_per_emit / static_cast<double>(sink_count);
  result.worker_p50_ns_per_emit = percentile(worker_ns, 0.50);
  result.worker_p99_ns_per_emit = percentile(worker_ns, 0.99);
  result.worker_max_ns_per_emit = *std::ranges::max_element(worker_ns);
  result.affinity_ok = affinity_ok;
  for (std::size_t index = 0; index < kWorkerCount; ++index) {
    result.requested_cpus[index] = samples[index].requested_cpu;
    result.observed_cpus[index] = samples[index].observed_cpu;
  }
  if (collect_chunks) {
    result.chunk_p50_ns_per_emit = percentile(chunk_ns, 0.50);
    result.chunk_p99_ns_per_emit = percentile(chunk_ns, 0.99);
    result.chunk_max_ns_per_emit = *std::ranges::max_element(chunk_ns);
  }
  return result;
}

void print_cpu_list(const std::array<unsigned, kWorkerCount> &cpus) {
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    if (index != 0)
      std::putchar(',');
    std::printf("%u", cpus[index]);
  }
}

void print_json_cpu_list(const std::array<unsigned, kWorkerCount> &cpus) {
  std::putchar('[');
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    if (index != 0)
      std::putchar(',');
    std::printf("%u", cpus[index]);
  }
  std::putchar(']');
}

void print_json_singleton_masks(
    const std::array<unsigned, kWorkerCount> &cpus) {
  std::putchar('[');
  for (std::size_t index = 0; index < cpus.size(); ++index) {
    if (index != 0)
      std::putchar(',');
    const auto mask = std::uint64_t{1} << cpus[index];
    std::printf("\"0x%llx\"", static_cast<unsigned long long>(mask));
  }
  std::putchar(']');
}

void print_measurement(const char *row_id, double operations_per_second,
                       const RunResult &result, std::size_t sink_count) {
  const std::uint64_t work_items =
      static_cast<std::uint64_t>(kWorkerCount) * kMeasuredIterations;
  const std::uint64_t expected_callbacks =
      work_items * static_cast<std::uint64_t>(sink_count);
  std::printf(
      "{\"record\":\"hyle_perf_structured_measurement\","
      "\"schema\":\"hyle.perf.structured-measurement/v1\","
      "\"row_id\":\"%s\",\"value\":%.9f,\"worker_count\":%zu,"
      "\"work_items_per_worker\":%zu,\"work_item_count\":%llu,"
      "\"fanout\":%zu,\"callback_count\":%llu,"
      "\"expected_callback_count\":%llu,\"affinity_ok\":true,"
      "\"requested_logical_cpus\":",
      row_id, operations_per_second, kWorkerCount, kMeasuredIterations,
      static_cast<unsigned long long>(work_items), sink_count,
      static_cast<unsigned long long>(result.writes),
      static_cast<unsigned long long>(expected_callbacks));
  print_json_cpu_list(result.requested_cpus);
  std::printf(",\"observed_logical_cpus\":");
  print_json_cpu_list(result.observed_cpus);
  std::printf(",\"worker_singleton_masks\":");
  print_json_singleton_masks(result.requested_cpus);
  std::printf(
      ",\"diagnostics\":{\"throughput_ns_per_emit\":%.9f,"
      "\"throughput_ns_per_callback\":%.9f,"
      "\"worker_p50_ns_per_emit\":%.9f,"
      "\"worker_p99_ns_per_emit\":%.9f,"
      "\"worker_max_ns_per_emit\":%.9f,"
      "\"chunk_p50_ns_per_emit\":%.9f,"
      "\"chunk_p99_ns_per_emit\":%.9f,"
      "\"chunk_max_ns_per_emit\":%.9f}}\n",
      result.throughput_ns_per_emit, result.throughput_ns_per_callback,
      result.worker_p50_ns_per_emit, result.worker_p99_ns_per_emit,
      result.worker_max_ns_per_emit, result.chunk_p50_ns_per_emit,
      result.chunk_p99_ns_per_emit, result.chunk_max_ns_per_emit);
}

} // namespace

int main() {
  const auto cpus = requested_cpus();
  if (!hyle::obs::set_default_sink_enabled(false).ok())
    fail("cannot disable the default sink");
  hyle::obs::set_level(hyle::obs::LogLevel::Info);

  for (const std::size_t sink_count : kSinkCounts) {
    std::array<Counter, 16> counters{};
    for (std::size_t index = 0; index < sink_count; ++index) {
      if (!hyle::obs::add_sink(counters[index].sink()).ok())
        fail("cannot register benchmark sink");
    }

    (void)run_workers(cpus, kWarmupIterations, sink_count, false);
    std::uint64_t writes_before = 0;
    for (std::size_t index = 0; index < sink_count; ++index) {
      writes_before += counters[index].writes.load(std::memory_order_relaxed);
    }
    RunResult result =
        run_workers(cpus, kMeasuredIterations, sink_count, true);
    std::uint64_t writes_after = 0;
    for (std::size_t index = 0; index < sink_count; ++index) {
      writes_after += counters[index].writes.load(std::memory_order_relaxed);
    }
    result.writes = writes_after - writes_before;
    const auto expected = static_cast<std::uint64_t>(kWorkerCount) *
                          static_cast<std::uint64_t>(kMeasuredIterations) *
                          static_cast<std::uint64_t>(sink_count);
    if (!result.affinity_ok)
      fail("a worker did not remain on its requested logical CPU");
    if (result.writes != expected)
      fail("sink callback count does not match the workload contract");

    std::array<char, 64> row_id{};
    std::snprintf(row_id.data(), row_id.size(), "fanout-%zu-throughput",
                  sink_count);
    print_measurement(row_id.data(), 1e9 / result.throughput_ns_per_emit,
                      result, sink_count);
    std::snprintf(row_id.data(), row_id.size(), "fanout-%zu-chunk-p99",
                  sink_count);
    print_measurement(row_id.data(), 1e9 / result.chunk_p99_ns_per_emit,
                      result, sink_count);

    for (std::size_t index = 0; index < sink_count; ++index) {
      if (!hyle::obs::remove_sink(counters[index].sink()).ok())
        fail("cannot unregister benchmark sink");
    }
  }
  const char *execution_order = std::getenv("HYLE_PERF_EXECUTION_ORDER");
  const char *topology = std::getenv("HYLE_PERF_SAMPLING_TOPOLOGY");
  const char *parent_mask = std::getenv("HYLE_PERF_PARENT_MASK");
  const char *priority = std::getenv("HYLE_PERF_PRIORITY");
  std::printf("# HYLE_PERF_STRUCTURED_RUN schema=v1 execution_order=%s "
              "affinity_topology=%s worker_count=%zu worker_cpus=",
              execution_order != nullptr ? execution_order : "unmanaged",
              topology != nullptr ? topology : "unmanaged", kWorkerCount);
  print_cpu_list(cpus);
  std::printf(" parent_mask=%s priority=%s affinity_ok=1 "
              "warmup_iterations=%zu measured_iterations=%zu "
              "chunk_iterations=%zu run_nonce=%s\n",
              parent_mask != nullptr ? parent_mask : "unmanaged",
              priority != nullptr ? priority : "unmanaged",
              kWarmupIterations, kMeasuredIterations, kChunkIterations,
              hyle::bench::perf::run_nonce());
  return 0;
}
