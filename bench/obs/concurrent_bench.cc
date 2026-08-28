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
  const char *value = std::getenv("HYLE_OBS_BENCH_WORKER_CPUS");
  if (value == nullptr || *value == '\0') {
    fail("HYLE_OBS_BENCH_WORKER_CPUS must list eight logical CPUs");
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

} // namespace

int main() {
  const auto cpus = requested_cpus();
  if (!hyle::obs::set_default_sink_enabled(false).ok())
    fail("cannot disable the default sink");
  hyle::obs::set_level(hyle::obs::LogLevel::Info);

  std::printf("# HYLE_OBS_CONCURRENT_BENCH schema=v1 worker_cpus=");
  print_cpu_list(cpus);
  std::printf(" workers=%zu warmup_iterations=%zu measured_iterations=%zu "
              "chunk_iterations=%zu\n",
              kWorkerCount, kWarmupIterations, kMeasuredIterations,
              kChunkIterations);

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

    std::printf(
        "{\"sink_count\":%zu,\"worker_count\":%zu,"
        "\"iterations_per_worker\":%zu,\"writes\":%llu,"
        "\"affinity_ok\":true,\"throughput_ns_per_emit\":%.9f,"
        "\"throughput_ns_per_callback\":%.9f,"
        "\"worker_p50_ns_per_emit\":%.9f,"
        "\"worker_p99_ns_per_emit\":%.9f,"
        "\"worker_max_ns_per_emit\":%.9f,"
        "\"chunk_p50_ns_per_emit\":%.9f,"
        "\"chunk_p99_ns_per_emit\":%.9f,"
        "\"chunk_max_ns_per_emit\":%.9f}\n",
        sink_count, kWorkerCount, kMeasuredIterations,
        static_cast<unsigned long long>(result.writes),
        result.throughput_ns_per_emit, result.throughput_ns_per_callback,
        result.worker_p50_ns_per_emit, result.worker_p99_ns_per_emit,
        result.worker_max_ns_per_emit, result.chunk_p50_ns_per_emit,
        result.chunk_p99_ns_per_emit, result.chunk_max_ns_per_emit);

    for (std::size_t index = 0; index < sink_count; ++index) {
      if (!hyle::obs::remove_sink(counters[index].sink()).ok())
        fail("cannot unregister benchmark sink");
    }
  }
  return 0;
}
