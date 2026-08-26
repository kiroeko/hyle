// Normal public clock reads. This belongs to time rather than the historical
// misc integration runner so the module can be built and recertified alone.
#include <nanobench.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "hyle/time/clock.h"

namespace {
namespace nb = ankerl::nanobench;

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
                 "hyle_time_clock_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int result = pthread_set_qos_class_self_np(qos, 0);
  std::fprintf(stderr,
               "# HYLE_TIME_CLOCK_BENCH apple_main_qos=%s qos_ok=%d "
               "affinity=unsupported\n",
               name, result == 0 ? 1 : 0);
  if (result != 0) std::abort();
#endif
}
}

int main() {
  configure_main_benchmark_thread();
  const bool profiling_counter_available =
      hyle::time::profiling_clock_available();
  std::fprintf(stderr,
               "# HYLE_TIME_CLOCK_BENCH profiling_clock_available=%d\n",
               profiling_counter_available ? 1 : 0);
  {
    nb::Bench bench;
    bench.title("profiling span boundaries")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
    bench.run("std::chrono steady pair", [&] {
      const auto start = std::chrono::steady_clock::now();
      const auto stop = std::chrono::steady_clock::now();
      nb::doNotOptimizeAway(stop - start);
    });
    bench.run("HYLE unordered sampling pair", [&] {
      const std::uint64_t start = hyle::time::profiling_clock_ns();
      const std::uint64_t stop = hyle::time::profiling_clock_ns();
      nb::doNotOptimizeAway(stop - start);
    });
    bench.run("HYLE ordered measurement pair", [&] {
      const std::uint64_t start =
          hyle::time::profiling_measurement_start_ns();
      const std::uint64_t stop =
          hyle::time::profiling_measurement_stop_ns();
      nb::doNotOptimizeAway(stop - start);
    });
  }
  {
    nb::Bench bench;
    bench.title("monotonic clock read")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
    bench.run("std::chrono::steady_clock::now", [&] {
      const auto value =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      const std::uint64_t result =
          value > 0 ? static_cast<std::uint64_t>(value) : 0;
      nb::doNotOptimizeAway(result);
    });
    bench.run("HYLE monotonic_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::monotonic_ns());
    });
    bench.run(profiling_counter_available
                  ? "HYLE profiling_clock_ns (hardware counter)"
                  : "HYLE profiling_clock_ns (monotonic fallback)",
              [&] {
      nb::doNotOptimizeAway(hyle::time::profiling_clock_ns());
    });
  }
  {
    nb::Bench bench;
    bench.title("realtime clock read")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
    bench.run("std::chrono::system_clock::now", [&] {
      const auto value =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      const std::uint64_t result =
          value > 0 ? static_cast<std::uint64_t>(value) : 0;
      nb::doNotOptimizeAway(result);
    });
    bench.run("HYLE realtime_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::realtime_ns());
    });
    bench.run("HYLE coarse_realtime_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::coarse_realtime_ns());
    });
  }
  return 0;
}
