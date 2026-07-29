// Normal public clock reads. This belongs to time rather than the historical
// misc integration runner so the module can be built and recertified alone.
#include <nanobench.h>

#include <chrono>
#include <cstdint>

#include "hyle/time/clock.h"

namespace {
namespace nb = ankerl::nanobench;
}

int main() {
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
    bench.run("HYLE profiling_clock_ns (hardware counter)", [&] {
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
