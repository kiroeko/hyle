// Trace-on adapter evidence for the Tracy 0.14.1 upgrade.
//
// The direct and HYLE rows deliberately use identical payload types and Tracy
// event contracts.  HYLE_TRACE_* remains the consumer-facing shape; direct
// Tracy macros appear only here to prove that the seam adds no hidden search,
// allocation, locking, or metadata work above the selected backend.

#include <nanobench.h>

#include <cstdint>
#include <cstdio>

#include "hyle/obs/trace.h"

#if !defined(HYLE_TRACE_TRACY) || !defined(TRACY_ENABLE)
#  error "hyle_obs_tracy_bench requires the trace-on Tracy backend"
#endif

namespace {
namespace nb = ankerl::nanobench;

void direct_zone(std::uint64_t value) {
  ZoneScopedN("obs.tracy.direct.zone");
  nb::doNotOptimizeAway(value);
}

void hyle_zone(std::uint64_t value) {
  HYLE_TRACE_ZONE_NAMED("obs.hyle.zone");
  nb::doNotOptimizeAway(value);
}

}  // namespace

int main() {
  std::fprintf(stderr,
               "# HYLE_OBS_TRACY_BENCH peer=tracy-0.14.1 "
               "contract=trace-on-native-macro\n");

  {
    std::int64_t direct_value = 0;
    std::int64_t hyle_value = 0;
    nb::Bench bench;
    bench.title("trace-on counter event: HYLE seam vs direct Tracy")
        .relative(true)
        .minEpochIterations(200'000);
    bench.run("direct TracyPlot(int64_t)", [&] {
      ++direct_value;
      TracyPlot("obs.tracy.direct.counter", direct_value);
      nb::doNotOptimizeAway(direct_value);
    });
    bench.run("HYLE_TRACE_COUNTER(int64_t)", [&] {
      ++hyle_value;
      HYLE_TRACE_COUNTER("obs.hyle.counter", hyle_value);
      nb::doNotOptimizeAway(hyle_value);
    });
  }

  {
    std::uint64_t direct_value = 0;
    std::uint64_t hyle_value = 0;
    nb::Bench bench;
    bench.title("trace-on named zone: HYLE seam vs direct Tracy")
        .relative(true)
        .minEpochIterations(200'000);
    bench.run("direct ZoneScopedN", [&] { direct_zone(++direct_value); });
    bench.run("HYLE_TRACE_ZONE_NAMED", [&] { hyle_zone(++hyle_value); });
  }
  return 0;
}
