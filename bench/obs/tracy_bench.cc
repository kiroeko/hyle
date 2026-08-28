// Same-contract trace adapter evidence against the selected Tracy backend.
// Direct Tracy use is confined to this validation target.

#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/obs/trace.h"

#include "../common/perf_process.h"

#if !defined(HYLE_TRACE_TRACY) || !defined(TRACY_ENABLE)
#error "hyle_obs_tracy_bench requires the trace-on Tracy backend"
#endif

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kEventBatch = 64;
constexpr std::string_view kZoneText = "operator=qwen attention shape=1x32x128";

bool rival_first_requested() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, "HYLE_BENCH_RIVAL_FIRST") == 0 &&
      value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
#endif
}

nb::Bench make_bench(const char *title) {
  nb::Bench bench;
  bench.title(title)
      .batch(kEventBatch)
      .minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool rival_first, const char *candidate_name,
              Candidate &&candidate, const char *peer_name, Peer &&peer) {
  if (rival_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

HYLE_NOINLINE void direct_zone_event(std::uint64_t value) {
  ZoneScopedN("hyle.io.reap");
  nb::doNotOptimizeAway(value);
}

HYLE_NOINLINE void hyle_zone_event(std::uint64_t value) {
  HYLE_TRACE_ZONE_NAMED("hyle.io.reap");
  nb::doNotOptimizeAway(value);
}

HYLE_NOINLINE void direct_annotated_zone_event(std::uint64_t value) {
  ZoneScopedN("hyle.inference.operator");
  ZoneText(kZoneText.data(), kZoneText.size());
  ZoneValue(value);
  nb::doNotOptimizeAway(value);
}

HYLE_NOINLINE void hyle_annotated_zone_event(std::uint64_t value) {
  HYLE_TRACE_ZONE_NAMED("hyle.inference.operator");
  HYLE_TRACE_ZONE_TEXT(kZoneText);
  HYLE_TRACE_ZONE_VALUE(value);
  nb::doNotOptimizeAway(value);
}

void bench_counter(bool rival_first) {
  std::int64_t hyle_value = 0;
  std::int64_t direct_value = 0;
  nb::Bench bench = make_bench("obs trace counter event batch N=64");
  run_pair(
      bench, rival_first, "HYLE_TRACE_COUNTER batch N=64",
      [&] {
        for (std::size_t i = 0; i < kEventBatch; ++i) {
          HYLE_TRACE_COUNTER("hyle.io.completions", ++hyle_value);
        }
        nb::doNotOptimizeAway(hyle_value);
      },
      "direct TracyPlot batch N=64",
      [&] {
        for (std::size_t i = 0; i < kEventBatch; ++i) {
          TracyPlot("hyle.io.completions", ++direct_value);
        }
        nb::doNotOptimizeAway(direct_value);
      });
}

void bench_zone(bool rival_first) {
  std::uint64_t hyle_value = 0;
  std::uint64_t direct_value = 0;
  nb::Bench bench = make_bench("obs trace named-zone event batch N=64");
  run_pair(
      bench, rival_first, "HYLE_TRACE_ZONE_NAMED batch N=64",
      [&] {
        for (std::size_t i = 0; i < kEventBatch; ++i) {
          hyle_zone_event(++hyle_value);
        }
      },
      "direct ZoneScopedN batch N=64",
      [&] {
        for (std::size_t i = 0; i < kEventBatch; ++i) {
          direct_zone_event(++direct_value);
        }
      });
}

void bench_annotation(bool rival_first) {
  std::uint64_t hyle_value = 0;
  std::uint64_t direct_value = 0;
  nb::Bench bench = make_bench("obs trace annotated-zone event batch N=64");
  run_pair(
      bench, rival_first, "HYLE current-zone text+value batch N=64",
      [&] {
        for (std::size_t i = 0; i < kEventBatch; ++i) {
          hyle_annotated_zone_event(++hyle_value);
        }
      },
      "direct Tracy ZoneText+Value batch N=64",
      [&] {
        for (std::size_t i = 0; i < kEventBatch; ++i) {
          direct_annotated_zone_event(++direct_value);
        }
      });
}

} // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool rival_first = rival_first_requested();
  std::fprintf(stderr,
               "# HYLE_OBS_TRACE_RUN execution_order=%s apple_qos=%s "
               "qos_ok=%s affinity=external run_nonce=%s\n",
               rival_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
               hyle::bench::perf::run_nonce());
  bench_counter(rival_first);
  bench_zone(rival_first);
  bench_annotation(rival_first);
  return 0;
}
