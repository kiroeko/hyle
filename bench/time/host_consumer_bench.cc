// Qualified host-time replay: one runtime phase batch performs the clock work
// used by schedulers and inference instrumentation, rather than treating one
// sub-nanosecond wrapper call as a product workload.
#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>

#include "hyle/time/clock.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

using Duration = hyle::time::Duration;
using TimePoint = hyle::time::MonotonicTimePoint;

constexpr std::size_t kBatch = 64;
constexpr std::uint64_t kTimeoutNs = 30'000'000'000u;

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

std::uint64_t peer_to_ns(TimePoint point) noexcept {
  const auto value = point.time_since_epoch().count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

std::uint64_t peer_saturating_add(std::uint64_t base,
                                  std::uint64_t delta) noexcept {
  return delta > UINT64_MAX - base ? UINT64_MAX : base + delta;
}

Duration peer_nonnegative_difference(TimePoint later,
                                     TimePoint earlier) noexcept {
  using Rep = Duration::rep;
  const Rep later_count = later.time_since_epoch().count();
  const Rep earlier_count = earlier.time_since_epoch().count();
  if (later_count <= earlier_count) return Duration::zero();
  constexpr Rep kMax = (std::numeric_limits<Rep>::max)();
  if (earlier_count < 0 && later_count > kMax + earlier_count) {
    return (Duration::max)();
  }
  return Duration(later_count - earlier_count);
}

std::uint64_t run_hyle_phase_batch() noexcept {
  std::uint64_t checksum = 0;
  for (std::size_t index = 0; index < kBatch; ++index) {
    const hyle::time::Instant begin = hyle::time::Instant::now();
    nb::doNotOptimizeAway(begin);
    const std::uint64_t deadline = hyle::time::saturating_add_ns(
        hyle::time::to_monotonic_ns(begin.time_point()), kTimeoutNs + index);
    const hyle::time::Instant end = hyle::time::Instant::now();
    const std::uint64_t elapsed = begin.until_ns(end);
    const std::uint64_t end_ns = hyle::time::to_monotonic_ns(end.time_point());
    const std::uint64_t remaining = deadline > end_ns ? deadline - end_ns : 0;
    checksum += elapsed ^ remaining ^ deadline;
  }
  nb::doNotOptimizeAway(checksum);
  return checksum;
}

std::uint64_t run_chrono_phase_batch() noexcept {
  std::uint64_t checksum = 0;
  for (std::size_t index = 0; index < kBatch; ++index) {
    const TimePoint begin = std::chrono::time_point_cast<Duration>(
        std::chrono::steady_clock::now());
    nb::doNotOptimizeAway(begin);
    const std::uint64_t deadline =
        peer_saturating_add(peer_to_ns(begin), kTimeoutNs + index);
    const TimePoint end = std::chrono::time_point_cast<Duration>(
        std::chrono::steady_clock::now());
    const std::uint64_t elapsed = static_cast<std::uint64_t>(
        peer_nonnegative_difference(end, begin).count());
    const std::uint64_t end_ns = peer_to_ns(end);
    const std::uint64_t remaining = deadline > end_ns ? deadline - end_ns : 0;
    checksum += elapsed ^ remaining ^ deadline;
  }
  nb::doNotOptimizeAway(checksum);
  return checksum;
}

}  // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_TIME_HOST_CONSUMER_RUN execution_order=%s apple_qos=%s "
      "qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  (void)run_hyle_phase_batch();
  (void)run_chrono_phase_batch();

  nb::Bench bench;
  bench.title("host phase timing and deadline aggregate N=64")
      .relative(true)
      .batch(kBatch)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(bench, peer_first, "HYLE host phase batch N=64",
           [] { (void)run_hyle_phase_batch(); },
           "C++ steady_clock phase batch N=64",
           [] { (void)run_chrono_phase_batch(); });
}
