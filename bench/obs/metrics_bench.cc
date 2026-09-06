/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include <nanobench.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/obs/metrics.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kRequestsPerIteration = 64;
constexpr std::array<double, 8> kLatencySeconds{0.0008, 0.0016, 0.0024, 0.0048,
                                                0.0075, 0.013,  0.031,  0.071};
constexpr std::array<double, 6> kLatencyBounds{0.001, 0.002, 0.005,
                                               0.01,  0.025, 0.05};

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "hyle_obs_metrics_bench: %s\n", message);
  std::abort();
}

void require(bool condition, const char *message) {
  if (!condition)
    fail(message);
}

bool peer_first_requested() {
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

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

std::size_t
latency_bucket(const std::array<double, kLatencyBounds.size()> &bounds,
               double value) noexcept {
  return static_cast<std::size_t>(
      std::lower_bound(bounds.begin(), bounds.end(), value) - bounds.begin());
}

HYLE_FORCE_INLINE void direct_add_counter(double &counter,
                                          double delta) noexcept {
  HYLE_FATAL_ASSERT_MSG(delta >= 0.0 &&
                            delta <= std::numeric_limits<double>::max(),
                        "counter delta must be finite and non-negative");
  counter += delta;
}

HYLE_FORCE_INLINE void direct_add_counter(std::atomic<double> &counter,
                                          double delta) noexcept {
  HYLE_FATAL_ASSERT_MSG(delta >= 0.0 &&
                            delta <= std::numeric_limits<double>::max(),
                        "counter delta must be finite and non-negative");
  counter.fetch_add(delta, std::memory_order_relaxed);
}

template <class Counts, class Sum>
HYLE_FORCE_INLINE bool
direct_observe(const std::array<double, kLatencyBounds.size()> &bounds,
               Counts &counts, Sum &sum, double value,
               std::uint64_t multiplicity) noexcept {
  if (multiplicity == 0)
    return true;
  if (!(value >= std::numeric_limits<double>::lowest() &&
        value <= std::numeric_limits<double>::max())) {
    return false;
  }
  const std::size_t index = latency_bucket(bounds, value);
  if constexpr (std::is_same_v<typename Counts::value_type, std::uint64_t>) {
    counts[index] += multiplicity;
    sum += value * static_cast<double>(multiplicity);
  } else {
    counts[index].fetch_add(multiplicity, std::memory_order_relaxed);
    sum.fetch_add(value * static_cast<double>(multiplicity),
                  std::memory_order_relaxed);
  }
  return true;
}

struct OwnerDirectMetrics {
  explicit OwnerDirectMetrics(
      std::array<double, kLatencyBounds.size()> histogram_bounds) noexcept
      : bounds(std::move(histogram_bounds)) {}

  std::uint64_t iterations = 0;
  std::uint64_t requests = 0;
  std::uint64_t tokens = 0;
  double scheduled_seconds = 0.0;
  std::uint64_t running = 0;
  double cache_usage = 0.0;
  std::array<double, kLatencyBounds.size()> bounds;
  std::array<std::uint64_t, kLatencyBounds.size() + 1> latency_counts{};
  double latency_sum = 0.0;
};

struct OwnerHyleMetrics {
  hyle::obs::Counter<> iterations;
  hyle::obs::Counter<> requests;
  hyle::obs::Counter<> tokens;
  hyle::obs::Counter<double> scheduled_seconds;
  hyle::obs::Gauge<std::uint64_t> running;
  hyle::obs::Gauge<> cache_usage;
  hyle::obs::FixedHistogram<kLatencyBounds.size()> latency;

  explicit OwnerHyleMetrics(
      hyle::obs::FixedBuckets<kLatencyBounds.size()> buckets) noexcept
      : latency(std::move(buckets)) {}
};

struct SharedDirectMetrics {
  explicit SharedDirectMetrics(
      std::array<double, kLatencyBounds.size()> histogram_bounds) noexcept
      : bounds(std::move(histogram_bounds)) {}

  std::atomic<std::uint64_t> iterations{};
  std::atomic<std::uint64_t> requests{};
  std::atomic<std::uint64_t> tokens{};
  std::atomic<double> scheduled_seconds{};
  std::atomic<std::uint64_t> running{};
  std::atomic<double> cache_usage{};
  std::array<double, kLatencyBounds.size()> bounds;
  std::array<std::atomic<std::uint64_t>, kLatencyBounds.size() + 1>
      latency_counts{};
  std::atomic<double> latency_sum{};
};

struct SharedHyleMetrics {
  hyle::obs::ConcurrentCounter<> iterations;
  hyle::obs::ConcurrentCounter<> requests;
  hyle::obs::ConcurrentCounter<> tokens;
  hyle::obs::ConcurrentCounter<double> scheduled_seconds;
  hyle::obs::ConcurrentGauge<std::uint64_t> running;
  hyle::obs::ConcurrentGauge<> cache_usage;
  hyle::obs::ConcurrentFixedHistogram<kLatencyBounds.size()> latency;

  explicit SharedHyleMetrics(
      hyle::obs::FixedBuckets<kLatencyBounds.size()> buckets) noexcept
      : latency(std::move(buckets)) {}
};

HYLE_NOINLINE void update_owner(OwnerHyleMetrics &metrics,
                                std::uint64_t seed) noexcept {
  metrics.iterations.add();
  metrics.running.set(12 + seed % 7);
  metrics.cache_usage.set(0.45 + static_cast<double>(seed % 31) / 100.0);
  for (std::size_t index = 0; index < kRequestsPerIteration; ++index) {
    const std::uint64_t tokens = 1 + (seed + index * 13) % 8;
    const double latency =
        kLatencySeconds[(seed + index) % kLatencySeconds.size()];
    metrics.requests.add();
    metrics.tokens.add(tokens);
    metrics.scheduled_seconds.add(latency);
    require(metrics.latency.observe(latency, tokens), "HYLE owner observe");
  }
}

HYLE_NOINLINE void update_owner(OwnerDirectMetrics &metrics,
                                std::uint64_t seed) noexcept {
  ++metrics.iterations;
  metrics.running = 12 + seed % 7;
  metrics.cache_usage = 0.45 + static_cast<double>(seed % 31) / 100.0;
  for (std::size_t index = 0; index < kRequestsPerIteration; ++index) {
    const std::uint64_t tokens = 1 + (seed + index * 13) % 8;
    const double latency =
        kLatencySeconds[(seed + index) % kLatencySeconds.size()];
    ++metrics.requests;
    metrics.tokens += tokens;
    direct_add_counter(metrics.scheduled_seconds, latency);
    require(direct_observe(metrics.bounds, metrics.latency_counts,
                           metrics.latency_sum, latency, tokens),
            "direct owner observe");
  }
}

HYLE_NOINLINE void update_shared(SharedHyleMetrics &metrics,
                                 std::uint64_t seed) noexcept {
  metrics.iterations.add();
  metrics.running.set(12 + seed % 7);
  metrics.cache_usage.set(0.45 + static_cast<double>(seed % 31) / 100.0);
  for (std::size_t index = 0; index < kRequestsPerIteration; ++index) {
    const std::uint64_t tokens = 1 + (seed + index * 13) % 8;
    const double latency =
        kLatencySeconds[(seed + index) % kLatencySeconds.size()];
    metrics.requests.add();
    metrics.tokens.add(tokens);
    metrics.scheduled_seconds.add(latency);
    require(metrics.latency.observe(latency, tokens), "HYLE shared observe");
  }
}

HYLE_NOINLINE void update_shared(SharedDirectMetrics &metrics,
                                 std::uint64_t seed) noexcept {
  metrics.iterations.fetch_add(1, std::memory_order_relaxed);
  metrics.running.store(12 + seed % 7, std::memory_order_relaxed);
  metrics.cache_usage.store(0.45 + static_cast<double>(seed % 31) / 100.0,
                            std::memory_order_relaxed);
  for (std::size_t index = 0; index < kRequestsPerIteration; ++index) {
    const std::uint64_t tokens = 1 + (seed + index * 13) % 8;
    const double latency =
        kLatencySeconds[(seed + index) % kLatencySeconds.size()];
    metrics.requests.fetch_add(1, std::memory_order_relaxed);
    metrics.tokens.fetch_add(tokens, std::memory_order_relaxed);
    direct_add_counter(metrics.scheduled_seconds, latency);
    require(direct_observe(metrics.bounds, metrics.latency_counts,
                           metrics.latency_sum, latency, tokens),
            "direct shared observe");
  }
}

bool equivalent_after_one_iteration(
    const hyle::obs::FixedBuckets<kLatencyBounds.size()> &buckets) {
  OwnerHyleMetrics candidate(buckets);
  OwnerDirectMetrics peer(buckets.values());
  update_owner(candidate, 17);
  update_owner(peer, 17);
  const auto snapshot = candidate.latency.snapshot();
  return candidate.iterations.value() == peer.iterations &&
         candidate.requests.value() == peer.requests &&
         candidate.tokens.value() == peer.tokens &&
         candidate.scheduled_seconds.value() == peer.scheduled_seconds &&
         candidate.running.value() == peer.running &&
         candidate.cache_usage.value() == peer.cache_usage &&
         snapshot.bucket_counts() == peer.latency_counts &&
         snapshot.sum() == peer.latency_sum;
}

bool equivalent_shared_after_one_iteration(
    const hyle::obs::FixedBuckets<kLatencyBounds.size()> &buckets) {
  SharedHyleMetrics candidate(buckets);
  SharedDirectMetrics peer(buckets.values());
  update_shared(candidate, 17);
  update_shared(peer, 17);
  const auto snapshot = candidate.latency.snapshot();
  std::array<std::uint64_t, kLatencyBounds.size() + 1> peer_counts{};
  for (std::size_t index = 0; index < peer_counts.size(); ++index) {
    peer_counts[index] =
        peer.latency_counts[index].load(std::memory_order_relaxed);
  }
  return candidate.iterations.value() ==
             peer.iterations.load(std::memory_order_relaxed) &&
         candidate.requests.value() ==
             peer.requests.load(std::memory_order_relaxed) &&
         candidate.tokens.value() ==
             peer.tokens.load(std::memory_order_relaxed) &&
         candidate.scheduled_seconds.value() ==
             peer.scheduled_seconds.load(std::memory_order_relaxed) &&
         candidate.running.value() ==
             peer.running.load(std::memory_order_relaxed) &&
         candidate.cache_usage.value() ==
             peer.cache_usage.load(std::memory_order_relaxed) &&
         snapshot.bucket_counts() == peer_counts &&
         snapshot.sum() == peer.latency_sum.load(std::memory_order_relaxed);
}

} // namespace

int main(int argc, char **argv) {
  auto buckets =
      hyle::obs::FixedBuckets<kLatencyBounds.size()>::create(kLatencyBounds);
  require(buckets.has_value(), "fixed bounds construction");
  require(equivalent_after_one_iteration(*buckets),
          "owner candidate/peer semantic mismatch");
  require(equivalent_shared_after_one_iteration(*buckets),
          "shared candidate/peer semantic mismatch");
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract") {
    std::printf("# HYLE_OBS_METRICS_CONTRACT schema=1 requests=%zu "
                "bounds=%zu owner_equivalent=PASS "
                "shared_equivalent=PASS verdict=PASS\n",
                kRequestsPerIteration, kLatencyBounds.size());
    return 0;
  }

  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = peer_first_requested();
  std::fprintf(stderr,
               "# HYLE_OBS_METRICS_RUN execution_order=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", affinity.topology,
               affinity.cpu, affinity.ok, hyle::bench::perf::run_nonce());

  OwnerHyleMetrics owner_candidate(*buckets);
  OwnerDirectMetrics owner_peer(buckets->values());
  {
    nb::Bench bench;
    bench.title("Owner scheduler metrics iteration")
        .relative(true)
        .batch(1)
        .minEpochTime(std::chrono::milliseconds(8));
    run_pair(
        bench, peer_first, "HYLE owner-thread metrics iteration",
        [&] {
          update_owner(owner_candidate, owner_candidate.iterations.value());
          nb::doNotOptimizeAway(owner_candidate.iterations.value());
        },
        "Direct owner-thread metrics iteration",
        [&] {
          update_owner(owner_peer, owner_peer.iterations);
          nb::doNotOptimizeAway(owner_peer.iterations);
        });
  }

  SharedHyleMetrics shared_candidate(*buckets);
  SharedDirectMetrics shared_peer(buckets->values());
  {
    nb::Bench bench;
    bench.title("Shared scheduler metrics iteration")
        .relative(true)
        .batch(1)
        .minEpochTime(std::chrono::milliseconds(8));
    run_pair(
        bench, peer_first, "HYLE concurrent metrics iteration",
        [&] {
          update_shared(shared_candidate, shared_candidate.iterations.value());
          nb::doNotOptimizeAway(shared_candidate.iterations.value());
        },
        "Direct relaxed-atomic metrics iteration",
        [&] {
          update_shared(shared_peer,
                        shared_peer.iterations.load(std::memory_order_relaxed));
          nb::doNotOptimizeAway(
              shared_peer.iterations.load(std::memory_order_relaxed));
        });
  }

  require(owner_candidate.iterations.value() != 0 && owner_peer.iterations != 0,
          "owner benchmark work was eliminated");
  require(shared_candidate.iterations.value() != 0 &&
              shared_peer.iterations.load(std::memory_order_relaxed) != 0,
          "shared benchmark work was eliminated");
  return 0;
}
