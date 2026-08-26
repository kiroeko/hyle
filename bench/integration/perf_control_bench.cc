#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../common/perf_process.h"
#include "hyle/core/compiler.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kLogicalOperationCount = 4096;
constexpr std::size_t kBaseWorkCount = kLogicalOperationCount * 4;
constexpr std::size_t kFivePercentWorkCount =
    kBaseWorkCount + kBaseWorkCount / 20;
constexpr std::size_t kTenPercentWorkCount =
    kBaseWorkCount + kBaseWorkCount / 10;

volatile std::uint64_t g_control_words[64]{};

nb::Bench make_bench() {
  nb::Bench bench;
  // One observation represents a large aggregate. This keeps the measured
  // interval above the clock/harness floor without pretending this synthetic
  // loop is a product workload.
  bench.minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

bool peer_first_requested() {
  return std::getenv("HYLE_PERF_PEER_FIRST") != nullptr;
}

HYLE_NOINLINE std::uint64_t
control_probe(const std::uint64_t *values, std::size_t work_count) noexcept {
  std::uint64_t checksum = 0;
  // Runtime-bounded, noinline and volatile-backed work prevents closed-form
  // folding. Every variant has the same loop body; only its trip count changes.
  for (std::size_t index = 0; index < work_count; ++index) {
    checksum += values[index & (kLogicalOperationCount - 1)];
    checksum += g_control_words[index & 63U];
  }
  return checksum;
}

template <typename First, typename Second>
void run_pair(nb::Bench &bench, bool peer_first, const char *candidate_name,
              First &&candidate, const char *peer_name, Second &&peer) {
  if (peer_first) {
    bench.run(peer_name, peer);
    bench.run(candidate_name, candidate);
  } else {
    bench.run(candidate_name, candidate);
    bench.run(peer_name, peer);
  }
}

template <std::size_t Batch>
void bench_timer_floor(const std::uint64_t *values, bool peer_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument timer floor batch=" + std::to_string(Batch))
      .batch(Batch)
      .minEpochIterations(2000);
  const auto empty = [&] { nb::doNotOptimizeAway(values); };
  run_pair(bench, peer_first, "timer floor candidate", empty,
           "timer floor peer", empty);
}

void bench_aa(const std::uint64_t *values, bool peer_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument A/A N=4096")
      .batch(kLogicalOperationCount)
      .minEpochIterations(20000);
  const auto identical = [&] {
    nb::doNotOptimizeAway(control_probe(values, kBaseWorkCount));
  };
  run_pair(bench, peer_first, "A/A candidate", identical, "A/A peer",
           identical);
}

void bench_known_delta(const std::uint64_t *values, bool peer_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument known delta N=4096")
      .batch(kLogicalOperationCount)
      .minEpochIterations(20000);
  const auto candidate = [&] {
    nb::doNotOptimizeAway(control_probe(values, kBaseWorkCount));
  };
  const auto peer_five_percent_slower = [&] {
    nb::doNotOptimizeAway(control_probe(values, kFivePercentWorkCount));
  };
  const auto peer_ten_percent_slower = [&] {
    nb::doNotOptimizeAway(control_probe(values, kTenPercentWorkCount));
  };

  // Each pair is isolated in its own table so row mapping is unambiguous.
  run_pair(bench, peer_first, "delta5 candidate", candidate,
           "delta5 peer slower", peer_five_percent_slower);

  nb::Bench ten_percent = make_bench();
  ten_percent.title("instrument known delta 10 percent N=4096")
      .batch(kLogicalOperationCount)
      .minEpochIterations(20000);
  run_pair(ten_percent, peer_first, "delta10 candidate", candidate,
           "delta10 peer slower", peer_ten_percent_slower);
}

} // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos();
  auto *values = new std::uint64_t[kLogicalOperationCount];
  for (std::size_t index = 0; index < kLogicalOperationCount; ++index)
    values[index] = static_cast<std::uint64_t>(index + 1);

  const bool peer_first = peer_first_requested();
  std::fprintf(stderr,
               "# HYLE_PERF_CONTROL_RUN execution_order=%s apple_qos=%s "
               "qos_ok=%s affinity=external base_work=%zu delta5_work=%zu "
               "delta10_work=%zu run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", qos.name,
               qos.ok, kBaseWorkCount, kFivePercentWorkCount,
               kTenPercentWorkCount, hyle::bench::perf::run_nonce());
  bench_timer_floor<1>(values, peer_first);
  bench_timer_floor<2>(values, peer_first);
  bench_timer_floor<4>(values, peer_first);
  bench_timer_floor<8>(values, peer_first);
  bench_timer_floor<16>(values, peer_first);
  bench_timer_floor<32>(values, peer_first);
  bench_timer_floor<64>(values, peer_first);
  bench_timer_floor<128>(values, peer_first);
  bench_timer_floor<256>(values, peer_first);
  bench_timer_floor<512>(values, peer_first);
  bench_timer_floor<1024>(values, peer_first);
  bench_timer_floor<2048>(values, peer_first);
  bench_timer_floor<4096>(values, peer_first);
  bench_aa(values, peer_first);
  bench_known_delta(values, peer_first);
  delete[] values;
}
