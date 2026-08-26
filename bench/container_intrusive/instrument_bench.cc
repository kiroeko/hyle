#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "hyle/core/compiler.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kAggregateCount = 4096;
constexpr std::size_t kBaseWorkCount = kAggregateCount * 4;
constexpr std::size_t kNearWorkCount = kBaseWorkCount * 67 / 64;
constexpr std::size_t kLargeWorkCount = kBaseWorkCount * 80 / 64;
volatile std::uint64_t g_known_delta_words[64]{};

extern "C" std::uintptr_t hyle_container_intrusive_observe_storage_phase(
    const void *storage, std::size_t byte_count) noexcept;

nb::Bench make_bench() {
  nb::Bench bench;
  // Every observation times a 4096-operation aggregate. This keeps the
  // measured interval far above the timer/read-counter floor without turning
  // a one-instruction synthetic loop into a product workload.
  bench.minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

struct ThreadQosState {
  const char *name;
  const char *ok;
};

ThreadQosState configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char *requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char *name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_container_intrusive_instrument_bench: "
                 "HYLE_BENCH_APPLE_QOS must be interactive or utility\n");
    std::abort();
  }
  const int result = pthread_set_qos_class_self_np(qos, 0);
  if (result != 0)
    std::abort();
  return {name, "1"};
#else
  return {"not-applicable", "not-applicable"};
#endif
}

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

HYLE_NOINLINE std::uint64_t
known_delta_probe(const std::uint64_t *values,
                  std::size_t work_count) noexcept {
  std::uint64_t checksum = 0;
  // Every variant enters this same runtime-bounded noinline scalar loop and
  // performs exactly the same work per iteration. A volatile load prevents
  // vectorization or closed-form folding. Only the trip count changes:
  // 64 base units, 67 near-MDE units (~4.69%), or 80 large units (25%).
  // This avoids the branch-predictor pathology of the rejected sparse-branch
  // control, where a nominal 1/32 event caused a ~28% wall-time slowdown.
  for (std::size_t index = 0; index < work_count; ++index) {
    checksum += values[index & (kAggregateCount - 1)];
    checksum += g_known_delta_words[index & 63U];
  }
  return checksum;
}

void bench_timer_floor(const std::uint64_t *values, bool rival_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument empty aggregate N=4096")
      .batch(kAggregateCount)
      .minEpochIterations(2000);
  const auto empty = [&] {
    // One compiler barrier per 4096 logical operations exposes the timer and
    // harness floor. It is telemetry only and never enters a product ratio.
    nb::doNotOptimizeAway(values);
  };
  if (rival_first) {
    bench.run("empty aggregate timer floor B", empty);
    bench.run("empty aggregate timer floor A", empty);
  } else {
    bench.run("empty aggregate timer floor A", empty);
    bench.run("empty aggregate timer floor B", empty);
  }
}

void bench_phase_observer_floor(const std::uint64_t *values,
                                bool rival_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument phase-observer floor N=4096")
      .batch(kAggregateCount)
      .minEpochIterations(2000);
  const auto two_phase_observers = [&] {
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        values, kAggregateCount * sizeof(*values)));
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        values, kAggregateCount * sizeof(*values)));
  };
  if (rival_first) {
    bench.run("phase observer fixed-call floor B", two_phase_observers);
    bench.run("phase observer fixed-call floor A", two_phase_observers);
  } else {
    bench.run("phase observer fixed-call floor A", two_phase_observers);
    bench.run("phase observer fixed-call floor B", two_phase_observers);
  }
}

void bench_aa(const std::uint64_t *values, bool rival_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument A/A aggregate N=4096")
      .batch(kAggregateCount)
      .minEpochIterations(2000);
  const auto identical = [&] {
    nb::doNotOptimizeAway(known_delta_probe(
        values, kBaseWorkCount));
  };
  if (rival_first) {
    bench.run("instrument identical B", identical);
    bench.run("instrument identical A", identical);
  } else {
    bench.run("instrument identical A", identical);
    bench.run("instrument identical B", identical);
  }
}

void bench_known_delta(const std::uint64_t *values, bool rival_first) {
  nb::Bench bench = make_bench();
  bench.title("instrument known-delta aggregate N=4096")
      .batch(kAggregateCount)
      .minEpochIterations(2000);
  const auto delta0 = [&] {
    nb::doNotOptimizeAway(known_delta_probe(values, kBaseWorkCount));
  };
  const auto delta_near = [&] {
    nb::doNotOptimizeAway(known_delta_probe(values, kNearWorkCount));
  };
  const auto delta_large = [&] {
    nb::doNotOptimizeAway(known_delta_probe(values, kLargeWorkCount));
  };
  if (rival_first) {
    bench.run("instrument control large=80/64 work", delta_large);
    bench.run("instrument control near=67/64 work", delta_near);
    bench.run("instrument control delta=0", delta0);
  } else {
    bench.run("instrument control delta=0", delta0);
    bench.run("instrument control near=67/64 work", delta_near);
    bench.run("instrument control large=80/64 work", delta_large);
  }
}

} // namespace

int main() {
  const ThreadQosState qos = configure_main_benchmark_thread();
  auto *values = new std::uint64_t[kAggregateCount];
  for (std::size_t index = 0; index < kAggregateCount; ++index)
    values[index] = static_cast<std::uint64_t>(index + 1);

  const bool rival_first = rival_first_requested();
  std::fprintf(stderr,
               "# HYLE_CONTAINER_INTRUSIVE_INSTRUMENT_RUN "
               "execution_order=%s apple_qos=%s qos_ok=%s "
               "affinity=external\n",
               rival_first ? "peer-first" : "candidate-first", qos.name,
               qos.ok);
  bench_timer_floor(values, rival_first);
  bench_phase_observer_floor(values, rival_first);
  bench_aa(values, rival_first);
  bench_known_delta(values, rival_first);
  delete[] values;
}
