// Consumer-shaped evidence for Obs call-site control.
//
// The ordinary row compares the natural HYLE and Abseil public macro forms.
// The inference row replays llama.cpp's per-slot three-second progress cadence
// over the same HYLE one-adapter emission path; its manual gate is a mechanism
// diagnostic, not an external SOTA peer.
#include <nanobench.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/obs/log.h"

#include "../common/perf_process.h"

#if defined(HYLE_BENCH_ABSL)
#include "absl/log/absl_log.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kOrdinaryChecks = 4096;
constexpr std::uint64_t kEveryN = 64;
constexpr std::size_t kOrdinaryEmits = kOrdinaryChecks / kEveryN;
constexpr std::size_t kInferenceSlots = 16;
constexpr std::size_t kInferenceTicks = 256;
constexpr std::uint64_t kTickNs = 50'000'000;
constexpr std::uint64_t kProgressIntervalNs = 3'000'000'000;
constexpr std::size_t kInferenceChecks = kInferenceSlots * kInferenceTicks;
constexpr std::size_t kEmitsPerSlot = 5; // ticks 0, 60, 120, 180, 240
constexpr std::size_t kInferenceEmits = kInferenceSlots * kEmitsPerSlot;

struct Counter {
  std::atomic<std::uint64_t> writes{0};

  hyle::obs::LogSink sink() noexcept { return hyle::obs::LogSink::from(*this); }
  void write(const hyle::obs::LogRecord &) noexcept {
    writes.fetch_add(1, std::memory_order_relaxed);
  }
};

#if defined(HYLE_BENCH_ABSL)
class AbslCounterSink final : public absl::LogSink {
public:
  explicit AbslCounterSink(Counter &counter) noexcept : counter_(&counter) {}

  void Send(const absl::LogEntry &) override {
    counter_->writes.fetch_add(1, std::memory_order_relaxed);
  }

private:
  Counter *counter_;
};
#endif

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

template <class CandidateOp, class PeerOp>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, CandidateOp &&candidate,
              std::string_view peer_name, PeerOp &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<PeerOp>(peer));
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
  } else {
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
    bench.run(peer_name, std::forward<PeerOp>(peer));
  }
}

HYLE_NOINLINE void hyle_ordinary_batch(std::uint64_t seed) {
  for (std::size_t i = 0; i < kOrdinaryChecks; ++i) {
    HYLE_LOG_INFO_EVERY_N(kEveryN, "request={} progress={}", seed, i);
  }
}

HYLE_NOINLINE void hyle_ordinary_validation_batch(std::uint64_t seed) {
  for (std::size_t i = 0; i < kOrdinaryChecks; ++i) {
    HYLE_LOG_INFO_EVERY_N(kEveryN, "request={} progress={}", seed, i);
  }
}

#if defined(HYLE_BENCH_ABSL)
#if defined(_MSC_VER)
// Abseil exposes a streamable COUNTER token from its stateful macro. MSVC
// diagnoses that token when the natural consumer spelling does not stream it.
// Keep /WX for HYLE and silence only this peer-owned expansion.
#pragma warning(push)
#pragma warning(disable : 4189)
#endif
HYLE_NOINLINE void absl_ordinary_batch(AbslCounterSink &sink,
                                       std::uint64_t seed) {
  for (std::size_t i = 0; i < kOrdinaryChecks; ++i) {
    ABSL_LOG_EVERY_N(INFO, kEveryN).ToSinkOnly(&sink)
        << "request=" << seed << " progress=" << i;
  }
}

HYLE_NOINLINE void absl_ordinary_validation_batch(AbslCounterSink &sink,
                                                  std::uint64_t seed) {
  for (std::size_t i = 0; i < kOrdinaryChecks; ++i) {
    ABSL_LOG_EVERY_N(INFO, kEveryN).ToSinkOnly(&sink)
        << "request=" << seed << " progress=" << i;
  }
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

struct ManualProgressGate {
  bool initialized = false;
  std::uint64_t last_ns = 0;

  bool should_log_at(std::uint64_t now_ns, std::uint64_t interval_ns) noexcept {
    if (interval_ns == 0)
      return true;
    if (!initialized) {
      initialized = true;
      last_ns = now_ns;
      return true;
    }
    if (now_ns < last_ns || now_ns - last_ns < interval_ns)
      return false;
    last_ns = now_ns;
    return true;
  }
};

template <class Gate> HYLE_NOINLINE std::uint64_t inference_progress_batch() {
  std::array<Gate, kInferenceSlots> slots{};
  std::uint64_t emitted = 0;
  for (std::size_t tick = 0; tick < kInferenceTicks; ++tick) {
    const std::uint64_t now_ns = tick * kTickNs;
    for (Gate &slot : slots) {
      if (slot.should_log_at(now_ns, kProgressIntervalNs)) {
        emitted += static_cast<std::uint64_t>(
            hyle::obs::log_message(hyle::obs::LogLevel::Info, "slot progress"));
      }
    }
  }
  return emitted;
}

template <class Gate> HYLE_NOINLINE std::uint64_t inference_gate_batch() {
  std::array<Gate, kInferenceSlots> slots{};
  std::uint64_t admitted = 0;
  for (std::size_t tick = 0; tick < kInferenceTicks; ++tick) {
    const std::uint64_t now_ns = tick * kTickNs;
    for (Gate &slot : slots) {
      admitted += static_cast<std::uint64_t>(
          slot.should_log_at(now_ns, kProgressIntervalNs));
    }
  }
  return admitted;
}

int validate_contract(Counter &hyle_counter
#if defined(HYLE_BENCH_ABSL)
                      ,
                      Counter &absl_counter, AbslCounterSink &absl_sink
#endif
) {
  const std::uint64_t hyle_before =
      hyle_counter.writes.load(std::memory_order_relaxed);
  hyle_ordinary_validation_batch(7);
  const std::uint64_t hyle_ordinary =
      hyle_counter.writes.load(std::memory_order_relaxed) - hyle_before;

  std::uint64_t absl_ordinary = 0;
#if defined(HYLE_BENCH_ABSL)
  const std::uint64_t absl_before =
      absl_counter.writes.load(std::memory_order_relaxed);
  absl_ordinary_validation_batch(absl_sink, 7);
  absl_ordinary =
      absl_counter.writes.load(std::memory_order_relaxed) - absl_before;
#endif

  const std::uint64_t inference_before =
      hyle_counter.writes.load(std::memory_order_relaxed);
  const std::uint64_t hyle_inference =
      inference_progress_batch<hyle::obs::LogIntervalLimiter>();
  const std::uint64_t after_hyle_inference =
      hyle_counter.writes.load(std::memory_order_relaxed);
  const std::uint64_t manual_inference =
      inference_progress_batch<ManualProgressGate>();
  const std::uint64_t inference_writes =
      hyle_counter.writes.load(std::memory_order_relaxed) - inference_before;
  const std::uint64_t hyle_gate =
      inference_gate_batch<hyle::obs::LogIntervalLimiter>();
  const std::uint64_t manual_gate = inference_gate_batch<ManualProgressGate>();

  const bool passed =
      hyle_ordinary == kOrdinaryEmits &&
#if defined(HYLE_BENCH_ABSL)
      absl_ordinary == kOrdinaryEmits &&
#endif
      hyle_inference == kInferenceEmits &&
      manual_inference == kInferenceEmits && hyle_gate == kInferenceEmits &&
      manual_gate == kInferenceEmits &&
      after_hyle_inference - inference_before == kInferenceEmits &&
      inference_writes == 2 * kInferenceEmits;
  std::printf("# HYLE_OBS_CONTROL_CONTRACT schema=1 ordinary_checks=%zu "
              "hyle_ordinary=%" PRIu64 " absl_ordinary=%" PRIu64
              " inference_slots=%zu inference_ticks=%zu hyle_inference=%" PRIu64
              " manual_inference=%" PRIu64 " hyle_gate=%" PRIu64
              " manual_gate=%" PRIu64 " verdict=%s\n",
              kOrdinaryChecks, hyle_ordinary, absl_ordinary, kInferenceSlots,
              kInferenceTicks, hyle_inference, manual_inference, hyle_gate,
              manual_gate, passed ? "PASS" : "FAIL");
  return passed ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
  if (!hyle::obs::set_default_sink_enabled(false).ok())
    return 1;
  hyle::obs::set_level(hyle::obs::LogLevel::Info);

  Counter hyle_counter;
  auto registration = hyle::obs::SinkRegistration::create(hyle_counter);
  if (!registration)
    return 1;

#if defined(HYLE_BENCH_ABSL)
  Counter absl_counter;
  AbslCounterSink absl_sink(absl_counter);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
#endif

  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract") {
    return validate_contract(hyle_counter
#if defined(HYLE_BENCH_ABSL)
                             ,
                             absl_counter, absl_sink
#endif
    );
  }

  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool peer_first = peer_first_requested();
  std::fprintf(stderr,
               "# HYLE_OBS_CONTROL_RUN benchmark=obs-control "
               "execution_order=%s apple_qos=%s qos_ok=%s "
               "affinity=external run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
               hyle::bench::perf::run_nonce());

  {
    nb::Bench bench;
    bench.title("Ordinary call-site every-N aggregate N=4096")
        .relative(true)
        .batch(kOrdinaryChecks)
        .minEpochIterations(512);
#if defined(HYLE_BENCH_ABSL)
    run_pair(
        bench, peer_first, "HYLE ordinary every-N one adapter",
        [&] {
          hyle_ordinary_batch(11);
          nb::doNotOptimizeAway(
              hyle_counter.writes.load(std::memory_order_relaxed));
        },
        "Abseil ordinary every-N ToSinkOnly",
        [&] {
          absl_ordinary_batch(absl_sink, 11);
          nb::doNotOptimizeAway(
              absl_counter.writes.load(std::memory_order_relaxed));
        });
#else
    bench.run("HYLE ordinary every-N one adapter", [&] {
      hyle_ordinary_batch(11);
      nb::doNotOptimizeAway(
          hyle_counter.writes.load(std::memory_order_relaxed));
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Inference interval gate 16 slots x 256 ticks")
        .relative(true)
        .batch(kInferenceChecks)
        .minEpochIterations(256);
    run_pair(
        bench, peer_first, "HYLE object-owned interval gate",
        [&] {
          nb::doNotOptimizeAway(
              inference_gate_batch<hyle::obs::LogIntervalLimiter>());
        },
        "llama-shaped manual interval gate",
        [&] {
          nb::doNotOptimizeAway(inference_gate_batch<ManualProgressGate>());
        });
  }

  {
    nb::Bench bench;
    bench.title("Inference progress 16 slots x 256 ticks")
        .relative(true)
        .batch(kInferenceChecks)
        .minEpochIterations(64);
    run_pair(
        bench, peer_first, "HYLE object-owned interval + one adapter",
        [&] {
          nb::doNotOptimizeAway(
              inference_progress_batch<hyle::obs::LogIntervalLimiter>());
        },
        "llama-shaped manual interval + same adapter",
        [&] {
          nb::doNotOptimizeAway(inference_progress_batch<ManualProgressGate>());
        });
  }

  return 0;
}
