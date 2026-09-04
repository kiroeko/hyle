// Complete structured-event consumer replay for Obs v4.
//
// Both routes build, validate, dispatch and fully consume the same borrowed
// typed records. The only paired difference is HYLE's concurrent-safe
// object-owned interval limiter versus the source-derived llama.cpp-shaped
// single-owner gate. This is intentionally not a getter microbenchmark.
#include <nanobench.h>

#include <array>
#include <atomic>
#include <bit>
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

namespace {
namespace nb = ankerl::nanobench;

constexpr std::array<std::string_view, 3> kStageNames{
    "text_model", "mmproj_model", "spec_model"};
constexpr std::size_t kTicksPerStage = 256;
constexpr std::size_t kMeasurementRounds = kTicksPerStage;
constexpr std::uint64_t kTickNs = 50'000'000;
constexpr std::uint64_t kIntervalNs = 200'000'000;
constexpr std::size_t kEmitsPerStage = 65; // ticks 0,4,...,252 plus final 255
constexpr std::size_t kEmits = kStageNames.size() * kEmitsPerStage;

struct HyleLoadProgressGate {
  hyle::obs::LogIntervalLimiter limiter;

  bool should_emit(std::uint64_t now_ns, bool done) noexcept {
    const bool interval = limiter.should_log_at(now_ns, kIntervalNs);
    return interval || done;
  }
};

// Source-derived from llama.cpp's load_progress_callback: t_last==0 marks the
// first sample, the final sample bypasses throttling, and intermediate samples
// are limited to one per 200 ms. The replay starts at a nonzero process time,
// matching ggml_time_ms() during model initialization.
struct ManualLoadProgressGate {
  std::uint64_t last_ns = 0;

  bool should_emit(std::uint64_t now_ns, bool done) noexcept {
    const bool first = last_ns == 0;
    const bool throttled = !first && !done && now_ns - last_ns < kIntervalNs;
    if (throttled)
      return false;
    last_ns = now_ns;
    return true;
  }
};

// Model a real synchronous logger adapter by reading every field into an
// observable deterministic checksum. Allocation, asynchronous copying and I/O
// belong to the host and would hide the native producer seam being assessed.
struct StructuredConsumer {
  std::atomic<std::uint64_t> writes{0};
  std::atomic<std::uint64_t> checksum{0};

  void write(const hyle::obs::LogRecord &record) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash ^= static_cast<std::uint64_t>(record.level);
    hash = consume_bytes(hash, record.category);
    hash = consume_bytes(hash, record.event);
    hash = consume_bytes(hash, record.message);
    for (const hyle::obs::LogAttribute &attribute : record.attributes) {
      hash = consume_bytes(hash, attribute.key);
      hash = consume_value(hash, attribute.value);
    }
    for (const hyle::obs::LogAttribute &attribute :
         record.correlation_attributes) {
      hash = consume_bytes(hash, attribute.key);
      hash = consume_value(hash, attribute.value);
    }
    hash = consume_bytes(hash, record.source.file);
    hash = consume_bytes(hash, record.source.function);
    hash ^= record.source.line;
    hash ^= static_cast<std::uint64_t>(record.source.column) << 32U;
    if (record.timestamp.has_value()) {
      hash ^= static_cast<std::uint64_t>(record.timestamp->seconds);
      hash ^= static_cast<std::uint64_t>(record.timestamp->nanoseconds) << 17U;
    }
    checksum.fetch_add(hash | 1U, std::memory_order_relaxed);
    writes.fetch_add(1, std::memory_order_relaxed);
  }

private:
  static std::uint64_t consume_bytes(std::uint64_t hash,
                                     std::string_view bytes) noexcept {
    for (const unsigned char byte : bytes) {
      hash = (hash ^ byte) * 1099511628211ULL;
    }
    return (hash ^ bytes.size()) * 1099511628211ULL;
  }

  static std::uint64_t
  consume_value(std::uint64_t hash, const hyle::obs::LogValue &value) noexcept {
    hash ^= static_cast<std::uint64_t>(value.kind()) + 1U;
    switch (value.kind()) {
    case hyle::obs::LogValueKind::String:
      return consume_bytes(hash, value.as_string());
    case hyle::obs::LogValueKind::SignedInteger:
      return hash ^ static_cast<std::uint64_t>(value.as_signed_integer());
    case hyle::obs::LogValueKind::UnsignedInteger:
      return hash ^ value.as_unsigned_integer();
    case hyle::obs::LogValueKind::FloatingPoint:
      return hash ^ std::bit_cast<std::uint64_t>(value.as_floating_point());
    case hyle::obs::LogValueKind::Boolean:
      return hash ^ static_cast<std::uint64_t>(value.as_boolean());
    }
    HYLE_UNREACHABLE();
  }
};

template <class Gate> HYLE_NOINLINE std::uint64_t run_progress_batch() {
  std::array<Gate, kStageNames.size()> stages{};
  std::uint64_t emitted = 0;
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    for (std::size_t tick = 0; tick < kTicksPerStage; ++tick) {
      const std::uint64_t now_ns = 1'000'000'000 + tick * kTickNs;
      const bool done = tick + 1 == kTicksPerStage;
      if (!stages[stage].should_emit(now_ns, done))
        continue;

      const hyle::obs::LogAttribute attributes[] = {
          {"stage", kStageNames[0]},
          {"stage", kStageNames[1]},
          {"stage", kStageNames[2]},
          {"current", kStageNames[stage]},
          {"value",
           static_cast<double>(tick) / static_cast<double>(kTicksPerStage - 1)},
      };
      const hyle::obs::LogAttribute correlation{"model_load_id",
                                                std::uint64_t{0xa7f2}};
      const hyle::obs::LogRecord record{
          .level = hyle::obs::LogLevel::Info,
          .category = "inference.model",
          .event = "model.load_progress",
          .attributes = attributes,
          .correlation_attributes =
              std::span<const hyle::obs::LogAttribute>(&correlation, 1),
          .source = {.file = "server-context.cpp",
                     .function = "load_progress_callback",
                     .line = 979,
                     .column = 7},
          .timestamp = hyle::time::unix_timestamp(
              1'700'000'000 + static_cast<std::int64_t>(tick / 20U),
              static_cast<std::uint32_t>(tick % 20U) * 50'000'000U),
      };
      emitted += static_cast<std::uint64_t>(hyle::obs::log_record(record));
    }
  }
  return emitted;
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

int validate(StructuredConsumer &consumer) {
  const std::uint64_t writes_before =
      consumer.writes.load(std::memory_order_relaxed);
  const std::uint64_t checksum_before =
      consumer.checksum.load(std::memory_order_relaxed);
  const std::uint64_t hyle = run_progress_batch<HyleLoadProgressGate>();
  const std::uint64_t manual = run_progress_batch<ManualLoadProgressGate>();
  const std::uint64_t writes =
      consumer.writes.load(std::memory_order_relaxed) - writes_before;
  const bool consumed =
      consumer.checksum.load(std::memory_order_relaxed) != checksum_before;
  const bool passed =
      hyle == kEmits && manual == kEmits && writes == 2 * kEmits && consumed;
  std::printf("# HYLE_OBS_STRUCTURED_CONTRACT schema=1 hyle=%" PRIu64
              " manual=%" PRIu64 " writes=%" PRIu64
              " fully_consumed=%d verdict=%s\n",
              hyle, manual, writes, consumed ? 1 : 0, passed ? "PASS" : "FAIL");
  return passed ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
  if (!hyle::obs::set_default_sink_enabled(false).ok())
    return 1;
  hyle::obs::set_level(hyle::obs::LogLevel::Info);

  StructuredConsumer consumer;
  auto registration = hyle::obs::SinkRegistration::create(consumer);
  if (!registration)
    return 1;
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract")
    return validate(consumer);

  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool peer_first = peer_first_requested();
  std::fprintf(stderr,
               "# HYLE_OBS_STRUCTURED_RUN benchmark=obs-structured "
               "execution_order=%s apple_qos=%s qos_ok=%s "
               "affinity=external run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
               hyle::bench::perf::run_nonce());

  nb::Bench bench;
  bench.title("Inference model-load structured progress 3 stages x 256 ticks")
      .relative(true)
      // One reported operation is one progress tick across the three possible
      // model stages; each nanobench callback performs 256 such rounds.
      .batch(kMeasurementRounds)
      .minEpochIterations(64);
  run_pair(
      bench, peer_first, "HYLE load gate + typed one adapter",
      [&] {
        nb::doNotOptimizeAway(run_progress_batch<HyleLoadProgressGate>());
        nb::doNotOptimizeAway(
            consumer.checksum.load(std::memory_order_relaxed));
      },
      "llama load gate + typed one adapter",
      [&] {
        nb::doNotOptimizeAway(run_progress_batch<ManualLoadProgressGate>());
        nb::doNotOptimizeAway(
            consumer.checksum.load(std::memory_order_relaxed));
      });
  return 0;
}
