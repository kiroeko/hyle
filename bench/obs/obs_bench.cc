#include <nanobench.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "hyle/obs/log.h"

#if defined(__APPLE__)
#include <pthread.h>
#endif

#if defined(HYLE_BENCH_ABSL)
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/log_entry.h>
#include <absl/log/log_sink.h>
#endif

namespace {
namespace nb = ankerl::nanobench;

// Darwin exposes QoS preference, not supported hard P/E-core affinity. Give an
// SSH-launched benchmark an explicit policy and retain utility as the
// heterogeneous-core audit; neither mode is described as physical pinning.
void configure_main_benchmark_thread() {
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
                 "hyle_obs_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int rc = pthread_set_qos_class_self_np(qos, 0);
  std::fprintf(stderr,
               "# HYLE_OBS_BENCH apple_main_qos=%s qos_ok=%d "
               "affinity=unsupported\n",
               name, rc == 0 ? 1 : 0);
  if (rc != 0)
    std::abort();
#endif
}

struct Counter {
  std::atomic<std::uint64_t> writes{0};
  std::atomic<std::uint64_t> flushes{0};

  hyle::obs::LogSink sink() noexcept {
    return hyle::obs::LogSink{this, &write, &flush};
  }
  static void write(void *state, const hyle::obs::LogRecord &) noexcept {
    static_cast<Counter *>(state)->writes.fetch_add(1,
                                                    std::memory_order_relaxed);
  }
  static void flush(void *state) noexcept {
    static_cast<Counter *>(state)->flushes.fetch_add(1,
                                                     std::memory_order_relaxed);
  }
};

#if defined(HYLE_BENCH_ABSL)
class AbslCounterSink final : public absl::LogSink {
public:
  explicit AbslCounterSink(Counter *counter) : counter_(counter) {}
  void Send(const absl::LogEntry &) override {
    counter_->writes.fetch_add(1, std::memory_order_relaxed);
  }

private:
  Counter *counter_;
};
#endif

std::uint64_t parallel_emit(std::size_t thread_count, std::size_t iterations) {
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([=] {
      for (std::size_t i = 0; i < iterations; ++i) {
        (void)hyle::obs::log_message(hyle::obs::LogLevel::Info, "ready");
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  return thread_count * iterations;
}

} // namespace

int main() {
  configure_main_benchmark_thread();
  using hyle::obs::LogField;
  using hyle::obs::LogLevel;
  using hyle::obs::LogRecord;

  if (!hyle::obs::set_default_sink_enabled(false).ok())
    return 1;

  {
    hyle::obs::set_level(LogLevel::Warn);
#if defined(HYLE_BENCH_ABSL)
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
#endif
    nb::Bench bench;
    bench.title("obs gated macro (arguments not formatted)")
        .relative(true)
        .batch(1)
        // Sub-nanosecond-to-low-nanosecond gates need a long epoch on Apple
        // heterogeneous cores; short epochs made the certificate measure QoS
        // migration more than the relaxed load itself.
        .minEpochIterations(3'500'000);
    bench.run("HYLE_LOG_INFO gated", [] {
      static int value = 7;
      HYLE_LOG_INFO("value={}", value);
      nb::doNotOptimizeAway(value);
    });
#if defined(HYLE_BENCH_ABSL)
    bench.run("Abseil ABSL_LOG(INFO) gated", [] {
      static int value = 7;
      ABSL_LOG(INFO) << "value=" << value;
      nb::doNotOptimizeAway(value);
    });
#endif
  }

  Counter counter;
#if defined(HYLE_BENCH_ABSL)
  AbslCounterSink absl_counter{&counter};
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
#endif
  if (!hyle::obs::add_sink(counter.sink()).ok())
    return 1;
  hyle::obs::set_level(LogLevel::Info);

  {
    constexpr LogField fields[] = {
        {"request_id", "a7f2"},
        {"peer", "10.0.0.8"},
        {"method", "GET"},
        {"route", "/models"},
    };
    const LogRecord structured{
        .level = LogLevel::Info,
        .message = "request complete",
        .fields = fields,
        .file = "server.cc",
        .function = "serve",
        .line = 42,
    };
    nb::Bench bench;
    bench.title("obs normal single-sink production")
        .relative(true)
        .batch(1)
        .minEpochIterations(250'000);
    bench.run("native preformatted", [] {
      nb::doNotOptimizeAway(hyle::obs::log_message(LogLevel::Info, "ready"));
    });
    bench.run("native structured 4 fields", [&] {
      nb::doNotOptimizeAway(hyle::obs::log_record(structured));
    });
    bench.run("native std::format macro", [] {
      static int value = 7;
      HYLE_LOG_INFO("value={} hex={:x}", value, value);
    });
#if defined(HYLE_BENCH_ABSL)
    bench.run("Abseil stream + ToSinkOnly", [&] {
      static int value = 7;
      ABSL_LOG(INFO).ToSinkOnly(&absl_counter)
          << "value=" << value << " hex=" << value;
    });
#endif
  }

  if (!hyle::obs::remove_sink(counter.sink()).ok())
    return 1;

  for (std::size_t sink_count :
       {std::size_t{1}, std::size_t{4}, std::size_t{16}}) {
    std::array<Counter, 16> counters{};
    for (std::size_t i = 0; i < sink_count; ++i) {
      if (!hyle::obs::add_sink(counters[i].sink()).ok())
        return 1;
    }
    nb::Bench bench;
    bench.title("obs fanout " + std::to_string(sink_count) + " sinks")
        .relative(true)
        .batch(sink_count)
        .minEpochIterations(50'000);
    bench.run("native preformatted", [] {
      nb::doNotOptimizeAway(hyle::obs::log_message(LogLevel::Info, "ready"));
    });
    for (std::size_t i = 0; i < sink_count; ++i) {
      if (!hyle::obs::remove_sink(counters[i].sink()).ok())
        return 1;
    }
  }

  Counter concurrent;
  if (!hyle::obs::add_sink(concurrent.sink()).ok())
    return 1;
  constexpr std::size_t kIterations = 100'000;
  for (std::size_t thread_count : {std::size_t{2}, std::size_t{8}}) {
    nb::Bench bench;
    bench
        .title("obs concurrent emit " + std::to_string(thread_count) +
               " threads")
        .relative(true)
        .batch(thread_count * kIterations)
        .epochs(7)
        .minEpochIterations(1);
    bench.run("immutable snapshot read path", [=] {
      nb::doNotOptimizeAway(parallel_emit(thread_count, kIterations));
    });
  }
  if (!hyle::obs::remove_sink(concurrent.sink()).ok())
    return 1;
  return 0;
}
