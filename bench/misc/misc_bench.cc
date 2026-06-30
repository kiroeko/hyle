// misc-module bench (gap-fill #37): the performance-relevant primitives that had
// no benchmark. Each races the std/obvious equivalent (win or lose, printed):
//   * util/random  -- Rng (xoshiro256++) vs std::mt19937_64 (the std PRNG SOTA-ish
//     baseline; xoshiro256++ is the modern recommended general-purpose PRNG).
//   * time/clock    -- monotonic_ns()/realtime_ns() read latency vs std::chrono
//     steady_clock/system_clock now() (same OS primitive; measures wrapper cost).
//   * obs/log       -- the PRODUCE path: gated-out (should_log false) cost, and a
//     counting-sink emit (format + fan-out, NO stderr I/O) so the number is the
//     library overhead, not the terminal.
#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>

#include "hyle/obs/log.h"
#include "hyle/time/clock.h"
#include "hyle/util/random.h"

namespace {
namespace nb = ankerl::nanobench;

// Counting sink: records calls, does NO I/O -> isolates format+fanout overhead.
std::uint64_t g_log_calls = 0;
void counting_write(void* /*state*/, hyle::obs::LogLevel, const char*) noexcept {
  ++g_log_calls;
}
}  // namespace

int main() {
  // ---- util/random: Rng (xoshiro256++) vs std::mt19937_64 ----
  {
    nb::Bench b;
    b.title("PRNG next_u64 (1 value/op)").relative(true).batch(1).minEpochIterations(2000000);
    b.run("std::mt19937_64", [&] {
      static std::mt19937_64 rng(12345);
      nb::doNotOptimizeAway(rng());
    });
    b.run("HYLE Rng (xoshiro256++)", [&] {
      static hyle::util::Rng rng(12345);
      nb::doNotOptimizeAway(rng.next_u64());
    });
  }
  // bounded range: HYLE Lemire unbiased vs std::uniform_int_distribution
  {
    nb::Bench b;
    b.title("PRNG bounded [0,1000)").relative(true).batch(1).minEpochIterations(1000000);
    b.run("std::uniform_int_distribution", [&] {
      static std::mt19937_64 rng(7);
      static std::uniform_int_distribution<std::uint64_t> dist(0, 999);
      nb::doNotOptimizeAway(dist(rng));
    });
    b.run("HYLE Rng::range_u64 (Lemire)", [&] {
      static hyle::util::Rng rng(7);
      nb::doNotOptimizeAway(rng.range_u64(0, 999));
    });
  }

  // ---- time/clock: read latency vs std::chrono (same OS primitive) ----
  {
    nb::Bench b;
    b.title("monotonic clock read").relative(true).batch(1).minEpochIterations(500000);
    b.run("std::chrono::steady_clock::now", [&] {
      auto t = std::chrono::steady_clock::now().time_since_epoch().count();
      nb::doNotOptimizeAway(t);
    });
    b.run("HYLE monotonic_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::monotonic_ns());
    });
  }
  {
    nb::Bench b;
    b.title("realtime clock read").relative(true).batch(1).minEpochIterations(500000);
    b.run("std::chrono::system_clock::now", [&] {
      auto t = std::chrono::system_clock::now().time_since_epoch().count();
      nb::doNotOptimizeAway(t);
    });
    b.run("HYLE realtime_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::realtime_ns());
    });
  }

  // ---- obs/log: produce-path overhead (no I/O) ----
  // Remove the auto-installed stderr sink so we measure library cost, not terminal.
  hyle::obs::remove_sink(hyle::obs::default_sink());
  {
    nb::Bench b;
    b.title("log gated-out (level below threshold)").relative(true).batch(1).minEpochIterations(2000000);
    hyle::obs::set_level(hyle::obs::LogLevel::Warn);  // Info logs are dropped
    b.run("HYLE_LOG_INFO (dropped at gate)", [&] {
      int x = static_cast<int>(g_log_calls);
      HYLE_LOG_INFO("value={}", x);  // should_log false -> no format, no fanout
      nb::doNotOptimizeAway(x);
    });
  }
  {
    hyle::obs::LogSink sink{nullptr, &counting_write};
    hyle::obs::add_sink(sink);
    hyle::obs::set_level(hyle::obs::LogLevel::Info);  // Info now passes
    nb::Bench b;
    b.title("log emit: format + fanout to 1 sink (no I/O)").relative(true).batch(1).minEpochIterations(500000);
    b.run("HYLE_LOG_INFO (counting sink)", [&] {
      int x = static_cast<int>(g_log_calls & 0xff);
      HYLE_LOG_INFO("iter={} hex={:x}", x, x);  // std::format into stack buf + fanout
      nb::doNotOptimizeAway(g_log_calls);
    });
    hyle::obs::remove_sink(sink);
  }
  return 0;
}
