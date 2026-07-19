// misc-module bench (gap-fill #37 + #47): the performance-relevant primitives that
// had no benchmark. Each races the std/obvious equivalent (win or lose, printed):
//   * util/random  -- Rng (xoshiro256++) vs std::mt19937_64 (the std PRNG SOTA-ish
//     baseline; xoshiro256++ is the modern recommended general-purpose PRNG).
//   * time/clock    -- monotonic_ns()/realtime_ns() read latency vs std::chrono
//     steady_clock/system_clock now() (same OS primitive; measures wrapper cost).
//   * obs/log       -- gated-out cost, safe preformatted fan-out through both
//     native C++ and stable C ABI, plus caller formatting + fan-out. All use
//     counting sinks (NO stderr I/O).
//   * util/spin_lock -- SpinLock lock/unlock uncontended vs std::mutex (the TTAS
//     spinlock claim; for short critical sections a spinlock should beat a mutex's
//     syscall-capable lock).
//   * util/ref_counted -- Ref copy+destroy (intrusive incref/decref) vs
//     std::shared_ptr copy+destroy (the "cheaper than shared_ptr, no control
//     block" claim).
#include <nanobench.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hyle/c/util.h"
#include "hyle/obs/log.h"
#include "hyle/time/clock.h"
#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace {
namespace nb = ankerl::nanobench;

// Counting sink: records calls, does NO I/O -> isolates format+fanout overhead.
std::uint64_t g_log_calls = 0;
void counting_write(void* /*state*/,
                    const hyle::obs::LogRecord& /*record*/) noexcept {
  ++g_log_calls;
}
void c_counting_write(void* /*state*/,
                      const hyle_obs_log_record* /*record*/) {
  ++g_log_calls;
}

// Minimal RefCounted subject for the Ref vs shared_ptr bench. One int payload, so
// the only difference measured is the refcount mechanism (intrusive atomic in the
// object vs shared_ptr's separate control block).
struct Widget : hyle::util::RefCounted<Widget> {
  int v = 0;
};

template <class Lock>
std::uint64_t contended_increment(std::size_t threads,
                                  std::size_t iterations) {
  Lock lock;
  std::uint64_t value = 0;
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (std::size_t thread = 0; thread < threads; ++thread) {
    workers.emplace_back([&] {
      for (std::size_t i = 0; i < iterations; ++i) {
        std::lock_guard<Lock> guard(lock);
        ++value;
      }
    });
  }
  for (auto& worker : workers) worker.join();
  return value;
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
  // A C/FFI scalar call crosses the stable DLL seam; batch fill below shows how
  // callers amortize that tax when producing more than one value.
  {
    hyle_util_rng c_rng{};
    if (!hyle_util_rng_seed(&c_rng, 12345)) return 1;
    nb::Bench b;
    b.title("PRNG C ABI scalar next_u64")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    b.run("C++ Rng::next_u64", [&] {
      static hyle::util::Rng rng(12345);
      nb::doNotOptimizeAway(rng.next_u64());
    });
    b.run("C ABI hyle_util_rng_next_u64", [&] {
      nb::doNotOptimizeAway(hyle_util_rng_next_u64(&c_rng));
    });
  }
  {
    constexpr std::size_t kBatch = 256;
    std::uint64_t values[kBatch];
    hyle_util_rng c_rng{};
    if (!hyle_util_rng_seed(&c_rng, 12345)) return 1;
    nb::Bench b;
    b.title("PRNG fill 256 u64")
        .relative(true)
        .batch(kBatch)
        .minEpochIterations(5000);
    b.run("C++ inline loop", [&] {
      static hyle::util::Rng rng(12345);
      for (std::uint64_t& value : values) value = rng.next_u64();
      nb::doNotOptimizeAway(values);
    });
    b.run("C ABI fill_u64", [&] {
      if (!hyle_util_rng_fill_u64(&c_rng, values, kBatch)) std::abort();
      nb::doNotOptimizeAway(values);
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
  // Runtime-varying bounds prevent constant-divisor folding from hiding the
  // actual bounded-generation path.
  {
    std::uint64_t bounds[256];
    for (std::size_t i = 0; i < 256; ++i) bounds[i] = 17 + i * 7919;
    std::size_t index = 0;
    nb::Bench b;
    b.title("PRNG bounded runtime-varying upper bound")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    b.run("std::uniform_int_distribution", [&] {
      static std::mt19937_64 rng(7);
      static std::uniform_int_distribution<std::uint64_t> dist;
      const std::uint64_t hi = bounds[index++ & 255];
      nb::doNotOptimizeAway(
          dist(rng, decltype(dist)::param_type(0, hi)));
    });
    index = 0;
    b.run("HYLE Rng::range_u64", [&] {
      static hyle::util::Rng rng(7);
      const std::uint64_t hi = bounds[index++ & 255];
      nb::doNotOptimizeAway(rng.range_u64(0, hi));
    });
  }
  {
    nb::Bench b;
    b.title("PRNG bounded u32 [0,1000)")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    b.run("std::uniform_int_distribution<u32>", [&] {
      static std::mt19937 rng(7);
      static std::uniform_int_distribution<std::uint32_t> dist(0, 999);
      nb::doNotOptimizeAway(dist(rng));
    });
    b.run("HYLE Rng::range_u32", [&] {
      static hyle::util::Rng rng(7);
      nb::doNotOptimizeAway(rng.range_u32(0, 999));
    });
  }
  // next_double [0,1): HYLE 53-bit scale vs std::generate_canonical / uniform_real
  {
    nb::Bench b;
    b.title("PRNG next_double [0,1)").relative(true).batch(1).minEpochIterations(1000000);
    b.run("std::uniform_real_distribution<double>", [&] {
      static std::mt19937_64 rng(7);
      static std::uniform_real_distribution<double> dist(0.0, 1.0);
      nb::doNotOptimizeAway(dist(rng));
    });
    b.run("HYLE Rng::next_double", [&] {
      static hyle::util::Rng rng(7);
      nb::doNotOptimizeAway(rng.next_double());
    });
  }
  // next_float [0,1): distinct 24-bit scale path (not a forward to next_double).
  {
    nb::Bench b;
    b.title("PRNG next_float [0,1)").relative(true).batch(1).minEpochIterations(1000000);
    b.run("std::uniform_real_distribution<float>", [&] {
      static std::mt19937_64 rng(7);
      static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      nb::doNotOptimizeAway(dist(rng));
    });
    b.run("HYLE Rng::next_float", [&] {
      static hyle::util::Rng rng(7);
      nb::doNotOptimizeAway(rng.next_float());
    });
  }
  // Fisher-Yates shuffle of 1000 ints: HYLE Rng::shuffle (swap callback) vs
  // std::shuffle(mt19937_64). Both are n-1 bounded draws + swaps; the rival is the
  // std shuffle algorithm over the std PRNG (the canonical "shuffle a range").
  {
    constexpr int kN = 1000;
    std::vector<int> v(kN);
    for (int i = 0; i < kN; ++i) v[i] = i;
    nb::Bench b;
    b.title("shuffle 1000 ints").relative(true).batch(kN).minEpochIterations(2000);
    b.run("std::shuffle(mt19937_64)", [&] {
      static std::mt19937_64 rng(7);
      std::shuffle(v.begin(), v.end(), rng);
      nb::doNotOptimizeAway(v.data());
    });
    b.run("HYLE Rng::shuffle (swap callback)", [&] {
      static hyle::util::Rng rng(7);
      rng.shuffle(kN, [&](std::uint64_t i, std::uint64_t j) {
        std::swap(v[i], v[j]);
      });
      nb::doNotOptimizeAway(v.data());
    });
  }
  {
    nb::Bench b;
    b.title("monotonic clock read").relative(true).batch(1).minEpochIterations(500000);
    b.run("std::chrono::steady_clock::now", [&] {
      const auto value =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      const std::uint64_t t =
          value > 0 ? static_cast<std::uint64_t>(value) : 0;
      nb::doNotOptimizeAway(t);
    });
    b.run("HYLE monotonic_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::monotonic_ns());
    });
    // Opt-in TSC fast path: should be ~2x the QPC/clock_gettime read (skips the
    // OS call shell). Falls back to monotonic_ns when no invariant TSC, so on
    // such hardware this row reads the same as the one above.
    b.run("HYLE fast_monotonic_ns (rdtsc)", [&] {
      nb::doNotOptimizeAway(hyle::time::fast_monotonic_ns());
    });
  }
  {
    nb::Bench b;
    b.title("realtime clock read").relative(true).batch(1).minEpochIterations(500000);
    b.run("std::chrono::system_clock::now", [&] {
      // Match HYLE's published job exactly: Unix-epoch nanoseconds as uint64,
      // including the 100ns->ns conversion MSVC's system_clock otherwise omits.
      const auto value =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      const std::uint64_t t =
          value > 0 ? static_cast<std::uint64_t>(value) : 0;
      nb::doNotOptimizeAway(t);
    });
    b.run("HYLE realtime_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::realtime_ns());
    });
    b.run("HYLE fast_realtime_ns (coarse)", [&] {
      nb::doNotOptimizeAway(hyle::time::fast_realtime_ns());
    });
  }

  // ---- util/spin_lock: uncontended lock/unlock vs std::mutex ----
  // Single-thread lock+unlock cycle (no contention). SpinLock is a TTAS atomic
  // exchange + release store -- a handful of instructions, no syscall path. A
  // std::mutex uncontended lock is also fast on modern libc (futex/SRWLOCK fast
  // path) but carries more machinery. For the very short critical sections
  // SpinLock targets, it should win uncontended.
  {
    nb::Bench b;
    b.title("uncontended lock+unlock").relative(true).batch(1).minEpochIterations(2000000);
    b.run("std::mutex", [&] {
      static std::mutex m;
      m.lock();
      nb::doNotOptimizeAway(&m);
      m.unlock();
    });
    b.run("HYLE SpinLock (TTAS)", [&] {
      static hyle::util::SpinLock s;
      s.lock();
      nb::doNotOptimizeAway(&s);
      s.unlock();
    });
  }
  {
    constexpr std::size_t kIterations = 100000;
    for (std::size_t threads : {std::size_t{2}, std::size_t{8}}) {
      nb::Bench b;
      b.title("contended increment, " + std::to_string(threads) + " threads")
          .relative(true)
          .batch(threads * kIterations)
          .minEpochIterations(1)
          .epochs(7);
      b.run("std::mutex", [&] {
        nb::doNotOptimizeAway(
            contended_increment<std::mutex>(threads, kIterations));
      });
      b.run("HYLE SpinLock", [&] {
        nb::doNotOptimizeAway(
            contended_increment<hyle::util::SpinLock>(threads, kIterations));
      });
    }
  }

  // ---- util/ref_counted: Ref vs std::shared_ptr ----
  // TWO patterns, because they stress different things (see the asm analysis in
  // the commit): both smart pointers emit EXACTLY 2 locked RMWs per copy+destroy
  // (incref + decref), so the algorithmic work is identical.
  //  (a) copy+destroy cycle: the two atomics land on the SAME cache line back-to-
  //      back; Ref does them adjacently (2nd serializes behind the 1st draining
  //      the store buffer) while shared_ptr happens to interleave its get()/_Rep
  //      reload between them -- a scheduling quirk, not an algorithmic gap.
  //  (b) fan-out then release: build a vector of N copies (N increfs) then clear
  //      (N decrefs). Incref burst and decref burst are separated, so the back-to-
  //      back serialization doesn't dominate -- the realistic "pass refs around"
  //      pattern, where Ref's leanness (1 pointer, in-object counter, no control
  //      block, no _Weaks) shows.
  {
    hyle::util::Ref<Widget> owner = hyle::util::make_ref<Widget>();
    std::shared_ptr<Widget> sp = std::make_shared<Widget>();
    nb::Bench b;
    b.title("smart-ptr copy+destroy (back-to-back same-line atomics)").relative(true).batch(1).minEpochIterations(2000000);
    b.run("std::shared_ptr", [&] {
      std::shared_ptr<Widget> c = sp;  // incref (control block)
      nb::doNotOptimizeAway(c.get());
    });                                 // decref on scope exit
    b.run("HYLE Ref (intrusive)", [&] {
      hyle::util::Ref<Widget> c = owner;  // incref (in-object atomic)
      nb::doNotOptimizeAway(c.get());
    });                                   // decref on scope exit
  }
  {
    constexpr int kFan = 64;  // refs held simultaneously (separates incref/decref bursts)
    hyle::util::Ref<Widget> owner = hyle::util::make_ref<Widget>();
    std::shared_ptr<Widget> sp = std::make_shared<Widget>();
    std::vector<std::shared_ptr<Widget>> spv;
    std::vector<hyle::util::Ref<Widget>> rv;
    spv.reserve(kFan);
    rv.reserve(kFan);
    nb::Bench b;
    b.title("smart-ptr fan-out 64 copies then release").relative(true).batch(kFan).minEpochIterations(20000);
    b.run("std::shared_ptr", [&] {
      for (int i = 0; i < kFan; ++i) spv.push_back(sp);  // 64 increfs
      nb::doNotOptimizeAway(spv.data());
      spv.clear();                                       // 64 decrefs
    });
    b.run("HYLE Ref (intrusive)", [&] {
      for (int i = 0; i < kFan; ++i) rv.push_back(owner);  // 64 increfs
      nb::doNotOptimizeAway(rv.data());
      rv.clear();                                          // 64 decrefs
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
    hyle::obs::set_level(hyle::obs::LogLevel::Info);
    nb::Bench b;
    b.title("log preformatted emit to 1 sink (no I/O)")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    b.run("C++ log_message", [&] {
      nb::doNotOptimizeAway(
          hyle::obs::log_message(hyle::obs::LogLevel::Info, "ready"));
    });
    hyle::obs::remove_sink(sink);

    hyle_obs_sink* c_sink = hyle_obs_sink_create(&c_counting_write, nullptr);
    if (c_sink == nullptr) return 1;
    b.run("C ABI hyle_obs_log", [&] {
      nb::doNotOptimizeAway(hyle_obs_log(HYLE_OBS_LOG_INFO, "ready"));
    });
    if (!hyle_obs_sink_destroy(c_sink)) return 1;
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
