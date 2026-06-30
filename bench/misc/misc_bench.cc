// misc-module bench (gap-fill #37 + #47): the performance-relevant primitives that
// had no benchmark. Each races the std/obvious equivalent (win or lose, printed):
//   * util/random  -- Rng (xoshiro256++) vs std::mt19937_64 (the std PRNG SOTA-ish
//     baseline; xoshiro256++ is the modern recommended general-purpose PRNG).
//   * time/clock    -- monotonic_ns()/realtime_ns() read latency vs std::chrono
//     steady_clock/system_clock now() (same OS primitive; measures wrapper cost).
//   * obs/log       -- the PRODUCE path: gated-out (should_log false) cost, and a
//     counting-sink emit (format + fan-out, NO stderr I/O) so the number is the
//     library overhead, not the terminal.
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
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "hyle/obs/log.h"
#include "hyle/time/clock.h"
#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace {
namespace nb = ankerl::nanobench;

// Counting sink: records calls, does NO I/O -> isolates format+fanout overhead.
std::uint64_t g_log_calls = 0;
void counting_write(void* /*state*/, hyle::obs::LogLevel, const char*) noexcept {
  ++g_log_calls;
}

// Minimal RefCounted subject for the Ref vs shared_ptr bench. One int payload, so
// the only difference measured is the refcount mechanism (intrusive atomic in the
// object vs shared_ptr's separate control block).
struct Widget : hyle::util::RefCounted<Widget> {
  int v = 0;
};
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
      auto t = std::chrono::steady_clock::now().time_since_epoch().count();
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
      auto t = std::chrono::system_clock::now().time_since_epoch().count();
      nb::doNotOptimizeAway(t);
    });
    b.run("HYLE realtime_ns", [&] {
      nb::doNotOptimizeAway(hyle::time::realtime_ns());
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
