#define _CRT_SECURE_NO_WARNINGS 1

// SOTA parity for the lock-free queue cluster (task #9). Until now these four
// structures claimed to match moodycamel / tokio / oneTBB but had ZERO head-to-
// head benchmark -- "coverage too low so assume SOTA" is exactly the self-
// deception this bench kills. Whatever the numbers say, they print.
//
// HYLE baselines ALWAYS compile (no gate). SOTA references are opt-in:
//   -DHYLE_BENCH_MOODYCAMEL=ON  -> moodycamel::ConcurrentQueue (header-only)
//   -DHYLE_BENCH_TBB=ON         -> tbb::concurrent_bounded_queue + concurrent_vector
//   -DHYLE_BENCH_RIGTORP=ON     -> rigtorp::mpmc::Queue (the SAME-CONTRACT rival:
//                                  strict-bounded MPMC, single ring -- the canonical
//                                  SOTA for exactly BoundedQueue's contract)
// The Rust crossbeam-deque parity (the true same-family work-stealing rival) is a
// SEPARATE crate (bench/rust/), since it cannot link into a C++ TU.
//
// FAIRNESS NOTES (read before trusting a ratio):
//  * BoundedQueue is BOUNDED + strict global FIFO (try_push fails when full). Its
//    APPLES-TO-APPLES rival is rigtorp::mpmc::Queue (also bounded single-ring MPMC)
//    -- both pay ONE enqueue/dequeue CAS serialization point, which is a THEOREM
//    WALL for strict-global-FIFO+bounded MPMC, not a HYLE defect. moodycamel is
//    UNBOUNDED + WEAK order (per-producer subqueues) -> it trades FIFO for producer-
//    side parallelism; the apples-to-apples answer to THAT trade is HYLE ShardedQueue
//    (same trade), which wins moodycamel 2.6-3x. tbb::concurrent_bounded_queue is
//    bounded too. So: vs rigtorp = "is our strict-bounded ring SOTA?"; vs moodycamel
//    single-ring = "the cost of the FIFO contract" (answered by ShardedQueue).
//  * WsFifoQueue / WorkStealingDeque OWNER paths are single-owner (no CAS on the
//    hot push/pop). Racing them against a general MPMC queue is NOT "HYLE beats
//    moodycamel": it shows the cost a general MPMC pays that a per-core local
//    queue avoids. The same-family rival is crossbeam (Rust bench).

#include <nanobench.h>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/concurrent_vector.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/work_stealing_deque.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/mem/allocator.h"

#if defined(HYLE_BENCH_MOODYCAMEL)
#include "concurrentqueue.h"
#endif
#if defined(HYLE_BENCH_TBB)
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/concurrent_vector.h"
#endif
#if defined(HYLE_BENCH_RIGTORP)
#include "rigtorp/MPMCQueue.h"  // strict-bounded-MPMC SAME-CONTRACT rival
#endif

using hyle::container::BoundedQueue;
using hyle::container::ConcurrentVector;
using hyle::container::ShardedQueue;
using hyle::container::WorkStealingDeque;
using hyle::container::WsFifoQueue;
using hyle::mem::system_allocator;

namespace {

constexpr std::size_t kCap = 1024;   // ring capacity (power of two)
constexpr int kBatch = 512;          // items per fill/drain wave (< kCap)

// ---- 1. single-thread enqueue+dequeue throughput ----------------------------
// Purest queue overhead: no contention, no thread scheduling. Fill `kBatch`
// then drain `kBatch`; one op == one push or one pop. This is the floor every
// MPMC algorithm pays even when uncontended (per-slot seq CAS for Vyukov).
void bench_single_thread() {
  ankerl::nanobench::Bench b;
  b.title("queue single-thread fill+drain (1024 cap, 512 batch)")
      .relative(true)
      .batch(static_cast<std::uint64_t>(kBatch) * 2)  // push + pop
      .minEpochIterations(2000);

  b.run("HYLE BoundedQueue", [&] {
    static BoundedQueue<std::uint64_t> q(kCap);
    for (int i = 0; i < kBatch; ++i) q.try_push(static_cast<std::uint64_t>(i));
    std::uint64_t out = 0, sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      q.try_pop(out);
      sum += out;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });

#if defined(HYLE_BENCH_MOODYCAMEL)
  b.run("moodycamel ConcurrentQueue", [&] {
    static moodycamel::ConcurrentQueue<std::uint64_t> q(kCap);
    for (int i = 0; i < kBatch; ++i) q.enqueue(static_cast<std::uint64_t>(i));
    std::uint64_t out = 0, sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      q.try_dequeue(out);
      sum += out;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });
#endif
#if defined(HYLE_BENCH_TBB)
  b.run("tbb concurrent_bounded_queue", [&] {
    static tbb::concurrent_bounded_queue<std::uint64_t> q;
    static bool init = [] { q.set_capacity(kCap); return true; }();
    (void)init;
    for (int i = 0; i < kBatch; ++i) q.try_push(static_cast<std::uint64_t>(i));
    std::uint64_t out = 0, sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      q.try_pop(out);
      sum += out;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });
#endif
#if defined(HYLE_BENCH_RIGTORP)
  b.run("rigtorp MPMCQueue (same-contract SOTA)", [&] {
    static rigtorp::mpmc::Queue<std::uint64_t> q(kCap);
    for (int i = 0; i < kBatch; ++i) q.try_push(static_cast<std::uint64_t>(i));
    std::uint64_t out = 0, sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      if (q.try_pop(out)) sum += out;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });
#endif
}

// ---- 2. MPMC throughput (P producers / C consumers, end-to-end) -------------
// Each producer pushes kPerThread disjoint ids; consumers drain until every
// producer has finished AND the queue reads empty. Spin-retry on a full bounded
// queue (the bounded queues only). batch == kTotal so the number is ns per item
// moved through the queue.
//
// Consumer-stop protocol (red-team fix): consumers count LOCALLY and only touch a
// shared atomic kProducers times total (each producer bumps `producers_done` when
// it finishes). A consumer exits when producers_done==kProducers AND its next
// try_pop fails (all produced -> empty == fully drained). The OLD design did a
// per-item fetch_add on a shared counter -- queue-EXTERNAL contention that all
// three rivals paid equally but which compressed the real algorithmic gap. This
// version keeps the hot loop free of cross-thread atomics beyond the queue itself.
template <int kProducers, int kConsumers>
void bench_mpmc() {
  constexpr int kPerThread = 50000;
  constexpr int kTotal = kProducers * kPerThread;

  ankerl::nanobench::Bench b;
  char title[96];
  std::snprintf(title, sizeof(title), "queue MPMC %dP/%dC (%d items)",
                kProducers, kConsumers, kTotal);
  // Each epoch is a full kTotal-item MPMC run (heavy); too few epochs and
  // nanobench flags it unstable. The sharded prototype is so fast its per-epoch
  // time shrinks, so it needs more iterations to stabilize -> 50.
  //
  // WHY NOT bump to nanobench's suggested ~543: at 50 iters the single ring
  // already costs ~8s/variant (200k items * ~70ns * 50); 543 would be ~90s/
  // variant -> a 15-run bench-stat sweep balloons past an hour. And it would not
  // fix the sharded variant's spread: its in-process err is only ~5%, but its
  // ACROSS-PROCESS spread is ~40% (measured, scripts/bench-stat.py -n 15). That
  // variance is thread spawn/join + OS scheduling jitter (sharded is ~14ns/op,
  // so per-process system noise dominates), NOT nanobench under-sampling -- more
  // iterations churn the same jittery spawn/join. The honest fix is the cross-
  // process median in bench-stat.py, which is what the perf doc reports; the
  // sharded win is stated as an ORDER OF MAGNITUDE (2-5x), not a point value.
  b.title(title).relative(true).batch(kTotal).minEpochIterations(50);

  b.run("HYLE BoundedQueue", [&] {
    BoundedQueue<std::uint64_t> q(kCap);
    std::atomic<int> producers_done{0};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(base + k)) std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done] {
        std::uint64_t out = 0, local = 0;
        for (;;) {
          if (q.try_pop(out)) {
            ++local;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            break;  // all produced and queue empty -> done
          } else {
            std::this_thread::yield();
          }
        }
        ankerl::nanobench::doNotOptimizeAway(local);
      });
    for (auto& t : ts) t.join();
  });

  // HYLE sharded: one shard per producer (zero inter-producer contention). Same
  // per-shard cap; drops global FIFO. Quantifies the headroom sharding unlocks
  // over the single ring under the SAME workload.
  b.run("HYLE ShardedQueue (1 shard/producer)", [&] {
    ShardedQueue<std::uint64_t> q(kProducers, kCap);
    std::atomic<int> producers_done{0};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(static_cast<std::size_t>(p), base + k))
            std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done, c] {
        std::uint64_t out = 0, local = 0;
        std::size_t cursor = static_cast<std::size_t>(c);  // spread across shards
        for (;;) {
          if (q.try_pop(cursor, out)) {
            ++local;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            break;
          } else {
            std::this_thread::yield();
          }
        }
        ankerl::nanobench::doNotOptimizeAway(local);
      });
    for (auto& t : ts) t.join();
  });

#if defined(HYLE_BENCH_MOODYCAMEL)
  b.run("moodycamel ConcurrentQueue", [&] {
    moodycamel::ConcurrentQueue<std::uint64_t> q(kCap);
    std::atomic<int> producers_done{0};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k) q.enqueue(base + k);
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done] {
        std::uint64_t out = 0, local = 0;
        for (;;) {
          if (q.try_dequeue(out)) {
            ++local;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            break;
          } else {
            std::this_thread::yield();
          }
        }
        ankerl::nanobench::doNotOptimizeAway(local);
      });
    for (auto& t : ts) t.join();
  });
#endif
#if defined(HYLE_BENCH_TBB)
  b.run("tbb concurrent_bounded_queue", [&] {
    tbb::concurrent_bounded_queue<std::uint64_t> q;
    q.set_capacity(kCap);
    std::atomic<int> producers_done{0};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(base + k)) std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done] {
        std::uint64_t out = 0, local = 0;
        for (;;) {
          if (q.try_pop(out)) {
            ++local;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            break;
          } else {
            std::this_thread::yield();
          }
        }
        ankerl::nanobench::doNotOptimizeAway(local);
      });
    for (auto& t : ts) t.join();
  });
#endif
#if defined(HYLE_BENCH_RIGTORP)
  // Same-contract rival: strict-bounded single-ring MPMC. Spin-retry on full (like
  // HYLE BoundedQueue) -> isolates the single-CAS serialization both pay at 4P.
  b.run("rigtorp MPMCQueue (same-contract SOTA)", [&] {
    rigtorp::mpmc::Queue<std::uint64_t> q(kCap);
    std::atomic<int> producers_done{0};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(base + k)) std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done] {
        std::uint64_t out = 0, local = 0;
        for (;;) {
          if (q.try_pop(out)) {
            ++local;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            break;
          } else {
            std::this_thread::yield();
          }
        }
        ankerl::nanobench::doNotOptimizeAway(local);
      });
    for (auto& t : ts) t.join();
  });
#endif
}

// ---- 3. work-stealing OWNER hot path (single-owner push+pop, no contention) -
// The whole point of a per-core local queue: the owner pushes/pops with no CAS.
// FIFO (tokio-style) and LIFO (Chase-Lev) are measured separately. The general
// MPMC queues are shown alongside ONLY to quantify the atomic the owner avoids
// (NOT a same-family comparison -- that's crossbeam in the Rust bench).
void bench_ws_owner() {
  ankerl::nanobench::Bench b;
  b.title("work-stealing OWNER fill+drain (1024 cap, 512 batch)")
      .relative(true)
      .batch(static_cast<std::uint64_t>(kBatch) * 2)
      .minEpochIterations(2000);

  b.run("HYLE WsFifoQueue (FIFO owner)", [&] {
    static WsFifoQueue<std::uint64_t> q(kCap);
    for (int i = 0; i < kBatch; ++i) q.push_back(static_cast<std::uint64_t>(i));
    std::uint64_t sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      auto r = q.pop();
      sum += r.value;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });

  b.run("HYLE WorkStealingDeque (LIFO owner)", [&] {
    static WorkStealingDeque<std::uint64_t> d(kCap);
    for (int i = 0; i < kBatch; ++i)
      d.push_bottom(static_cast<std::uint64_t>(i));
    std::uint64_t sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      auto r = d.pop_bottom();
      sum += r.value;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });

#if defined(HYLE_BENCH_MOODYCAMEL)
  b.run("moodycamel (general MPMC ref)", [&] {
    static moodycamel::ConcurrentQueue<std::uint64_t> q(kCap);
    for (int i = 0; i < kBatch; ++i) q.enqueue(static_cast<std::uint64_t>(i));
    std::uint64_t out = 0, sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      q.try_dequeue(out);
      sum += out;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });
#endif
}

// ---- 4. work-stealing STEAL throughput (thief drains victim ~half at a time) -
// Owner fills kBatch; a single thief steals everything in steal-half waves.
// Measures the steal path (the contended end). Reported as ns per item stolen.
void bench_ws_steal() {
  ankerl::nanobench::Bench b;
  b.title("work-stealing STEAL drain (owner fills 512)")
      .relative(true)
      .batch(kBatch)
      .minEpochIterations(1000);

  b.run("HYLE WsFifoQueue steal_into", [&] {
    static WsFifoQueue<std::uint64_t> victim(kCap);
    static WsFifoQueue<std::uint64_t> thief(kCap);
    for (int i = 0; i < kBatch; ++i)
      victim.push_back(static_cast<std::uint64_t>(i));
    std::uint64_t sum = 0;
    // Drain victim: each steal_into grabs ~half into `thief`, returns one.
    for (;;) {
      auto r = victim.steal_into(thief);
      if (!r.ok) break;
      sum += r.value;
      // empty the thief's haul so capacity guard never trips
      for (;;) {
        auto t = thief.pop();
        if (!t.ok) break;
        sum += t.value;
      }
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });

  b.run("HYLE WorkStealingDeque steal", [&] {
    static WorkStealingDeque<std::uint64_t> d(kCap);
    for (int i = 0; i < kBatch; ++i)
      d.push_bottom(static_cast<std::uint64_t>(i));
    std::uint64_t sum = 0;
    for (int i = 0; i < kBatch; ++i) {
      auto r = d.steal();
      if (r.status == hyle::container::StealStatus::kOk) sum += r.value;
    }
    ankerl::nanobench::doNotOptimizeAway(sum);
  });
}

// ---- 5. ConcurrentVector: push_back + indexed read (vs tbb) ------------------
void bench_concurrent_vector() {
  constexpr int kN = 100000;
  const char* variant = std::getenv("HYLE_QUEUE_BENCH_VARIANT");
  const bool run_hyle =
      variant == nullptr || std::string_view(variant) == "hyle";
  const bool run_tbb =
      variant == nullptr || std::string_view(variant) == "tbb";
  const char* selected_case = std::getenv("HYLE_QUEUE_BENCH_CV_CASE");
  const auto run_case = [selected_case](std::string_view name) {
    return selected_case == nullptr || std::string_view(selected_case) == name;
  };
  if (run_case("push")) {
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector single-thread push_back (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      for (int i = 0; i < kN; ++i) v.push_back(static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#if defined(HYLE_BENCH_TBB)
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      for (int i = 0; i < kN; ++i) v.push_back(static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#endif
  }
  if (run_case("writers")) {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 25000;
    constexpr int kTotal = kThreads * kPerThread;
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector 8-writer reserved append (200k)")
        .relative(true)
        .batch(kTotal)
        .minEpochIterations(10);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      v.reserve(kTotal);
      std::atomic<bool> go{false};
      std::vector<std::thread> writers;
      writers.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&, t] {
          while (!go.load(std::memory_order_acquire)) {}
          for (int i = 0; i < kPerThread; ++i)
            v.emplace_back(static_cast<std::uint64_t>(t) * kPerThread + i);
        });
      }
      go.store(true, std::memory_order_release);
      for (auto& writer : writers) writer.join();
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#if defined(HYLE_BENCH_TBB)
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      v.reserve(kTotal);
      std::atomic<bool> go{false};
      std::vector<std::thread> writers;
      writers.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&, t] {
          while (!go.load(std::memory_order_acquire)) {}
          for (int i = 0; i < kPerThread; ++i)
            v.emplace_back(static_cast<std::uint64_t>(t) * kPerThread + i);
        });
      }
      go.store(true, std::memory_order_release);
      for (auto& writer : writers) writer.join();
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#endif
  }
  if (run_case("index")) {
    // Lock-free read hot path: sequential index sum over a prebuilt vector.
    ConcurrentVector<std::uint64_t> hv;
    for (int i = 0; i < kN; ++i) hv.push_back(static_cast<std::uint64_t>(i));
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector sequential read sum (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      std::uint64_t sum = 0;
      for (int i = 0; i < kN; ++i) sum += hv[static_cast<std::size_t>(i)];
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#if defined(HYLE_BENCH_TBB)
    static tbb::concurrent_vector<std::uint64_t> tv;
    if (tv.empty())
      for (int i = 0; i < kN; ++i) tv.push_back(static_cast<std::uint64_t>(i));
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      std::uint64_t sum = 0;
      for (int i = 0; i < kN; ++i) sum += tv[static_cast<std::size_t>(i)];
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#endif
  }
  if (run_case("iterate")) {
    ConcurrentVector<std::uint64_t> hv;
    for (int i = 0; i < kN; ++i) hv.push_back(static_cast<std::uint64_t>(i));
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector range iteration sum (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      std::uint64_t sum = 0;
      for (std::uint64_t value : hv) sum += value;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#if defined(HYLE_BENCH_TBB)
    static tbb::concurrent_vector<std::uint64_t> tv;
    if (tv.empty())
      for (int i = 0; i < kN; ++i) tv.push_back(static_cast<std::uint64_t>(i));
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      std::uint64_t sum = 0;
      for (std::uint64_t value : tv) sum += value;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#endif
  }

  // ---- CONCURRENT lock-free read WHILE a writer appends (the headline) -------
  // The single-thread reads above don't exercise ConcurrentVector's actual claim:
  // "insert once, read many, FROM MANY THREADS" with LOCK-FREE readers. Here R
  // reader threads sum a FULLY-BUILT prefix [0,kPre) in a loop while ONE writer
  // appends another kPre items concurrently (forcing segment growth). Readers only
  // touch the pre-built prefix, so neither HYLE nor tbb can read a half-constructed
  // tail (fair). This stresses stable segmented indexing while the reservation
  // counter and cold growth path are active on another thread. Win or lose,
  // printed. batch =
  // R*kPre*kRounds (total reads); the number is ns per lock-free read under writer
  // contention. minEpochIterations low (each epoch spawns+joins R+1 threads).
  if (run_case("read-write")) {
    constexpr int kPre = 100000;     // pre-built, fully-constructed prefix
    constexpr int kReaders = 4;      // concurrent lock-free readers
    constexpr int kRounds = 8;       // full-prefix sweeps per reader thread
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector 4 readers (lock-free) + 1 writer append")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kReaders) * kPre * kRounds)
        .minEpochIterations(10);

    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      for (int i = 0; i < kPre; ++i) v.push_back(static_cast<std::uint64_t>(i));
      std::atomic<bool> go{false};
      std::vector<std::thread> ts;
      std::atomic<std::uint64_t> sink{0};
      for (int r = 0; r < kReaders; ++r)
        ts.emplace_back([&] {
          while (!go.load(std::memory_order_acquire)) {}
          std::uint64_t s = 0;
          for (int round = 0; round < kRounds; ++round)
            for (int i = 0; i < kPre; ++i) s += v[static_cast<std::size_t>(i)];
          sink.fetch_add(s, std::memory_order_relaxed);
        });
      std::thread writer([&] {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < kPre; ++i)
          v.push_back(static_cast<std::uint64_t>(kPre + i));  // forces growth
      });
      go.store(true, std::memory_order_release);
      for (auto& t : ts) t.join();
      writer.join();
      ankerl::nanobench::doNotOptimizeAway(sink.load());
    });
#if defined(HYLE_BENCH_TBB)
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      for (int i = 0; i < kPre; ++i) v.push_back(static_cast<std::uint64_t>(i));
      std::atomic<bool> go{false};
      std::vector<std::thread> ts;
      std::atomic<std::uint64_t> sink{0};
      for (int r = 0; r < kReaders; ++r)
        ts.emplace_back([&] {
          while (!go.load(std::memory_order_acquire)) {}
          std::uint64_t s = 0;
          for (int round = 0; round < kRounds; ++round)
            for (int i = 0; i < kPre; ++i) s += v[static_cast<std::size_t>(i)];
          sink.fetch_add(s, std::memory_order_relaxed);
        });
      std::thread writer([&] {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < kPre; ++i)
          v.push_back(static_cast<std::uint64_t>(kPre + i));
      });
      go.store(true, std::memory_order_release);
      for (auto& t : ts) t.join();
      writer.join();
      ankerl::nanobench::doNotOptimizeAway(sink.load());
    });
#endif
  }
}

}  // namespace

int main() {
  const char* filter = std::getenv("HYLE_QUEUE_BENCH_FILTER");
  if (filter != nullptr) {
    if (std::string_view(filter) == "concurrent-vector") {
      bench_concurrent_vector();
      return 0;
    }
    return 2;
  }
  bench_single_thread();
  bench_mpmc<2, 2>();
  bench_mpmc<4, 4>();
  bench_ws_owner();
  bench_ws_steal();
  bench_concurrent_vector();
  return 0;
}
