// SOTA parity for the lock-free queue cluster (task #9). Until now these four
// structures claimed to match moodycamel / tokio / oneTBB but had ZERO head-to-
// head benchmark -- "coverage too low so assume SOTA" is exactly the self-
// deception this bench kills. Whatever the numbers say, they print.
//
// HYLE baselines ALWAYS compile (no gate). SOTA references are opt-in:
//   -DHYLE_BENCH_MOODYCAMEL=ON  -> moodycamel::ConcurrentQueue (header-only)
//   -DHYLE_BENCH_TBB=ON         -> tbb::concurrent_bounded_queue + concurrent_vector
// The Rust crossbeam-deque parity (the true same-family work-stealing rival) is a
// SEPARATE crate (bench/rust/), since it cannot link into a C++ TU.
//
// FAIRNESS NOTES (read before trusting a ratio):
//  * BoundedQueue is BOUNDED (try_push fails when full). moodycamel ConcurrentQueue
//    is UNBOUNDED (grows). tbb::concurrent_bounded_queue is bounded (set_capacity)
//    -> tbb is the apples-to-apples MPMC rival; moodycamel is the "industry default"
//    reference whose unboundedness we call out, not hide.
//  * WsFifoQueue / WorkStealingDeque OWNER paths are single-owner (no CAS on the
//    hot push/pop). Racing them against a general MPMC queue is NOT "HYLE beats
//    moodycamel": it shows the cost a general MPMC pays that a per-core local
//    queue avoids. The same-family rival is crossbeam (Rust bench).

#include <nanobench.h>

#include <atomic>
#include <cstdint>
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
  {
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector single-thread push_back (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      for (int i = 0; i < kN; ++i) v.push_back(static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#if defined(HYLE_BENCH_TBB)
    b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      for (int i = 0; i < kN; ++i) v.push_back(static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#endif
  }
  {
    // Lock-free read hot path: sequential index sum over a prebuilt vector.
    ConcurrentVector<std::uint64_t> hv;
    for (int i = 0; i < kN; ++i) hv.push_back(static_cast<std::uint64_t>(i));
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector sequential read sum (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);
    b.run("HYLE ConcurrentVector", [&] {
      std::uint64_t sum = 0;
      for (int i = 0; i < kN; ++i) sum += hv[static_cast<std::size_t>(i)];
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#if defined(HYLE_BENCH_TBB)
    static tbb::concurrent_vector<std::uint64_t> tv;
    if (tv.empty())
      for (int i = 0; i < kN; ++i) tv.push_back(static_cast<std::uint64_t>(i));
    b.run("tbb concurrent_vector", [&] {
      std::uint64_t sum = 0;
      for (int i = 0; i < kN; ++i) sum += tv[static_cast<std::size_t>(i)];
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#endif
  }
}

}  // namespace

int main() {
  bench_single_thread();
  bench_mpmc<2, 2>();
  bench_mpmc<4, 4>();
  bench_ws_owner();
  bench_ws_steal();
  bench_concurrent_vector();
  return 0;
}
