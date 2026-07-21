// HYLE specialized-allocator cluster vs SOTA. performance.md:127 claims arena/bump
// allocators are "5-10x+ vs new/delete" -- never benchmarked until now. The honest
// SOTA references are in the STANDARD LIBRARY (no external clone):
//   * std::pmr::monotonic_buffer_resource   -- std's arena (bump, bulk release)
//   * std::pmr::unsynchronized_pool_resource -- std's pool (size-class free-list)
// plus raw mi_malloc/mi_free (the general malloc = what `new/delete` resolve to,
// since HYLE's global backing is mimalloc) as the "5-10x" target baseline.
//
// Workload families (fair contracts only — no "wrong tool" rows in SOTA tables):
//  A. BUMP: Arena + StackAllocator vs pmr::monotonic vs raw mi.
//  B. SINGLE-THREAD pool churn: GrowablePool ONLY vs pmr::unsynchronized_pool vs
//     raw mi. MemoryArena/ObjectPool are multi-thread shared tools — they must NOT
//     appear here (no fair ST baseline; atomic CAS tax is misuse, not a rival).
//  C. 8-thread: GrowablePool-per-thread vs mi; MemoryArena shared vs pmr sync +
//     optional oneTBB scalable (gated HYLE_BENCH_TBB).

#include <mimalloc.h>
#include <nanobench.h>

#if defined(HYLE_BENCH_TBB)
#  include <tbb/scalable_allocator.h>  // scalable_malloc/free: tbb's MT-scaling allocator
#endif

#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory_resource>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/memory_arena.h"
#include "hyle/mem/stack_allocator.h"

using hyle::mem::Allocator;
using hyle::mem::Arena;
using hyle::mem::GrowablePool;
using hyle::mem::MemoryArena;
using hyle::mem::StackAllocator;
using hyle::mem::system_allocator;

namespace {
constexpr std::size_t kObj = 64;       // one small same-size object
constexpr int kN = 8192;               // allocations per round
constexpr std::size_t kRegion = kN * kObj + 4096;  // bump region (+slack)
namespace nb = ankerl::nanobench;

// Section filter: HYLE_BENCH_SECTIONS="D" runs only section D, "AD" runs A+D, etc.
// Unset/empty runs everything. The full sweep is minutes-per-run (the bump/ST epochs
// dominate), so a bench-stat -n sweep over just the shared-path section D is orders
// of magnitude faster; this gate makes that possible without touching the rows.
bool want_section(char id) {
  const char* env = std::getenv("HYLE_BENCH_SECTIONS");
  if (env == nullptr || env[0] == '\0') return true;  // no filter -> all
  for (const char* p = env; *p; ++p)
    if (*p == id) return true;
  return false;
}
}  // namespace

int main() {
  Allocator sys = system_allocator();  // mimalloc backing

  // ---- A. BUMP: alloc N small, no per-object free, bulk release ----
  if (want_section('A')) {
    nb::Bench b;
    b.title("bump: 8192x64B alloc + bulk release")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);

    // HYLE Arena (borrows ONE pooled block; reset() reuses it per round).
    MemoryArena pool(sys, kRegion, 1);  // 1 block of kRegion bytes
    Arena arena(&pool);
    b.run("HYLE Arena (bump, pooled block)", [&] {
      arena.reset();
      for (int i = 0; i < kN; ++i) {
        void* p = arena.allocate(kObj);
        nb::doNotOptimizeAway(p);
      }
    });

    // HYLE StackAllocator (owns its slab; reset() per round).
    StackAllocator stack(sys, kRegion);
    b.run("HYLE StackAllocator (bump, own slab)", [&] {
      stack.reset();
      for (int i = 0; i < kN; ++i) {
        void* p = stack.allocate(kObj);
        nb::doNotOptimizeAway(p);
      }
    });

    // std::pmr::monotonic_buffer_resource (SOTA: std's arena).
    std::vector<std::byte> buf(kRegion);
    b.run("std::pmr::monotonic_buffer_resource (SOTA)", [&] {
      std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size());
      for (int i = 0; i < kN; ++i) {
        void* p = mbr.allocate(kObj, alignof(std::max_align_t));
        nb::doNotOptimizeAway(p);
      }
      // dtor releases all (no per-object free) -- the bump lifecycle.
    });

    // raw mi_malloc: general malloc has NO bulk free -> alloc-all then free-all.
    b.run("raw mi_malloc/free (general, no bulk)", [&] {
      std::vector<void*> ps(kN);
      for (int i = 0; i < kN; ++i) ps[i] = mi_malloc(kObj);
      for (int i = 0; i < kN; ++i) mi_free(ps[i]);
      nb::doNotOptimizeAway(ps.data());
    });
  }

  // ---- A2. StackAllocator mark/rewind: nested-scope LIFO release ---- (gated 'A')
  // mark()/rewind() is the StackAllocator capability reset() can't express: save a
  // checkpoint, allocate a temporary batch, roll back to the checkpoint (freeing
  // exactly that batch) while KEEPING everything before it. The scratch/frame
  // pattern (e.g. a taskgraph pass that allocates per-node temporaries and rewinds
  // between nodes). Each round: alloc a base batch once, then inner loop {mark;
  // alloc temps; rewind}. No std rival -- std::pmr::monotonic has no partial
  // rewind (only whole-buffer release) -- so reset()-per-round is the baseline the
  // mark/rewind cost is read against (both are pointer moves; this proves rewind
  // is as cheap as a fresh bump, not a free-list walk).
  if (want_section('A')) {
    constexpr int kBase = 256;     // persistent allocations kept across rewinds
    constexpr int kInner = 64;     // temporaries per rewind cycle
    constexpr int kCycles = 128;   // rewind cycles per round
    StackAllocator stack(sys, kRegion);
    nb::Bench b;
    b.title("StackAllocator mark/rewind: 128 cycles x 64 temps")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kCycles) * kInner)
        .minEpochIterations(200);
    b.run("HYLE StackAllocator mark/rewind (LIFO scratch)", [&] {
      stack.reset();
      for (int i = 0; i < kBase; ++i) nb::doNotOptimizeAway(stack.allocate(kObj));
      for (int c = 0; c < kCycles; ++c) {
        auto m = stack.mark();
        for (int i = 0; i < kInner; ++i) nb::doNotOptimizeAway(stack.allocate(kObj));
        stack.rewind(m);  // free exactly this cycle's temps, keep the base batch
      }
    });
    // Baseline: a full reset() can't keep the base batch, but as a pure pointer-
    // move reference it shows rewind is the same O(1) cost (re-bump the temps each
    // cycle from a full reset of just the temp region).
    StackAllocator stack2(sys, kRegion);
    b.run("HYLE StackAllocator reset re-bump (baseline)", [&] {
      for (int c = 0; c < kCycles; ++c) {
        stack2.reset();
        for (int i = 0; i < kInner; ++i) nb::doNotOptimizeAway(stack2.allocate(kObj));
      }
    });
  }

  // ---- B. SINGLE-THREAD pool churn (fair tools only) ----
  // GrowablePool is the ST same-size free-list tool. Rivals: pmr unsync pool (std
  // ST pool SOTA) + raw mi (general baseline). MemoryArena/ObjectPool intentionally
  // omitted: they pay atomics for multi-thread sharing — putting them here invents
  // a "loss" with no fair ST SOTA denominator (see multi-thread section C).
  if (want_section('B')) {
    nb::Bench b;
    b.title("single-thread pool churn: 8192x alloc/free 64B")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);

    struct Obj {
      std::byte b[kObj];
    };
    GrowablePool<Obj> gpool(sys, 64);
    b.run("HYLE GrowablePool (single-thread)", [&] {
      for (int i = 0; i < kN; ++i) {
        Obj* p = gpool.create();
        nb::doNotOptimizeAway(p);
        gpool.destroy(p);
      }
    });

    b.run("std::pmr::unsynchronized_pool_resource (SOTA)", [&] {
      std::pmr::unsynchronized_pool_resource upr;
      for (int i = 0; i < kN; ++i) {
        void* p = upr.allocate(kObj, alignof(std::max_align_t));
        nb::doNotOptimizeAway(p);
        upr.deallocate(p, kObj, alignof(std::max_align_t));
      }
    });

    b.run("raw mi_malloc/free (general)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = mi_malloc(kObj);
        nb::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
  }

  // ---- C. CONCURRENT pool churn: APPLES-TO-APPLES, not apples-to-oranges ----
  // The headline "8-thread mimalloc beats shared MemoryArena 1.95x" is NOT a fair
  // fight, and saying otherwise would be a self-deception. mimalloc's 8-thread
  // number is EIGHT PER-THREAD HEAPS: every alloc/free here is SAME-THREAD, which
  // mimalloc serves with a NON-ATOMIC thread-local free list. A single SHARED
  // fixed-slab pool can't do that -- coordinating one slab across threads costs an
  // atomic CAS per op (even sharded + uncontended, the `lock` prefix isn't free).
  // So the rows below separate the two honest questions:
  //
  //   (1) "I want per-thread same-size churn" -> the SOTA pattern is a pool PER
  //       THREAD (that IS what mimalloc does internally). HYLE's matching answer
  //       is GrowablePool-per-thread: no atomics, no sharing -> parity/beat mi.
  //   (2) "I genuinely need ONE shared fixed pool" (a bounded resource every
  //       thread draws from) -> MemoryArena. The APPLES-TO-APPLES rival is std::
  //       pmr::synchronized_pool_resource (the std thread-safe SHARED pool, same
  //       contract), NOT per-thread-heap mimalloc. HYLE's lock-free sharded pool
  //       vs std's mutex-serialized pool -> THAT is the honest ledger row. The
  //       mimalloc per-thread-heap number is kept only as context (a different
  //       contract), explicitly NOT the win/loss denominator.
  // (std::pmr::unsynchronized_pool_resource is NOT thread-safe -> not a rival;
  //  synchronized_pool_resource IS, and is the fair shared-pool SOTA below.)
  if (want_section('C')) {
    constexpr int kThreads = 8;
    constexpr int kPer = 50000;
    auto run_threads = [](auto&& body) {
      std::vector<std::thread> ts;
      ts.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) ts.emplace_back(body);
      for (auto& th : ts) th.join();
    };
    struct Obj { std::byte b[kObj]; };
    nb::Bench b;
    b.title("8-thread pool churn: per-thread (fair) vs shared (coordination cost)")
        .relative(true)
        .batch(kThreads * kPer)
        .minEpochIterations(10);

    // (1) APPLES-TO-APPLES: a GrowablePool PER THREAD (no sharing, no atomics) --
    // HYLE's SOTA-parity answer for per-thread same-size churn, the SAME shape as
    // mimalloc's per-thread heaps.
    b.run("HYLE GrowablePool (8 threads, 1 pool EACH = fair vs mi)", [&] {
      run_threads([&] {
        GrowablePool<Obj> local(sys, 256);
        for (int i = 0; i < kPer; ++i) {
          Obj* p = local.create();
          nb::doNotOptimizeAway(p);
          local.destroy(p);
        }
      });
    });

    // (2) ONE MemoryArena shared by all 8 threads. NOTE (2026-07-21 audit): this
    // tight alloc->dealloc shape is a SAME-BLOCK TLS ping-pong -- after the first
    // shard pop, every op hits this thread's TLS freelist (depth 1) and NEVER
    // touches the shared shards again. It measures the TLS fast-path constant only,
    // NOT the shared-backend coordination it claims to. The honest shared-path
    // stress (TLS overflow-flush + shard refill, and real cross-thread routing) is
    // section D below; keep this row as the steady-state hot-path number.
    MemoryArena shared(sys, kObj, kThreads * 256);  // small pool -> real reuse churn
    b.run("HYLE MemoryArena (8 threads, SHARED, TLS ping-pong hot path)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = shared.allocate();
          nb::doNotOptimizeAway(p);
          shared.deallocate(p);
        }
      });
    });

    // (2-FAIR) The APPLES-TO-APPLES rival for row (2): std::pmr::synchronized_
    // pool_resource is the STANDARD LIBRARY's thread-safe SHARED pool -- same
    // contract as MemoryArena (one pool every thread draws from), so this is the
    // honest "is our shared pool SOTA for its class?" check (vs the apples-to-
    // oranges per-thread-heap mimalloc baseline). std's shared pool serializes on
    // an internal mutex; HYLE's is lock-free sharded -> HYLE should win here, and
    // THIS comparison (not the mimalloc one) is what belongs in the ledger.
    b.run("std::pmr::synchronized_pool_resource (8 threads, SHARED SOTA)", [&] {
      std::pmr::synchronized_pool_resource spr;
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = spr.allocate(kObj, alignof(std::max_align_t));
          nb::doNotOptimizeAway(p);
          spr.deallocate(p, kObj, alignof(std::max_align_t));
        }
      });
    });

    // (2-STRONG) The STRONG same-contract rival: oneTBB's scalable_allocator is a
    // thread-safe SHARED allocator engineered for MT scaling (per-thread caches +
    // shared backing), unlike std's mutex-serialized synchronized_pool. This is the
    // honest "is our shared pool SOTA vs a REAL scalable allocator, not just the weak
    // std one?" check -- synchronized_pool (44x) flatters us because std's impl is a
    // single mutex; tbb is what a production shared allocator actually costs. Gated
    // on -DHYLE_BENCH_TBB (needs the cloned oneTBB + tbbmalloc).
#if defined(HYLE_BENCH_TBB)
    b.run("oneTBB scalable_allocator (8 threads, SHARED SOTA)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = scalable_malloc(kObj);
          nb::doNotOptimizeAway(p);
          scalable_free(p);
        }
      });
    });
#endif

    // raw mi_malloc/free, 8 threads -- mimalloc's per-thread heaps (non-atomic
    // same-thread free). The fair rival to row (1), the apples-to-oranges baseline
    // for row (2).
    b.run("raw mi_malloc/free (8 threads, per-thread heaps)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = mi_malloc(kObj);
          nb::doNotOptimizeAway(p);
          mi_free(p);
        }
      });
    });
  }

  // ---- D. SHARED-PATH stress: the shapes section C's ping-pong CANNOT reach ----
  // Audit (2026-07-21): section C row (2) is a depth-1 TLS ping-pong -- it never
  // spills to the shared shards, so it does NOT measure the lock-free backend that
  // is MemoryArena's reason to exist. These two rows exercise the real machinery a
  // shared bounded pool is chosen for, against the same-contract SOTA rivals.
  if (want_section('D')) {
    constexpr int kThreads = 8;
    struct Obj { std::byte b[kObj]; };
    auto run_threads = [](auto&& body) {
      std::vector<std::thread> ts;
      ts.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) ts.emplace_back(body, t);
      for (auto& th : ts) th.join();
    };

    // ---- D1. BURST churn: working set > TLS cap (32) ----
    // Each round holds kHold live blocks then frees them all. kHold=64 > kTlsCacheCap
    // (32) so every round forces the TLS cache to OVERFLOW-FLUSH half to the shards on
    // the way up and REFILL from the shards on the way down -- i.e. it actually drives
    // the Treiber-stack push/pop path, not just the owner-only TLS stack. This is the
    // pooled-fixed-object working-set pattern (e.g. N in-flight fiber stacks / pooled
    // nodes), the realistic shape section C skips.
    constexpr int kHold = 64;         // > kTlsCacheCap (32): forces flush + refill
    constexpr int kRounds = 500;
    MemoryArena burst(sys, kObj, kThreads * (kHold + 16));  // room for all threads
    {
      nb::Bench b;
      b.title("8-thread BURST churn: hold 64 (>TLS cap) then free all")
          .relative(true)
          .batch(static_cast<std::uint64_t>(kThreads) * kRounds * kHold)
          .minEpochIterations(3);
      b.run("HYLE MemoryArena (burst, TLS overflow+shard refill)", [&] {
        run_threads([&](int) {
          std::vector<void*> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i) held[i] = burst.allocate();
            for (int i = 0; i < kHold; ++i) burst.deallocate(held[i]);
          }
        });
      });
      b.run("std::pmr::synchronized_pool_resource (burst)", [&] {
        std::pmr::synchronized_pool_resource spr;
        run_threads([&](int) {
          std::vector<void*> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = spr.allocate(kObj, alignof(std::max_align_t));
            for (int i = 0; i < kHold; ++i)
              spr.deallocate(held[i], kObj, alignof(std::max_align_t));
          }
        });
      });
#if defined(HYLE_BENCH_TBB)
      b.run("oneTBB scalable_allocator (burst)", [&] {
        run_threads([&](int) {
          std::vector<void*> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i) held[i] = scalable_malloc(kObj);
            for (int i = 0; i < kHold; ++i) scalable_free(held[i]);
          }
        });
      });
#endif
      b.run("raw mi_malloc/free (burst, per-thread heaps)", [&] {
        run_threads([&](int) {
          std::vector<void*> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i) held[i] = mi_malloc(kObj);
            for (int i = 0; i < kHold; ++i) mi_free(held[i]);
          }
        });
      });
    }

    // ---- D2. CROSS-THREAD handoff: allocate on thread t, free on thread t+1 ----
    // The work-stealing / producer-consumer pattern MemoryArena's address-routed free
    // is built for (design: "fiber stack allocated on core A, freed on core B"). Each
    // thread allocates a batch, a barrier rotates ownership, then each thread frees the
    // NEIGHBOR's batch -- so deallocate() cannot hit the freeing thread's TLS (foreign
    // blocks) and must route by address to the owning shard's Treiber stack. This is
    // the genuine shared-backend contention section C's same-thread ping-pong hides.
    // (mimalloc is shown as the apples-to-oranges baseline: its cross-thread free is a
    // deferred atomic push to the owning heap -- a different contract, context only.)
    // kBatch is large so the two per-round barrier syncs amortize over 256 alloc + 256
    // free ops -- otherwise the barrier latency (not the allocator) dominates ns/op.
    constexpr int kBatch = 256;
    constexpr int kHandoffRounds = 250;
    MemoryArena handoff(sys, kObj, kThreads * kBatch * 2);
    {
      std::barrier sync(kThreads);
      std::vector<std::vector<void*>> slots(kThreads, std::vector<void*>(kBatch));
      nb::Bench b;
      b.title("8-thread CROSS-THREAD handoff: alloc on t, free on t+1")
          .relative(true)
          .batch(static_cast<std::uint64_t>(kThreads) * kHandoffRounds * kBatch)
          .minEpochIterations(3);
      b.run("HYLE MemoryArena (cross-thread, address-routed shard free)", [&] {
        run_threads([&](int t) {
          const int nbr = (t + 1) % kThreads;
          for (int r = 0; r < kHandoffRounds; ++r) {
            for (int i = 0; i < kBatch; ++i) slots[t][i] = handoff.allocate();
            sync.arrive_and_wait();                 // publish my batch
            for (int i = 0; i < kBatch; ++i) handoff.deallocate(slots[nbr][i]);
            sync.arrive_and_wait();                 // all frees done before next round
          }
        });
      });
      b.run("std::pmr::synchronized_pool_resource (cross-thread)", [&] {
        std::pmr::synchronized_pool_resource spr;
        run_threads([&](int t) {
          const int nbr = (t + 1) % kThreads;
          for (int r = 0; r < kHandoffRounds; ++r) {
            for (int i = 0; i < kBatch; ++i)
              slots[t][i] = spr.allocate(kObj, alignof(std::max_align_t));
            sync.arrive_and_wait();
            for (int i = 0; i < kBatch; ++i)
              spr.deallocate(slots[nbr][i], kObj, alignof(std::max_align_t));
            sync.arrive_and_wait();
          }
        });
      });
#if defined(HYLE_BENCH_TBB)
      b.run("oneTBB scalable_allocator (cross-thread)", [&] {
        run_threads([&](int t) {
          const int nbr = (t + 1) % kThreads;
          for (int r = 0; r < kHandoffRounds; ++r) {
            for (int i = 0; i < kBatch; ++i) slots[t][i] = scalable_malloc(kObj);
            sync.arrive_and_wait();
            for (int i = 0; i < kBatch; ++i) scalable_free(slots[nbr][i]);
            sync.arrive_and_wait();
          }
        });
      });
#endif
      b.run("raw mi_malloc/free (cross-thread, deferred atomic push)", [&] {
        run_threads([&](int t) {
          const int nbr = (t + 1) % kThreads;
          for (int r = 0; r < kHandoffRounds; ++r) {
            for (int i = 0; i < kBatch; ++i) slots[t][i] = mi_malloc(kObj);
            sync.arrive_and_wait();
            for (int i = 0; i < kBatch; ++i) mi_free(slots[nbr][i]);
            sync.arrive_and_wait();
          }
        });
      });
    }
  }

  return 0;
}
