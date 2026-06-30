// HYLE specialized-allocator cluster vs SOTA. performance.md:127 claims arena/bump
// allocators are "5-10x+ vs new/delete" -- never benchmarked until now. The honest
// SOTA references are in the STANDARD LIBRARY (no external clone):
//   * std::pmr::monotonic_buffer_resource   -- std's arena (bump, bulk release)
//   * std::pmr::unsynchronized_pool_resource -- std's pool (size-class free-list)
// plus raw mi_malloc/mi_free (the general malloc = what `new/delete` resolve to,
// since HYLE's global backing is mimalloc) as the "5-10x" target baseline.
//
// TWO workload families:
//  A. BUMP (alloc N, no per-object free, bulk release): HYLE Arena + StackAllocator
//     vs std::pmr::monotonic_buffer_resource vs raw mi (alloc-all then free-all).
//  B. POOL CHURN (alloc/free same size class, reuse): HYLE MemoryArena (lock-free)
//     + GrowablePool (single-thread) vs std::pmr::unsynchronized_pool_resource vs
//     raw mi (malloc/free churn).

#include <mimalloc.h>
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/memory_arena.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/stack_allocator.h"

using hyle::mem::Allocator;
using hyle::mem::Arena;
using hyle::mem::GrowablePool;
using hyle::mem::MemoryArena;
using hyle::mem::ObjectPool;
using hyle::mem::StackAllocator;
using hyle::mem::system_allocator;

namespace {
constexpr std::size_t kObj = 64;       // one small same-size object
constexpr int kN = 8192;               // allocations per round
constexpr std::size_t kRegion = kN * kObj + 4096;  // bump region (+slack)
namespace nb = ankerl::nanobench;
}  // namespace

int main() {
  Allocator sys = system_allocator();  // mimalloc backing

  // ---- A. BUMP: alloc N small, no per-object free, bulk release ----
  {
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

  // ---- A2. StackAllocator mark/rewind: nested-scope LIFO release ----
  // mark()/rewind() is the StackAllocator capability reset() can't express: save a
  // checkpoint, allocate a temporary batch, roll back to the checkpoint (freeing
  // exactly that batch) while KEEPING everything before it. The scratch/frame
  // pattern (e.g. a taskgraph pass that allocates per-node temporaries and rewinds
  // between nodes). Each round: alloc a base batch once, then inner loop {mark;
  // alloc temps; rewind}. No std rival -- std::pmr::monotonic has no partial
  // rewind (only whole-buffer release) -- so reset()-per-round is the baseline the
  // mark/rewind cost is read against (both are pointer moves; this proves rewind
  // is as cheap as a fresh bump, not a free-list walk).
  {
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

  // ---- B. POOL CHURN: alloc/free same size class, slot reuse ----
  {
    nb::Bench b;
    b.title("pool churn: 8192x alloc/free same 64B class")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);

    // HYLE MemoryArena (lock-free fixed pool).
    MemoryArena mpool(sys, kObj, kN);
    b.run("HYLE MemoryArena (lock-free pool)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = mpool.allocate();
        nb::doNotOptimizeAway(p);
        mpool.deallocate(p);
      }
    });

    // HYLE GrowablePool (single-threaded, no atomics).
    struct Obj { std::byte b[kObj]; };
    GrowablePool<Obj> gpool(sys, 64);
    b.run("HYLE GrowablePool (single-thread)", [&] {
      for (int i = 0; i < kN; ++i) {
        Obj* p = gpool.create();
        nb::doNotOptimizeAway(p);
        gpool.destroy(p);
      }
    });

    // HYLE ObjectPool<T> (typed create/destroy over the lock-free MemoryArena).
    // Measures the typed shell's overhead atop the arena: it adds placement-new
    // + ~T over MemoryArena's raw alloc/free, so it should track MemoryArena (NOT
    // GrowablePool -- ObjectPool is the lock-free/shared-capable one, the same
    // atomic-CAS class). Was a DEAD include (object_pool.h:28) until this row.
    ObjectPool<Obj> opool(sys, kN);
    b.run("HYLE ObjectPool (typed, lock-free arena)", [&] {
      for (int i = 0; i < kN; ++i) {
        Obj* p = opool.create();
        nb::doNotOptimizeAway(p);
        opool.destroy(p);
      }
    });

    // std::pmr::unsynchronized_pool_resource (SOTA: std's pool).
    b.run("std::pmr::unsynchronized_pool_resource (SOTA)", [&] {
      std::pmr::unsynchronized_pool_resource upr;
      for (int i = 0; i < kN; ++i) {
        void* p = upr.allocate(kObj, alignof(std::max_align_t));
        nb::doNotOptimizeAway(p);
        upr.deallocate(p, kObj, alignof(std::max_align_t));
      }
    });

    // raw mi_malloc/free churn (general malloc baseline).
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
  {
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

    // (2) ONE MemoryArena shared by all 8 threads: the honest shared-fixed-slab
    // coordination cost (atomic CAS/op), NOT a per-thread heap. Sharding removed
    // the O(threads) contention (278x->1.95x); the residual is the CAS constant.
    MemoryArena shared(sys, kObj, kThreads * 256);  // small pool -> real reuse churn
    b.run("HYLE MemoryArena (8 threads, 1 SHARED pool)", [&] {
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

  return 0;
}
