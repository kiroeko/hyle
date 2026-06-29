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

  // ---- C. CONCURRENT pool churn: the lock-free design POINT ----
  // The single-thread rows above measure MemoryArena OUTSIDE its design point --
  // a lock-free pool pays an atomic CAS per op that a single-thread pool skips.
  // Its reason to exist is SHARED concurrent churn. Fair read: 8 threads hammer
  // ONE shared MemoryArena vs raw mi (per-thread heaps = the concurrent SOTA).
  // (std::pmr::unsynchronized_pool_resource is NOT thread-safe -> not a rival
  // here; the synchronized_pool_resource would just be a global lock.)
  {
    constexpr int kThreads = 8;
    constexpr int kPer = 50000;
    auto run_threads = [](auto&& body) {
      std::vector<std::thread> ts;
      ts.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) ts.emplace_back(body);
      for (auto& th : ts) th.join();
    };
    nb::Bench b;
    b.title("8-thread SHARED pool churn (lock-free design point)")
        .relative(true)
        .batch(kThreads * kPer)
        .minEpochIterations(10);

    // One MemoryArena shared by all 8 threads (its lock-free reason to exist).
    MemoryArena shared(sys, kObj, kThreads * 256);  // small pool -> real reuse churn
    b.run("HYLE MemoryArena (8 threads, 1 shared pool)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = shared.allocate();
          nb::doNotOptimizeAway(p);
          shared.deallocate(p);
        }
      });
    });

    // raw mi_malloc/free, 8 threads -- mimalloc's per-thread heaps are the SOTA
    // concurrent allocator (no inter-thread contention by design).
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
