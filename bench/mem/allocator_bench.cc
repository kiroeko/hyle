// HYLE Allocator(mimalloc-backed) overhead audit + OS-allocator comparison.
// THREE rivals per workload:
//   * raw mi_malloc/mi_free  -- the SOTA reference (set as the relative baseline,
//     so HYLE's % reads DIRECTLY as the vtable-wrapper overhead: one non-inlinable
//     indirect call (Win64 ABI spills the 5th arg `alignment` to the stack) plus a
//     few branches in system_realloc. In an EMPTY alloc/free loop (denominator
//     ~7-8ns) this fixed cost shows as a large %; the "alloc + touch memory" row
//     below is the honest read -- with any real work the wrapper cost amortizes.
//   * HYLE system_allocator  -- our Allocator vtable over the same mimalloc.
//   * std::malloc/free       -- the OS allocator (weak baseline; shows mimalloc's win).
// mimalloc is a hard vendored dep (third_party/mimalloc), so this needs no gate.

#include <mimalloc.h>
#include <nanobench.h>

#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;

namespace {
constexpr int kAllocs = 200000;
// A spread of small sizes (the common allocator workload).
std::uint32_t size_at(int i) { return 16 + ((i * 2654435761u) % 496); }  // 16..511
}  // namespace

int main() {
  Allocator a = system_allocator();  // mimalloc backing (HYLE_USE_MIMALLOC=ON)

  // ---- single-thread alloc/free churn ----
  {
    ankerl::nanobench::Bench b;
    b.title("alloc+free churn, 200k small blocks")
        .relative(true)
        .batch(kAllocs)
        .minEpochIterations(20);
    b.run("raw mi_malloc/free (SOTA)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        void* p = mi_malloc(size_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
    b.run("HYLE system_allocator (vtable)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        void* p = a.allocate(size_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, size_at(i));
      }
    });
    b.run("HYLE SystemAllocator (compile-time, inlined)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        void* p = hyle::mem::SystemAllocator::allocate(size_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::SystemAllocator::deallocate(p, size_at(i));
      }
    });
    b.run("std::malloc/free (OS)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        void* p = std::malloc(size_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        std::free(p);
      }
    });
  }

  // ---- realistic: alloc + TOUCH the memory + free (the honest read) ----
  // The empty-loop churn above magnifies the wrapper's fixed indirect-call cost
  // because the denominator is tiny. Real callers WRITE the block they ask for.
  // Touching first+last byte (a cache miss on fresh memory dwarfs the indirect
  // call) is the apples-to-apples read of what the vtable actually costs in use.
  {
    ankerl::nanobench::Bench b;
    b.title("alloc + touch + free, 200k small blocks")
        .relative(true)
        .batch(kAllocs)
        .minEpochIterations(20);
    auto touch = [](void* p, std::uint32_t sz) {
      auto* c = static_cast<volatile char*>(p);
      c[0] = 1;
      c[sz - 1] = 1;
    };
    b.run("raw mi_malloc/free (SOTA)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        std::uint32_t sz = size_at(i);
        void* p = mi_malloc(sz);
        touch(p, sz);
        ankerl::nanobench::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
    b.run("HYLE system_allocator (vtable)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        std::uint32_t sz = size_at(i);
        void* p = a.allocate(sz);
        touch(p, sz);
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, sz);
      }
    });
    b.run("std::malloc/free (OS)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        std::uint32_t sz = size_at(i);
        void* p = std::malloc(sz);
        touch(p, sz);
        ankerl::nanobench::doNotOptimizeAway(p);
        std::free(p);
      }
    });
  }

  // ---- batched alloc-all then free-all (exercises the free list / reuse) ----
  {
    constexpr int kN = 50000;
    ankerl::nanobench::Bench b;
    b.title("alloc-all then free-all, 50k blocks")
        .relative(true)
        .batch(kN)
        .minEpochIterations(300);
    b.run("raw mi_malloc/free (SOTA)", [&] {
      std::vector<void*> ps(kN);
      for (int i = 0; i < kN; ++i) ps[i] = mi_malloc(size_at(i));
      for (int i = 0; i < kN; ++i) mi_free(ps[i]);
      ankerl::nanobench::doNotOptimizeAway(ps.data());
    });
    b.run("HYLE system_allocator (vtable)", [&] {
      std::vector<void*> ps(kN);
      for (int i = 0; i < kN; ++i) ps[i] = a.allocate(size_at(i));
      for (int i = 0; i < kN; ++i) a.deallocate(ps[i], size_at(i));
      ankerl::nanobench::doNotOptimizeAway(ps.data());
    });
    b.run("std::malloc/free (OS)", [&] {
      std::vector<void*> ps(kN);
      for (int i = 0; i < kN; ++i) ps[i] = std::malloc(size_at(i));
      for (int i = 0; i < kN; ++i) std::free(ps[i]);
      ankerl::nanobench::doNotOptimizeAway(ps.data());
    });
  }

  // ---- concurrent allocation (mimalloc's design point: per-thread heaps) ----
  {
    constexpr int kThreads = 8;
    constexpr int kPer = 50000;
    auto run_threads = [](auto&& body) {
      std::vector<std::thread> ts;
      for (int t = 0; t < kThreads; ++t) ts.emplace_back(body);
      for (auto& th : ts) th.join();
    };
    ankerl::nanobench::Bench b;
    b.title("8-thread concurrent alloc/free")
        .relative(true)
        .batch(kThreads * kPer)
        .minEpochIterations(150);
    b.run("raw mi_malloc/free (SOTA)", [&] {
      run_threads([] {
        for (int i = 0; i < kPer; ++i) {
          void* p = mi_malloc(size_at(i));
          ankerl::nanobench::doNotOptimizeAway(p);
          mi_free(p);
        }
      });
    });
    b.run("HYLE system_allocator (vtable)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = a.allocate(size_at(i));
          ankerl::nanobench::doNotOptimizeAway(p);
          a.deallocate(p, size_at(i));
        }
      });
    });
    b.run("std::malloc/free (OS)", [&] {
      run_threads([] {
        for (int i = 0; i < kPer; ++i) {
          void* p = std::malloc(size_at(i));
          ankerl::nanobench::doNotOptimizeAway(p);
          std::free(p);
        }
      });
    });
  }

  return 0;
}
