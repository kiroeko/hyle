// mimalloc-backed system_allocator vs the OS allocator (std::malloc/free).
// mem-design §101 gate. Two workloads: single-thread alloc/free churn, and the
// case mimalloc is built for -- many threads allocating concurrently (the OS
// allocator serializes on a global lock; mimalloc has per-thread heaps).

#include <nanobench.h>

#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"

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
    b.title("alloc+free churn, 200k small blocks").relative(true).batch(kAllocs);
    b.run("std::malloc/free", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        void* p = std::malloc(size_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        std::free(p);
      }
    });
    b.run("HYLE system_allocator (mimalloc)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        void* p = a.allocate(size_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, size_at(i));
      }
    });
  }

  // ---- batched alloc-all then free-all (exercises the free list / reuse) ----
  {
    constexpr int kN = 50000;
    ankerl::nanobench::Bench b;
    b.title("alloc-all then free-all, 50k blocks").relative(true).batch(kN);
    b.run("std::malloc/free", [&] {
      std::vector<void*> ps(kN);
      for (int i = 0; i < kN; ++i) ps[i] = std::malloc(size_at(i));
      for (int i = 0; i < kN; ++i) std::free(ps[i]);
      ankerl::nanobench::doNotOptimizeAway(ps.data());
    });
    b.run("HYLE system_allocator (mimalloc)", [&] {
      std::vector<void*> ps(kN);
      for (int i = 0; i < kN; ++i) ps[i] = a.allocate(size_at(i));
      for (int i = 0; i < kN; ++i) a.deallocate(ps[i], size_at(i));
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
    b.title("8-thread concurrent alloc/free").relative(true).batch(kThreads * kPer);
    b.run("std::malloc/free", [&] {
      run_threads([] {
        for (int i = 0; i < kPer; ++i) {
          void* p = std::malloc(size_at(i));
          ankerl::nanobench::doNotOptimizeAway(p);
          std::free(p);
        }
      });
    });
    b.run("HYLE system_allocator (mimalloc)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void* p = a.allocate(size_at(i));
          ankerl::nanobench::doNotOptimizeAway(p);
          a.deallocate(p, size_at(i));
        }
      });
    });
  }

  return 0;
}
