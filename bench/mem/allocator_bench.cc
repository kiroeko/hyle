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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <pthread/qos.h>
#endif

#include "hyle/mem/allocator.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/mem/system_inline.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;

namespace {
constexpr int kAllocs = 200000;
// A spread of small sizes (the common allocator workload).
std::uint32_t size_at(int i) { return 16 + ((i * 2654435761u) % 496); }  // 16..511

// HYLE_BENCH_SECTIONS="R" runs only realloc. Unset/empty preserves the full
// historical sweep; the filter keeps multi-process certification bounded.
bool want_section(char id) {
  const char* sections = std::getenv("HYLE_BENCH_SECTIONS");
  if (sections == nullptr || sections[0] == '\0') return true;
  for (const char* current = sections; *current != '\0'; ++current) {
    if (*current == id) return true;
  }
  return false;
}

// Section L can run one variant per process.  Certification cycles the variants
// between fresh processes instead of executing them in one fixed order, which
// otherwise turns allocator warmup/thermal drift into a fake adapter delta.
bool want_link_variant(char id) {
  const char* variant = std::getenv("HYLE_MEM_LINK_VARIANT");
  return variant == nullptr || variant[0] == '\0' || variant[0] == id;
}

bool configure_apple_qos() {
#if defined(__APPLE__) && defined(__MACH__)
  const char* requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  if (requested == nullptr || requested[0] == '\0') requested = "interactive";
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  if (std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
  } else if (std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_mem_bench: HYLE_BENCH_APPLE_QOS must be interactive "
                 "or utility\n");
    return false;
  }
  const int rc = pthread_set_qos_class_self_np(qos, 0);
  std::printf("HYLE_MEM_LINK_QOS requested=%s qos_ok=%d affinity=unsupported\n",
              requested, rc == 0 ? 1 : 0);
  return rc == 0;
#else
  return true;
#endif
}

// std::pmr::memory_resource over mimalloc -- the SAME-CLASS rival to HYLE's
// runtime Allocator vtable. BOTH are runtime polymorphism over the same backing
// (mimalloc), so racing them isolates the DISPATCH cost (HYLE 3-fn fn-ptr vtable
// vs std::pmr's virtual do_allocate). Racing the vtable against RAW mi (non-
// polymorphic, inlined) measures "polymorphism vs none" -- not apples-to-apples;
// THIS is the fair "is our runtime path SOTA for its class?" check.
class MiResource final : public std::pmr::memory_resource {
  void* do_allocate(std::size_t bytes, std::size_t align) override {
    return mi_malloc_aligned(bytes, align);
  }
  void do_deallocate(void* p, std::size_t, std::size_t) override { mi_free(p); }
  bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override {
    return this == &o;
  }
};

// A PMR reference over the exact same hyle_mem backend entry used by the
// shared SystemAllocator.  This is deliberately not mi_malloc in the
// executable: on a shared build that would compare an executable-local static
// mimalloc against a DLL call and attribute the different link shape to the
// adapter.  BackendResource and PmrResource below both cross the same import
// seam; their delta therefore isolates HYLE's extra runtime Allocator dispatch.
class BackendResource final : public std::pmr::memory_resource {
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* pointer = hyle::mem::detail::system_backend_allocate(
        hyle::mem::detail::pmr_backing_size(bytes), alignment);
    if (pointer == nullptr) throw std::bad_alloc{};
    return pointer;
  }
  void do_deallocate(void* p, std::size_t, std::size_t) override {
    hyle::mem::detail::system_backend_deallocate(p);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};
}  // namespace

int main() {
  if (!configure_apple_qos()) return 2;
  Allocator a = system_allocator();  // mimalloc backing (HYLE_USE_MIMALLOC=ON)

#if defined(HYLE_STATIC)
  std::puts("HYLE_MEM_LINKAGE static");
#else
  std::puts("HYLE_MEM_LINKAGE shared");
#endif
#if defined(HYLE_TESTING)
  std::puts("HYLE_MEM_BUILD_MODE testing");
#else
  std::puts("HYLE_MEM_BUILD_MODE shipping");
#endif

  // ---- shipping same-link certificate ------------------------------------
  // Every formal row crosses the same hyle_mem import seam and reaches the
  // same mimalloc instance.  Shared is the normal binary SDK shape; static is
  // also certifiable when the driver explicitly requests that linkage. Both
  // avoid the old
  // benchmark distortion where raw mi_* lived in the executable while HYLE's
  // backend lived in a DLL/DSO.  The direct mi_* row remains a useful absolute
  // lower bound, but is intentionally excluded from shared SOTA judgments.
  if (want_section('L')) {
    constexpr int kN = 200000;
    auto bytes_at = [](int i) -> std::size_t {
      return 16u + 8u * ((static_cast<std::uint32_t>(i) * 2654435761u) % 62u);
    };
    BackendResource backend_resource;
    hyle::mem::PmrResource hyle_resource(a);
    std::pmr::polymorphic_allocator<std::byte> backend_pmr(&backend_resource);
    std::pmr::polymorphic_allocator<std::byte> system_pmr(
        hyle::mem::system_memory_resource());
    std::pmr::polymorphic_allocator<std::byte> hyle_pmr(&hyle_resource);
    hyle::mem::SystemStdAllocator<std::uint64_t> system_std_allocator;
    hyle::mem::StdAllocator<std::uint64_t> std_allocator(a);

    ankerl::nanobench::Bench b;
    b.title("shared same-link alloc/free, 200k small blocks")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    auto run = [&]<class Function>(char id, const char* name,
                                  Function&& function) {
      if (want_link_variant(id)) b.run(name, std::forward<Function>(function));
    };
    run('B', "backend raw seam (same-link baseline)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = bytes_at(i);
        void* p = hyle::mem::detail::system_backend_allocate(
            bytes, hyle::mem::kSystemDefaultAlignment);
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::detail::system_backend_deallocate(p);
      }
    });
    run('S', "SystemAllocator (same-link direct)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = bytes_at(i);
        void* p = hyle::mem::SystemAllocator::allocate(bytes);
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::SystemAllocator::deallocate(p, bytes);
      }
    });
    run('A', "Allocator (same-link runtime)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = bytes_at(i);
        void* p = a.allocate(bytes);
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, bytes);
      }
    });
    run('P', "PMR direct backend (same-link rival)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = bytes_at(i);
        void* p = backend_pmr.allocate_bytes(bytes);
        ankerl::nanobench::doNotOptimizeAway(p);
        backend_pmr.deallocate_bytes(p, bytes);
      }
    });
    run('M', "system_memory_resource (same-link direct)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = bytes_at(i);
        void* p = system_pmr.allocate_bytes(bytes);
        ankerl::nanobench::doNotOptimizeAway(p);
        system_pmr.deallocate_bytes(p, bytes);
      }
    });
    run('R', "HYLE PmrResource (same-link runtime)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = bytes_at(i);
        void* p = hyle_pmr.allocate_bytes(bytes);
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle_pmr.deallocate_bytes(p, bytes);
      }
    });
    run('T', "HYLE SystemStdAllocator<uint64_t> (same-link direct)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t count = bytes_at(i) / sizeof(std::uint64_t);
        auto* p = system_std_allocator.allocate(count);
        ankerl::nanobench::doNotOptimizeAway(p);
        system_std_allocator.deallocate(p, count);
      }
    });
    run('D', "HYLE StdAllocator<uint64_t> (same-link runtime)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t count = bytes_at(i) / sizeof(std::uint64_t);
        auto* p = std_allocator.allocate(count);
        ankerl::nanobench::doNotOptimizeAway(p);
        std_allocator.deallocate(p, count);
      }
    });
    // Reallocation is a colder but still ordinary SDK operation (container
    // growth, byte buffers).  Keep it in the same-link certificate instead of
    // comparing the DLL/DSO path with a second mimalloc instance in the exe.
    constexpr std::array<std::size_t, 7> kReallocSizes = {
        16, 32, 64, 128, 256, 512, 1024};
    run('E', "backend realloc seam (same-link baseline)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = hyle::mem::detail::system_backend_allocate(
            kReallocSizes[0], hyle::mem::kSystemDefaultAlignment);
        for (std::size_t index = 1; index < kReallocSizes.size(); ++index) {
          p = hyle::mem::detail::system_backend_reallocate(
              p, kReallocSizes[index], hyle::mem::kSystemDefaultAlignment);
        }
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::detail::system_backend_deallocate(p);
      }
    });
    run('F', "SystemAllocator realloc (same-link direct)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = hyle::mem::SystemAllocator::allocate(kReallocSizes[0]);
        for (std::size_t index = 1; index < kReallocSizes.size(); ++index) {
          p = hyle::mem::SystemAllocator::reallocate(
              p, kReallocSizes[index - 1], kReallocSizes[index]);
        }
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::SystemAllocator::deallocate(p, kReallocSizes.back());
      }
    });
    run('G', "Allocator realloc (same-link runtime)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = a.allocate(kReallocSizes[0]);
        for (std::size_t index = 1; index < kReallocSizes.size(); ++index) {
          p = a.reallocate(p, kReallocSizes[index - 1], kReallocSizes[index]);
        }
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, kReallocSizes.back());
      }
    });
    run('X', "raw mi_* in executable (non-symmetric lower bound)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = mi_malloc(bytes_at(i));
        ankerl::nanobench::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
  }

  // ---- mimalloc v3 sub-default alignment regression guard -----------------
  // v3 may naturally align blocks smaller than max_align_t to their smaller
  // size class. HYLE pads those requests to 16 and stays on mi_malloc instead
  // of sending the hottest tiny-object path through aligned over-allocation.
  // Keep both raw vendor shapes visible so a future backend upgrade cannot
  // trade the public alignment contract for an unmeasured slow path.
  if (want_section('M')) {
    constexpr int kN = 200000;
    auto tiny_size_at = [](int i) -> std::size_t {
      return 1u + (static_cast<std::uint32_t>(i) % 15u);
    };
    ankerl::nanobench::Bench b;
    b.title("sub-max-align alloc/free, 200k tiny blocks")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    b.run("raw mi_malloc padded to 16 (backend shape)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = mi_malloc(16);
        ankerl::nanobench::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
    b.run("raw mi_malloc_aligned(size, 16)", [&] {
      for (int i = 0; i < kN; ++i) {
        void* p = mi_malloc_aligned(tiny_size_at(i), 16);
        ankerl::nanobench::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
    b.run("HYLE SystemAllocator (contract path)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = tiny_size_at(i);
        void* p = hyle::mem::SystemAllocator::allocate(bytes, 1);
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::SystemAllocator::deallocate(p, bytes, 1);
      }
    });
    b.run("HYLE system_allocator (vtable contract path)", [&] {
      for (int i = 0; i < kN; ++i) {
        const std::size_t bytes = tiny_size_at(i);
        void* p = a.allocate(bytes, 1);
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, bytes, 1);
      }
    });
  }

  // ---- single-thread alloc/free churn ----
  if (want_section('A')) {
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
    // Same-class rival: std::pmr runtime polymorphism over the SAME mimalloc.
    MiResource mr;
    std::pmr::polymorphic_allocator<std::byte> pa(&mr);
    b.run("std::pmr over mimalloc (same-class rival)", [&] {
      for (int i = 0; i < kAllocs; ++i) {
        std::uint32_t sz = size_at(i);
        void* p = pa.allocate_bytes(sz);
        ankerl::nanobench::doNotOptimizeAway(p);
        pa.deallocate_bytes(p, sz);
      }
    });
    b.run("HYLE SystemAllocator (default compile-time binding)", [&] {
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
  if (want_section('B')) {
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
  if (want_section('C')) {
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
  if (want_section('D')) {
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

  // ---- reallocate: grow an existing block (the COLD path, but still benched) --
  // Allocator splits alloc/free/realloc (header rationale: register-friendly, no
  // dispatch branch). The realloc path had no direct bench (PodVector growth exercises
  // it only indirectly). Grow a block through a sequence of sizes (16->1024 via
  // doubling) so realloc must sometimes copy. HYLE vtable reallocate vs raw
  // mi_realloc (SOTA) vs the compile-time SystemAllocator path vs the runtime
  // Allocator vtable vs std::realloc (OS). Each iteration starts from a fresh
  // small block so the grow chain is identical across rivals. Keeping the two
  // HYLE rows separate prevents runtime injection cost from being attributed to
  // the default compile-time path.
  if (want_section('R')) {
    constexpr int kChains = 20000;
    static constexpr std::size_t kSizes[] = {16, 32, 64, 128, 256, 512, 1024};
    ankerl::nanobench::Bench b;
    b.title("realloc grow chain 16->1024 (7 steps)")
        .relative(true)
        .batch(kChains)
        .minEpochIterations(50);
    b.run("raw mi_realloc (SOTA)", [&] {
      for (int i = 0; i < kChains; ++i) {
        void* p = mi_malloc(kSizes[0]);
        for (int s = 1; s < 7; ++s) p = mi_realloc(p, kSizes[s]);
        ankerl::nanobench::doNotOptimizeAway(p);
        mi_free(p);
      }
    });
    b.run("HYLE SystemAllocator::reallocate (compile-time)", [&] {
      for (int i = 0; i < kChains; ++i) {
        void* p = hyle::mem::SystemAllocator::allocate(kSizes[0]);
        for (int s = 1; s < 7; ++s) {
          p = hyle::mem::SystemAllocator::reallocate(
              p, kSizes[s - 1], kSizes[s]);
        }
        ankerl::nanobench::doNotOptimizeAway(p);
        hyle::mem::SystemAllocator::deallocate(p, kSizes[6]);
      }
    });
    b.run("HYLE Allocator::reallocate (vtable)", [&] {
      for (int i = 0; i < kChains; ++i) {
        void* p = a.allocate(kSizes[0]);
        for (int s = 1; s < 7; ++s) p = a.reallocate(p, kSizes[s - 1], kSizes[s]);
        ankerl::nanobench::doNotOptimizeAway(p);
        a.deallocate(p, kSizes[6]);
      }
    });
    b.run("std::realloc (OS)", [&] {
      for (int i = 0; i < kChains; ++i) {
        void* p = std::malloc(kSizes[0]);
        for (int s = 1; s < 7; ++s) p = std::realloc(p, kSizes[s]);
        ankerl::nanobench::doNotOptimizeAway(p);
        std::free(p);
      }
    });
  }

  return 0;
}
