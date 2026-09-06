#include <mimalloc.h>
#include <nanobench.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <new>
#include <utility>

#include "hyle/mem/heap.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;
namespace mem = hyle::mem;

constexpr std::size_t kBatch = 64;

constexpr std::array<std::size_t, kBatch> make_sizes() noexcept {
  std::array<std::size_t, kBatch> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = 24u * (1u + ((index * 13u + 5u) % 128u));
  }
  return result;
}

constexpr std::array<std::size_t, kBatch> make_alignments() noexcept {
  std::array<std::size_t, kBatch> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = std::size_t{16} << ((index * 5u + 1u) % 3u);
  }
  return result;
}

constexpr auto kSizes = make_sizes();
constexpr auto kAlignments = make_alignments();

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

void touch_allocation(void* pointer, std::size_t size,
                      std::uint8_t value) noexcept {
  if (pointer == nullptr || size == 0) std::abort();
  auto* bytes = static_cast<std::byte*>(pointer);
  bytes[0] = static_cast<std::byte>(value);
  bytes[size - 1] = static_cast<std::byte>(value ^ 0x5au);
  nb::doNotOptimizeAway(pointer);
}

class HyleHeapReplay {
 public:
  HyleHeapReplay()
      : heap_(mem::Heap::create().value()), allocator_(heap_.allocator()) {}

  void run() noexcept {
    std::array<void*, kBatch> allocations{};
    for (std::size_t index = 0; index < kBatch; ++index) {
      allocations[index] = allocator_.allocate(kSizes[index], kAlignments[index]);
      touch_allocation(allocations[index], kSizes[index],
                       static_cast<std::uint8_t>(index));
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      const std::size_t item = index - 1;
      allocator_.deallocate(allocations[item], kSizes[item],
                            kAlignments[item]);
    }
  }

 private:
  mem::Heap heap_;
  mem::AllocatorRef allocator_;
};

class PmrHeapResource final : public std::pmr::memory_resource {
 public:
  PmrHeapResource() : heap_(mi_heap_new()) {
    if (heap_ == nullptr) std::abort();
  }
  ~PmrHeapResource() override { mi_heap_destroy(heap_); }

 private:
  void* do_allocate(std::size_t size, std::size_t alignment) override {
    void* result = alignment <= 16
                       ? mi_heap_malloc(heap_, size < 16 ? 16 : size)
                       : mi_heap_malloc_aligned(heap_, size, alignment);
    if (result == nullptr) throw std::bad_alloc();
    return result;
  }

  void do_deallocate(void* pointer, std::size_t size,
                     std::size_t alignment) override {
    if (alignment <= 16) {
      mi_free_size(pointer, size < 16 ? 16 : size);
    } else {
      mi_free_size_aligned(pointer, size, alignment);
    }
  }

  bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  mi_heap_t* heap_ = nullptr;
};

HYLE_NOINLINE std::pmr::memory_resource* erase_resource(
    PmrHeapResource* resource) noexcept {
  return resource;
}

class PmrHeapReplay {
 public:
  PmrHeapReplay() : resource_(erase_resource(&backing_)) {}

  void run() noexcept {
    std::array<void*, kBatch> allocations{};
    for (std::size_t index = 0; index < kBatch; ++index) {
      allocations[index] = resource_->allocate(kSizes[index], kAlignments[index]);
      touch_allocation(allocations[index], kSizes[index],
                       static_cast<std::uint8_t>(index));
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      const std::size_t item = index - 1;
      resource_->deallocate(allocations[item], kSizes[item],
                            kAlignments[item]);
    }
  }

 private:
  PmrHeapResource backing_;
  std::pmr::memory_resource* resource_ = nullptr;
};

struct HeapPair {
  std::unique_ptr<HyleHeapReplay> hyle;
  std::unique_ptr<PmrHeapReplay> pmr;

  explicit HeapPair(bool peer_allocation_first) {
    if (peer_allocation_first) {
      pmr = std::make_unique<PmrHeapReplay>();
      hyle = std::make_unique<HyleHeapReplay>();
    } else {
      hyle = std::make_unique<HyleHeapReplay>();
      pmr = std::make_unique<PmrHeapReplay>();
    }
  }
};

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first, Candidate&& candidate,
              Peer&& peer) {
  if (peer_first) {
    bench.run("std pmr erased mimalloc heap lifecycle N=64",
              std::forward<Peer>(peer));
    bench.run("HYLE Heap allocator lifecycle N=64",
              std::forward<Candidate>(candidate));
  } else {
    bench.run("HYLE Heap allocator lifecycle N=64",
              std::forward<Candidate>(candidate));
    bench.run("std pmr erased mimalloc heap lifecycle N=64",
              std::forward<Peer>(peer));
  }
}

}  // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_MEM_V6_HEAP_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  HeapPair heaps(peer_allocation_first);
  heaps.hyle->run();
  heaps.pmr->run();

  nb::Bench bench;
  bench.title("isolated heap lifecycle aggregate N=64")
      .relative(true)
      .batch(kBatch)
      .minEpochIterations(512)
      .minEpochTime(std::chrono::milliseconds(50));
  run_pair(bench, peer_first, [&] { heaps.hyle->run(); },
           [&] { heaps.pmr->run(); });
}
