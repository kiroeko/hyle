// Qualified source-consumer replay for the four Mem v2 P1 families.
#include <nanobench.h>
#include <mimalloc.h>
#include <tbb/memory_pool.h>
#include <tbb/scalable_allocator.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <memory_resource>
#include <optional>
#include <utility>

#include "hyle/mem/arena.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/buffer_pool.h"
#include "hyle/mem/heap.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;
namespace mem = hyle::mem;

constexpr std::size_t kBatch = 64;
constexpr std::size_t kScratchBytes = 128 * 1024;

constexpr std::array<std::size_t, kBatch> make_sizes(
    std::size_t unit, std::size_t classes) noexcept {
  std::array<std::size_t, kBatch> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = unit * (1 + ((index * 13u + 5u) % classes));
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

constexpr auto kHeapSizes = make_sizes(24, 128);
constexpr auto kArenaSizes = make_sizes(24, 32);
constexpr auto kPoolSizes = make_sizes(256, 16);
constexpr auto kAlignments = make_alignments();
constexpr std::size_t kPoolAlignment = alignof(std::max_align_t);

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

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              const char* candidate_name, Candidate&& candidate,
              const char* peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

nb::Bench make_bench(const char* title) {
  nb::Bench bench;
  bench.title(title)
      .relative(true)
      .batch(kBatch)
      .minEpochTime(std::chrono::milliseconds(8));
  return bench;
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
  HyleHeapReplay() : heap_(mem::Heap::create().value()), allocator_(heap_.allocator()) {}

  void run() noexcept {
    std::array<void*, kBatch> allocations{};
    for (std::size_t index = 0; index < kBatch; ++index) {
      allocations[index] =
          allocator_.allocate(kHeapSizes[index], kAlignments[index]);
      touch_allocation(allocations[index], kHeapSizes[index],
                       static_cast<std::uint8_t>(index));
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      const std::size_t item = index - 1;
      allocator_.deallocate(allocations[item], kHeapSizes[item],
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
      allocations[index] =
          resource_->allocate(kHeapSizes[index], kAlignments[index]);
      touch_allocation(allocations[index], kHeapSizes[index],
                       static_cast<std::uint8_t>(index));
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      const std::size_t item = index - 1;
      resource_->deallocate(allocations[item], kHeapSizes[item],
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

class FixedArenaReplay {
 public:
  FixedArenaReplay() : arena_(storage_) {}

  void run() noexcept {
    arena_.clear();
    for (std::size_t index = 0; index < kBatch; ++index) {
      void* pointer = arena_.allocate(kArenaSizes[index], kAlignments[index]);
      touch_allocation(pointer, kArenaSizes[index],
                       static_cast<std::uint8_t>(index));
    }
  }

 private:
  alignas(64) std::array<std::byte, kScratchBytes> storage_{};
  mem::FixedArena arena_;
};

class PmrArenaReplay {
 public:
  PmrArenaReplay() { reset_owner(); }

  void run() {
    // std::pmr::monotonic_buffer_resource::release() releases upstream
    // blocks, but does not rewind a caller-provided initial buffer. Rebuild
    // the tiny owner in place to provide the same reusable external-buffer
    // epoch contract as FixedArena::clear().
    reset_owner();
    for (std::size_t index = 0; index < kBatch; ++index) {
      void* pointer = nullptr;
      try {
        pointer = arena_->allocate(kArenaSizes[index], kAlignments[index]);
      } catch (const std::bad_alloc&) {
        pointer = nullptr;
      }
      touch_allocation(pointer, kArenaSizes[index],
                       static_cast<std::uint8_t>(index));
    }
  }

 private:
  void reset_owner() {
    arena_.reset();
    arena_.emplace(storage_.data(), storage_.size(),
                   std::pmr::null_memory_resource());
  }

  alignas(64) std::array<std::byte, kScratchBytes> storage_{};
  std::optional<std::pmr::monotonic_buffer_resource> arena_;
};

struct ArenaPair {
  std::unique_ptr<FixedArenaReplay> hyle;
  std::unique_ptr<PmrArenaReplay> pmr;

  explicit ArenaPair(bool peer_allocation_first) {
    if (peer_allocation_first) {
      pmr = std::make_unique<PmrArenaReplay>();
      hyle = std::make_unique<FixedArenaReplay>();
    } else {
      hyle = std::make_unique<FixedArenaReplay>();
      pmr = std::make_unique<PmrArenaReplay>();
    }
  }
};

class BufferViewReplay {
 public:
  BufferViewReplay()
      : buffer_(mem::Buffer::create(mem::system_resource(), 64 * 4096, 64)
                    .value()) {}

  void hyle_run() noexcept {
    mem::BufferView whole = buffer_.view();
    for (std::size_t index = 0; index < kBatch; ++index) {
      const std::size_t offset = index * 4096;
      const std::size_t size = 3072 + ((index * 17u) & 1023u);
      auto slice = whole.subview(offset, size);
      if (!slice || !slice->fill(static_cast<std::byte>(index)).ok()) {
        std::abort();
      }
      nb::doNotOptimizeAway(slice->native_handle());
    }
  }

  void direct_run() noexcept {
    mem::BufferView whole = buffer_.view();
    for (std::size_t index = 0; index < kBatch; ++index) {
      const std::size_t relative = index * 4096;
      const std::size_t size = 3072 + ((index * 17u) & 1023u);
      if (!direct_subview_fill(whole, relative, size,
                               static_cast<std::byte>(index))
               .ok()) {
        std::abort();
      }
      nb::doNotOptimizeAway(whole.native_handle());
    }
  }

 private:
  static hyle::Status direct_subview_fill(mem::BufferView whole,
                                          std::size_t relative,
                                          std::size_t size,
                                          std::byte value) noexcept {
    if (!whole.valid()) {
      return hyle::Status::FailedPrecondition("buffer view is invalid");
    }
    if (relative > whole.size()) {
      return hyle::Status::OutOfRange(
          "buffer subview offset exceeds its range");
    }
    const std::size_t available = whole.size() - relative;
    if (size > available || whole.offset() > SIZE_MAX - relative) {
      return hyle::Status::OutOfRange(
          "buffer subview size exceeds its range");
    }
    return whole.resource().resource().fill(
        whole.native_handle(), whole.offset() + relative, size, value);
  }

  mem::Buffer buffer_;
};

class BufferPoolReplay {
 public:
  BufferPoolReplay()
      : pool_(mem::BufferPool::create(mem::system_resource(), options())
                  .value()) {
    candidate_run();
    const mem::BufferPoolStats warmed = pool_.stats();
    if (warmed.provider_allocations != kBatch ||
        warmed.cached_buffers != kBatch || warmed.live_buffers != 0) {
      std::abort();
    }
  }

  void candidate_run() noexcept {
    std::array<mem::BufferLease, kBatch> leases{};
    for (std::size_t index = 0; index < kBatch; ++index) {
      auto lease = pool_.acquire(kPoolSizes[index], kPoolAlignment);
      if (!lease) std::abort();
      leases[index] = std::move(*lease);
      nb::doNotOptimizeAway(leases[index].view().native_handle());
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      leases[index - 1].reset();
    }
  }

  void direct_run() noexcept {
    std::array<mem::Buffer, kBatch> buffers{};
    for (std::size_t index = 0; index < kBatch; ++index) {
      auto buffer = mem::Buffer::create(mem::system_resource(),
                                        kPoolSizes[index], kPoolAlignment);
      if (!buffer) std::abort();
      buffers[index] = std::move(*buffer);
      nb::doNotOptimizeAway(buffers[index].native_handle());
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      if (!buffers[index - 1].reset().ok()) std::abort();
    }
  }

  mem::BufferPoolStats stats() const noexcept { return pool_.stats(); }

 private:
  static mem::BufferPoolOptions options() noexcept {
    std::size_t retained = 0;
    for (std::size_t size : kPoolSizes) retained += size;
    mem::BufferPoolOptions result;
    result.max_cached_bytes = retained;
    result.max_cached_buffers = kBatch;
    result.max_reuse_slack = 0;
    result.growth_percent = 0;
    result.allocation_granularity = 256;
    result.max_entries_scanned_per_bucket = 8;
    return result;
  }

  mem::BufferPool pool_;
};

struct PeerPoolStats {
  std::size_t cached_bytes = 0;
  std::size_t cached_buffers = 0;
  std::size_t live_bytes = 0;
  std::size_t live_buffers = 0;
  std::uint64_t provider_allocations = 0;
  std::uint64_t provider_deallocations = 0;
  std::uint64_t cache_reuses = 0;
};

struct TbbUpstreamStats {
  std::atomic<std::uint64_t> allocations{0};
  std::atomic<std::uint64_t> deallocations{0};
};

template <class T>
class CountingTbbAllocator {
 public:
  using value_type = T;
  template <class U>
  struct rebind {
    using other = CountingTbbAllocator<U>;
  };

  CountingTbbAllocator() noexcept = default;
  explicit CountingTbbAllocator(TbbUpstreamStats* stats) noexcept
      : stats_(stats) {}
  template <class U>
  CountingTbbAllocator(const CountingTbbAllocator<U>& other) noexcept
      : stats_(other.stats()) {}

  T* allocate(std::size_t count) {
    T* result = tbb::scalable_allocator<T>{}.allocate(count);
    if (stats_ != nullptr) {
      stats_->allocations.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
  }
  void deallocate(T* pointer, std::size_t count) noexcept {
    if (stats_ != nullptr) {
      stats_->deallocations.fetch_add(1, std::memory_order_relaxed);
    }
    tbb::scalable_allocator<T>{}.deallocate(pointer, count);
  }
  TbbUpstreamStats* stats() const noexcept { return stats_; }

  template <class U>
  friend bool operator==(const CountingTbbAllocator& left,
                         const CountingTbbAllocator<U>& right) noexcept {
    return left.stats_ == right.stats();
  }
  template <class U>
  friend bool operator!=(const CountingTbbAllocator& left,
                         const CountingTbbAllocator<U>& right) noexcept {
    return !(left == right);
  }

 private:
  template <class>
  friend class CountingTbbAllocator;
  TbbUpstreamStats* stats_ = nullptr;
};

class TbbPoolReplay {
 public:
  struct View {
    const void* resource = nullptr;
    void* native_handle = nullptr;
    std::size_t offset = 0;
    std::size_t size = 0;
  };

  class Lease {
   public:
    Lease() noexcept = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          view_(std::exchange(other.view_, View{})),
          alignment_(std::exchange(other.alignment_, 0)) {}
    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        view_ = std::exchange(other.view_, View{});
        alignment_ = std::exchange(other.alignment_, 0);
      }
      return *this;
    }
    ~Lease() { reset(); }

    View view() const noexcept { return view_; }
    void reset() noexcept {
      if (owner_ == nullptr) return;
      owner_->return_block(view_.native_handle, view_.size, alignment_);
      owner_ = nullptr;
      view_ = {};
      alignment_ = 0;
    }

   private:
    friend class TbbPoolReplay;
    Lease(TbbPoolReplay* owner, void* pointer, std::size_t size,
          std::size_t alignment) noexcept
        : owner_(owner), view_{owner, pointer, 0, size},
          alignment_(alignment) {}

    TbbPoolReplay* owner_ = nullptr;
    View view_{};
    std::size_t alignment_ = 0;
  };

  TbbPoolReplay()
      : pool_(CountingTbbAllocator<std::byte>(&upstream_)) {
    run();
    warm_provider_allocations_ =
        upstream_.allocations.load(std::memory_order_relaxed);
    warm_provider_deallocations_ =
        upstream_.deallocations.load(std::memory_order_relaxed);
    warmed_.store(true, std::memory_order_relaxed);
  }

  void run() {
    std::array<Lease, kBatch> leases{};
    for (std::size_t index = 0; index < kBatch; ++index) {
      auto lease = acquire(kPoolSizes[index], kPoolAlignment);
      if (!lease) std::abort();
      leases[index] = std::move(*lease);
      nb::doNotOptimizeAway(leases[index].view().native_handle);
    }
    for (std::size_t index = kBatch; index != 0; --index) {
      leases[index - 1].reset();
    }
  }

  hyle::Result<Lease> acquire(std::size_t size,
                              std::size_t alignment) noexcept {
    void* pointer = nullptr;
    try {
      pointer = pool_.malloc(size);
    } catch (const std::bad_alloc&) {
      return std::unexpected(
          hyle::Status::ResourceExhausted("oneTBB pool allocation failed"));
    }
    if (pointer == nullptr) {
      return std::unexpected(
          hyle::Status::ResourceExhausted("oneTBB pool allocation failed"));
    }
    live_bytes_.fetch_add(size, std::memory_order_relaxed);
    live_buffers_.fetch_add(1, std::memory_order_relaxed);
    if (warmed_.load(std::memory_order_relaxed)) {
      cached_bytes_.fetch_sub(size, std::memory_order_relaxed);
      cached_buffers_.fetch_sub(1, std::memory_order_relaxed);
      cache_reuses_.fetch_add(1, std::memory_order_relaxed);
    }
    return Lease(this, pointer, size, alignment);
  }

  PeerPoolStats stats() const noexcept {
    return PeerPoolStats{
        .cached_bytes = cached_bytes_.load(std::memory_order_relaxed),
        .cached_buffers = cached_buffers_.load(std::memory_order_relaxed),
        .live_bytes = live_bytes_.load(std::memory_order_relaxed),
        .live_buffers = live_buffers_.load(std::memory_order_relaxed),
        .provider_allocations =
            upstream_.allocations.load(std::memory_order_relaxed),
        .provider_deallocations =
            upstream_.deallocations.load(std::memory_order_relaxed),
        .cache_reuses = cache_reuses_.load(std::memory_order_relaxed),
    };
  }

  bool provider_stable_after_warm() const noexcept {
    return warm_provider_allocations_ ==
               upstream_.allocations.load(std::memory_order_relaxed) &&
           warm_provider_deallocations_ ==
               upstream_.deallocations.load(std::memory_order_relaxed);
  }

 private:
  void return_block(void* pointer, std::size_t size,
                    std::size_t alignment) noexcept {
    (void)alignment;
    pool_.free(pointer);
    live_bytes_.fetch_sub(size, std::memory_order_relaxed);
    live_buffers_.fetch_sub(1, std::memory_order_relaxed);
    cached_bytes_.fetch_add(size, std::memory_order_relaxed);
    cached_buffers_.fetch_add(1, std::memory_order_relaxed);
  }

  TbbUpstreamStats upstream_;
  tbb::memory_pool<CountingTbbAllocator<std::byte>> pool_;
  std::atomic<bool> warmed_{false};
  std::atomic<std::size_t> cached_bytes_{0};
  std::atomic<std::size_t> cached_buffers_{0};
  std::atomic<std::size_t> live_bytes_{0};
  std::atomic<std::size_t> live_buffers_{0};
  std::atomic<std::uint64_t> cache_reuses_{0};
  std::uint64_t warm_provider_allocations_ = 0;
  std::uint64_t warm_provider_deallocations_ = 0;
};

struct PoolPair {
  std::unique_ptr<BufferPoolReplay> hyle;
  std::unique_ptr<TbbPoolReplay> tbb;

  explicit PoolPair(bool peer_allocation_first) {
    if (peer_allocation_first) {
      tbb = std::make_unique<TbbPoolReplay>();
      hyle = std::make_unique<BufferPoolReplay>();
    } else {
      hyle = std::make_unique<BufferPoolReplay>();
      tbb = std::make_unique<TbbPoolReplay>();
    }
  }
};

void validate_contract() {
  HeapPair heaps(false);
  ArenaPair arenas(false);
  BufferViewReplay views;
  PoolPair buffers(false);
  heaps.hyle->run();
  heaps.pmr->run();
  arenas.hyle->run();
  arenas.pmr->run();
  views.hyle_run();
  views.direct_run();
  buffers.hyle->candidate_run();
  buffers.tbb->run();
  const mem::BufferPoolStats stats = buffers.hyle->stats();
  const PeerPoolStats peer_stats = buffers.tbb->stats();
  if (stats.provider_allocations != kBatch ||
      stats.cached_buffers != kBatch || stats.live_buffers != 0 ||
      stats.cache_reuses != kBatch ||
      peer_stats.cached_buffers != kBatch ||
      peer_stats.live_buffers != 0 || peer_stats.cache_reuses != kBatch) {
    std::abort();
  }
  if (!buffers.tbb->provider_stable_after_warm()) std::abort();
  std::fprintf(
      stderr,
      "# HYLE_MEM_V2_CONTRACT schema=1 heap=PASS arena=PASS view=PASS "
      "pool=PASS batch=%zu provider_allocations=%llu cache_reuses=%llu "
      "cached_buffers=%zu live_buffers=%zu peer_provider_allocations=%llu "
      "peer_provider_deallocations=%llu peer_cache_reuses=%llu "
      "peer_cached_buffers=%zu peer_live_buffers=%zu verdict=PASS\n",
      kBatch, static_cast<unsigned long long>(stats.provider_allocations),
      static_cast<unsigned long long>(stats.cache_reuses),
      stats.cached_buffers, stats.live_buffers,
      static_cast<unsigned long long>(peer_stats.provider_allocations),
      static_cast<unsigned long long>(peer_stats.provider_deallocations),
      static_cast<unsigned long long>(peer_stats.cache_reuses),
      peer_stats.cached_buffers, peer_stats.live_buffers);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--validate-contract") == 0) {
    validate_contract();
    return 0;
  }

  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_MEM_V2_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  HeapPair heaps(peer_allocation_first);
  heaps.hyle->run();
  heaps.pmr->run();
  nb::Bench heap = make_bench("isolated heap lifecycle aggregate N=64");
  run_pair(heap, peer_first, "HYLE Heap allocator lifecycle N=64",
           [&] { heaps.hyle->run(); },
           "std pmr erased mimalloc heap lifecycle N=64",
           [&] { heaps.pmr->run(); });

  ArenaPair arenas(peer_allocation_first);
  arenas.hyle->run();
  arenas.pmr->run();
  nb::Bench arena = make_bench("external scratch epoch aggregate N=64");
  run_pair(arena, peer_first, "HYLE FixedArena scratch epoch N=64",
           [&] { arenas.hyle->run(); },
           "std pmr monotonic scratch epoch N=64",
           [&] { arenas.pmr->run(); });

  BufferViewReplay views;
  views.hyle_run();
  views.direct_run();
  nb::Bench view = make_bench("buffer view range fill aggregate N=64");
  run_pair(view, peer_first, "HYLE BufferView subview fill N=64",
           [&] { views.hyle_run(); },
           "direct MemoryResource range fill N=64",
           [&] { views.direct_run(); });

  PoolPair buffers(peer_allocation_first);
  buffers.hyle->candidate_run();
  buffers.tbb->run();
  nb::Bench pool = make_bench("transient buffer lifecycle aggregate N=64");
  run_pair(pool, peer_first, "HYLE BufferPool warmed lifecycle N=64",
           [&] { buffers.hyle->candidate_run(); },
           "oneTBB memory_pool adapted lifecycle N=64",
           [&] { buffers.tbb->run(); });

  const mem::BufferPoolStats stats = buffers.hyle->stats();
  const PeerPoolStats peer_stats = buffers.tbb->stats();
  std::fprintf(
      stderr,
      "# HYLE_MEM_V2_RESOURCE provider_allocations=%llu cache_reuses=%llu "
      "cached_buffers=%zu cached_bytes=%zu live_buffers=%zu live_bytes=%zu "
      "peer_provider_allocations=%llu peer_provider_deallocations=%llu "
      "peer_cache_reuses=%llu peer_cached_buffers=%zu "
      "peer_live_buffers=%zu peer_provider_stable=%d\n",
      static_cast<unsigned long long>(stats.provider_allocations),
      static_cast<unsigned long long>(stats.cache_reuses),
      stats.cached_buffers, stats.cached_bytes, stats.live_buffers,
      stats.live_bytes,
      static_cast<unsigned long long>(peer_stats.provider_allocations),
      static_cast<unsigned long long>(peer_stats.provider_deallocations),
      static_cast<unsigned long long>(peer_stats.cache_reuses),
      peer_stats.cached_buffers, peer_stats.live_buffers,
      buffers.tbb->provider_stable_after_warm() ? 1 : 0);
}
