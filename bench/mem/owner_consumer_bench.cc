#include <nanobench.h>

#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/mem/region.h"
#include "hyle/mem/system_inline.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#include "../common/perf_process.h"

// Defined in a separate translation unit and built without LTCG. Returning
// the peer through the standard erased interface prevents this benchmark TU
// from devirtualizing oneTBB's concrete scalable_resource_impl while HYLE's
// runtime Allocator still pays its function-pointer dispatch.
std::pmr::memory_resource* tbb_runtime_resource() noexcept;

namespace {
namespace nb = ankerl::nanobench;
namespace bpmr = boost::container::pmr;

constexpr std::size_t kAllocations = 64;
constexpr std::size_t kFibers = 16;
constexpr std::size_t kTailSamples = 256;

struct TailLatency {
  std::uint64_t p50_ns = 0;
  std::uint64_t p95_ns = 0;
  std::uint64_t p99_ns = 0;
  std::uint64_t max_ns = 0;
};

template <class Function>
std::uint64_t elapsed_ns(Function&& function) {
  const auto begin = std::chrono::steady_clock::now();
  function();
  const auto end = std::chrono::steady_clock::now();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

TailLatency summarize_tail(
    std::array<std::uint64_t, kTailSamples> samples) {
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](std::size_t numerator) {
    const std::size_t index =
        (samples.size() * numerator + 99u) / 100u - 1u;
    return samples[index];
  };
  return {percentile(50), percentile(95), percentile(99), samples.back()};
}

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
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
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
      // One operation is one complete consumer batch. This avoids claiming
      // sub-nanosecond precision for an individual allocator instruction.
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

constexpr std::size_t allocation_size(std::size_t index) noexcept {
  constexpr std::array<std::size_t, 8> sizes{
      24, 40, 72, 128, 320, 1024, 4096, 16384};
  return sizes[(index * 5u + 3u) % sizes.size()];
}

constexpr std::size_t allocation_alignment(std::size_t index) noexcept {
  constexpr std::array<std::size_t, 4> alignments{16, 16, 32, 64};
  return alignments[index % alignments.size()];
}

inline void observe_allocation(void* pointer, std::size_t bytes,
                               std::size_t index) noexcept {
  if (pointer == nullptr) std::abort();
  auto* data = static_cast<std::byte*>(pointer);
  data[0] = static_cast<std::byte>((index + 1u) & 0xffu);
  data[bytes - 1] = static_cast<std::byte>((index * 17u + 3u) & 0xffu);
  nb::doNotOptimizeAway(data[0]);
  nb::doNotOptimizeAway(data[bytes - 1]);
}

template <class Allocate, class Deallocate>
void allocation_batch(Allocate&& allocate, Deallocate&& deallocate) {
  std::array<void*, kAllocations> pointers{};
  for (std::size_t index = 0; index < pointers.size(); ++index) {
    const std::size_t bytes = allocation_size(index);
    const std::size_t alignment = allocation_alignment(index);
    pointers[index] = allocate(bytes, alignment);
    observe_allocation(pointers[index], bytes, index);
  }
  for (std::size_t index = pointers.size(); index-- != 0;) {
    deallocate(pointers[index], allocation_size(index),
               allocation_alignment(index));
  }
}

class BoostAllocatorResource final : public bpmr::memory_resource {
 public:
  explicit BoostAllocatorResource(hyle::mem::Allocator allocator) noexcept
      : allocator_(allocator) {}

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* pointer = allocator_.allocate(bytes, alignment);
    if (pointer == nullptr) throw std::bad_alloc{};
    return pointer;
  }

  void do_deallocate(void* pointer, std::size_t bytes,
                     std::size_t alignment) noexcept override {
    allocator_.deallocate(pointer, bytes, alignment);
  }

  bool do_is_equal(const bpmr::memory_resource& other) const
      noexcept override {
    return this == &other;
  }

  hyle::mem::Allocator allocator_;
};

constexpr std::size_t region_size(std::size_t index) noexcept {
  constexpr std::array<std::size_t, 8> sizes{
      24, 64, 128, 320, 1024, 2048, 4096, 8192};
  // One dedicated large allocation exercises the long-lived owner's release
  // path; the other 63 allocations exercise ordinary retained blocks.
  return index + 1u == kAllocations ? 64u * 1024u
                                    : sizes[(index * 3u) % sizes.size()];
}

template <class Reset, class Allocate>
void region_epoch(Reset&& reset, Allocate&& allocate) {
  reset();
  for (std::size_t index = 0; index < kAllocations; ++index) {
    const std::size_t bytes = region_size(index);
    void* pointer = allocate(bytes, alignof(std::max_align_t));
    observe_allocation(pointer, bytes, index);
  }
}

struct FiberWork {
  std::uint64_t value = 0;
};

void fiber_entry(void* argument) noexcept {
  auto* work = static_cast<FiberWork*>(argument);
  work->value = (reinterpret_cast<std::uintptr_t>(argument) >> 4u) | 1u;
}

class FiberLifecycle {
 public:
  FiberLifecycle()
      : scheduler_(1, hyle::mem::system_allocator(), 4096),
        pool_(scheduler_.fiber_stack_pool(hyle::mem::kFiberStackDefault)) {
    if (pool_ == nullptr) std::abort();
    fibers_.reserve(kFibers);
  }

  ~FiberLifecycle() { scheduler_.shutdown(); }

  void run(bool pooled) {
    fibers_.clear();
    for (FiberWork& item : work_) item.value = 0;

    for (std::size_t index = 0; index < kFibers; ++index) {
      auto acquired = pooled
                          ? pool_->acquire()
                          : hyle::mem::FiberStack::allocate(
                                hyle::mem::kFiberStackDefault);
      if (!acquired) std::abort();
      hyle::mem::FiberStack stack = *std::move(acquired);
      fibers_.emplace_back(std::move(stack), &fiber_entry, &work_[index],
                           pooled ? pool_ : nullptr);
      if (!scheduler_.spawn(&fibers_.back())) std::abort();
    }
    scheduler_.drain();

    std::uint64_t checksum = 0;
    for (const FiberWork& item : work_) checksum += item.value;
    if (checksum == 0) std::abort();
    nb::doNotOptimizeAway(checksum);
    // Fiber destruction is charged: it either returns the guarded stack to
    // the scheduler-owned pool or releases the VM mapping directly.
    fibers_.clear();
  }

  std::size_t cached() const noexcept { return pool_->cached(); }
  std::size_t capacity() const noexcept { return pool_->capacity(); }

 private:
  hyle::sched::Scheduler scheduler_;
  hyle::mem::FiberStackPool* pool_ = nullptr;
  std::vector<hyle::sched::Fiber> fibers_;
  std::array<FiberWork, kFibers> work_{};
};

}  // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_MEM_OWNER_CONSUMER_RUN execution_order=%s apple_qos=%s "
      "qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  const hyle::mem::Allocator runtime = hyle::mem::system_allocator();
  auto runtime_batch = [&] {
    allocation_batch(
        [&](std::size_t bytes, std::size_t alignment) {
          return runtime.allocate(bytes, alignment);
        },
        [&](void* pointer, std::size_t bytes, std::size_t alignment) {
          runtime.deallocate(pointer, bytes, alignment);
        });
  };
  auto direct_batch = [&] {
    allocation_batch(
        [](std::size_t bytes, std::size_t alignment) {
          return hyle::mem::SystemAllocator::allocate(bytes, alignment);
        },
        [](void* pointer, std::size_t bytes, std::size_t alignment) {
          hyle::mem::SystemAllocator::deallocate(pointer, bytes, alignment);
        });
  };
  std::pmr::memory_resource* const tbb_runtime = tbb_runtime_resource();
  if (tbb_runtime == nullptr) std::abort();
  auto tbb_batch = [&] {
    allocation_batch(
        [&](std::size_t bytes, std::size_t alignment) {
          return tbb_runtime->allocate(bytes, alignment);
        },
        [&](void* pointer, std::size_t bytes, std::size_t alignment) {
          tbb_runtime->deallocate(pointer, bytes, alignment);
        });
  };
  runtime_batch();
  direct_batch();
  tbb_batch();

  nb::Bench direct = make_bench(
      "runtime allocator consumer aggregate N=64 versus direct");
  run_pair(direct, peer_first, "HYLE runtime Allocator batch N=64",
           runtime_batch, "HYLE SystemAllocator direct batch N=64",
           direct_batch);

  nb::Bench tbb = make_bench(
      "runtime allocator consumer aggregate N=64 versus oneTBB");
  run_pair(tbb, peer_first, "HYLE runtime Allocator batch N=64",
           runtime_batch, "oneTBB scalable memory_resource batch N=64",
           tbb_batch);

  hyle::mem::RegionOptions options;
  // Match HttpServeOptions::request_region: long-lived keep-alive owners start
  // small, grow ordinary blocks to 64 KiB, and return >=64 KiB outliers on
  // every clear rather than retaining one pathological request forever.
  options.initial_block_size = 4u * 1024u;
  options.max_block_size = 64u * 1024u;
  options.large_allocation_threshold = 64u * 1024u;
  hyle::mem::Region region(runtime, options);
  BoostAllocatorResource boost_upstream(runtime);
  bpmr::monotonic_buffer_resource boost_region(options.initial_block_size,
                                                &boost_upstream);
  auto hyle_region_epoch = [&] {
    region_epoch([&] { region.clear(); },
                 [&](std::size_t bytes, std::size_t alignment) {
                   return region.allocate(bytes, alignment);
                 });
  };
  auto boost_region_epoch = [&] {
    region_epoch([&] { boost_region.release(); },
                 [&](std::size_t bytes, std::size_t alignment) {
                   return boost_region.allocate(bytes, alignment);
                 });
  };
  hyle_region_epoch();
  boost_region_epoch();

  nb::Bench region_bench = make_bench(
      "request region epoch aggregate N=64");
  run_pair(region_bench, peer_first, "HYLE Region clear/reuse N=64",
           hyle_region_epoch, "Boost monotonic release N=64",
           boost_region_epoch);

  FiberLifecycle fibers;
  for (int index = 0; index < 8; ++index) {
    fibers.run(true);
    fibers.run(false);
  }
  if (fibers.cached() < kFibers || fibers.cached() > fibers.capacity()) {
    std::abort();
  }
  nb::Bench fiber_bench = make_bench(
      "scheduler guarded fiber lifecycle aggregate N=16");
  run_pair(fiber_bench, peer_first,
           "HYLE scheduler pooled guarded stacks N=16",
           [&] { fibers.run(true); },
           "HYLE scheduler direct guarded VM stacks N=16",
           [&] { fibers.run(false); });

  std::array<std::uint64_t, kTailSamples> pooled_tail{};
  std::array<std::uint64_t, kTailSamples> direct_tail{};
  for (std::size_t index = 0; index < kTailSamples; ++index) {
    if (peer_first) {
      direct_tail[index] = elapsed_ns([&] { fibers.run(false); });
      pooled_tail[index] = elapsed_ns([&] { fibers.run(true); });
    } else {
      pooled_tail[index] = elapsed_ns([&] { fibers.run(true); });
      direct_tail[index] = elapsed_ns([&] { fibers.run(false); });
    }
  }
  const TailLatency pooled_latency = summarize_tail(pooled_tail);
  const TailLatency direct_latency = summarize_tail(direct_tail);

  std::fprintf(stderr,
               "# HYLE_MEM_OWNER_RESOURCE fiber_cached=%zu "
               "fiber_capacity=%zu region_reserved=%zu region_peak_used=%zu "
               "fiber_pooled_p50_ns=%llu fiber_pooled_p95_ns=%llu "
               "fiber_pooled_p99_ns=%llu fiber_pooled_max_ns=%llu "
               "fiber_direct_p50_ns=%llu fiber_direct_p95_ns=%llu "
               "fiber_direct_p99_ns=%llu fiber_direct_max_ns=%llu\n",
               fibers.cached(), fibers.capacity(), region.reserved(),
               region.peak_used(),
               static_cast<unsigned long long>(pooled_latency.p50_ns),
               static_cast<unsigned long long>(pooled_latency.p95_ns),
               static_cast<unsigned long long>(pooled_latency.p99_ns),
               static_cast<unsigned long long>(pooled_latency.max_ns),
               static_cast<unsigned long long>(direct_latency.p50_ns),
               static_cast<unsigned long long>(direct_latency.p95_ns),
               static_cast<unsigned long long>(direct_latency.p99_ns),
               static_cast<unsigned long long>(direct_latency.max_ns));
}
