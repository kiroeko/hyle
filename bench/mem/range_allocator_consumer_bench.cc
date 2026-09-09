#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>

#include <offsetAllocator.hpp>

#include "hyle/mem/range_allocator.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;
namespace mem = hyle::mem;

constexpr std::size_t kLiveRanges = 256;
constexpr std::size_t kOperations = 512;
constexpr std::uint32_t kCapacity = 512u << 20;

constexpr std::array<std::uint32_t, kOperations> make_sizes() noexcept {
  std::array<std::uint32_t, kOperations> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    // Resource-like powers from 256 B to 1 MiB, with a deterministic
    // non-power-of-two tail so the trace exercises size classes rather than a
    // benchmark-specific exact-bin loop.
    const unsigned exponent =
        8u + static_cast<unsigned>((index * 7u + index / 11u) % 13u);
    result[index] = (std::uint32_t{1} << exponent) +
                    static_cast<std::uint32_t>((index * 131u) & 255u);
  }
  return result;
}

constexpr std::array<std::uint16_t, kOperations> make_slots() noexcept {
  std::array<std::uint16_t, kOperations> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] =
        static_cast<std::uint16_t>((index * 73u + index / 17u) % kLiveRanges);
  }
  return result;
}

constexpr auto kSizes = make_sizes();
constexpr auto kSlots = make_slots();

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

class HyleRangeReplay {
 public:
  HyleRangeReplay()
      : allocator_(mem::RangeAllocator::create(
                       kCapacity, {.max_allocations = kLiveRanges})
                       .value()) {
    for (std::size_t index = 0; index < allocations_.size(); ++index) {
      allocations_[index] = allocator_.allocate(kSizes[index]).value();
    }
  }

  void run() noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t operation = 0; operation < kOperations; ++operation) {
      const std::size_t slot = kSlots[operation];
      allocations_[slot].reset();
      auto next = allocator_.allocate(kSizes[operation]);
      if (!next) std::abort();
      checksum += next->offset();
      allocations_[slot] = std::move(*next);
    }
    nb::doNotOptimizeAway(checksum);
  }

  mem::RangeAllocatorStats stats() const noexcept { return allocator_.stats(); }

 private:
  mem::RangeAllocator allocator_;
  std::array<mem::RangeAllocation, kLiveRanges> allocations_{};
};

struct OffsetControl {
  OffsetControl(std::uint32_t capacity, std::uint32_t max_allocations)
      : allocator(capacity, 2 * max_allocations + 1),
        capacity_bytes(capacity),
        max_live_allocations(max_allocations) {}

  OffsetAllocator::Allocator allocator;
  std::uint64_t capacity_bytes = 0;
  std::uint64_t allocated_bytes = 0;
  std::uint64_t peak_allocated_bytes = 0;
  std::uint64_t generation = 1;
  std::uint32_t references = 1;
  std::uint32_t live_allocations = 0;
  std::uint32_t max_live_allocations = 0;
  bool owner_alive = true;
};

void offset_release_reference(OffsetControl* control) noexcept {
  if (--control->references == 0) delete control;
}

class OffsetLease {
 public:
  OffsetLease() noexcept = default;
  OffsetLease(const OffsetLease&) = delete;
  OffsetLease& operator=(const OffsetLease&) = delete;
  OffsetLease(OffsetLease&& other) noexcept
      : control_(other.control_),
        generation_(other.generation_),
        offset_(other.offset_),
        size_(other.size_),
        metadata_(other.metadata_) {
    other.control_ = nullptr;
  }
  OffsetLease& operator=(OffsetLease&& other) noexcept {
    if (this != &other) {
      if (control_ != nullptr) reset();
      control_ = other.control_;
      generation_ = other.generation_;
      offset_ = other.offset_;
      size_ = other.size_;
      metadata_ = other.metadata_;
      other.control_ = nullptr;
    }
    return *this;
  }
  ~OffsetLease() {
    if (control_ != nullptr) reset();
  }

  bool valid() const noexcept {
    return control_ != nullptr && control_->owner_alive &&
           generation_ == control_->generation;
  }
  explicit operator bool() const noexcept { return valid(); }
  std::uint64_t offset() const noexcept { return offset_; }
  std::uint64_t size() const noexcept { return size_; }

  void reset() noexcept {
    if (control_ == nullptr) return;
    OffsetControl* control = control_;
    if (control->owner_alive && generation_ == control->generation) {
      control->allocator.free(
          {.offset = static_cast<std::uint32_t>(offset_),
           .metadata = metadata_});
      control->allocated_bytes -= size_;
      --control->live_allocations;
    }
    offset_release_reference(control);
    control_ = nullptr;
  }

 private:
  friend class OffsetSafeAllocator;
  OffsetLease(OffsetControl* control, std::uint64_t generation,
              std::uint32_t offset, std::uint32_t size,
              OffsetAllocator::NodeIndex metadata) noexcept
      : control_(control),
        generation_(generation),
        offset_(offset),
        size_(size),
        metadata_(metadata) {}

  OffsetControl* control_ = nullptr;
  std::uint64_t generation_ = 0;
  std::uint64_t offset_ = 0;
  std::uint32_t size_ = 0;
  OffsetAllocator::NodeIndex metadata_ = 0;
};

static_assert(sizeof(OffsetLease) == sizeof(mem::RangeAllocation));

class OffsetSafeAllocator {
 public:
  OffsetSafeAllocator(std::uint32_t capacity, std::uint32_t max_allocations)
      : control_(new OffsetControl(capacity, max_allocations)) {}
  OffsetSafeAllocator(const OffsetSafeAllocator&) = delete;
  OffsetSafeAllocator& operator=(const OffsetSafeAllocator&) = delete;
  ~OffsetSafeAllocator() {
    control_->owner_alive = false;
    offset_release_reference(control_);
  }

  hyle::Result<OffsetLease> allocate(std::uint64_t size,
                                     std::uint64_t alignment = 1) noexcept {
    if (size == 0 ||
        size > (std::numeric_limits<std::uint32_t>::max)() ||
        alignment != 1) {
      return std::unexpected(hyle::Status::InvalidArgument());
    }
    if (control_->live_allocations == control_->max_live_allocations) {
      return std::unexpected(hyle::Status::ResourceExhausted());
    }
    const auto allocation =
        control_->allocator.allocate(static_cast<std::uint32_t>(size));
    if (allocation.offset == OffsetAllocator::Allocation::NO_SPACE) {
      return std::unexpected(hyle::Status::ResourceExhausted());
    }
    control_->allocated_bytes += size;
    control_->peak_allocated_bytes =
        (std::max)(control_->peak_allocated_bytes,
                   control_->allocated_bytes);
    ++control_->live_allocations;
    ++control_->references;
    return OffsetLease(control_, control_->generation, allocation.offset,
                       static_cast<std::uint32_t>(size),
                       allocation.metadata);
  }

  void reset() noexcept {
    ++control_->generation;
    control_->allocator.reset();
    control_->allocated_bytes = 0;
    control_->live_allocations = 0;
  }

  mem::RangeAllocatorStats stats() const noexcept {
    return {.capacity_bytes = control_->capacity_bytes,
            .allocated_bytes = control_->allocated_bytes,
            .free_bytes =
                control_->capacity_bytes - control_->allocated_bytes,
            .peak_allocated_bytes = control_->peak_allocated_bytes,
            .outstanding_handles = control_->references - 1u,
            .live_allocations = control_->live_allocations,
            .max_allocations = control_->max_live_allocations};
  }

 private:
  OffsetControl* control_;
};

class OffsetRangeReplay {
 public:
  OffsetRangeReplay() : allocator_(kCapacity, kLiveRanges) {
    for (std::size_t index = 0; index < allocations_.size(); ++index) {
      allocations_[index] = allocator_.allocate(kSizes[index]).value();
    }
  }

  void run() noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t operation = 0; operation < kOperations; ++operation) {
      const std::size_t slot = kSlots[operation];
      allocations_[slot].reset();
      auto next = allocator_.allocate(kSizes[operation]);
      if (!next) std::abort();
      checksum += next->offset();
      allocations_[slot] = std::move(*next);
    }
    nb::doNotOptimizeAway(checksum);
  }

  mem::RangeAllocatorStats stats() const noexcept { return allocator_.stats(); }

 private:
  OffsetSafeAllocator allocator_;
  std::array<OffsetLease, kLiveRanges> allocations_{};
};

struct ReplayPair {
  std::unique_ptr<HyleRangeReplay> hyle;
  std::unique_ptr<OffsetRangeReplay> peer;

  explicit ReplayPair(bool peer_allocation_first) {
    if (peer_allocation_first) {
      peer = std::make_unique<OffsetRangeReplay>();
      hyle = std::make_unique<HyleRangeReplay>();
    } else {
      hyle = std::make_unique<HyleRangeReplay>();
      peer = std::make_unique<OffsetRangeReplay>();
    }
  }
};

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first, Candidate&& candidate,
              Peer&& peer) {
  if (peer_first) {
    bench.run("OffsetAllocator adapted mixed range churn N=512",
              std::forward<Peer>(peer));
    bench.run("HYLE RangeAllocator mixed range churn N=512",
              std::forward<Candidate>(candidate));
  } else {
    bench.run("HYLE RangeAllocator mixed range churn N=512",
              std::forward<Candidate>(candidate));
    bench.run("OffsetAllocator adapted mixed range churn N=512",
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

  ReplayPair ranges(peer_allocation_first);
  ranges.hyle->run();
  ranges.peer->run();
  const auto stats = ranges.hyle->stats();
  const auto peer_stats = ranges.peer->stats();
  if (stats.capacity_bytes != peer_stats.capacity_bytes ||
      stats.allocated_bytes != peer_stats.allocated_bytes ||
      stats.free_bytes != peer_stats.free_bytes ||
      stats.peak_allocated_bytes != peer_stats.peak_allocated_bytes ||
      stats.outstanding_handles != peer_stats.outstanding_handles ||
      stats.live_allocations != peer_stats.live_allocations ||
      stats.max_allocations != peer_stats.max_allocations) {
    std::abort();
  }
  std::fprintf(
      stderr,
      "# HYLE_MEM_V7_RANGE_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s run_nonce=%s capacity=%u live=%zu operations=%zu "
      "hyle_metadata_bytes=%llu hyle_handle_bytes=%zu "
      "offset_handle_bytes=%zu offset_allocator_commit=3610a7377088\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce(), kCapacity, kLiveRanges, kOperations,
      static_cast<unsigned long long>(stats.metadata_bytes),
      sizeof(mem::RangeAllocation), sizeof(OffsetLease));

  nb::Bench bench;
  bench.title("logical range mixed resource churn aggregate N=512")
      .relative(true)
      .batch(kOperations)
      .minEpochIterations(64)
      .minEpochTime(std::chrono::milliseconds(50));
  run_pair(bench, peer_first, [&] { ranges.hyle->run(); },
           [&] { ranges.peer->run(); });
}
