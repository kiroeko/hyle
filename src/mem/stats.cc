#include "hyle/mem/stats.h"

#include <limits>
#include <memory_resource>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "hyle/mem/pmr.h"

namespace hyle::mem {

// Nested Impl must be completed in this namespace (not an anon ns) for MSVC.
struct LeakTracker::Impl {
  explicit Impl(Allocator allocator)
      : backing(allocator), resource(allocator), live(&resource) {}

  Allocator backing;
  // unordered_map insert/erase may allocate, rehash and walk buckets. This is
  // diagnostic code, not a few-instruction spin-lock critical section.
  mutable std::mutex lock;
  PmrResource resource;
  std::pmr::unordered_map<void*, std::size_t> live;
  std::uint64_t dropped_records = 0;
};

namespace {

// ---- StatsAllocator vtable state (TU-local) ---------------------------------

struct StatsState {
  StatsState(Allocator b, Allocator m, LeakTracker&& tracker,
             bool track) noexcept
      : backing(b), metadata(m), leaks(std::move(tracker)),
        track_leaks(track) {}

  Allocator backing;
  Allocator metadata;
  AllocStats stats;
  LeakTracker leaks;
  bool track_leaks = false;
};

void* stats_alloc(void* st, std::size_t size, std::size_t alignment) noexcept {
  auto* s = static_cast<StatsState*>(st);
  void* p = s->backing.allocate(size, alignment);
  if (p != nullptr && size > 0) {
    s->stats.record_alloc(size);
    if (s->track_leaks) s->leaks.track_alloc(p, size);
  }
  return p;
}

void stats_free(void* st, void* ptr, std::size_t size,
                std::size_t alignment) noexcept {
  auto* s = static_cast<StatsState*>(st);
  if (ptr == nullptr) {
    s->backing.deallocate(ptr, size, alignment);
    return;
  }
  // When leak tracking is on, account the free by the size RECORDED at alloc
  // time (robust to a caller passing the wrong size to deallocate). Falls back
  // to the caller's size for untracked ptrs / when tracking is off.
  std::size_t account = size;
  if (s->track_leaks) {
    std::size_t recorded = 0;
    if (s->leaks.track_free(ptr, &recorded)) account = recorded;
  }
  if (account > 0) s->stats.record_free(account);
  s->backing.deallocate(ptr, size, alignment);
}

void* stats_realloc(void* st, void* ptr, std::size_t old_size,
                    std::size_t new_size, std::size_t alignment) noexcept {
  auto* s = static_cast<StatsState*>(st);
  void* p = s->backing.reallocate(ptr, old_size, new_size, alignment);
  // OOM on grow: original block intact, no accounting change.
  if (new_size > 0 && p == nullptr) return nullptr;

  std::size_t accounted_old_size = old_size;
  if (s->track_leaks) {
    if (ptr != nullptr) {
      std::size_t recorded = 0;
      if (s->leaks.track_free(ptr, &recorded)) {
        accounted_old_size = recorded;
      }
    }
    if (p != nullptr && new_size > 0) s->leaks.track_alloc(p, new_size);
  }
  s->stats.record_realloc(accounted_old_size, new_size);
  return p;
}

}  // namespace

// ---- AllocStats::record_realloc (correctness-critical; keep in .cc? header) -
// Implemented inline in header.

// ---- LeakTracker ------------------------------------------------------------

Result<LeakTracker> LeakTracker::create(Allocator metadata) noexcept {
  if (!metadata.valid()) {
    return std::unexpected(
        Status::InvalidArgument("LeakTracker create: invalid metadata"));
  }
  void* memory = metadata.allocate(sizeof(Impl), alignof(Impl));
  if (memory == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "LeakTracker create: metadata allocation failed"));
  }
  LeakTracker tracker;
  try {
    tracker.impl_ = ::new (memory) Impl(metadata);
  } catch (...) {
    metadata.deallocate(memory, sizeof(Impl), alignof(Impl));
    return std::unexpected(Status::ResourceExhausted(
        "LeakTracker create: map construction failed"));
  }
  return tracker;
}

LeakTracker::~LeakTracker() { destroy(); }

LeakTracker::LeakTracker(LeakTracker&& o) noexcept : impl_(o.impl_) {
  o.impl_ = nullptr;
}
LeakTracker& LeakTracker::operator=(LeakTracker&& o) noexcept {
  if (this != &o) {
    destroy();
    impl_ = o.impl_;
    o.impl_ = nullptr;
  }
  return *this;
}

void LeakTracker::destroy() noexcept {
  if (impl_ == nullptr) return;
  const Allocator backing = impl_->backing;
  impl_->~Impl();
  backing.deallocate(impl_, sizeof(Impl), alignof(Impl));
  impl_ = nullptr;
}

void LeakTracker::track_alloc(void* p, std::size_t size) noexcept {
  if (impl_ == nullptr || p == nullptr || size == 0) return;
  std::lock_guard<std::mutex> lock(impl_->lock);
  // Leak tracking is diagnostic and runs inside Allocator's noexcept vtable.
  // Hash-table growth may throw on OOM; losing one debug record is preferable
  // to terminating an otherwise successful allocation.
  try {
    impl_->live[p] = size;
  } catch (...) {
    if (impl_->dropped_records !=
        (std::numeric_limits<std::uint64_t>::max)()) {
      ++impl_->dropped_records;
    }
  }
}

bool LeakTracker::track_free(void* p, std::size_t* out_size) noexcept {
  if (impl_ == nullptr || p == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->lock);
  auto it = impl_->live.find(p);
  if (it == impl_->live.end()) return false;
  if (out_size != nullptr) *out_size = it->second;
  impl_->live.erase(it);
  return true;
}

std::size_t LeakTracker::outstanding() const noexcept {
  if (impl_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(impl_->lock);
  return impl_->live.size();
}

std::uint64_t LeakTracker::dropped_records() const noexcept {
  if (impl_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(impl_->lock);
  return impl_->dropped_records;
}

std::size_t LeakTracker::copy_leaks(LeakEntry* out, std::size_t cap) const noexcept {
  if (impl_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(impl_->lock);
  const std::size_t total = impl_->live.size();
  std::size_t i = 0;
  if (out != nullptr && cap > 0) {
    for (const auto& kv : impl_->live) {
      if (i >= cap) break;
      out[i].ptr = kv.first;
      out[i].size = kv.second;
      ++i;
    }
  }
  return total;
}

void LeakTracker::clear() noexcept {
  if (impl_ == nullptr) return;
  std::lock_guard<std::mutex> lock(impl_->lock);
  impl_->live.clear();
  impl_->dropped_records = 0;
}

// ---- StatsAllocator ---------------------------------------------------------

// State_ pointer stores StatsState*.
Result<StatsAllocator> StatsAllocator::create(Allocator backing,
                                             bool track_leaks,
                                             Allocator metadata) noexcept {
  if (!backing.valid() || !metadata.valid()) {
    return std::unexpected(
        Status::InvalidArgument(
            "StatsAllocator create: invalid backing or metadata"));
  }

  LeakTracker tracker;
  if (track_leaks) {
    auto result = LeakTracker::create(metadata);
    if (!result) return std::unexpected(std::move(result).error());
    tracker = *std::move(result);
  }

  void* memory = metadata.allocate(sizeof(StatsState), alignof(StatsState));
  if (memory == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "StatsAllocator create: state allocation failed"));
  }
  StatsAllocator out;
  out.state_ = reinterpret_cast<State*>(
      ::new (memory)
          StatsState(backing, metadata, std::move(tracker), track_leaks));
  return out;
}

StatsAllocator::~StatsAllocator() { destroy(); }

void StatsAllocator::destroy() noexcept {
  if (state_ == nullptr) return;
  auto* s = reinterpret_cast<StatsState*>(state_);
  const Allocator metadata = s->metadata;
  s->~StatsState();
  metadata.deallocate(s, sizeof(StatsState), alignof(StatsState));
  state_ = nullptr;
}

StatsAllocator::StatsAllocator(StatsAllocator&& o) noexcept : state_(o.state_) {
  o.state_ = nullptr;
}

StatsAllocator& StatsAllocator::operator=(StatsAllocator&& o) noexcept {
  if (this != &o) {
    destroy();
    state_ = o.state_;
    o.state_ = nullptr;
  }
  return *this;
}

Allocator StatsAllocator::get() const noexcept {
  HYLE_FATAL_ASSERT(state_ != nullptr);
  return Allocator(&stats_alloc, &stats_free, &stats_realloc,
                   reinterpret_cast<StatsState*>(state_));
}

AllocStats& StatsAllocator::stats() noexcept {
  HYLE_FATAL_ASSERT(state_ != nullptr);
  return reinterpret_cast<StatsState*>(state_)->stats;
}

const AllocStats& StatsAllocator::stats() const noexcept {
  HYLE_FATAL_ASSERT(state_ != nullptr);
  return reinterpret_cast<StatsState*>(state_)->stats;
}

LeakTracker* StatsAllocator::leaks() noexcept {
  if (state_ == nullptr) return nullptr;
  auto* s = reinterpret_cast<StatsState*>(state_);
  return s->track_leaks ? &s->leaks : nullptr;
}

const LeakTracker* StatsAllocator::leaks() const noexcept {
  if (state_ == nullptr) return nullptr;
  const auto* s = reinterpret_cast<const StatsState*>(state_);
  return s->track_leaks ? &s->leaks : nullptr;
}

}  // namespace hyle::mem
