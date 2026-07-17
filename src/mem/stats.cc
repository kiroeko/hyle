#include "hyle/mem/stats.h"

#include <unordered_map>
#include <utility>

#include "hyle/util/spin_lock.h"

namespace hyle::mem {

// Nested Impl must be completed in this namespace (not an anon ns) for MSVC.
struct LeakTracker::Impl {
  mutable util::SpinLock lock;
  std::unordered_map<void*, std::size_t> live;
};

namespace {

// ---- StatsAllocator vtable state (TU-local) ---------------------------------

struct StatsState {
  Allocator backing;
  AllocStats stats;
  LeakTracker* leaks = nullptr;
  bool owns_leaks = false;
};

void* stats_alloc(void* st, std::size_t size, std::size_t alignment) noexcept {
  auto* s = static_cast<StatsState*>(st);
  void* p = s->backing.allocate(size, alignment);
  if (p != nullptr && size > 0) {
    s->stats.record_alloc(size);
    if (s->leaks) s->leaks->track_alloc(p, size);
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
  if (s->leaks) {
    std::size_t recorded = 0;
    if (s->leaks->track_free(ptr, &recorded)) account = recorded;
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

  if (s->leaks) {
    if (ptr != nullptr) (void)s->leaks->track_free(ptr);
    if (p != nullptr && new_size > 0) s->leaks->track_alloc(p, new_size);
  }
  s->stats.record_realloc(old_size, new_size);
  return p;
}

}  // namespace

// ---- AllocStats::record_realloc (correctness-critical; keep in .cc? header) -
// Implemented inline in header.

// ---- LeakTracker ------------------------------------------------------------

LeakTracker::LeakTracker() : impl_(new Impl) {}
LeakTracker::~LeakTracker() { delete impl_; }

LeakTracker::LeakTracker(LeakTracker&& o) noexcept : impl_(o.impl_) {
  o.impl_ = nullptr;
}
LeakTracker& LeakTracker::operator=(LeakTracker&& o) noexcept {
  if (this != &o) {
    delete impl_;
    impl_ = o.impl_;
    o.impl_ = nullptr;
  }
  return *this;
}

void LeakTracker::track_alloc(void* p, std::size_t size) noexcept {
  if (impl_ == nullptr || p == nullptr || size == 0) return;
  impl_->lock.lock();
  impl_->live[p] = size;
  impl_->lock.unlock();
}

bool LeakTracker::track_free(void* p, std::size_t* out_size) noexcept {
  if (impl_ == nullptr || p == nullptr) return false;
  impl_->lock.lock();
  auto it = impl_->live.find(p);
  if (it == impl_->live.end()) {
    impl_->lock.unlock();
    return false;
  }
  if (out_size != nullptr) *out_size = it->second;
  impl_->live.erase(it);
  impl_->lock.unlock();
  return true;
}

std::size_t LeakTracker::outstanding() const noexcept {
  if (impl_ == nullptr) return 0;
  impl_->lock.lock();
  std::size_t n = impl_->live.size();
  impl_->lock.unlock();
  return n;
}

std::size_t LeakTracker::copy_leaks(LeakEntry* out, std::size_t cap) const noexcept {
  if (impl_ == nullptr) return 0;
  impl_->lock.lock();
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
  impl_->lock.unlock();
  return total;
}

void LeakTracker::clear() noexcept {
  if (impl_ == nullptr) return;
  impl_->lock.lock();
  impl_->live.clear();
  impl_->lock.unlock();
}

// ---- StatsAllocator ---------------------------------------------------------

// State_ pointer stores StatsState*.
StatsAllocator::StatsAllocator(Allocator backing, bool track_leaks) {
  auto* s = new StatsState;
  s->backing = backing;
  if (track_leaks) {
    s->leaks = new LeakTracker();
    s->owns_leaks = true;
  }
  state_ = reinterpret_cast<State*>(s);
}

StatsAllocator::~StatsAllocator() {
  if (state_ == nullptr) return;
  auto* s = reinterpret_cast<StatsState*>(state_);
  if (s->owns_leaks) delete s->leaks;
  delete s;
  state_ = nullptr;
}

StatsAllocator::StatsAllocator(StatsAllocator&& o) noexcept : state_(o.state_) {
  o.state_ = nullptr;
}

StatsAllocator& StatsAllocator::operator=(StatsAllocator&& o) noexcept {
  if (this != &o) {
    if (state_ != nullptr) {
      auto* s = reinterpret_cast<StatsState*>(state_);
      if (s->owns_leaks) delete s->leaks;
      delete s;
    }
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
  return state_ ? reinterpret_cast<StatsState*>(state_)->leaks : nullptr;
}

const LeakTracker* StatsAllocator::leaks() const noexcept {
  return state_ ? reinterpret_cast<const StatsState*>(state_)->leaks : nullptr;
}

}  // namespace hyle::mem
