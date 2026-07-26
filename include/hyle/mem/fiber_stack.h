#pragma once

#include <cstddef>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/platform/virtual_memory.h"
#include "hyle/util/spin_lock.h"

// Fiber stack storage (mem-design fiber_stack): page-aligned usable region + a
// permanent low guard page (Protect::kNone). Built on platform::VmRegion::
// reserve_guarded. Stack grows down: put SP at top() and grow toward base()/guard.
//
// Tier sizes (sched-design §1): small for shallow io fibers, default for common
// work, large for deep compute/recursion. Callers pick a tier; the OS rounds
// usable up to a page and always adds one guard page below.

namespace hyle::mem {

// Usable bytes (guard page is EXTRA, not counted here).
inline constexpr std::size_t kFiberStackSmall = 16 * 1024;
inline constexpr std::size_t kFiberStackDefault = 64 * 1024;
inline constexpr std::size_t kFiberStackLarge = 256 * 1024;

// One fiber stack: move-only owner of a guarded VmRegion.
class FiberStack {
 public:
  FiberStack() noexcept = default;

  // Reserve + commit `usable` (page-rounded) RW bytes with a low guard page.
  static Result<FiberStack> allocate(std::size_t usable_bytes) noexcept {
    FiberStack s;
    HYLE_ASSIGN_OR_RETURN(s.region_,
                          platform::VmRegion::reserve_guarded(usable_bytes));
    return s;
  }

  FiberStack(FiberStack&& o) noexcept : region_(std::move(o.region_)) {}
  FiberStack& operator=(FiberStack&& o) noexcept {
    if (this != &o) region_ = std::move(o.region_);
    return *this;
  }
  HYLE_DISABLE_COPY(FiberStack);

  bool valid() const noexcept { return region_.valid(); }
  explicit operator bool() const noexcept { return valid(); }

  // First RW byte (above the guard page). Stack grows down toward this.
  void* base() const noexcept { return region_.usable(); }
  // Usable RW bytes (page-aligned; may be > requested usable).
  std::size_t size() const noexcept { return region_.usable_size(); }
  // Highest address of the usable range (exclusive). make_fcontext wants this.
  void* top() const noexcept {
    return static_cast<char*>(base()) + size();
  }

  bool has_guard() const noexcept { return region_.has_guard(); }
  // Low address of the permanent no-access guard page (1 page).
  void* guard() const noexcept { return region_.data(); }
  std::size_t total_reserved() const noexcept { return region_.size(); }

  // Early release (also done by the destructor via VmRegion).
  Status reset() noexcept { return region_.release(); }

 private:
  platform::VmRegion region_;
};

// Bounded free-list of same-tier stacks. Thread-safe (SpinLock). Caps cached
// stacks so idle processes do not retain unbounded address space.
//
// Contract: every release() must pass a stack whose size() matches the pool's
// usable tier (page-rounded). Mismatched sizes are dropped (destroyed), not cached.
class FiberStackPool {
 public:
  static constexpr std::size_t kDefaultCacheCap = 64;
  static constexpr std::size_t kMaxCacheCap = 256;

  explicit FiberStackPool(std::size_t usable_tier,
                          std::size_t cache_cap = kDefaultCacheCap) noexcept
      : usable_tier_(usable_tier),
        cache_cap_(cache_cap < kMaxCacheCap ? cache_cap : kMaxCacheCap) {}

  HYLE_DISABLE_COPY(FiberStackPool);

  std::size_t usable_tier() const noexcept { return usable_tier_; }
  // The effective retention bound. A requested cap above the fixed inline
  // storage is clamped; zero intentionally disables caching.
  std::size_t capacity() const noexcept { return cache_cap_; }
  std::size_t cached() const noexcept {
    lock_.lock();
    std::size_t n = n_free_;
    lock_.unlock();
    return n;
  }

  // Pop a cached stack or allocate a fresh one.
  Result<FiberStack> acquire() noexcept {
    lock_.lock();
    if (n_free_ > 0) {
      FiberStack s = std::move(free_[--n_free_]);
      lock_.unlock();
      return s;
    }
    lock_.unlock();
    return FiberStack::allocate(usable_tier_);
  }

  // Return a stack to the cache, or destroy if full / wrong size / invalid.
  void release(FiberStack&& s) noexcept {
    if (!s.valid()) return;
    // usable_tier_ is the REQUEST; VmRegion rounds up -- match on actual size.
    const std::size_t want = platform::page_align_up(usable_tier_);
    if (s.size() != want) {
      s.reset().IgnoreError();
      return;
    }
    lock_.lock();
    if (n_free_ < cache_cap_) {
      free_[n_free_++] = std::move(s);
      lock_.unlock();
      return;
    }
    lock_.unlock();
    s.reset().IgnoreError();
  }

  // Release cached stacks until at most `keep` remain. VM release happens
  // outside the SpinLock; concurrent release() may refill the pool after trim
  // observes it, so owners wanting a stable zero first quiesce their users.
  // Returns the number of mappings released by this call.
  std::size_t trim(std::size_t keep = 0) noexcept {
    const std::size_t target = keep < cache_cap_ ? keep : cache_cap_;
    std::size_t released = 0;
    for (;;) {
      FiberStack stack;
      lock_.lock();
      if (n_free_ <= target) {
        lock_.unlock();
        return released;
      }
      stack = std::move(free_[--n_free_]);
      lock_.unlock();
      stack.reset().IgnoreError();
      ++released;
    }
  }

  void clear() noexcept { (void)trim(); }

 private:
  // Fixed slots avoid heap in the pool itself (fiber stacks are already VM).
  std::size_t usable_tier_;
  std::size_t cache_cap_;
  mutable util::SpinLock lock_;
  FiberStack free_[kMaxCacheCap]{};
  std::size_t n_free_ = 0;
};

}  // namespace hyle::mem
