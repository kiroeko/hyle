#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"
#include "hyle/util/spin_lock.h"

// Read-optimized append-only concurrent vector (clay parallel/concurrent_vector).
// Usage pattern: insert once, read many, from many threads. Concurrent readers
// are LOCK-FREE (one atomic load + an index computation); writers serialize on a
// spin lock. Elements, once inserted, never move -- so a reference/index stays
// valid for the life of the container, even across growth.
//
// HYLE redesign vs clay (clay had five issues, all fixed here):
//  1. clay stored std::vector<std::vector<T>> and relied on a hidden reserve(65)
//     in the ctor to stop the OUTER vector from reallocating (which would free
//     buffers out from under lock-free readers). HYLE uses a fixed C-array of
//     segment pointers -- the "outer never reallocates" invariant is STRUCTURAL,
//     not a fragile reserve() call.
//  2. clay copied the whole old buffer into the new one on every grow and never
//     freed the old -> ~2x live memory and O(n) copy per grow. HYLE is SEGMENTED:
//     each segment is independent, growth allocates one new segment, nothing is
//     copied. Memory is ~1x; insert is O(1).
//  3. clay required T copyable (because of the copy-on-grow). HYLE never moves an
//     existing element, so T may be move-only or even immovable after construction.
//  4. clay bounds-checked every read (a branch on the hot path). HYLE's
//     operator[] is unchecked (the std::vector contract: caller ensures < size()).
//  5. no std owning container -- segments come from an injected Allocator.
//
// Memory ordering: the published size_ (release on write / acquire on read) is
// the single synchronization point. A writer constructs the element and stores
// the segment pointer BEFORE the size_ release; a reader does the size_ acquire
// (via size()) BEFORE indexing, so for any index < the acquired size both the
// segment pointer and the element are fully visible (no torn/!uninit reads).
// operator[] itself is a PLAIN read -- the caller's size() acquire is the fence;
// re-acquiring per element would be redundant and contends with the writer.

namespace hyle::container {

template <class T, class Alloc = mem::SystemAllocator>
class ConcurrentVector {
 public:
  // `initial_capacity` sizes the first segment (rounded up to a power of two);
  // later segments double. Nothing is allocated until the first insert. Alloc defaults to
  // compile-time SystemAllocator ([[no_unique_address]] -> 0 bytes); pass mem::Allocator to
  // opt into injection.
  explicit ConcurrentVector(std::size_t initial_capacity = 16) noexcept
      requires std::is_default_constructible_v<Alloc> {
    std::size_t first = std::bit_ceil(initial_capacity < 1 ? 1 : initial_capacity);
    first_shift_ = static_cast<std::uint32_t>(std::countr_zero(first));
  }
  explicit ConcurrentVector(Alloc alloc, std::size_t initial_capacity = 16) noexcept
      : alloc_(alloc) {
    std::size_t first = std::bit_ceil(initial_capacity < 1 ? 1 : initial_capacity);
    first_shift_ = static_cast<std::uint32_t>(std::countr_zero(first));
  }

  ~ConcurrentVector() {
    std::size_t n = size_.load(std::memory_order_relaxed);
    // Destroy constructed elements (skip for trivial T) then free segments.
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (std::size_t i = 0; i < n; ++i) (*this)[i].~T();
    }
    for (std::uint32_t k = 0; k < num_segments_; ++k) {
      if (segments_[k] != nullptr)
        alloc_.deallocate_n(segments_[k], seg_capacity(k));
    }
  }

  HYLE_DISABLE_COPY_AND_MOVE(ConcurrentVector);

  // Append, returning the new element's index. Thread-safe with other writers
  // (serialized) and with concurrent readers (they see a consistent prefix).
  template <class... Args>
  std::size_t emplace_back(Args&&... args) {
    std::lock_guard<util::SpinLock> guard(write_lock_);
    std::size_t i = size_.load(std::memory_order_relaxed);  // sole writer region
    auto [k, off] = locate(i);
    if (k >= num_segments_) {
      // New segment. Allocate before publishing any size that reaches into it.
      segments_[k] = alloc_.template allocate_n<T>(seg_capacity(k));
      num_segments_ = k + 1;
    }
    ::new (static_cast<void*>(segments_[k] + off)) T(std::forward<Args>(args)...);
    size_.store(i + 1, std::memory_order_release);  // publish (HB for readers)
    return i;
  }

  std::size_t push_back(const T& v) { return emplace_back(v); }
  std::size_t push_back(T&& v) { return emplace_back(std::move(v)); }

  // Unchecked indexed access (caller ensures index < size()). LOCK-FREE and, by
  // design, a PLAIN read -- no per-element atomic. The synchronization point is
  // size(): to know index < size() under a concurrent writer the caller must have
  // observed size() >= index+1, and size() is an acquire load that synchronizes-
  // with the writer's release-store of size_. That single acquire makes the
  // segment pointer AND the element for every index < the observed size visible,
  // so indexing afterwards needs no further fence (the std::vector / tbb::
  // concurrent_vector model: observe size once, then read freely). An earlier
  // version issued an acquire load of size_ on EVERY operator[]; that was
  // redundant given the contract and made each read contend on the size_ cache
  // line with the writer's release-store (a measured ~2x slowdown under a
  // concurrent writer). locate() reads only the index and the immutable
  // first_shift_, never size_.
  T& operator[](std::size_t index) noexcept {
    auto [k, off] = locate(index);
    return segments_[k][off];
  }
  const T& operator[](std::size_t index) const noexcept {
    auto [k, off] = locate(index);
    return segments_[k][off];
  }

  // Count of elements visible to this thread (monotonic; conservative).
  std::size_t size() const noexcept {
    return size_.load(std::memory_order_acquire);
  }
  bool empty() const noexcept { return size() == 0; }

  // Total allocated element capacity across segments.
  std::size_t capacity() const noexcept {
    std::size_t cap = 0;
    for (std::uint32_t k = 0; k < num_segments_; ++k) cap += seg_capacity(k);
    return cap;
  }

 private:
  // Up to 48 doubling segments from a >=1-element base spans >1e15 elements --
  // far beyond any real use, while keeping the pointer array tiny (48*8 B).
  static constexpr std::uint32_t kMaxSegments = 48;

  std::size_t seg_capacity(std::uint32_t k) const noexcept {
    return (std::size_t{1} << first_shift_) << k;  // base<<k
  }

  // index -> (segment, offset). With base B=2^first_shift_, segment k holds
  // [B*(2^k - 1), B*(2^(k+1) - 1)); see header note for the derivation. Reads
  // only `i` and the immutable first_shift_ -- never size_, so it carries no
  // ordering of its own (the caller's size() acquire is the synchronization).
  std::pair<std::uint32_t, std::size_t> locate(std::size_t i) const noexcept {
    std::size_t m = i >> first_shift_;
    std::uint32_t k = static_cast<std::uint32_t>(std::bit_width(m + 1)) - 1;
    std::size_t prefix = (std::size_t{1} << first_shift_) * ((std::size_t{1} << k) - 1);
    return {k, i - prefix};
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  std::atomic<std::size_t> size_{0};       // published count (release/acquire)
  util::SpinLock write_lock_;              // serializes writers (rare by design)
  std::uint32_t first_shift_ = 0;          // log2(first segment capacity)
  std::uint32_t num_segments_ = 0;         // allocated segments (writer-owned)
  T* segments_[kMaxSegments] = {};         // fixed: never reallocates (the trick)
};

}  // namespace hyle::container
