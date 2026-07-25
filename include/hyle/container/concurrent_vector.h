#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

// Append-only segmented vector for concurrent producers.
//
// This follows oneTBB concurrent_vector's practical publication model:
// writers atomically reserve disjoint indices and construct in parallel;
// growth adds stable segments and never moves existing values. size() is the
// number of RESERVED indices, not a completion fence for writers that are
// currently inside append. Therefore these patterns are safe:
//
//   * append concurrently, join/quiesce writers, then traverse [0,size());
//   * use the returned index after that append call returns;
//   * read a prefix known to predate concurrent appends.
//
// Taking a fresh size() while writers are active and immediately reading every
// new tail element is not safe: another writer may still be constructing one.
// This avoids a global ordered-commit line that measured 5.8x slower than
// oneTBB with eight writers. Users needing per-element live publication should
// store atomics/futures as T rather than making every vector append pay for it.
//
// The growth lock is cold: it is entered only when current capacity is
// exhausted. The fallible try_* path allocates before reserving an index, so OOM
// never strands a permanent hole (stronger than oneTBB's documented
// broken-vector allocator-failure state). The ordinary fail-fast path can use a
// single fetch_add like oneTBB: allocation failure terminates, so there is no
// recoverable object whose hole must be repaired. Element construction must be
// noexcept because the index is already reserved when placement construction
// begins.

namespace hyle::container {

template <class T, class Alloc = mem::SystemAllocator>
class ConcurrentVector {
  static_assert(std::is_nothrow_destructible_v<T>,
                "ConcurrentVector requires noexcept destruction");

 public:
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  // initial_capacity sizes the first segment (rounded to a power of two); it
  // does not allocate eagerly. Later segments double.
  explicit ConcurrentVector(std::size_t initial_capacity = 16) noexcept
      requires std::is_default_constructible_v<Alloc> {
    init_first_segment(initial_capacity);
  }

  explicit ConcurrentVector(Alloc alloc,
                            std::size_t initial_capacity = 16) noexcept
      : alloc_(std::move(alloc)) {
    init_first_segment(initial_capacity);
  }

  ~ConcurrentVector() {
    const std::size_t count = size_.load(std::memory_order_relaxed);
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (std::size_t i = 0; i < count; ++i) (*this)[i].~T();
    }
    for (std::uint32_t k = 0; k < num_segments_; ++k) {
      if (segments_[k] != nullptr)
        alloc_.deallocate_n(segments_[k], segment_capacity(k));
    }
  }

  HYLE_DISABLE_COPY_AND_MOVE(ConcurrentVector);

  // Preallocate enough stable segments for count values. Concurrent with
  // append; only this cold path serializes. Failure preserves existing state.
  bool try_reserve(std::size_t count) noexcept {
    if (count <= capacity_.load(std::memory_order_acquire)) return true;
    std::lock_guard<std::mutex> guard(grow_lock_);
    std::size_t current = capacity_.load(std::memory_order_relaxed);
    while (current < count) {
      const std::uint32_t k = num_segments_;
      if (k >= max_segments()) return false;
      const std::size_t segment_count = segment_capacity(k);
      if (segment_count > static_cast<std::size_t>(-1) - current) return false;
      T* values = alloc_.template allocate_n<T>(segment_count);
      if (values == nullptr) return false;
      segments_[k] = values;
      num_segments_ = k + 1;
      current += segment_count;
      capacity_.store(current, std::memory_order_release);
    }
    return true;
  }

  void reserve(std::size_t count) noexcept {
    HYLE_FATAL_ASSERT(try_reserve(count));
  }

  // Fallible append for allocator-aware/FFI paths. SIZE_MAX means capacity
  // allocation failed before any index was claimed.
  template <class... Args>
    requires std::is_nothrow_constructible_v<T, Args...>
  std::size_t try_emplace_back(Args&&... args) noexcept {
    const std::size_t index = reserve_index();
    if (index == npos) return npos;
    const auto [segment, offset] = locate(index);
    ::new (static_cast<void*>(segments_[segment] + offset))
        T(std::forward<Args>(args)...);
    return index;
  }

  template <class... Args>
    requires std::is_nothrow_constructible_v<T, Args...>
  std::size_t emplace_back(Args&&... args) noexcept {
    // This is intentionally distinct from try_emplace_back(). In the common
    // pre-reserved case one fetch_add gives each writer a disjoint slot; the
    // recoverable API pays the CAS loop needed to avoid claiming past capacity.
    const std::size_t index = size_.fetch_add(1, std::memory_order_relaxed);
    HYLE_FATAL_ASSERT(index != npos);
    if (HYLE_UNLIKELY(index >= capacity_.load(std::memory_order_acquire)))
      reserve(index + 1);
    const auto [segment, offset] = locate(index);
    ::new (static_cast<void*>(segments_[segment] + offset))
        T(std::forward<Args>(args)...);
    return index;
  }

  std::size_t try_push_back(const T& value) noexcept
      requires std::is_nothrow_copy_constructible_v<T> {
    return try_emplace_back(value);
  }
  std::size_t try_push_back(T&& value) noexcept
      requires std::is_nothrow_move_constructible_v<T> {
    return try_emplace_back(std::move(value));
  }
  std::size_t push_back(const T& value) noexcept
      requires std::is_nothrow_copy_constructible_v<T> {
    return emplace_back(value);
  }
  std::size_t push_back(T&& value) noexcept
      requires std::is_nothrow_move_constructible_v<T> {
    return emplace_back(std::move(value));
  }

  // Unchecked stable random access. The caller supplies the completion
  // synchronization described above; indexing itself is a plain read.
  T& operator[](std::size_t index) noexcept {
    const auto [segment, offset] = locate(index);
    return segments_[segment][offset];
  }
  const T& operator[](std::size_t index) const noexcept {
    const auto [segment, offset] = locate(index);
    return segments_[segment][offset];
  }

  // Reservation count. It is exact after writers quiesce, but deliberately is
  // not an acquire-publication barrier for in-flight element constructors.
  std::size_t size() const noexcept {
    return size_.load(std::memory_order_relaxed);
  }
  bool empty() const noexcept { return size() == 0; }

  std::size_t capacity() const noexcept {
    return capacity_.load(std::memory_order_acquire);
  }

 private:
  // 48 doubling segments cover >2^48 values from a one-element base while
  // keeping the fixed pointer table modest. 32-bit builds clamp earlier.
  static constexpr std::uint32_t kMaxSegments = 48;

  void init_first_segment(std::size_t initial_capacity) noexcept {
    constexpr std::size_t kHighestPower =
        std::size_t{1} << (sizeof(std::size_t) * 8 - 1);
    HYLE_FATAL_ASSERT(initial_capacity <= kHighestPower);
    const std::size_t first =
        std::bit_ceil(initial_capacity < 1 ? 1 : initial_capacity);
    first_shift_ =
        static_cast<std::uint32_t>(std::countr_zero(first));
  }

  std::uint32_t max_segments() const noexcept {
    constexpr auto kSizeBits =
        static_cast<std::uint32_t>(sizeof(std::size_t) * 8);
    const std::uint32_t addressable = kSizeBits - first_shift_;
    return addressable < kMaxSegments ? addressable : kMaxSegments;
  }

  std::size_t segment_capacity(std::uint32_t k) const noexcept {
    return (std::size_t{1} << first_shift_) << k;
  }

  std::pair<std::uint32_t, std::size_t> locate(
      std::size_t index) const noexcept {
    const std::size_t scaled = index >> first_shift_;
    const std::uint32_t segment =
        static_cast<std::uint32_t>(std::bit_width(scaled + 1)) - 1;
    const std::size_t prefix =
        (std::size_t{1} << first_shift_) *
        ((std::size_t{1} << segment) - 1);
    return {segment, index - prefix};
  }

  std::size_t reserve_index() noexcept {
    std::size_t candidate = size_.load(std::memory_order_relaxed);
    for (;;) {
      const std::size_t available = capacity_.load(std::memory_order_acquire);
      if (candidate >= available) {
        if (candidate == npos || !try_reserve(candidate + 1)) return npos;
        candidate = size_.load(std::memory_order_relaxed);
        continue;
      }
      if (size_.compare_exchange_weak(candidate, candidate + 1,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed))
        return candidate;
    }
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  alignas(core::kDestructiveInterferenceSize)
      std::atomic<std::size_t> size_{0};
  std::atomic<std::size_t> capacity_{0};
  std::mutex grow_lock_;
  std::uint32_t first_shift_ = 0;
  std::uint32_t num_segments_ = 0;
  T* segments_[kMaxSegments] = {};
};

}  // namespace hyle::container
