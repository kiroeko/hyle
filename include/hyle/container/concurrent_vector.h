#pragma once

#include <atomic>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
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

  template <bool IsConst>
  class BasicIterator {
    template <bool>
    friend class BasicIterator;
    friend class ConcurrentVector;

    using Owner =
        std::conditional_t<IsConst, const ConcurrentVector, ConcurrentVector>;

   public:
    using iterator_concept = std::random_access_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = std::conditional_t<IsConst, const T&, T&>;
    using pointer = std::conditional_t<IsConst, const T*, T*>;

    BasicIterator() noexcept = default;

    template <bool Enabled = IsConst>
      requires Enabled
    BasicIterator(const BasicIterator<false>& other) noexcept
        : owner_(other.owner_), index_(other.index_), limit_(other.limit_),
          pointer_(other.pointer_), segment_(other.segment_),
          segment_begin_(other.segment_begin_),
          segment_end_(other.segment_end_) {}

    reference operator*() const noexcept { return *pointer_; }
    pointer operator->() const noexcept { return pointer_; }
    reference operator[](difference_type offset) const noexcept {
      return (*owner_)[static_cast<std::size_t>(
          static_cast<difference_type>(index_) + offset)];
    }

    BasicIterator& operator++() noexcept {
      ++index_;
      if (index_ >= limit_) {
        pointer_ = nullptr;
      } else if (index_ == segment_end_) {
        ++segment_;
        segment_begin_ = segment_end_;
        segment_end_ += owner_->segment_capacity(segment_);
        pointer_ = owner_->segment_at(segment_);
      } else {
        ++pointer_;
      }
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    BasicIterator& operator--() noexcept {
      const std::size_t next = index_ - 1;
      if (pointer_ != nullptr && next >= segment_begin_) {
        --index_;
        --pointer_;
      } else {
        seek(next);
      }
      return *this;
    }
    BasicIterator operator--(int) noexcept {
      BasicIterator copy = *this;
      --*this;
      return copy;
    }

    BasicIterator& operator+=(difference_type offset) noexcept {
      seek(static_cast<std::size_t>(static_cast<difference_type>(index_) +
                                    offset));
      return *this;
    }
    BasicIterator& operator-=(difference_type offset) noexcept {
      return *this += -offset;
    }

    friend BasicIterator operator+(BasicIterator it,
                                   difference_type offset) noexcept {
      it += offset;
      return it;
    }
    friend BasicIterator operator+(difference_type offset,
                                   BasicIterator it) noexcept {
      it += offset;
      return it;
    }
    friend BasicIterator operator-(BasicIterator it,
                                   difference_type offset) noexcept {
      it -= offset;
      return it;
    }
    friend difference_type operator-(const BasicIterator& left,
                                     const BasicIterator& right) noexcept {
      return static_cast<difference_type>(left.index_) -
             static_cast<difference_type>(right.index_);
    }
    friend bool operator==(const BasicIterator& left,
                           const BasicIterator& right) noexcept {
      return left.owner_ == right.owner_ && left.index_ == right.index_;
    }
    friend auto operator<=>(const BasicIterator& left,
                            const BasicIterator& right) noexcept {
      return left.index_ <=> right.index_;
    }

   private:
    BasicIterator(Owner* owner, std::size_t index,
                  std::size_t limit) noexcept
        : owner_(owner), limit_(limit) {
      seek(index);
    }

    void seek(std::size_t index) noexcept {
      index_ = index;
      if (index_ >= limit_) {
        pointer_ = nullptr;
        return;
      }
      const auto [segment, offset] = owner_->locate(index_);
      segment_ = segment;
      segment_begin_ = index_ - offset;
      segment_end_ = segment_begin_ + owner_->segment_capacity(segment_);
      pointer_ = owner_->segment_at(segment_) + offset;
    }

    Owner* owner_ = nullptr;
    std::size_t index_ = 0;
    std::size_t limit_ = 0;
    pointer pointer_ = nullptr;
    std::uint32_t segment_ = 0;
    std::size_t segment_begin_ = 0;
    std::size_t segment_end_ = 0;
  };

  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;

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
    clear();
    for (std::uint32_t k = 0; k < num_segments_; ++k) {
      T* segment = segment_at(k);
      if (segment != nullptr)
        alloc_.deallocate_n(segment, segment_capacity(k));
    }
    if (extended_segments_ != nullptr)
      alloc_.deallocate_n(extended_segments_, max_segments());
  }

  ConcurrentVector(const ConcurrentVector&) = delete;
  ConcurrentVector& operator=(const ConcurrentVector&) = delete;
  ConcurrentVector(ConcurrentVector&&) = delete;
  ConcurrentVector& operator=(ConcurrentVector&&) = delete;

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

      T** promoted_table = nullptr;
      if (k == kEmbeddedSegments && extended_segments_ == nullptr) {
        const std::size_t table_size = max_segments();
        promoted_table = alloc_.template allocate_n<T*>(table_size);
        if (promoted_table == nullptr) return false;
        for (std::size_t i = 0; i < table_size; ++i)
          std::construct_at(promoted_table + i, nullptr);
        for (std::uint32_t i = 0; i < kEmbeddedSegments; ++i)
          promoted_table[i] = embedded_segments_[i];
      }

      T* values = alloc_.template allocate_n<T>(segment_count);
      if (values == nullptr) {
        if (promoted_table != nullptr)
          alloc_.deallocate_n(promoted_table, max_segments());
        return false;
      }
      if (promoted_table != nullptr) {
        promoted_table[k] = values;
        extended_segments_ = promoted_table;
      } else {
        store_segment(k, values);
      }
      num_segments_ = k + 1;
      current += segment_count;
      capacity_.store(current, std::memory_order_release);
    }
    return true;
  }

  void reserve(std::size_t count) noexcept {
    HYLE_FATAL_ASSERT(try_reserve(count));
  }

  // Fallible append for allocator-aware paths. SIZE_MAX means capacity
  // allocation failed before any index was claimed.
  template <class... Args>
    requires std::is_nothrow_constructible_v<T, Args...>
  std::size_t try_emplace_back(Args&&... args) noexcept {
    const std::size_t index = reserve_index();
    if (index == npos) return npos;
    const auto [segment, offset] = locate(index);
    ::new (static_cast<void*>(segment_at(segment) + offset))
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
    ::new (static_cast<void*>(segment_at(segment) + offset))
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
    return segment_at(segment)[offset];
  }
  const T& operator[](std::size_t index) const noexcept {
    const auto [segment, offset] = locate(index);
    return segment_at(segment)[offset];
  }

  // Iteration snapshots the reservation count. Use it only after writers have
  // quiesced; it deliberately does not turn size() into a completion fence.
  // The iterator caches its current segment pointer, so ++ is a plain pointer
  // increment except at segment boundaries.
  iterator begin() noexcept {
    const std::size_t count = size();
    return iterator(this, 0, count);
  }
  iterator end() noexcept {
    const std::size_t count = size();
    return iterator(this, count, count);
  }
  const_iterator begin() const noexcept {
    const std::size_t count = size();
    return const_iterator(this, 0, count);
  }
  const_iterator end() const noexcept {
    const std::size_t count = size();
    return const_iterator(this, count, count);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }

  // Reservation count. It is exact after writers quiesce, but deliberately is
  // not an acquire-publication barrier for in-flight element constructors.
  std::size_t size() const noexcept {
    return size_.load(std::memory_order_relaxed);
  }
  bool empty() const noexcept { return size() == 0; }

  std::size_t capacity() const noexcept {
    return capacity_.load(std::memory_order_acquire);
  }

  // Destroy every completed element after all writers have quiesced. Segment
  // allocations and the promoted pointer table are retained, so subsequent
  // appends reuse capacity without allocation. All prior references/iterators
  // become invalid even when their addresses are reused.
  void clear() noexcept {
    std::lock_guard<std::mutex> guard(grow_lock_);
    const std::size_t count = size_.load(std::memory_order_relaxed);
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (std::size_t i = count; i != 0; --i)
        std::destroy_at(std::addressof((*this)[i - 1]));
    }
    size_.store(0, std::memory_order_relaxed);
  }

 private:
  // Match oneTBB's normal object shape: three segment pointers cover the common
  // small vector, and a complete table is allocated only when segment 3 is
  // needed. 48 doubling segments cover >2^48 values from a one-element base;
  // 32-bit builds clamp earlier.
  static constexpr std::uint32_t kEmbeddedSegments = 3;
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

  T* segment_at(std::uint32_t segment) const noexcept {
    // Do not validate against num_segments_: readers of a completed prefix may
    // run while a writer grows a different segment, and num_segments_ is owned
    // by the cold growth lock. The index contract plus locate() proves this
    // compile-time table bound without adding a racing hot read.
    HYLE_ASSUME(segment < kMaxSegments);
    return segment < kEmbeddedSegments ? embedded_segments_[segment]
                                       : extended_segments_[segment];
  }

  void store_segment(std::uint32_t segment, T* values) noexcept {
    if (segment < kEmbeddedSegments)
      embedded_segments_[segment] = values;
    else
      extended_segments_[segment] = values;
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
  T* embedded_segments_[kEmbeddedSegments] = {};
  T** extended_segments_ = nullptr;
};

}  // namespace hyle::container
