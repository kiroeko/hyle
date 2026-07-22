#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>

#include "hyle/macros.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {

// Single-threaded variable-size allocator over one caller-owned fixed buffer.
// All per-block/allocation metadata lives inside that buffer; the control
// object owns only indexes and counters. Construction and every operation
// perform no hidden heap allocation and there is no fallback. Free
// blocks are indexed by logarithmic size bins; a one-entry recent-free cache
// avoids an immediate bin round trip on common free-then-allocate churn, while
// boundary metadata still gives O(1) adjacent coalescing. Allocation normally
// visits one matching size class instead of scanning every physical block. The
// storage must outlive the pool and every live allocation.
class FixedPool {
public:
  FixedPool() noexcept = default;
  explicit FixedPool(std::span<std::byte> storage) noexcept {
    init(storage.data(), storage.size());
  }
  FixedPool(void *data, std::size_t size) noexcept { init(data, size); }

  HYLE_DISABLE_COPY_AND_MOVE(FixedPool);

  bool valid() const noexcept { return begin_ != nullptr; }

  // Keep the complete allocation/free graph inline. MSVC otherwise outlines
  // nested bin helpers even under /O2, adding several calls to every operation.
  // The whole-request latency certificate in scenario_bench.cc guards this.
  HYLE_FORCE_INLINE void *
  allocate(std::size_t bytes,
           std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment));
    if (bytes == 0 || !valid())
      return nullptr;
    const std::size_t effective_alignment =
        alignment > alignof(Allocation) ? alignment : alignof(Allocation);

    std::size_t lower = sizeof(Block) + sizeof(Allocation);
    if (bytes > std::numeric_limits<std::size_t>::max() - lower)
      return nullptr;
    lower += bytes;

    if (recent_free_ != nullptr) {
      if (void *out = allocate_from(recent_free_, bytes, effective_alignment,
                                    false);
          out != nullptr)
        return out;
      publish_recent();
    }

    std::uint64_t candidates =
        nonempty_bins_ & (~std::uint64_t{0} << bin_index(lower));
    while (candidates != 0) {
      const unsigned bin = static_cast<unsigned>(std::countr_zero(candidates));
      candidates &= candidates - 1;
      for (Block *b = bins_[bin]; b != nullptr; b = b->free_next) {
#if defined(HYLE_TESTING)
        validate(b);
#endif
        if (void *out = allocate_from(b, bytes, effective_alignment, true);
            out != nullptr)
          return out;
      }
    }
    return nullptr;
  }

  HYLE_FORCE_INLINE void deallocate(void *ptr) noexcept {
    if (ptr == nullptr)
      return;
    Allocation *a = allocation_for(ptr);
    Block *b = a->block;
    HYLE_FATAL_ASSERT(a->magic == kAllocationMagic);
#if defined(HYLE_TESTING)
    validate(b);
#endif
    publish_recent();
    HYLE_FATAL_ASSERT(!b->free);
    HYLE_FATAL_ASSERT(used_ >= a->requested);
    used_ -= a->requested;
    a->magic = 0;
    b->free = true;
    recent_free_ = coalesce(b);
  }

  void *reallocate(void *ptr, std::size_t old_size, std::size_t new_size,
                   std::size_t alignment = alignof(std::max_align_t)) noexcept {
    if (ptr == nullptr)
      return allocate(new_size, alignment);
    if (new_size == 0) {
      deallocate(ptr);
      return nullptr;
    }
    Allocation *a = allocation_for(ptr);
    HYLE_FATAL_ASSERT(a->magic == kAllocationMagic && a->requested == old_size);
    if (new_size <= old_size) {
      used_ -= old_size - new_size;
      a->requested = new_size;
      return ptr;
    }
    void *out = allocate(new_size, alignment);
    if (out == nullptr)
      return nullptr;
    std::memcpy(out, ptr, old_size);
    deallocate(ptr);
    return out;
  }

  void reset() noexcept {
    if (!valid())
      return;
    auto *b = ::new (begin_) Block;
    b->span = capacity_;
    b->previous_span = 0;
    b->magic = kBlockMagic;
    b->free = true;
    for (Block *&head : bins_)
      head = nullptr;
    nonempty_bins_ = 0;
    recent_free_ = nullptr;
    insert_free(b);
    used_ = 0;
  }

  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t used() const noexcept { return used_; }
  std::size_t peak() const noexcept { return peak_; }

  Allocator as_allocator() noexcept {
    return Allocator(&pool_alloc, &pool_free, &pool_realloc, this);
  }

private:
  struct Block {
    std::size_t span = 0;
    std::size_t previous_span = 0;
    Block *free_prev = nullptr;
    Block *free_next = nullptr;
    std::uint32_t magic = 0;
    bool free = false;
  };
  struct Allocation {
    Block *block = nullptr;
    std::size_t requested = 0;
    std::uint32_t magic = 0;
  };

  static constexpr std::uint32_t kBlockMagic = 0x48594C45u;
  static constexpr std::uint32_t kAllocationMagic = 0x4D454D41u;
  static constexpr unsigned kBinCount = sizeof(std::size_t) * 8;

  static constexpr std::size_t minimum_block_span() noexcept {
    return sizeof(Block) + sizeof(Allocation) + alignof(Allocation) + 1;
  }

  void init(void *data, std::size_t size) noexcept {
    if (data == nullptr || size < minimum_block_span())
      return;
    const auto raw = reinterpret_cast<std::uintptr_t>(data);
    std::uintptr_t aligned = 0;
    if (!detail::try_align_up_address(raw, alignof(Block), &aligned))
      return;
    const std::size_t prefix = static_cast<std::size_t>(aligned - raw);
    if (prefix >= size || size - prefix < minimum_block_span())
      return;
    begin_ = reinterpret_cast<std::byte *>(aligned);
    capacity_ = size - prefix;
    capacity_ -= capacity_ % alignof(Block);
    if (capacity_ < minimum_block_span()) {
      begin_ = nullptr;
      capacity_ = 0;
      return;
    }
    if (capacity_ > std::numeric_limits<std::uintptr_t>::max() - aligned) {
      begin_ = nullptr;
      capacity_ = 0;
      return;
    }
    end_ = reinterpret_cast<std::byte *>(aligned + capacity_);
    reset();
  }

  HYLE_FORCE_INLINE Block *next(Block *b) const noexcept {
    auto *n = reinterpret_cast<std::byte *>(b) + b->span;
    return n < end_ ? reinterpret_cast<Block *>(n) : nullptr;
  }
  void validate(const Block *b) const noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(b);
    const auto begin = reinterpret_cast<std::uintptr_t>(begin_);
    const auto end = reinterpret_cast<std::uintptr_t>(end_);
    HYLE_FATAL_ASSERT(address >= begin && address <= end &&
                      sizeof(Block) <= end - address);
    HYLE_FATAL_ASSERT(b->magic == kBlockMagic && b->span >= sizeof(Block));
    HYLE_FATAL_ASSERT(b->span <= static_cast<std::size_t>(end - address));
  }
  HYLE_FORCE_INLINE Allocation *allocation_for(void *ptr) const noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const auto begin = reinterpret_cast<std::uintptr_t>(begin_);
    const auto end = reinterpret_cast<std::uintptr_t>(end_);
    HYLE_FATAL_ASSERT(valid() && address >= begin &&
                      address - begin >= sizeof(Allocation) && address < end);
    const auto header = address - sizeof(Allocation);
    HYLE_FATAL_ASSERT(header % alignof(Allocation) == 0);
    return reinterpret_cast<Allocation *>(header);
  }

  HYLE_FORCE_INLINE void *allocate_from(Block *b, std::size_t bytes,
                                        std::size_t effective_alignment,
                                        bool indexed) noexcept {
    const auto block_addr = reinterpret_cast<std::uintptr_t>(b);
    const auto raw_payload = block_addr + sizeof(Block) + sizeof(Allocation);
    const auto block_end = block_addr + b->span;
    if (raw_payload > block_end)
      return nullptr;
    const auto payload_padding =
        (std::uintptr_t{0} - raw_payload) & (effective_alignment - 1);
    if (payload_padding > block_end - raw_payload)
      return nullptr;
    const std::uintptr_t payload = raw_payload + payload_padding;
    if (bytes > block_end - payload)
      return nullptr;
    const std::uintptr_t payload_end = payload + bytes;
    const auto split_padding =
        (std::uintptr_t{0} - payload_end) & (alignof(Block) - 1);
    if (split_padding > block_end - payload_end)
      return nullptr;
    const std::uintptr_t split_addr = payload_end + split_padding;

    if (indexed)
      remove_free(b);
    else {
      HYLE_FATAL_ASSERT(recent_free_ == b);
      recent_free_ = nullptr;
    }
    const std::size_t consumed =
        static_cast<std::size_t>(split_addr - block_addr);
    const std::size_t remainder = b->span - consumed;
    if (remainder >= minimum_block_span()) {
      auto *tail = ::new (reinterpret_cast<void *>(split_addr)) Block;
      tail->span = remainder;
      tail->previous_span = consumed;
      tail->magic = kBlockMagic;
      tail->free = true;
      b->span = consumed;
      if (Block *after = next(tail); after != nullptr)
        after->previous_span = tail->span;
      insert_free(tail);
    }

    b->free = false;
    auto *a = reinterpret_cast<Allocation *>(payload - sizeof(Allocation));
    a->block = b;
    a->requested = bytes;
    a->magic = kAllocationMagic;
    HYLE_FATAL_ASSERT(used_ <=
                      std::numeric_limits<std::size_t>::max() - bytes);
    used_ += bytes;
    if (used_ > peak_)
      peak_ = used_;
    return reinterpret_cast<void *>(payload);
  }

  HYLE_FORCE_INLINE void publish_recent() noexcept {
    if (recent_free_ == nullptr)
      return;
    Block *b = recent_free_;
    recent_free_ = nullptr;
    insert_free(b);
  }

  HYLE_FORCE_INLINE static unsigned bin_index(std::size_t span) noexcept {
    HYLE_FATAL_ASSERT(span != 0);
    return static_cast<unsigned>(std::bit_width(span) - 1);
  }
  HYLE_FORCE_INLINE void insert_free(Block *b) noexcept {
    HYLE_FATAL_ASSERT(b->free);
    const unsigned bin = bin_index(b->span);
    b->free_prev = nullptr;
    b->free_next = bins_[bin];
    if (bins_[bin] != nullptr)
      bins_[bin]->free_prev = b;
    bins_[bin] = b;
    nonempty_bins_ |= std::uint64_t{1} << bin;
  }
  HYLE_FORCE_INLINE void remove_free(Block *b) noexcept {
    HYLE_FATAL_ASSERT(b->free);
    const unsigned bin = bin_index(b->span);
    if (b->free_prev != nullptr)
      b->free_prev->free_next = b->free_next;
    else {
      HYLE_FATAL_ASSERT(bins_[bin] == b);
      bins_[bin] = b->free_next;
    }
    if (b->free_next != nullptr)
      b->free_next->free_prev = b->free_prev;
    if (bins_[bin] == nullptr)
      nonempty_bins_ &= ~(std::uint64_t{1} << bin);
    b->free_prev = b->free_next = nullptr;
  }
  HYLE_FORCE_INLINE Block *coalesce(Block *b) noexcept {
    if (Block *n = next(b); n != nullptr) {
#if defined(HYLE_TESTING)
      validate(n);
#endif
      if (n->free) {
        remove_free(n);
        b->span += n->span;
      }
    }
    if (b->previous_span != 0) {
      auto *p = reinterpret_cast<Block *>(reinterpret_cast<std::byte *>(b) -
                                          b->previous_span);
#if defined(HYLE_TESTING)
      validate(p);
#endif
      if (p->free) {
        remove_free(p);
        p->span += b->span;
        b = p;
      }
    }
    if (Block *n = next(b); n != nullptr)
      n->previous_span = b->span;
    return b;
  }

  static void *pool_alloc(void *state, std::size_t size,
                          std::size_t alignment) noexcept {
    return static_cast<FixedPool *>(state)->allocate(size, alignment);
  }
  static void pool_free(void *state, void *ptr, std::size_t,
                        std::size_t) noexcept {
    static_cast<FixedPool *>(state)->deallocate(ptr);
  }
  static void *pool_realloc(void *state, void *ptr, std::size_t old_size,
                            std::size_t new_size,
                            std::size_t alignment) noexcept {
    return static_cast<FixedPool *>(state)->reallocate(ptr, old_size, new_size,
                                                       alignment);
  }

  std::byte *begin_ = nullptr;
  std::byte *end_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
  std::size_t peak_ = 0;
  Block *bins_[kBinCount]{};
  std::uint64_t nonempty_bins_ = 0;
  Block *recent_free_ = nullptr;
};

} // namespace hyle::mem
