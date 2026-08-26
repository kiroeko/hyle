#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>

#include "hyle/core/compiler.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {

// Single-threaded variable-size allocator over one caller-owned fixed buffer.
// All per-block/allocation metadata lives inside that buffer; the control
// object owns only indexes and counters. Construction and every operation
// perform no hidden heap allocation and there is no fallback. Free
// blocks are indexed by a two-level segregated-fit bitmap. The first level is
// logarithmic and every range has eight linear subdivisions, so a suitable
// free-list head is found in O(1) without walking unrelated blocks. A one-entry
// free cursor keeps a recent return or split remainder on the active allocation
// path. Boundary metadata gives O(1) adjacent coalescing. The storage must
// outlive the pool and every live allocation.
class FixedPool {
public:
  FixedPool() noexcept = default;
  explicit FixedPool(std::span<std::byte> storage) noexcept {
    init(storage.data(), storage.size());
  }
  FixedPool(void *data, std::size_t size) noexcept { init(data, size); }

  FixedPool(const FixedPool&) = delete;
  FixedPool& operator=(const FixedPool&) = delete;
  FixedPool(FixedPool&&) = delete;
  FixedPool& operator=(FixedPool&&) = delete;

  bool valid() const noexcept { return begin_ != nullptr; }

  // Keep the complete allocation/free graph inline. MSVC otherwise outlines
  // nested bin helpers even under /O2, adding several calls to every operation.
  // The whole-request latency certificate in scenario_bench.cc guards this.
  HYLE_FORCE_INLINE void *
  allocate(std::size_t bytes,
           std::size_t alignment = alignof(std::max_align_t)) noexcept {
#if defined(HYLE_TESTING)
    debug_last_fit_candidates_ = 0;
#endif
    HYLE_FATAL_ASSERT(is_pow2(alignment));
    if (bytes == 0 || !valid())
      return nullptr;
    const std::size_t effective_alignment =
        alignment > alignof(Allocation) ? alignment : alignof(Allocation);

    if (recent_free_ != nullptr) {
#if defined(HYLE_TESTING)
      ++debug_last_fit_candidates_;
#endif
      if (void *out = allocate_from(recent_free_, bytes, effective_alignment,
                                    false);
          out != nullptr)
        return out;
      publish_recent();
    }

    // Only indexed lookup needs the conservative, address-independent span.
    // A hot recent cursor can prove the exact fit directly and avoids this
    // overflow/mapping work entirely.
    std::size_t required = 0;
    if (!required_span(bytes, effective_alignment, &required))
      return nullptr;

    unsigned first = 0;
    unsigned second = 0;
    Block *b = find_suitable(required, &first, &second);
    if (b == nullptr)
      return nullptr;
#if defined(HYLE_TESTING)
    ++debug_last_fit_candidates_;
    validate(b);
#endif
    void *out =
        allocate_from(b, bytes, effective_alignment, true, first, second);
    HYLE_FATAL_ASSERT(out != nullptr);
    return out;
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
    HYLE_FATAL_ASSERT(!b->free);
    HYLE_FATAL_ASSERT(used_ >= a->requested);
    used_ -= a->requested;
    a->magic = 0;
    b->free = true;

    // A split remainder is commonly adjacent to the next block returned by a
    // request window. Merge it directly instead of indexing the remainder only
    // to remove it again here. A non-adjacent cursor remains fully discoverable
    // by publishing it before the ordinary boundary-tag coalescing pass.
    if (recent_free_ != nullptr) {
      Block *recent = recent_free_;
      if (next(b) == recent) {
        recent_free_ = nullptr;
        b->span += recent->span;
      } else if (next(recent) == b) {
        recent_free_ = nullptr;
        recent->span += b->span;
        b = recent;
      } else {
        publish_recent();
      }
    }
    recent_free_ = coalesce(b);
  }

  HYLE_FORCE_INLINE void *
  reallocate(void *ptr, std::size_t old_size, std::size_t new_size,
             std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment));
    if (ptr == nullptr || new_size == 0)
      return reallocate_slow(ptr, old_size, new_size, alignment);
    Allocation *a = allocation_for(ptr);
    HYLE_FATAL_ASSERT(a->magic == kAllocationMagic && a->requested == old_size);
    if ((reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0 &&
        resize_in_place(a, ptr, new_size)) {
      account_resize(a, old_size, new_size);
      return ptr;
    }
    return reallocate_slow(ptr, old_size, new_size, alignment);
  }

private:
  HYLE_NOINLINE void *reallocate_slow(void *ptr, std::size_t old_size,
                                      std::size_t new_size,
                                      std::size_t alignment) noexcept {
    if (ptr == nullptr)
      return allocate(new_size, alignment);
    if (new_size == 0) {
      deallocate(ptr);
      return nullptr;
    }
    void *out = allocate(new_size, alignment);
    if (out == nullptr)
      return nullptr;
    std::memcpy(out, ptr, old_size < new_size ? old_size : new_size);
    deallocate(ptr);
    return out;
  }

public:
  void reset() noexcept {
    if (!valid())
      return;
    auto *b = ::new (begin_) Block;
    b->span = capacity_;
    b->previous_span = 0;
    b->magic = kBlockMagic;
    b->free = true;
    for (auto &row : bins_)
      for (Block *&head : row)
        head = nullptr;
    for (std::uint8_t &map : second_level_nonempty_)
      map = 0;
    first_level_nonempty_ = 0;
    recent_free_ = nullptr;
    insert_free(b);
    used_ = 0;
  }

  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t used() const noexcept { return used_; }
  std::size_t peak() const noexcept { return peak_; }

#if defined(HYLE_TESTING)
  // TEST-ONLY: the TLSF index examines at most the optional recent
  // cursor and one bitmap-selected free-list head, regardless of fragmentation.
  std::uint32_t debug_last_fit_candidates() const noexcept {
    return debug_last_fit_candidates_;
  }
#endif

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
  // Eight subdivisions are a mature TLSF configuration: they bound lookup to
  // one bitmap-selected list while keeping this explicit control object near
  // 4 KiB on 64-bit targets. A finer 16/32-way index would spend 8/16 KiB per
  // pool for lower class rounding; an RB tree would reduce control footprint
  // but make allocation/free O(log free_blocks).
  static constexpr unsigned kSecondLevelBits = 3;
  static constexpr unsigned kSecondLevelCount = 1u << kSecondLevelBits;
  static constexpr unsigned kAlignmentBits =
      static_cast<unsigned>(std::countr_zero(alignof(Block)));
  static constexpr unsigned kFirstLevelShift =
      kSecondLevelBits + kAlignmentBits;
  static constexpr unsigned kFirstLevelCount =
      std::numeric_limits<std::size_t>::digits - kFirstLevelShift + 1;
  static constexpr std::size_t kSmallBlock =
      std::size_t{1} << kFirstLevelShift;

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
    if (capacity_ > (std::numeric_limits<std::uintptr_t>::max)() - aligned) {
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

  HYLE_FORCE_INLINE void *
  allocate_from(Block *b, std::size_t bytes,
                std::size_t effective_alignment, bool indexed,
                unsigned indexed_first = 0,
                unsigned indexed_second = 0) noexcept {
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
      remove_free(b, indexed_first, indexed_second);
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
      HYLE_FATAL_ASSERT(recent_free_ == nullptr);
      recent_free_ = tail;
    }

    b->free = false;
    auto *a = reinterpret_cast<Allocation *>(payload - sizeof(Allocation));
    a->block = b;
    a->requested = bytes;
    a->magic = kAllocationMagic;
    HYLE_FATAL_ASSERT(used_ <=
                      (std::numeric_limits<std::size_t>::max)() - bytes);
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

  HYLE_FORCE_INLINE void detach_free(Block *b) noexcept {
    if (recent_free_ == b)
      recent_free_ = nullptr;
    else
      remove_free(b);
  }

  // Return representable tail bytes to the pool after an in-place resize. The
  // preceding block remains live, so only the following neighbor can merge.
  HYLE_FORCE_INLINE void split_resized_tail(Block *b, void *ptr,
                                             std::size_t bytes) noexcept {
    const auto block_addr = reinterpret_cast<std::uintptr_t>(b);
    const auto block_end = block_addr + b->span;
    const auto payload_end = reinterpret_cast<std::uintptr_t>(ptr) + bytes;
    const auto padding =
        (std::uintptr_t{0} - payload_end) & (alignof(Block) - 1);
    if (padding > block_end - payload_end)
      return;
    const auto split_addr = payload_end + padding;
    const std::size_t consumed =
        static_cast<std::size_t>(split_addr - block_addr);
    const std::size_t remainder = b->span - consumed;
    if (remainder < minimum_block_span())
      return;

    b->span = consumed;
    auto *tail = ::new (reinterpret_cast<void *>(split_addr)) Block;
    tail->span = remainder;
    tail->previous_span = consumed;
    tail->magic = kBlockMagic;
    tail->free = true;

    if (Block *following = next(tail);
        following != nullptr && following->free) {
      detach_free(following);
      tail->span += following->span;
    }
    if (Block *following = next(tail); following != nullptr)
      following->previous_span = tail->span;

    if (recent_free_ == nullptr)
      recent_free_ = tail;
    else
      insert_free(tail);
  }

  // O(1) resize when the allocation already has enough physical span or its
  // immediately following block is free. This avoids a bin search and copy in
  // the common vector/buffer growth shape while preserving hard capacity.
  HYLE_FORCE_INLINE bool resize_in_place(Allocation *a, void *ptr,
                                          std::size_t new_size) noexcept {
    Block *b = a->block;
    const auto payload = reinterpret_cast<std::uintptr_t>(ptr);
    const auto block_end = reinterpret_cast<std::uintptr_t>(b) + b->span;
    // Match mature realloc semantics: a shrink changes the requested size but
    // retains the physical span, making a later regrow within it a true O(1)
    // metadata update. Compacting every shrink would mutate free indexes on a
    // normal vector/buffer resize loop and lose the retained-capacity benefit.
    if (new_size <= block_end - payload)
      return true;

    Block *following = next(b);
    if (following == nullptr || !following->free)
      return false;
#if defined(HYLE_TESTING)
    validate(following);
#endif
    const auto combined_end =
        reinterpret_cast<std::uintptr_t>(following) + following->span;
    if (new_size > combined_end - payload)
      return false;

    detach_free(following);
    b->span = static_cast<std::size_t>(combined_end -
                                      reinterpret_cast<std::uintptr_t>(b));
    if (Block *after = next(b); after != nullptr)
      after->previous_span = b->span;
    split_resized_tail(b, ptr, new_size);
    return true;
  }

  HYLE_FORCE_INLINE void account_resize(Allocation *a, std::size_t old_size,
                                         std::size_t new_size) noexcept {
    HYLE_FATAL_ASSERT(used_ >= old_size &&
                      new_size <=
                          (std::numeric_limits<std::size_t>::max)() -
                              (used_ - old_size));
    used_ = used_ - old_size + new_size;
    if (used_ > peak_)
      peak_ = used_;
    a->requested = new_size;
  }

  HYLE_FORCE_INLINE static bool
  required_span(std::size_t bytes, std::size_t alignment,
                std::size_t *out) noexcept {
    HYLE_FATAL_ASSERT(out != nullptr && is_pow2(alignment));
    constexpr std::size_t fixed = sizeof(Block) + sizeof(Allocation);
    std::size_t required = fixed;
    static_assert(alignof(Allocation) == alignof(Block));
    static_assert((sizeof(Block) + sizeof(Allocation)) % alignof(Block) == 0);
    // raw_payload is Block-aligned. Since both values are powers of two and
    // effective alignment is never smaller, the largest possible leading pad
    // is alignment - alignof(Block), not alignment - 1.
    const std::size_t payload_padding = alignment - alignof(Block);
    const std::size_t split_padding = alignof(Block) - 1;
    const std::size_t max = (std::numeric_limits<std::size_t>::max)();
    if (payload_padding > max - required)
      return false;
    required += payload_padding;
    if (bytes > max - required)
      return false;
    required += bytes;
    if (split_padding > max - required)
      return false;
    *out = required + split_padding;
    return true;
  }

  // Floor-map a free span into one FL/SL class. Small blocks use linear
  // alignof(Block) steps; larger blocks divide every power-of-two interval
  // into eight equal ranges.
  HYLE_FORCE_INLINE static void mapping_insert(std::size_t span,
                                               unsigned *first,
                                               unsigned *second) noexcept {
    HYLE_FATAL_ASSERT(span != 0 && first != nullptr && second != nullptr);
    if (span < kSmallBlock) {
      *first = 0;
      *second = static_cast<unsigned>(span >> kAlignmentBits);
      return;
    }
    const unsigned msb = static_cast<unsigned>(std::bit_width(span) - 1);
    *first = msb - (kFirstLevelShift - 1);
    *second = static_cast<unsigned>(span >> (msb - kSecondLevelBits)) ^
              kSecondLevelCount;
  }

  // Ceil-map a request so every block in the selected class is large enough;
  // this is what removes the old per-bin linear fit scan.
  HYLE_FORCE_INLINE static bool mapping_search(std::size_t span,
                                               unsigned *first,
                                               unsigned *second) noexcept {
    if (span >= kSmallBlock) {
      const unsigned msb = static_cast<unsigned>(std::bit_width(span) - 1);
      const std::size_t round =
          (std::size_t{1} << (msb - kSecondLevelBits)) - 1;
      if (span > (std::numeric_limits<std::size_t>::max)() - round)
        return false;
      span += round;
    }
    mapping_insert(span, first, second);
    return *first < kFirstLevelCount;
  }

  HYLE_FORCE_INLINE Block *find_suitable(std::size_t span, unsigned *first,
                                         unsigned *second) noexcept {
    HYLE_FATAL_ASSERT(first != nullptr && second != nullptr);
    if (!mapping_search(span, first, second))
      return nullptr;

    std::uint32_t second_map =
        static_cast<std::uint32_t>(second_level_nonempty_[*first]) &
        (std::uint32_t{0xFF} << *second);
    if (second_map == 0) {
      const std::uint64_t first_map =
          first_level_nonempty_ &
          (~std::uint64_t{0} << static_cast<unsigned>(*first + 1));
      if (first_map == 0)
        return nullptr;
      *first = static_cast<unsigned>(std::countr_zero(first_map));
      second_map = second_level_nonempty_[*first];
    }
    *second = static_cast<unsigned>(std::countr_zero(second_map));
    Block *b = bins_[*first][*second];
    HYLE_FATAL_ASSERT(b != nullptr && b->span >= span);
    return b;
  }

  HYLE_FORCE_INLINE void insert_free(Block *b) noexcept {
    HYLE_FATAL_ASSERT(b->free);
    unsigned first = 0;
    unsigned second = 0;
    mapping_insert(b->span, &first, &second);
    HYLE_FATAL_ASSERT(first < kFirstLevelCount &&
                      second < kSecondLevelCount);
    b->free_prev = nullptr;
    b->free_next = bins_[first][second];
    if (bins_[first][second] != nullptr)
      bins_[first][second]->free_prev = b;
    bins_[first][second] = b;
    first_level_nonempty_ |= std::uint64_t{1} << first;
    second_level_nonempty_[first] |=
        static_cast<std::uint8_t>(1u << second);
  }
  HYLE_FORCE_INLINE void remove_free(Block *b) noexcept {
    HYLE_FATAL_ASSERT(b->free);
    unsigned first = 0;
    unsigned second = 0;
    mapping_insert(b->span, &first, &second);
    remove_free(b, first, second);
  }

  HYLE_FORCE_INLINE void remove_free(Block *b, unsigned first,
                                     unsigned second) noexcept {
    HYLE_FATAL_ASSERT(b->free && first < kFirstLevelCount &&
                      second < kSecondLevelCount);
    if (b->free_prev != nullptr)
      b->free_prev->free_next = b->free_next;
    else {
      HYLE_FATAL_ASSERT(bins_[first][second] == b);
      bins_[first][second] = b->free_next;
    }
    if (b->free_next != nullptr)
      b->free_next->free_prev = b->free_prev;
    if (bins_[first][second] == nullptr) {
      second_level_nonempty_[first] &=
          static_cast<std::uint8_t>(~(1u << second));
      if (second_level_nonempty_[first] == 0)
        first_level_nonempty_ &= ~(std::uint64_t{1} << first);
    }
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
  Block *bins_[kFirstLevelCount][kSecondLevelCount]{};
  std::uint8_t second_level_nonempty_[kFirstLevelCount]{};
  std::uint64_t first_level_nonempty_ = 0;
  Block *recent_free_ = nullptr;
#if defined(HYLE_TESTING)
  std::uint32_t debug_last_fit_candidates_ = 0;
#endif
};

} // namespace hyle::mem
