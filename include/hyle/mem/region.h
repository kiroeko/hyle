#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {

namespace detail {

// Shared single-threaded bump cursor used by external/fixed region shapes. It
// owns no storage and performs no fallback allocation.
class BumpCursor {
public:
  BumpCursor() noexcept = default;
  BumpCursor(void *data, std::size_t size) noexcept {
    reset_storage(data, size);
  }

  void reset_storage(void *data, std::size_t size) noexcept {
    begin_ = cur_ = end_ = nullptr;
    if (data == nullptr || size == 0)
      return;
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    if (size > (std::numeric_limits<std::uintptr_t>::max)() - address)
      return;
    begin_ = reinterpret_cast<std::byte *>(address);
    cur_ = begin_;
    end_ = reinterpret_cast<std::byte *>(address + size);
  }

  void *allocate(std::size_t bytes, std::size_t alignment) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment));
    if (bytes == 0 || cur_ == nullptr)
      return nullptr;
    const auto cur = reinterpret_cast<std::uintptr_t>(cur_);
    const auto end = reinterpret_cast<std::uintptr_t>(end_);
    const auto mask = alignment - 1;
    if (HYLE_LIKELY((cur & mask) == 0)) {
      if (bytes > end - cur)
        return nullptr;
      cur_ = reinterpret_cast<std::byte *>(cur + bytes);
      return reinterpret_cast<void *>(cur);
    }
    const auto padding = (std::uintptr_t{0} - cur) & mask;
    const auto available = end - cur;
    const auto consumed = bytes + padding;
    if (consumed < bytes || consumed > available)
      return nullptr;
    const auto aligned = cur + padding;
    cur_ = reinterpret_cast<std::byte *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
  }

  void clear() noexcept { cur_ = begin_; }
  std::byte *mark() const noexcept { return cur_; }
  bool owns_mark(const std::byte *p) const noexcept {
    if (begin_ == nullptr || p == nullptr)
      return false;
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return address >= reinterpret_cast<std::uintptr_t>(begin_) &&
           address <= reinterpret_cast<std::uintptr_t>(cur_);
  }
  void rewind(std::byte *p) noexcept {
    HYLE_FATAL_ASSERT(begin_ != nullptr && owns_mark(p));
    cur_ = p;
  }
  bool valid() const noexcept { return begin_ != nullptr && end_ > begin_; }
  std::size_t used() const noexcept {
    return valid() ? static_cast<std::size_t>(cur_ - begin_) : 0;
  }
  std::size_t capacity() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - begin_) : 0;
  }
  std::size_t remaining() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - cur_) : 0;
  }

private:
  std::byte *begin_ = nullptr;
  std::byte *cur_ = nullptr;
  std::byte *end_ = nullptr;
};

} // namespace detail

// Non-owning bump region over caller-provided storage. Single-threaded. It
// never allocates, grows, frees individual objects, or falls back to another
// allocator. The storage must outlive the region and everything allocated from
// it.
class FixedRegion {
public:
  FixedRegion() noexcept = default;
  explicit FixedRegion(std::span<std::byte> storage) noexcept
      : cursor_(storage.data(), storage.size()) {}
  FixedRegion(void *data, std::size_t size) noexcept : cursor_(data, size) {}

  // A bitwise move would leave two active cursors over the same external
  // storage.
  FixedRegion(const FixedRegion&) = delete;
  FixedRegion& operator=(const FixedRegion&) = delete;
  FixedRegion(FixedRegion&&) = delete;
  FixedRegion& operator=(FixedRegion&&) = delete;

  bool valid() const noexcept { return cursor_.valid(); }
  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    return cursor_.allocate(bytes, alignment);
  }
  void clear() noexcept { cursor_.clear(); }
  std::size_t used() const noexcept { return cursor_.used(); }
  std::size_t capacity() const noexcept { return cursor_.capacity(); }
  std::size_t remaining() const noexcept { return cursor_.remaining(); }

  Allocator as_allocator() noexcept {
    return Allocator(&fixed_alloc, &fixed_free, &fixed_realloc, this);
  }

private:
  static void *fixed_alloc(void *state, std::size_t size,
                           std::size_t alignment) noexcept {
    return static_cast<FixedRegion *>(state)->allocate(size, alignment);
  }
  static void fixed_free(void *, void *, std::size_t, std::size_t) noexcept {}
  static void *fixed_realloc(void *state, void *ptr, std::size_t old_size,
                             std::size_t new_size,
                             std::size_t alignment) noexcept {
    if (new_size == 0)
      return nullptr;
    auto *self = static_cast<FixedRegion *>(state);
    void *out = self->allocate(new_size, alignment);
    if (out != nullptr && ptr != nullptr)
      std::memcpy(out, ptr, old_size < new_size ? old_size : new_size);
    return out;
  }

  detail::BumpCursor cursor_;
};

struct RegionOptions {
  // Ordinary-block backing allocation targets, including Region's internal
  // header and worst-case alignment padding. A single allocation may force a
  // larger block. Treating these as total backing bytes avoids crossing an
  // allocator size-class boundary merely to store Region metadata.
  std::size_t initial_block_size = 64 * 1024;
  // Four MiB avoids backing-allocator page/segment amplification from many
  // one-MiB blocks on large request scratch, while keeping retained tail slack
  // bounded. Callers with a tighter working-set ceiling can lower it.
  std::size_t max_block_size = 4 * 1024 * 1024;
  std::size_t large_allocation_threshold = 256 * 1024;
  std::size_t size_limit = (std::numeric_limits<std::size_t>::max)();
};

// Owned growable single-threaded region. Allocations are bump-only and
// individual free is a no-op. clear() bulk-releases logical allocations,
// retains ordinary blocks for reuse, and returns dedicated large blocks.
// release() returns every block to the backing allocator. size_limit is a
// strict bound on backing bytes reserved by this Region (headers and alignment
// padding included), not merely on requested payload bytes.
class Region {
public:
  explicit Region(Allocator backing, RegionOptions options = {}) noexcept
      : backing_(backing), options_(options),
        next_block_size_(options.initial_block_size) {
    HYLE_FATAL_ASSERT(backing_.valid());
    HYLE_FATAL_ASSERT(options_.initial_block_size != 0);
    HYLE_FATAL_ASSERT(options_.max_block_size >= options_.initial_block_size);
    HYLE_FATAL_ASSERT(options_.large_allocation_threshold != 0);
  }

  ~Region() { release(); }
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
  Region(Region &&other) noexcept { steal(other); }
  Region &operator=(Region &&other) noexcept {
    if (this != &other) {
      release();
      steal(other);
    }
    return *this;
  }

  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment));
    if (bytes == 0)
      return nullptr;

    if (bytes < options_.large_allocation_threshold) {
      if (void *p = allocate_current(bytes, alignment))
        return account(p, bytes);
    }
    return allocate_slow(bytes, alignment);
  }

  void clear() noexcept {
    record_peak();
    Block **link = &blocks_;
    while (*link != nullptr) {
      Block *b = *link;
      if (b->large) {
        *link = b->next;
        reserved_ -= b->allocation_bytes;
        backing_.deallocate(b, b->allocation_bytes, b->allocation_alignment);
      } else {
        b->cursor = b->begin;
        link = &b->next;
      }
    }
    // Reuse retained ordinary blocks in list order. `reuse_next_` only moves
    // forward during one clear epoch: a block whose remaining tail cannot
    // satisfy a request is never rescanned later in that epoch. This is the
    // same monotonic-block discipline used by mature arenas and bounds total
    // retained-block inspection to O(blocks) per epoch (amortized O(1) per
    // block transition), instead of restarting at the list head and becoming
    // O(blocks^2) over a long-lived multi-block arena.
    select_block(blocks_);
    reuse_next_ = blocks_ != nullptr ? blocks_->next : nullptr;
    used_ = 0;
#if defined(HYLE_TESTING)
    debug_reuse_candidates_ = 0;
#endif
  }

  void release() noexcept {
    record_peak();
    Block *b = blocks_;
    while (b != nullptr) {
      Block *next = b->next;
      backing_.deallocate(b, b->allocation_bytes, b->allocation_alignment);
      b = next;
    }
    blocks_ = nullptr;
    select_block(nullptr);
    reuse_next_ = nullptr;
    used_ = reserved_ = 0;
    next_block_size_ = options_.initial_block_size;
#if defined(HYLE_TESTING)
    debug_reuse_candidates_ = 0;
#endif
  }

  std::size_t bytes_used() const noexcept { return used_; }
  std::size_t reserved() const noexcept { return reserved_; }
  // The completed-epoch high water is folded lazily. Returning the maximum
  // keeps this query exact during an active epoch without adding a peak
  // load/compare/store dependency to every allocation.
  std::size_t peak_used() const noexcept {
    return used_ > peak_used_ ? used_ : peak_used_;
  }
  std::size_t size_limit() const noexcept { return options_.size_limit; }

#if defined(HYLE_TESTING)
  // TEST-ONLY: after clear(), each retained ordinary block is
  // considered at most once before the Region grows again.
  std::size_t debug_reuse_candidates() const noexcept {
    return debug_reuse_candidates_;
  }
#endif

  Allocator as_allocator() noexcept {
    return Allocator(&region_alloc, &region_free, &region_realloc, this);
  }

private:
  struct Block {
    Block *next = nullptr;
    std::size_t allocation_bytes = 0;
    std::size_t allocation_alignment = alignof(std::max_align_t);
    std::byte *begin = nullptr;
    std::byte *cursor = nullptr;
    std::byte *end = nullptr;
    bool large = false;
  };

  static void *allocate_from(Block *b, std::size_t bytes,
                             std::size_t alignment) noexcept {
    const auto cur = reinterpret_cast<std::uintptr_t>(b->cursor);
    const auto end = reinterpret_cast<std::uintptr_t>(b->end);
    const auto mask = alignment - 1;
    if (HYLE_LIKELY((cur & mask) == 0)) {
      if (bytes > end - cur)
        return nullptr;
      b->cursor = reinterpret_cast<std::byte *>(cur + bytes);
      return reinterpret_cast<void *>(cur);
    }
    const auto padding = (std::uintptr_t{0} - cur) & mask;
    const auto available = end - cur;
    const auto consumed = bytes + padding;
    if (consumed < bytes || consumed > available)
      return nullptr;
    const auto aligned = cur + padding;
    b->cursor = reinterpret_cast<std::byte *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
  }

  void *allocate_current(std::size_t bytes, std::size_t alignment) noexcept {
    if (cursor_ == nullptr)
      return nullptr;
    const auto cur = reinterpret_cast<std::uintptr_t>(cursor_);
    const auto end = reinterpret_cast<std::uintptr_t>(end_);
    const auto mask = alignment - 1;
    if (HYLE_LIKELY((cur & mask) == 0)) {
      if (bytes > end - cur)
        return nullptr;
      cursor_ = reinterpret_cast<std::byte *>(cur + bytes);
      return reinterpret_cast<void *>(cur);
    }
    const auto padding = (std::uintptr_t{0} - cur) & mask;
    const auto available = end - cur;
    const auto consumed = bytes + padding;
    if (consumed < bytes || consumed > available)
      return nullptr;
    const auto aligned = cur + padding;
    cursor_ = reinterpret_cast<std::byte *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
  }

  void select_block(Block *b) noexcept {
    current_block_ = b;
    cursor_ = b != nullptr ? b->cursor : nullptr;
    end_ = b != nullptr ? b->end : nullptr;
  }

  HYLE_NOINLINE void *allocate_slow(std::size_t bytes,
                                    std::size_t alignment) noexcept {
    if (current_block_ != nullptr)
      current_block_->cursor = cursor_;

    const bool large = bytes >= options_.large_allocation_threshold;
    if (!large) {
      while (reuse_next_ != nullptr) {
        Block *b = reuse_next_;
        reuse_next_ = b->next;
#if defined(HYLE_TESTING)
        ++debug_reuse_candidates_;
#endif
        HYLE_FATAL_ASSERT(!b->large);
        if (void *p = allocate_from(b, bytes, alignment)) {
          select_block(b);
          return account(p, bytes);
        }
      }
    }

    std::size_t payload = bytes;
    if (!large) {
      std::size_t overhead = 0;
      if (!try_block_overhead(alignment, &overhead))
        return nullptr;
      payload = next_block_size_ > overhead ? next_block_size_ - overhead : 0;
      if (payload < bytes)
        payload = bytes;
    }
    Block *b = add_block(payload, alignment, large);
    if (b == nullptr)
      return nullptr;
    void *p = allocate_from(b, bytes, alignment);
    HYLE_FATAL_ASSERT(p != nullptr);
    if (!large) {
      select_block(b);
      if (next_block_size_ < options_.max_block_size) {
        const std::size_t room = options_.max_block_size - next_block_size_;
        next_block_size_ += next_block_size_ < room ? next_block_size_ : room;
      }
    }
    return account(p, bytes);
  }

  Block *add_block(std::size_t payload, std::size_t request_alignment,
                   bool large) noexcept {
    const std::size_t alignment =
        request_alignment > alignof(Block) ? request_alignment : alignof(Block);
    std::size_t overhead = 0;
    if (!try_block_overhead(request_alignment, &overhead))
      return nullptr;
    if (payload > (std::numeric_limits<std::size_t>::max)() - overhead)
      return nullptr;
    const std::size_t bytes = overhead + payload;
    if (reserved_ > options_.size_limit ||
        bytes > options_.size_limit - reserved_)
      return nullptr;
    void *raw = backing_.allocate(bytes, alignment);
    if (raw == nullptr)
      return nullptr;
    auto *b = ::new (raw) Block;
    b->next = blocks_;
    b->allocation_bytes = bytes;
    b->allocation_alignment = alignment;
    b->large = large;
    const auto first = reinterpret_cast<std::uintptr_t>(b) + sizeof(Block);
    std::uintptr_t begin = 0;
    if (!detail::try_align_up_address(first, request_alignment, &begin)) {
      backing_.deallocate(raw, bytes, alignment);
      return nullptr;
    }
    b->begin = b->cursor = reinterpret_cast<std::byte *>(begin);
    b->end = b->begin + payload;
    blocks_ = b;
    reserved_ += bytes;
    return b;
  }

  static bool try_block_overhead(std::size_t request_alignment,
                                 std::size_t *out) noexcept {
    if (request_alignment - 1 >
        (std::numeric_limits<std::size_t>::max)() - sizeof(Block))
      return false;
    *out = sizeof(Block) + (request_alignment - 1);
    return true;
  }

  void *account(void *p, std::size_t bytes) noexcept {
    // Every successful request consumes at least `bytes` from finite retained
    // backing, so used_ + bytes cannot overflow while Region invariants hold.
    // Peak folding is deferred to clear/release/query to keep the bump path to
    // one dependent counter update.
    used_ += bytes;
    return p;
  }

  void record_peak() noexcept {
    if (used_ > peak_used_)
      peak_used_ = used_;
  }

  static void *region_alloc(void *state, std::size_t size,
                            std::size_t alignment) noexcept {
    return static_cast<Region *>(state)->allocate(size, alignment);
  }
  static void region_free(void *, void *, std::size_t, std::size_t) noexcept {}
  static void *region_realloc(void *state, void *ptr, std::size_t old_size,
                              std::size_t new_size,
                              std::size_t alignment) noexcept {
    if (new_size == 0)
      return nullptr;
    auto *self = static_cast<Region *>(state);
    void *out = self->allocate(new_size, alignment);
    if (out != nullptr && ptr != nullptr)
      std::memcpy(out, ptr, old_size < new_size ? old_size : new_size);
    return out;
  }

  void steal(Region &other) noexcept {
    backing_ = other.backing_;
    options_ = other.options_;
    blocks_ = other.blocks_;
    current_block_ = other.current_block_;
    cursor_ = other.cursor_;
    end_ = other.end_;
    next_block_size_ = other.next_block_size_;
    used_ = other.used_;
    reserved_ = other.reserved_;
    peak_used_ = other.peak_used_;
    reuse_next_ = other.reuse_next_;
#if defined(HYLE_TESTING)
    debug_reuse_candidates_ = other.debug_reuse_candidates_;
#endif
    other.blocks_ = nullptr;
    other.select_block(nullptr);
    other.reuse_next_ = nullptr;
    other.used_ = other.reserved_ = other.peak_used_ = 0;
    other.next_block_size_ = other.options_.initial_block_size;
#if defined(HYLE_TESTING)
    other.debug_reuse_candidates_ = 0;
#endif
  }

  Allocator backing_;
  RegionOptions options_;
  Block *blocks_ = nullptr;
  Block *current_block_ = nullptr;
  std::byte *cursor_ = nullptr;
  std::byte *end_ = nullptr;
  std::size_t next_block_size_ = 0;
  std::size_t used_ = 0;
  std::size_t reserved_ = 0;
  std::size_t peak_used_ = 0;
  // First retained ordinary block not yet considered in the current clear
  // epoch. It is deliberately independent of `blocks_`: newly grown blocks
  // are prepended to the ownership list but must not make already exhausted
  // blocks eligible for another scan.
  Block *reuse_next_ = nullptr;
#if defined(HYLE_TESTING)
  std::size_t debug_reuse_candidates_ = 0;
#endif
};

} // namespace hyle::mem
