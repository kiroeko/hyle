#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "hyle/macros.h"
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
    if (size > std::numeric_limits<std::uintptr_t>::max() - address)
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
    std::uintptr_t aligned = 0;
    if (!try_align_up_address(cur, alignment, &aligned))
      return nullptr;
    const auto end = reinterpret_cast<std::uintptr_t>(end_);
    if (aligned > end || bytes > end - aligned)
      return nullptr;
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
  HYLE_DISABLE_COPY_AND_MOVE(FixedRegion);

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
  std::size_t initial_block_size = 64 * 1024;
  // Four MiB avoids backing-allocator page/segment amplification from many
  // one-MiB blocks on large request scratch, while keeping retained tail slack
  // bounded. Callers with a tighter working-set ceiling can lower it.
  std::size_t max_block_size = 4 * 1024 * 1024;
  std::size_t large_allocation_threshold = 256 * 1024;
  std::size_t size_limit = std::numeric_limits<std::size_t>::max();
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
  HYLE_DISABLE_COPY(Region);
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
    select_block(blocks_);
    used_ = 0;
  }

  void release() noexcept {
    Block *b = blocks_;
    while (b != nullptr) {
      Block *next = b->next;
      backing_.deallocate(b, b->allocation_bytes, b->allocation_alignment);
      b = next;
    }
    blocks_ = nullptr;
    select_block(nullptr);
    used_ = reserved_ = 0;
    next_block_size_ = options_.initial_block_size;
  }

  std::size_t bytes_used() const noexcept { return used_; }
  std::size_t reserved() const noexcept { return reserved_; }
  std::size_t peak_used() const noexcept { return peak_used_; }
  std::size_t size_limit() const noexcept { return options_.size_limit; }

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
    std::uintptr_t aligned = 0;
    if (!detail::try_align_up_address(cur, alignment, &aligned))
      return nullptr;
    const auto end = reinterpret_cast<std::uintptr_t>(b->end);
    if (aligned > end || bytes > end - aligned)
      return nullptr;
    b->cursor = reinterpret_cast<std::byte *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
  }

  void *allocate_current(std::size_t bytes, std::size_t alignment) noexcept {
    if (cursor_ == nullptr)
      return nullptr;
    const auto cur = reinterpret_cast<std::uintptr_t>(cursor_);
    std::uintptr_t aligned = 0;
    if (!detail::try_align_up_address(cur, alignment, &aligned))
      return nullptr;
    const auto end = reinterpret_cast<std::uintptr_t>(end_);
    if (aligned > end || bytes > end - aligned)
      return nullptr;
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
      for (Block *b = blocks_; b != nullptr; b = b->next) {
        if (b == current_block_ || b->large)
          continue;
        if (void *p = allocate_from(b, bytes, alignment)) {
          select_block(b);
          return account(p, bytes);
        }
      }
    }

    std::size_t payload = bytes;
    if (!large) {
      payload = next_block_size_;
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
    if (request_alignment - 1 >
        std::numeric_limits<std::size_t>::max() - sizeof(Block))
      return nullptr;
    const std::size_t overhead = sizeof(Block) + (request_alignment - 1);
    if (payload > std::numeric_limits<std::size_t>::max() - overhead)
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

  void *account(void *p, std::size_t bytes) noexcept {
    // requested live bytes cannot exceed reserved backing bytes; reaching this
    // assertion therefore indicates internal accounting corruption.
    HYLE_FATAL_ASSERT(used_ <= std::numeric_limits<std::size_t>::max() - bytes);
    used_ += bytes;
    if (used_ > peak_used_)
      peak_used_ = used_;
    return p;
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
    other.blocks_ = nullptr;
    other.select_block(nullptr);
    other.used_ = other.reserved_ = other.peak_used_ = 0;
    other.next_block_size_ = other.options_.initial_block_size;
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
};

} // namespace hyle::mem
