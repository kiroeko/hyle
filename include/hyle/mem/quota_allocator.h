#pragma once

#include <atomic>
#include <cstddef>
#include <limits>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {

// Thread-safe strict byte-budget decorator over any sized-free Allocator.
//
// The quota accounts REQUESTED live bytes, not the backing allocator's
// usable-size or metadata overhead. A reservation is acquired before entering
// the backing, so concurrent successful calls can never make used() exceed
// limit(). Backing OOM rolls the reservation back. The Allocator returned by
// get() borrows this object and is valid only while *this remains alive.
//
// As with every HYLE Allocator, deallocate/reallocate must receive the original
// requested size. Passing a different size is a caller contract violation and
// corrupts accounting.
class QuotaAllocator {
public:
  explicit QuotaAllocator(
      Allocator backing,
      std::size_t limit = (std::numeric_limits<std::size_t>::max)()) noexcept
      : backing_(backing), limit_(limit) {
    HYLE_FATAL_ASSERT(backing_.valid());
  }

  ~QuotaAllocator() {
    HYLE_FATAL_ASSERT(accounted_.load(std::memory_order_relaxed) == 0);
    HYLE_FATAL_ASSERT(live_.load(std::memory_order_relaxed) == 0);
  }

  QuotaAllocator(const QuotaAllocator&) = delete;
  QuotaAllocator& operator=(const QuotaAllocator&) = delete;
  QuotaAllocator(QuotaAllocator&&) = delete;
  QuotaAllocator& operator=(QuotaAllocator&&) = delete;

  Allocator get() noexcept {
    return Allocator(&quota_alloc, &quota_free, &quota_realloc, this);
  }

  std::size_t used() const noexcept {
    return live_.load(std::memory_order_relaxed);
  }
  std::size_t peak() const noexcept {
    return peak_.load(std::memory_order_relaxed);
  }
  std::size_t limit() const noexcept { return limit_; }
  std::size_t remaining() const noexcept {
    return limit_ - accounted_.load(std::memory_order_relaxed);
  }

private:
  bool reserve(std::size_t bytes) noexcept {
    if (bytes == 0)
      return true;
    std::size_t cur = accounted_.load(std::memory_order_relaxed);
    for (;;) {
      if (cur > limit_ || bytes > limit_ - cur)
        return false;
      const std::size_t next = cur + bytes;
      if (accounted_.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                           std::memory_order_relaxed))
        return true;
    }
  }

  void unreserve(std::size_t bytes) noexcept {
    if (bytes == 0)
      return;
    const std::size_t before =
        accounted_.fetch_sub(bytes, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT(before >= bytes);
  }

  void commit_growth(std::size_t bytes) noexcept {
    if (bytes == 0)
      return;
    const std::size_t next =
        live_.fetch_add(bytes, std::memory_order_acq_rel) + bytes;
    std::size_t peak = peak_.load(std::memory_order_relaxed);
    while (next > peak &&
           !peak_.compare_exchange_weak(peak, next, std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
    }
  }

  void commit_shrink(std::size_t bytes) noexcept {
    if (bytes == 0)
      return;
    const std::size_t before =
        live_.fetch_sub(bytes, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT(before >= bytes);
    unreserve(bytes);
  }

  static void *quota_alloc(void *state, std::size_t size,
                           std::size_t alignment) noexcept {
    auto *self = static_cast<QuotaAllocator *>(state);
    if (size == 0 || !self->reserve(size))
      return nullptr;
    void *p = self->backing_.allocate(size, alignment);
    if (p == nullptr)
      self->unreserve(size);
    else
      self->commit_growth(size);
    return p;
  }

  static void quota_free(void *state, void *ptr, std::size_t size,
                         std::size_t alignment) noexcept {
    if (ptr == nullptr)
      return;
    auto *self = static_cast<QuotaAllocator *>(state);
    self->backing_.deallocate(ptr, size, alignment);
    self->commit_shrink(size);
  }

  static void *quota_realloc(void *state, void *ptr, std::size_t old_size,
                             std::size_t new_size,
                             std::size_t alignment) noexcept {
    auto *self = static_cast<QuotaAllocator *>(state);
    if (ptr == nullptr)
      return quota_alloc(state, new_size, alignment);
    if (new_size == 0) {
      self->backing_.reallocate(ptr, old_size, 0, alignment);
      self->commit_shrink(old_size);
      return nullptr;
    }

    if (new_size > old_size) {
      const std::size_t growth = new_size - old_size;
      if (!self->reserve(growth))
        return nullptr;
      void *out = self->backing_.reallocate(ptr, old_size, new_size, alignment);
      if (out == nullptr)
        self->unreserve(growth);
      else
        self->commit_growth(growth);
      return out;
    }

    void *out = self->backing_.reallocate(ptr, old_size, new_size, alignment);
    if (out != nullptr && old_size > new_size)
      self->commit_shrink(old_size - new_size);
    return out;
  }

  Allocator backing_;
  const std::size_t limit_;
  // accounted_ includes live bytes plus reservations currently inside the
  // backing call; live_ includes only successful allocations and is what
  // used()/peak() expose.
  std::atomic<std::size_t> accounted_{0};
  std::atomic<std::size_t> live_{0};
  std::atomic<std::size_t> peak_{0};
};

} // namespace hyle::mem
