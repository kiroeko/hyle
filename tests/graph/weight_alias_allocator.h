#pragma once

#include <atomic>
#include <hyle/mem/allocator.h>

namespace hyle::graph::test {

// Only WeightSlots uses this allocator; Graph/MLX runtime allocation stays live.
struct WeightAliasAllocator final {
  std::atomic<bool> deny{false};
  std::atomic<unsigned> calls{0}, rejected{0};

  static void *allocate(void *opaque, std::size_t bytes,
                        std::size_t alignment) noexcept {
    auto &self = *static_cast<WeightAliasAllocator *>(opaque);
    ++self.calls;
    if (self.deny.load()) {
      ++self.rejected;
      return nullptr;
    }
    return mem::system_allocator().allocate(bytes, alignment);
  }
  static void free(void *, void *pointer, std::size_t bytes,
                   std::size_t alignment) noexcept {
    mem::system_allocator().deallocate(pointer, bytes, alignment);
  }
  static void *resize(void *opaque, void *pointer, std::size_t old_bytes,
                      std::size_t new_bytes, std::size_t alignment) noexcept {
    if (pointer == nullptr) return allocate(opaque, new_bytes, alignment);
    if (new_bytes == 0) {
      free(opaque, pointer, old_bytes, alignment);
      return nullptr;
    }
    auto &self = *static_cast<WeightAliasAllocator *>(opaque);
    ++self.calls;
    if (self.deny.load()) {
      ++self.rejected;
      return nullptr;
    }
    return mem::system_allocator().reallocate(pointer, old_bytes, new_bytes, alignment);
  }
  mem::AllocatorRef get() noexcept { return {allocate, free, resize, this}; }
};

} // namespace hyle::graph::test
