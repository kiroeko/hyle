#pragma once

#include <atomic>
#include <cstddef>
#include <new>

#include "hyle/graph/executable.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::detail {

// Graph owns only family/lifetime composition. Mem owns the strict byte
// admission algorithm. Each successful metadata allocation retains this
// allocator owner, so independently retained signatures need no Plan cycle.
class InvocationStorageState final
    : public util::RefCounted<InvocationStorageState> {
public:
  // Returns the constructing +1, immediately adopted by compile_plan. Do not
  // instantiate constrained Ref<Self> in an incomplete class declaration.
  static Result<InvocationStorageState *>
  create(mem::AllocatorRef backing, InvocationStorageLimits limits) noexcept {
    void *storage = backing.allocate(sizeof(InvocationStorageState),
                                     alignof(InvocationStorageState));
    if (storage == nullptr)
      return std::unexpected(Status::ResourceExhausted(
          "graph invocation budget allocation failed"));
    return ::new (storage) InvocationStorageState(backing, limits);
  }

  void destroy() noexcept {
    const auto backing = backing_;
    this->~InvocationStorageState();
    backing.deallocate(this, sizeof(InvocationStorageState),
                       alignof(InvocationStorageState));
  }

  mem::AllocatorRef allocator() noexcept {
    return {&allocate, &deallocate, &reallocate, this};
  }
  std::optional<mem::QuotaAllocator::Reservation>
  reserve(std::size_t bytes) noexcept {
    return quota_.try_reserve_bytes(bytes);
  }
  bool rejects_unknown() const noexcept {
    return limits_.max_bytes != SIZE_MAX &&
           limits_.unknown == UnknownStoragePolicy::kReject;
  }
  void note_unknown() noexcept {
    unknown_.store(true, std::memory_order_relaxed);
  }
  InvocationStorageUsage usage() const noexcept {
    return {quota_.used(), quota_.peak(), quota_.limit(),
            unknown_.load(std::memory_order_relaxed)};
  }

private:
  InvocationStorageState(mem::AllocatorRef backing,
                         InvocationStorageLimits limits) noexcept
      : backing_(backing), limits_(limits), quota_(backing, limits.max_bytes) {}

  static void *allocate(void *opaque, std::size_t bytes,
                        std::size_t alignment) noexcept {
    auto &self = *static_cast<InvocationStorageState *>(opaque);
    void *result = self.quota_.get().allocate(bytes, alignment);
    if (result != nullptr)
      self.incref();
    return result;
  }
  static void deallocate(void *opaque, void *pointer, std::size_t bytes,
                         std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return;
    auto &self = *static_cast<InvocationStorageState *>(opaque);
    self.quota_.get().deallocate(pointer, bytes, alignment);
    self.decref();
  }
  static void *reallocate(void *opaque, void *pointer, std::size_t old_bytes,
                          std::size_t new_bytes,
                          std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return allocate(opaque, new_bytes, alignment);
    auto &self = *static_cast<InvocationStorageState *>(opaque);
    void *result =
        self.quota_.get().reallocate(pointer, old_bytes, new_bytes, alignment);
    if (new_bytes == 0)
      self.decref();
    return result;
  }

  mem::AllocatorRef backing_;
  InvocationStorageLimits limits_;
  mem::QuotaAllocator quota_;
  std::atomic<bool> unknown_{false};
};

} // namespace hyle::graph::detail
