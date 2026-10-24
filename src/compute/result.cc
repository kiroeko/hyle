#include "hyle/compute/detail/generated_result.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <memory>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::detail {

struct ResultStorage final : util::RefCounted<ResultStorage> {
  mem::AllocatorRef allocator{};
  std::size_t allocation_size = 0;
  std::size_t allocation_alignment = 0;
  std::size_t payload_offset = 0;
  ResultDestroyFn destroy_payload = nullptr;
  CompletionRetention retention{};
  std::atomic<bool> busy{false};
  bool constructed = false;

  void clear_payload() noexcept {
    if (constructed) {
      destroy_payload(result_payload(this));
      constructed = false;
    }
  }
  void destroy() noexcept {
    clear_payload();
    const auto owner = allocator;
    const auto bytes = allocation_size;
    const auto alignment = allocation_alignment;
    std::destroy_at(this);
    owner.deallocate(this, bytes, alignment);
  }
};

Result<ResultStorage *>
create_result_storage(mem::AllocatorRef allocator, std::size_t size,
                      std::size_t alignment, ResultDestroyFn destroy) noexcept {
  if (!allocator.valid() || size == 0 || !std::has_single_bit(alignment) ||
      destroy == nullptr)
    return std::unexpected(Status::InvalidArgument("invalid result storage"));
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (alignment - 1 > limit - sizeof(ResultStorage))
    return std::unexpected(
        Status::ResourceExhausted("result alignment overflows"));
  const auto offset =
      (sizeof(ResultStorage) + alignment - 1) & ~(alignment - 1);
  if (size > limit - offset)
    return std::unexpected(
        Status::ResourceExhausted("result storage overflows"));
  const auto allocation_alignment = std::max(alignment, alignof(ResultStorage));
  void *memory = allocator.allocate(offset + size, allocation_alignment);
  if (memory == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("result storage allocation failed"));
  auto *storage = std::construct_at(static_cast<ResultStorage *>(memory));
  storage->allocator = allocator;
  storage->allocation_size = offset + size;
  storage->allocation_alignment = allocation_alignment;
  storage->payload_offset = offset;
  storage->destroy_payload = destroy;
  return storage;
}

void retain_result(ResultStorage *storage) noexcept {
  if (storage != nullptr)
    storage->incref();
}

void release_result(ResultStorage *storage) noexcept {
  if (storage != nullptr)
    storage->decref();
}

void *result_payload(ResultStorage *storage) noexcept {
  return reinterpret_cast<std::byte *>(storage) + storage->payload_offset;
}

Status begin_result(ResultStorage *storage) noexcept {
  if (storage == nullptr)
    return Status::FailedPrecondition("result slot is invalid");
  bool expected = false;
  if (!storage->busy.compare_exchange_strong(expected, true,
                                             std::memory_order_acquire))
    return Status::FailedPrecondition("result slot is in flight");
  // Once unique, no new outside reference can form. Admission excludes a
  // second writer; the acquire observes destruction by the final old reader.
  if (!storage->is_unique()) {
    storage->busy.store(false, std::memory_order_release);
    return Status::FailedPrecondition("result is still retained");
  }
  storage->clear_payload();
  storage->incref(); // transferred from write to OwnedResult on publish
  return Status::Ok();
}

Status can_detach_result(ResultStorage *storage,
                         const ResultStorageOwner &known_private) noexcept {
  if (storage == nullptr)
    return Status::FailedPrecondition("result slot is invalid");
  if (known_private.valid() && known_private.get() != storage)
    return Status::InvalidArgument("private result owner belongs to another slot");
  if (storage->busy.load(std::memory_order_acquire))
    return Status::FailedPrecondition("result slot is in flight");
  const auto expected = known_private.valid() ? 2u : 1u;
  if (storage->use_count() != expected)
    return Status::FailedPrecondition("result is still retained");
  // Pair final old-reader release with this observational reference check.
  // The caller's existing producer gate excludes begin/private-owner copying
  // throughout validation and commit; this check does not claim that gate.
  std::atomic_thread_fence(std::memory_order_acquire);
  return Status::Ok();
}

void discard_retired_result_payload(ResultStorage *storage) noexcept {
  HYLE_FATAL_ASSERT(storage != nullptr);
  HYLE_FATAL_ASSERT(!storage->busy.load(std::memory_order_acquire));
  HYLE_FATAL_ASSERT(storage->is_unique());
  storage->clear_payload();
}

void rollback_result(ResultStorage *storage) noexcept {
  if (storage == nullptr)
    return;
  storage->clear_payload();
  storage->busy.store(false, std::memory_order_release);
  storage->decref();
}

bool result_constructed(ResultStorage *storage) noexcept {
  return storage->constructed;
}

void mark_result_constructed(ResultStorage *storage) noexcept {
  storage->constructed = true;
}

Status publish_result(ResultStorage *storage, Completion &completion) noexcept {
  if (storage == nullptr || !storage->constructed || !completion.valid())
    return Status::FailedPrecondition(
        "result publish needs an owner and completion");
  storage->incref(); // real terminal/abandonment retains the native owner
  storage->retention.own = {storage,
                            [](void *state, const Status &) noexcept {
                              static_cast<ResultStorage *>(state)->busy.store(
                                  false, std::memory_order_release);
                            },
                            [](void *state) noexcept {
                              static_cast<ResultStorage *>(state)->decref();
                            }};
  const Status status =
      CompletionAccess::attach_retention(completion, storage->retention);
  if (!status.ok()) {
    storage->retention.own = {};
    storage->decref();
  }
  return status;
}

} // namespace hyle::compute::detail
