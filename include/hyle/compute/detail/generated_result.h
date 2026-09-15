#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/compute/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {

using ResultDestroyFn = void (*)(void *) noexcept;
HYLE_COMPUTE_API Result<ResultStorage *>
create_result_storage(mem::AllocatorRef allocator, std::size_t size,
                      std::size_t alignment, ResultDestroyFn destroy) noexcept;
HYLE_COMPUTE_API Status begin_result(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API void rollback_result(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API bool result_constructed(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API void mark_result_constructed(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API Status publish_result(ResultStorage *storage,
                                       Completion &completion) noexcept;

template <class T> class ResultSlot;

// Producer-only write admission, obtained before accepting any native work.
// After acceptance, publish the real Completion even if it carries a late
// error. Never abandon an accepted prefix by returning an ordinary Status.
template <class T> class [[nodiscard]] ResultWrite final {
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_destructible_v<T>);

public:
  ResultWrite() noexcept = default;
  ResultWrite(const ResultWrite &) = delete;
  ResultWrite &operator=(const ResultWrite &) = delete;
  ResultWrite(ResultWrite &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)) {}
  ResultWrite &operator=(ResultWrite &&other) noexcept {
    if (this != &other) {
      rollback_result(storage_);
      storage_ = std::exchange(other.storage_, nullptr);
    }
    return *this;
  }
  ~ResultWrite() { rollback_result(storage_); }

  bool valid() const noexcept { return storage_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  Status emplace(T &&owner) noexcept {
    if (!valid() || result_constructed(storage_))
      return Status::FailedPrecondition("result write is invalid or populated");
    std::construct_at(static_cast<T *>(result_payload(storage_)),
                      std::move(owner));
    mark_result_constructed(storage_);
    return Status::Ok();
  }
  // On error the caller still owns completion, including any accepted work.
  Result<OwnedResult<T>> publish(Completion &&completion) noexcept {
    const Status status = publish_result(storage_, completion);
    if (!status.ok())
      return std::unexpected(status);
    return OwnedResult<T>(std::exchange(storage_, nullptr),
                          std::move(completion));
  }

private:
  friend class ResultSlot<T>;
  explicit ResultWrite(ResultStorage *storage) noexcept : storage_(storage) {}
  ResultStorage *storage_ = nullptr;
};

// Optional route implementation helper; native Programs need not share an
// executor. One cold allocation holds metadata and T. The injected allocator
// must outlive all submissions and leases; T owns its native payload itself.
// begin is thread-safe and nonblocking; moving/destroying the slot requires
// external synchronization. A ResultWrite is exclusive and thread-compatible.
template <class T> class [[nodiscard]] ResultSlot final {
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_destructible_v<T>);

public:
  ResultSlot() noexcept = default;
  ResultSlot(const ResultSlot &) = delete;
  ResultSlot &operator=(const ResultSlot &) = delete;
  ResultSlot(ResultSlot &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)) {}
  ResultSlot &operator=(ResultSlot &&other) noexcept {
    if (this != &other) {
      release_result(storage_);
      storage_ = std::exchange(other.storage_, nullptr);
    }
    return *this;
  }
  ~ResultSlot() { release_result(storage_); }

  static Result<ResultSlot> create(mem::AllocatorRef allocator) noexcept {
    auto storage = create_result_storage(
        allocator, sizeof(T), alignof(T),
        [](void *owner) noexcept { std::destroy_at(static_cast<T *>(owner)); });
    if (!storage)
      return std::unexpected(std::move(storage).error());
    return ResultSlot(*storage);
  }
  bool valid() const noexcept { return storage_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  // Concurrent begin attempts are checked. A retained old result prevents
  // overwrite; use a second slot to submit while that result is still held.
  Result<ResultWrite<T>> begin() noexcept {
    const Status status = begin_result(storage_);
    if (!status.ok())
      return std::unexpected(status);
    return ResultWrite<T>(storage_);
  }

private:
  explicit ResultSlot(ResultStorage *storage) noexcept : storage_(storage) {}
  ResultStorage *storage_ = nullptr;
};

} // namespace hyle::compute::detail
