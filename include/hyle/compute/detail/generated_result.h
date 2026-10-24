#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/compute/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {

// An opaque reference to the original result slot, not another tensor/result
// system or a readiness observation. Copies prevent that same slot's overwrite.
// Only the original successful Completion authorizes access to its payload.
class ResultStorageOwner final {
public:
  ResultStorageOwner() noexcept = default;
  ResultStorageOwner(const ResultStorageOwner &other) noexcept : storage_(other.storage_) {
    retain_result(storage_);
  }
  ResultStorageOwner &operator=(const ResultStorageOwner &other) noexcept {
    if (this != &other) {
      retain_result(other.storage_);
      release_result(storage_);
      storage_ = other.storage_;
    }
    return *this;
  }
  ResultStorageOwner(ResultStorageOwner &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)) {}
  ResultStorageOwner &operator=(ResultStorageOwner &&other) noexcept {
    if (this != &other) {
      reset();
      storage_ = std::exchange(other.storage_, nullptr);
    }
    return *this;
  }
  ~ResultStorageOwner() { reset(); }
  bool valid() const noexcept { return storage_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  ResultStorage *get() const noexcept { return storage_; }
  void reset() noexcept { release_result(std::exchange(storage_, nullptr)); }

private:
  friend struct ResultAccess;
  explicit ResultStorageOwner(ResultStorage *storage) noexcept : storage_(storage) {}
  ResultStorage *storage_ = nullptr;
};

// One move-only handoff. A native adapter stores only owner in its original
// slot and returns completion to the original execution cursor. Keeping a
// Completion inside a copied slot owner could create an invocation refcycle.
struct ResultTransfer final {
  ResultTransfer() noexcept = default;
  ResultTransfer(ResultStorageOwner retained, Completion accepted) noexcept
      : owner(std::move(retained)), completion(std::move(accepted)) {}
  ResultTransfer(const ResultTransfer &) = delete;
  ResultTransfer &operator=(const ResultTransfer &) = delete;
  ResultTransfer(ResultTransfer &&) noexcept = default;
  ResultTransfer &operator=(ResultTransfer &&other) noexcept {
    if (this != &other) {
      // Same release order as OwnedResult: abandon/retire the original work
      // before releasing this handle's storage reference. No hidden wait.
      completion = Completion{};
      owner.reset();
      owner = std::move(other.owner);
      completion = std::move(other.completion);
    }
    return *this;
  }
  // Declaration order makes destruction release completion before owner.
  ResultStorageOwner owner;
  Completion completion;
};

// Trusted same-build observation/ownership adapters only. Transferring or
// replacing the original completion never creates another result slot.
struct ResultAccess final {
  template<class T>
  static ResultTransfer take_storage(OwnedResult<T> &&result) noexcept {
    return {ResultStorageOwner(std::exchange(result.storage_, nullptr)),
            std::move(result.completion_)};
  }
  // No wait, poll, type test or ready permission is implied. The native
  // adapter knows T and must first establish the original success boundary.
  static const void *payload(const ResultStorageOwner &owner) noexcept {
    return owner.valid() ? result_payload(owner.get()) : nullptr;
  }
  template<class T>
  static ResultStorage *storage(const OwnedResult<T> &result) noexcept {
    return result.storage_;
  }
  template<class T>
  static Completion &completion(OwnedResult<T> &result) noexcept {
    return result.completion_;
  }
  template<class T>
  static Completion replace_completion(OwnedResult<T> &result,
                                       Completion replacement) noexcept {
    Completion original=std::move(result.completion_);
    result.completion_=std::move(replacement);
    return original;
  }
};

using ResultDestroyFn = void (*)(void *) noexcept;
HYLE_COMPUTE_API Result<ResultStorage *>
create_result_storage(mem::AllocatorRef allocator, std::size_t size,
                      std::size_t alignment, ResultDestroyFn destroy) noexcept;
HYLE_COMPUTE_API Status begin_result(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API Status can_detach_result(
    ResultStorage *storage, const ResultStorageOwner &known_private) noexcept;
HYLE_COMPUTE_API void discard_retired_result_payload(ResultStorage *storage) noexcept;
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
  // Observation only, under the original producer's exclusive cold-mutation
  // gate. The optional owner must be this exact slot's one private reference;
  // no other escaped reader is discounted. This reserves no write admission.
  Status can_detach_retired(
      const ResultStorageOwner &known_private = {}) const noexcept {
    return can_detach_result(storage_, known_private);
  }
  // After the same uninterrupted cold gate's successful preflight and release
  // of its private owner. Destroys only the retired payload, without begin(),
  // changing generation or invalidating any independently escaped owner.
  void discard_retired_payload() noexcept {
    discard_retired_result_payload(storage_);
  }

private:
  explicit ResultSlot(ResultStorage *storage) noexcept : storage_(storage) {}
  ResultStorage *storage_ = nullptr;
};

template <class T>
Status can_detach_retired(const ResultSlot<T> &slot,
    const ResultStorageOwner &known_private = {}) noexcept {
  return slot.can_detach_retired(known_private);
}
template <class T>
void discard_retired_payload(ResultSlot<T> &slot) noexcept {
  slot.discard_retired_payload();
}

} // namespace hyle::compute::detail
