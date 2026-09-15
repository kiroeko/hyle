#pragma once

#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/core/compiler.h"

namespace hyle::compute {

namespace detail {
struct ResultStorage;
template <class T> class ResultWrite;
HYLE_COMPUTE_API void retain_result(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API void release_result(ResultStorage *storage) noexcept;
HYLE_COMPUTE_API void *result_payload(ResultStorage *storage) noexcept;
} // namespace detail

template <class T> class OwnedResult;

// A retained native owning handle, available only after successful completion.
// T's runtime defines how to read its data; this does not turn T into a Mem
// allocation or add a host copy. Copies retain the same immutable handle.
// Distinct leases may be released on different threads only when T and the
// injected allocator permit destruction there. Each handle is
// thread-compatible.
template <class T> class [[nodiscard]] ResultLease final {
public:
  ResultLease() noexcept = default;
  ResultLease(const ResultLease &other) noexcept : storage_(other.storage_) {
    detail::retain_result(storage_);
  }
  ResultLease &operator=(const ResultLease &other) noexcept {
    if (this != &other) {
      detail::retain_result(other.storage_);
      detail::release_result(storage_);
      storage_ = other.storage_;
    }
    return *this;
  }
  ResultLease(ResultLease &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)) {}
  ResultLease &operator=(ResultLease &&other) noexcept {
    if (this != &other) {
      detail::release_result(storage_);
      storage_ = std::exchange(other.storage_, nullptr);
    }
    return *this;
  }
  ~ResultLease() { detail::release_result(storage_); }

  bool valid() const noexcept { return storage_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  const T *get() const noexcept {
    return storage_ != nullptr
               ? static_cast<const T *>(detail::result_payload(storage_))
               : nullptr;
  }
  const T &operator*() const noexcept {
    HYLE_FATAL_ASSERT(valid());
    return *get();
  }
  const T *operator->() const noexcept {
    HYLE_FATAL_ASSERT(valid());
    return get();
  }

private:
  friend class OwnedResult<T>;
  explicit ResultLease(detail::ResultStorage *storage) noexcept
      : storage_(storage) {
    detail::retain_result(storage_);
  }
  detail::ResultStorage *storage_ = nullptr;
};

// Move-only accepted submission and its native result owner. Destruction never
// hides a wait: a pending Completion retains the result until real retirement.
// As with Completion, do not concurrently settle or move this same handle.
template <class T> class [[nodiscard]] OwnedResult final {
public:
  OwnedResult() noexcept = default;
  OwnedResult(const OwnedResult &) = delete;
  OwnedResult &operator=(const OwnedResult &) = delete;
  OwnedResult(OwnedResult &&other) noexcept
      : completion_(std::move(other.completion_)),
        storage_(std::exchange(other.storage_, nullptr)) {}
  OwnedResult &operator=(OwnedResult &&other) noexcept {
    if (this != &other) {
      reset();
      completion_ = std::move(other.completion_);
      storage_ = std::exchange(other.storage_, nullptr);
    }
    return *this;
  }
  ~OwnedResult() { reset(); }

  bool valid() const noexcept { return storage_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  const Completion &completion() const noexcept { return completion_; }
  Result<CompletionPoll> poll() const noexcept { return completion_.poll(); }
  Status request_cancel() noexcept { return completion_.request_cancel(); }

  Result<ResultLease<T>> wait() const noexcept {
    const Status terminal = completion_.wait();
    if (!terminal.ok())
      return std::unexpected(terminal);
    return ResultLease<T>(storage_);
  }
  // Nonblocking. Pending work and failed work never publish a ready lease.
  Result<ResultLease<T>> result() const noexcept {
    auto state = completion_.poll();
    if (!state)
      return std::unexpected(std::move(state).error());
    if (*state != CompletionPoll::kComplete)
      return std::unexpected(Status::FailedPrecondition("result is pending"));
    return ResultLease<T>(storage_);
  }

private:
  friend class detail::ResultWrite<T>;
  OwnedResult(detail::ResultStorage *storage, Completion &&completion) noexcept
      : completion_(std::move(completion)), storage_(storage) {}
  void reset() noexcept {
    completion_ = Completion{};
    detail::release_result(std::exchange(storage_, nullptr));
  }
  Completion completion_{};
  detail::ResultStorage *storage_ = nullptr;
};

} // namespace hyle::compute
