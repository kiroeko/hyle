#pragma once

#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"

namespace hyle::compute {

enum class CompletionPoll : unsigned char {
  kPending = 0,
  kComplete,
};

namespace detail {
struct CompletionAccess;
struct CompletionVTable;
} // namespace detail

// Move-only terminal owner returned after a Queue accepts work. Immediate CPU
// completions stay inline; asynchronous implementations own their event state
// through a static vtable. Destroying a Completion never performs a hidden
// wait, so a backend must keep late work and required runtime state safe.
class [[nodiscard]] Completion final {
public:
  Completion() noexcept = default;
  Completion(const Completion &) = delete;
  Completion &operator=(const Completion &) = delete;
  HYLE_COMPUTE_API Completion(Completion &&other) noexcept;
  HYLE_COMPUTE_API Completion &operator=(Completion &&other) noexcept;
  HYLE_COMPUTE_API ~Completion();

  HYLE_COMPUTE_API static Completion
  completed(Status terminal = Status::Ok()) noexcept;

  bool valid() const noexcept { return valid_; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API Result<CompletionPoll> poll() const noexcept;
  HYLE_COMPUTE_API Status wait() const noexcept;
  HYLE_COMPUTE_API Status request_cancel() noexcept;

private:
  Completion(const detail::CompletionVTable *vtable, void *state) noexcept;
  explicit Completion(Status terminal) noexcept;
  void reset() noexcept;

  friend struct detail::CompletionAccess;
  const detail::CompletionVTable *vtable_ = nullptr;
  void *state_ = nullptr;
  Status terminal_{};
  bool valid_ = false;
};

} // namespace hyle::compute
