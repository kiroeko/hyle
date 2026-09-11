#pragma once

#include <utility>

#include "hyle/error/status.h"

namespace hyle::compute {

// Terminal state returned by a Queue submission. The current CPU provider
// completes work before
// submit returns, but callers still use the same object/lifetime boundary that
// asynchronous backends will implement. An invalid/moved-from Completion is a
// programming-state error, not a successful no-op.
class [[nodiscard]] Completion final {
 public:
  Completion() noexcept = default;
  Completion(const Completion&) = delete;
  Completion& operator=(const Completion&) = delete;

  Completion(Completion&& other) noexcept
      : terminal_(std::move(other.terminal_)), valid_(other.valid_) {
    other.valid_ = false;
  }

  Completion& operator=(Completion&& other) noexcept {
    if (this != &other) {
      terminal_ = std::move(other.terminal_);
      valid_ = other.valid_;
      other.valid_ = false;
    }
    return *this;
  }

  static Completion completed(Status terminal = Status::Ok()) noexcept {
    return Completion(std::move(terminal));
  }

  bool valid() const noexcept { return valid_; }
  explicit operator bool() const noexcept { return valid(); }
  bool ready() const noexcept { return valid_; }

  Status wait() const noexcept {
    if (!valid_) {
      return Status::FailedPrecondition("completion is invalid");
    }
    return terminal_;
  }

  Status status() const noexcept { return wait(); }

 private:
  explicit Completion(Status terminal) noexcept
      : terminal_(std::move(terminal)), valid_(true) {}

  Status terminal_{};
  bool valid_ = false;
};

}  // namespace hyle::compute
