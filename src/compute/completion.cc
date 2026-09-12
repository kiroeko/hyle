#include "hyle/compute/completion.h"

#include <utility>

#include "completion_internal.h"

namespace hyle::compute {

Completion
detail::CompletionAccess::make(const detail::CompletionVTable *vtable,
                               void *state) noexcept {
  return Completion(vtable, state);
}

Completion::Completion(const detail::CompletionVTable *vtable,
                       void *state) noexcept
    : vtable_(vtable), state_(state), valid_(vtable != nullptr) {}

Completion::Completion(Status terminal) noexcept
    : terminal_(std::move(terminal)), valid_(true) {}

Completion::Completion(Completion &&other) noexcept
    : vtable_(other.vtable_), state_(other.state_),
      terminal_(std::move(other.terminal_)), valid_(other.valid_) {
  other.vtable_ = nullptr;
  other.state_ = nullptr;
  other.valid_ = false;
}

Completion &Completion::operator=(Completion &&other) noexcept {
  if (this != &other) {
    reset();
    vtable_ = other.vtable_;
    state_ = other.state_;
    terminal_ = std::move(other.terminal_);
    valid_ = other.valid_;
    other.vtable_ = nullptr;
    other.state_ = nullptr;
    other.valid_ = false;
  }
  return *this;
}

Completion::~Completion() { reset(); }

Completion Completion::completed(Status terminal) noexcept {
  return Completion(std::move(terminal));
}

Result<CompletionPoll> Completion::poll() const noexcept {
  if (!valid_) {
    return std::unexpected(Status::FailedPrecondition("completion is invalid"));
  }
  if (vtable_ != nullptr) {
    if (vtable_->poll == nullptr) {
      return std::unexpected(
          Status::Internal("completion poll callback is missing"));
    }
    return vtable_->poll(state_);
  }
  if (!terminal_.ok())
    return std::unexpected(terminal_);
  return CompletionPoll::kComplete;
}

Status Completion::wait() const noexcept {
  if (!valid_)
    return Status::FailedPrecondition("completion is invalid");
  if (vtable_ != nullptr) {
    if (vtable_->wait == nullptr)
      return Status::Internal("completion wait callback is missing");
    return vtable_->wait(state_);
  }
  return terminal_;
}

Status Completion::request_cancel() noexcept {
  if (!valid_)
    return Status::FailedPrecondition("completion is invalid");
  if (vtable_ == nullptr)
    return Status::Ok();
  if (vtable_->request_cancel == nullptr)
    return Status::Unimplemented("completion cannot be cancelled");
  return vtable_->request_cancel(state_);
}

void Completion::reset() noexcept {
  if (valid_ && vtable_ != nullptr && vtable_->destroy != nullptr)
    vtable_->destroy(state_);
  vtable_ = nullptr;
  state_ = nullptr;
  valid_ = false;
}

} // namespace hyle::compute
