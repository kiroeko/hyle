#include "hyle/compute/completion.h"

#include <utility>

#include "completion_internal.h"

namespace hyle::compute {

Completion
detail::CompletionAccess::make(const detail::CompletionVTable *vtable,
                               void *state) noexcept {
  if (vtable == nullptr || state == nullptr || vtable->destroy == nullptr ||
      vtable->poll == nullptr || vtable->wait == nullptr) {
    if (vtable != nullptr && state != nullptr && vtable->destroy != nullptr)
      vtable->destroy(state, {});
    return Completion{};
  }
  return Completion(vtable, state);
}

Status detail::CompletionAccess::attach_lifetime(
    Completion &completion, detail::CompletionLifetime &lifetime) noexcept {
  return completion.attach_lifetime(lifetime);
}

void detail::CompletionAccess::append_terminal(Completion &completion,
                                               Status terminal) noexcept {
  completion.append_terminal(std::move(terminal));
}

Completion::Completion(const detail::CompletionVTable *vtable,
                       void *state) noexcept
    : vtable_(vtable), state_(state), valid_(vtable != nullptr) {}

Completion::Completion(Status terminal) noexcept
    : terminal_(std::move(terminal)), valid_(true) {}

Completion::Completion(Completion &&other) noexcept
    : vtable_(other.vtable_), state_(other.state_),
      terminal_(std::move(other.terminal_)), lifetime_(other.lifetime_),
      valid_(other.valid_) {
  other.vtable_ = nullptr;
  other.state_ = nullptr;
  other.lifetime_ = nullptr;
  other.valid_ = false;
}

Completion &Completion::operator=(Completion &&other) noexcept {
  if (this != &other) {
    reset();
    vtable_ = other.vtable_;
    state_ = other.state_;
    terminal_ = std::move(other.terminal_);
    lifetime_ = other.lifetime_;
    valid_ = other.valid_;
    other.vtable_ = nullptr;
    other.state_ = nullptr;
    other.lifetime_ = nullptr;
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
    Result<CompletionPoll> result = vtable_->poll(state_);
    if (result && *result == CompletionPoll::kPending)
      return result;
    Status terminal = Status::Ok();
    if (!result)
      terminal = std::move(result).error();
    settle(std::move(terminal));
    if (!terminal_.ok())
      return std::unexpected(terminal_);
    return CompletionPoll::kComplete;
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
    settle(vtable_->wait(state_));
    return terminal_;
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
  if (valid_ && vtable_ != nullptr && vtable_->destroy != nullptr) {
    const detail::CompletionLifetime lifetime =
        lifetime_ != nullptr ? *lifetime_ : detail::CompletionLifetime{};
    lifetime_ = nullptr;
    vtable_->destroy(state_, lifetime);
  } else if (lifetime_ != nullptr && lifetime_->valid()) {
    const detail::CompletionLifetime lifetime = *lifetime_;
    lifetime_ = nullptr;
    lifetime.terminal(lifetime.state, terminal_);
    lifetime.release(lifetime.state);
  }
  vtable_ = nullptr;
  state_ = nullptr;
  valid_ = false;
}

void Completion::settle(Status terminal) const noexcept {
  if (vtable_ == nullptr)
    return;
  if (terminal.ok() && !terminal_.ok())
    terminal = terminal_;

  const detail::CompletionVTable *vtable = vtable_;
  void *state = state_;
  vtable_ = nullptr;
  state_ = nullptr;
  terminal_ = std::move(terminal);

  const detail::CompletionLifetime lifetime =
      lifetime_ != nullptr ? *lifetime_ : detail::CompletionLifetime{};
  lifetime_ = nullptr;
  // The provider completion state may itself be embedded in the retained
  // invocation.  Retire it before releasing the attachment's final reference.
  if (vtable->destroy != nullptr)
    vtable->destroy(state, {});
  if (lifetime.valid()) {
    lifetime.terminal(lifetime.state, terminal_);
    lifetime.release(lifetime.state);
  }
}

Status
Completion::attach_lifetime(detail::CompletionLifetime &lifetime) noexcept {
  if (!lifetime.valid())
    return Status::InvalidArgument("completion lifetime attachment is invalid");
  if (!valid_) {
    const Status error = Status::FailedPrecondition("completion is invalid");
    lifetime.terminal(lifetime.state, error);
    lifetime.release(lifetime.state);
    return error;
  }
  if (lifetime_ != nullptr) {
    const Status error = Status::FailedPrecondition(
        "completion already owns a lifetime attachment");
    lifetime.terminal(lifetime.state, error);
    lifetime.release(lifetime.state);
    return error;
  }
  if (vtable_ == nullptr) {
    lifetime.terminal(lifetime.state, terminal_);
    lifetime.release(lifetime.state);
    return Status::Ok();
  }
  if (vtable_->destroy == nullptr) {
    const Status error = Status::Internal(
        "asynchronous completion cannot retain borrowed execution state");
    lifetime.terminal(lifetime.state, error);
    lifetime.release(lifetime.state);
    return error;
  }
  lifetime_ = &lifetime;
  return Status::Ok();
}

void Completion::append_terminal(Status terminal) noexcept {
  if (!terminal.ok() && terminal_.ok())
    terminal_ = std::move(terminal);
}

} // namespace hyle::compute
