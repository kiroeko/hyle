#include "hyle/compute/detail/retained_completion.h"

#include <memory>
#include <mutex>
#include <utility>

#include "completion_internal.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace hyle::compute::detail {
struct RetainedCompletionState final : util::RefCounted<RetainedCompletionState> {
  explicit RetainedCompletionState(mem::AllocatorRef selected) noexcept : allocator(selected) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this);
    selected.deallocate(this, sizeof(RetainedCompletionState), alignof(RetainedCompletionState));
  }
  mem::AllocatorRef allocator;
  Completion original;
  CompletionRetention retention;
  CompletionLifetime forwarded;
  util::SpinLock lock;
  Status retired_terminal, forwarded_hint;
  bool retired = false;
  bool deferred_prepared = false;
  Status terminal;
};
namespace {
void native_terminal(void *opaque, const Status &terminal) noexcept {
  auto &state = *static_cast<RetainedCompletionState *>(opaque);
  CompletionLifetime lifetime;
  Status final;
  {
    std::lock_guard guard(state.lock);
    state.retired = true;
    state.retired_terminal = terminal;
    lifetime = std::exchange(state.forwarded, {});
    final = terminal;
    final.Update(state.forwarded_hint);
  }
  if (lifetime.valid()) {
    lifetime.terminal(lifetime.state, final);
    lifetime.release(lifetime.state);
  }
}
void native_release(void *opaque) noexcept {
  static_cast<RetainedCompletionState *>(opaque)->decref();
}
const CompletionVTable proxy_vtable{
    .destroy = [](void *opaque, CompletionLifetime lifetime, Status hint) noexcept {
      auto *state = static_cast<RetainedCompletionState *>(opaque);
      CompletionAccess::append_terminal(state->original, hint);
      if (lifetime.valid()) {
        bool retired;
        Status terminal;
        {
          std::lock_guard guard(state->lock);
          HYLE_FATAL_ASSERT(!state->forwarded.valid());
          retired = state->retired;
          if (retired) { terminal = state->retired_terminal; terminal.Update(hint); }
          else { state->forwarded = lifetime; state->forwarded_hint = hint; }
        }
        if (retired) {
          lifetime.terminal(lifetime.state, terminal);
          lifetime.release(lifetime.state);
        }
      }
      // Never infer retirement from a proxy error. The original native destroy
      // receives both the original and forwarded attachments and its hint.
      state->original = {};
      state->decref(); // public proxy
    },
    .poll = [](void *opaque) noexcept -> Result<CompletionPoll> {
      auto &state = *static_cast<RetainedCompletionState *>(opaque);
      auto ready = state.original.poll();
      if (!ready) state.terminal.Update(ready.error());
      return ready;
    },
    .wait = [](void *opaque) noexcept {
      auto &state = *static_cast<RetainedCompletionState *>(opaque);
      const auto terminal = state.original.wait();
      state.terminal.Update(terminal);
      return terminal;
    },
    .request_cancel = [](void *opaque) noexcept {
      return static_cast<RetainedCompletionState *>(opaque)->original.request_cancel();
    }};
} // namespace

RetainedCompletion::RetainedCompletion(RetainedCompletion &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
RetainedCompletion &RetainedCompletion::operator=(RetainedCompletion &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
RetainedCompletion::~RetainedCompletion() { reset(); }
Result<RetainedCompletion> RetainedCompletion::create(mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid()) return std::unexpected(Status::InvalidArgument("completion observer allocator"));
  auto *storage = allocator.allocate(sizeof(RetainedCompletionState), alignof(RetainedCompletionState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("completion observer metadata"));
  return RetainedCompletion(std::construct_at(static_cast<RetainedCompletionState *>(storage), allocator));
}
bool RetainedCompletion::available() const noexcept { return state_ && state_->is_unique(); }
CompletionLifetime RetainedCompletion::reserve_retirement() noexcept {
  HYLE_FATAL_ASSERT(available());
  {
    std::lock_guard guard(state_->lock);
    state_->retired = false;
    state_->retired_terminal = Status::Ok();
    state_->forwarded_hint = Status::Ok();
  }
  state_->terminal = Status::Ok();
  state_->deferred_prepared = true;
  state_->incref();
  return {state_, &native_terminal, &native_release};
}
Completion RetainedCompletion::publish(Completion completion, bool deferred_retirement) noexcept {
  HYLE_FATAL_ASSERT(state_ && completion.valid());
  if (!deferred_retirement) {
    auto retirement = reserve_retirement();
    state_->retention.own = retirement;
  }
  HYLE_FATAL_ASSERT(state_->deferred_prepared);
  state_->deferred_prepared = false;
  state_->original = std::move(completion);
  // Install the actual-retirement latch before any public observation. An
  // unknown observation may cache an error without firing this callback.
  if (!deferred_retirement)
    HYLE_FATAL_ASSERT(CompletionAccess::attach_retention(state_->original, state_->retention).ok());
  state_->incref();
  return CompletionAccess::make(&proxy_vtable, state_);
}
Status RetainedCompletion::wait() noexcept {
  if (!state_) return Status::Ok();
  if (state_->original.valid()) state_->terminal.Update(state_->original.wait());
  return state_->terminal;
}
void RetainedCompletion::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) state->decref();
}
} // namespace hyle::compute::detail
