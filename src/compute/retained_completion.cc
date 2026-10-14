#include "hyle/compute/detail/retained_completion.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

#include "completion_internal.h"
#include "hyle/compute/detail/dependent_completion.h"
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
Result<CompletionDependency> RetainedCompletion::native_dependency(const Completion &source) noexcept {
  if (!source.valid())
    return std::unexpected(Status::FailedPrecondition("copy producer completion is empty"));
  auto dependency = CompletionAccess::dependency(source);
  const Status *error = nullptr;
  for (unsigned depth = 0; depth < 32; ++depth) {
    if (!error && dependency.terminal_hint && !dependency.terminal_hint->ok())
      error = dependency.terminal_hint;
    if (dependency.vtable != &proxy_vtable) {
      if (error) dependency.terminal_hint = error;
      return dependency;
    }
    auto &state = *static_cast<RetainedCompletionState *>(dependency.state);
    if (!error && !state.terminal.ok()) error = &state.terminal;
    // Read an identity, never poll/wait or infer retirement from cached status.
    dependency = CompletionAccess::dependency(state.original);
  }
  return std::unexpected(Status::ResourceExhausted("copy producer retained dependency depth exceeded"));
}
struct DependentCompletionState final : util::RefCounted<DependentCompletionState> {
  explicit DependentCompletionState(mem::AllocatorRef selected) noexcept
      : allocator(selected) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this);
    selected.deallocate(this, sizeof(DependentCompletionState),
                        alignof(DependentCompletionState));
  }
  mem::AllocatorRef allocator;
  Completion suffix;
  OutcomeSources sources;
  CompletionRetention retention;
  CompletionLifetime forwarded;
  util::SpinLock lock;
  Status terminal;
  std::atomic<unsigned> retired{0};
  bool proxy_destroyed = false;
  bool handoff_started = false;
  bool handoff_active = false;
};

namespace {
constexpr unsigned kSuffixRetired = 1;
constexpr unsigned kSourcesRetired = 2;
constexpr unsigned kAllRetired = kSuffixRetired | kSourcesRetired;

void dependent_finish(DependentCompletionState &state) noexcept {
  CompletionLifetime forwarded;
  OutcomeSources sources;
  Status terminal;
  {
    std::lock_guard guard(state.lock);
    if (!state.proxy_destroyed || state.handoff_active ||
        state.retired.load(std::memory_order_acquire) != kAllRetired)
      return;
    forwarded = std::exchange(state.forwarded, {});
    sources = std::exchange(state.sources, {});
    terminal = state.terminal;
  }
  if (forwarded.valid()) {
    forwarded.terminal(forwarded.state, terminal);
    forwarded.release(forwarded.state);
  }
  if (sources.valid()) sources.release(sources.state);
}

void dependent_handoff(DependentCompletionState &state) noexcept {
  OutcomeSources sources;
  {
    std::lock_guard guard(state.lock);
    if (!state.proxy_destroyed || state.handoff_started ||
        !(state.retired.load(std::memory_order_acquire) & kSuffixRetired))
      return;
    state.handoff_started = true;
    state.handoff_active = true;
    sources = state.sources;
  }
  // Called only after the suffix releases its borrowed dependency. The source
  // transfers its original completions to their existing retirement owners.
  sources.abandon(sources.state);
  {
    std::lock_guard guard(state.lock);
    state.handoff_active = false;
  }
  dependent_finish(state);
}

void dependent_terminal(void *opaque, const Status &terminal) noexcept {
  auto &state = *static_cast<DependentCompletionState *>(opaque);
  std::lock_guard guard(state.lock);
  state.terminal.Update(terminal);
}
void dependent_suffix_release(void *opaque) noexcept {
  auto &state = *static_cast<DependentCompletionState *>(opaque);
  state.retired.fetch_or(kSuffixRetired, std::memory_order_release);
  state.retired.notify_all();
  dependent_handoff(state);
  dependent_finish(state);
  state.decref();
}
void dependent_sources_release(void *opaque) noexcept {
  auto &state = *static_cast<DependentCompletionState *>(opaque);
  state.retired.fetch_or(kSourcesRetired, std::memory_order_release);
  state.retired.notify_all();
  dependent_finish(state);
  state.decref();
}
void dependent_record(DependentCompletionState &state,
                      const Status &terminal) noexcept {
  std::lock_guard guard(state.lock);
  state.terminal.Update(terminal);
}
Status dependent_status(DependentCompletionState &state) noexcept {
  std::lock_guard guard(state.lock);
  return state.terminal;
}
void dependent_wait_retirement(DependentCompletionState &state,
                               unsigned mask) noexcept {
  auto retired = state.retired.load(std::memory_order_acquire);
  while ((retired & mask) != mask) {
    state.retired.wait(retired, std::memory_order_acquire);
    retired = state.retired.load(std::memory_order_acquire);
  }
}

const CompletionVTable dependent_vtable{
    .destroy = [](void *opaque, CompletionLifetime lifetime, Status hint) noexcept {
      auto &state = *static_cast<DependentCompletionState *>(opaque);
      {
        std::lock_guard guard(state.lock);
        state.terminal.Update(hint);
        HYLE_FATAL_ASSERT(!state.proxy_destroyed && !state.forwarded.valid());
        state.proxy_destroyed = true;
        state.forwarded = lifetime;
      }
      CompletionAccess::append_terminal(state.suffix, hint);
      state.suffix = {};
      dependent_handoff(state);
      dependent_finish(state);
      state.decref();
    },
    .poll = [](void *opaque) noexcept -> Result<CompletionPoll> {
      auto &state = *static_cast<DependentCompletionState *>(opaque);
      auto suffix = state.suffix.poll();
      if (!suffix) dependent_record(state, suffix.error());
      if (!(state.retired.load(std::memory_order_acquire) & kSuffixRetired))
        return CompletionPoll::kPending;
      auto sources = state.sources.poll(state.sources.state);
      if (!sources) dependent_record(state, sources.error());
      if (state.retired.load(std::memory_order_acquire) != kAllRetired ||
          (sources && *sources == CompletionPoll::kPending))
        return CompletionPoll::kPending;
      auto terminal = dependent_status(state);
      if (!terminal.ok()) return std::unexpected(terminal);
      return CompletionPoll::kComplete;
    },
    .wait = [](void *opaque) noexcept {
      auto &state = *static_cast<DependentCompletionState *>(opaque);
      dependent_record(state, state.suffix.wait());
      dependent_wait_retirement(state, kSuffixRetired);
      dependent_record(state, state.sources.wait(state.sources.state));
      dependent_wait_retirement(state, kAllRetired);
      return dependent_status(state);
    },
    .request_cancel = [](void *opaque) noexcept {
      auto &state = *static_cast<DependentCompletionState *>(opaque);
      auto status = state.suffix.request_cancel();
      status.Update(state.sources.request_cancel(state.sources.state));
      return status;
    }};
} // namespace

DependentCompletion::DependentCompletion(DependentCompletion &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
DependentCompletion &DependentCompletion::operator=(DependentCompletion &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
DependentCompletion::~DependentCompletion() { reset(); }
Result<DependentCompletion> DependentCompletion::create(mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid())
    return std::unexpected(Status::InvalidArgument("dependent completion allocator"));
  auto *storage = allocator.allocate(sizeof(DependentCompletionState),
                                     alignof(DependentCompletionState));
  if (!storage)
    return std::unexpected(Status::ResourceExhausted("dependent completion metadata"));
  return DependentCompletion(std::construct_at(
      static_cast<DependentCompletionState *>(storage), allocator));
}
bool DependentCompletion::available() const noexcept {
  return state_ && state_->is_unique();
}
Completion DependentCompletion::publish(Completion &&suffix,
                                        OutcomeSources sources) noexcept {
  HYLE_FATAL_ASSERT(available() && suffix.valid() && sources.valid());
  state_->terminal = Status::Ok();
  state_->retired.store(0, std::memory_order_relaxed);
  state_->proxy_destroyed = false;
  state_->handoff_started = false;
  state_->handoff_active = false;
  state_->sources = sources;
  state_->suffix = std::move(suffix);
  // References for the public proxy and both independent retirement callbacks.
  state_->incref();
  state_->incref();
  state_->incref();
  state_->retention.own = {state_, &dependent_terminal, &dependent_suffix_release};
  HYLE_FATAL_ASSERT(CompletionAccess::attach_retention(
      state_->suffix, state_->retention).ok());
  sources.subscribe_retirement(sources.state,
      {state_, &dependent_terminal, &dependent_sources_release});
  return CompletionAccess::make(&dependent_vtable, state_);
}
void DependentCompletion::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) state->decref();
}
} // namespace hyle::compute::detail
