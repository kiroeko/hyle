#include "hyle/compute/detail/retained_completion.h"

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "completion_internal.h"
#include "hyle/compute/detail/dependent_completion.h"
#include "hyle/compute/detail/copy_outcome_source.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace hyle::compute::detail {
struct RetainedCompletionState final : util::RefCounted<RetainedCompletionState> {
  enum class Phase : unsigned char { Unpublished, Pending, Installing, Installed };
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
  std::atomic<std::uint64_t> revision{0};
  RetainedCompletionControl control;
  std::size_t control_users = 0;
  Status retired_terminal, forwarded_hint;
  bool retired = false;
  bool deferred_prepared = false;
  bool deferred_retirement = false;
  Phase phase = Phase::Unpublished;
  bool native_active = false;
  bool abandoned = false;
  bool cancel_requested = false;
  bool control_closing = false;
  Status terminal;
};
namespace {
using RetainedPhase = RetainedCompletionState::Phase;
// A control reference stays in the state until handoff has closed hook entry
// and all outstanding hook calls have left. No arbitrary hook runs under lock.
RetainedCompletionControl borrow_control(RetainedCompletionState &state) noexcept {
  std::lock_guard guard(state.lock);
  if (!state.control.state || state.control_closing) return {};
  ++state.control_users;
  return state.control;
}
void return_control(RetainedCompletionState &state,
                    RetainedCompletionControl borrowed) noexcept {
  if (!borrowed.state) return;
  RetainedCompletionControl release;
  {
    std::lock_guard guard(state.lock);
    HYLE_FATAL_ASSERT(state.control_users);
    if (--state.control_users == 0 && state.control_closing)
      release = std::exchange(state.control, {});
  }
  if (release.state) release.release(release.state);
}
void notify_observers(RetainedCompletionState &state,
                      RetainedCompletionControl control = {}) noexcept {
  state.revision.fetch_add(1, std::memory_order_release);
  state.revision.notify_all();
  if (control.notify) control.notify(control.state);
}
void wait_for_change(RetainedCompletionState &state,
                     std::uint64_t revision) noexcept {
  auto control = borrow_control(state);
  if (control.wait) control.wait(control.state, state.revision, revision);
  else state.revision.wait(revision, std::memory_order_acquire);
  return_control(state, control);
}

void native_terminal(void *opaque, const Status &terminal) noexcept {
  auto &state = *static_cast<RetainedCompletionState *>(opaque);
  CompletionLifetime lifetime;
  Status final;
  {
    std::lock_guard guard(state.lock);
    state.retired = true;
    state.retired_terminal = terminal;
    state.terminal.Update(terminal);
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

// Borrow the move-only native handle, not its callbacks, under the short state
// lock. Native calls and Completion destruction may invoke retirement inline.
bool borrow_native(RetainedCompletionState &state, Completion &native) noexcept {
  std::lock_guard guard(state.lock);
  if (state.phase != RetainedPhase::Installed || state.native_active) return false;
  state.native_active = true;
  native = std::move(state.original); // original is empty until returned
  return true;
}
void return_native(RetainedCompletionState &state, Completion native,
                   const Status &terminal = Status::Ok()) noexcept {
  bool abandoned = false;
  Status hint;
  {
    std::lock_guard guard(state.lock);
    HYLE_FATAL_ASSERT(state.native_active && !state.original.valid());
    state.terminal.Update(terminal);
    abandoned = state.abandoned;
    hint = state.terminal;
    if (!abandoned) state.original = std::move(native);
    state.native_active = false;
  }
  if (abandoned) {
    CompletionAccess::append_terminal(native, hint);
    native = {};
  }
  notify_observers(state);
}
Status wait_retained(RetainedCompletionState &state) noexcept {
  for (;;) {
    const auto revision = state.revision.load(std::memory_order_acquire);
    Completion native;
    if (borrow_native(state, native)) {
      const auto terminal = native.valid() ? native.wait() : Status::Ok();
      return_native(state, std::move(native), terminal);
      std::lock_guard guard(state.lock);
      return state.terminal;
    }
    wait_for_change(state, revision);
  }
}
Status cancel_retained(RetainedCompletionState &state) noexcept {
  Completion native;
  bool borrowed = false;
  RetainedCompletionControl control;
  {
    std::lock_guard guard(state.lock);
    const bool first = !std::exchange(state.cancel_requested, true);
    if (first && state.control.state && !state.control_closing) {
      ++state.control_users;
      control = state.control;
    }
    // Decide which side owns cancellation under the installation lock. A
    // pre-handoff request must not also cancel again after handoff publishes.
    if (state.phase == RetainedPhase::Installed && !state.native_active) {
      state.native_active = true;
      native = std::move(state.original);
      borrowed = true;
    }
  }
  if (control.request_cancel) control.request_cancel(control.state);
  notify_observers(state, control);
  return_control(state, control);
  if (!borrowed) return Status::Ok();
  const auto result = native.valid() ? native.request_cancel() : Status::Ok();
  return_native(state, std::move(native));
  return result;
}
const CompletionVTable proxy_vtable{
    .destroy = [](void *opaque, CompletionLifetime lifetime, Status hint) noexcept {
      auto *state = static_cast<RetainedCompletionState *>(opaque);
      Completion original;
      bool retired = false;
      Status terminal;
      {
        std::lock_guard guard(state->lock);
        state->abandoned = true;
        state->terminal.Update(hint);
        if (state->phase == RetainedPhase::Installed && !state->native_active)
          original = std::move(state->original);
        if (lifetime.valid()) {
          HYLE_FATAL_ASSERT(!state->forwarded.valid());
          retired = state->retired;
          if (retired) {
            terminal = state->retired_terminal;
            terminal.Update(hint);
          } else {
            state->forwarded = lifetime;
            state->forwarded_hint = hint;
          }
        }
      }
      if (lifetime.valid()) {
        if (retired) {
          lifetime.terminal(lifetime.state, terminal);
          lifetime.release(lifetime.state);
        }
      }
      // Never infer retirement from a proxy error. The original native destroy
      // receives both the original and forwarded attachments and its hint.
      CompletionAccess::append_terminal(original, hint);
      original = {};
      state->decref(); // public proxy
    },
    .poll = [](void *opaque) noexcept -> Result<CompletionPoll> {
      auto &state = *static_cast<RetainedCompletionState *>(opaque);
      Completion native;
      if (!borrow_native(state, native)) return CompletionPoll::kPending;
      auto ready = native.valid() ? native.poll() : Result<CompletionPoll>(CompletionPoll::kComplete);
      return_native(state, std::move(native),
                    ready ? Status::Ok() : Status{ready.error()});
      if (ready && *ready == CompletionPoll::kComplete) {
        std::lock_guard guard(state.lock);
        if (!state.terminal.ok()) return std::unexpected(state.terminal);
      }
      return ready;
    },
    .wait = [](void *opaque) noexcept {
      return wait_retained(*static_cast<RetainedCompletionState *>(opaque));
    },
    .request_cancel = [](void *opaque) noexcept {
      return cancel_retained(*static_cast<RetainedCompletionState *>(opaque));
    }};
} // namespace

RetainedCompletionHandoff::RetainedCompletionHandoff(RetainedCompletionHandoff &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
RetainedCompletionHandoff &RetainedCompletionHandoff::operator=(RetainedCompletionHandoff &&other) noexcept {
  if (this != &other) {
    HYLE_FATAL_ASSERT(!state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
RetainedCompletionHandoff::~RetainedCompletionHandoff() {
  HYLE_FATAL_ASSERT(!state_);
}
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
bool RetainedCompletion::owns(const Completion &completion) const noexcept {
  const auto proxy = CompletionAccess::dependency(completion);
  return state_ && proxy.vtable == &proxy_vtable && proxy.state == state_;
}
CompletionLifetime RetainedCompletion::reserve_retirement() noexcept {
  HYLE_FATAL_ASSERT(available());
  {
    std::lock_guard guard(state_->lock);
    HYLE_FATAL_ASSERT(state_->phase != RetainedPhase::Pending &&
                      state_->phase != RetainedPhase::Installing);
    HYLE_FATAL_ASSERT(!state_->original.valid() && !state_->native_active &&
                      !state_->control.state && !state_->control_users);
    state_->retired = false;
    state_->retired_terminal = Status::Ok();
    state_->forwarded_hint = Status::Ok();
    state_->terminal = Status::Ok();
    state_->phase = RetainedPhase::Unpublished;
    state_->abandoned = false;
    state_->cancel_requested = false;
    state_->control_closing = false;
  }
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
  {
    std::lock_guard guard(state_->lock);
    state_->phase = RetainedPhase::Installed;
  }
  state_->incref();
  return CompletionAccess::make(&proxy_vtable, state_);
}
RetainedCompletionPending RetainedCompletion::publish_pending(
    RetainedCompletionControl control, bool deferred_retirement) noexcept {
  HYLE_FATAL_ASSERT(state_);
  HYLE_FATAL_ASSERT(control.state ? control.release != nullptr :
      !control.request_cancel && !control.wait && !control.notify && !control.release);
  HYLE_FATAL_ASSERT(!control.wait || control.notify);
  if (!deferred_retirement) state_->retention.own = reserve_retirement();
  HYLE_FATAL_ASSERT(state_->deferred_prepared);
  {
    std::lock_guard guard(state_->lock);
    state_->deferred_prepared = false;
    state_->deferred_retirement = deferred_retirement;
    state_->phase = RetainedPhase::Pending;
    state_->control = control;
  }
  state_->incref(); // original posted progress driver, consumed by handoff
  state_->incref(); // public proxy
  return {CompletionAccess::make(&proxy_vtable, state_), RetainedCompletionHandoff(state_)};
}
Status RetainedCompletionHandoff::handoff(Completion &&completion) noexcept {
  if (!state_ || !completion.valid())
    return Status::FailedPrecondition("pending completion handoff is invalid");
  auto &state = *state_;
  bool deferred;
  {
    std::lock_guard guard(state.lock);
    if (state.phase != RetainedPhase::Pending)
      return Status::FailedPrecondition("completion handoff is not pending");
    state.phase = RetainedPhase::Installing;
    deferred = state.deferred_retirement;
  }
  // From here every path installs the actual completion. Callbacks may destroy
  // the ticket's enclosing owner; only the local state/driver reference remains.
  state_ = nullptr;
  Completion original = std::move(completion);
  if (!deferred)
    HYLE_FATAL_ASSERT(CompletionAccess::attach_retention(original, state.retention).ok());
  bool cancelled = false;
  bool abandoned = false;
  Status hint;
  RetainedCompletionControl control;
  for (;;) {
    bool cancel;
    {
      std::lock_guard guard(state.lock);
      cancel = state.cancel_requested && !cancelled;
      if (!cancel) {
        abandoned = state.abandoned;
        hint = state.terminal;
        if (!abandoned) state.original = std::move(original);
        state.phase = RetainedPhase::Installed;
        // Pin the existing owner for notify before closing new hook entry.
        control = state.control;
        if (control.state) ++state.control_users;
        state.control_closing = true;
      }
    }
    if (!cancel) break;
    // Cancellation of the driver was already accepted. A native prefix may
    // not support cancel; it must still be installed and actually retired.
    (void)original.request_cancel();
    cancelled = true;
  }
  notify_observers(state, control);
  return_control(state, control);
  if (abandoned) {
    CompletionAccess::append_terminal(original, hint);
    original = {};
  }
  state.decref(); // posted progress driver
  return Status::Ok();
}
Result<CompletionPoll> RetainedCompletion::poll() noexcept {
  auto *const state = state_;
  if (!state) return CompletionPoll::kComplete;
  {
    std::lock_guard guard(state->lock);
    if (state->phase == RetainedPhase::Unpublished) {
      if (!state->terminal.ok()) return std::unexpected(state->terminal);
      return CompletionPoll::kComplete;
    }
  }
  auto ready = proxy_vtable.poll(state);
  if (!ready || *ready == CompletionPoll::kPending) return ready;
  // The public proxy may already have settled and cleared original. Preserve
  // the same cached verdict as root.wait(), without inferring actual retirement.
  std::lock_guard guard(state->lock);
  if (!state->terminal.ok()) return std::unexpected(state->terminal);
  return CompletionPoll::kComplete;
}
Status RetainedCompletion::wait() noexcept {
  if (!state_) return Status::Ok();
  {
    std::lock_guard guard(state_->lock);
    if (state_->phase == RetainedPhase::Unpublished) return state_->terminal;
  }
  return wait_retained(*state_);
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
    std::lock_guard guard(state.lock);
    if (state.phase != RetainedPhase::Installed || state.native_active)
      return std::unexpected(Status::Unavailable("copy producer completion handoff is pending"));
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
    // A completed observation can return the original controls to a cold
    // source slot for reuse. Only pending abandonment needs native handoff.
    if (state.retired.load(std::memory_order_acquire) == kAllRetired) return;
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
struct CopyOutcomeSourceState final : util::RefCounted<CopyOutcomeSourceState> {
  struct Node final {
    CopyOutcomeSourceState *owner = nullptr;
    CompletionRetention retention;
  };
  CopyOutcomeSourceState(mem::AllocatorRef selected, std::size_t selected_capacity,
      std::size_t bytes, std::size_t alignment, CopyOutcomeSourceControl selected_control) noexcept
      : allocator(selected), control(selected_control), capacity(selected_capacity),
        payload_bytes(bytes), payload_alignment(alignment) {}
  void destroy() noexcept {
    if (constructed) control.destroy(original);
    if (nodes) {
      std::destroy_n(nodes, capacity);
      allocator.deallocate(nodes, capacity * sizeof(Node), alignof(Node));
    }
    if (completions) allocator.deallocate(completions,
        capacity * sizeof(Completion *), alignof(Completion *));
    if (original) allocator.deallocate(original, payload_bytes, payload_alignment);
    const auto selected = allocator;
    std::destroy_at(this);
    selected.deallocate(this, sizeof(CopyOutcomeSourceState), alignof(CopyOutcomeSourceState));
  }
  mem::AllocatorRef allocator;
  CopyOutcomeSourceControl control;
  std::size_t capacity, payload_bytes, payload_alignment;
  void *original = nullptr;
  Completion **completions = nullptr;
  Node *nodes = nullptr;
  std::size_t count = 0;
  std::atomic<std::size_t> remaining{0};
  util::SpinLock lock;
  Status terminal;
  CompletionLifetime retirement;
  bool constructed = false, bound = false, subscribed = false;
};
namespace {
void source_record(CopyOutcomeSourceState &state, const Status &status) noexcept {
  std::lock_guard guard(state.lock);
  state.terminal.Update(status);
}
Status source_status(CopyOutcomeSourceState &state) noexcept {
  std::lock_guard guard(state.lock);
  return state.terminal;
}
void source_terminal(void *opaque, const Status &status) noexcept {
  source_record(*static_cast<CopyOutcomeSourceState::Node *>(opaque)->owner, status);
}
void source_release(void *opaque) noexcept {
  auto &state = *static_cast<CopyOutcomeSourceState::Node *>(opaque)->owner;
  if (state.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    const auto lifetime = state.retirement;
    const auto terminal = source_status(state);
    lifetime.terminal(lifetime.state, terminal);
    lifetime.release(lifetime.state);
  }
  state.decref();
}
OutcomeSources source_protocol(CopyOutcomeSourceState &state) noexcept {
  return {&state,
      [](void *opaque) noexcept -> Result<CompletionPoll> {
        auto &source = *static_cast<CopyOutcomeSourceState *>(opaque);
        source.control.serialize(source.original, [](void *context) noexcept {
          auto &source=*static_cast<CopyOutcomeSourceState *>(context);
          for (std::size_t i = 0; i < source.count; ++i) {
            auto observed = source.completions[i]->poll();
            if (!observed) source_record(source, observed.error());
          }
        }, &source);
        if (source.remaining.load(std::memory_order_acquire))
          return CompletionPoll::kPending;
        source_record(source, source.control.observe(source.original));
        const auto terminal = source_status(source);
        if (!terminal.ok()) return std::unexpected(terminal);
        return CompletionPoll::kComplete;
      },
      [](void *opaque) noexcept {
        auto &source = *static_cast<CopyOutcomeSourceState *>(opaque);
        // The original Update waits outside its observation lock so native
        // worker preflight can progress. Shared completions are then retired.
        source_record(source, source.control.wait(source.original));
        source.control.serialize(source.original, [](void *context) noexcept {
          auto &source=*static_cast<CopyOutcomeSourceState *>(context);
          for (std::size_t i = 0; i < source.count; ++i)
            source_record(source, source.completions[i]->wait());
        }, &source);
        return source_status(source);
      },
      [](void *opaque) noexcept {
        auto &source = *static_cast<CopyOutcomeSourceState *>(opaque);
        return source.control.request_cancel(source.original);
      },
      [](void *opaque) noexcept {
        auto &source = *static_cast<CopyOutcomeSourceState *>(opaque);
        source.control.abandon(source.original);
      },
      [](void *opaque, CompletionLifetime retirement) noexcept {
        auto &source = *static_cast<CopyOutcomeSourceState *>(opaque);
        HYLE_FATAL_ASSERT(source.bound && !source.subscribed && retirement.valid());
        source.subscribed = true;
        source.retirement = retirement;
        source.remaining.store(source.count, std::memory_order_release);
        source.incref(static_cast<std::uint32_t>(source.count));
        source.control.serialize(source.original, [](void *context) noexcept {
          auto &source=*static_cast<CopyOutcomeSourceState *>(context);
          for (std::size_t i = 0; i < source.count; ++i) {
            auto &node = source.nodes[i];
            node.owner = &source;
            node.retention.own = {&node, &source_terminal, &source_release};
            HYLE_FATAL_ASSERT(CompletionAccess::attach_retention(
                *source.completions[i], node.retention).ok());
          }
        }, &source);
      },
      [](void *opaque) noexcept {
        auto &source = *static_cast<CopyOutcomeSourceState *>(opaque);
        // Unsubscribed rejected suffixes have no borrow. Subscribed references
        // arrive here only after both source and suffix actually retire.
        if (!source.subscribed) source.control.abandon(source.original);
        source.bound = false;
        source.decref();
      }};
}
} // namespace

Result<CopyOutcomeSourceState *> CopyOutcomeSourceAccess::create(
    std::size_t capacity, std::size_t payload_bytes, std::size_t payload_alignment,
    CopyOutcomeSourceControl control, CopyOutcomeSourceOptions options) noexcept {
  if (!capacity || capacity > std::numeric_limits<std::uint32_t>::max() - 2 ||
      capacity > SIZE_MAX / sizeof(CopyOutcomeSourceState::Node) ||
      capacity > SIZE_MAX / sizeof(Completion *) || !payload_bytes || !payload_alignment ||
      (payload_alignment & (payload_alignment - 1)) || !options.allocator.valid() ||
      !control.construct || !control.destroy || !control.observe || !control.wait ||
      !control.request_cancel || !control.abandon || !control.serialize)
    return std::unexpected(Status::InvalidArgument("copy outcome cold source configuration"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *storage = allocator.allocate(sizeof(CopyOutcomeSourceState), alignof(CopyOutcomeSourceState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("copy outcome state"));
  auto *state = std::construct_at(static_cast<CopyOutcomeSourceState *>(storage),
      allocator, capacity, payload_bytes, payload_alignment, control);
  state->original = allocator.allocate(payload_bytes, payload_alignment);
  if (state->original) {
    state->completions = static_cast<Completion **>(allocator.allocate(
        capacity * sizeof(Completion *), alignof(Completion *)));
    if (state->completions) {
      state->nodes = static_cast<CopyOutcomeSourceState::Node *>(allocator.allocate(
          capacity * sizeof(CopyOutcomeSourceState::Node), alignof(CopyOutcomeSourceState::Node)));
      if (state->nodes) std::uninitialized_value_construct_n(state->nodes, capacity);
    }
  }
  if (!state->original || !state->completions || !state->nodes) {
    state->decref();
    return std::unexpected(Status::ResourceExhausted("copy outcome payload/retirement nodes"));
  }
  control.construct(state->original);
  state->constructed = true;
  return state;
}
void *CopyOutcomeSourceAccess::payload(CopyOutcomeSourceState *state) noexcept {
  return state ? state->original : nullptr;
}
bool CopyOutcomeSourceAccess::available(const CopyOutcomeSourceState *state) noexcept {
  return state && state->is_unique() && !state->bound;
}
Result<std::span<Completion *>> CopyOutcomeSourceAccess::prepare(
    CopyOutcomeSourceState *state, std::size_t count) noexcept {
  if (!available(state))
    return std::unexpected(Status::Unavailable("copy outcome source is retained"));
  if (!count || count > state->capacity)
    return std::unexpected(Status::ResourceExhausted("copy outcome source capacity"));
  return std::span(state->completions, count);
}
OutcomeSources CopyOutcomeSourceAccess::commit(CopyOutcomeSourceState *state,
                                               std::size_t count) noexcept {
  HYLE_FATAL_ASSERT(available(state) && count && count <= state->capacity);
  state->count = count;
  state->terminal = Status::Ok();
  state->retirement = {};
  state->subscribed = false;
  state->bound = true;
  state->incref();
  return source_protocol(*state);
}
void CopyOutcomeSourceAccess::release(CopyOutcomeSourceState *state) noexcept {
  if (state) state->decref();
}
} // namespace hyle::compute::detail
