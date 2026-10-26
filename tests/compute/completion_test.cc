#include <gtest/gtest.h>

#include <atomic>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

#include "completion_internal.h"
#include "hyle/compute/completion.h"
#include "hyle/compute/completion_retention.h"
#include "hyle/compute/detail/dependent_completion.h"
#include "hyle/compute/detail/copy_outcome_source.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/detail/retained_completion.h"
#if !defined(HYLE_STATIC)
#include "result_owner_fixture.h"
#endif

namespace {

static_assert(!std::is_copy_constructible_v<hyle::compute::CompletionRetention>);
static_assert(!std::is_move_constructible_v<hyle::compute::CompletionRetention>);
static_assert(sizeof(hyle::compute::CompletionRetention) ==
    sizeof(hyle::compute::detail::CompletionRetention));

struct AsyncProbe final {
  bool complete = false;
  bool cancelled = false;
  bool destroyed = false;
  bool observation_failed = false;
  std::size_t polls = 0;
  std::size_t waits = 0;
  std::size_t cancels = 0;
  hyle::compute::detail::CompletionLifetime abandoned{};
  hyle::Status abandoned_hint{};
};

void destroy_async(
    void *state, hyle::compute::detail::CompletionLifetime lifetime,
    hyle::Status terminal_hint) noexcept {
  auto &probe = *static_cast<AsyncProbe *>(state);
  probe.destroyed = true;
  if (!lifetime.valid())
    return;
  if (!probe.complete) {
    probe.abandoned = lifetime;
    probe.abandoned_hint = std::move(terminal_hint);
    return;
  }
  hyle::Status terminal =
      probe.cancelled ? hyle::Status::Cancelled("work was cancelled")
                      : hyle::Status::Ok();
  if (terminal.ok() && !terminal_hint.ok())
    terminal = std::move(terminal_hint);
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

hyle::Result<hyle::compute::CompletionPoll> poll_async(void *state) noexcept {
  auto &probe = *static_cast<AsyncProbe *>(state);
  ++probe.polls;
  if (probe.observation_failed)
    return std::unexpected(hyle::Status::Unavailable("observation failed"));
  if (!probe.complete)
    return hyle::compute::CompletionPoll::kPending;
  if (probe.cancelled)
    return std::unexpected(hyle::Status::Cancelled("work was cancelled"));
  return hyle::compute::CompletionPoll::kComplete;
}

hyle::Status wait_async(void *state) noexcept {
  auto &probe = *static_cast<AsyncProbe *>(state);
  ++probe.waits;
  probe.complete = true;
  return probe.cancelled ? hyle::Status::Cancelled("work was cancelled")
                         : hyle::Status::Ok();
}

hyle::Status cancel_async(void *state) noexcept {
  auto &probe = *static_cast<AsyncProbe *>(state);
  ++probe.cancels;
  probe.cancelled = true;
  probe.complete = true;
  return hyle::Status::Ok();
}

const hyle::compute::detail::CompletionVTable kAsyncVTable{
    .destroy = &destroy_async,
    .poll = &poll_async,
    .wait = &wait_async,
    .request_cancel = &cancel_async,
};

struct LifetimeProbe final {
  std::size_t terminal_count = 0;
  std::size_t release_count = 0;
  bool terminal_ok = false;
  hyle::StatusCode terminal_code = hyle::StatusCode::Ok;
};

void observe_terminal(void *state, const hyle::Status &terminal) noexcept {
  auto &probe = *static_cast<LifetimeProbe *>(state);
  ++probe.terminal_count;
  probe.terminal_ok = terminal.ok();
  probe.terminal_code = terminal.code();
}

void release_lifetime(void *state) noexcept {
  ++static_cast<LifetimeProbe *>(state)->release_count;
}

struct OutcomeSourceProbe final {
  hyle::compute::Completion original;
  hyle::compute::detail::CompletionRetention retention;
  unsigned observations = 0;
  unsigned abandons = 0;
  unsigned releases = 0;

  hyle::compute::detail::OutcomeSources protocol() noexcept {
    using namespace hyle::compute;
    return {this,
        [](void *opaque) noexcept -> hyle::Result<CompletionPoll> {
          auto &source = *static_cast<OutcomeSourceProbe *>(opaque);
          ++source.observations;
          return source.original.poll();
        },
        [](void *opaque) noexcept {
          auto &source = *static_cast<OutcomeSourceProbe *>(opaque);
          ++source.observations;
          return source.original.wait();
        },
        [](void *opaque) noexcept {
          return static_cast<OutcomeSourceProbe *>(opaque)->original.request_cancel();
        },
        [](void *opaque) noexcept {
          auto &source = *static_cast<OutcomeSourceProbe *>(opaque);
          ++source.abandons;
          source.original = {};
        },
        [](void *opaque, detail::CompletionLifetime lifetime) noexcept {
          auto &source = *static_cast<OutcomeSourceProbe *>(opaque);
          source.retention.own = lifetime;
          const auto attached = detail::CompletionAccess::attach_retention(
              source.original, source.retention);
          EXPECT_TRUE(attached.ok());
        },
        [](void *opaque) noexcept {
          ++static_cast<OutcomeSourceProbe *>(opaque)->releases;
        }};
  }
};

void retire_abandoned(AsyncProbe &native) {
  ASSERT_TRUE(native.abandoned.valid());
  native.complete = true;
  auto lifetime = std::exchange(native.abandoned, {});
  lifetime.terminal(lifetime.state, native.abandoned_hint);
  lifetime.release(lifetime.state);
}

struct RetireAsyncOnExit final {
  AsyncProbe &native;
  hyle::compute::Completion &completion;
  ~RetireAsyncOnExit() {
    native.complete = true;
    completion = {};
    if (native.abandoned.valid()) retire_abandoned(native);
  }
};

TEST(PublicCompletionRetention, InlineSuccessAndErrorReleaseExactlyOnce) {
  using namespace hyle::compute;
  CompletionRetention node;
  LifetimeProbe lifetime;
  auto success = Completion::completed();
  ASSERT_TRUE(node.attach(success, &lifetime, &release_lifetime).ok());
  EXPECT_EQ(lifetime.release_count, 1u);
  EXPECT_TRUE(success.wait().ok());
  auto failure = Completion::completed(hyle::Status::Cancelled("inline cancellation"));
  ASSERT_TRUE(node.attach(failure, &lifetime, &release_lifetime).ok());
  EXPECT_EQ(lifetime.release_count, 2u);
  EXPECT_TRUE(failure.wait().IsCancelled());
  EXPECT_EQ(lifetime.release_count, 2u);

  struct SelfOwned final {
    CompletionRetention node;
    unsigned *releases;
  };
  unsigned releases = 0;
  auto *owner = new SelfOwned{{}, &releases};
  ASSERT_TRUE(owner->node.attach(success, owner, +[](void *opaque) noexcept {
    auto *retained = static_cast<SelfOwned *>(opaque);
    ++*retained->releases;
    delete retained;
  }).ok());
  EXPECT_EQ(releases, 1u);
}

TEST(PublicCompletionRetention, RejectionPreservesCallerReferenceAndActiveAttachment) {
  using namespace hyle::compute;
  AsyncProbe native;
  CompletionRetention node;
  LifetimeProbe original, rejected;
  Completion invalid;
  EXPECT_TRUE(node.attach(invalid, &rejected, &release_lifetime).IsFailedPrecondition());
  auto other = Completion::completed();
  EXPECT_TRUE(node.attach(other, nullptr, &release_lifetime).IsInvalidArgument());
  EXPECT_TRUE(node.attach(other, &rejected, nullptr).IsInvalidArgument());
  EXPECT_EQ(rejected.release_count, 0u);

  auto completion = detail::CompletionAccess::make(&kAsyncVTable, &native);
  RetireAsyncOnExit retire{native, completion};
  ASSERT_TRUE(node.attach(completion, &original, &release_lifetime).ok());
  EXPECT_TRUE(node.attach(other, &rejected, &release_lifetime).IsFailedPrecondition());
  EXPECT_EQ(original.release_count, 0u);
  EXPECT_EQ(rejected.release_count, 0u);
  EXPECT_EQ(native.polls, 0u);
  ASSERT_TRUE(completion.wait().ok());
  EXPECT_EQ(original.release_count, 1u);
  ASSERT_TRUE(node.attach(other, &rejected, &release_lifetime).ok());
  EXPECT_EQ(rejected.release_count, 1u);
}

TEST(PublicCompletionRetention, ObservationErrorRetainsUntilActualRuntimeRetirement) {
  using namespace hyle::compute;
  AsyncProbe native;
  native.observation_failed = true;
  CompletionRetention node;
  LifetimeProbe lifetime;
  auto completion = detail::CompletionAccess::make(&kAsyncVTable, &native);
  RetireAsyncOnExit retire{native, completion};
  ASSERT_TRUE(node.attach(completion, &lifetime, &release_lifetime).ok());
  auto observed = completion.poll();
  ASSERT_FALSE(observed.has_value());
  EXPECT_TRUE(observed.error().IsUnavailable());
  EXPECT_TRUE(native.abandoned.valid());
  EXPECT_EQ(lifetime.release_count, 0u);
  completion = {};
  EXPECT_EQ(lifetime.release_count, 0u);
  retire_abandoned(native);
  EXPECT_EQ(lifetime.release_count, 1u);
}

TEST(PublicCompletionRetention, AbandonedChainPreservesOriginalOwnerAndTailError) {
  using namespace hyle::compute;
  AsyncProbe native;
  LifetimeProbe original, second, third;
  detail::CompletionLifetime inherited{&original, &observe_terminal, &release_lifetime};
  CompletionRetention inner, outer;
  auto completion = detail::CompletionAccess::make(&kAsyncVTable, &native);
  RetireAsyncOnExit retire{native, completion};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, inherited).ok());
  ASSERT_TRUE(inner.attach(completion, &second, &release_lifetime).ok());
  ASSERT_TRUE(outer.attach(completion, &third, &release_lifetime).ok());
  detail::CompletionAccess::append_terminal(completion,
      hyle::Status::Unavailable("accepted prefix then rejected tail"));
  completion = {};
  EXPECT_TRUE(native.abandoned.valid());
  EXPECT_TRUE(native.abandoned_hint.IsUnavailable());
  EXPECT_EQ(original.release_count, 0u);
  EXPECT_EQ(second.release_count, 0u);
  EXPECT_EQ(third.release_count, 0u);
  EXPECT_EQ(native.waits, 0u);
  retire_abandoned(native);
  EXPECT_EQ(original.terminal_count, 1u);
  EXPECT_EQ(original.terminal_code, hyle::StatusCode::Unavailable);
  EXPECT_EQ(original.release_count, 1u);
  EXPECT_EQ(second.release_count, 1u);
  EXPECT_EQ(third.release_count, 1u);
}

struct SourceUpdateProbe final {
  struct NativeControls final {
    std::array<hyle::compute::Completion, 4> *storage = nullptr;
    auto &operator[](std::size_t i) noexcept { return (*storage)[i]; }
    const auto &operator[](std::size_t i) const noexcept { return (*storage)[i]; }
    auto begin() noexcept { return storage->begin(); }
    auto end() noexcept { return storage->end(); }
  } completions;
  hyle::Status known_terminal;
  std::size_t accepted = 0;
  bool active = false;
  SourceUpdateProbe() noexcept = default;
  explicit SourceUpdateProbe(std::array<hyle::compute::Completion, 4> &storage) noexcept
      : completions{&storage} {}
  ~SourceUpdateProbe() {
    if (completions.storage) for (auto &completion : completions) completion = {};
  }
  SourceUpdateProbe(SourceUpdateProbe &&other) noexcept { *this = std::move(other); }
  SourceUpdateProbe &operator=(SourceUpdateProbe &&other) noexcept {
    if (completions.storage) for (auto &completion : completions) completion = {};
    completions = std::exchange(other.completions, {});
    known_terminal = std::move(other.known_terminal);
    accepted = other.accepted;
    active = std::exchange(other.active, false);
    return *this;
  }
  bool valid() const noexcept { return active; }
  hyle::Status observe() noexcept {
    auto status = known_terminal;
    for (std::size_t i = 0; i < accepted; ++i) {
      auto polled = completions[i].poll();
      if (!polled) status.Update(polled.error());
    }
    return status;
  }
  hyle::Status wait() noexcept {
    auto status = known_terminal;
    for (std::size_t i = 0; i < accepted; ++i) status.Update(completions[i].wait());
    return status;
  }
  hyle::Status request_cancel() noexcept {
    hyle::Status status;
    for (std::size_t i = 0; i < accepted; ++i) status.Update(completions[i].request_cancel());
    return status;
  }
};
struct SourceUpdateAccessProbe final {
  static hyle::Result<hyle::compute::detail::CompletionDependency> dependency(
      const SourceUpdateProbe &source) noexcept {
    if (!source.valid() || !source.accepted)
      return std::unexpected(hyle::Status::InvalidArgument("source probe"));
    if (!source.known_terminal.ok()) return std::unexpected(source.known_terminal);
    return hyle::compute::detail::CompletionAccess::dependency(source.completions[source.accepted - 1]);
  }
  static hyle::Result<std::size_t> retirement_sources(SourceUpdateProbe &source,
      std::span<hyle::compute::Completion *> output) noexcept {
    if (output.empty()) return source.accepted;
    if (output.size() < source.accepted)
      return std::unexpected(hyle::Status::ResourceExhausted("source probe capacity"));
    for (std::size_t i = 0; i < source.accepted; ++i) output[i] = &source.completions[i];
    return source.accepted;
  }
};
using ColdSourceProbe = hyle::compute::detail::CopyOutcomeSource<SourceUpdateProbe, SourceUpdateAccessProbe>;

struct SourceAllocations final {
  unsigned attempts = 0, allocations = 0, frees = 0, fail_at = 0;
  hyle::mem::AllocatorRef ref() noexcept {
    return {
      [](void *opaque, std::size_t bytes, std::size_t alignment) noexcept -> void * {
        auto &state = *static_cast<SourceAllocations *>(opaque);
        if (++state.attempts == state.fail_at) return nullptr;
        auto *pointer = hyle::mem::system_allocator().allocate(bytes, alignment);
        if (pointer) ++state.allocations;
        return pointer;
      },
      [](void *opaque, void *pointer, std::size_t bytes, std::size_t alignment) noexcept {
        ++static_cast<SourceAllocations *>(opaque)->frees;
        hyle::mem::system_allocator().deallocate(pointer, bytes, alignment);
      },
      [](void *, void *pointer, std::size_t before, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        return hyle::mem::system_allocator().reallocate(pointer, before, bytes, alignment);
      }, this};
  }
};

TEST(CopyOutcomeSource, EveryColdAllocationOomRollsBackWithoutAnyNativeWork) {
  for (unsigned failure = 1; failure <= 4; ++failure) {
    SourceAllocations counts;
    counts.fail_at = failure;
    auto source = ColdSourceProbe::create(4, {.allocator = counts.ref()});
    ASSERT_FALSE(source.has_value());
    EXPECT_TRUE(source.error().IsResourceExhausted());
    EXPECT_EQ(counts.attempts, failure);
    EXPECT_EQ(counts.allocations, counts.frees);
  }
  EXPECT_FALSE(ColdSourceProbe::create(0));
  EXPECT_FALSE(ColdSourceProbe::create(SIZE_MAX));
}

TEST(CopyOutcomeSource, ResourceBudgetChargesColdNodesAndReleasesEveryFailedOrSuccessfulBlock) {
  using namespace hyle::compute;
  const auto resource = hyle::mem::system_resource();
  const std::array limits{ResourceBudgetLimit{resource, 1u << 16}};
  auto budget = ResourceBudget::create({.limits = limits}).value();
  const auto baseline = budget.usage(resource)->live_bytes;
  {
    auto source = ColdSourceProbe::create(4, {.resource_budget = budget}).value();
    EXPECT_GT(budget.usage(resource)->live_bytes, baseline);
  }
  EXPECT_EQ(budget.usage(resource)->live_bytes, baseline);
  const std::array tiny_limits{ResourceBudgetLimit{resource, baseline + 1}};
  auto tiny = ResourceBudget::create({.limits = tiny_limits}).value();
  EXPECT_TRUE(ColdSourceProbe::create(4, {.resource_budget = tiny}).error().IsResourceExhausted());
  EXPECT_EQ(tiny.usage(resource)->live_bytes, baseline);
}

TEST(CopyOutcomeSource, CapacityAndKnownPrefixFailureDoNotConsumeOriginalControl) {
  auto source = ColdSourceProbe::create(1).value();
  std::array<hyle::compute::Completion, 4> storage;
  SourceUpdateProbe update(storage);
  update.active = true; update.accepted = 2;
  update.completions[0] = hyle::compute::Completion::completed();
  update.completions[1] = hyle::compute::Completion::completed();
  EXPECT_TRUE(source.take(std::move(update)).error().IsResourceExhausted());
  EXPECT_TRUE(update.valid()); EXPECT_TRUE(source.available());
  update.accepted = 1;
  update.known_terminal = hyle::Status::Cancelled("accepted prefix then rejected tail");
  EXPECT_TRUE(source.take(std::move(update)).error().IsCancelled());
  EXPECT_TRUE(update.valid()); EXPECT_TRUE(source.available());
  auto two = ColdSourceProbe::create(2).value();
  update.known_terminal = hyle::Status::Ok(); update.accepted = 2;
  update.completions[0] = hyle::compute::Completion::completed(hyle::Status::Cancelled("used producer"));
  EXPECT_TRUE(two.take(std::move(update)).error().IsCancelled());
  EXPECT_TRUE(update.valid()); EXPECT_TRUE(two.available());
}

TEST(CopyOutcomeSource, UsedProducerUnknownCannotReleaseSourceOrResultBeforeActualRetirement) {
  using namespace hyle::compute;
  auto source = ColdSourceProbe::create(2).value();
  auto observer = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe producer;
  producer.observation_failed = true;
  std::array<Completion, 4> storage;
  SourceUpdateProbe update(storage);
  update.active = true; update.accepted = 2;
  update.completions[0] = detail::CompletionAccess::make(&kAsyncVTable, &producer);
  update.completions[1] = Completion::completed();
  auto protocol = source.take(std::move(update)).value();
  auto completion = observer.publish(Completion::completed(), protocol);
  LifetimeProbe result;
  detail::CompletionLifetime lifetime{&result, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  EXPECT_FALSE(source.available()); EXPECT_EQ(result.release_count, 0u);
  retire_abandoned(producer);
  EXPECT_TRUE(completion.wait().IsUnavailable());
  EXPECT_EQ(result.release_count, 1u);
  EXPECT_TRUE(source.available()); EXPECT_TRUE(source.update().valid());
  EXPECT_TRUE(source.update().observe().IsUnavailable());
  source.update() = {};
}

TEST(CopyOutcomeSource, AbandonWaitsForEveryAcceptedActualCallbackWithoutCallbackBlocking) {
  using namespace hyle::compute;
  auto source = ColdSourceProbe::create(2).value();
  auto observer = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe one, two, suffix;
  std::array<Completion, 4> storage;
  SourceUpdateProbe update(storage);
  update.active = true; update.accepted = 2;
  update.completions[0] = detail::CompletionAccess::make(&kAsyncVTable, &one);
  update.completions[1] = detail::CompletionAccess::make(&kAsyncVTable, &two);
  auto protocol = source.take(std::move(update)).value();
  auto completion = observer.publish(detail::CompletionAccess::make(&kAsyncVTable, &suffix), protocol);
  completion = {};
  EXPECT_FALSE(one.destroyed); EXPECT_FALSE(two.destroyed);
  retire_abandoned(suffix);
  EXPECT_TRUE(one.destroyed); EXPECT_TRUE(two.destroyed);
  retire_abandoned(two);
  EXPECT_FALSE(source.available()); EXPECT_FALSE(observer.available());
  retire_abandoned(one);
  EXPECT_TRUE(source.available()); EXPECT_TRUE(observer.available());
}

TEST(CopyOutcomeSource, RejectedSuffixAndDroppedColdHandleRetainExactlyTheirOwnReferences) {
  using namespace hyle::compute;
  SourceAllocations counts;
  auto source = ColdSourceProbe::create(2, {.allocator = counts.ref()}).value();
  std::array<Completion, 4> rejected_storage, accepted_storage;
  SourceUpdateProbe rejected(rejected_storage);
  rejected.active = true; rejected.accepted = 1;
  rejected.completions[0] = Completion::completed();
  auto protocol = source.take(std::move(rejected)).value();
  protocol.release(protocol.state); // No suffix accepted and no subscription.
  EXPECT_TRUE(source.available()); EXPECT_FALSE(source.update().valid());
  SourceUpdateProbe accepted(accepted_storage);
  accepted.active = true; accepted.accepted = 1;
  accepted.completions[0] = Completion::completed();
  protocol = source.take(std::move(accepted)).value();
  source.reset();
  EXPECT_EQ(counts.frees, 0u);
  auto observer = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  auto completion = observer.publish(Completion::completed(), protocol);
  EXPECT_TRUE(completion.wait().ok());
  EXPECT_EQ(counts.allocations, 4u); EXPECT_EQ(counts.frees, 4u);
}

TEST(CopyOutcomeSource, TwoSourceSlotsAndObserversReuse64EpisodesWithOnlyColdAllocations) {
  using namespace hyle::compute;
  SourceAllocations counts;
  auto one = ColdSourceProbe::create(4, {.allocator = counts.ref()}).value();
  auto two = ColdSourceProbe::create(4, {.allocator = counts.ref()}).value();
  auto left = detail::DependentCompletion::create(counts.ref()).value();
  auto right = detail::DependentCompletion::create(counts.ref()).value();
  EXPECT_EQ(counts.allocations, 10u);
  for (unsigned episode = 0; episode < 64; ++episode) {
    std::array<Completion, 4> first_storage, second_storage;
    SourceUpdateProbe first(first_storage), second(second_storage);
    first.active = true; first.accepted = 4;
    second.active = true; second.accepted = 4;
    for (auto &completion : first.completions) completion = Completion::completed();
    for (auto &completion : second.completions) completion = Completion::completed();
    auto a = left.publish(Completion::completed(), one.take(std::move(first)).value());
    auto b = right.publish(Completion::completed(), two.take(std::move(second)).value());
    EXPECT_FALSE(one.available()); EXPECT_FALSE(two.available());
    ASSERT_TRUE(b.wait().ok()); ASSERT_TRUE(a.wait().ok());
    EXPECT_TRUE(one.available()); EXPECT_TRUE(two.available());
    EXPECT_EQ(counts.allocations, 10u); EXPECT_EQ(counts.frees, 0u);
    one.update() = {}; two.update() = {};
  }
  one.reset(); two.reset(); left.reset(); right.reset();
  EXPECT_EQ(counts.allocations, counts.frees);
}

TEST(DependentCompletion, SourcesAreObservedOnlyAfterSuffixActuallyRetires) {
  using namespace hyle::compute;
  auto slot = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe source_native, suffix_native;
  OutcomeSourceProbe source;
  source.original = detail::CompletionAccess::make(&kAsyncVTable, &source_native);
  auto completion = slot.publish(
      detail::CompletionAccess::make(&kAsyncVTable, &suffix_native), source.protocol());
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  EXPECT_EQ(source.observations, 0u);
  EXPECT_FALSE(slot.available());
  suffix_native.complete = true;
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  EXPECT_EQ(source.observations, 1u);
  source_native.cancelled = true;
  source_native.complete = true;
  auto terminal = completion.poll();
  ASSERT_FALSE(terminal.has_value());
  EXPECT_TRUE(terminal.error().IsCancelled());
  EXPECT_TRUE(completion.wait().IsCancelled());
  EXPECT_EQ(source.releases, 1u);
  EXPECT_TRUE(slot.available());
}

TEST(DependentCompletion, UnknownSuffixRemainsPendingUntilActualCallback) {
  using namespace hyle::compute;
  auto slot = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe suffix_native;
  suffix_native.observation_failed = true;
  OutcomeSourceProbe source;
  source.original = Completion::completed(hyle::Status::Cancelled("source failed"));
  auto completion = slot.publish(
      detail::CompletionAccess::make(&kAsyncVTable, &suffix_native), source.protocol());
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  EXPECT_EQ(source.observations, 0u);
  EXPECT_EQ(source.releases, 0u);
  EXPECT_FALSE(slot.available());
  retire_abandoned(suffix_native);
  auto terminal = completion.poll();
  ASSERT_FALSE(terminal.has_value());
  // The source's inline known error was delivered by its subscription first.
  EXPECT_TRUE(terminal.error().IsCancelled());
  EXPECT_EQ(source.releases, 1u);
  EXPECT_TRUE(slot.available());
}

TEST(DependentCompletion, UnknownSourceKeepsResultAttachmentUntilBothRetire) {
  using namespace hyle::compute;
  auto slot = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe source_native;
  source_native.observation_failed = true;
  OutcomeSourceProbe source;
  source.original = detail::CompletionAccess::make(&kAsyncVTable, &source_native);
  auto completion = slot.publish(Completion::completed(), source.protocol());
  LifetimeProbe result;
  detail::CompletionLifetime lifetime{&result, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  EXPECT_EQ(result.release_count, 0u);
  EXPECT_EQ(source.releases, 0u);
  retire_abandoned(source_native);
  EXPECT_TRUE(completion.wait().IsUnavailable());
  EXPECT_EQ(result.release_count, 1u);
  EXPECT_EQ(result.terminal_code, hyle::StatusCode::Unavailable);
  EXPECT_EQ(source.releases, 1u);
  EXPECT_TRUE(slot.available());
}

TEST(DependentCompletion, AbandonTransfersSourceOnlyAfterBorrowingSuffixRetires) {
  using namespace hyle::compute;
  auto slot = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe source_native, suffix_native;
  OutcomeSourceProbe source;
  source.original = detail::CompletionAccess::make(&kAsyncVTable, &source_native);
  auto completion = slot.publish(
      detail::CompletionAccess::make(&kAsyncVTable, &suffix_native), source.protocol());
  LifetimeProbe result;
  detail::CompletionLifetime lifetime{&result, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  completion = {};
  EXPECT_EQ(source.abandons, 0u);
  EXPECT_EQ(source.releases, 0u);
  EXPECT_EQ(result.release_count, 0u);
  retire_abandoned(suffix_native);
  EXPECT_EQ(source.abandons, 1u);
  EXPECT_EQ(source.observations, 0u);
  EXPECT_EQ(source.releases, 0u);
  EXPECT_EQ(result.release_count, 0u);
  retire_abandoned(source_native);
  EXPECT_EQ(source.releases, 1u);
  EXPECT_EQ(result.release_count, 1u);
  EXPECT_TRUE(slot.available());
}

TEST(DependentCompletion, OriginalOwnedResultHasOneVerdictAndRetainsItsSlot) {
  using namespace hyle::compute;
  auto observer = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  auto slot = detail::ResultSlot<int>::create(hyle::mem::system_allocator()).value();
  for (int episode = 0; episode < 2; ++episode) {
    auto write = slot.begin().value();
    ASSERT_TRUE(write.emplace(int{23 + episode}).ok());
    auto result = write.publish(Completion::completed()).value();
    OutcomeSourceProbe source;
    source.original = Completion::completed();
    auto &original = detail::ResultAccess::completion(result);
    auto ignored = detail::ResultAccess::replace_completion(
        result, observer.publish(std::move(original), source.protocol()));
    EXPECT_FALSE(ignored.valid());
    ASSERT_TRUE(result.completion().wait().ok()); // Actual Graph consumption shape.
    auto lease = result.wait().value();
    EXPECT_EQ(*lease, 23 + episode);
    EXPECT_EQ(result.result().value().get(), lease.get());
    result = {};
    EXPECT_FALSE(slot.begin().has_value());
    lease = {};
    EXPECT_TRUE(observer.available());
    EXPECT_EQ(source.releases, 1u);
  }
  auto write = slot.begin().value();
  ASSERT_TRUE(write.emplace(99).ok());
  auto result = write.publish(Completion::completed()).value();
  OutcomeSourceProbe failed;
  failed.original = Completion::completed(hyle::Status::Cancelled("source"));
  auto &original = detail::ResultAccess::completion(result);
  auto replaced = detail::ResultAccess::replace_completion(
      result, observer.publish(std::move(original), failed.protocol()));
  EXPECT_FALSE(replaced.valid());
  EXPECT_TRUE(result.completion().wait().IsCancelled());
  EXPECT_TRUE(result.wait().error().IsCancelled());
  EXPECT_TRUE(result.result().error().IsCancelled());
}

TEST(DependentCompletion, ColdAllocationOomAndInvalidAllocatorAreReported) {
  using namespace hyle::compute;
  EXPECT_FALSE(detail::DependentCompletion::create({}).has_value());
  hyle::mem::AllocatorRef exhausted{
      [](void *, std::size_t, std::size_t) noexcept -> void * { return nullptr; },
      [](void *, void *, std::size_t, std::size_t) noexcept {},
      [](void *, void *, std::size_t, std::size_t, std::size_t) noexcept -> void * {
        return nullptr;
      }, nullptr};
  auto failed = detail::DependentCompletion::create(exhausted);
  ASSERT_FALSE(failed.has_value());
  EXPECT_TRUE(failed.error().IsResourceExhausted());
}

TEST(DependentCompletion, WaitDrainsUnknownNativeRetirementWithoutCallbackWaiting) {
  using namespace hyle::compute;
  auto slot = detail::DependentCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe suffix_native;
  suffix_native.observation_failed = true;
  OutcomeSourceProbe source;
  source.original = Completion::completed();
  auto completion = slot.publish(
      detail::CompletionAccess::make(&kAsyncVTable, &suffix_native), source.protocol());
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  std::barrier start(2);
  std::thread retirement([&] {
    start.arrive_and_wait();
    retire_abandoned(suffix_native);
  });
  start.arrive_and_wait();
  EXPECT_TRUE(completion.wait().IsUnavailable());
  retirement.join();
  EXPECT_EQ(source.releases, 1u);
  EXPECT_TRUE(slot.available());
}

TEST(DependentCompletion, TwoColdSlotsReuseWithoutWarmMetadataAllocations) {
  using namespace hyle::compute;
  struct Allocations final { std::size_t allocations = 0, frees = 0; } counters;
  hyle::mem::AllocatorRef allocator{
      [](void *opaque, std::size_t bytes, std::size_t alignment) noexcept -> void * {
        ++static_cast<Allocations *>(opaque)->allocations;
        return hyle::mem::system_allocator().allocate(bytes, alignment);
      },
      [](void *opaque, void *pointer, std::size_t bytes, std::size_t alignment) noexcept {
        ++static_cast<Allocations *>(opaque)->frees;
        hyle::mem::system_allocator().deallocate(pointer, bytes, alignment);
      },
      [](void *, void *pointer, std::size_t old_bytes, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        return hyle::mem::system_allocator().reallocate(pointer, old_bytes, bytes, alignment);
      }, &counters};
  auto first = detail::DependentCompletion::create(allocator).value();
  auto second = detail::DependentCompletion::create(allocator).value();
  EXPECT_EQ(counters.allocations, 2u);
  for (unsigned episode = 0; episode < 64; ++episode) {
    OutcomeSourceProbe one, two;
    one.original = Completion::completed();
    two.original = Completion::completed();
    auto left = first.publish(Completion::completed(), one.protocol());
    auto right = second.publish(Completion::completed(), two.protocol());
    EXPECT_FALSE(first.available());
    EXPECT_FALSE(second.available());
    ASSERT_TRUE(right.wait().ok());
    ASSERT_TRUE(left.wait().ok());
    EXPECT_EQ(one.releases, 1u);
    EXPECT_EQ(two.releases, 1u);
    EXPECT_TRUE(first.available());
    EXPECT_TRUE(second.available());
    EXPECT_EQ(counters.allocations, 2u);
    EXPECT_EQ(counters.frees, 0u);
  }
  first.reset(); second.reset();
  EXPECT_EQ(counters.frees, 2u);
}

TEST(CopyDependency, RetainedNativeIdentityIsResolvedWithoutObservationOrRetirement) {
  using namespace hyle::compute;
  auto inner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  auto outer = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe probe;
  auto wrapped = inner.publish(detail::CompletionAccess::make(&kAsyncVTable, &probe));
  auto source = outer.publish(std::move(wrapped));
  const auto dependency = detail::RetainedCompletion::native_dependency(source);
  ASSERT_TRUE(dependency.has_value());
  EXPECT_EQ(dependency->vtable, &kAsyncVTable);
  EXPECT_EQ(dependency->state, &probe);
  EXPECT_TRUE(dependency->pending());
  EXPECT_EQ(probe.polls, 0u); EXPECT_EQ(probe.waits, 0u); EXPECT_FALSE(probe.destroyed);
  EXPECT_FALSE(inner.available()); EXPECT_FALSE(outer.available());
  ASSERT_TRUE(source.wait().ok());
  EXPECT_TRUE(inner.available()); EXPECT_TRUE(outer.available());
}

TEST(CopyDependency, InlineErrorInvalidAndOuterTailHintArePreserved) {
  using namespace hyle::compute;
  auto invalid = detail::RetainedCompletion::native_dependency({});
  EXPECT_FALSE(invalid.has_value());
  auto inline_error = Completion::completed(hyle::Status::Cancelled("producer cancelled"));
  auto dependency = detail::RetainedCompletion::native_dependency(inline_error).value();
  EXPECT_FALSE(dependency.pending());
  ASSERT_NE(dependency.terminal_hint, nullptr);
  EXPECT_TRUE(dependency.terminal_hint->IsCancelled());
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe probe;
  auto source = owner.publish(detail::CompletionAccess::make(&kAsyncVTable, &probe));
  detail::CompletionAccess::append_terminal(source, hyle::Status::Unavailable("producer tail rejected"));
  dependency = detail::RetainedCompletion::native_dependency(source).value();
  EXPECT_EQ(dependency.vtable, &kAsyncVTable);
  EXPECT_TRUE(dependency.terminal_hint->IsUnavailable());
  EXPECT_FALSE(source.wait().ok());
}

TEST(CopyDependency, CachedUnknownErrorCannotManufactureNativeSuccessOrRelease) {
  using namespace hyle::compute;
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe probe; probe.observation_failed = true;
  auto source = owner.publish(detail::CompletionAccess::make(&kAsyncVTable, &probe));
  EXPECT_FALSE(source.poll().has_value());
  EXPECT_FALSE(owner.available());
  auto dependency = detail::RetainedCompletion::native_dependency(source).value();
  EXPECT_FALSE(dependency.pending());
  EXPECT_TRUE(dependency.terminal_hint->IsUnavailable());
  EXPECT_FALSE(probe.complete);
  ASSERT_TRUE(probe.abandoned.valid());
  probe.complete = true;
  probe.abandoned.terminal(probe.abandoned.state, probe.abandoned_hint);
  probe.abandoned.release(probe.abandoned.state); probe.abandoned = {};
  EXPECT_TRUE(owner.available());
}

TEST(Completion, RetainedObserverDrainsWhilePublicHandleIsHeld) {
  auto owner = hyle::compute::detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe probe;
  auto completion = owner.publish(hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &probe));
  EXPECT_FALSE(owner.available());
  ASSERT_TRUE(owner.wait().ok());
  EXPECT_EQ(probe.waits, 1u);
  EXPECT_FALSE(owner.available()); // the old public episode must not be overwritten
  ASSERT_TRUE(completion.wait().ok());
  EXPECT_TRUE(owner.available());
  EXPECT_EQ(probe.waits, 1u);
}

TEST(Completion, RetainedObserverForwardsAttachmentOnlyAfterActualUnknownRetirement) {
  auto owner = hyle::compute::detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe native;
  native.observation_failed = true;
  auto completion = owner.publish(hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &native));
  LifetimeProbe probe;
  hyle::compute::detail::CompletionLifetime lifetime{&probe, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(hyle::compute::detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  EXPECT_FALSE(completion.poll().has_value());
  EXPECT_FALSE(owner.wait().ok());
  EXPECT_EQ(probe.release_count, 0u);
  EXPECT_FALSE(owner.available());
  ASSERT_TRUE(native.abandoned.valid());
  native.complete = true;
  native.abandoned.terminal(native.abandoned.state, native.abandoned_hint);
  native.abandoned.release(native.abandoned.state);
  native.abandoned = {};
  EXPECT_EQ(probe.terminal_count, 1u);
  EXPECT_EQ(probe.release_count, 1u);
  EXPECT_FALSE(probe.terminal_ok);
  EXPECT_TRUE(owner.available());
}

TEST(Completion, RetainedObserverMayBeDroppedBeforePublicCompletion) {
  auto owner = hyle::compute::detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  AsyncProbe native;
  auto completion = owner.publish(hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &native));
  owner.reset();
  EXPECT_FALSE(native.destroyed);
  EXPECT_TRUE(completion.wait().ok());
  EXPECT_TRUE(native.destroyed);
}

TEST(Completion, RetainedCompoundObserverDoesNotTreatCachedRootErrorAsRetirement) {
  auto owner = hyle::compute::detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  const auto retirement = owner.reserve_retirement();
  auto completion = owner.publish(hyle::compute::Completion::completed(
      hyle::Status::Unavailable("accepted prefix observation failed")), true);
  LifetimeProbe probe;
  hyle::compute::detail::CompletionLifetime lifetime{&probe, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(hyle::compute::detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  EXPECT_FALSE(completion.poll().has_value());
  EXPECT_EQ(probe.release_count, 0u);
  EXPECT_FALSE(owner.available());
  retirement.terminal(retirement.state, hyle::Status::Unavailable("accepted prefix observation failed"));
  retirement.release(retirement.state);
  EXPECT_EQ(probe.release_count, 1u);
  EXPECT_FALSE(probe.terminal_ok);
  EXPECT_TRUE(owner.available());
}

struct PendingControlProbe final {
  std::atomic<unsigned> cancels{0}, waits{0}, notifications{0}, releases{0};
  std::atomic<std::uint64_t> event{0};
  std::atomic<bool> entered{false}, leave_wait{true};

  hyle::compute::detail::RetainedCompletionControl control() noexcept {
    return {
        this,
        [](void *opaque) noexcept {
          static_cast<PendingControlProbe *>(opaque)->cancels.fetch_add(1);
        },
        [](void *opaque, const std::atomic<std::uint64_t> &revision,
           std::uint64_t observed) noexcept {
          auto &probe = *static_cast<PendingControlProbe *>(opaque);
          probe.waits.fetch_add(1);
          probe.entered.store(true, std::memory_order_release);
          probe.entered.notify_all();
          while (revision.load(std::memory_order_acquire) == observed) {
            const auto event = probe.event.load(std::memory_order_acquire);
            if (revision.load(std::memory_order_acquire) != observed) break;
            probe.event.wait(event, std::memory_order_acquire);
          }
          probe.leave_wait.wait(false, std::memory_order_acquire);
        },
        [](void *opaque) noexcept {
          auto &probe = *static_cast<PendingControlProbe *>(opaque);
          probe.notifications.fetch_add(1);
          probe.event.fetch_add(1, std::memory_order_release);
          probe.event.notify_all();
        },
        [](void *opaque) noexcept {
          static_cast<PendingControlProbe *>(opaque)->releases.fetch_add(1);
        }};
  }
};

struct HandoffCancelProbe final {
  AsyncProbe native;
  std::atomic<bool> entered{false}, resume{false};
};
const hyle::compute::detail::CompletionVTable kHandoffCancelVTable{
    [](void *opaque, hyle::compute::detail::CompletionLifetime lifetime,
       hyle::Status hint) noexcept {
      destroy_async(&static_cast<HandoffCancelProbe *>(opaque)->native,
                    lifetime, std::move(hint));
    },
    [](void *opaque) noexcept {
      return poll_async(&static_cast<HandoffCancelProbe *>(opaque)->native);
    },
    [](void *opaque) noexcept {
      return wait_async(&static_cast<HandoffCancelProbe *>(opaque)->native);
    },
    [](void *opaque) noexcept {
      auto &probe = *static_cast<HandoffCancelProbe *>(opaque);
      probe.entered.store(true, std::memory_order_release);
      probe.entered.notify_all();
      probe.resume.wait(false, std::memory_order_acquire);
      return cancel_async(&probe.native);
    }};
TEST(Completion, RetainedPendingWaitPinsControlUntilConcurrentHandoffIsObserved) {
  using namespace hyle::compute;
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  PendingControlProbe control;
  control.leave_wait = false;
  auto pending = owner.publish_pending(control.control());
  auto &completion = pending.completion;
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  auto dependency = detail::RetainedCompletion::native_dependency(completion);
  EXPECT_FALSE(dependency);
  if (!dependency) EXPECT_TRUE(dependency.error().IsUnavailable());
  hyle::Status waited;
  std::thread waiter([&] { waited = completion.wait(); });
  control.entered.wait(false, std::memory_order_acquire);
  auto actual = Completion::completed();
  EXPECT_TRUE(pending.ticket.handoff(std::move(actual)).ok());
  EXPECT_FALSE(actual.valid());
  EXPECT_EQ(control.releases.load(), 0u); // the wait callback is still using it
  EXPECT_FALSE(owner.available());
  control.leave_wait.store(true, std::memory_order_release);
  control.leave_wait.notify_all();
  waiter.join();
  EXPECT_TRUE(waited.ok());
  EXPECT_EQ(control.waits.load(), 1u);
  EXPECT_EQ(control.releases.load(), 1u);
  EXPECT_TRUE(owner.available());
  auto extra = Completion::completed();
  EXPECT_TRUE(pending.ticket.handoff(std::move(extra)).IsFailedPrecondition());
  EXPECT_TRUE(extra.valid());
}

TEST(Completion, RetainedPendingCancelAndAbandonRaceActualInstallationSafely) {
  using namespace hyle::compute;
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  PendingControlProbe control;
  auto pending = owner.publish_pending(control.control());
  auto &completion = pending.completion;
  LifetimeProbe lifetime_probe;
  detail::CompletionLifetime lifetime{&lifetime_probe, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  EXPECT_TRUE(completion.request_cancel().ok());
  EXPECT_TRUE(completion.request_cancel().ok());
  EXPECT_EQ(control.cancels.load(), 1u);
  HandoffCancelProbe actual;
  hyle::Status handed;
  std::thread driver([&] {
    auto native = detail::CompletionAccess::make(&kHandoffCancelVTable, &actual);
    handed = pending.ticket.handoff(std::move(native));
  });
  actual.entered.wait(false, std::memory_order_acquire);
  // Actual native cancellation is running, so installation is not observable.
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  const auto unavailable = detail::RetainedCompletion::native_dependency(completion);
  EXPECT_FALSE(unavailable);
  if (!unavailable) EXPECT_TRUE(unavailable.error().IsUnavailable());
  completion = {};
  EXPECT_EQ(lifetime_probe.release_count, 0u);
  EXPECT_FALSE(owner.available());
  actual.resume.store(true, std::memory_order_release);
  actual.resume.notify_all();
  driver.join();
  EXPECT_TRUE(handed.ok());
  EXPECT_EQ(actual.native.cancels, 1u);
  EXPECT_TRUE(actual.native.destroyed);
  EXPECT_EQ(lifetime_probe.terminal_count, 1u);
  EXPECT_EQ(lifetime_probe.release_count, 1u);
  EXPECT_EQ(lifetime_probe.terminal_code, hyle::StatusCode::Cancelled);
  EXPECT_EQ(control.releases.load(), 1u);
  EXPECT_TRUE(owner.available());
}

TEST(Completion, RetainedPendingEarlyAbandonStillTransfersActualNativeRetirement) {
  using namespace hyle::compute;
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  auto pending = owner.publish_pending();
  auto &completion = pending.completion;
  LifetimeProbe probe;
  detail::CompletionLifetime lifetime{&probe, &observe_terminal, &release_lifetime};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  completion = {};
  EXPECT_FALSE(owner.available());
  EXPECT_EQ(probe.release_count, 0u);
  AsyncProbe actual;
  auto native = detail::CompletionAccess::make(&kAsyncVTable, &actual);
  ASSERT_TRUE(pending.ticket.handoff(std::move(native)).ok());
  EXPECT_TRUE(actual.destroyed);
  EXPECT_FALSE(actual.complete);
  EXPECT_EQ(probe.release_count, 0u);
  EXPECT_FALSE(owner.available());
  retire_abandoned(actual);
  EXPECT_EQ(probe.terminal_count, 1u);
  EXPECT_EQ(probe.release_count, 1u);
  EXPECT_TRUE(owner.available());
}

TEST(Completion, RetainedPendingInlineHandoffAllowsSynchronousRetirementCallbacks) {
  using namespace hyle::compute;
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  std::optional<detail::RetainedCompletionPending> pending(owner.publish_pending());
  auto completion = std::move(pending->completion);
  struct ReentrantProbe final {
    std::optional<detail::RetainedCompletionPending> *pending;
    LifetimeProbe lifetime;
    bool rejected = false, preserved = false;
  } probe{&pending};
  detail::CompletionLifetime lifetime{
      &probe,
      [](void *opaque, const hyle::Status &terminal) noexcept {
        auto &probe = *static_cast<ReentrantProbe *>(opaque);
        auto duplicate = Completion::completed();
        probe.rejected = (*probe.pending)->ticket.handoff(std::move(duplicate)).IsFailedPrecondition();
        probe.preserved = duplicate.valid();
        probe.pending->reset(); // may destroy the handoff ticket inline
        observe_terminal(&probe.lifetime, terminal);
      },
      [](void *opaque) noexcept {
        release_lifetime(&static_cast<ReentrantProbe *>(opaque)->lifetime);
      }};
  ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(completion, lifetime).ok());
  completion = {};
  auto actual = Completion::completed(hyle::Status::Unavailable("actual accepted prefix failed"));
  ASSERT_TRUE(pending->ticket.handoff(std::move(actual)).ok());
  EXPECT_EQ(probe.lifetime.terminal_count, 1u);
  EXPECT_EQ(probe.lifetime.release_count, 1u);
  EXPECT_EQ(probe.lifetime.terminal_code, hyle::StatusCode::Unavailable);
  EXPECT_TRUE(probe.rejected);
  EXPECT_TRUE(probe.preserved);
  EXPECT_FALSE(pending.has_value());
  EXPECT_TRUE(owner.available());
}

TEST(Completion, RetainedPendingTicketSurvivesObservationRootResetAndDestruction) {
  using namespace hyle::compute;
  for (const bool explicit_reset : {false, true}) {
    SourceAllocations allocations;
    detail::RetainedCompletionPending pending;
    {
      auto owner = detail::RetainedCompletion::create(allocations.ref()).value();
      pending = owner.publish_pending();
      if (explicit_reset) owner.reset();
    }
    EXPECT_EQ(allocations.frees, 0u);
    LifetimeProbe probe;
    detail::CompletionLifetime lifetime{&probe, &observe_terminal, &release_lifetime};
    ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(pending.completion, lifetime).ok());
    auto ticket = std::move(pending.ticket);
    if (explicit_reset) pending.completion = {}; // abandon before actual handoff
    AsyncProbe actual;
    actual.observation_failed = !explicit_reset;
    auto native = detail::CompletionAccess::make(&kAsyncVTable, &actual);
    ASSERT_TRUE(ticket.handoff(std::move(native)).ok());
    if (!explicit_reset) EXPECT_FALSE(pending.completion.poll());
    EXPECT_TRUE(actual.destroyed);
    EXPECT_FALSE(actual.complete);
    EXPECT_EQ(probe.release_count, 0u);
    EXPECT_EQ(allocations.frees, 0u);
    retire_abandoned(actual);
    EXPECT_EQ(probe.terminal_count, 1u);
    EXPECT_EQ(probe.release_count, 1u);
    EXPECT_EQ(allocations.frees, 1u);
  }
}

TEST(Completion, RetainedPendingDeferredLatchAndColdSlotReuseKeepOriginalContracts) {
  using namespace hyle::compute;
  SourceAllocations allocations;
  auto owner = detail::RetainedCompletion::create(allocations.ref()).value();
  for (unsigned episode = 0; episode < 32; ++episode) {
    const auto retirement = owner.reserve_retirement();
    auto pending = owner.publish_pending({}, true);
    auto actual = Completion::completed(hyle::Status::Unavailable("prefix observation failed"));
    ASSERT_TRUE(pending.ticket.handoff(std::move(actual)).ok());
    EXPECT_TRUE(pending.completion.wait().IsUnavailable());
    EXPECT_FALSE(owner.available());
    retirement.terminal(retirement.state, hyle::Status::Unavailable("native prefix retired"));
    retirement.release(retirement.state);
    EXPECT_TRUE(owner.available());
    EXPECT_EQ(allocations.attempts, 1u);
    EXPECT_EQ(allocations.frees, 0u);
  }
  owner.reset();
  EXPECT_EQ(allocations.allocations, 1u);
  EXPECT_EQ(allocations.frees, 1u);
}

TEST(Completion, RetainedControlPollKeepsPublicAndUnknownNativeRetirementGates) {
  using namespace hyle::compute;
  detail::RetainedCompletion empty;
  EXPECT_EQ(empty.poll().value(), CompletionPoll::kComplete);
  auto owner = detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  EXPECT_EQ(owner.poll().value(), CompletionPoll::kComplete);
  {
    auto pending = owner.publish_pending();
    EXPECT_EQ(owner.poll().value(), CompletionPoll::kPending);
    AsyncProbe actual;
    auto native = detail::CompletionAccess::make(&kAsyncVTable, &actual);
    ASSERT_TRUE(pending.ticket.handoff(std::move(native)).ok());
    EXPECT_EQ(owner.poll().value(), CompletionPoll::kPending);
    actual.complete = true;
    EXPECT_EQ(owner.poll().value(), CompletionPoll::kComplete);
    EXPECT_FALSE(owner.available()); // successful root observation keeps old proxy
    ASSERT_TRUE(pending.completion.wait().ok());
    EXPECT_TRUE(owner.available());
    EXPECT_EQ(actual.waits, 0u); // the original native poll already settled it
  }
  {
    auto pending = owner.publish_pending();
    LifetimeProbe probe;
    detail::CompletionLifetime lifetime{&probe, &observe_terminal, &release_lifetime};
    ASSERT_TRUE(detail::CompletionAccess::attach_lifetime(pending.completion, lifetime).ok());
    AsyncProbe actual;
    actual.observation_failed = true;
    auto native = detail::CompletionAccess::make(&kAsyncVTable, &actual);
    ASSERT_TRUE(pending.ticket.handoff(std::move(native)).ok());
    const auto failed = owner.poll();
    EXPECT_FALSE(failed);
    if (!failed) EXPECT_TRUE(failed.error().IsUnavailable());
    EXPECT_FALSE(owner.available());
    EXPECT_FALSE(pending.completion.poll());
    const auto cached = owner.poll();
    EXPECT_FALSE(cached);
    if (!cached) EXPECT_TRUE(cached.error().IsUnavailable());
    EXPECT_EQ(actual.polls, 1u);
    EXPECT_EQ(probe.release_count, 0u);
    EXPECT_FALSE(owner.available()); // cached error/public settle is not retirement
    retire_abandoned(actual);
    EXPECT_EQ(probe.terminal_count, 1u);
    EXPECT_EQ(probe.release_count, 1u);
    EXPECT_TRUE(owner.available());
    const auto retired = owner.poll();
    EXPECT_FALSE(retired);
    if (!retired) EXPECT_TRUE(retired.error().IsUnavailable());
  }
  owner.reset();
  EXPECT_EQ(owner.poll().value(), CompletionPoll::kComplete);
}

TEST(Completion, ImmediateSuccessAndFailureRemainInline) {
  auto success = hyle::compute::Completion::completed();
  auto ready = success.poll();
  ASSERT_TRUE(ready.has_value());
  EXPECT_EQ(*ready, hyle::compute::CompletionPoll::kComplete);
  EXPECT_TRUE(success.wait().ok());
  EXPECT_TRUE(success.request_cancel().ok());

  auto failure = hyle::compute::Completion::completed(
      hyle::Status::Unavailable("device disappeared"));
  auto failed_poll = failure.poll();
  ASSERT_FALSE(failed_poll.has_value());
  EXPECT_TRUE(failed_poll.error().IsUnavailable());
  EXPECT_TRUE(failure.wait().IsUnavailable());
}

TEST(Completion, DeferredTerminalDiagnosticReachesPublicPollAndWait) {
  using namespace hyle::compute;
  for (const bool use_poll : {false, true}) {
    auto made = detail::RetainedCompletion::create(hyle::mem::system_allocator());
    ASSERT_TRUE(made);
    auto observer = std::move(*made);
    auto retirement = observer.reserve_retirement();
    auto done = observer.publish(Completion::completed(), true);
    // A successfully retired native suffix may still expose a producer's
    // cached diagnostic through the common owner's final publication.
    retirement.terminal(retirement.state, hyle::Status::Unavailable("producer observation failed"));
    retirement.release(retirement.state);
    if (use_poll) {
      auto ready = done.poll();
      ASSERT_FALSE(ready);
      EXPECT_TRUE(ready.error().IsUnavailable());
    } else {
      EXPECT_TRUE(done.wait().IsUnavailable());
    }
    EXPECT_TRUE(observer.available());
  }
}

TEST(Completion, CompositeRetentionPreservesBothOwnersAndRejectsReuse) {
  AsyncProbe probe;
  LifetimeProbe first, second, third;
  using namespace hyle::compute::detail;
  CompletionLifetime original{&first, &observe_terminal, &release_lifetime};
  CompletionRetention extra{{&second, &observe_terminal, &release_lifetime}};
  CompletionRetention outer{{&third, &observe_terminal, &release_lifetime}};
  auto completion = CompletionAccess::make(&kAsyncVTable, &probe);
  ASSERT_TRUE(CompletionAccess::attach_lifetime(completion, original).ok());
  ASSERT_TRUE(CompletionAccess::attach_retention(completion, extra).ok());
  EXPECT_TRUE(CompletionAccess::attach_retention(completion, extra)
                  .IsInvalidArgument());
  ASSERT_TRUE(CompletionAccess::attach_retention(completion, outer).ok());
  ASSERT_TRUE(completion.wait().ok());
  for (const auto *owner : {&first, &second, &third}) {
    EXPECT_EQ(owner->terminal_count, 1u);
    EXPECT_EQ(owner->release_count, 1u);
    EXPECT_TRUE(owner->terminal_ok);
  }
  EXPECT_FALSE(extra.attachment.valid());
  EXPECT_FALSE(outer.attachment.valid());
}

struct TrackedResult final {
  std::unique_ptr<int> value;
  int *destroyed = nullptr;
  TrackedResult(int number, int &counter)
      : value(std::make_unique<int>(number)), destroyed(&counter) {}
  TrackedResult(TrackedResult &&) noexcept = default;
  ~TrackedResult() {
    if (value)
      ++*destroyed;
  }
};

TEST(OwnedResult, RetainedResultSurvivesSlotAndSubmissionAndBlocksOverwrite) {
  using namespace hyle::compute;
  auto first =
      detail::ResultSlot<TrackedResult>::create(hyle::mem::system_allocator());
  auto second =
      detail::ResultSlot<TrackedResult>::create(hyle::mem::system_allocator());
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  int destroyed = 0;
  ResultLease<TrackedResult> lease;
  {
    auto write = first->begin();
    ASSERT_TRUE(write.has_value());
    EXPECT_FALSE(first->begin().has_value());
    ASSERT_TRUE(write->emplace(TrackedResult(42, destroyed)).ok());
    auto submission = write->publish(Completion::completed());
    ASSERT_TRUE(submission.has_value());
    auto ready = submission->wait();
    ASSERT_TRUE(ready.has_value());
    lease = *ready;
    EXPECT_EQ(*lease->value, 42);
    EXPECT_FALSE(first->begin().has_value());
    auto other = second->begin();
    ASSERT_TRUE(other.has_value());
    ASSERT_TRUE(other->emplace(TrackedResult(81, destroyed)).ok());
    auto another = other->publish(Completion::completed());
    ASSERT_TRUE(another.has_value());
    EXPECT_EQ(*another->wait().value()->value, 81);
  }
  first = detail::ResultSlot<TrackedResult>{};
  second = detail::ResultSlot<TrackedResult>{};
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(*lease->value, 42);
  auto copy = lease;
  lease = {};
  EXPECT_EQ(destroyed, 1);
  copy = {};
  EXPECT_EQ(destroyed, 2);
}

TEST(OwnedResult, AbandonmentKeepsNativeOwnerUntilRealTerminal) {
  using namespace hyle::compute;
  int destroyed = 0;
  AsyncProbe probe;
  LifetimeProbe invocation;
  detail::CompletionLifetime original{&invocation, &observe_terminal,
                                      &release_lifetime};
  {
    auto slot = detail::ResultSlot<TrackedResult>::create(
        hyle::mem::system_allocator());
    ASSERT_TRUE(slot.has_value());
    auto write = slot->begin();
    ASSERT_TRUE(write.has_value());
    ASSERT_TRUE(write->emplace(TrackedResult(17, destroyed)).ok());
    auto completion = detail::CompletionAccess::make(&kAsyncVTable, &probe);
    ASSERT_TRUE(
        detail::CompletionAccess::attach_lifetime(completion, original).ok());
    auto submission = write->publish(std::move(completion));
    ASSERT_TRUE(submission.has_value());
    EXPECT_FALSE(submission->result().has_value());
    EXPECT_FALSE(slot->begin().has_value());
    EXPECT_EQ(probe.waits, 0u);
  }
  EXPECT_EQ(destroyed, 0);
  EXPECT_EQ(invocation.release_count, 0u);
  ASSERT_TRUE(probe.abandoned.valid());
  const auto retired = std::exchange(probe.abandoned, {});
  retired.terminal(retired.state, hyle::Status::Ok());
  retired.release(retired.state);
  EXPECT_EQ(invocation.release_count, 1u);
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(probe.waits, 0u);
}

TEST(OwnedResult, FailureAndCancellationDoNotPublishReadyLease) {
  using namespace hyle::compute;
  auto slot =
      detail::ResultSlot<TrackedResult>::create(hyle::mem::system_allocator());
  ASSERT_TRUE(slot.has_value());
  int destroyed = 0;
  for (bool cancel : {false, true}) {
    AsyncProbe probe;
    auto write = slot->begin();
    ASSERT_TRUE(write.has_value());
    ASSERT_TRUE(write->emplace(TrackedResult(7, destroyed)).ok());
    auto completion = detail::CompletionAccess::make(&kAsyncVTable, &probe);
    if (!cancel)
      detail::CompletionAccess::append_terminal(
          completion, hyle::Status::Unavailable("late failure"));
    auto submission = write->publish(std::move(completion));
    ASSERT_TRUE(submission.has_value());
    if (cancel) {
      ASSERT_TRUE(submission->request_cancel().ok());
    }
    auto ready = submission->wait();
    ASSERT_FALSE(ready.has_value());
    EXPECT_EQ(ready.error().code(), cancel ? hyle::StatusCode::Cancelled
                                           : hyle::StatusCode::Unavailable);
    EXPECT_FALSE(submission->result().has_value());
    EXPECT_FALSE(slot->begin().has_value());
  }
  auto next = slot->begin();
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(destroyed, 2);
}

TEST(OwnedResult, InvalidPublishPreservesAcceptedCompletionAndWriteCanRecover) {
  using namespace hyle::compute;
  int destroyed = 0;
  auto slot =
      detail::ResultSlot<TrackedResult>::create(hyle::mem::system_allocator());
  ASSERT_TRUE(slot.has_value());
  auto write = slot->begin();
  ASSERT_TRUE(write.has_value());
  auto completion = Completion::completed();
  EXPECT_FALSE(write->publish(std::move(completion)).has_value());
  EXPECT_TRUE(completion.valid());
  ASSERT_TRUE(write->emplace(TrackedResult(9, destroyed)).ok());
  Completion invalid;
  EXPECT_FALSE(write->publish(std::move(invalid)).has_value());
  auto accepted = write->publish(std::move(completion));
  ASSERT_TRUE(accepted.has_value());
  EXPECT_FALSE(completion.valid());
  EXPECT_EQ(*accepted->wait().value()->value, 9);
  EXPECT_FALSE(write->publish(Completion::completed()).has_value());
}

TEST(OwnedResult, ColdAllocationFailureAndUnsubmittedRollback) {
  using namespace hyle::compute;
  EXPECT_FALSE(detail::ResultSlot<int>::create({}).has_value());
  hyle::mem::AllocatorRef exhausted{
      [](void *, std::size_t, std::size_t) noexcept -> void * {
        return nullptr;
      },
      [](void *, void *, std::size_t, std::size_t) noexcept {},
      [](void *, void *, std::size_t, std::size_t,
         std::size_t) noexcept -> void * { return nullptr; },
      nullptr};
  auto failed = detail::ResultSlot<int>::create(exhausted);
  ASSERT_FALSE(failed.has_value());
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  int destroyed = 0;
  auto slot =
      detail::ResultSlot<TrackedResult>::create(hyle::mem::system_allocator());
  ASSERT_TRUE(slot.has_value());
  {
    auto write = slot->begin();
    ASSERT_TRUE(write.has_value());
    ASSERT_TRUE(write->emplace(TrackedResult(3, destroyed)).ok());
  }
  EXPECT_EQ(destroyed, 1);
  EXPECT_TRUE(slot->begin().has_value());
  OwnedResult<int> invalid;
  EXPECT_FALSE(invalid.wait().has_value());
  EXPECT_FALSE(invalid.result().has_value());
}

TEST(OwnedResult, OveralignedOwnerAndConcurrentBeginAdmission) {
  using namespace hyle::compute;
  struct alignas(128) AlignedOwner final {
    int value = 0;
  };
  auto slot =
      detail::ResultSlot<AlignedOwner>::create(hyle::mem::system_allocator());
  ASSERT_TRUE(slot.has_value());
  std::barrier start(3);
  std::barrier admitted(3);
  std::atomic<int> winners{0};
  auto attempt = [&] {
    start.arrive_and_wait();
    auto write = slot->begin();
    if (write)
      winners.fetch_add(1, std::memory_order_relaxed);
    admitted.arrive_and_wait(); // keep a successful write until both attempted
  };
  std::thread first(attempt), second(attempt);
  start.arrive_and_wait();
  admitted.arrive_and_wait();
  first.join();
  second.join();
  EXPECT_EQ(winners.load(), 1);
  auto write = slot->begin();
  ASSERT_TRUE(write.has_value());
  ASSERT_TRUE(write->emplace(AlignedOwner{13}).ok());
  auto submitted = write->publish(Completion::completed());
  ASSERT_TRUE(submitted.has_value());
  auto ready = submitted->wait();
  ASSERT_TRUE(ready.has_value());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ready->get()) % 128, 0u);
  EXPECT_EQ((*ready)->value, 13);
}

#if !defined(HYLE_STATIC)
TEST(OwnedResult,
     ProducerDsoOwnsDestructionAfterConsumerLeaseOutlivesSubmission) {
  using namespace hyle::compute;
  int destroyed = 0;
  ResultLease<test::SharedResultOwner> lease;
  {
    auto submitted = test::make_shared_result(101, destroyed);
    ASSERT_TRUE(submitted.has_value());
    auto ready = submitted->wait();
    ASSERT_TRUE(ready.has_value());
    lease = *ready;
  }
  EXPECT_EQ(destroyed, 0);
  EXPECT_EQ(lease->value, 101);
  lease = {};
  EXPECT_EQ(destroyed, 1);
}
#endif

TEST(Completion, SimulatedAsyncStateSupportsMovePollWaitAndCancel) {
  AsyncProbe probe;
  {
    auto completion =
        hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &probe);
    auto pending = completion.poll();
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(*pending, hyle::compute::CompletionPoll::kPending);

    auto moved = std::move(completion);
    EXPECT_FALSE(completion.valid()); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(moved.request_cancel().ok());
    auto cancelled = moved.poll();
    ASSERT_FALSE(cancelled.has_value());
    EXPECT_TRUE(cancelled.error().IsCancelled());
    EXPECT_TRUE(moved.wait().IsCancelled());
    // First terminal observation detaches the provider state; later wait is an
    // inline cached-status query and cannot repeat backend cleanup.
    EXPECT_TRUE(probe.destroyed);
  }
  EXPECT_TRUE(probe.destroyed);
  EXPECT_EQ(probe.polls, 2u);
  EXPECT_EQ(probe.waits, 0u);
  EXPECT_EQ(probe.cancels, 1u);
}

TEST(Completion, LifetimeSettlesExactlyOnceAtObservedTerminal) {
  AsyncProbe async;
  LifetimeProbe lifetime;
  auto completion =
      hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &async);
  hyle::compute::detail::CompletionLifetime attachment{
      .state = &lifetime,
      .terminal = &observe_terminal,
      .release = &release_lifetime,
  };
  ASSERT_TRUE(hyle::compute::detail::CompletionAccess::attach_lifetime(
                  completion, attachment)
                  .ok());
  EXPECT_TRUE(completion.wait().ok());
  EXPECT_EQ(lifetime.terminal_count, 1u);
  EXPECT_EQ(lifetime.release_count, 1u);
  EXPECT_TRUE(lifetime.terminal_ok);
  EXPECT_TRUE(completion.wait().ok());
  EXPECT_EQ(lifetime.terminal_count, 1u);
  EXPECT_EQ(lifetime.release_count, 1u);
}

TEST(Completion, AbandonedHandleDefersLifetimeUntilBackendTerminal) {
  AsyncProbe async;
  LifetimeProbe lifetime;
  hyle::compute::detail::CompletionLifetime attachment{
      .state = &lifetime,
      .terminal = &observe_terminal,
      .release = &release_lifetime,
  };
  {
    auto completion =
        hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &async);
    ASSERT_TRUE(hyle::compute::detail::CompletionAccess::attach_lifetime(
                    completion, attachment)
                    .ok());
  }
  EXPECT_TRUE(async.destroyed);
  EXPECT_TRUE(async.abandoned.valid());
  EXPECT_EQ(lifetime.terminal_count, 0u);
  EXPECT_EQ(lifetime.release_count, 0u);

  async.complete = true;
  async.abandoned.terminal(async.abandoned.state, hyle::Status::Ok());
  async.abandoned.release(async.abandoned.state);
  async.abandoned = {};
  EXPECT_EQ(lifetime.terminal_count, 1u);
  EXPECT_EQ(lifetime.release_count, 1u);
  EXPECT_TRUE(lifetime.terminal_ok);
}

TEST(Completion, AbandonedHandleTransfersSavedTailErrorToBackend) {
  AsyncProbe async;
  LifetimeProbe lifetime;
  hyle::compute::detail::CompletionLifetime attachment{
      .state = &lifetime,
      .terminal = &observe_terminal,
      .release = &release_lifetime,
  };
  {
    auto completion =
        hyle::compute::detail::CompletionAccess::make(&kAsyncVTable, &async);
    hyle::compute::detail::CompletionAccess::append_terminal(
        completion, hyle::Status::Unavailable("tail enqueue failed"));
    ASSERT_TRUE(hyle::compute::detail::CompletionAccess::attach_lifetime(
                    completion, attachment)
                    .ok());
  }
  EXPECT_TRUE(async.abandoned.valid());
  EXPECT_TRUE(async.abandoned_hint.IsUnavailable());

  async.complete = true;
  async.abandoned.terminal(async.abandoned.state, async.abandoned_hint);
  async.abandoned.release(async.abandoned.state);
  async.abandoned = {};
  EXPECT_EQ(lifetime.terminal_code, hyle::StatusCode::Unavailable);
  EXPECT_EQ(lifetime.release_count, 1u);
}

TEST(Completion, FailedObservationHandsRetentionToActualRuntimeRetirement) {
  AsyncProbe probe;
  probe.observation_failed = true;
  LifetimeProbe lifetime;
  using namespace hyle::compute::detail;
  CompletionLifetime attachment{&lifetime, &observe_terminal, &release_lifetime};
  auto completion = CompletionAccess::make(&kAsyncVTable, &probe);
  ASSERT_TRUE(CompletionAccess::attach_lifetime(completion, attachment).ok());
  auto observed = completion.poll();
  ASSERT_FALSE(observed.has_value());
  EXPECT_TRUE(observed.error().IsUnavailable());
  EXPECT_TRUE(probe.abandoned.valid());
  EXPECT_TRUE(probe.abandoned_hint.IsUnavailable());
  EXPECT_EQ(lifetime.release_count, 0u);
  probe.complete = true;
  probe.abandoned.terminal(probe.abandoned.state, probe.abandoned_hint);
  probe.abandoned.release(probe.abandoned.state);
  probe.abandoned = {};
  EXPECT_EQ(lifetime.release_count, 1u);
  EXPECT_EQ(lifetime.terminal_count, 1u);
  EXPECT_EQ(lifetime.terminal_code, hyle::StatusCode::Unavailable);
}

} // namespace
