#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include "completion_internal.h"
#include "hyle/compute/completion.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/detail/retained_completion.h"
#if !defined(HYLE_STATIC)
#include "result_owner_fixture.h"
#endif

namespace {

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
