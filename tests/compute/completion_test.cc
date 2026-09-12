#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

#include "completion_internal.h"
#include "hyle/compute/completion.h"

namespace {

struct AsyncProbe final {
  bool complete = false;
  bool cancelled = false;
  bool destroyed = false;
  std::size_t polls = 0;
  std::size_t waits = 0;
  std::size_t cancels = 0;
  hyle::compute::detail::CompletionLifetime abandoned{};
};

void destroy_async(
    void *state, hyle::compute::detail::CompletionLifetime lifetime) noexcept {
  auto &probe = *static_cast<AsyncProbe *>(state);
  probe.destroyed = true;
  if (!lifetime.valid())
    return;
  if (!probe.complete) {
    probe.abandoned = lifetime;
    return;
  }
  const hyle::Status terminal =
      probe.cancelled ? hyle::Status::Cancelled("work was cancelled")
                      : hyle::Status::Ok();
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

hyle::Result<hyle::compute::CompletionPoll> poll_async(void *state) noexcept {
  auto &probe = *static_cast<AsyncProbe *>(state);
  ++probe.polls;
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
};

void observe_terminal(void *state, const hyle::Status &terminal) noexcept {
  auto &probe = *static_cast<LifetimeProbe *>(state);
  ++probe.terminal_count;
  probe.terminal_ok = terminal.ok();
}

void release_lifetime(void *state) noexcept {
  ++static_cast<LifetimeProbe *>(state)->release_count;
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

} // namespace
