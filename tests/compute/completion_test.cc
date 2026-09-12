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
};

void destroy_async(void *state) noexcept {
  static_cast<AsyncProbe *>(state)->destroyed = true;
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
    EXPECT_FALSE(probe.destroyed);
  }
  EXPECT_TRUE(probe.destroyed);
  EXPECT_EQ(probe.polls, 2u);
  EXPECT_EQ(probe.waits, 1u);
  EXPECT_EQ(probe.cancels, 1u);
}

} // namespace
