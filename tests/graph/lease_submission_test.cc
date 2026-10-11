#include <gtest/gtest.h>

#include "../../src/compute/completion_internal.h"
#include "hyle/graph/detail/lease_submission.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
struct NativePrefix final { cd::CompletionLifetime abandoned; };
const cd::CompletionVTable prefix_vtable{
    .destroy = [](void *owner, cd::CompletionLifetime lifetime, hyle::Status) noexcept {
      static_cast<NativePrefix *>(owner)->abandoned = lifetime;
    },
    .poll = [](void *) noexcept -> hyle::Result<c::CompletionPoll> {
      return std::unexpected(hyle::Status::Unavailable("unknown prefix observation"));
    },
    .wait = [](void *) noexcept { return hyle::Status::Unavailable("unknown prefix observation"); }};
struct Parent final {
  unsigned references = 1, terminals = 0;
  bool accepted = false;
  hyle::Status terminal;
};
hyle::Status begin(hyle::graph::detail::LeaseSubmission &submission, Parent &parent) {
  return submission.begin(&parent,
      [](void *owner) noexcept { ++static_cast<Parent *>(owner)->references; },
      [](void *owner) noexcept { --static_cast<Parent *>(owner)->references; },
      [](void *owner, hyle::Status terminal, bool accepted) noexcept {
        auto &parent = *static_cast<Parent *>(owner);
        ++parent.terminals; parent.terminal = std::move(terminal); parent.accepted = accepted;
      });
}
TEST(LeaseSubmission, UnknownObservationCannotRetireReadOrParentBeforeNativeRelease) {
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto write = pool.allocate(16).value();
  auto read = write.publish().value();
  auto submission = hyle::graph::detail::LeaseSubmission::create(1, 2).value();
  submission.bind(0, read);
  Parent parent;
  ASSERT_TRUE(begin(submission, parent).ok());
  NativePrefix native;
  auto completion = cd::CompletionAccess::make(&prefix_vtable, &native);
  ASSERT_TRUE(submission.protect(completion).ok());
  ASSERT_FALSE(completion.poll().has_value());
  submission.finish(hyle::Status::Unavailable("body failed after accepted prefix"), true);
  EXPECT_EQ(parent.terminals, 0u);
  EXPECT_EQ(parent.references, 2u);
  EXPECT_EQ(pool.stats().active_uses, 1u);
  ASSERT_TRUE(native.abandoned.valid());
  native.abandoned.terminal(native.abandoned.state, hyle::Status::Unavailable("native retired with saved error"));
  native.abandoned.release(native.abandoned.state);
  native.abandoned = {};
  EXPECT_EQ(parent.terminals, 1u);
  EXPECT_EQ(parent.references, 1u);
  EXPECT_TRUE(parent.accepted);
  EXPECT_FALSE(parent.terminal.ok());
  EXPECT_EQ(parent.terminal.message(), "body failed after accepted prefix");
  EXPECT_EQ(pool.stats().active_uses, 0u);
}
TEST(LeaseSubmission, UnacceptedBodyRollsBackAndClosedSourceRejectsBeforeAcceptance) {
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto write = pool.allocate(16).value();
  auto read = write.publish().value();
  auto submission = hyle::graph::detail::LeaseSubmission::create(1, 1).value();
  submission.bind(0, read);
  Parent parent;
  ASSERT_TRUE(begin(submission, parent).ok());
  submission.finish(hyle::Status::ResourceExhausted("scheduler rejected"), false);
  EXPECT_EQ(parent.terminals, 1u);
  EXPECT_FALSE(parent.accepted);
  EXPECT_EQ(pool.stats().active_uses, 0u);
  ASSERT_TRUE(pool.close().ok());
  EXPECT_FALSE(begin(submission, parent).ok());
  EXPECT_EQ(parent.terminals, 1u);
  EXPECT_EQ(pool.stats().active_uses, 0u);
}
TEST(LeaseSubmission, ColdMetadataBudgetRejectsWithoutBeginningAnEpisode) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1);
  auto rejected = hyle::graph::detail::LeaseSubmission::create(2, 3, {.allocator = quota.get()});
  EXPECT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
}
} // namespace
