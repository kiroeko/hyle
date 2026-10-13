#include <gtest/gtest.h>

#include <array>

#include "../compute_batch/allocation_fault_fixture.h"
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
TEST(LeaseSubmission, ExactGroupsRetainEveryBlockUntilBothNativePrefixesRetire) {
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 1024, {.max_blocks = 4}).value();
  std::array<c::ReadLease, 3> reads;
  for (auto &read : reads) read = pool.allocate(16).value().publish().value();
  auto unused = pool.allocate(16).value();
  auto submission = hyle::graph::detail::LeaseSubmission::create(2, 2, {.max_page_reads = 4}).value();
  const std::array first{reads[0], reads[1]}, second{reads[1], reads[2]};
  ASSERT_TRUE(submission.bind_pages(0, first).ok());
  ASSERT_TRUE(submission.bind_pages(1, second).ok());
  Parent parent;
  ASSERT_TRUE(begin(submission, parent).ok());
  EXPECT_EQ(pool.stats().active_uses, 3u); // Shared block is admitted once.
  const std::array<std::byte, 16> bytes{};
  EXPECT_TRUE(unused.view()->write(bytes).ok());
  NativePrefix a, b;
  auto one = cd::CompletionAccess::make(&prefix_vtable, &a);
  auto two = cd::CompletionAccess::make(&prefix_vtable, &b);
  ASSERT_TRUE(submission.protect(one).ok());
  ASSERT_TRUE(submission.protect(two).ok());
  ASSERT_FALSE(one.poll()); ASSERT_FALSE(two.poll());
  submission.finish(hyle::Status::Unavailable("body failed after two accepted prefixes"), true);
  EXPECT_TRUE(submission.bind_pages(0, {}).IsUnavailable());
  ASSERT_TRUE(a.abandoned.valid()); ASSERT_TRUE(b.abandoned.valid());
  a.abandoned.terminal(a.abandoned.state, hyle::Status::Ok());
  a.abandoned.release(a.abandoned.state); a.abandoned = {};
  EXPECT_EQ(parent.terminals, 0u);
  EXPECT_EQ(pool.stats().active_uses, 3u);
  b.abandoned.terminal(b.abandoned.state, hyle::Status::Unavailable("last native prefix retired"));
  b.abandoned.release(b.abandoned.state); b.abandoned = {};
  EXPECT_EQ(parent.terminals, 1u); EXPECT_EQ(parent.references, 1u);
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_EQ(parent.terminal.message(), "body failed after two accepted prefixes");
  submission.reset();
}
TEST(LeaseSubmission, GroupReplacementAndRejectedBindingsPreserveOtherInputPermissions) {
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 1024, {.max_blocks = 4}).value();
  std::array<c::ReadLease, 4> reads;
  for (auto &read : reads) read = pool.allocate(16).value().publish().value();
  auto submission = hyle::graph::detail::LeaseSubmission::create(3, 1, {.max_page_reads = 4}).value();
  ASSERT_TRUE(submission.bind_pages(1, std::span(reads).subspan(1, 2)).ok());
  ASSERT_TRUE(submission.bind_pages(0, std::span(reads).first(1)).ok());
  ASSERT_TRUE(submission.bind_pages(2, std::span(reads).last(1)).ok());
  EXPECT_TRUE(submission.bind_pages(0, reads).IsResourceExhausted());
  const std::array<c::ReadLease, 1> empty_guard{};
  EXPECT_TRUE(submission.bind_pages(1, empty_guard).IsInvalidArgument());
  EXPECT_TRUE(submission.bind_pages(3, {}).IsInvalidArgument());
  Parent parent;
  ASSERT_TRUE(begin(submission, parent).ok());
  EXPECT_EQ(pool.stats().active_uses, 4u);
  submission.finish(hyle::Status::Ok(), false);
  ASSERT_TRUE(submission.bind_pages(1, {}).ok());
  ASSERT_TRUE(begin(submission, parent).ok());
  EXPECT_EQ(pool.stats().active_uses, 2u);
  submission.finish(hyle::Status::Ok(), false);
  ASSERT_TRUE(submission.bind_pages(1, std::span(reads).subspan(1, 2)).ok());
  submission.clear(0);
  submission.bind(1, reads[0]); // Ordinary binding replaces its whole page group.
  ASSERT_TRUE(begin(submission, parent).ok());
  EXPECT_EQ(pool.stats().active_uses, 2u);
  submission.finish(hyle::Status::Ok(), false);
  submission.reset();
  ASSERT_TRUE(begin(submission, parent).ok());
  EXPECT_EQ(pool.stats().active_uses, 0u);
  submission.finish(hyle::Status::Ok(), false);
}
TEST(LeaseSubmission, GroupOwnsCallerGuardsAndPartialAdmissionFailureRollsBack) {
  auto first = c::LeasePool::create(hyle::mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto second = c::LeasePool::create(hyle::mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto submission = hyle::graph::detail::LeaseSubmission::create(1, 1, {.max_page_reads = 2}).value();
  {
    const std::array reads{first.allocate(16).value().publish().value(),
                           second.allocate(16).value().publish().value()};
    ASSERT_TRUE(submission.bind_pages(0, reads).ok());
  }
  Parent parent;
  ASSERT_TRUE(begin(submission, parent).ok());
  EXPECT_EQ(first.stats().active_uses, 1u); EXPECT_EQ(second.stats().active_uses, 1u);
  submission.finish(hyle::Status::ResourceExhausted("unaccepted scheduling failure"), false);
  ASSERT_TRUE(second.close().ok());
  EXPECT_FALSE(begin(submission, parent).ok());
  EXPECT_EQ(first.stats().active_uses, 0u); EXPECT_EQ(second.stats().active_uses, 0u);
  EXPECT_EQ(parent.terminals, 1u);
  submission.reset();
}
TEST(LeaseSubmission, ExactGroupColdAllocationFaultsLeaveNoMetadataOrWarmAllocation) {
  hyle_f5_fault_fixture::Allocator measured;
  {
    auto created = hyle::graph::detail::LeaseSubmission::create(2, 2,
        {.allocator = measured.ref(), .max_page_reads = 4});
    ASSERT_TRUE(created);
  }
  EXPECT_EQ(measured.live, 0u);
  const auto attempts = measured.attempts;
  ASSERT_GT(attempts, 0u);
  for (std::size_t fail = 0; fail < attempts; ++fail) {
    hyle_f5_fault_fixture::Allocator fault;
    fault.fail = fail;
    {
      auto rejected = hyle::graph::detail::LeaseSubmission::create(2, 2,
          {.allocator = fault.ref(), .max_page_reads = 4});
      ASSERT_FALSE(rejected) << fail;
      EXPECT_TRUE(rejected.error().IsResourceExhausted()) << fail;
    }
    EXPECT_EQ(fault.live, 0u) << fail;
  }
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256, {.max_blocks = 2}).value();
  const std::array reads{pool.allocate(16).value().publish().value(),
                         pool.allocate(16).value().publish().value()};
  hyle_f5_fault_fixture::Allocator cold;
  auto submission = hyle::graph::detail::LeaseSubmission::create(2, 1,
      {.allocator = cold.ref(), .max_page_reads = 2}).value();
  const auto cold_attempts = cold.attempts;
  cold.fail = cold_attempts;
  Parent parent;
  for (unsigned step = 0; step < 64; ++step) {
    ASSERT_TRUE(submission.bind_pages(step % 2, reads).ok());
    ASSERT_TRUE(begin(submission, parent).ok());
    EXPECT_EQ(pool.stats().active_uses, 2u);
    submission.finish(hyle::Status::Ok(), false);
    submission.clear(step % 2);
  }
  EXPECT_EQ(cold.attempts, cold_attempts);
}
TEST(LeaseSubmission, UnselectedGroupsAndOverflowRejectBeforeAllocating) {
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto read = pool.allocate(16).value().publish().value();
  auto ordinary = hyle::graph::detail::LeaseSubmission::create(1, 1).value();
  ordinary.bind(0, read);
  EXPECT_TRUE(ordinary.bind_pages(0, std::span(&read, 1)).IsResourceExhausted());
  Parent parent;
  ASSERT_TRUE(begin(ordinary, parent).ok());
  EXPECT_EQ(pool.stats().active_uses, 1u);
  ordinary.finish(hyle::Status::Ok(), false);
  hyle_f5_fault_fixture::Allocator cold;
  auto rejected = hyle::graph::detail::LeaseSubmission::create(1, 1,
      {.allocator = cold.ref(), .max_page_reads = SIZE_MAX});
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_EQ(cold.attempts, 0u);
}
} // namespace
