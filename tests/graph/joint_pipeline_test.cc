#include <gtest/gtest.h>

#include <array>

#include "../../src/graph/joint_pipeline.h"
#include "hyle/mem/quota_allocator.h"

namespace {
using Plan = hyle::graph::detail::JointPipelinePlan;
using Frontier = hyle::graph::detail::JointPipelineFrontier;
using Lane = Plan::Lane;

TEST(JointPipeline, JoinWaitsForBothOriginalBranchesAcrossRepeatedEpisodes) {
  Plan plan(hyle::mem::system_allocator());
  plan.append(Lane::kCpu, {});
  plan.append(Lane::kNative, {});
  plan.append(Lane::kCpu, std::array<std::size_t, 2>{0, 1});
  ASSERT_TRUE(plan.prepare().ok()); ASSERT_TRUE(plan.has_parallel_lanes);
  Frontier frontier(hyle::mem::system_allocator());
  ASSERT_TRUE(frontier.prepare(plan).ok());
  for (unsigned episode = 0; episode < 65; ++episode) {
    frontier.reset(plan);
    EXPECT_EQ(frontier.take(Lane::kNative), 1);
    EXPECT_EQ(frontier.take(Lane::kCpu), 0);
    EXPECT_FALSE(frontier.ready(Lane::kCpu));
    if (episode % 2 == 0) {
      frontier.complete(plan, 0);
      EXPECT_FALSE(frontier.ready(Lane::kCpu));
      frontier.complete(plan, 1);
    } else {
      frontier.complete(plan, 1);
      EXPECT_FALSE(frontier.ready(Lane::kCpu));
      frontier.complete(plan, 0);
    }
    EXPECT_EQ(frontier.take(Lane::kCpu), 2);
    EXPECT_FALSE(frontier.done());
    frontier.complete(plan, 2);
    EXPECT_TRUE(frontier.done());
  }
}

TEST(JointPipeline, ReadyNativeBranchPassesUnreadyBoundaryWithoutLosingControlOrder) {
  Plan plan(hyle::mem::system_allocator());
  plan.append(Lane::kCpu, {});
  plan.append(Lane::kNative, std::array<std::size_t, 1>{0}); // CPU-produced boundary.
  plan.append(Lane::kNative, {}); // Independent native work must start now.
  plan.append(Lane::kCpu, std::array<std::size_t, 2>{1, 2});
  ASSERT_TRUE(plan.prepare().ok());
  EXPECT_TRUE(plan.has_parallel_lanes);
  Frontier frontier(hyle::mem::system_allocator());
  ASSERT_TRUE(frontier.prepare(plan).ok());
  EXPECT_EQ(frontier.take(Lane::kNative), 2);
  EXPECT_EQ(frontier.take(Lane::kCpu), 0);
  frontier.complete(plan, 2);
  EXPECT_EQ(frontier.take(Lane::kNative), Frontier::kNone);
  frontier.complete(plan, 0);
  EXPECT_EQ(frontier.take(Lane::kNative), 1);
  EXPECT_FALSE(frontier.ready(Lane::kCpu));
  frontier.complete(plan, 1);
  EXPECT_EQ(frontier.take(Lane::kCpu), 3);
  frontier.complete(plan, 3); EXPECT_TRUE(frontier.done());
}

TEST(JointPipeline, StrictSerialAndSingleEndpointPlansNeedNoParallelDriver) {
  Plan serial(hyle::mem::system_allocator());
  serial.append(Lane::kCpu, {});
  serial.append(Lane::kNative, std::array<std::size_t, 1>{0});
  serial.append(Lane::kCpu, std::array<std::size_t, 1>{1});
  ASSERT_TRUE(serial.prepare().ok()); EXPECT_FALSE(serial.has_parallel_lanes);
  Plan native(hyle::mem::system_allocator());
  native.append(Lane::kNative, {}); native.append(Lane::kNative, {});
  ASSERT_TRUE(native.prepare().ok()); EXPECT_FALSE(native.has_parallel_lanes);
  Plan invalid(hyle::mem::system_allocator());
  invalid.append(Lane::kCpu, std::array<std::size_t, 1>{0});
  EXPECT_TRUE(invalid.prepare().IsInvalidArgument());
}

TEST(JointPipeline, FrontierQuotaFailureReleasesItsPartialColdAllocation) {
  Plan plan(hyle::mem::system_allocator()); plan.append(Lane::kCpu, {});
  ASSERT_TRUE(plan.prepare().ok());
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), sizeof(std::size_t));
  {
    Frontier frontier(quota.get());
    EXPECT_TRUE(frontier.prepare(plan).IsResourceExhausted());
  }
  EXPECT_EQ(quota.used(), 0);
}

TEST(JointPipeline, BindingReadyInputDoesNotWaitForUnrelatedNativeUpload) {
  Plan plan(hyle::mem::system_allocator());
  plan.append(Lane::kNative, {}); // Host view is already ready: no download.
  plan.append(Lane::kNative, {}); // Another input still requires real upload.
  plan.append(Lane::kCpu, std::array<std::size_t, 1>{0});
  plan.append(Lane::kNative, std::array<std::size_t, 1>{1});
  plan.append(Lane::kCpu, std::array<std::size_t, 2>{2, 3});
  ASSERT_TRUE(plan.prepare().ok());
  const std::array<std::uint8_t, 5> ready_inputs{1, 0, 0, 0, 0};
  EXPECT_TRUE(plan.parallel_after(ready_inputs));
  Frontier frontier(hyle::mem::system_allocator());
  ASSERT_TRUE(frontier.prepare(plan).ok());
  frontier.reset(plan, ready_inputs);
  EXPECT_EQ(frontier.take(Lane::kCpu), 2);
  EXPECT_EQ(frontier.take(Lane::kNative), 1);
  frontier.complete(plan, 2);
  EXPECT_FALSE(frontier.ready(Lane::kCpu));
  frontier.complete(plan, 1);
  EXPECT_EQ(frontier.take(Lane::kNative), 3);
  frontier.complete(plan, 3);
  EXPECT_EQ(frontier.take(Lane::kCpu), 4);
  frontier.complete(plan, 4); EXPECT_TRUE(frontier.done());
}

TEST(JointPipeline, AlreadyNativeInputDoesNotAddDriverToNecessarySerialChain) {
  Plan plan(hyle::mem::system_allocator());
  plan.append(Lane::kCpu, {}); // Conditional original CPU repack.
  plan.append(Lane::kNative, std::array<std::size_t, 1>{0}); // Conditional upload.
  plan.append(Lane::kCpu, {});
  plan.append(Lane::kNative, std::array<std::size_t, 2>{1, 2});
  plan.append(Lane::kCpu, std::array<std::size_t, 1>{3});
  ASSERT_TRUE(plan.prepare().ok());
  const std::array<std::uint8_t, 5> native_ready{1, 1, 0, 0, 0};
  EXPECT_FALSE(plan.parallel_after(native_ready));
  Frontier frontier(hyle::mem::system_allocator());
  ASSERT_TRUE(frontier.prepare(plan).ok()); frontier.reset(plan, native_ready);
  EXPECT_EQ(frontier.take(Lane::kCpu), 2);
  EXPECT_FALSE(frontier.ready(Lane::kNative));
  frontier.complete(plan, 2);
  EXPECT_EQ(frontier.take(Lane::kNative), 3);
  frontier.complete(plan, 3);
  EXPECT_EQ(frontier.take(Lane::kCpu), 4);
  frontier.complete(plan, 4); EXPECT_TRUE(frontier.done());
}
} // namespace
