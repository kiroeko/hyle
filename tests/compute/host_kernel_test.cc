#include <gtest/gtest.h>

#include <type_traits>

#include "hyle/compute/host_kernel.h"

namespace {

using hyle::Status;
using hyle::compute::Context;
using hyle::compute::Device;
using hyle::compute::Queue;

struct NothrowVoidKernel final {
  void operator()(int* value, int increment) const noexcept {
    *value += increment;
  }
};

struct NothrowStatusKernel final {
  Status operator()(bool succeed) const noexcept {
    return succeed ? Status::Ok()
                   : Status::InvalidArgument("kernel rejected input");
  }
};

struct ThrowingKernel final {
  void operator()() const {}
};

template <class Function, class... Args>
concept HostSubmittable = requires(Queue& queue, Function&& function,
                                   Args&&... args) {
  hyle::compute::submit_host(queue, static_cast<Function&&>(function),
                             static_cast<Args&&>(args)...);
};

static_assert(HostSubmittable<NothrowVoidKernel, int*, int>);
static_assert(HostSubmittable<NothrowStatusKernel, bool>);
static_assert(!HostSubmittable<ThrowingKernel>);

TEST(HostKernel, TypedVoidAndStatusCallablesCompleteSynchronously) {
  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  int value = 4;

  auto first = hyle::compute::submit_host(queue, NothrowVoidKernel{}, &value, 7);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(value, 11);
  EXPECT_TRUE(first->ready());
  EXPECT_TRUE(first->wait().ok());

  auto second =
      hyle::compute::submit_host(queue, NothrowStatusKernel{}, false);
  ASSERT_TRUE(second.has_value());
  EXPECT_TRUE(second->ready());
  EXPECT_TRUE(second->status().IsInvalidArgument());
}

TEST(HostKernel, ClosedAndInvalidQueuesRejectBeforeInvoking) {
  int calls = 0;
  const auto kernel = [&calls]() noexcept { ++calls; };

  Queue invalid;
  auto rejected = hyle::compute::submit_host(invalid, kernel);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(calls, 0);

  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(queue.close().ok());
  rejected = hyle::compute::submit_host(queue, kernel);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(calls, 0);
}

TEST(Completion, MoveMakesTheSourceInvalid) {
  auto completion = hyle::compute::Completion::completed();
  auto moved = std::move(completion);
  EXPECT_FALSE(completion.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(completion.wait().IsFailedPrecondition());
  EXPECT_TRUE(moved.ready());
  EXPECT_TRUE(moved.wait().ok());
}

}  // namespace
