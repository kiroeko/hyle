#include "hyle/compute/apple.h"

#include <array>
#include <atomic>
#include <thread>

#include <gtest/gtest.h>
#include <mlx/ops.h>

namespace {
namespace apple = hyle::compute::apple;
namespace mx = mlx::core;

TEST(AppleRegion, ChangedInputsRetainedResultsAndNoRetrace) {
  auto context = apple::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  std::atomic<int> traces{0};
  const std::vector<mx::array> inputs{
      mx::array({1.f, 2.f, 3.f, 4.f}, {2, 2}),
      mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
  auto program = context->compile([&](const auto &args, mx::Stream stream) {
    traces.fetch_add(1);
    return std::vector<mx::array>{mx::matmul(args[0], args[1], stream)};
  }, inputs);
  ASSERT_TRUE(program) << program.error().message();
  auto a = program->create_invocation();
  auto b = program->create_invocation();
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(a->bind(inputs).ok());
  auto first = a->submit();
  ASSERT_TRUE(first);
  auto old = first->wait();
  ASSERT_TRUE(old);
  EXPECT_FLOAT_EQ(old->get()->arrays()[0].data<float>()[3], 12.f);
  first = hyle::compute::OwnedResult<apple::Results>{};
  EXPECT_FALSE(a->submit()); // a retained lease prevents slot overwrite
  for (int run = 1; run <= 3; ++run) {
    auto changed = inputs;
    changed[0] = mx::array({float(run), 1.f, 2.f, 3.f}, {2, 2});
    ASSERT_TRUE(b->bind(changed).ok());
    auto submitted = b->submit();
    ASSERT_TRUE(submitted);
    auto result = submitted->wait();
    ASSERT_TRUE(result);
    EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[0], float(run * 2));
    EXPECT_FLOAT_EQ(old->get()->arrays()[0].data<float>()[3], 12.f);
  }
  EXPECT_EQ(traces.load(), 1);
  ASSERT_TRUE(context->close().ok());
  EXPECT_FLOAT_EQ(old->get()->arrays()[0].data<float>()[3], 12.f);
  EXPECT_FALSE(b->submit());
}

TEST(AppleRegion, RejectsForeignStreamPendingInputAndWrongShape) {
  auto context = apple::Context::create();
  ASSERT_TRUE(context);
  const std::vector<mx::array> inputs{mx::array({1.f, 2.f})};
  auto wrong_stream = context->compile([](const auto &args, mx::Stream) {
    return std::vector<mx::array>{mx::negative(args[0], mx::Device::cpu)};
  }, inputs);
  EXPECT_FALSE(wrong_stream);
  auto program = context->compile([](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::negative(args[0], stream)};
  }, inputs);
  ASSERT_TRUE(program);
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  EXPECT_FALSE(slot->submit());
  const std::array pending{mx::negative(inputs[0], mx::Device::gpu)};
  EXPECT_FALSE(slot->bind(pending).ok());
  const std::array wrong_shape{mx::array({1.f, 2.f, 3.f})};
  EXPECT_FALSE(slot->bind(wrong_shape).ok());
  ASSERT_TRUE(slot->bind(inputs).ok());
  auto work = slot->submit();
  ASSERT_TRUE(work);
  auto result = work->wait();
  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[1], -2.f);
}

TEST(AppleRegion, AbandonAndConcurrentWaitSurviveContextClose) {
  auto context = apple::Context::create();
  ASSERT_TRUE(context);
  const std::vector<mx::array> inputs{mx::array({1.f, 2.f, 3.f})};
  auto program = context->compile([](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::square(args[0], stream)};
  }, inputs);
  ASSERT_TRUE(program);
  auto a = program->create_invocation();
  auto b = program->create_invocation();
  ASSERT_TRUE(a && b);
  ASSERT_TRUE(a->bind(inputs).ok());
  ASSERT_TRUE(b->bind(inputs).ok());
  {
    auto abandoned = a->submit();
    ASSERT_TRUE(abandoned);
  }
  auto submitted = b->submit();
  ASSERT_TRUE(submitted);
  std::atomic<bool> checked{false};
  std::thread waiter([work = std::move(*submitted), &checked]() mutable {
    auto result = work.wait();
    checked.store(result && result->get()->arrays()[0].data<float>()[2] == 9.f);
  });
  EXPECT_TRUE(context->close().ok());
  waiter.join();
  EXPECT_TRUE(checked.load());
}
} // namespace
