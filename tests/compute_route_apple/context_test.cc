#include "hyle/compute/apple.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

#include <gtest/gtest.h>
#include <mlx/backend/common/utils.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "hyle/compute/low_precision.h"

namespace {
namespace apple = hyle::compute::apple;
namespace mx = mlx::core;

TEST(AppleRegion, BroadcastScalarReshapeKeepsOneStoredElementAcrossWideAxes) {
  const auto stream = mx::default_stream(mx::Device::gpu);
  auto input = mx::broadcast_to(mx::array(std::uint8_t{37}),
                                {805306368, 4}, stream);
  mx::eval(input);
  ASSERT_EQ(input.data_size(), 1u);
  const mx::Shape shape{96, 4096, 64, 128};
  // Inspect the planner before evaluation: an unpatched SDK must fail the
  // assertion here instead of attempting the multi-gigabyte logical copy.
  mx::array description(shape, mx::uint8, nullptr, {});
  const auto [copy, strides] = mx::prepare_reshape(input, description);
  ASSERT_FALSE(copy);
  EXPECT_EQ(strides, mx::Strides(shape.size(), 0));
  auto reshaped = mx::reshape(input, shape, stream);
  mx::eval(reshaped);
  EXPECT_EQ(reshaped.data_size(), 1u);
  EXPECT_EQ(reshaped.data_shared_ptr(), input.data_shared_ptr());
  EXPECT_EQ(reshaped.data<std::uint8_t>()[0], 37);
}

TEST(AppleRegion, BroadcastByteViewMaterializesOnlyOneWord) {
  const auto stream = mx::default_stream(mx::Device::gpu);
  auto owner = mx::array({std::uint8_t{0}, std::uint8_t{37}});
  auto byte = mx::slice(owner, {1}, {2}, stream);
  auto repeated = mx::broadcast_to(mx::reshape(byte, {1, 1, 1}, stream),
                                  {2, 3, 4}, stream);
  mx::eval(repeated);
  ASSERT_EQ(repeated.data_size(), 1u);
  ASSERT_EQ(repeated.offset(), 1);
  ASSERT_EQ(repeated.strides(), (mx::Strides{0, 0, 0}));
  auto viewed = mx::view(repeated, mx::float32, stream);
  mx::eval(viewed);
  EXPECT_EQ(viewed.shape(), (mx::Shape{2, 3, 1}));
  EXPECT_EQ(viewed.data_size(), 1u);
  EXPECT_EQ(viewed.strides(), (mx::Strides{0, 0, 0}));
  // A real aligned four-byte word owns the repeated bits independently of
  // the original single byte, which need not have an aligned byte offset.
  EXPECT_NE(viewed.data_shared_ptr(), repeated.data_shared_ptr());
  EXPECT_EQ(viewed.offset(), 0);
  owner = mx::array(std::uint8_t{0});
  byte = mx::array(std::uint8_t{0});
  repeated = mx::array(std::uint8_t{0});
  auto words = mx::contiguous(mx::view(viewed, mx::uint32, stream), false, stream);
  mx::eval(words);
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(words.data<std::uint32_t>()[i], 0x25252525u);
}

TEST(AppleRegion, GroupedWeightPreparationDoesNotFreezeBroadcastStrides) {
  namespace c = hyle::compute;
  auto context = apple::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  for (const bool four : {false, true}) {
    auto description = four ? c::float4_grouped_weight_description(2, 129, 128)
                            : c::float8_grouped_weight_description(2, 129, 128);
    ASSERT_TRUE(description) << description.error().message();
    auto spec = apple::input_spec(*description);
    ASSERT_TRUE(spec) << spec.error().message();
    ASSERT_EQ(spec->shape.size(), 1u);
    const std::array specs{*spec};
    auto program = context->compile(
        [](const auto &args, mx::Stream stream) {
          return std::vector<mx::array>{
              mx::slice(args[0], {0}, {4}, stream),
              mx::slice(args[0], {args[0].shape(0) - 4},
                        {args[0].shape(0)}, stream)};
        }, specs);
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation);
    std::vector<std::uint8_t> payload(spec->shape[0]);
    for (int phase : {1, 2}) {
      for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<std::uint8_t>((i * 7 + phase * 13) % 251);
      std::array inputs{mx::array(payload.data(), spec->shape, mx::uint8)};
      ASSERT_TRUE(invocation->bind(inputs).ok());
      auto submitted = invocation->submit();
      ASSERT_TRUE(submitted);
      auto result = submitted->wait();
      ASSERT_TRUE(result);
      const auto outputs = result->get()->arrays();
      ASSERT_EQ(outputs.size(), 2u);
      for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(outputs[0].data<std::uint8_t>()[i], payload[i]);
        EXPECT_EQ(outputs[1].data<std::uint8_t>()[i],
                  payload[payload.size() - 4 + i]);
      }
    }
    std::array short_input{mx::array(std::uint8_t{0})};
    EXPECT_FALSE(invocation->bind(short_input).ok());
  }
}

TEST(AppleRegion, ChangedInputsRetainedResultsAndNoRetrace) {
  auto context = apple::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  std::atomic<int> traces{0};
  const std::vector<mx::array> inputs{mx::array({1.f, 2.f, 3.f, 4.f}, {2, 2}),
                                      mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
  auto program = context->compile(
      [&](const auto &args, mx::Stream stream) {
        traces.fetch_add(1);
        return std::vector<mx::array>{mx::matmul(args[0], args[1], stream)};
      },
      inputs);
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
    EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[0],
                    float(run * 2));
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
  auto wrong_stream = context->compile(
      [](const auto &args, mx::Stream) {
        return std::vector<mx::array>{mx::negative(args[0], mx::Device::cpu)};
      },
      inputs);
  EXPECT_FALSE(wrong_stream);
  auto program = context->compile(
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::negative(args[0], stream)};
      },
      inputs);
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
  auto program = context->compile(
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::square(args[0], stream)};
      },
      inputs);
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
