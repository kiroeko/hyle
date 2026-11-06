#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_models/d5_gguf_test_support.h"
#include <array>
#include <vector>

namespace c = hyle::compute;
namespace g = hyle::graph;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

TEST(SharedMmvq, CapturedReplaysRecomputeForDistinctInputsAndFormats) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cuda::create_context().value();
  auto owner =
      c::StorageOwner::adopt(
          context, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value())
          .value();
  auto queue = context.create_queue().value();
  constexpr std::int64_t k = 512, n = 257;
  const auto a_desc = dense(c::DataType::kFloat32, {1, k});
  const auto y_desc = dense(c::DataType::kFloat32, {1, n});
  for (const auto format :
       {c::GgufQuantizedType::kQ8_0, c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K}) {
    SCOPED_TRACE(static_cast<int>(format));
    const auto w_desc = c::gguf_weight_description(format, c::DataType::kFloat32, k, n).value();
    auto builder = g::GraphBuilder::create().value();
    const auto a = builder.input(a_desc).value();
    const std::array weights{builder.input(w_desc).value(), builder.input(w_desc).value(),
                             builder.input(w_desc).value()};
    std::array<g::ValueId, 3> outputs;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto node =
          builder.add_node(c::Operation(c::MatMul{}), std::array{a, weights[i]}).value();
      outputs[i] = builder.result(node).value();
      ASSERT_TRUE(builder.publish(outputs[i]).ok());
    }
    auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, owner).value();
    bool grouped = false;
    for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
      const auto step = compiled.report.step(i).value();
      grouped |=
          step.policy == c::RoutePolicyId{0x48594c4547474355ULL, 1, 1} && step.node_count == 3;
    }
    ASSERT_TRUE(grouped);
    for (int lane = 0; lane < 2; ++lane) {
      auto slot = compiled.executable.create_invocation().value();
      auto values = d5_gguf_test::values(1, k, 11 + lane);
      auto input = upload<float>(owner.ref(), values);
      const auto p0 = d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 3), n, k,
                                         w_desc.required_bytes().value());
      const auto p1 = d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 7), n, k,
                                         w_desc.required_bytes().value());
      const auto p2 = d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 9), n, k,
                                         w_desc.required_bytes().value());
      std::array storage{upload<std::byte>(owner.ref(), p0.bytes),
                         upload<std::byte>(owner.ref(), p1.bytes),
                         upload<std::byte>(owner.ref(), p2.bytes)};
      std::array actual{output_buffer(owner.ref(), y_desc), output_buffer(owner.ref(), y_desc),
                        output_buffer(owner.ref(), y_desc)};
      const auto av = c::ConstTensorView::create(a_desc, input.view()).value();
      ASSERT_TRUE(slot.bind(a, av).ok());
      for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(
            slot.bind(weights[i], c::ConstTensorView::create(w_desc, storage[i].view()).value())
                .ok());
        ASSERT_TRUE(
            slot.bind(outputs[i], c::TensorView::create(y_desc, actual[i].view()).value()).ok());
      }
      for (int round = 0; round < 2; ++round) {
        if (round) {
          values = d5_gguf_test::values(1, k, 41 + lane);
          ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
        }
        ASSERT_TRUE(
            g::cuda::prepare_replay(slot, queue, {.mode = g::ReplayMode::kRequireCapture}).ok());
        if (round)
          EXPECT_GT(slot.replay_report().value().reused_capture_range_count, 0U);
        auto done = slot.submit(queue).value();
        ASSERT_TRUE(done.wait().ok());
        for (std::size_t i = 0; i < 3; ++i) {
          auto expected = output_buffer(owner.ref(), y_desc);
          const std::array args{av, c::ConstTensorView::create(w_desc, storage[i].view()).value()};
          ASSERT_TRUE(
              ggml_cuda_tensor_test::execute(context, c::Operation(c::MatMul{}), args,
                                             c::TensorView::create(y_desc, expected.view()).value())
                  .ok());
          EXPECT_EQ(half_test::read<float>(actual[i]), half_test::read<float>(expected));
        }
      }
    }
  }
}

TEST(SharedMmvq, TwoQueuesAndInvocationWorkspacesRemainIndependent) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cuda::create_context().value();
  auto owner =
      c::StorageOwner::adopt(
          context, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value())
          .value();
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  constexpr std::int64_t k = 512, n = 257;
  const auto ad = dense(c::DataType::kFloat32, {1, k}), yd = dense(c::DataType::kFloat32, {1, n});
  const auto format = c::GgufQuantizedType::kQ4K;
  const auto wd = c::gguf_weight_description(format, c::DataType::kFloat32, k, n).value();
  auto builder = g::GraphBuilder::create().value();
  const auto a = builder.input(ad).value();
  const std::array weights{builder.input(wd).value(), builder.input(wd).value(),
                           builder.input(wd).value()};
  std::array<g::ValueId, 3> outputs;
  for (std::size_t i = 0; i < 3; ++i) {
    outputs[i] =
        builder
            .result(builder.add_node(c::Operation(c::MatMul{}), std::array{a, weights[i]}).value())
            .value();
    ASSERT_TRUE(builder.publish(outputs[i]).ok());
  }
  auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, owner).value();
  std::array slots{compiled.executable.create_invocation().value(),
                   compiled.executable.create_invocation().value()};
  const auto p0 =
      d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 3), n, k, wd.required_bytes().value());
  const auto p1 =
      d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 7), n, k, wd.required_bytes().value());
  const auto p2 =
      d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 9), n, k, wd.required_bytes().value());
  std::array ws{upload<std::byte>(owner.ref(), p0.bytes), upload<std::byte>(owner.ref(), p1.bytes),
                upload<std::byte>(owner.ref(), p2.bytes)};
  auto first = d5_gguf_test::values(1, k, 11), second = d5_gguf_test::values(1, k, 27);
  std::array inputs{upload<float>(owner.ref(), first), upload<float>(owner.ref(), second)};
  std::array output0{output_buffer(owner.ref(), yd), output_buffer(owner.ref(), yd),
                     output_buffer(owner.ref(), yd)};
  std::array output1{output_buffer(owner.ref(), yd), output_buffer(owner.ref(), yd),
                     output_buffer(owner.ref(), yd)};
  for (std::size_t lane = 0; lane < 2; ++lane) {
    ASSERT_TRUE(
        slots[lane].bind(a, c::ConstTensorView::create(ad, inputs[lane].view()).value()).ok());
    for (std::size_t i = 0; i < 3; ++i) {
      ASSERT_TRUE(
          slots[lane].bind(weights[i], c::ConstTensorView::create(wd, ws[i].view()).value()).ok());
      auto &out = lane ? output1[i] : output0[i];
      ASSERT_TRUE(slots[lane].bind(outputs[i], c::TensorView::create(yd, out.view()).value()).ok());
    }
    ASSERT_TRUE(
        g::cuda::prepare_replay(slots[lane], queues[lane], {.mode = g::ReplayMode::kRequireCapture})
            .ok());
  }
  for (int round = 0; round < 2; ++round) {
    if (round) {
      first = d5_gguf_test::values(1, k, 51);
      second = d5_gguf_test::values(1, k, 67);
      ASSERT_TRUE(inputs[0].view().write(std::as_bytes(std::span(first))).ok());
      ASSERT_TRUE(inputs[1].view().write(std::as_bytes(std::span(second))).ok());
    }
    auto done0 = slots[0].submit(queues[0]).value();
    auto done1 = slots[1].submit(queues[1]).value();
    ASSERT_TRUE(done0.wait().ok());
    ASSERT_TRUE(done1.wait().ok());
    for (std::size_t lane = 0; lane < 2; ++lane)
      for (std::size_t i = 0; i < 3; ++i) {
        auto expected = output_buffer(owner.ref(), yd);
        const std::array args{c::ConstTensorView::create(ad, inputs[lane].view()).value(),
                              c::ConstTensorView::create(wd, ws[i].view()).value()};
        ASSERT_TRUE(
            ggml_cuda_tensor_test::execute(context, c::Operation(c::MatMul{}), args,
                                           c::TensorView::create(yd, expected.view()).value())
                .ok());
        auto &out = lane ? output1[i] : output0[i];
        EXPECT_EQ(half_test::read<float>(out), half_test::read<float>(expected));
      }
  }
}

TEST(SharedMmvq, UnsupportedWidthDoesNotOfferSharedRoute) {
  auto context = g::cuda::create_context().value();
  auto owner =
      c::StorageOwner::adopt(
          context, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value())
          .value();
  constexpr std::int64_t k = 256, n = 257;
  const auto ad = dense(c::DataType::kFloat32, {1, k});
  const auto wd =
      c::gguf_weight_description(c::GgufQuantizedType::kQ4K, c::DataType::kFloat32, k, n).value();
  auto builder = g::GraphBuilder::create().value();
  const auto a = builder.input(ad).value();
  for (int i = 0; i < 3; ++i) {
    const auto w = builder.input(wd).value();
    const auto node = builder.add_node(c::Operation(c::MatMul{}), std::array{a, w}).value();
    ASSERT_TRUE(builder.publish(builder.result(node).value()).ok());
  }
  const auto compiled =
      g::cuda::compile(std::move(builder).finish().value(), context, owner).value();
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = compiled.report.step(i).value();
    EXPECT_FALSE((step.policy == c::RoutePolicyId{0x48594c4547474355ULL, 1, 1} &&
                  step.preference == 201 && step.node_count == 3));
  }
}
