#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_models/d5_gguf_test_support.h"
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace c = hyle::compute;
namespace g = hyle::graph;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

TEST(SharedMatvecAdd, OriginalLeafArithmeticAndCapturedRebinding) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cuda::create_context().value();
  auto owner =
      c::StorageOwner::adopt(
          context, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value())
          .value();
  auto queue = context.create_queue().value();
  constexpr std::int64_t k = 512, n = 257;
  const auto ad = dense(c::DataType::kFloat32, {1, k}), yd = dense(c::DataType::kFloat32, {1, n});
  for (const auto format :
       {c::GgufQuantizedType::kQ8_0, c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K}) {
    const auto wd = c::gguf_weight_description(format, c::DataType::kFloat32, k, n).value();
    for (bool expose : {false, true}) {
      SCOPED_TRACE(static_cast<int>(format));
      SCOPED_TRACE(expose);
      auto builder = g::GraphBuilder::create().value();
      const auto a = builder.input(ad).value(), w = builder.input(wd).value(),
                 residual = builder.input(yd).value();
      const auto mm = builder.add_node(c::Operation(c::MatMul{}), std::array{a, w}).value();
      const auto projected = builder.result(mm).value();
      const auto add = builder
                           .add_node(c::Operation(c::Binary{c::BinaryKind::kAdd}),
                                     std::array{residual, projected})
                           .value();
      const auto y = builder.result(add).value();
      ASSERT_TRUE(builder.publish(y).ok());
      if (expose)
        ASSERT_TRUE(builder.publish(projected).ok());
      auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, owner).value();
      bool pair = false;
      for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
        const auto step = compiled.report.step(i).value();
        pair |= step.policy == c::RoutePolicyId{0x48594c4547474355ULL, 1, 1} &&
                step.node_count == 2 && step.preference == 201;
      }
      EXPECT_EQ(pair, !expose);
      auto slot = compiled.executable.create_invocation().value();
      auto source_values = d5_gguf_test::values(1, k, 21);
      auto src = upload<float>(owner.ref(), source_values);
      auto packed = d5_gguf_test::pack(format, d5_gguf_test::values(n, k, 9), n, k,
                                      wd.required_bytes().value());
      // For Q8_0 the first half is the first block scale. Make the projected
      // first row NaN as well as the residual, with a different payload. This
      // checks the real binary operand order, not just one NaN plus a number.
      if (format == c::GgufQuantizedType::kQ8_0) {
        packed.bytes[0] = std::byte{0x23};
        packed.bytes[1] = std::byte{0x7e};
      }
      auto weight = upload<std::byte>(owner.ref(), packed.bytes);
      auto bias_values = d5_gguf_test::values(1, n, 33);
      bias_values[0] = std::bit_cast<float>(std::uint32_t{0x7fc00123});
      bias_values[1] = std::bit_cast<float>(std::uint32_t{0x80000000});
      auto bias = upload<float>(owner.ref(), bias_values);
      auto actual = output_buffer(owner.ref(), yd), observed = output_buffer(owner.ref(), yd);
      const auto av = c::ConstTensorView::create(ad, src.view()).value();
      const auto wv = c::ConstTensorView::create(wd, weight.view()).value();
      const auto bv = c::ConstTensorView::create(yd, bias.view()).value();
      ASSERT_TRUE(slot.bind(a, av).ok());
      ASSERT_TRUE(slot.bind(w, wv).ok());
      ASSERT_TRUE(slot.bind(residual, bv).ok());
      ASSERT_TRUE(slot.bind(y, c::TensorView::create(yd, actual.view()).value()).ok());
      if (expose)
        ASSERT_TRUE(slot.bind(projected, c::TensorView::create(yd, observed.view()).value()).ok());
      for (int round = 0; round < 2; ++round) {
        if (round) {
          source_values = d5_gguf_test::values(1, k, 51);
          ASSERT_TRUE(src.view().write(std::as_bytes(std::span(source_values))).ok());
        }
        auto plain = output_buffer(owner.ref(), yd), expected = output_buffer(owner.ref(), yd);
        const std::array original{av, wv};
        ASSERT_TRUE(ggml_cuda_tensor_test::execute(context, c::Operation(c::MatMul{}), original,
                                                   c::TensorView::create(yd, plain.view()).value())
                        .ok());
        if (format == c::GgufQuantizedType::kQ8_0) {
          const auto projected_nan = std::bit_cast<std::uint32_t>(half_test::read<float>(plain)[0]);
          ASSERT_EQ(projected_nan & 0x7f800000U, 0x7f800000U);
          ASSERT_NE(projected_nan & 0x007fffffU, 0U);
          ASSERT_NE(projected_nan, std::bit_cast<std::uint32_t>(bias_values[0]));
        }
        const std::array addends{bv, c::ConstTensorView::create(yd, plain.view()).value()};
        ASSERT_TRUE(ggml_cuda_tensor_test::execute(
                        context, c::Operation(c::Binary{c::BinaryKind::kAdd}), addends,
                        c::TensorView::create(yd, expected.view()).value())
                        .ok());
        ASSERT_TRUE(
            g::cuda::prepare_replay(slot, queue, {.mode = g::ReplayMode::kRequireCapture}).ok());
        if (round)
          EXPECT_GT(slot.replay_report().value().reused_capture_range_count, 0U);
        auto done = slot.submit(queue).value();
        ASSERT_TRUE(done.wait().ok());
        const auto got = half_test::read<float>(actual), want = half_test::read<float>(expected);
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t i = 0; i < got.size(); ++i)
          EXPECT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i]))
              << i;
        if (expose) {
          const auto actual_projection = half_test::read<float>(observed);
          const auto expected_projection = half_test::read<float>(plain);
          ASSERT_EQ(actual_projection.size(), expected_projection.size());
          for (std::size_t i = 0; i < actual_projection.size(); ++i)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_projection[i]),
                      std::bit_cast<std::uint32_t>(expected_projection[i])) << i;
        }
      }
    }
  }
}

TEST(SharedMatvecAdd, ExternalControlReaderPreventsContraction) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(
                   context, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
                                .value())
                   .value();
  constexpr std::int64_t k = 512, n = 257;
  const auto ad = dense(c::DataType::kFloat32, {1, k});
  const auto yd = dense(c::DataType::kFloat32, {1, n});
  const auto wd = c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
                                              c::DataType::kFloat32, k, n)
                      .value();
  auto builder = g::GraphBuilder::create().value();
  const auto a = builder.input(ad).value();
  const auto w = builder.input(wd).value();
  const auto residual = builder.input(yd).value();
  const auto mm = builder.add_node(c::Operation(c::MatMul{}), std::array{a, w}).value();
  const auto projected = builder.result(mm).value();
  const auto control = builder.add_node(c::Operation(c::Copy{}), std::array{a}).value();
  ASSERT_TRUE(builder.depends_on(control, mm).ok());
  const auto add = builder
                       .add_node(c::Operation(c::Binary{c::BinaryKind::kAdd}),
                                 std::array{residual, projected})
                       .value();
  ASSERT_TRUE(builder.depends_on(add, control).ok());
  ASSERT_TRUE(builder.publish(builder.result(control).value()).ok());
  ASSERT_TRUE(builder.publish(builder.result(add).value()).ok());
  auto compiled = g::cuda::compile(std::move(builder).finish().value(), context, owner).value();
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = compiled.report.step(i).value();
    if (step.node_count != 2 || step.preference != 201 ||
        step.policy != c::RoutePolicyId{0x48594c4547474355ULL, 1, 1})
      continue;
    bool has_mm = false, has_add = false;
    for (std::size_t j = 0; j < step.node_count; ++j) {
      const auto node = compiled.report.step_node(i, j).value();
      has_mm |= node == mm;
      has_add |= node == add;
    }
    EXPECT_FALSE(has_mm && has_add);
  }
}
