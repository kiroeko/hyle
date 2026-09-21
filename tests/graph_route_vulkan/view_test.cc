#include <gtest/gtest.h>

#include "../compute_route_vulkan/tensor_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_tensor_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     PublishedViewsKeepTheirComputedRootsLiveAcrossLaterDispatches) {
  auto context = vulkan_test::context().value();
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto desc = dense(type, {17, 129});
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(desc).value();
    const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, x).value();
    const auto view =
        builder.add(c::View::slice(1, 1, 127).value(), negative).value();
    const auto exp = builder.add(c::Unary{c::UnaryKind::kExp}, x).value();
    const auto end = builder.add(c::Unary{c::UnaryKind::kNeg}, exp).value();
    ASSERT_TRUE(builder.publish(view).ok());
    ASSERT_TRUE(builder.publish(end).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program = gv::compile(context, graph, {}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    // The view splits F16 into two regions; BF16 retains one ggml/IREE pair
    // per public unary rounding point. Both paths adapt the input twice.
    EXPECT_EQ(report.native_dispatch_count, 2U);
    EXPECT_EQ(report.managed_region_count,
              type == c::DataType::kFloat16 ? 2U : 6U);
    EXPECT_EQ(report.invocation_storage_block_bytes.size(), 3U);
    graph = {};
    std::vector<double> data(17 * 129);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = (static_cast<int>(i % 31) - 15) * 0.0625;
    const auto source = make(context, desc, data);
    auto first = program->create_invocation().value(),
         second = program->create_invocation().value();
    v::Tensor retained;
    std::shared_ptr<const void> token;
    {
      ASSERT_TRUE(first.bind(source).ok());
      auto work = first.submit().value();
      auto result = work.wait().value();
      retained = result.get()->tensors()[0];
      token = retained.native_owner();
      EXPECT_EQ(retained.native_offset(), 2U);
      std::vector<double> expected;
      for (int row = 0; row < 17; ++row)
        for (int column = 1; column < 128; ++column)
          expected.push_back(-data[row * 129 + column]);
      expect(context, retained, expected, true);
      for (auto &value : data)
        value = rounded(-std::exp(value), type);
      expect(context, result.get()->tensors()[1], data);
    }
    const auto saved = bits(context, retained);
    EXPECT_FALSE(first.bind(source).ok());
    ASSERT_TRUE(second.bind(source).ok());
    {
      auto work = second.submit().value();
      ASSERT_TRUE(work.wait());
    }
    EXPECT_EQ(bits(context, retained), saved);
    retained = {};
    EXPECT_FALSE(first.bind(source).ok());
    token.reset();
    ASSERT_TRUE(first.bind(source).ok());
    {
      auto work = first.submit().value();
      ASSERT_TRUE(work.wait());
    }
  }
}

TEST(VulkanGraph, InputAliasChainsRebindOddOriginsAndKeepOldInputStorage) {
  auto context = vulkan_test::context().value();
  const auto parent_desc =
      c::TensorDesc::strided_bytes(c::DataType::kFloat16,
                                   std::array<std::int64_t, 2>{3, 5},
                                   std::array<std::int64_t, 2>{15, 2})
          .value();
  const auto input_desc =
      c::TensorDesc::strided_bytes(c::DataType::kFloat16,
                                   std::array<std::int64_t, 2>{2, 5},
                                   std::array<std::int64_t, 2>{15, 2})
          .value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(input_desc).value();
  const auto sliced =
      builder.add(c::View::slice(1, 1, 2, 2).value(), x).value();
  const auto transposed =
      builder
          .add(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(),
               sliced)
          .value();
  const auto copied = builder.add(c::Copy{}, transposed).value();
  const auto negative =
      builder.add(c::Unary{c::UnaryKind::kNeg}, copied).value();
  ASSERT_TRUE(builder.publish(transposed).ok());
  ASSERT_TRUE(builder.publish(negative).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report}).value();
  // Copy materializes the strided view; F16 casts stay inside the Neg region.
  EXPECT_EQ(report.native_dispatch_count, 1U);
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_EQ(report.invocation_storage_block_bytes.size(), 2U);
  auto first = program.create_invocation().value(),
       second = program.create_invocation().value();
  graph = {};
  v::Tensor retained;
  std::vector<std::uint64_t> saved;
  for (int phase = 0; phase < 3; ++phase) {
    std::vector<double> data(15);
    for (int i = 0; i < 15; ++i)
      data[i] = i + phase * 16;
    auto parent = make(context, parent_desc, data);
    const int start = phase % 2;
    auto input = parent.view(c::View::slice(0, start, 2).value()).value();
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait().value();
    const auto &view = result.get()->tensors()[0];
    EXPECT_EQ(view.native_buffer(), parent.native_buffer());
    EXPECT_EQ(view.native_offset(), static_cast<std::uint64_t>(15 * start + 2));
    std::array<double, 4> expected{
        data[5 * start + 1], data[5 * (start + 1) + 1], data[5 * start + 3],
        data[5 * (start + 1) + 3]};
    expect(context, view, expected, true);
    for (auto &value : expected)
      value = -value;
    expect(context, result.get()->tensors()[1], expected, true);
    if (phase == 0) {
      retained = view;
      saved = bits(context, retained);
    } else
      EXPECT_EQ(bits(context, retained), saved);
  }
  first = {};
  second = {};
  program = {};
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(bits(context, retained), saved);
}

TEST(VulkanGraph, ConstantViewsShareColdWeightPackingAndDropUnusedRoots) {
  auto context = vulkan_test::context().value();
  constexpr int rows = 3, k = 17, n = 33;
  const auto host_weights = vulkan_test::data(k * n, 2);
  for (bool publish_view : {false, true}) {
    auto weights =
        vulkan_test::upload(context, vulkan_test::dense({k * n}), host_weights)
            .value();
    const std::weak_ptr<const void> original_owner = weights.native_owner();
    const auto original_buffer = weights.native_buffer();
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(vulkan_test::dense({rows, k})).value();
    const auto w = builder.constant(weights.description()).value();
    const auto matrix =
        builder
            .add(c::View::reshape(
                     c::TensorShape::create(std::array<std::int64_t, 2>{k, n})
                         .value())
                     .value(),
                 w)
            .value();
    const auto same_matrix =
        builder.add(c::View::slice(0, 0, k).value(), matrix).value();
    const auto first = builder.add(c::MatMul{}, x, matrix).value();
    const auto second = builder.add(c::MatMul{}, x, same_matrix).value();
    ASSERT_TRUE(builder.publish(first).ok());
    ASSERT_TRUE(builder.publish(second).ok());
    if (publish_view)
      ASSERT_TRUE(builder.publish(same_matrix).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program =
        gv::compile(context, graph, std::array{weights}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.native_dispatch_count, 2U); // Only the two matmuls replay.
    EXPECT_EQ(report.invocation_storage_block_bytes.size(), 2U);
    EXPECT_EQ(report.retained_constant_owner_count, publish_view ? 2U : 1U);
    graph = {};
    weights = {};
    EXPECT_EQ(original_owner.expired(), !publish_view);
    auto slot = program->create_invocation().value();
    for (int phase = 0; phase < 2; ++phase) {
      const auto host_input = vulkan_test::data(rows * k, phase + 3);
      const auto input = vulkan_test::upload(
                             context, vulkan_test::dense({rows, k}), host_input)
                             .value();
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      const auto expected =
          vulkan_test::matmul(host_input, host_weights, rows, k, n);
      for (std::size_t output = 0; output < 2; ++output) {
        const auto actual =
            vulkan_test::read(context, result.get()->tensors()[output]).value();
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_NEAR(actual[i], expected[i],
                      8e-5f + 3e-5f * std::abs(expected[i]));
      }
      if (publish_view) {
        const auto &view = result.get()->tensors()[2];
        EXPECT_EQ(view.native_buffer(), original_buffer);
        EXPECT_EQ(view.native_offset(), 0U);
        EXPECT_EQ(vulkan_test::read(context, view).value(), host_weights);
      }
    }
  }
}

TEST(VulkanGraph, ConstantViewsNeedNoDeviceStorageAndRetainOriginalOwners) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kInt64, {17});
  std::vector<std::uint64_t> raw(17);
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = 0xffff123456780000ULL + i;
  auto constant = context.upload(desc, payload(desc, raw)).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.constant(desc).value();
  const auto view = builder.add(c::View::slice(0, 1, 7, 2).value(), x).value();
  ASSERT_TRUE(builder.publish(view).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{constant}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.native_dispatch_count, 0U);
  EXPECT_TRUE(report.invocation_storage_block_bytes.empty());
  EXPECT_EQ(report.retained_constant_owner_count, 1U);
  auto slot = program->create_invocation().value();
  const auto handle = constant.native_buffer();
  graph = {};
  constant = {};
  auto work = slot.submit().value();
  auto result = work.wait().value();
  auto retained = result.get()->tensors()[0];
  EXPECT_EQ(retained.native_buffer(), handle);
  EXPECT_EQ(retained.native_offset(), 8U);
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(bits(context, retained),
            (std::vector<std::uint64_t>{raw[1], raw[3], raw[5], raw[7], raw[9],
                                        raw[11], raw[13]}));
}
} // namespace
