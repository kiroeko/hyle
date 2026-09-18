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
    EXPECT_EQ(report.native_dispatch_count, 3U);
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
  EXPECT_EQ(report.native_dispatch_count, 2U);
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
