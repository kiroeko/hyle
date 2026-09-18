#include <gtest/gtest.h>

#include "../compute_route_vulkan/topk_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_topk_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph, TopKBothPortsHaveIndependentLivenessAndOldResultOwners) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  const auto desc = dense(c::DataType::kFloat32, {17, 288});
  const c::TopK selection{.count = 8};
  for (int published : {1, 2, 3}) {
    SCOPED_TRACE(published);
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(desc).value();
    const auto chosen =
        builder.add_node(c::Operation(selection), std::array{x}).value();
    const auto values = builder.result(chosen, 0).value();
    const auto indices = builder.result(chosen, 1).value();
    const auto index_copy = builder.add(c::Copy{}, indices).value();
    if (published & 1)
      ASSERT_TRUE(builder.publish(values).ok());
    if (published & 2)
      ASSERT_TRUE(builder.publish(index_copy).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto prepared = gv::compile(context, graph, {}, {.report = &report});
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(report.managed_region_count, 1U);
    graph = {};
    auto first = prepared->create_invocation().value();
    auto second = prepared->create_invocation().value();
    *prepared = {};
    std::vector<v::Tensor> retained;
    std::vector<std::vector<std::uint64_t>> previous;
    for (int phase = 0; phase < 3; ++phase) {
      const auto raw = pattern(*desc.element_count(), phase);
      const auto input = context.upload(desc, payload(desc, raw)).value();
      const auto expected = reference(desc, raw, selection);
      auto &slot = phase == 0 ? first : second;
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      const auto outputs = result->get()->tensors();
      ASSERT_EQ(outputs.size(), published == 3 ? 2U : 1U);
      if (published & 1)
        EXPECT_EQ(bits(context, outputs[0]), expected.values);
      if (published & 2)
        EXPECT_EQ(bits(context, outputs.back()), expected.indices);
      if (phase == 0) {
        retained.assign(outputs.begin(), outputs.end());
        for (const auto &tensor : retained)
          previous.push_back(bits(context, tensor));
      } else {
        for (std::size_t i = 0; i < retained.size(); ++i)
          EXPECT_EQ(bits(context, retained[i]), previous[i]);
      }
    }
    first = {};
    second = {};
    for (std::size_t i = 0; i < retained.size(); ++i)
      EXPECT_EQ(bits(context, retained[i]), previous[i]);
  }
  EXPECT_TRUE(context.close().ok());
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}
TEST(VulkanGraph, TopKComposesWithMaturePointwiseAndPublishesRawIndices) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {2, 288});
  const c::TopK selection{.count = 8, .largest = false};
  std::vector<std::uint64_t> raw(*desc.element_count()), absolute(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = encode(static_cast<int>(i % 43) - 21.0, c::DataType::kFloat32);
    absolute[i] = raw[i] & 0x7fffffffU;
  }
  const auto input = context.upload(desc, payload(desc, raw)).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(desc).value();
  const auto positive = builder.add(c::Unary{c::UnaryKind::kAbs}, x).value();
  const auto chosen =
      builder.add_node(c::Operation(selection), std::array{positive}).value();
  const auto values = builder.result(chosen, 0).value();
  const auto indices = builder.result(chosen, 1).value();
  const auto negative =
      builder.add(c::Unary{c::UnaryKind::kNeg}, values).value();
  ASSERT_TRUE(builder.publish(negative).ok());
  ASSERT_TRUE(builder.publish(indices).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_EQ(report.nodes.size(), 3U);
  auto result = execute_all(*program, input);
  ASSERT_TRUE(result) << result.error().message();
  auto expected = reference(desc, absolute, selection);
  for (auto &value : expected.values)
    value ^= 0x80000000U;
  expect_pair(context, *result, expected);
}
} // namespace
