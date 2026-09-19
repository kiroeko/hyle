#include <gtest/gtest.h>

#include "../compute_route_vulkan/scan_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_scan_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph, ScanUsesNativeRegionAndPreservesIndependentReplays) {
  auto context = vulkan_test::context().value();
  for (const std::int64_t rows : {17, 513}) {
    const auto desc = dense(c::DataType::kFloat32, {rows, 129});
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(desc).value();
    const auto positive =
        builder.add(c::Unary{c::UnaryKind::kAbs}, input).value();
    const auto inclusive = builder.add(c::Scan{}, positive).value();
    const auto exclusive =
        builder.add(c::Scan{.exclusive = true}, positive).value();
    const auto consumed =
        builder.add(c::Unary{c::UnaryKind::kNeg}, inclusive).value();
    ASSERT_TRUE(builder.publish(inclusive).ok());
    ASSERT_TRUE(builder.publish(exclusive).ok());
    ASSERT_TRUE(builder.publish(consumed).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program = gv::compile(context, graph, {}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_GE(report.managed_region_count, 1U);
    auto first = program->create_invocation().value();
    auto second = program->create_invocation().value();
    graph = {};
    *program = {};
    std::vector<v::Tensor> retained;
    std::vector<std::vector<std::uint64_t>> saved;
    for (int phase = 0; phase < 5; ++phase) {
      auto data = pattern(*desc.element_count(), phase);
      const auto tensor = make(context, desc, data);
      for (auto &value : data)
        value = std::abs(value);
      auto &slot = phase == 0 ? first : second;
      ASSERT_TRUE(slot.bind(tensor).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      const auto outputs = result->get()->tensors();
      expect_prefix(context, outputs[0], desc, data, c::Scan{});
      expect_prefix(context, outputs[1], desc, data,
                    c::Scan{.exclusive = true});
      auto negative = data;
      for (auto &value : negative)
        value = -value;
      expect_prefix(context, outputs[2], desc, negative, c::Scan{});
      if (phase == 0) {
        retained.assign(outputs.begin(), outputs.end());
        for (const auto &output : retained)
          saved.push_back(bits(context, output));
      } else {
        for (std::size_t i = 0; i < retained.size(); ++i)
          EXPECT_EQ(bits(context, retained[i]), saved[i]);
      }
    }
  }
  EXPECT_TRUE(context.close().ok());
}
} // namespace
