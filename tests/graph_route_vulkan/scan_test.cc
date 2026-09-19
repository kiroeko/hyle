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
TEST(VulkanGraph, IntegerScanOffsetsFeedNativeGatherAndNormWithCheckedFailure) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kInt32, {17});
  const auto table_desc = dense(c::DataType::kFloat32, {34, 129});
  const auto table_data = pattern(*table_desc.element_count());
  const auto table = make(context, table_desc, table_data);
  const auto original = bytes(context, table);
  auto builder = g::GraphBuilder::create().value();
  const auto weights = builder.constant(table_desc).value();
  const auto counts = builder.input(desc).value();
  const auto offsets = builder.add(c::Scan{.exclusive = true}, counts).value();
  const auto selected = builder.add(c::Gather{0}, weights, offsets).value();
  const auto normalized =
      builder.add(c::RmsNorm{.epsilon = 1e-5f}, selected).value();
  ASSERT_TRUE(builder.publish(offsets).ok());
  ASSERT_TRUE(builder.publish(selected).ok());
  ASSERT_TRUE(builder.publish(normalized).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{table}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_GE(report.managed_region_count, 1U);
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  auto failed = program->create_invocation().value();
  auto fresh = program->create_invocation().value();
  graph = {};
  *program = {};
  std::vector<v::Tensor> retained;
  std::vector<std::vector<std::uint64_t>> saved;
  for (int phase = 0; phase < 5; ++phase) {
    std::vector<std::int32_t> data(17);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = (static_cast<int>(i) + phase) % 2;
    const auto input = make_integer(context, desc, data);
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto outputs = result->get()->tensors();
    expect_integer_prefix(context, outputs[0], desc, data,
                          c::Scan{.exclusive = true});
    std::vector<double> selected_data(17 * 129), normalized_data(17 * 129);
    std::int64_t prefix = 0;
    for (std::size_t row = 0; row < 17; ++row) {
      double squares = 0;
      for (std::size_t col = 0; col < 129; ++col) {
        const auto value =
            table_data[static_cast<std::size_t>(prefix) * 129 + col];
        selected_data[row * 129 + col] = value;
        squares += value * value;
      }
      const auto scale = 1 / std::sqrt(squares / 129 + 1e-5);
      for (std::size_t col = 0; col < 129; ++col)
        normalized_data[row * 129 + col] =
            selected_data[row * 129 + col] * scale;
      prefix += data[row];
    }
    expect(context, outputs[1], selected_data, true);
    expect(context, outputs[2], normalized_data);
    if (phase == 0) {
      retained.assign(outputs.begin(), outputs.end());
      for (const auto &output : retained)
        saved.push_back(bits(context, output));
      std::vector<std::int32_t> bad_data(17, 0);
      bad_data[0] = INT32_MAX;
      bad_data[1] = 1;
      const auto bad = make_integer(context, desc, bad_data);
      ASSERT_TRUE(failed.bind(bad).ok());
      auto bad_work = failed.submit().value();
      auto bad_result = bad_work.wait();
      ASSERT_FALSE(bad_result);
      EXPECT_TRUE(bad_result.error().IsOutOfRange());
      EXPECT_TRUE(failed.bind(input).IsOutOfRange());
      ASSERT_TRUE(fresh.bind(input).ok());
      auto recovered = fresh.submit().value();
      auto recovered_result = recovered.wait();
      ASSERT_TRUE(recovered_result) << recovered_result.error().message();
      expect(context, recovered_result->get()->tensors()[2], normalized_data);
    } else {
      for (std::size_t i = 0; i < retained.size(); ++i)
        EXPECT_EQ(bits(context, retained[i]), saved[i]);
    }
  }
  EXPECT_TRUE(context.close().ok());
  for (std::size_t i = 0; i < retained.size(); ++i)
    EXPECT_EQ(bits(context, retained[i]), saved[i]);
  EXPECT_EQ(bytes(context, table), original);
}
} // namespace
