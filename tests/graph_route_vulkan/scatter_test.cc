#include <gtest/gtest.h>

#include "../compute_route_vulkan/scan_test_support.h"
#include "../compute_route_vulkan/scatter_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_scatter_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     ScanScatterGatherNormKeepsOrderedUpdatesAndIndependentResults) {
  auto context = vulkan_test::context().value();
  const auto counts_desc = dense(c::DataType::kInt32, {17});
  const auto table_desc = dense(c::DataType::kBFloat16, {34, 129});
  const auto update_desc = dense(c::DataType::kBFloat16, {17, 129});
  const auto original = numeric(table_desc);
  auto table =
      context.upload(table_desc, payload(table_desc, original)).value();
  auto builder = g::GraphBuilder::create().value();
  const auto base = builder.constant(table_desc).value();
  const auto counts = builder.input(counts_desc).value();
  const auto update = builder.input(update_desc).value();
  const auto offsets = builder.add(c::Scan{.exclusive = true}, counts).value();
  const auto assigned =
      builder.add(c::Scatter{0}, base, offsets, update).value();
  const auto added =
      builder
          .add(c::Scatter{0, c::ScatterKind::kAdd}, assigned, offsets, update)
          .value();
  const auto selected = builder.add(c::Gather{0}, added, offsets).value();
  const auto floats =
      builder.add(c::Cast{c::DataType::kFloat32}, selected).value();
  const auto normalized = builder.add(c::RmsNorm{}, floats).value();
  for (const auto output : {offsets, assigned, added, selected, normalized})
    ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{table}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_GE(report.managed_region_count, 3U);
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  auto failed = program->create_invocation().value();
  auto fresh = program->create_invocation().value();
  graph = {};
  *program = {};
  std::vector<v::Tensor> retained;
  std::vector<std::vector<std::uint64_t>> saved;
  for (int phase = 0; phase < 5; ++phase) {
    std::vector<std::int32_t> count_data(17);
    std::vector<std::uint64_t> positions(17);
    std::uint64_t prefix = 0;
    for (std::size_t i = 0; i < count_data.size(); ++i) {
      count_data[i] = (static_cast<int>(i) + phase) % 2;
      positions[i] = prefix;
      prefix += static_cast<std::uint64_t>(count_data[i]);
    }
    const auto updates = numeric(update_desc, phase);
    const auto count_tensor =
        vulkan_scan_test::make_integer(context, counts_desc, count_data);
    const auto update_tensor =
        context.upload(update_desc, payload(update_desc, updates)).value();
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(std::array{count_tensor, update_tensor}).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto outputs = result->get()->tensors();
    EXPECT_EQ(bits(context, outputs[0]), positions);
    const auto assigned_data =
        reference(table_desc, original, positions, updates, c::Scatter{0});
    const auto added_data =
        reference(table_desc, assigned_data, positions, updates,
                  c::Scatter{0, c::ScatterKind::kAdd});
    expect_scatter(context, outputs[1], assigned_data, false);
    expect_scatter(context, outputs[2], added_data, true);
    const auto gathered = vulkan_gather_test::reference(
        table_desc, payload(table_desc, added_data), positions, 0);
    EXPECT_EQ(bits(context, outputs[3]), gathered);
    std::vector<double> normalized_data(gathered.size());
    for (std::size_t row = 0; row < 17; ++row) {
      double squares = 0;
      for (std::size_t col = 0; col < 129; ++col) {
        const auto x =
            decode(gathered[row * 129 + col], c::DataType::kBFloat16);
        squares += x * x;
      }
      const auto scale = 1 / std::sqrt(squares / 129 + 1e-5);
      for (std::size_t col = 0; col < 129; ++col)
        normalized_data[row * 129 + col] =
            decode(gathered[row * 129 + col], c::DataType::kBFloat16) * scale;
    }
    expect(context, outputs[4], normalized_data);
    if (phase == 0) {
      retained.assign(outputs.begin(), outputs.end());
      for (const auto &output : retained)
        saved.push_back(bits(context, output));
      count_data.assign(17, 0);
      count_data[0] =
          -1; // Scan itself is valid; Scatter rejects negative offsets.
      const auto bad =
          vulkan_scan_test::make_integer(context, counts_desc, count_data);
      ASSERT_TRUE(failed.bind(std::array{bad, update_tensor}).ok());
      auto bad_work = failed.submit().value();
      auto bad_result = bad_work.wait();
      ASSERT_FALSE(bad_result);
      EXPECT_TRUE(bad_result.error().IsOutOfRange());
      EXPECT_TRUE(
          failed.bind(std::array{count_tensor, update_tensor}).IsOutOfRange());
      ASSERT_TRUE(fresh.bind(std::array{count_tensor, update_tensor}).ok());
      auto recovered = fresh.submit().value();
      auto good_result = recovered.wait();
      ASSERT_TRUE(good_result) << good_result.error().message();
      expect(context, good_result->get()->tensors()[4], normalized_data);
    } else {
      for (std::size_t i = 0; i < retained.size(); ++i)
        EXPECT_EQ(bits(context, retained[i]), saved[i]);
    }
  }
  EXPECT_TRUE(context.close().ok());
  for (std::size_t i = 0; i < retained.size(); ++i)
    EXPECT_EQ(bits(context, retained[i]), saved[i]);
  EXPECT_EQ(bits(context, table), original);
}
} // namespace
