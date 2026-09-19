#include <gtest/gtest.h>

#include <algorithm>

#include "../compute_route_vulkan/gather_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_gather_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph, GatherSharesCapturedTableAndPreservesIndependentResults) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kBFloat16, {7, 129});
  const auto index_desc = dense(c::DataType::kInt64, {2});
  std::vector<double> data(*desc.element_count());
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = (static_cast<int>(i % 37) - 18) * 0.125;
  std::vector<std::uint64_t> raw;
  for (double value : data)
    raw.push_back(encode(value, desc.data_type()));
  const auto host = payload(desc, raw);
  const auto table = context.upload(desc, host).value();
  auto builder = g::GraphBuilder::create().value();
  const auto weights = builder.constant(desc).value();
  const auto index = builder.input(index_desc).value();
  const auto selected = builder.add(c::Gather{0}, weights, index).value();
  const auto floats =
      builder.add(c::Cast{c::DataType::kFloat32}, selected).value();
  const auto norm =
      builder
          .add(c::RmsNorm{.epsilon = 1e-5f,
                          .accumulation_type = c::DataType::kFloat32},
               floats)
          .value();
  ASSERT_TRUE(builder.publish(selected).ok());
  ASSERT_TRUE(builder.publish(norm).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{table}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 2U);
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  auto failed = program->create_invocation().value();
  graph = {};
  *program = {};
  std::vector<v::Tensor> retained;
  std::vector<std::vector<std::uint64_t>> saved;
  for (int phase = 0; phase < 3; ++phase) {
    const std::vector<std::uint64_t> positions{
        static_cast<std::uint64_t>(6 - phase), 1};
    const auto indices =
        context.upload(index_desc, payload(index_desc, positions)).value();
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(indices).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto outputs = result->get()->tensors();
    const auto expected = reference(desc, host, positions, 0);
    EXPECT_EQ(bits(context, outputs[0]), expected);
    std::vector<double> normalized(expected.size());
    for (std::size_t row = 0; row < 2; ++row) {
      double squares = 0;
      for (std::size_t col = 0; col < 129; ++col) {
        const auto x = decode(expected[row * 129 + col], desc.data_type());
        squares += x * x;
      }
      const auto scale = 1 / std::sqrt(squares / 129 + 1e-5);
      for (std::size_t col = 0; col < 129; ++col)
        normalized[row * 129 + col] =
            decode(expected[row * 129 + col], desc.data_type()) * scale;
    }
    expect(context, outputs[1], normalized);
    if (phase == 0) {
      retained.assign(outputs.begin(), outputs.end());
      for (const auto &output : retained)
        saved.push_back(bits(context, output));
      const auto bad =
          context
              .upload(index_desc,
                      payload(index_desc, std::array<std::uint64_t, 2>{0, 7}))
              .value();
      ASSERT_TRUE(failed.bind(bad).ok());
      auto bad_work = failed.submit().value();
      auto bad_result = bad_work.wait();
      ASSERT_FALSE(bad_result);
      EXPECT_TRUE(bad_result.error().IsOutOfRange());
    } else {
      for (std::size_t i = 0; i < retained.size(); ++i)
        EXPECT_EQ(bits(context, retained[i]), saved[i]);
    }
  }
  EXPECT_EQ(bytes(context, table), host);
  EXPECT_TRUE(context.close().ok());
}
} // namespace
