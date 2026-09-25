#include "hyle/graph/vulkan.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_route_vulkan/recurrent_test_support.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"

namespace {
using namespace vulkan_recurrent_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

class VulkanGraphQwen38GdnProfiles
    : public ::testing::TestWithParam<
          const c::ModelArtifactProfileDescription *> {};

TEST_P(VulkanGraphQwen38GdnProfiles,
       D5Qwen38GroupedHeadGdnUsesOrdinaryGraphAndReplays) {
  const d5_qwen38_gdn_fixture::Fixture fixture;
  auto context = vulkan_test::context().value();
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 6> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port)
    inputs[port] = builder.input(fixture.inputs[port]).value();
  const auto node = builder.add_node(c::Operation(c::GatedDeltaNet{}), inputs)
                        .value();
  const std::array outputs{builder.result(node, 0).value(),
                           builder.result(node, 1).value()};
  ASSERT_TRUE(builder.publish(outputs[0]).ok());
  ASSERT_TRUE(builder.publish(outputs[1]).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  const auto workload = d5_qwen38_gdn_fixture::workload(
      *GetParam(), c::WorkloadPhase::kDecode);
  auto program = gv::compile(context, graph, {},
                             {.report = &report, .workload = workload});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->workload(), workload);
  EXPECT_EQ(report.managed_region_count, 1U);
  std::array<c::vulkan::Tensor, 6> resident;
  for (std::size_t port = 0; port < resident.size(); ++port)
    resident[port] =
        context.upload(fixture.inputs[port],
                       std::as_bytes(std::span(fixture.data[port])))
            .value();
  auto invocation = program->create_invocation().value();
  ASSERT_TRUE(invocation.bind(resident).ok());
  const auto expected = fixture.oracle();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto values = result->get()->tensors();
    ASSERT_EQ(values.size(), expected.size());
    for (std::size_t port = 0; port < values.size(); ++port)
      expect(context, values[port], expected[port]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, VulkanGraphQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kOfficialProfile,
                      &d5_qwen38_gdn_fixture::kPrimaryGgufProfile));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, VulkanGraphQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kFidelityGgufProfile));

TEST(VulkanGraph, MatureMoeAndHyperConnectionBatchUsesOrdinaryGraphEntry) {
  auto context = vulkan_test::context().value();
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    auto builder = g::GraphBuilder::create().value();
    std::vector<g::ValueId> inputs;
    inputs.reserve(test.inputs.size());
    for (const auto &description : test.inputs)
      inputs.push_back(builder.input(description).value());
    const auto node = builder.add_node(test.operation, inputs).value();
    const auto output = builder.result(node).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program = gv::compile(context, graph, {}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.managed_region_count, 1U);
    auto invocation = program->create_invocation().value();
    std::vector<c::vulkan::Tensor> resident;
    resident.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i)
      resident.push_back(
          context.upload(test.inputs[i], test.payloads[i]).value());
    ASSERT_TRUE(invocation.bind(resident).ok());
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto actual =
        vulkan_test::read(context, result->get()->tensors()[0]).value();
    ASSERT_EQ(actual.size(), test.expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(VulkanGraph, MatureLightningIndexerUsesOrdinaryGraphEntry) {
  const sparse_fragment_test::Fixture test;
  auto context = vulkan_test::context().value();
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = builder.input(test.inputs[i]).value();
  const auto node = builder.add_node(test.operation, inputs).value();
  const auto output = builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report}).value();
  EXPECT_EQ(report.managed_region_count, 1U);
  auto invocation = program.create_invocation().value();
  std::vector<c::vulkan::Tensor> resident;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    resident.push_back(context.upload(test.inputs[i], test.payloads[i]).value());
  ASSERT_TRUE(invocation.bind(resident).ok());
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto work = invocation.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto actual =
        vulkan_test::read(context, result->get()->tensors()[0]).value();
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(VulkanGraph, OriginalRecurrentRegionsKeepStateEdgesAndPublishedForks) {
  Fixture fixture(5, 2, 32);
  auto context = vulkan_test::context().value();
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 6> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = builder.input(fixture.descriptions[i]).value();
  const auto query = builder.add(c::Binary{}, inputs[0], inputs[0]).value();
  auto ports = inputs;
  ports[0] = query;
  const auto first = builder.add_node(c::Operation(c::GatedDeltaNet{}), ports).value();
  const auto y1 = builder.result(first, 0).value();
  const auto s1 = builder.result(first, 1).value();
  ports[2] = builder.add(c::Binary{}, y1, inputs[2]).value();
  ports[5] = s1;
  const auto second = builder.add_node(c::Operation(c::GatedDeltaNet{}), ports).value();
  const auto y2 = builder.result(second, 0).value();
  const auto s2 = builder.result(second, 1).value();
  const auto residual = builder.add(c::Binary{}, y2, inputs[2]).value();
  ASSERT_TRUE(builder.publish(residual).ok());
  ASSERT_TRUE(builder.publish(s2).ok());
  ASSERT_TRUE(builder.publish(y1).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_NE(report.numerical_policy.find("recurrent F32"), std::string::npos);
  graph = {};
  auto invocation = program->create_invocation().value();
  const auto original = fixture.oracle(2);
  auto next = fixture;
  for (std::size_t i = 0; i < next.data[2].size(); ++i)
    next.data[2][i] = float(original[0][i]) + fixture.data[2][i];
  for (std::size_t i = 0; i < next.data[5].size(); ++i)
    next.data[5][i] = float(original[1][i]);
  auto expected = next.oracle(2);
  for (std::size_t i = 0; i < expected[0].size(); ++i)
    expected[0][i] = float(expected[0][i]) + fixture.data[2][i];
  auto resident = fixture.upload(context);
  ASSERT_TRUE(invocation.bind(resident).ok());
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto work = invocation.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto values = result->get()->tensors();
    ASSERT_EQ(values.size(), 3U);
    expect(context, values[0], expected[0]);
    expect(context, values[1], expected[1]);
    expect(context, values[2], original[0]);
  }
  EXPECT_EQ(vulkan_test::read(context, resident[5]).value(), fixture.data[5]);
}
} // namespace
