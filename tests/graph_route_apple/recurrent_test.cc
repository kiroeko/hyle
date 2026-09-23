#include "hyle/graph/apple.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>

#include "../compute_route_apple/recurrent_fixture.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace ga = g::apple;
namespace mx = mlx::core;

TEST(AppleGraph, MatureMoeAndHyperConnectionBatchUsesOrdinaryGraphEntry) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    auto builder = g::GraphBuilder::create().value();
    std::vector<g::ValueId> ports;
    ports.reserve(test.inputs.size());
    for (const auto &description : test.inputs)
      ports.push_back(builder.input(description).value());
    const auto node = builder.add_node(test.operation, ports).value();
    const auto output = builder.result(node).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    ga::CompileReport report;
    auto program = ga::compile(*context, graph, {}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_NE(report.implementation_policy.find("ggml Metal"),
              std::string::npos);
    auto invocation = program->create_invocation().value();
    std::vector<mx::array> inputs;
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i) {
      mx::Shape shape;
      for (const auto extent : test.inputs[i].dimensions())
        shape.push_back(static_cast<int>(extent));
      if (test.inputs[i].data_type() == c::DataType::kInt32)
        inputs.emplace_back(
            reinterpret_cast<const std::int32_t *>(test.payloads[i].data()),
            shape);
      else
        inputs.emplace_back(
            reinterpret_cast<const float *>(test.payloads[i].data()), shape);
    }
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto arrays = result->get()->arrays();
    ASSERT_EQ(arrays.size(), 1U);
    ASSERT_EQ(arrays[0].size(), test.expected.size());
    for (std::size_t i = 0; i < test.expected.size(); ++i)
      EXPECT_NEAR(arrays[0].data<float>()[i], test.expected[i], 2.0e-5F)
          << i;
  }
}

TEST(AppleGraph, MatureLightningIndexerUsesOrdinaryGraphEntry) {
  const sparse_fragment_test::Fixture test;
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> ports;
  for (std::size_t i = 0; i < ports.size(); ++i)
    ports[i] = builder.input(test.inputs[i]).value();
  const auto node = builder.add_node(test.operation, ports).value();
  const auto output = builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  ga::CompileReport report;
  auto program = ga::compile(*context, graph, {}, {.report = &report}).value();
  EXPECT_NE(report.implementation_policy.find("ggml Metal"),
            std::string::npos);
  auto invocation = program.create_invocation().value();
  std::vector<mx::array> inputs;
  for (std::size_t i = 0; i < test.inputs.size(); ++i) {
    mx::Shape shape;
    for (const auto extent : test.inputs[i].dimensions())
      shape.push_back(static_cast<int>(extent));
    if (test.inputs[i].data_type() == c::DataType::kFloat16) {
      inputs.emplace_back(
          std::initializer_list<float>{-1, 0, 0.5F, -2, 0.25F, -0.5F, 1, 0},
          shape, mx::float16);
    } else {
      inputs.emplace_back(
          reinterpret_cast<const float *>(test.payloads[i].data()), shape);
    }
  }
  for (int repeat = 0; repeat < 2; ++repeat) {
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto work = invocation.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto arrays = result->get()->arrays();
    for (std::size_t i = 0; i < test.expected.size(); ++i)
      EXPECT_NEAR(arrays[0].data<float>()[i], test.expected[i], 2.0e-5F)
          << i;
  }
}

TEST(AppleGraph, OriginalRecurrentRegionsComposeWithMlxAndExplicitState) {
  apple_test::Recurrent fixture(5, 2, 32);
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 6> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::vector<std::int64_t> shape(fixture.inputs[i].shape().begin(), fixture.inputs[i].shape().end());
    inputs[i] = builder.input(c::TensorDesc::dense(c::DataType::kFloat32, shape).value()).value();
  }
  const auto query = builder.add(c::Binary{}, inputs[0], inputs[0]).value();
  auto ports = inputs;
  ports[0] = query;
  const auto first = builder.add_node(c::Operation(c::GatedDeltaNet{}), ports).value();
  const auto y1 = builder.result(first, 0).value();
  const auto s1 = builder.result(first, 1).value();
  const auto value = builder.add(c::Binary{}, y1, inputs[2]).value();
  ports[2] = value;
  ports[5] = s1;
  const auto second = builder.add_node(c::Operation(c::GatedDeltaNet{}), ports).value();
  const auto y2 = builder.result(second, 0).value();
  const auto s2 = builder.result(second, 1).value();
  const auto residual = builder.add(c::Binary{}, y2, inputs[2]).value();
  ASSERT_TRUE(builder.publish(residual).ok());
  ASSERT_TRUE(builder.publish(s2).ok());
  ASSERT_TRUE(builder.publish(y1).ok());
  auto graph = std::move(builder).finish().value();
  ga::CompileReport report;
  auto program = ga::compile(*context, graph, {}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_NE(report.implementation_policy.find("ggml Metal"), std::string::npos);
  EXPECT_FALSE(report.environment.ggml_revision.empty());
  graph = g::Graph{};
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();

  const auto original = fixture.oracle(2);
  auto next = fixture;
  for (std::size_t i = 0; i < next.data[2].size(); ++i)
    next.data[2][i] = float(original[0][i]) + fixture.data[2][i];
  for (std::size_t i = 0; i < next.data[5].size(); ++i)
    next.data[5][i] = float(original[1][i]);
  auto expected = next.oracle(2);
  for (std::size_t i = 0; i < expected[0].size(); ++i)
    expected[0][i] = float(expected[0][i]) + fixture.data[2][i];
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_TRUE(invocation->bind(fixture.inputs).ok());
    auto work = invocation->submit();
    ASSERT_TRUE(work);
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto values = result->get()->arrays();
    ASSERT_EQ(values.size(), 3u);
    for (std::size_t port = 0; port < values.size(); ++port) {
      const auto &oracle = port == 2 ? original[0] : expected[port];
      ASSERT_EQ(values[port].size(), oracle.size());
      double max_error = 0;
      for (std::size_t i = 0; i < oracle.size(); ++i)
        max_error = std::max(max_error, std::abs(double(values[port].data<float>()[i]) - oracle[i]));
      EXPECT_LT(max_error, 3e-6);
    }
  }
}
} // namespace
