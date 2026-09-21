#include <gtest/gtest.h>

#include "../compute_route_vulkan/gate_norm_test_support.h"
#include "../compute_route_vulkan/reduction_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_reduction_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     MatureHighRankL2RetainsPublishedReductionAndIndependentSlots) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  const auto description = dense(c::DataType::kFloat32, {2, 1, 3, 1, 2, 128});
  const auto scalar = dense(c::DataType::kFloat32, {});
  auto epsilon = make(context, scalar, std::array<double, 1>{1e-6F});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto eps = builder.constant(scalar).value();
  const auto copied = builder.add(c::Copy{}, x).value();
  const auto squared =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, copied, copied).value();
  const c::Reduce operation{.axes = c::Axes::from(std::array{-1}).value(),
                            .keep_dimensions = true,
                            .accumulation_type = c::DataType::kFloat32};
  const auto sum = builder.add(operation, squared).value();
  const auto shifted = builder.add(c::Binary{}, sum, eps).value();
  const auto inverse =
      builder.add(c::Unary{c::UnaryKind::kRsqrt}, shifted).value();
  const auto normalized =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, copied, inverse).value();
  ASSERT_TRUE(builder.publish(sum).ok());
  ASSERT_TRUE(builder.publish(normalized).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{epsilon}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_EQ(report.native_dispatch_count, 1U); // Input origin adapter only.
  EXPECT_FALSE(report.opaque_storage_bytes);
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  graph = {};
  epsilon = {};
  v::Tensor retained_sum, retained_output;
  std::vector<double> saved_sum, saved_output;
  for (int phase = 0; phase != 3; ++phase) {
    const auto data = pattern(*description.element_count(), phase);
    auto input = make(context, description, data);
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    std::vector<double> squares(data.size()), output(data.size());
    for (std::size_t i = 0; i != data.size(); ++i)
      squares[i] = data[i] * data[i];
    const auto sums = reduce_reference(squares, description, operation);
    for (std::size_t i = 0; i != data.size(); ++i)
      output[i] = data[i] / std::sqrt(sums[i / 128] + 1e-6F);
    expect(context, result->get()->tensors()[0], sums);
    expect(context, result->get()->tensors()[1], output);
    if (phase == 0) {
      retained_sum = result->get()->tensors()[0];
      retained_output = result->get()->tensors()[1];
      saved_sum = sums;
      saved_output = output;
    } else {
      expect(context, retained_sum, saved_sum);
      expect(context, retained_output, saved_output);
    }
  }
  EXPECT_FALSE(first.submit());
  first = {};
  second = {};
  *program = {};
  ASSERT_TRUE(context.close().ok());
  expect(context, retained_sum, saved_sum);
  expect(context, retained_output, saved_output);
  retained_sum = {};
  retained_output = {};
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanGraph,
     BFloatPublishedRoundingBoundariesRetainIndependentMixedProviderResults) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  256 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  const auto description = dense(c::DataType::kFloat32, {3, 1, 3, 1, 1, 129});
  const auto gamma_description = dense(c::DataType::kFloat32, {129});
  std::vector<double> gamma(129);
  for (std::size_t i = 0; i != gamma.size(); ++i)
    gamma[i] = static_cast<float>(0.739 + 0.007 * (i % 17));
  auto scale_tensor = make(context, gamma_description, gamma);
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto scale = builder.constant(gamma_description).value();
  const auto cast = builder.add(c::Cast{c::DataType::kBFloat16}, x).value();
  const auto normalized = builder.add(c::RmsNorm{}, cast, scale).value();
  const auto activated =
      builder.add(c::Unary{c::UnaryKind::kSilu}, normalized).value();
  const auto residual = builder.add(c::Binary{}, activated, cast).value();
  for (auto value : {cast, normalized, residual})
    ASSERT_TRUE(builder.publish(value).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, std::array{scale_tensor},
                             {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_GE(report.managed_region_count, 3U);
  EXPECT_FALSE(report.opaque_storage_bytes);
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  graph = {};
  scale_tensor = {};
  std::vector<v::Tensor> retained;
  std::vector<std::vector<double>> saved;
  for (int phase = 0; phase != 3; ++phase) {
    auto data = pattern(*description.element_count(), phase);
    for (auto &value : data)
      value = static_cast<float>(value + 0.000913 * (phase + 1));
    auto input = make(context, description, data);
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    for (auto &value : data)
      value = rounded(value, c::DataType::kBFloat16);
    auto norm = vulkan_gate_norm_test::norm(data, 129, false, 1e-5F, gamma);
    for (auto &value : norm)
      value = rounded(value, c::DataType::kBFloat16);
    auto output = norm;
    for (std::size_t i = 0; i != output.size(); ++i)
      output[i] =
          rounded(unary(c::UnaryKind::kSilu, norm[i]), c::DataType::kBFloat16) +
          data[i];
    const std::array expected{data, norm, output};
    for (std::size_t port = 0; port != expected.size(); ++port) {
      expect(context, result->get()->tensors()[port], expected[port],
             port == 0);
      if (phase == 0) {
        retained.push_back(result->get()->tensors()[port]);
        saved.push_back(expected[port]);
      } else
        expect(context, retained[port], saved[port], port == 0);
    }
  }
  EXPECT_FALSE(first.submit());
  first = {};
  second = {};
  *program = {};
  ASSERT_TRUE(context.close().ok());
  for (std::size_t port = 0; port != retained.size(); ++port)
    expect(context, retained[port], saved[port], port == 0);
  retained.clear();
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}
} // namespace
