#include <gtest/gtest.h>

#include "../compute_route_vulkan/gate_norm_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_gate_norm_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph, ManagedNormFusionPreservesPublishedForksAndNativeBoundaries) {
  auto context = vulkan_test::context().value();
  constexpr int width = 2048;
  v::Tensor surviving;
  std::vector<double> surviving_values;
  for (int rows : {1, 17, 128}) {
    for (bool native_prefix : {false, true}) {
      SCOPED_TRACE(::testing::Message() << rows << ':' << native_prefix);
      const auto desc = dense(c::DataType::kFloat32, {rows, width});
      const auto parameter = dense(c::DataType::kFloat32, {width});
      auto builder = g::GraphBuilder::create().value();
      const auto x = builder.input(desc).value();
      const auto skip = builder.input(desc).value();
      const auto scale = builder.constant(parameter).value();
      const auto input =
          native_prefix ? builder.add(c::Unary{c::UnaryKind::kRelu}, x).value()
                        : x;
      const auto rms = builder.add(c::RmsNorm{}, input).value();
      const auto scaled =
          builder.add(c::Binary{c::BinaryKind::kMultiply}, rms, scale).value();
      const auto activated =
          builder.add(c::Unary{c::UnaryKind::kSilu}, scaled).value();
      const auto added =
          builder.add(c::Binary{c::BinaryKind::kAdd}, activated, skip).value();
      const auto fork =
          builder.add(c::Binary{c::BinaryKind::kMultiply}, rms, rms).value();
      const auto output =
          builder.add(c::Unary{c::UnaryKind::kRelu}, added).value();
      ASSERT_TRUE(builder.publish(rms).ok());
      ASSERT_TRUE(builder.publish(activated).ok());
      ASSERT_TRUE(builder.publish(fork).ok());
      ASSERT_TRUE(builder.publish(output).ok());
      auto graph = std::move(builder).finish().value();
      std::vector<double> weight(width);
      for (int i = 0; i < width; ++i)
        weight[i] = 1 + (i % 7 - 3) * 0.0625;
      std::array constants{make(context, parameter, weight)};
      gv::CompileReport report;
      auto program =
          gv::compile(context, graph, constants, {.report = &report});
      ASSERT_TRUE(program) << program.error().message();
      EXPECT_EQ(report.managed_region_count, 1U);
      EXPECT_EQ(report.native_dispatch_count,
                3U); // Native or copy prefix, skip copy, suffix.
      EXPECT_FALSE(report.opaque_storage_bytes);
      graph = {};
      constants = {};
      auto first = program->create_invocation().value();
      auto second = program->create_invocation().value();
      v::Tensor retained;
      std::vector<double> saved;
      for (int phase = 0; phase < 3; ++phase) {
        std::vector<double> data(static_cast<std::size_t>(rows) * width),
            bypass(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
          data[i] = (static_cast<int>((i + phase * 7) % 61) - 30) * 0.125;
          bypass[i] = (static_cast<int>((i + phase) % 17) - 8) * 0.0625;
        }
        const auto activation = make(context, desc, data);
        const auto residual = make(context, desc, bypass);
        if (native_prefix)
          for (auto &value : data)
            value = (std::max)(value, 0.0);
        const auto normalized = norm(data, width, false, 1e-5F);
        std::vector<double> silu(data.size()), squared(data.size()),
            expected(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
          silu[i] = activate(c::GatedActivationKind::kSilu,
                             normalized[i] * weight[i % width]);
          squared[i] = normalized[i] * normalized[i];
          expected[i] = (std::max)(silu[i] + bypass[i], 0.0);
        }
        auto &slot = phase == 0 ? first : second;
        ASSERT_TRUE(slot.bind(activation, residual).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        expect(context, result->get()->tensors()[0], normalized);
        expect(context, result->get()->tensors()[1], silu);
        expect(context, result->get()->tensors()[2], squared);
        expect(context, result->get()->tensors()[3], expected);
        if (phase == 0) {
          retained = result->get()->tensors()[3];
          saved = expected;
        } else {
          expect(context, retained, saved);
        }
      }
      first = {};
      second = {};
      *program = {};
      expect(context, retained, saved);
      surviving = retained;
      surviving_values = std::move(saved);
    }
  }
  EXPECT_TRUE(context.close().ok());
  expect(context, surviving, surviving_values);
}
TEST(VulkanGraph, ManagedSiluKeepsTheCommonInfinityAndNanSemantics) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {8});
  const std::array<double, 8> bias{-INFINITY, INFINITY, NAN, -100,
                                   -0.0,      0,        5,   -1e30F};
  const std::array<double, 8> data{1, 2, -3, 4, -5, 6, -7, 8};
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(desc).value();
  const auto b = builder.constant(desc).value();
  const auto normalized = builder.add(c::RmsNorm{}, x).value();
  const auto shifted =
      builder.add(c::Binary{c::BinaryKind::kAdd}, normalized, b).value();
  const auto y = builder.add(c::Unary{c::UnaryKind::kSilu}, shifted).value();
  ASSERT_TRUE(builder.publish(y).ok());
  auto graph = std::move(builder).finish().value();
  const std::array constants{make(context, desc, bias)};
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, constants, {.report = &report}).value();
  ASSERT_EQ(report.managed_region_count, 1U);
  auto slot = program.create_invocation().value();
  const auto input = make(context, desc, data);
  ASSERT_TRUE(slot.bind(input).ok());
  auto work = slot.submit().value();
  auto result = work.wait().value();
  auto expected = norm(data, 8, false, 1e-5F);
  for (std::size_t i = 0; i < expected.size(); ++i)
    expected[i] =
        activate(c::GatedActivationKind::kSilu, expected[i] + bias[i]);
  expect(context, result.get()->tensors()[0], expected);
}

TEST(VulkanGraph, PointwiseRegionPreservesScalarOrderPublishedValuesAndRebinds) {
  auto context = vulkan_test::context().value();
  for (int width : {129, 2048}) {
    const auto desc = dense(c::DataType::kFloat32, {17, width});
    const auto scalar = dense(c::DataType::kFloat32, {});
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(desc).value();
    const auto scale = builder.constant(scalar).value();
    const auto absolute = builder.add(c::Unary{c::UnaryKind::kAbs}, x).value();
    const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, absolute).value();
    const auto exponent = builder.add(c::Unary{c::UnaryKind::kExp}, negative).value();
    const auto bounded = builder.add(c::Unary{c::UnaryKind::kTanh}, exponent).value();
    const auto sigmoid = builder.add(c::Unary{c::UnaryKind::kSigmoid}, bounded).value();
    const auto difference =
        builder.add(c::Binary{c::BinaryKind::kSubtract}, scale, sigmoid).value();
    const auto output =
        builder.add(c::Binary{c::BinaryKind::kDivide}, scale, difference).value();
    ASSERT_TRUE(builder.publish(exponent).ok());
    ASSERT_TRUE(builder.publish(difference).ok());
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    const auto parameter = make(context, scalar, std::array{2.0});
    gv::CompileReport report;
    auto program = gv::compile(context, graph, std::array{parameter},
                               {.report = &report}).value();
    EXPECT_EQ(report.managed_region_count, 1U);
    EXPECT_EQ(report.native_dispatch_count, 1U); // Dynamic input origin adapter.
    graph = {};
    auto first = program.create_invocation().value();
    auto second = program.create_invocation().value();
    v::Tensor retained;
    std::vector<double> saved;
    for (int phase = 0; phase < 3; ++phase) {
      std::vector<double> data(*desc.element_count()), exp(data.size()),
          sub(data.size()), div(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = (static_cast<int>((i + phase * 7) % 37) - 18) * 0.125;
        exp[i] = std::exp(-std::abs(data[i]));
        sub[i] = 2.0 - 1.0 / (1.0 + std::exp(-std::tanh(exp[i])));
        div[i] = 2.0 / sub[i];
      }
      auto &slot = phase == 0 ? first : second;
      const auto input = make(context, desc, data);
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      expect(context, result.get()->tensors()[0], exp);
      expect(context, result.get()->tensors()[1], sub);
      expect(context, result.get()->tensors()[2], div);
      if (phase == 0) {
        retained = result.get()->tensors()[2];
        saved = div;
      } else {
        expect(context, retained, saved);
      }
    }
    first = {};
    second = {};
    program = {};
    expect(context, retained, saved);
  }
}
} // namespace
