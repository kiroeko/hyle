#include <gtest/gtest.h>

#include "../compute/workload_fixture.h"
#include "../compute_route_vulkan/gate_norm_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_gate_norm_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     ManagedLayerNormKeepsMisalignedCapturedRowsAndPublishedForks) {
  auto context = vulkan_test::context().value();
  constexpr int width = 129;
  std::vector<double> storage(4 * width);
  for (std::size_t i = 0; i < storage.size(); ++i)
    storage[i] = (static_cast<int>(i % 29) - 14) * 0.125;
  auto parent =
      make(context, dense(c::DataType::kFloat32, {4, width}), storage);
  auto input = parent.view(c::View::slice(0, 1, 3).value()).value();
  auto builder = g::GraphBuilder::create().value();
  const auto captured = builder.constant(input.description()).value();
  const auto normalized = builder.add(c::LayerNorm{}, captured).value();
  const auto residual =
      builder.add(c::Binary{c::BinaryKind::kAdd}, normalized, captured).value();
  ASSERT_TRUE(builder.publish(normalized).ok());
  ASSERT_TRUE(builder.publish(residual).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto workload = workload_test::known(c::WorkloadPhase::kPrefill);
  auto program = gv::compile(context, graph, std::array{input},
                             {.report = &report, .workload = workload});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->workload(), workload);
  workload.region.version = 0;
  EXPECT_FALSE(
      gv::compile(context, graph, std::array{input}, {.workload = workload}));
  workload = {};
  EXPECT_TRUE(program->workload().profile.matches(workload_test::kModel));
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_EQ(report.native_dispatch_count, 1U); // Descriptor alignment adapter.
  auto slot = program->create_invocation().value();
  graph = {};
  input = {};
  parent = {};
  ASSERT_TRUE(slot.bind(std::span<const v::Tensor>{}).ok());
  v::Tensor retained;
  const auto data = std::span<const double>(storage).subspan(width);
  const auto expected = norm(data, width, true, 1e-5F);
  std::vector<double> added(expected.size());
  for (std::size_t i = 0; i < added.size(); ++i)
    added[i] = expected[i] + data[i];
  {
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    expect(context, result->get()->tensors()[0], expected);
    expect(context, result->get()->tensors()[1], added);
    retained = result->get()->tensors()[0];
  }
  slot = {};
  *program = {};
  ASSERT_TRUE(context.close().ok());
  expect(context, retained, expected);
}

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

TEST(VulkanGraph,
     PointwiseRegionPreservesScalarOrderPublishedValuesAndRebinds) {
  auto context = vulkan_test::context().value();
  for (int width : {129, 2048}) {
    const auto desc = dense(c::DataType::kFloat32, {17, width});
    const auto scalar = dense(c::DataType::kFloat32, {});
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(desc).value();
    const auto scale = builder.constant(scalar).value();
    const auto absolute = builder.add(c::Unary{c::UnaryKind::kAbs}, x).value();
    const auto negative =
        builder.add(c::Unary{c::UnaryKind::kNeg}, absolute).value();
    const auto exponent =
        builder.add(c::Unary{c::UnaryKind::kExp}, negative).value();
    const auto bounded =
        builder.add(c::Unary{c::UnaryKind::kTanh}, exponent).value();
    const auto sigmoid =
        builder.add(c::Unary{c::UnaryKind::kSigmoid}, bounded).value();
    const auto difference =
        builder.add(c::Binary{c::BinaryKind::kSubtract}, scale, sigmoid)
            .value();
    const auto output =
        builder.add(c::Binary{c::BinaryKind::kDivide}, scale, difference)
            .value();
    ASSERT_TRUE(builder.publish(exponent).ok());
    ASSERT_TRUE(builder.publish(difference).ok());
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    const auto parameter = make(context, scalar, std::array{2.0});
    gv::CompileReport report;
    auto program =
        gv::compile(context, graph, std::array{parameter}, {.report = &report})
            .value();
    EXPECT_EQ(report.managed_region_count, 1U);
    EXPECT_EQ(report.native_dispatch_count,
              1U); // Dynamic input origin adapter.
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
TEST(VulkanGraph, StableModelGatesShareMatureRegionAndKeepPublishedForks) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  v::Tensor retained;
  std::vector<double> saved;
  auto context = vulkan_test::context(&quota).value();
  {
    constexpr int rows = 3, width = 129;
    const auto desc =
        c::TensorDesc::strided_bytes(
            c::DataType::kBFloat16, std::array<std::int64_t, 2>{rows, width},
            std::array<std::int64_t, 2>{width * 3 + 1, 3})
            .value();
    const auto parameter = dense(c::DataType::kFloat32, {width});
    std::vector<double> bias(width), logarithm(width);
    for (int i = 0; i < width; ++i) {
      bias[i] = (i % 5 - 2) * 0.125;
      logarithm[i] = (i % 7 - 3) * 0.0625;
    }
    const std::array constants{make(context, parameter, bias),
                               make(context, parameter, logarithm)};
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(desc).value();
    const auto bias_id = builder.constant(parameter).value();
    const auto log_id = builder.constant(parameter).value();
    const auto wide =
        builder.add(c::Cast{c::DataType::kFloat32}, input).value();
    const auto added =
        builder.add(c::Binary{c::BinaryKind::kAdd}, wide, bias_id).value();
    const auto plus =
        builder.add(c::Unary{c::UnaryKind::kSoftplus}, added).value();
    const auto score = builder.add(c::Unary{c::UnaryKind::kSqrt}, plus).value();
    const auto exponential =
        builder.add(c::Unary{c::UnaryKind::kExp}, log_id).value();
    const auto decay =
        builder.add(c::Unary{c::UnaryKind::kNeg}, exponential).value();
    const auto gate =
        builder.add(c::Binary{c::BinaryKind::kMultiply}, plus, decay).value();
    const auto sigmoid =
        builder.add(c::Unary{c::UnaryKind::kSigmoid}, wide).value();
    const auto attention =
        builder.add(c::Binary{c::BinaryKind::kMultiply}, sigmoid, wide).value();
    for (auto value : {plus, score, gate, attention})
      ASSERT_TRUE(builder.publish(value).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program =
        gv::compile(context, graph, constants, {.report = &report}).value();
    EXPECT_EQ(report.managed_region_count, 1U);
    auto first = program.create_invocation().value();
    auto second = program.create_invocation().value();
    for (int phase = 0; phase < 3; ++phase) {
      std::vector<double> data(rows * width), softplus(data.size()),
          root(data.size()), gated(data.size()), attended(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = rounded((static_cast<int>((i + phase * 3) % 49) - 40) * 0.5,
                          c::DataType::kBFloat16);
        const auto x = static_cast<float>(data[i] + bias[i % width]);
        softplus[i] = static_cast<float>(unary(c::UnaryKind::kSoftplus, x));
        root[i] = std::sqrt(softplus[i]);
        gated[i] =
            softplus[i] * -static_cast<float>(std::exp(logarithm[i % width]));
        attended[i] =
            static_cast<float>(unary(c::UnaryKind::kSigmoid, data[i])) *
            data[i];
      }
      const auto tensor = make(context, desc, data);
      auto &slot = phase == 0 ? first : second;
      ASSERT_TRUE(slot.bind(tensor).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      expect(context, result.get()->tensors()[0], softplus);
      expect(context, result.get()->tensors()[1], root);
      expect(context, result.get()->tensors()[2], gated);
      expect(context, result.get()->tensors()[3], attended);
      if (phase == 0) {
        retained = result.get()->tensors()[2];
        saved = gated;
      } else {
        expect(context, retained, saved);
      }
    }
  }
  ASSERT_TRUE(context.close().ok());
  expect(context, retained, saved);
  retained = {};
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanGraph, B2BatchedLinearAndHadamardShareOneMatureRegion) {
  auto context = vulkan_test::context().value();
  const auto left_desc = dense(c::DataType::kFloat16, {1, 2, 3});
  const auto right_desc = dense(c::DataType::kFloat16, {1, 3, 64});
  auto builder = g::GraphBuilder::create().value();
  const auto left = builder.input(left_desc).value();
  const auto right = builder.constant(right_desc).value();
  const auto product = builder.add(c::MatMul{}, left, right).value();
  const auto output = builder.add(c::Hadamard{}, product).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  std::array<double, 192> right_values{};
  right_values.fill(1.0);
  const std::array constants{make(context, right_desc, right_values)};
  gv::CompileReport report;
  auto program = gv::compile(context, graph, constants, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  ASSERT_TRUE(report.status.ok());
  EXPECT_EQ(report.managed_region_count, 1U);
  ASSERT_EQ(report.nodes.size(), 2U);
  EXPECT_EQ(report.nodes[0].kind, c::OperationKind::kMatMul);
  EXPECT_EQ(report.nodes[1].kind, c::OperationKind::kHadamard);
  auto slot = program->create_invocation().value();
  for (int replay = 0; replay < 2; ++replay) {
    const std::array<double, 6> left_values{1.0 + replay, 2, 3,
                                            4.0 + replay, 5, 6};
    const auto input = make(context, left_desc, left_values);
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait().value();
    std::array<double, 128> expected{};
    expected[0] = 8.0 * (left_values[0] + left_values[1] + left_values[2]);
    expected[64] = 8.0 * (left_values[3] + left_values[4] + left_values[5]);
    expect(context, result.get()->tensors()[0], expected);
  }
}
} // namespace
