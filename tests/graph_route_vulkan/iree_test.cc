#include "../compute_route_vulkan/iree_test_support.h"
#include "../compute_route_vulkan/artifact_test_support.h"
#include "../compute_route_vulkan/mlx_test_support.h"
#include "../compute_route_vulkan/official_test_support.h"
#include "../compute_route_vulkan/scan_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_iree_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;
namespace t = vulkan_tensor_test;
namespace s = vulkan_scan_test;

hyle::Result<v::Program> compile_artifact(const v::Context &context,
                                        c::Operation operation,
                                        std::span<const c::TensorDesc> descriptions) {
  auto builder = g::GraphBuilder::create().value();
  std::vector<g::ValueId> inputs;
  for (const auto &description : descriptions)
    inputs.push_back(builder.input(description).value());
  const auto node = builder.add_node(operation, inputs).value();
  const auto result = builder.result(node).value();
  EXPECT_TRUE(builder.publish(result).ok());
  return gv::compile(context, std::move(builder).finish().value());
}

TEST(VulkanGraph, OriginalOfficialDynamicLinearsAndWeightTailDecode) {
  vulkan_official_test::linears(compile_artifact);vulkan_official_test::decodes(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialGroupedFp4Fp8SharedDistinctAndChecked) {
  vulkan_official_test::grouped(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialQatFiniteBfloatExhaustiveAndFailure) {
  vulkan_official_test::roundtrips(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialIndependentLeaseFiniteCancelCloseAndOom) {
  vulkan_official_test::lifecycle(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialEncodedBlock128TailAndReplay) {
  auto context = vulkan_test::context().value();
  for (int n : {5, 129}) {
    Float8Inputs a(3, 256, n, c::Float8ScaleMode::kFloat32Block128), b(3, 256, n, c::Float8ScaleMode::kFloat32Block128, 1);
    const std::array descriptions{a.activation, a.weight};
    auto program = compile_artifact(context, c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}), descriptions);
    ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
    for (const auto *data : {&a, &b}) {
      const std::array inputs{context.upload(data->activation, data->a).value(), context.upload(data->weight, data->w).value()};
      ASSERT_TRUE(slot.bind(inputs).ok());auto work = slot.submit().value();auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();t::expect(context, result->get()->tensors()[0], data->multiply(), true);
    }
  }
}

TEST(VulkanGraph, OriginalMlxBatchUsesOrdinaryTypedNodesAndCheckedSlots) {
  vulkan_mlx_test::batch(compile_artifact);
}

TEST(VulkanGraph, OriginalMlxConstantsKeepBanksAcrossRebindingAndSlots) {
  vulkan_mlx_test::constants([](const v::Context &context, c::Operation operation,
                               std::span<const v::Tensor> resident) {
    auto builder = g::GraphBuilder::create().value();std::vector<g::ValueId> inputs;
    std::vector<v::Tensor> constants;
    for (std::size_t i = 0; i < resident.size(); ++i)
      if (i == 0 || i + 1 == resident.size()) inputs.push_back(builder.input(resident[i].description()).value());
      else { inputs.push_back(builder.constant(resident[i].description()).value());constants.push_back(resident[i]); }
    const auto result = builder.result(builder.add_node(operation, inputs).value()).value();
    EXPECT_TRUE(builder.publish(result).ok());
    return gv::compile(context, std::move(builder).finish().value(), constants);
  });
}

TEST(VulkanGraph, OriginalArtifactBatchUsesOrdinaryTypedEntry) {
  vulkan_artifact_test::batch(compile_artifact);
}

TEST(VulkanGraph, OriginalArtifactColdResultBindsMatureConstantProjection) {
  vulkan_artifact_test::cold_projection(compile_artifact,
      [](const v::Context &context, const c::TensorDesc &activation, const v::Tensor &weight) {
    auto builder = g::GraphBuilder::create().value();
    const auto a = builder.input(activation).value();
    const auto w = builder.constant(weight.description()).value();
    const auto result = builder.add(
        c::MatMul{.accumulation_type = c::DataType::kFloat32}, a, w).value();
    EXPECT_TRUE(builder.publish(result).ok());
    return gv::compile(context, std::move(builder).finish().value(), std::array{weight});
  });
}

TEST(VulkanGraph, OriginalFloat8RegionContinuesIntoGgmlAndNativeCopy) {
  auto context = vulkan_test::context().value();
  for (auto mode : {c::Float8ScaleMode::kFloat32Block128,
                    c::Float8ScaleMode::kUnsignedExponent8Group32}) {
    Float8Inputs data(3, 128, 128, mode, 1);
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(data.activation).value();
    const auto weight = builder.constant(data.weight).value();
    const auto middle = builder.add(c::MatMul{}, input, weight).value();
    const auto absolute =
        builder.add(c::Unary{c::UnaryKind::kAbs}, middle).value();
    const auto output = builder.add(c::Copy{}, absolute).value();
    const auto low =
        builder.add(c::Cast{c::DataType::kBFloat16}, middle).value();
    ASSERT_TRUE(builder.publish(output).ok());
    ASSERT_TRUE(builder.publish(middle).ok());
    ASSERT_TRUE(builder.publish(low).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    const auto resident = context.upload(data.weight, data.w).value();
    auto program =
        gv::compile(context, graph, std::array{resident}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_GE(report.managed_region_count, 2U);
    EXPECT_NE(report.numerical_policy.find("FP8"), std::string::npos);
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(
        slot.bind(context.upload(data.activation, data.a).value()).ok());
    const auto expected = data.multiply();
    auto positive = expected;
    for (auto &value : positive)
      value = std::abs(value);
    graph = {};
    *program = {};
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      t::expect(context, result->get()->tensors()[0], positive, true);
      t::expect(context, result->get()->tensors()[1], expected, true);
      t::expect(context, result->get()->tensors()[2], expected, true);
    }
    EXPECT_EQ(t::bytes(context, resident), data.w);
  }
}

TEST(VulkanGraph, CheckedScanChainsTwoOriginalProgramsWithIndependentResults) {
  auto context = vulkan_test::context().value();
  const auto desc = t::dense(c::DataType::kInt32, {2, 257});
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(desc).value();
  const auto middle = builder.add(c::Scan{}, input).value();
  const auto output = builder.add(c::Scan{.exclusive = true}, middle).value();
  ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(middle).ok());
  const auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report}).value();
  EXPECT_EQ(report.managed_region_count, 2U);
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  v::Tensor retained;
  std::vector<std::int32_t> saved;
  for (int phase = 0; phase < 3; ++phase) {
    const auto data = s::integer_pattern(514, phase);
    auto prefix = data;
    for (std::size_t row = 0; row < 2; ++row) {
      std::int64_t sum = 0;
      for (std::size_t col = 0; col < 257; ++col) {
        sum += data[row * 257 + col];
        prefix[row * 257 + col] = static_cast<std::int32_t>(sum);
      }
    }
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(s::make_integer(context, desc, data)).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    s::expect_integer_prefix(context, result->get()->tensors()[0], desc, prefix,
                             c::Scan{.exclusive = true});
    EXPECT_EQ(s::integer_values(context, result->get()->tensors()[1]), prefix);
    if (phase == 0) {
      retained = result->get()->tensors()[0];
      saved = s::integer_values(context, retained);
    } else {
      EXPECT_EQ(s::integer_values(context, retained), saved);
    }
  }
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(s::integer_values(context, retained), saved);
}
} // namespace
