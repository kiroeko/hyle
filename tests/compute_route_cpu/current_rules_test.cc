#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <vector>

#include "../../src/compute_route_cpu/default_policy.h"
#include "../../src/compute_packed_shared/artifact_packed_linear.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/buffer.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#endif

namespace {
using namespace hyle::compute;
constexpr std::uint64_t kEngineeringPolicy = 0x48594c4543505544ULL;
constexpr std::uint64_t kStoragePolicy = 0x48594c4553544f52ULL;

auto create_context(cpu::DefaultContextOptions options = {}) {
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  return hyle::graph::cpu::create_context(options);
#else
  return cpu::create_context(options);
#endif
}
TensorDesc dense(DataType type, std::initializer_list<std::int64_t> dimensions) {
  return TensorDesc::dense(type, {dimensions.begin(), dimensions.size()}).value();
}

TEST(CpuCurrentRules, FlatExpUsesTheSameEngineeringPolicyAcrossRanksAndSizes) {
  auto context = create_context().value();
  const std::array descriptions{
      dense(DataType::kFloat32, {3072}),
      dense(DataType::kFloat32, {3, 1024}),
      dense(DataType::kFloat32, {2, 3, 512}),
      dense(DataType::kFloat32, {17, 241}),
      dense(DataType::kFloat32, {1, 2048}),
      dense(DataType::kFloat32, {1, 4096}),
      dense(DataType::kFloat32, {32, 2048})};
  RoutePolicyId policy;
  for (const auto &description : descriptions) {
    auto prepared = prepare(context, Unary{UnaryKind::kExp}, description);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, native::kProviderId);
    EXPECT_EQ(*prepared->output_description(), description);
    EXPECT_EQ(prepared->route_policy().namespace_id, kEngineeringPolicy);
    EXPECT_TRUE(prepared->route_policy().valid());
    if (policy.empty()) policy = prepared->route_policy();
    EXPECT_EQ(prepared->route_policy(), policy);
  }
  // native_threads is ggml's independent team budget, not an ATen setter or
  // a claim that Native's flat loop acquired two workers. Keep the other
  // engineering baseline when this single-lane comparison no longer applies.
  auto team = create_context({.native_threads = 2}).value();
  auto prepared = prepare(team, Unary{UnaryKind::kExp}, descriptions.front());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_NE(prepared->provider().id, native::kProviderId);
  EXPECT_EQ(prepared->route_policy().namespace_id, kEngineeringPolicy);
  EXPECT_NE(prepared->route_policy(), policy);
}

TEST(CpuCurrentRules, OtherMathAttributesAndPhysicalLayoutsKeepVisibleDefaults) {
  auto context = create_context().value();
  const auto tensor = dense(DataType::kBFloat16, {3, 37});
  const std::array inputs{tensor, dense(DataType::kBFloat16, {37})};
  auto norm = prepare(context, Operation(RmsNorm{.epsilon = 2e-5F}), inputs,
                      std::span(&tensor, 1), hyle::mem::system_resource());
  ASSERT_TRUE(norm) << norm.error().message();
  EXPECT_EQ(norm->route_policy().namespace_id, kEngineeringPolicy);
  EXPECT_EQ(norm->output_description()->data_type(), DataType::kBFloat16);
  const auto left = dense(DataType::kBFloat16, {5, 19});
  const auto weight = dense(DataType::kBFloat16, {23, 19});
  for (const auto &right : {weight.transposed().value(), dense(DataType::kBFloat16, {19, 23})}) {
    auto program = prepare(context, MatMul{}, left, right);
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(program->output_description()->data_type(), DataType::kBFloat16);
    EXPECT_EQ(program->route_policy().namespace_id, kEngineeringPolicy);
  }
}

TEST(CpuCurrentRules, SelectedByteGatherDoesNotDependOnBankModelOrThreadRecipe) {
  for (auto threads : {1u, 2u}) {
    auto context = create_context({.native_threads = threads}).value();
    const std::array sources{
        dense(DataType::kFloat16, {5, 7}),
        dense(DataType::kBFloat16, {5, 7}),
        TensorDesc::strided_bytes(DataType::kFloat32,
            std::array<std::int64_t, 2>{5, 7},
            std::array<std::int64_t, 2>{48, 4}).value()};
    for (const auto &source : sources) {
      for (const auto index_type : {DataType::kInt32, DataType::kInt64}) {
        auto program = prepare(context, Gather{}, source, dense(index_type, {3}));
        ASSERT_TRUE(program) << program.error().message();
        EXPECT_EQ(program->provider().id, native::kProviderId);
        EXPECT_EQ(program->route_policy().namespace_id, kStoragePolicy);
        EXPECT_EQ(program->output_description()->data_type(), source.data_type());
      }
    }
  }
}

TEST(CpuCurrentRules, ExpertNativeHasNoFacadePolicyAndPackedPriorityStaysOriginal) {
  auto expert = Context::create(cpu::device(), native::provider()).value();
  auto exp = prepare(expert, Unary{UnaryKind::kExp}, dense(DataType::kFloat32, {513}));
  ASSERT_TRUE(exp) << exp.error().message();
  EXPECT_TRUE(exp->route_policy().empty());
  const auto operation = Operation::custom(packed::internal::ArtifactPackedLinear{});
  const std::array inputs{dense(DataType::kBFloat16, {2, 32}),
      dense(DataType::kInt32, {8, 4}), dense(DataType::kBFloat16, {8, 1})};
  const auto output = infer_single_output(operation, inputs);
  ASSERT_TRUE(output) << output.error().message();
  // Direct uses this original native vtable. Graph's public factory instead
  // owns its CT-region wrapper; do not ask a different vtable to inherit that
  // already assembled facade's policy through the original identity guard.
  auto provider = cpu::internal::with_defaults(native::provider(),
      cpu::internal::DefaultProvider::kNative, cpu::internal::DefaultScope::kDirect, true);
  auto context = Context::create(cpu::device(), provider).value();
  auto program = prepare(context, operation, inputs, std::span(&*output, 1), hyle::mem::system_resource());
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->provider().id, native::kProviderId);
  EXPECT_EQ(program->route_preference(), 1001u);
  EXPECT_EQ(program->route_policy().namespace_id, kEngineeringPolicy);
}

TEST(CpuCurrentRules, PreparedExpReplayPreservesIeeeValuesAndReadsChangedInputs) {
  auto context = create_context().value();
  auto queue = context.create_queue().value();
  constexpr std::size_t count = 17;
  const auto description = dense(DataType::kFloat32, {count});
  auto program = prepare(context, Unary{UnaryKind::kExp}, description).value();
  ASSERT_EQ(program.provider().id, native::kProviderId);
  auto input = hyle::mem::Buffer::create(hyle::mem::system_resource(), count * sizeof(float), 64).value();
  auto output = hyle::mem::Buffer::create(hyle::mem::system_resource(), count * sizeof(float), 64).value();
  auto invocation = program.create_invocation().value();
  const std::array inputs{ConstTensorView::create(description, input.view()).value()};
  const std::array outputs{TensorView::create(description, output.view()).value()};
  std::array<float, count> values{std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      0.0F, -0.0F, -200.0F, 100.0F, -90.0F, -103.0F, 88.0F,
      -1.0F, 1.0F, -0.5F, 0.25F, 2.0F, -2.0F, 0.75F};
  std::array<float, count> actual{};
  const auto policy = program.route_policy();
  for (unsigned round = 0; round < 2; ++round) {
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
    auto done = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t i = 0; i < count; ++i) {
      const double reference = std::exp(static_cast<double>(values[i]));
      if (std::isnan(reference)) EXPECT_TRUE(std::isnan(actual[i])) << i;
      else if (reference > std::numeric_limits<float>::max()) {
        EXPECT_TRUE(std::isinf(actual[i]) && !std::signbit(actual[i])) << i;
      } else {
        const auto expected = static_cast<float>(reference);
        if (expected == 0.0F || values[i] == 0.0F)
          EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected)) << i;
        else {
          EXPECT_GT(actual[i], 0.0F) << i;
          // Same8e-7 relative budget as the existing SIMD math oracle; use a
          // subnormal-ULP floor rather than an absolute1e-6 loophole near zero.
          const double tolerance = std::abs(reference) * 8e-7 +
              2.0 * std::numeric_limits<float>::denorm_min();
          EXPECT_NEAR(static_cast<double>(actual[i]), reference, tolerance) << i;
        }
      }
    }
    EXPECT_EQ(program.route_policy(), policy);
    EXPECT_FALSE(invocation.in_flight());
    for (std::size_t i = 0; i < count; ++i) values[i] = -0.5F + 0.0625F * static_cast<float>(i);
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuCurrentRules, OrdinaryGraphKeepsCompleteMatureRegionsBeforeFlatLeaves) {
  auto context = create_context().value();
  const auto description = dense(DataType::kFloat32, {3, 37});
  auto builder = hyle::graph::GraphBuilder::create().value();
  auto input = builder.input(description).value();
  auto expanded = builder.add(Unary{UnaryKind::kExp}, input).value();
  auto output = builder.add(Binary{BinaryKind::kMultiply}, expanded, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = hyle::graph::compile(graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(compiled->report.step_count(), 1u);
  // The complete region100 wins before Native's individual Graph leaf175.
  EXPECT_NE(compiled->report.step(0)->provider.id, native::kProviderId);
}
#endif
} // namespace
