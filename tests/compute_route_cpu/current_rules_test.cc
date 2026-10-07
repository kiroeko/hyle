#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>
#include <vector>

#include "../../src/compute_provider_native/default_rules.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/buffer.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#endif

namespace {
using namespace hyle::compute;
auto create_context(cpu::DefaultContextOptions options = {}) {
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  return hyle::graph::cpu::create_context(options);
#else
  return cpu::create_context(options);
#endif
}
TensorDesc dense(DataType t, std::initializer_list<std::int64_t> dimensions) {
  return TensorDesc::dense(t, {dimensions.begin(), dimensions.size()}).value();
}

TEST(CpuCurrentRules, ExactQualifiedRequestsAndNearbyMissesUseOrdinaryFactories) {
  struct Case { const char *id; std::string_view family; int variant, type, rows, width, other; ProviderId winner; };
  // Fixed independent expectation from the completed all-candidate evidence.
  const Case cases[]{
      {"binary-0-t6-r32-h2048-o6", "binary", 0, 6, 32, 2048, 6, native::kProviderId},
      {"binary-1-t4-r32-h2048-o6", "binary", 1, 4, 32, 2048, 6, native::kProviderId},
      {"binary-1-t6-r1-h2048-o6", "binary", 1, 6, 1, 2048, 6, native::kProviderId},
      {"binary-3-t4-r1-h2048-o6", "binary", 3, 4, 1, 2048, 6, native::kProviderId},
      {"binary-3-t4-r1-h4096-o6", "binary", 3, 4, 1, 4096, 6, native::kProviderId},
      {"binary-3-t4-r32-h2048-o6", "binary", 3, 4, 32, 2048, 6, native::kProviderId},
      {"binary-3-t6-r1-h4096-o6", "binary", 3, 6, 1, 4096, 6, native::kProviderId},
      {"binary-4-t4-r1-h2048-o6", "binary", 4, 4, 1, 2048, 6, native::kProviderId},
      {"cast-t2-r1-h4096-o4", "cast", 0, 2, 1, 4096, 4, native::kProviderId},
      {"cast-t3-r1-h2048-o5", "cast", 0, 3, 1, 2048, 5, native::kProviderId},
      {"cast-t3-r1-h4096-o3", "cast", 0, 3, 1, 4096, 3, native::kProviderId},
      {"cast-t3-r32-h2048-o2", "cast", 0, 3, 32, 2048, 2, native::kProviderId},
      {"fill-t6-r1-h2048-o6", "fill", 0, 6, 1, 2048, 6, native::kProviderId},
      {"gather-t4-r1-h4096-o12", "gather", 0, 4, 1, 4096, 12, native::kProviderId},
      {"gather-t4-r1-h4096-o14", "gather", 0, 4, 1, 4096, 14, native::kProviderId},
      {"gather-t5-r1-h4096-o14", "gather", 0, 5, 1, 4096, 14, native::kProviderId},
      {"gather-t6-r1-h2048-o12", "gather", 0, 6, 1, 2048, 12, native::kProviderId},
      {"norm-0-t5-r1-h4096-o6", "norm", 0, 5, 1, 4096, 6, native::kProviderId},
      {"reduce-2-t4-r32-h2048-o6", "reduce", 2, 4, 32, 2048, 6, native::kProviderId},
      {"reduce-3-t4-r1-h2048-o6", "reduce", 3, 4, 1, 2048, 6, native::kProviderId},
      {"reduce-3-t5-r1-h4096-o6", "reduce", 3, 5, 1, 4096, 6, native::kProviderId},
      {"silu-gate-0-t4-r32-h2048-o6", "silu-gate", 0, 4, 32, 2048, 6, native::kProviderId},
      {"softmax-2-t4-r1-h2048-o6", "softmax", 2, 4, 1, 2048, 6, native::kProviderId},
      {"softmax-2-t4-r32-h2048-o6", "softmax", 2, 4, 32, 2048, 6, aten_cpu::kProviderId},
      {"unary-1-t5-r1-h2048-o6", "unary", 1, 5, 1, 2048, 6, native::kProviderId},
      {"unary-1-t5-r32-h2048-o6", "unary", 1, 5, 32, 2048, 6, native::kProviderId},
      {"unary-11-t4-r32-h2048-o6", "unary", 11, 4, 32, 2048, 6, native::kProviderId},
      {"unary-2-t4-r1-h2048-o6", "unary", 2, 4, 1, 2048, 6, native::kProviderId},
      {"unary-2-t4-r32-h2048-o6", "unary", 2, 4, 32, 2048, 6, native::kProviderId},
      {"unary-2-t5-r32-h2048-o6", "unary", 2, 5, 32, 2048, 6, native::kProviderId},
      {"unary-2-t6-r1-h2048-o6", "unary", 2, 6, 1, 2048, 6, native::kProviderId},
      {"unary-2-t6-r1-h4096-o6", "unary", 2, 6, 1, 4096, 6, native::kProviderId},
      {"unary-2-t6-r32-h2048-o6", "unary", 2, 6, 32, 2048, 6, native::kProviderId},
      {"unary-3-t4-r1-h2048-o6", "unary", 3, 4, 1, 2048, 6, native::kProviderId},
      {"unary-4-t4-r1-h2048-o6", "unary", 4, 4, 1, 2048, 6, native::kProviderId},
      {"unary-4-t5-r32-h2048-o6", "unary", 4, 5, 32, 2048, 6, native::kProviderId},
      {"unary-5-t4-r1-h2048-o6", "unary", 5, 4, 1, 2048, 6, native::kProviderId},
      {"unary-5-t4-r32-h2048-o6", "unary", 5, 4, 32, 2048, 6, native::kProviderId},
      {"unary-5-t5-r32-h2048-o6", "unary", 5, 5, 32, 2048, 6, native::kProviderId},
      {"unary-6-t4-r1-h4096-o6", "unary", 6, 4, 1, 4096, 6, native::kProviderId},
      {"unary-8-t5-r1-h2048-o6", "unary", 8, 5, 1, 2048, 6, native::kProviderId},
      {"unary-8-t5-r32-h2048-o6", "unary", 8, 5, 32, 2048, 6, native::kProviderId},
  };
  auto context_result = create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto multiple_result = create_context({.native_threads = 2});
  ASSERT_TRUE(multiple_result) << multiple_result.error().message();
  auto multiple = std::move(*multiple_result);
  const bool qualified = native::internal::qualified_default_environment();
  ::testing::Test::RecordProperty("qualified_default_environment", qualified ? "1" : "0");
  for (const auto &item : cases) {
    SCOPED_TRACE(item.id);
    const auto t = DataType(item.type);
    const auto input = dense(t, {item.rows, item.width});
    std::vector<TensorDesc> inputs{input};
    Operation op;
    if (item.family == "unary") op = Operation(Unary{UnaryKind(item.variant)});
    else if (item.family == "binary") {
      op = Operation(Binary{BinaryKind(item.variant)}); inputs.push_back(input);
    } else if (item.family == "norm") {
      op = Operation(RmsNorm{}); inputs.push_back(dense(t, {item.width}));
    } else if (item.family == "reduce") {
      op = Operation(Reduce{.kind = ReductionKind(item.variant),
                  .axes = Axes::from(std::array<std::int32_t, 1>{1}).value(),
                  .accumulation_type = DataType::kFloat32});
    } else if (item.family == "softmax") {
      op = Operation(Softmax{.mask = MaskKind::kBoolean});
      inputs.push_back(dense(DataType::kBool, {item.rows, item.width}));
    } else if (item.family == "silu-gate") {
      op = Operation(GatedActivation{.inputs = GatedInputKind::kSeparate}); inputs.push_back(input);
    } else if (item.family == "fill") {
      op = Operation(Fill{.output_shape = TensorShape::create(std::array<std::int64_t, 2>{item.rows, item.width}).value(),
                .output_type = t, .value = ScalarValue::from_bits(t, 0x3fa00000ULL)});
      inputs.clear();
    } else if (item.family == "gather") {
      op = Operation(Gather{}); inputs = {dense(t, {64, item.width}), dense(DataType(item.other), {item.rows})};
    } else if (item.family == "cast") op = Operation(Cast{DataType(item.other)});
    const auto inferred = infer_single_output(op, inputs);
    ASSERT_TRUE(inferred) << inferred.error().message();
    const auto output = *inferred;
    const auto outputs = std::span(&output, 1);
    auto prepared = prepare(context, op, inputs, outputs, hyle::mem::system_resource());
    ASSERT_TRUE(prepared) << prepared.error().message();
    if (qualified) {
      EXPECT_EQ(prepared->provider().id, item.winner);
      EXPECT_EQ(prepared->route_policy().version, 3u);
      EXPECT_EQ(prepared->route_policy().namespace_id, 0x48594c4543505552ULL);
    } else {
      EXPECT_NE(prepared->route_policy().namespace_id, 0x48594c4543505552ULL);
    }
    // This is a real public thread-policy miss; math stays available.
    auto fallback = prepare(multiple, op, inputs, outputs, hyle::mem::system_resource());
    ASSERT_TRUE(fallback) << fallback.error().message();
    EXPECT_NE(fallback->provider().id, native::kProviderId);
    EXPECT_NE(fallback->route_policy().namespace_id, 0x48594c4543505552ULL);
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
    // Exercise actual Graph lowering of the same leaf with regions explicitly
    // disabled. Ordinary region planning is checked separately and keeps priority.
    auto builder = hyle::graph::GraphBuilder::create().value();
    std::vector<hyle::graph::ValueId> values;
    for (const auto &description : inputs)
      values.push_back(builder.input(description).value());
    auto result = builder.result(builder.add_node(op, values).value()).value();
    ASSERT_TRUE(builder.publish(result).ok());
    auto graph = std::move(builder).finish().value();
    auto compiled = hyle::graph::compile(
        graph, context, {.region_selection = hyle::graph::RegionSelectionPolicy::kDisabled});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.step(0)->provider.id, prepared->provider().id);
#endif
  }
  // Shape and attribute misses do not acquire a sampled WIN.
  for (const auto width : {2047, 2049}) {
    auto program = prepare(context, Unary{UnaryKind::kExp}, dense(DataType::kFloat32, {1, width}));
    ASSERT_TRUE(program);
    EXPECT_NE(program->provider().id, native::kProviderId);
  }
  const auto norm_input = dense(DataType::kBFloat16, {1, 4096});
  const std::array norm_inputs{norm_input, dense(DataType::kBFloat16, {4096})};
  auto epsilon_miss = prepare(context, Operation(RmsNorm{.epsilon = 2e-5F}), norm_inputs,
                              std::span(&norm_input, 1), hyle::mem::system_resource());
  ASSERT_TRUE(epsilon_miss);
  EXPECT_NE(epsilon_miss->provider().id, native::kProviderId);
}

TEST(CpuCurrentRules, MatureProjectionWinnerAndPhysicalLayoutMissKeepPrecision) {
  auto context = create_context().value();
  auto weight = dense(DataType::kBFloat16, {6144, 2048});
  auto left = dense(DataType::kBFloat16, {32, 2048});
  auto program = prepare(context, MatMul{}, left, weight.transposed().value());
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->provider().id, ggml_cpu::kProviderId);
  EXPECT_EQ(program->output_description()->data_type(), DataType::kBFloat16);
  if (native::internal::qualified_default_environment()) {
    EXPECT_EQ(program->route_policy().version, 3u);
  }
  auto layout_miss = prepare(context, MatMul{}, left, dense(DataType::kBFloat16, {2048, 6144}));
  ASSERT_TRUE(layout_miss);
  EXPECT_NE(layout_miss->route_policy().namespace_id, 0x48594c4543505552ULL);
}

TEST(CpuCurrentRules, PreparedReplayKeepsTheSelectedProgramAndChangedInputs) {
  const auto desc = dense(DataType::kFloat32, {1, 2048});
  auto context = create_context().value();
  auto queue = context.create_queue().value();
  auto program = prepare(context, Unary{UnaryKind::kExp}, desc).value();
  auto input = hyle::mem::Buffer::create(hyle::mem::system_resource(), 8192, 64).value();
  auto output = hyle::mem::Buffer::create(hyle::mem::system_resource(), 8192, 64).value();
  auto invocation = program.create_invocation().value();
  const std::array inputs{ConstTensorView::create(desc, input.view()).value()};
  const std::array outputs{TensorView::create(desc, output.view()).value()};
  std::vector<float> values(2048), actual(2048);
  const auto provider = program.provider().id;
  const auto policy = program.route_policy();
  for (float value : {0.25F, -0.5F}) {
    std::fill(values.begin(), values.end(), value);
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
    auto completion = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    EXPECT_FALSE(invocation.in_flight());
    ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (float result : actual)
      EXPECT_NEAR(result, std::exp(value), 3e-5F);
    EXPECT_EQ(program.provider().id, provider);
    EXPECT_EQ(program.route_policy(), policy);
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuCurrentRules, OrdinaryGraphKeepsCompleteMatureRegionsBeforeLeafWins) {
  auto context = create_context().value();
  const auto desc = dense(DataType::kFloat32, {1, 2048});
  auto builder = hyle::graph::GraphBuilder::create().value();
  auto input = builder.input(desc).value();
  auto expanded = builder.add(Unary{UnaryKind::kExp}, input).value();
  auto output = builder.add(Binary{BinaryKind::kMultiply}, expanded, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = hyle::graph::compile(graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id, ggml_cpu::kProviderId);
}
#endif
} // namespace
