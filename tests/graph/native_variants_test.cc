#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "hyle/graph/native_variants.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/quota_allocator.h"
#include "../../src/graph/native_analysis.h"

namespace {
namespace g = hyle::graph;
namespace c = hyle::compute;
namespace m = hyle::mem;

struct Fixture {
  g::Graph graph;
  g::DimensionId rows;
  g::ValueId input, output;
};

Fixture make_graph() {
  auto builder = g::GraphBuilder::create().value();
  auto rows = builder.dimension({1, 4}).value();
  auto spec = g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(8)}).value();
  auto input = builder.input(spec).value();
  auto node = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{input}, std::array{spec}).value();
  auto output = builder.result(node).value();
  EXPECT_TRUE(builder.publish(output).ok());
  return {std::move(builder).finish().value(), rows, input, output};
}

TEST(ExactNativeVariants, ConcreteSnapshotKeepsOriginalIdentitiesAndInference) {
  auto f = make_graph();
  const std::array shape{g::ShapeBinding{f.rows, 4}};
  auto point = g::specialize_exact(f.graph, shape);
  ASSERT_TRUE(point);
  EXPECT_EQ(point->graph.dimension_count(), 0);
  EXPECT_EQ(point->graph.input_id().value(), f.input);
  EXPECT_EQ(point->graph.output_id().value(), f.output);
  EXPECT_EQ(point->graph.input_description()->dimensions()[0], 4);
  EXPECT_EQ(point->graph.output_description()->dimensions()[0], 4);
  EXPECT_EQ(point->signature.binding(0)->value, 4);
  EXPECT_FALSE(f.graph.input_description()); // Logical graph stays symbolic.
}

TEST(ExactNativeVariants, FiniteCoverageRejectsHoleAndForeignOrIncompleteBindings) {
  auto f = make_graph();
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  unsigned factories = 0;
  auto family = g::ExactNativeVariants<g::Graph>::prepare(f.graph, table,
      [&](const g::Graph &point) -> hyle::Result<g::Graph> { ++factories; return point; });
  ASSERT_TRUE(family);
  EXPECT_EQ(factories, 2);
  EXPECT_EQ(family->variant_count(), 2);
  EXPECT_EQ(family->select(four)->output_description()->dimensions()[0], 4);
  const std::array two{g::ShapeBinding{f.rows, 2}};
  auto missing = family->select(two);
  ASSERT_FALSE(missing); EXPECT_TRUE(missing.error().IsUnimplemented());
  EXPECT_FALSE(family->select());
  auto foreign = make_graph();
  EXPECT_FALSE(family->select(std::array{g::ShapeBinding{foreign.rows, 1}}));
  EXPECT_FALSE(family->select(std::array{g::ShapeBinding{f.rows, 0}}));
  EXPECT_FALSE(family->select(std::array{g::ShapeBinding{f.rows, 1}, g::ShapeBinding{f.rows, 1}}));
  EXPECT_FALSE(family->signature(2));
  EXPECT_EQ(factories, 2); // Selection cannot compile or grow the family.
}

TEST(ExactNativeVariants, WholeTableIsValidatedBeforeNativeFactory) {
  auto f = make_graph();
  const std::array one{g::ShapeBinding{f.rows, 1}}, invalid{g::ShapeBinding{f.rows, 5}};
  unsigned factories = 0;
  auto factory = [&](const g::Graph &point) -> hyle::Result<g::Graph> { ++factories; return point; };
  const std::array invalid_table{g::ExactShapeVariant{one}, g::ExactShapeVariant{invalid}};
  EXPECT_FALSE(g::ExactNativeVariants<g::Graph>::prepare(f.graph, invalid_table, factory));
  const std::array duplicate{g::ExactShapeVariant{one}, g::ExactShapeVariant{one}};
  EXPECT_FALSE(g::ExactNativeVariants<g::Graph>::prepare(f.graph, duplicate, factory));
  EXPECT_FALSE(g::ExactNativeVariants<g::Graph>::prepare(f.graph, {}, factory));
  EXPECT_EQ(factories, 0);
}

TEST(ExactNativeVariants, DerivedGuardAndSymbolicViewUseCommonBinding) {
  auto builder = g::GraphBuilder::create().value();
  auto n = builder.dimension({1, 4}).value();
  auto half = builder.dimension_quotient(g::Dimension(n), g::Dimension::fixed(2)).value();
  auto input = builder.input(g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(n), g::Dimension::fixed(8)}).value()).value();
  auto view = g::ViewSpec::reshape(std::array{g::Dimension(half), g::Dimension::fixed(16)}).value();
  auto output = builder.add(view, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  EXPECT_FALSE(g::specialize_exact(graph, std::array{g::ShapeBinding{n, 3}}));
  EXPECT_FALSE(g::specialize_exact(graph, std::array{g::ShapeBinding{half, 2}}));
  auto point = g::specialize_exact(graph, std::array{g::ShapeBinding{n, 4}});
  ASSERT_TRUE(point);
  EXPECT_EQ(point->graph.output_description()->dimensions()[0], 2);
  EXPECT_TRUE(point->graph.operation(point->graph.node_id().value()));
}

TEST(ExactNativeVariants, SnapshotPreservesControlOnlyPredecessor) {
  auto builder = g::GraphBuilder::create().value();
  auto desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{8}).value();
  auto input = builder.input(desc).value();
  auto first = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}), std::array{input}).value();
  auto last = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{input}).value();
  ASSERT_TRUE(builder.depends_on(last, first).ok());
  ASSERT_TRUE(builder.publish(builder.result(last).value()).ok());
  auto graph = std::move(builder).finish().value();
  auto point = g::specialize_exact(graph);
  ASSERT_TRUE(point);
  EXPECT_EQ(point->graph.node_id(0).value(), first);
  EXPECT_EQ(point->graph.node_dependency(last, 0).value(), first);
}

TEST(ExactNativeVariants, FailedFactoryReleasesEarlierArtifacts) {
  auto f = make_graph();
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  auto stats = m::StatsAllocator::create(m::system_allocator()).value();
  unsigned factories = 0;
  auto family = g::ExactNativeVariants<g::Graph>::prepare(f.graph, table,
      [&](const g::Graph &point) -> hyle::Result<g::Graph> {
        if (++factories == 2) return std::unexpected(hyle::Status::Unavailable("native preparation rejected"));
        return point;
      }, stats.get());
  EXPECT_FALSE(family);
  EXPECT_EQ(factories, 2);
  EXPECT_EQ(stats.snapshot().live_bytes, 0);
}

TEST(ExactNativeVariants, MetadataQuotaFailureDoesNotCallFactoryOrLoseLogicalGraph) {
  auto f = make_graph();
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  m::QuotaAllocator quota(m::system_allocator(), 1);
  unsigned factories = 0;
  auto family = g::ExactNativeVariants<g::Graph>::prepare(f.graph, table,
      [&](const g::Graph &point) -> hyle::Result<g::Graph> { ++factories; return point; }, quota.get());
  ASSERT_FALSE(family);
  EXPECT_TRUE(family.error().IsResourceExhausted());
  EXPECT_EQ(factories, 0);
  EXPECT_EQ(quota.used(), 0);
  EXPECT_EQ(f.graph.input_id().value(), f.input);
  EXPECT_TRUE(g::specialize_exact(f.graph, four));
}

TEST(NativeReachability, ControlChainKeepsDataAncestorsWithoutRevivingUnrelatedDeadNodes) {
  auto builder = g::GraphBuilder::create().value();
  auto desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{8}).value();
  auto input = builder.input(desc).value();
  auto ancestor = builder.add(c::Unary{c::UnaryKind::kAbs}, input).value();
  auto first = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{ancestor}).value();
  auto dead = builder.add(c::Unary{c::UnaryKind::kAbs}, input).value();
  (void)dead;
  auto last = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{input}).value();
  ASSERT_TRUE(builder.depends_on(last, first).ok());
  ASSERT_TRUE(builder.publish(builder.result(last).value()).ok());
  auto graph = std::move(builder).finish().value();
  auto analysis = g::detail::analyze_native_pure(graph);
  ASSERT_TRUE(analysis);
  EXPECT_TRUE(analysis->needed[0]); EXPECT_TRUE(analysis->needed[1]);
  EXPECT_FALSE(analysis->needed[2]); EXPECT_TRUE(analysis->needed[3]);
  EXPECT_TRUE(analysis->boundary[1]); EXPECT_TRUE(analysis->boundary[3]);
  EXPECT_TRUE(analysis->control_predecessor[1]);
  EXPECT_FALSE(analysis->control_predecessor[3]);
  m::QuotaAllocator quota(m::system_allocator(), 1);
  auto oom = g::detail::analyze_native_pure(graph, quota.get());
  ASSERT_FALSE(oom); EXPECT_TRUE(oom.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), 0);
}

TEST(NativeReachability, UnusedResourceDeclarationDoesNotCreateAnEffectCarrier) {
  auto builder=g::GraphBuilder::create().value();
  ASSERT_TRUE(builder.effect_resource());
  auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{8}).value();
  auto input=builder.input(desc).value();
  auto output=builder.add(c::Unary{c::UnaryKind::kNeg},input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph=std::move(builder).finish().value();
  auto analysis=g::detail::analyze_native_pure(graph);
  ASSERT_TRUE(analysis);
  EXPECT_TRUE(analysis->needed[0]);
}

struct MandatoryAlias {
  static constexpr c::OperationId operation_id{0x464f554e44415449ULL,0x414c494153544553ULL};
  static constexpr std::string_view operation_name="mandatory-alias-test";
  c::OperationTraits operation_traits()const noexcept {
    return {.inputs={1,1},.outputs=1,.alias=c::AliasPolicy::kMustAliasInput0,.materializes=false};
  }
  hyle::Status infer_output_descriptions(std::span<const c::TensorDesc> inputs,std::span<c::TensorDesc> outputs)const noexcept {
    if(inputs.size()!=1||outputs.size()!=1)return hyle::Status::InvalidArgument("alias test arity");
    outputs[0]=inputs[0];return hyle::Status::Ok();
  }
};
TEST(NativeReachability, MandatoryCustomAliasNeedsNativeQualificationWhileCpuKeepsOriginalAdmission) {
  auto builder=g::GraphBuilder::create().value();
  auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{8}).value();
  auto input=builder.input(desc).value();
  auto output=builder.add(c::Operation::custom(MandatoryAlias{}),input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph=std::move(builder).finish().value();
  auto native=g::detail::analyze_native_pure(graph);
  ASSERT_FALSE(native);EXPECT_TRUE(native.error().IsUnimplemented());
  const std::array<std::uint8_t,1> cpu{1};
  EXPECT_TRUE(g::detail::analyze_native_pure(graph,m::system_allocator(),cpu));
}
} // namespace
