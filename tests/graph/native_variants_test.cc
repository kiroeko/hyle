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

TEST(NativePartition, IndependentPrefixDoesNotInheritJoinWaitInEitherInsertionOrder) {
  for (const bool cpu_first : {false, true}) {
    auto builder = g::GraphBuilder::create().value();
    auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 1>{64}).value();
    auto x = builder.input(desc).value(), a = builder.input(desc).value();
    g::ValueId cpu, gpu;
    if (cpu_first) {
      cpu = builder.add(c::Unary{c::UnaryKind::kAbs}, x).value();
      gpu = builder.add(c::Unary{c::UnaryKind::kNeg}, a).value();
    } else {
      gpu = builder.add(c::Unary{c::UnaryKind::kNeg}, a).value();
      cpu = builder.add(c::Unary{c::UnaryKind::kAbs}, x).value();
    }
    auto join = builder.add(c::Binary{}, cpu, gpu).value();
    ASSERT_TRUE(builder.publish(join).ok());
    auto graph = std::move(builder).finish().value();
    auto analysis = g::detail::analyze_native_pure(graph).value();
    const std::array targets{g::ExecutionTargetId{cpu_first ? 1u : 0u},
        g::ExecutionTargetId{cpu_first ? 0u : 1u}, g::ExecutionTargetId{1}};
    auto partition = g::detail::partition_native_joint(graph, analysis, targets);
    ASSERT_TRUE(partition);
    ASSERT_EQ(partition->fragments.size(), 3);
    EXPECT_TRUE(partition->prerequisites(partition->fragments[0]).empty());
    EXPECT_TRUE(partition->prerequisites(partition->fragments[1]).empty());
    auto dependencies = partition->prerequisites(partition->fragments[2]);
    ASSERT_EQ(dependencies.size(), 2);
    EXPECT_NE(dependencies[0], dependencies[1]);
    EXPECT_LT(dependencies[0], 2); EXPECT_LT(dependencies[1], 2);
    EXPECT_EQ(partition->fragments[1].end, 2);
  }
}

TEST(NativePartition, ExistingEntryDependencyKeepsWholeNativeOrCpuChain) {
  auto builder = g::GraphBuilder::create().value();
  auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{64}).value();
  auto input = builder.input(desc).value();
  auto gpu = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  auto cpu = builder.add(c::Unary{c::UnaryKind::kAbs}, gpu).value();
  auto output = builder.add(c::Binary{}, cpu, gpu).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto analysis = g::detail::analyze_native_pure(graph).value();
  const std::array targets{g::ExecutionTargetId{0}, g::ExecutionTargetId{1},
      g::ExecutionTargetId{1}};
  auto partition = g::detail::partition_native_joint(graph, analysis, targets);
  ASSERT_TRUE(partition); ASSERT_EQ(partition->fragments.size(), 2);
  EXPECT_EQ(partition->fragments[1].begin, 1);
  EXPECT_EQ(partition->fragments[1].end, 3);
  auto dependencies = partition->prerequisites(partition->fragments[1]);
  ASSERT_EQ(dependencies.size(), 1); EXPECT_EQ(dependencies[0], 0);
  const std::array one_target{g::ExecutionTargetId{0}, g::ExecutionTargetId{0},
      g::ExecutionTargetId{0}};
  auto fused = g::detail::partition_native_joint(graph, analysis, one_target);
  ASSERT_TRUE(fused); EXPECT_EQ(fused->fragments.size(), 1);
}

TEST(NativePartition, ControlOnlyPredecessorSurvivesDeadValuesAndDeduplicatesDataEdge) {
  auto builder = g::GraphBuilder::create().value();
  auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{8}).value();
  auto input = builder.input(desc).value();
  auto first = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{input}).value();
  (void)builder.add(c::Unary{c::UnaryKind::kNeg}, input).value(); // Unrelated dead node.
  auto last = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{input}).value();
  ASSERT_TRUE(builder.depends_on(last, first).ok());
  auto join = builder.add_node(c::Operation(c::Binary{}),
      std::array{builder.result(first).value(), builder.result(last).value()}).value();
  ASSERT_TRUE(builder.depends_on(join, first).ok());
  ASSERT_TRUE(builder.publish(builder.result(join).value()).ok());
  auto graph = std::move(builder).finish().value();
  auto analysis = g::detail::analyze_native_pure(graph).value();
  const std::array targets{g::ExecutionTargetId{0}, g::ExecutionTargetId{0},
      g::ExecutionTargetId{1}, g::ExecutionTargetId{1}};
  auto partition = g::detail::partition_native_joint(graph, analysis, targets);
  ASSERT_TRUE(partition); ASSERT_EQ(partition->fragments.size(), 3);
  EXPECT_EQ(partition->fragments[0].end, 2); // Dead gap adds no fragment.
  auto control = partition->prerequisites(partition->fragments[1]);
  ASSERT_EQ(control.size(), 1); EXPECT_EQ(control[0], 0);
  auto join_dependencies = partition->prerequisites(partition->fragments[2]);
  ASSERT_EQ(join_dependencies.size(), 2); // first appears as both data and control.
  EXPECT_NE(join_dependencies[0], join_dependencies[1]);
}

TEST(NativePartition, MalformedInputsAndQuotaFailureKeepOriginalGraphAndReleaseMetadata) {
  auto builder = g::GraphBuilder::create().value();
  auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{8}).value();
  auto input = builder.input(desc).value();
  ASSERT_TRUE(builder.publish(builder.add(c::Unary{}, input).value()).ok());
  auto graph = std::move(builder).finish().value();
  auto analysis = g::detail::analyze_native_pure(graph).value();
  const std::array targets{g::ExecutionTargetId{0}};
  EXPECT_FALSE(g::detail::partition_native_joint(graph, analysis, {}));
  m::QuotaAllocator quota(m::system_allocator(), 1);
  auto failed = g::detail::partition_native_joint(graph, analysis, targets, quota.get());
  ASSERT_FALSE(failed); EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), 0);
  EXPECT_EQ(graph.input_id().value(), input);
  EXPECT_TRUE(g::detail::partition_native_joint(graph, analysis, targets));
}
} // namespace
