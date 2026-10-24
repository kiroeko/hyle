#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;

c::TensorDesc description(std::int64_t count) {
  return c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 5>{1, 1, 1, 1, count}).value();
}

struct Tensor {
  hyle::mem::Buffer buffer;
  c::TensorView view;
  explicit Tensor(c::TensorDesc desc)
      : buffer(hyle::mem::Buffer::create(hyle::mem::system_resource(),
            desc.required_bytes().value(), 64).value()),
        view(c::TensorView::create(desc, buffer.view()).value()) {}
};

c::Context context() {
  const std::array providers{
      c::ProviderBinding{c::aten_cpu::provider(), 100}};
  return c::Context::create(c::cpu::device(), {.providers = providers}).value();
}

std::array<float, 2> read_pair(c::ConstTensorView tensor) {
  std::array<float, 2> values{};
  EXPECT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(values))).ok());
  return values;
}

TEST(GraphAliasLiveness, LaterOverlappingViewKeepsItsOriginalValues) {
  auto ctx = context();
  auto prepared = c::prepare(ctx, c::Unary{c::UnaryKind::kNeg}, description(2));
  ASSERT_TRUE(prepared) << prepared.error().message();
  ASSERT_TRUE(prepared->alias_support(0, 0).value().supported);
  for (bool view_before_write : {false, true}) {
    SCOPED_TRACE(view_before_write);
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(description(4)).value();
    const auto root = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
    const auto first = builder.add(c::View::slice(4, 0, 4).value(), root).value();
    g::ValueId later;
    if (view_before_write)
      later = builder.add(c::View::slice(4, 0, 2).value(), root).value();
    const auto changed = builder.add(c::Unary{c::UnaryKind::kNeg}, first).value();
    if (!view_before_write)
      later = builder.add(c::View::slice(4, 0, 2).value(), root).value();
    const auto changed_part = builder.add(c::View::slice(4, 1, 2).value(), changed).value();
    const auto output = builder.add(c::Binary{c::BinaryKind::kAdd}, changed_part, later).value();
    ASSERT_TRUE(builder.publish(output).ok());
    const auto graph = std::move(builder).finish().value();
    auto compiled = g::compile(graph, ctx);
    ASSERT_TRUE(compiled) << compiled.error().message();
    auto run = compiled->executable.create_invocation().value();
    Tensor source(description(4)), destination(description(2));
    const std::array values{1.F, 2.F, 3.F, 4.F};
    ASSERT_TRUE(source.buffer.view().write(std::as_bytes(std::span(values))).ok());
    ASSERT_TRUE(run.bind(input, source.view).ok());
    ASSERT_TRUE(run.bind(output, destination.view).ok());
    auto queue = ctx.create_queue().value();
    for (int replay = 0; replay < 2; ++replay) {
      auto completion = run.submit(queue);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      EXPECT_EQ(read_pair(destination.view.as_const()), (std::array{1.F, 1.F}));
    }
  }
}

TEST(GraphAliasLiveness, PublishedPrivateViewSurvivesLaterDestructiveCandidates) {
  auto ctx = context();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description(4)).value();
  const auto root = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  const auto kept = builder.add(c::View::slice(4, 1, 2).value(), root).value();
  const auto changed = builder.add(c::Unary{c::UnaryKind::kNeg}, root).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, changed).value();
  ASSERT_TRUE(builder.publish(kept).ok());
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = g::compile(graph, ctx);
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto run = compiled->executable.create_invocation().value();
  Tensor source(description(4)), destination(description(4));
  const std::array values{1.F, 2.F, 3.F, 4.F};
  ASSERT_TRUE(source.buffer.view().write(std::as_bytes(std::span(values))).ok());
  ASSERT_TRUE(run.bind(input, source.view).ok());
  ASSERT_TRUE(run.bind(output, destination.view).ok());
  auto queue = ctx.create_queue().value();
  auto completion = run.submit(queue);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  auto result = run.output(kept);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(read_pair(*result), (std::array{-2.F, -3.F}));
}

TEST(GraphAliasLiveness, LastReaderStillAllowsInternalInPlaceReuse) {
  auto ctx = context();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description(4)).value();
  const auto first = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  const auto second = builder.add(c::Unary{c::UnaryKind::kNeg}, first).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, second).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = g::compile(graph, ctx);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.workspace_bytes(), description(4).required_bytes().value());
  auto run = compiled->executable.create_invocation().value();
  Tensor source(description(4)), destination(description(4));
  const std::array values{1.F, 2.F, 3.F, 4.F};
  ASSERT_TRUE(source.buffer.view().write(std::as_bytes(std::span(values))).ok());
  ASSERT_TRUE(run.bind(input, source.view).ok());
  ASSERT_TRUE(run.bind(output, destination.view).ok());
  auto queue = ctx.create_queue().value();
  ASSERT_TRUE(run.submit(queue)->wait().ok());
  std::array<float, 4> actual{};
  ASSERT_TRUE(destination.buffer.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array{-1.F, -2.F, -3.F, -4.F}));
}

TEST(GraphAliasLiveness, UnorderedEarlierLaneReaderPreventsDestructiveAlias) {
  auto reader_context = context(), writer_context = context();
  const auto desc = description(4);
  auto prepared = c::prepare(writer_context, c::Unary{c::UnaryKind::kNeg}, desc);
  ASSERT_TRUE(prepared) << prepared.error().message();
  ASSERT_TRUE(prepared->alias_support(0, 0).value().supported);
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(desc).value();
  const auto seed_input = builder.input(desc).value();
  // Seed first: scheduling then stays on the reader target for P and R.
  // R's extra seed prerequisite splits its segment from P's prefix, so W
  // depending on P does not implicitly wait for R's read of the same bytes.
  const auto seed = builder.add(c::Unary{c::UnaryKind::kNeg}, seed_input).value();
  const auto root = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  const auto earlier_read = builder.add(c::Binary{c::BinaryKind::kAdd}, root, seed).value();
  const auto changed = builder.add(c::Unary{c::UnaryKind::kNeg}, root).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, changed).value();
  ASSERT_TRUE(builder.publish(earlier_read).ok());
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array storage{g::StorageDomain{hyle::mem::system_resource()}};
  const std::array targets{
      g::ExecutionTarget{g::ExecutionSource::acquire(reader_context).value(), g::StorageDomainId{0}},
      g::ExecutionTarget{g::ExecutionSource::acquire(writer_context).value(), g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{g::ExecutionTargetId{0}},
                         g::ExecutionLane{g::ExecutionTargetId{1}}};
  const std::array placements{
      g::NodePlacement{graph.node_id(0).value(), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(1).value(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(2).value(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(3).value(), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(4).value(), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired}};
  auto compiled = g::compile(graph,
      {.storage_domains = storage, .targets = targets, .lanes = lanes},
      {.node_placements = placements, .region_selection = g::RegionSelectionPolicy::kDisabled});
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto &report = compiled->report;
  ASSERT_EQ(report.step_count(), 5u);
  ASSERT_EQ(report.copy_bytes(), 0u);
  EXPECT_EQ(report.serialization_edge_count(), 0u);
  std::array<std::size_t, 5> steps;
  steps.fill(report.step_count());
  for (std::size_t step = 0; step < report.step_count(); ++step) {
    auto node = report.step_node(step, 0);
    ASSERT_TRUE(node) << node.error().message();
    ASSERT_LT(node->ordinal(), steps.size());
    steps[node->ordinal()] = step;
  }
  for (const auto step : steps) ASSERT_LT(step, report.step_count());
  const auto producer = steps[1], reader = steps[2], writer = steps[3];
  ASSERT_LT(reader, writer); // This scalar order is not a retirement proof.
  ASSERT_NE(report.step(reader)->lane, report.step(writer)->lane);
  std::vector<std::size_t> segments(report.step_count(), report.segment_count());
  for (std::size_t index = 0; index < report.segment_count(); ++index) {
    const auto segment = report.segment(index).value();
    for (std::size_t step = segment.step_begin;
         step < segment.step_begin + segment.step_count; ++step)
      segments[step] = index;
  }
  ASSERT_LT(segments[reader], report.segment_count());
  ASSERT_LT(segments[writer], report.segment_count());
  ASSERT_NE(segments[producer], segments[reader]);
  ASSERT_NE(segments[reader], segments[writer]);
  // Prove that no direct or transitive Plan edge makes R retire before W.
  std::vector<bool> ancestors(report.segment_count());
  ancestors[segments[writer]] = true;
  for (std::size_t index = report.segment_count(); index > 0; --index) {
    const auto segment = index - 1;
    if (!ancestors[segment]) continue;
    const auto description = report.segment(segment).value();
    for (std::size_t edge = 0; edge < description.dependency_count; ++edge) {
      auto dependency = report.segment_dependency(segment, edge);
      ASSERT_TRUE(dependency) << dependency.error().message();
      ASSERT_LT(dependency->source_segment, segment);
      ancestors[dependency->source_segment] = true;
    }
  }
  ASSERT_FALSE(ancestors[segments[reader]]);
  const auto root_instance = report.step_output(producer, 0).value();
  ASSERT_EQ(report.step_input(reader, 0).value(), root_instance);
  ASSERT_EQ(report.step_input(writer, 0).value(), root_instance);
  const auto changed_instance = report.step_output(writer, 0).value();
  const auto physical = report.physical_value(changed_instance).value();
  EXPECT_TRUE(physical.workspace_backed);
  EXPECT_FALSE(physical.alias_source.has_value());
  EXPECT_GE(report.workspace_bytes(), 2 * desc.required_bytes().value());
}
} // namespace
