#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <vector>

#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace a = c::apple;
namespace m = hyle::mem;
constexpr g::ExecutionTargetId apple_target{0}, cpu_target{1};
constexpr g::StorageDomainId host_storage{0}, metal_storage{1};

hyle::Result<c::Context> cpu_context() {
  return g::cpu::create_context();
}
hyle::Result<std::vector<g::QueueBinding>> cpu_queues(
    const g::CompileReport &report, c::Queue &queue) {
  std::vector<g::QueueBinding> result;
  for (std::size_t index = 0; index < report.lane_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(index));
    if (!lane.queue_required) continue;
    if (lane.target != cpu_target || lane.runtime_mode != c::ProviderRuntimeMode::kSharedQueue)
      return std::unexpected(hyle::Status::Internal("mixed graph requested a synthetic native Queue"));
    result.push_back({lane.lane, &queue});
  }
  return result;
}

TEST(AppleMixedRegion, FiniteCpuAppleCpuKeepsTwoSlotsAndStrictReplay) {
  auto apple = a::Context::create().value();
  auto memory = a::MemoryResource::create(apple).value();
  auto cpu = cpu_context().value();
  auto queue = cpu.create_queue().value();
  const std::array domains{g::StorageDomain{m::system_resource()}, g::StorageDomain{memory.ref()}};
  const std::array targets{
      g::ExecutionTarget{g::apple::execution_source(apple).value(), metal_storage},
      g::ExecutionTarget{g::ExecutionSource::acquire(cpu).value(), host_storage}};
  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4}).value();
  const auto spec = g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(8)}).value();
  const auto input = builder.input(spec).value();
  const auto first = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{input}, std::array{spec}).value();
  const auto host_value = builder.result(first).value();
  const auto middle = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{host_value}, std::array{spec}).value();
  const auto generated = builder.result(middle).value();
  const auto last = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{generated}, std::array{spec}).value();
  const auto output = builder.result(last).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const std::array small{g::ShapeBinding{rows, 1}}, large{g::ShapeBinding{rows, 4}};
  const std::array points{g::ExactShapeVariant{small}, g::ExactShapeVariant{large}};
  const std::array placements{
      g::NodePlacement{first, cpu_target, g::PlacementStrength::kRequired},
      g::NodePlacement{middle, apple_target, g::PlacementStrength::kRequired},
      g::NodePlacement{last, cpu_target, g::PlacementStrength::kRequired}};
  const std::array locality{g::ValuePlacement{host_value, host_storage, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{input, host_storage},
                            g::ExternalResourceContract{output, host_storage}};
  auto compiled = g::compile(graph, {domains, targets},
      {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = points,
       .node_placements = placements, .value_placements = locality, .external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.exact_variant_count(), 2U);
  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    const auto shape = point_index == 0 ? std::span<const g::ShapeBinding>(small)
                                      : std::span<const g::ShapeBinding>(large);
    const auto point = g::specialize_exact(graph, shape).value();
    const auto description = point.graph.input_description().value();
    const auto bytes = description.required_bytes().value();
    const auto report = compiled->report.exact_variant(point_index).value();
    const auto queues = cpu_queues(report, queue).value();
    std::array<g::Invocation, 2> slots;
    std::array<m::Buffer, 2> sources, destinations;
    for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
      slots[slot_index] = compiled->executable.create_invocation({.shape = shape}).value();
      sources[slot_index] = m::Buffer::create(m::system_resource(), bytes, 64).value();
      destinations[slot_index] = m::Buffer::create(m::system_resource(), bytes, 64).value();
      ASSERT_TRUE(slots[slot_index].bind(input,
          c::ConstTensorView::create(description, sources[slot_index].view()).value()).ok());
      ASSERT_TRUE(slots[slot_index].bind(output,
          c::TensorView::create(description, destinations[slot_index].view()).value()).ok());
      EXPECT_FALSE(slots[slot_index].prepare_replay(queues, {.host_wait = g::HostWaitPolicy::kReject}).ok());
      ASSERT_TRUE(slots[slot_index].prepare_replay(queues).ok());
      EXPECT_GT(slots[slot_index].replay_report().value().host_wait_dependency_count, 0U);
    }
    std::vector<float> values(bytes / sizeof(float)), observed(values.size());
    for (int phase = 0; phase < 2; ++phase) {
      std::array<g::GraphOutput, 2> retained;
      for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
        for (std::size_t index = 0; index < values.size(); ++index)
          values[index] = 0.5f * (float(index) - 7.0f) * float(1 + phase + slot_index);
        values[0] = phase == 0 ? 0.0f : -0.0f;
        ASSERT_TRUE(sources[slot_index].view().write(std::as_bytes(std::span(values))).ok());
        {
          auto accepted = slots[slot_index].submit(queues);
          ASSERT_TRUE(accepted) << accepted.error().message();
          ASSERT_TRUE(accepted->wait().ok());
        }
        retained[slot_index] = slots[slot_index].result(output).value();
        EXPECT_FALSE(slots[slot_index].reset().ok());
      }
      // Read both results only after the other original slot has also run.
      for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
        ASSERT_TRUE(retained[slot_index].tensor()->storage().read(
            std::as_writable_bytes(std::span(observed))).ok());
        for (std::size_t index = 0; index < observed.size(); ++index) {
          const float original = index == 0 ? (phase == 0 ? 0.0f : -0.0f)
              : 0.5f * (float(index) - 7.0f) * float(1 + phase + slot_index);
          EXPECT_EQ(std::bit_cast<std::uint32_t>(observed[index]),
                    std::bit_cast<std::uint32_t>(std::abs(original)));
        }
      }
    } // Same fixed bindings/preparation, two real producer epochs.
  }
}

TEST(AppleMixedRegion, CheckedProducerFailureDoesNotWriteCpuSuccessor) {
  auto apple = a::Context::create().value();
  auto memory = a::MemoryResource::create(apple).value();
  auto cpu = cpu_context().value();
  auto queue = cpu.create_queue().value();
  const std::array domains{g::StorageDomain{m::system_resource()}, g::StorageDomain{memory.ref()}};
  const std::array targets{
      g::ExecutionTarget{g::apple::execution_source(apple).value(), metal_storage},
      g::ExecutionTarget{g::ExecutionSource::acquire(cpu).value(), host_storage}};
  const auto table_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 8}).value();
  const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto table = builder.input(table_desc).value(), ids = builder.input(ids_desc).value();
  const auto gather = builder.add_node(c::Operation(c::Gather{0, c::IndexBoundsPolicy::kChecked}),
      std::array{table, ids}).value();
  const auto chosen = builder.result(gather).value();
  const auto last = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{chosen}).value();
  const auto output = builder.result(last).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const auto output_desc = graph.value_description(output).value();
  const std::array placements{g::NodePlacement{gather, apple_target, g::PlacementStrength::kRequired},
                              g::NodePlacement{last, cpu_target, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{table, host_storage},
      g::ExternalResourceContract{ids, host_storage}, g::ExternalResourceContract{output, host_storage}};
  auto compiled = g::compile(graph, {domains, targets}, {.node_placements = placements, .external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto queues = cpu_queues(compiled->report, queue).value();
  auto table_bytes = m::Buffer::create(m::system_resource(), table_desc.required_bytes().value(), 64).value();
  auto id_bytes = m::Buffer::create(m::system_resource(), ids_desc.required_bytes().value(), 64).value();
  auto destination = m::Buffer::create(m::system_resource(), output_desc.required_bytes().value(), 64).value();
  const std::array<float, 16> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  ASSERT_TRUE(table_bytes.view().write(std::as_bytes(std::span(data))).ok());
  for (const auto id : {2, 1}) {
    auto slot = compiled->executable.create_invocation().value();
    const std::array indices{id};
    const std::array<float, 8> sentinel{99,99,99,99,99,99,99,99};
    ASSERT_TRUE(id_bytes.view().write(std::as_bytes(std::span(indices))).ok());
    ASSERT_TRUE(destination.view().write(std::as_bytes(std::span(sentinel))).ok());
    ASSERT_TRUE(slot.bind(table, c::ConstTensorView::create(table_desc, table_bytes.view()).value()).ok());
    ASSERT_TRUE(slot.bind(ids, c::ConstTensorView::create(ids_desc, id_bytes.view()).value()).ok());
    ASSERT_TRUE(slot.bind(output, c::TensorView::create(output_desc, destination.view()).value()).ok());
    ASSERT_TRUE(slot.prepare_replay(queues).ok());
    auto accepted = slot.submit(queues);
    ASSERT_TRUE(accepted) << accepted.error().message();
    const auto terminal = accepted->wait();
    std::array<float, 8> observed{};
    ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(observed))).ok());
    if (id == 2) {
      EXPECT_TRUE(terminal.IsOutOfRange()) << terminal.message();
      EXPECT_FALSE(slot.result(output));
      EXPECT_EQ(observed, sentinel);
    } else {
      ASSERT_TRUE(terminal.ok()) << terminal.message();
      for (std::size_t index = 0; index < observed.size(); ++index)
        EXPECT_EQ(observed[index], -data[8 + index]);
    }
  }
}
} // namespace
