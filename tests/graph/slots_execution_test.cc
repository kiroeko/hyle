#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/result.h"
#include "hyle/graph/slots.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;

constexpr std::size_t kItemBytes = 4 * sizeof(float);
constexpr std::size_t kStride = 64;
constexpr std::array<float, 4> kSentinel{101, 103, 107, 109};
constexpr std::array<float, 4> kUntouched{-1, -2, -3, -4};

c::TensorDesc item_description() {
  return c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
}

// A bare CPU Context deliberately has no numerical provider. The ordinary CPU
// Graph route must supply this external store without special consumer routing.
struct Program final {
  c::Context context = c::Context::create(c::cpu::device()).value();
  c::Queue queue = context.create_queue().value();
  c::TensorDesc rows_description, ids_description;
  g::ValueId rows, ids;
  g::NodeId store;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  const std::array<g::QueueBinding, 1> queues{{{g::ExecutionLaneId{0}, &queue}}};

  explicit Program(std::int64_t count = 2) {
    rows_description = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{count, 4}).value();
    ids_description = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{count}).value();
    auto builder = g::GraphBuilder::create().value();
    rows = builder.input(rows_description).value();
    ids = builder.input(ids_description).value();
    const auto schema = s::Schema::create(item_description(), 3, kStride).value();
    store = builder.add_node(c::Operation::custom(s::Store{schema}),
        std::array{rows, ids}).value();
    // Publishing the ordinary source also exercises the original Graph result
    // lifetime, independently of the external destination transaction.
    EXPECT_TRUE(builder.publish(rows).ok());
    graph = std::move(builder).finish().value();
    compiled = g::cpu::compile(graph, context).value();
    invocation = compiled.executable.create_invocation().value();
  }
};

struct Storage final {
  c::LeasePool pool = c::LeasePool::create(m::system_resource(),
      3 * kStride, {.max_blocks = 3}).value();
  c::LeaseSlotLayout layout = c::LeaseSlotLayout::create(pool,
      item_description(), 3, kStride).value();
  std::array<c::WriteLease, 2> destinations;
  c::ReadLease sentinel;

  Storage() {
    destinations[0] = pool.allocate(kItemBytes, kStride).value();
    auto middle = pool.allocate(kItemBytes, kStride).value();
    destinations[1] = pool.allocate(kItemBytes, kStride).value();
    EXPECT_EQ(layout.locate(destinations[0])->index, 0u);
    EXPECT_EQ(layout.locate(middle)->index, 1u);
    EXPECT_EQ(layout.locate(destinations[1])->index, 2u);
    EXPECT_TRUE(middle.view()->write(std::as_bytes(std::span(kSentinel))).ok());
    sentinel = middle.publish().value();
    for (auto &destination : destinations)
      EXPECT_TRUE(destination.view()->write(std::as_bytes(std::span(kUntouched))).ok());
  }
};

std::array<float, 4> read_item(const c::ReadLease &ready) {
  std::array<float, 4> result{};
  EXPECT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

std::array<float, 4> read_item(const c::WriteLease &writer) {
  std::array<float, 4> result{};
  EXPECT_TRUE(writer.view()->read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

TEST(GraphSlotsExecution, RealRetirementAndResultsGuardTakeAndRebindingUpdatesOrder) {
  Program program;
  Storage storage;
  auto rows = m::Buffer::create(m::system_resource(), 2 * kItemBytes, 64).value();
  constexpr std::array<float, 8> first{1, 2, 3, 4, 11, 12, 13, 14};
  ASSERT_TRUE(rows.view().write(std::as_bytes(std::span(first))).ok());
  const auto tensor = c::ConstTensorView::create(program.rows_description,
      rows.view()).value();
  ASSERT_TRUE(program.invocation.bind(program.rows, tensor).ok());
  auto bindings = s::SlotBindings::create(storage.layout, {},
      storage.destinations).value();
  const std::array original_generations{bindings.destinations()[0].generation(),
      bindings.destinations()[1].generation()};

  EXPECT_TRUE(s::bind(program.invocation, program.graph, program.store,
      std::move(bindings)).IsFailedPrecondition());
  ASSERT_TRUE(bindings.valid());
  EXPECT_EQ(bindings.destinations()[0].generation(), original_generations[0]);
  EXPECT_EQ(bindings.destinations()[1].generation(), original_generations[1]);
  ASSERT_TRUE(s::prepare(program.invocation, program.queues).ok());
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store,
      std::move(bindings)).ok());
  EXPECT_FALSE(bindings.valid());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  auto done = program.invocation.submit(program.queues);
  ASSERT_TRUE(done) << done.error().message();
  // Inline CPU work has finished, but its unobserved public Completion still
  // owns the original observation slot. take must not steal that episode.
  auto unobserved = s::take(program.invocation, program.store);
  ASSERT_FALSE(unobserved);
  EXPECT_TRUE(unobserved.error().IsUnavailable());
  ASSERT_TRUE(done->wait().ok());
  *done = {};
  {
    auto result = program.invocation.result(program.rows).value();
    std::array<float, 8> actual{};
    ASSERT_TRUE(result.tensor()->storage().read(
        std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, first);
    auto observed = s::take(program.invocation, program.store);
    ASSERT_FALSE(observed);
    EXPECT_TRUE(observed.error().IsUnavailable());
  }
  auto taken = s::take(program.invocation, program.store);
  ASSERT_TRUE(taken) << taken.error().message();
  bindings = std::move(*taken);
  EXPECT_EQ(bindings.destinations()[0].generation(), original_generations[0]);
  EXPECT_EQ(bindings.destinations()[1].generation(), original_generations[1]);
  auto stale = program.invocation.submit(program.queues);
  ASSERT_FALSE(stale);

  auto low = bindings.destinations()[0].publish().value();
  auto high = bindings.destinations()[1].publish().value();
  EXPECT_EQ(read_item(low), (std::array<float, 4>{1, 2, 3, 4}));
  EXPECT_EQ(read_item(high), (std::array<float, 4>{11, 12, 13, 14}));
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
  bindings = {};

  // Reuse the same two allocations with the writer order reversed. The
  // original fixed control carrier must describe this new episode, rather
  // than retaining the first episode's destination order or reader owner.
  std::array<c::WriteLease, 2> reversed;
  reversed[0] = high.try_write().value();
  reversed[1] = low.try_write().value();
  bindings = s::SlotBindings::create(storage.layout, {}, reversed).value();
  EXPECT_EQ(bindings.destination_addresses()[0].index, 2u);
  EXPECT_EQ(bindings.destination_addresses()[1].index, 0u);
  constexpr std::array<float, 8> second{21, 22, 23, 24, 31, 32, 33, 34};
  ASSERT_TRUE(rows.view().write(std::as_bytes(std::span(second))).ok());
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store,
      std::move(bindings)).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  done = program.invocation.submit(program.queues);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  *done = {};
  taken = s::take(program.invocation, program.store);
  ASSERT_TRUE(taken) << taken.error().message();
  bindings = std::move(*taken);
  high = bindings.destinations()[0].publish().value();
  low = bindings.destinations()[1].publish().value();
  EXPECT_EQ(read_item(high), (std::array<float, 4>{21, 22, 23, 24}));
  EXPECT_EQ(read_item(low), (std::array<float, 4>{31, 32, 33, 34}));
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
}

TEST(GraphSlotsExecution, ReplacedPrivateIdsAreRejectedWithoutPublishingOrWriting) {
  Program program;
  Storage storage;
  auto rows = m::Buffer::create(m::system_resource(), 2 * kItemBytes, 64).value();
  constexpr std::array<float, 8> values{1, 2, 3, 4, 11, 12, 13, 14};
  ASSERT_TRUE(rows.view().write(std::as_bytes(std::span(values))).ok());
  ASSERT_TRUE(program.invocation.bind(program.rows, c::ConstTensorView::create(
      program.rows_description, rows.view()).value()).ok());
  ASSERT_TRUE(s::prepare(program.invocation, program.queues).ok());
  auto bindings = s::SlotBindings::create(storage.layout, {},
      storage.destinations).value();
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store,
      std::move(bindings)).ok());

  auto arbitrary = m::Buffer::create(m::system_resource(), 2 * sizeof(std::int32_t), 16).value();
  constexpr std::array<std::int32_t, 2> invalid_ids{1, 1};
  ASSERT_TRUE(arbitrary.view().write(std::as_bytes(std::span(invalid_ids))).ok());
  const auto replacement = c::ConstTensorView::create(program.ids_description,
      arbitrary.view()).value();
  const auto replaced = program.invocation.bind(program.ids, replacement);
  if (replaced.ok()) {
    const auto rejected = program.invocation.prepare_replay(program.queues);
    EXPECT_TRUE(rejected.IsInvalidArgument() || rejected.IsFailedPrecondition());
  } else {
    EXPECT_TRUE(replaced.IsFailedPrecondition());
  }
  auto taken = s::take(program.invocation, program.store);
  ASSERT_TRUE(taken) << taken.error().message();
  ASSERT_EQ(taken->destination_count(), 2u);
  for (const auto &writer : taken->destinations()) {
    EXPECT_TRUE(writer.valid());
    EXPECT_EQ(read_item(writer), kUntouched);
  }
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
}

TEST(GraphSlotsExecution, SourceDestinationOverlapIsRejectedBeforeAnyStore) {
  Program program(1);
  Storage storage;
  const auto overlapping_rows = c::ConstTensorView::create(
      program.rows_description, storage.destinations[0].view().value()).value();
  ASSERT_TRUE(s::prepare(program.invocation, program.queues).ok());
  auto bindings = s::SlotBindings::create(storage.layout, {},
      std::span(storage.destinations).first(1)).value();
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store,
      std::move(bindings)).ok());
  const auto bound = program.invocation.bind(program.rows, overlapping_rows);
  if (bound.ok()) {
    const auto rejected = program.invocation.prepare_replay(program.queues);
    EXPECT_TRUE(rejected.IsInvalidArgument() || rejected.IsFailedPrecondition());
  } else {
    EXPECT_TRUE(bound.IsInvalidArgument() || bound.IsFailedPrecondition());
  }
  auto taken = s::take(program.invocation, program.store);
  ASSERT_TRUE(taken) << taken.error().message();
  ASSERT_EQ(taken->destination_count(), 1u);
  EXPECT_TRUE(taken->destinations()[0].valid());
  EXPECT_EQ(read_item(taken->destinations()[0]), kUntouched);
  EXPECT_EQ(read_item(storage.destinations[1]), kUntouched);
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
}

TEST(GraphSlotsExecution, ResetRevokesTheActualWriteEpisodeAndReturnsItsOwner) {
  Program program;
  Storage storage;
  auto rows = m::Buffer::create(m::system_resource(), 2 * kItemBytes, 64).value();
  constexpr std::array<float, 8> values{1, 2, 3, 4, 11, 12, 13, 14};
  ASSERT_TRUE(rows.view().write(std::as_bytes(std::span(values))).ok());
  const auto input = c::ConstTensorView::create(program.rows_description,
      rows.view()).value();
  ASSERT_TRUE(program.invocation.bind(program.rows, input).ok());
  ASSERT_TRUE(s::prepare(program.invocation, program.queues).ok());
  auto bindings = s::SlotBindings::create(storage.layout, {},
      storage.destinations).value();
  const std::array generations{bindings.destinations()[0].generation(),
      bindings.destinations()[1].generation()};
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store,
      std::move(bindings)).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());

  // Borrow the real prepared I32 carrier, without making or retaining any
  // lease authority. Rebinding exactly these same bytes after reset must not
  // resurrect the write episode that reset withdrew from the original Graph.
  c::ConstTensorView original_ids;
  ASSERT_TRUE(g::detail::RegionAccess::visit_bound(program.invocation,
      +[](void *opaque, const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
        auto &ids = *static_cast<c::ConstTensorView *>(opaque);
        for (const auto &input : view.inputs)
          if (input.description().data_type() == c::DataType::kInt32 &&
              input.description().rank() == 1)
            ids = input;
        return hyle::Status::Ok();
      }, &original_ids).ok());
  ASSERT_TRUE(original_ids.valid());
  ASSERT_TRUE(program.invocation.reset().ok());
  ASSERT_TRUE(program.invocation.bind(program.rows, input).ok());
  ASSERT_TRUE(program.invocation.bind(program.ids, original_ids).ok());
  EXPECT_FALSE(program.invocation.prepare_replay(program.queues).ok());
  auto stale = program.invocation.submit(program.queues);
  ASSERT_FALSE(stale);
  auto recovered = s::take(program.invocation, program.store);
  ASSERT_TRUE(recovered) << recovered.error().message();
  ASSERT_EQ(recovered->destination_count(), 2u);
  for (std::size_t i = 0; i < recovered->destination_count(); ++i) {
    EXPECT_EQ(recovered->destinations()[i].generation(), generations[i]);
    EXPECT_EQ(read_item(recovered->destinations()[i]), kUntouched);
  }
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
}

// A single logical Graph is used with the ordinary native Gather and fallback
// Store. Only the real initialized source items are readable; the [P,E] pool
// description does not manufacture initialized storage for the other slots.
struct SourceProgram final {
  c::Context context;
  c::Queue queue;
  s::Schema schema;
  c::TensorDesc ids_description;
  g::ValueId pool, source_ids, destination_ids, gathered;
  g::NodeId gather, store;
  g::Graph graph;
  g::CompiledGraph compiled;
  m::Buffer gathered_buffer;
  g::Invocation invocation;
  const std::array<g::QueueBinding, 1> queues{{{g::ExecutionLaneId{0}, &queue}}};

  explicit SourceProgram(std::size_t stride, bool defaults = false)
      : context(defaults ? g::cpu::create_context({.native_threads = 2}).value()
                         : c::Context::create(c::cpu::device(), c::native::provider()).value()),
        queue(context.create_queue().value()) {
    schema = s::Schema::create(item_description(), 5, stride).value();
    ids_description = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{2}).value();
    auto builder = g::GraphBuilder::create().value();
    pool = builder.input(schema.capacity_description().value()).value();
    source_ids = builder.input(ids_description).value();
    destination_ids = builder.input(ids_description).value();
    gather = builder.add_node(c::Operation(c::Gather{.axis = 0,
        .bounds = c::IndexBoundsPolicy::kChecked}), std::array{pool, source_ids}).value();
    gathered = builder.result(gather, 0).value();
    store = builder.add_node(c::Operation::custom(s::Store{schema}),
        std::array{gathered, destination_ids}).value();
    EXPECT_TRUE(builder.publish(gathered).ok());
    graph = std::move(builder).finish().value();
    compiled = g::cpu::compile(graph, context).value();
    invocation = compiled.executable.create_invocation().value();
    const auto description = graph.value_description(gathered).value();
    gathered_buffer = m::Buffer::create(m::system_resource(), description.required_bytes().value(), 64).value();
    EXPECT_TRUE(invocation.bind(gathered,
        c::TensorView::create(description, gathered_buffer.view()).value()).ok());
  }
};

struct SourceStorage final {
  c::LeasePool pool;
  c::LeaseSlotLayout layout;
  std::array<c::ReadLease, 2> sources;
  std::array<c::WriteLease, 2> destinations;
  c::ReadLease sentinel;
  explicit SourceStorage(std::size_t stride) {
    pool = c::LeasePool::create(m::system_resource(), 5 * stride,
        {.max_blocks = 5, .backing_alignment = 64}).value();
    layout = c::LeaseSlotLayout::create(pool, item_description(), 5, stride).value();
    std::array<c::WriteLease, 5> blocks;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      blocks[i] = pool.allocate(kItemBytes, stride).value();
      EXPECT_EQ(layout.locate(blocks[i])->index, i);
    }
    constexpr std::array<float, 4> first{1, 2, 3, 4}, second{11, 12, 13, 14};
    EXPECT_TRUE(blocks[1].view()->write(std::as_bytes(std::span(first))).ok());
    EXPECT_TRUE(blocks[3].view()->write(std::as_bytes(std::span(second))).ok());
    EXPECT_TRUE(blocks[2].view()->write(std::as_bytes(std::span(kSentinel))).ok());
    sources[0] = blocks[1].publish().value(); sources[1] = blocks[3].publish().value();
    sentinel = blocks[2].publish().value();
    destinations[0] = std::move(blocks[0]); destinations[1] = std::move(blocks[4]);
    for (auto &writer : destinations)
      EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(kUntouched))).ok());
  }
};

void exercise_source_gather(std::size_t stride, bool defaults = false) {
  SourceProgram program(stride, defaults);
  if (defaults) {
    bool selected = false;
    for (std::size_t step = 0; step < program.compiled.report.step_count(); ++step)
      for (std::size_t node = 0; node < program.compiled.report.step(step)->node_count; ++node)
        if (program.compiled.report.step_node(step, node).value() == program.gather) {
          EXPECT_EQ(program.compiled.report.step(step)->provider.id, c::native::kProviderId);
          selected = true;
        }
    ASSERT_TRUE(selected);
  }
  SourceStorage storage(stride);
  const std::array associations{s::SourceBinding{program.store, program.gather}};
  ASSERT_TRUE(s::prepare(program.invocation, program.graph, program.queues, associations).ok());
  auto bindings = s::SlotBindings::create(storage.layout, storage.sources,
      storage.destinations).value();
  ASSERT_EQ(bindings.source_addresses()[0].index, 1u);
  ASSERT_EQ(bindings.source_addresses()[1].index, 3u);
  ASSERT_EQ(bindings.destination_addresses()[0].index, 0u);
  ASSERT_EQ(bindings.destination_addresses()[1].index, 4u);
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store, std::move(bindings)).ok());
  const auto prepared = program.invocation.prepare_replay(program.queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  auto done = program.invocation.submit(program.queues);
  ASSERT_TRUE(done) << done.error().message();
  EXPECT_FALSE(s::take(program.invocation, program.store));
  ASSERT_TRUE(done->wait().ok()); *done = {};
  {
    auto output = program.invocation.result(program.gathered).value();
    std::array<float, 8> actual{};
    ASSERT_TRUE(output.tensor()->storage().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, (std::array<float, 8>{1, 2, 3, 4, 11, 12, 13, 14}));
    EXPECT_FALSE(s::take(program.invocation, program.store));
  }
  auto taken = s::take(program.invocation, program.store);
  ASSERT_TRUE(taken) << taken.error().message();
  auto low = taken->destinations()[0].publish().value();
  auto high = taken->destinations()[1].publish().value();
  EXPECT_EQ(read_item(low), (std::array<float, 4>{1, 2, 3, 4}));
  EXPECT_EQ(read_item(high), (std::array<float, 4>{11, 12, 13, 14}));
  EXPECT_EQ(read_item(storage.sources[0]), read_item(low));
  EXPECT_EQ(read_item(storage.sources[1]), read_item(high));
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
  *taken = {};
  // No Graph-owned source guard or private ID reader may survive take. The
  // original source can become writable again without resetting the Graph.
  auto source_writer = storage.sources[0].try_write();
  ASSERT_TRUE(source_writer) << source_writer.error().message();
  constexpr std::array<float, 4> updated{21, 22, 23, 24};
  ASSERT_TRUE(source_writer->view()->write(std::as_bytes(std::span(updated))).ok());
  storage.sources[0] = source_writer->publish().value();
  std::swap(storage.sources[0], storage.sources[1]);
  storage.destinations[0] = high.try_write().value();
  storage.destinations[1] = low.try_write().value();
  bindings = s::SlotBindings::create(storage.layout, storage.sources, storage.destinations).value();
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store, std::move(bindings)).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  done = program.invocation.submit(program.queues);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok()); *done = {};
  taken = s::take(program.invocation, program.store);
  ASSERT_TRUE(taken) << taken.error().message();
  high = taken->destinations()[0].publish().value(); low = taken->destinations()[1].publish().value();
  EXPECT_EQ(read_item(high), (std::array<float, 4>{11, 12, 13, 14}));
  EXPECT_EQ(read_item(low), updated);
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
}

TEST(GraphSlotsExecution, SourceGatherDenseUsesOriginalGuardsAndRecyclesBothIdSlices) {
  exercise_source_gather(kItemBytes);
}
TEST(GraphSlotsExecution, SourceGatherPaddedUsesOriginalGuardsAndRecyclesBothIdSlices) {
  exercise_source_gather(kStride);
}
TEST(GraphSlotsExecution, DefaultPaddedGatherUsesOriginalGuardsWithoutForcedProvider) {
  exercise_source_gather(kStride, true);
}

TEST(GraphSlotsExecution, InvalidAssociationCountAndReplacedSourceIdsPreserveOriginalOwners) {
  SourceProgram program(kStride);
  SourceStorage storage(kStride);
  const std::array invalid{s::SourceBinding{program.store, program.store}};
  EXPECT_TRUE(s::prepare(program.invocation, program.graph, program.queues, invalid).IsInvalidArgument());
  const std::array duplicate{s::SourceBinding{program.store, program.gather},
      s::SourceBinding{program.store, program.gather}};
  EXPECT_TRUE(s::prepare(program.invocation, program.graph, program.queues, duplicate).IsInvalidArgument());
  const std::array associations{s::SourceBinding{program.store, program.gather}};
  ASSERT_TRUE(s::prepare(program.invocation, program.graph, program.queues, associations).ok());
  auto bindings = s::SlotBindings::create(storage.layout, std::span(storage.sources).first(1),
      storage.destinations).value();
  const std::array generations{bindings.destinations()[0].generation(), bindings.destinations()[1].generation()};
  EXPECT_TRUE(s::bind(program.invocation, program.graph, program.store, std::move(bindings)).IsInvalidArgument());
  ASSERT_TRUE(bindings.valid());
  for (std::size_t i = 0; i < storage.destinations.size(); ++i) {
    EXPECT_EQ(bindings.destinations()[i].generation(), generations[i]);
    EXPECT_EQ(read_item(bindings.destinations()[i]), kUntouched);
    storage.destinations[i] = std::move(bindings.destinations()[i]);
  }
  bindings = {};
  bindings = s::SlotBindings::create(storage.layout, storage.sources, storage.destinations).value();
  ASSERT_TRUE(s::bind(program.invocation, program.graph, program.store, std::move(bindings)).ok());
  const auto valid = program.invocation.prepare_replay(program.queues);
  ASSERT_TRUE(valid.ok()) << valid.message();
  auto arbitrary = m::Buffer::create(m::system_resource(), 2 * sizeof(std::int32_t), 16).value();
  constexpr std::array<std::int32_t, 2> unowned_sources{2, 2};
  ASSERT_TRUE(arbitrary.view().write(std::as_bytes(std::span(unowned_sources))).ok());
  auto tensor = c::ConstTensorView::create(program.ids_description, arbitrary.view()).value();
  const auto replacement = program.invocation.bind(program.source_ids, tensor);
  if (replacement.ok()) EXPECT_FALSE(program.invocation.prepare_replay(program.queues).ok());
  else EXPECT_TRUE(replacement.IsFailedPrecondition());
  EXPECT_FALSE(program.invocation.submit(program.queues));
  auto recovered = s::take(program.invocation, program.store);
  ASSERT_TRUE(recovered) << recovered.error().message();
  for (std::size_t i = 0; i < recovered->destination_count(); ++i) {
    EXPECT_EQ(recovered->destinations()[i].generation(), generations[i]);
    EXPECT_EQ(read_item(recovered->destinations()[i]), kUntouched);
  }
  EXPECT_EQ(read_item(storage.sources[0]), (std::array<float, 4>{1, 2, 3, 4}));
  EXPECT_EQ(read_item(storage.sources[1]), (std::array<float, 4>{11, 12, 13, 14}));
  EXPECT_EQ(read_item(storage.sentinel), kSentinel);
}

void exercise_default_gather(c::DataType type, bool padded, bool graph) {
  constexpr std::int64_t rows = 5, width = 5;
  const auto scalar = c::data_type_size(type);
  const auto row_bytes = static_cast<std::size_t>(width) * scalar;
  const auto pitch = row_bytes + (padded ? 16 : 0);
  const auto source_desc = c::TensorDesc::strided_bytes(type,
      std::array{rows, width},
      std::array<std::int64_t, 2>{static_cast<std::int64_t>(pitch),
          static_cast<std::int64_t>(scalar)}).value();
  const auto ids_desc = c::TensorDesc::strided_bytes(c::DataType::kInt32,
      std::array<std::int64_t, 1>{3}, std::array<std::int64_t, 1>{8}).value();
  const c::Gather operation{.axis = -2, .bounds = c::IndexBoundsPolicy::kChecked};
  const std::array descriptions{source_desc, ids_desc};
  const auto output_desc = c::infer_single_output(c::Operation(operation), descriptions).value();
  std::vector<std::byte> original(source_desc.required_bytes().value(), std::byte{0xa5});
  const std::array<std::uint32_t, 5> single{0x80000000u, 0x7f812345u,
      0x7fc54321u, 0u, 0x3f800000u};
  const std::array<std::uint16_t, 5> half{0x8000u, 0x7c01u, 0x7e55u, 0u, 0x3c00u};
  const std::array<std::uint16_t, 5> brain{0x8000u, 0x7f81u, 0x7fc5u, 0u, 0x3f80u};
  for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row)
    for (std::size_t column = 0; column < static_cast<std::size_t>(width); ++column) {
      const auto i = (row + column) % single.size();
      const void *bits = type == c::DataType::kFloat32
          ? static_cast<const void *>(&single[i])
          : type == c::DataType::kFloat16 ? static_cast<const void *>(&half[i])
                                        : static_cast<const void *>(&brain[i]);
      std::memcpy(original.data() + row * pitch + column * scalar, bits, scalar);
    }
  auto source = m::Buffer::create(m::system_resource(), original.size(), 64).value();
  auto ids = m::Buffer::create(m::system_resource(), ids_desc.required_bytes().value(), 64).value();
  auto destination = m::Buffer::create(m::system_resource(), output_desc.required_bytes().value(), 64).value();
  ASSERT_TRUE(source.view().write(original).ok());
  const std::array inputs{c::ConstTensorView::create(source_desc, source.view()).value(),
      c::ConstTensorView::create(ids_desc, ids.view()).value()};
  auto output = c::TensorView::create(output_desc, destination.view()).value();
  // Selected-byte Gather is independent of the native thread budget. These
  // ordinary factories choose from formats/algorithms, without an explicit
  // provider or a Store annotation.
  auto context = graph ? g::cpu::create_context({.native_threads = 2}).value()
                       : c::cpu::create_context({.native_threads = 2}).value();
  auto queue = context.create_queue().value();
  const bool native = padded || type != c::DataType::kFloat32;
  c::PreparedOperation direct;
  c::OperationInvocation direct_invocation;
  g::CompiledGraph compiled;
  g::ValueId table_node, ids_node, output_node;
  if (graph) {
    auto builder = g::GraphBuilder::create().value();
    table_node = builder.input(source_desc).value(); ids_node = builder.input(ids_desc).value();
    const auto node = builder.add_node(c::Operation(operation), std::array{table_node, ids_node}).value();
    output_node = builder.result(node).value(); ASSERT_TRUE(builder.publish(output_node).ok());
    auto description = std::move(builder).finish().value();
    auto result = g::cpu::compile(description, context);
    ASSERT_TRUE(result) << result.error().message(); compiled = std::move(*result);
    ASSERT_EQ(compiled.report.step_count(), 1u);
    EXPECT_EQ(compiled.report.step(0)->provider.id == c::native::kProviderId, native);
  } else {
    auto result = c::prepare(context, c::Operation(operation), inputs, std::span(&output, 1));
    ASSERT_TRUE(result) << result.error().message(); direct = std::move(*result);
    EXPECT_EQ(direct.provider().id == c::native::kProviderId, native);
    auto invocation = direct.create_invocation(); ASSERT_TRUE(invocation) << invocation.error().message();
    direct_invocation = std::move(*invocation);
  }
  std::array<std::int32_t, 5> index_bytes{4, -99, 0, -99, 4};
  const auto submit = [&]() -> hyle::Status {
    if (!graph) {
      auto done = direct_invocation.submit(queue, inputs, std::span(&output, 1));
      if (!done) return done.error();
      return done->wait();
    }
    auto invocation = compiled.executable.create_invocation();
    if (!invocation) return invocation.error();
    HYLE_RETURN_IF_ERROR(invocation->bind(table_node, inputs[0]));
    HYLE_RETURN_IF_ERROR(invocation->bind(ids_node, inputs[1]));
    HYLE_RETURN_IF_ERROR(invocation->bind(output_node, output));
    const std::array<g::QueueBinding, 1> queues{{{g::ExecutionLaneId{0}, &queue}}};
    HYLE_RETURN_IF_ERROR(invocation->prepare_replay(queues));
    auto done = invocation->submit(queues);
    if (!done) return done.error();
    return done->wait();
  };
  std::vector<std::byte> expected(destination.size()), actual(destination.size());
  for (int run = 0; run < 2; ++run) {
    if (run) index_bytes = {1, -99, 3, -99, 1};
    ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(index_bytes))).ok());
    const auto status = submit(); ASSERT_TRUE(status.ok()) << status.message();
    for (std::size_t i = 0; i < 3; ++i)
      std::memcpy(expected.data() + i * row_bytes,
          original.data() + static_cast<std::size_t>(index_bytes[2 * i]) * pitch, row_bytes);
    ASSERT_TRUE(destination.view().read(actual).ok()); EXPECT_EQ(actual, expected);
  }
  // The last ID is invalid: checking a prefix and partially publishing rows
  // would corrupt this exact previous output and fail the byte comparison.
  index_bytes[4] = static_cast<std::int32_t>(rows);
  ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(index_bytes))).ok());
  EXPECT_TRUE(submit().IsOutOfRange());
  ASSERT_TRUE(destination.view().read(actual).ok()); EXPECT_EQ(actual, expected);
  std::vector<std::byte> unchanged(original.size());
  ASSERT_TRUE(source.view().read(unchanged).ok()); EXPECT_EQ(unchanged, original);
}
TEST(CpuStorageGather, OrdinaryDefaultComputePreservesRawRowsAndChecksAllIdsBeforeWriting) {
  for (const auto type : {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16})
    for (const auto padded : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << " padded=" << padded);
      exercise_default_gather(type, padded, false);
    }
}
TEST(CpuStorageGather, OrdinaryDefaultGraphPreservesRawRowsWithoutStoreAnnotations) {
  for (const auto type : {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16})
    for (const auto padded : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << " padded=" << padded);
      exercise_default_gather(type, padded, true);
    }
}

// Two independent logical pools share one original Queue/resource. The
// resource adapter, not the caller, groups their private source/destination IDs.
template <class T> T batch_fixture(hyle::Result<T> value, const char *stage) {
  if (!value) throw std::runtime_error(std::string(stage) + ": " + std::string(value.error().message()));
  return std::move(*value);
}
void batch_fixture_status(hyle::Status status, const char *stage) {
  if (!status.ok()) throw std::runtime_error(std::string(stage) + ": " + std::string(status.message()));
}
struct BatchedSourceProgram final {
  c::Context context = batch_fixture(c::Context::create(c::cpu::device(), c::native::provider()), "batch Context");
  c::Queue queue = batch_fixture(context.create_queue(), "batch Queue");
  s::Schema schema = batch_fixture(s::Schema::create(item_description(), 5, kStride), "batch schema");
  c::TensorDesc ids_description = batch_fixture(c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}), "batch IDs description");
  std::array<g::ValueId, 2> pools{}, gathered{};
  std::array<g::NodeId, 2> gathers{}, stores{};
  std::array<m::Buffer, 2> outputs;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  const std::array<g::QueueBinding, 1> queues{{{g::ExecutionLaneId{0}, &queue}}};
  BatchedSourceProgram() {
    auto builder = batch_fixture(g::GraphBuilder::create(), "batch GraphBuilder");
    for (std::size_t i = 0; i < stores.size(); ++i) {
      pools[i] = batch_fixture(builder.input(batch_fixture(schema.capacity_description(), "batch capacity")), "batch pool input");
      const auto sources = batch_fixture(builder.input(ids_description), "batch source IDs input");
      const auto destinations = batch_fixture(builder.input(ids_description), "batch destination IDs input");
      gathers[i] = batch_fixture(builder.add_node(c::Operation(c::Gather{.axis = 0,
          .bounds = c::IndexBoundsPolicy::kChecked}), std::array{pools[i], sources}), "batch Gather");
      gathered[i] = batch_fixture(builder.result(gathers[i], 0), "batch Gather result");
      stores[i] = batch_fixture(builder.add_node(c::Operation::custom(s::Store{schema}),
          std::array{gathered[i], destinations}), "batch Store");
      // Store deliberately has an external effect rather than a named pool
      // resource. Independent pools still require its explicit Graph order.
      if (i) batch_fixture_status(builder.depends_on(stores[i], stores[i - 1]), "batch effect dependency");
      batch_fixture_status(builder.publish(gathered[i]), "batch publish Gather");
    }
    graph = batch_fixture(std::move(builder).finish(), "batch finish Graph");
    compiled = batch_fixture(g::cpu::compile(graph, context), "batch compile Graph");
    invocation = batch_fixture(compiled.executable.create_invocation(), "batch create invocation");
    for (std::size_t i = 0; i < outputs.size(); ++i)
      outputs[i] = batch_fixture(m::Buffer::create(m::system_resource(), 2 * kItemBytes, 64), "batch output allocation");
    bind_outputs();
  }
  void bind_outputs() {
    for (std::size_t i = 0; i < outputs.size(); ++i)
      batch_fixture_status(invocation.bind(gathered[i], batch_fixture(c::TensorView::create(
          batch_fixture(graph.value_description(gathered[i]), "batch output description"), outputs[i].view()),
          "batch output view")), "batch bind output");
  }
  std::array<s::SourceBinding, 2> associations() const {
    return {{{stores[0], gathers[0]}, {stores[1], gathers[1]}}};
  }
};

struct BatchedSourceStorage final {
  c::LeasePool pool;
  c::LeaseSlotLayout layout;
  m::ConstBufferView diagnostic;
  std::array<c::ReadLease, 2> sources;
  std::array<c::WriteLease, 2> destinations;
  c::ReadLease sentinel;
  std::array<c::ReadLease, 5> padding;
  std::vector<std::byte> expected = std::vector<std::byte>(5 * kStride, std::byte{0xa5});
  explicit BatchedSourceStorage(float offset = 0) {
    const std::array<float, 4> first{1 + offset, 2 + offset, 3 + offset, 4 + offset};
    const std::array<float, 4> second{11 + offset, 12 + offset, 13 + offset, 14 + offset};
    const auto put = [&](std::size_t slot, const auto &item) {
      std::memcpy(expected.data() + slot * kStride, item.data(), kItemBytes);
    };
    put(0, kUntouched); put(1, first); put(2, kSentinel); put(3, second); put(4, kUntouched);
    auto backing = batch_fixture(m::Buffer::create(m::system_resource(), expected.size(), 64), "batch pool backing");
    batch_fixture_status(backing.view().write(expected), "batch initialize backing"); diagnostic = backing.view();
    pool = batch_fixture(c::LeasePool::adopt(std::move(backing), {.max_blocks = 10}), "batch adopt backing");
    layout = batch_fixture(c::LeaseSlotLayout::create(pool, item_description(), 5, kStride), "batch slot layout");
    std::array<c::WriteLease, 5> blocks;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      blocks[i] = batch_fixture(pool.allocate(kItemBytes, kStride), "batch item writer");
      EXPECT_EQ(batch_fixture(layout.locate(blocks[i]), "batch item coordinates").index, i);
      auto gap = batch_fixture(pool.allocate(kStride - kItemBytes, 1), "batch padding writer");
      padding[i] = batch_fixture(gap.publish(), "batch publish padding");
    }
    sources[0] = batch_fixture(blocks[1].publish(), "batch publish source 0");
    sources[1] = batch_fixture(blocks[3].publish(), "batch publish source 1");
    sentinel = batch_fixture(blocks[2].publish(), "batch publish sentinel");
    destinations[0] = std::move(blocks[0]); destinations[1] = std::move(blocks[4]);
  }
  void expect_storage() const {
    std::vector<std::byte> actual(expected.size());
    ASSERT_TRUE(diagnostic.read(actual).ok()); EXPECT_EQ(actual, expected);
  }
  void expect_stored(const s::SlotBindings &transaction) {
    for (std::size_t i = 0; i < 2; ++i) {
      const auto value = read_item(sources[i]);
      EXPECT_EQ(read_item(transaction.destinations()[i]), value);
      const auto id = batch_fixture(layout.locate(transaction.destinations()[i]), "batch stored coordinates").index;
      std::memcpy(expected.data() + id * kStride, value.data(), kItemBytes);
    }
    expect_storage();
  }
};

std::array<c::ConstTensorView, 2> batched_control_views(BatchedSourceProgram &program) {
  struct Call final {
    const std::array<g::NodeId, 2> &stores;
    std::array<c::ConstTensorView, 2> ids{};
  } call{program.stores};
  EXPECT_TRUE(g::detail::RegionAccess::visit_bound(program.invocation,
      +[](void *opaque, const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
        auto &call = *static_cast<Call *>(opaque);
        for (std::size_t i = 0; i < call.stores.size(); ++i) {
          bool selected = false;
          for (const auto node : view.node_ordinals) selected |= node == call.stores[i].ordinal();
          if (!selected) continue;
          for (const auto &input : view.inputs)
            if (input.description().data_type() == c::DataType::kInt32) call.ids[i] = input;
        }
        return hyle::Status::Ok();
      }, &call).ok());
  return call.ids;
}

TEST(GraphSlotsExecution, BatchedStoresShareFixedControlSlicesAndRequireTheWholeGroup) {
  BatchedSourceProgram program;
  std::array<BatchedSourceStorage, 2> storage{BatchedSourceStorage{0}, BatchedSourceStorage{100}};
  const auto associations = program.associations();
  ASSERT_TRUE(s::prepare(program.invocation, program.graph, program.queues, associations).ok());
  std::array<s::SlotBindings, 2> transactions;
  for (std::size_t i = 0; i < transactions.size(); ++i)
    transactions[i] = s::SlotBindings::create(storage[i].layout, storage[i].sources, storage[i].destinations).value();
  const std::array requests{s::BindRequest{program.stores[0], &transactions[0]},
      s::BindRequest{program.stores[1], &transactions[1]}};
  EXPECT_TRUE(s::bind(program.invocation, program.graph, program.stores[0], std::move(transactions[0])).IsInvalidArgument());
  EXPECT_TRUE(transactions[0].valid()); EXPECT_TRUE(transactions[1].valid());
  const std::array duplicated{s::BindRequest{program.stores[0], &transactions[0]},
      s::BindRequest{program.stores[1], &transactions[0]}};
  EXPECT_TRUE(s::bind(program.invocation, program.graph, duplicated).IsInvalidArgument());
  EXPECT_TRUE(transactions[0].valid()); EXPECT_TRUE(transactions[1].valid());
  std::array<c::ConstTensorView, 2> first_controls{};
  for (std::size_t round = 0; round < 2; ++round) {
    ASSERT_TRUE(s::bind(program.invocation, program.graph, requests).ok());
    ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
    const auto controls = batched_control_views(program);
    ASSERT_TRUE(controls[0].valid()); ASSERT_TRUE(controls[1].valid());
    EXPECT_EQ(controls[0].storage().native_handle(), controls[1].storage().native_handle());
    EXPECT_EQ(controls[0].storage().resource(), controls[1].storage().resource());
    EXPECT_NE(controls[0].storage().offset(), controls[1].storage().offset());
    if (!round) first_controls = controls;
    for (std::size_t i = 0; i < controls.size(); ++i) {
      EXPECT_EQ(controls[i].storage().native_handle(), first_controls[i].storage().native_handle());
      EXPECT_EQ(controls[i].storage().offset(), first_controls[i].storage().offset());
      std::array<std::int32_t, 2> actual{};
      ASSERT_TRUE(controls[i].storage().read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, (round ? std::array<std::int32_t, 2>{4, 0} : std::array<std::int32_t, 2>{0, 4}));
    }
    auto done = program.invocation.submit(program.queues).value();
    EXPECT_FALSE(s::take(program.invocation, program.stores[0]));
    ASSERT_TRUE(done.wait().ok()); done = {};
    transactions[0] = s::take(program.invocation, program.stores[0]).value();
    // A complete next cohort cannot modify the carrier while the other
    // original transaction is still retained, even though work has retired.
    BatchedSourceStorage spare(200);
    auto replacement = s::SlotBindings::create(spare.layout, spare.sources, spare.destinations).value();
    const std::array partial{s::BindRequest{program.stores[0], &transactions[0]},
        s::BindRequest{program.stores[1], &replacement}};
    EXPECT_TRUE(s::bind(program.invocation, program.graph, partial).IsFailedPrecondition());
    EXPECT_TRUE(transactions[0].valid()); EXPECT_TRUE(replacement.valid()); spare.expect_storage();
    transactions[1] = s::take(program.invocation, program.stores[1]).value();
    for (std::size_t i = 0; i < storage.size(); ++i) {
      storage[i].expect_stored(transactions[i]);
      if (!round) {
        std::array<c::ReadLease, 2> ready;
        for (std::size_t row = 0; row < ready.size(); ++row)
          ready[row] = transactions[i].destinations()[row].publish().value();
        transactions[i].clear();
        std::array<c::WriteLease, 2> reversed{ready[1].try_write().value(), ready[0].try_write().value()};
        ASSERT_TRUE(transactions[i].refill(storage[i].sources, reversed).ok());
      }
    }
  }
  // Third legal preparation is revoked by the original reset, not by an
  // invented Store lifetime. The same caller owners are returned, unpublished.
  ASSERT_TRUE(s::bind(program.invocation, program.graph, requests).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  ASSERT_TRUE(program.invocation.reset().ok()); program.bind_outputs();
  EXPECT_FALSE(program.invocation.prepare_replay(program.queues).ok());
  for (std::size_t i = 0; i < storage.size(); ++i) {
    transactions[i] = s::take(program.invocation, program.stores[i]).value();
    storage[i].expect_stored(transactions[i]);
  }
}

TEST(GraphSlotsExecution, BatchedStoresLateReadPermissionFailureReturnsEveryOriginalTransaction) {
  BatchedSourceProgram program;
  std::array<std::size_t, 2> selected_steps{SIZE_MAX, SIZE_MAX};
  for (std::size_t step = 0; step < program.compiled.report.step_count(); ++step)
    for (std::size_t node = 0; node < program.compiled.report.step(step)->node_count; ++node)
      for (std::size_t i = 0; i < program.stores.size(); ++i)
        if (program.compiled.report.step_node(step, node).value() == program.stores[i]) selected_steps[i] = step;
  ASSERT_LT(selected_steps[0], selected_steps[1]); // The blocker is deliberately AFTER the first installation.
  ASSERT_NE(selected_steps[1], SIZE_MAX);
  std::array<BatchedSourceStorage, 2> storage{BatchedSourceStorage{0}, BatchedSourceStorage{100}};
  const auto associations = program.associations();
  ASSERT_TRUE(s::prepare(program.invocation, program.graph, program.queues, associations).ok());
  const auto capacity = program.schema.capacity_description().value();
  auto blocking_buffer = m::Buffer::create(m::system_resource(), capacity.required_bytes().value(), 64).value();
  std::vector<std::byte> initialized(blocking_buffer.size(), std::byte{0x5a});
  ASSERT_TRUE(blocking_buffer.view().write(initialized).ok());
  auto blocker = c::ReadLease::adopt(std::move(blocking_buffer)).value();
  ASSERT_TRUE(program.invocation.bind(program.pools[1], capacity, blocker).ok());
  std::array<s::SlotBindings, 2> transactions;
  for (std::size_t i = 0; i < transactions.size(); ++i)
    transactions[i] = s::SlotBindings::create(storage[i].layout, storage[i].sources, storage[i].destinations).value();
  const std::array requests{s::BindRequest{program.stores[0], &transactions[0]},
      s::BindRequest{program.stores[1], &transactions[1]}};
  const auto rejected = s::bind(program.invocation, program.graph, requests);
  EXPECT_TRUE(rejected.IsFailedPrecondition());
  EXPECT_EQ(rejected.message(), "read projection needs reset before rebinding");
  for (std::size_t i = 0; i < transactions.size(); ++i) {
    ASSERT_TRUE(transactions[i].valid());
    EXPECT_EQ(transactions[i].destination_addresses()[0].index, 0u);
    EXPECT_EQ(transactions[i].destination_addresses()[1].index, 4u);
    storage[i].expect_storage(); EXPECT_FALSE(s::take(program.invocation, program.stores[i]));
  }
  // No reset: withdrawing only the intentional blocker and successfully
  // rebinding proves that the earlier effect AND source projection rolled back.
  ASSERT_TRUE(program.invocation.unbind(program.pools[1]).ok());
  ASSERT_TRUE(s::bind(program.invocation, program.graph, requests).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  auto done = program.invocation.submit(program.queues).value();
  ASSERT_TRUE(done.wait().ok()); done = {};
  for (std::size_t i = 0; i < transactions.size(); ++i) {
    transactions[i] = s::take(program.invocation, program.stores[i]).value();
    storage[i].expect_stored(transactions[i]);
  }
}

TEST(GraphSlotsExecution, CohortTakeRejectsWithoutMovingAndReturnsRetiredOriginalOwners) {
  BatchedSourceProgram program;
  std::array<BatchedSourceStorage, 2> storage{BatchedSourceStorage{0}, BatchedSourceStorage{100}};
  const auto associations = program.associations();
  ASSERT_TRUE(s::prepare(program.invocation, program.graph, program.queues, associations).ok());
  std::array<s::SlotBindings, 2> transactions;
  for (std::size_t i = 0; i < transactions.size(); ++i)
    transactions[i] = s::SlotBindings::create(storage[i].layout, storage[i].sources, storage[i].destinations).value();
  const std::array generations{transactions[0].destinations()[0].generation(), transactions[1].destinations()[0].generation()};
  const std::array bindings{s::BindRequest{program.stores[0], &transactions[0]},
      s::BindRequest{program.stores[1], &transactions[1]}};
  const std::array returned{s::TakeRequest{program.stores[0], &transactions[0]},
      s::TakeRequest{program.stores[1], &transactions[1]}};
  ASSERT_TRUE(s::bind(program.invocation, program.graph, bindings).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  auto done = program.invocation.submit(program.queues).value();
  EXPECT_TRUE(s::take(program.invocation, returned).IsUnavailable());
  ASSERT_TRUE(done.wait().ok()); done = {};
  {
    auto result = program.invocation.result(program.gathered[0]).value();
    EXPECT_TRUE(s::take(program.invocation, returned).IsUnavailable());
    EXPECT_FALSE(transactions[0].valid()); EXPECT_FALSE(transactions[1].valid());
  }
  const std::array duplicate_store{s::TakeRequest{program.stores[0], &transactions[0]},
      s::TakeRequest{program.stores[0], &transactions[1]}};
  EXPECT_TRUE(s::take(program.invocation, duplicate_store).IsInvalidArgument());
  const std::array duplicate_output{s::TakeRequest{program.stores[0], &transactions[0]},
      s::TakeRequest{program.stores[1], &transactions[0]}};
  EXPECT_TRUE(s::take(program.invocation, duplicate_output).IsInvalidArgument());
  const std::array bad_last{s::TakeRequest{program.stores[0], &transactions[0]},
      s::TakeRequest{{}, &transactions[1]}};
  EXPECT_TRUE(s::take(program.invocation, bad_last).IsInvalidArgument());
  BatchedSourceStorage spare(200);
  auto occupied = s::SlotBindings::create(spare.layout, spare.sources, spare.destinations).value();
  const auto occupied_generation = occupied.destinations()[0].generation();
  const std::array occupied_output{s::TakeRequest{program.stores[0], &transactions[0]},
      s::TakeRequest{program.stores[1], &occupied}};
  EXPECT_TRUE(s::take(program.invocation, occupied_output).IsInvalidArgument());
  EXPECT_EQ(occupied.destinations()[0].generation(), occupied_generation);
  EXPECT_FALSE(transactions[0].valid()); EXPECT_FALSE(transactions[1].valid());
  ASSERT_TRUE(s::take(program.invocation, returned).ok());
  for (std::size_t i = 0; i < transactions.size(); ++i) {
    EXPECT_EQ(transactions[i].destinations()[0].generation(), generations[i]);
    storage[i].expect_stored(transactions[i]);
    auto repeated = s::take(program.invocation, program.stores[i]);
    ASSERT_FALSE(repeated);
    EXPECT_TRUE(repeated.error().IsInvalidArgument());
  }
  // Recovery after reset returns the same unpublished transactions without
  // requiring a successful execution or disturbing the unrelated spare owner.
  ASSERT_TRUE(s::bind(program.invocation, program.graph, bindings).ok());
  ASSERT_TRUE(program.invocation.prepare_replay(program.queues).ok());
  ASSERT_TRUE(program.invocation.reset().ok());
  ASSERT_TRUE(s::take(program.invocation, returned).ok());
  for (std::size_t i = 0; i < transactions.size(); ++i) {
    EXPECT_EQ(transactions[i].destinations()[0].generation(), generations[i]);
    storage[i].expect_stored(transactions[i]);
  }
  EXPECT_EQ(occupied.destinations()[0].generation(), occupied_generation);
  spare.expect_storage();
}

} // namespace
