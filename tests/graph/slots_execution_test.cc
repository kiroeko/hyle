#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
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
  // Two native threads deliberately miss the sampled one-thread kWins. These
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

} // namespace
