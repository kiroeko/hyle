#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/backends/cpu.h"
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

} // namespace
