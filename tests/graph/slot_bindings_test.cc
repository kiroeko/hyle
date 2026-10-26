#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "../compute_batch/allocation_fault_fixture.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/slots.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;

static_assert(std::is_trivially_copyable_v<s::Schema>);
static_assert(!std::is_copy_constructible_v<s::SlotBindings>);
static_assert(!std::is_copy_assignable_v<s::SlotBindings>);
static_assert(std::is_nothrow_move_constructible_v<s::SlotBindings>);
static_assert(std::is_nothrow_move_assignable_v<s::SlotBindings>);

c::TensorDesc item() {
  return c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{16}).value();
}

TEST(GraphSlots, SchemaPreservesPaddingAndRejectsInvalidGeometry) {
  const auto multidimensional = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{2, 3}).value();
  const auto schema = s::Schema::create(multidimensional, 3, 32).value();
  EXPECT_EQ(schema.type(), c::DataType::kBFloat16);
  EXPECT_EQ(schema.item_elements(), 6);
  EXPECT_EQ(schema.slot_count(), 3u);
  EXPECT_EQ(schema.stride_bytes(), 32u);
  const auto capacity = schema.capacity_description().value();
  EXPECT_EQ(capacity.rank(), 2u);
  EXPECT_EQ(capacity.dimensions()[0], 3);
  EXPECT_EQ(capacity.dimensions()[1], 6);
  EXPECT_EQ(capacity.byte_strides()[0], 32);
  EXPECT_EQ(capacity.byte_strides()[1], 2);
  EXPECT_EQ(capacity.required_bytes().value(), 76u);
  EXPECT_FALSE(capacity.is_dense());
  EXPECT_TRUE(s::Schema{}.capacity_description().error().IsFailedPrecondition());
  EXPECT_TRUE(s::Schema::create(multidimensional, 0, 32).error().IsInvalidArgument());
  EXPECT_TRUE(s::Schema::create(multidimensional, 3, 10).error().IsInvalidArgument());
  EXPECT_TRUE(s::Schema::create(multidimensional, 3, 13).error().IsInvalidArgument());
  EXPECT_TRUE(s::Schema::create(capacity, 3, 128).error().IsInvalidArgument());
  const auto empty = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{0}).value();
  EXPECT_TRUE(s::Schema::create(empty, 1, 16).error().IsInvalidArgument());
  EXPECT_TRUE(s::Schema::create(item(), std::uint32_t(INT32_MAX) + 1u, 256)
      .error().IsOutOfRange());
  const auto byte = c::TensorDesc::dense(c::DataType::kUInt8,
      std::array<std::int64_t, 1>{1}).value();
  EXPECT_TRUE(s::Schema::create(byte, 4, std::size_t(INT64_MAX)).error().IsOutOfRange());
}

TEST(GraphSlots, StoreInfersExactDenseWholeItemsAndIsAnExternalEffect) {
  const auto scalar_type = c::DataType::kBFloat16;
  const auto dense_item = c::TensorDesc::dense(scalar_type,
      std::array<std::int64_t, 2>{2, 3}).value();
  const s::Store store{s::Schema::create(dense_item, 3, 32).value()};
  const auto rows = c::TensorDesc::dense(scalar_type,
      std::array<std::int64_t, 2>{2, 6}).value();
  const auto ids = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  std::array inputs{rows, ids};
  EXPECT_TRUE(store.infer_output_descriptions(inputs, {}).ok());
  const auto operation = c::Operation::custom(store);
  const auto traits = c::operation_traits(operation);
  EXPECT_EQ(traits.effect, c::EffectKind::kExternalSideEffect);
  EXPECT_EQ(traits.resource_count, 0u);
  EXPECT_EQ(traits.outputs, 0u);
  EXPECT_EQ(traits.alias, c::AliasPolicy::kNone);
  EXPECT_EQ(traits.precision, c::PrecisionPolicy::kExact);
  std::array<c::TensorDesc, 1> output{};
  EXPECT_TRUE(store.infer_output_descriptions(inputs, output).IsInvalidArgument());
  inputs[0] = c::TensorDesc::strided_bytes(scalar_type,
      std::array<std::int64_t, 2>{2, 6},
      std::array<std::int64_t, 2>{20, 2}).value();
  EXPECT_TRUE(store.infer_output_descriptions(inputs, {}).IsInvalidArgument());
  inputs[0] = rows;
  inputs[1] = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  EXPECT_TRUE(store.infer_output_descriptions(inputs, {}).IsInvalidArgument());
  inputs[1] = c::TensorDesc::dense(c::DataType::kInt64,
      std::array<std::int64_t, 1>{2}).value();
  EXPECT_TRUE(store.infer_output_descriptions(inputs, {}).IsInvalidArgument());
  inputs = {c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 6}).value(), ids};
  EXPECT_TRUE(store.infer_output_descriptions(inputs, {}).IsInvalidArgument());
  inputs = {c::TensorDesc::dense(scalar_type,
      std::array<std::int64_t, 2>{4, 6}).value(),
      c::TensorDesc::dense(c::DataType::kInt32,
          std::array<std::int64_t, 1>{4}).value()};
  EXPECT_TRUE(store.infer_output_descriptions(inputs, {}).IsInvalidArgument());
  auto builder = g::GraphBuilder::create().value();
  const std::array ports{builder.input(rows).value(), builder.input(ids).value()};
  const auto node = builder.add_node(operation, ports).value();
  const auto graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.node_output_count(node).value(), 0u);
  EXPECT_EQ(graph.effect_resource_count(), 0u);
  EXPECT_EQ(graph.output_count(), 0u);
}

TEST(GraphSlots, LateForeignSourceAndWriterNeverConsumeCallerDestinations) {
  auto pool = c::LeasePool::create(m::system_resource(), 1024, {.max_blocks = 4}).value();
  const auto layout = c::LeaseSlotLayout::create(pool, item(), 4, 256).value();
  const auto ready = pool.allocate(64).value().publish().value();
  auto other_pool = c::LeasePool::create(m::system_resource(), 512, {.max_blocks = 2}).value();
  const auto foreign = other_pool.allocate(64).value().publish().value();
  std::array<c::ReadLease, 2> sources{ready, foreign};
  std::array<c::WriteLease, 2> destinations;
  destinations[0] = pool.allocate(64).value();
  destinations[1] = pool.allocate(64).value();
  const auto first_generation = destinations[0].generation();
  const auto second_generation = destinations[1].generation();
  auto rejected_source = s::SlotBindings::create(layout, sources, destinations);
  ASSERT_FALSE(rejected_source);
  EXPECT_TRUE(rejected_source.error().IsInvalidArgument());
  EXPECT_EQ(destinations[0].generation(), first_generation);
  EXPECT_EQ(destinations[1].generation(), second_generation);
  sources[1] = ready;
  destinations[1] = other_pool.allocate(64).value();
  const auto foreign_generation = destinations[1].generation();
  auto rejected_writer = s::SlotBindings::create(layout, sources, destinations);
  ASSERT_FALSE(rejected_writer);
  EXPECT_TRUE(rejected_writer.error().IsInvalidArgument());
  EXPECT_EQ(destinations[0].generation(), first_generation);
  EXPECT_EQ(destinations[1].generation(), foreign_generation);
  EXPECT_TRUE(destinations[0].valid());
  EXPECT_TRUE(destinations[1].valid());
  EXPECT_TRUE(destinations[0].view()->fill(std::byte{7}).ok());
}

TEST(GraphSlots, FailedLateWriterIsNotTurnedIntoAnOwnedReadyResult) {
  auto pool = c::LeasePool::create(m::system_resource(), 1024, {.max_blocks = 4}).value();
  const auto layout = c::LeaseSlotLayout::create(pool, item(), 4, 256).value();
  const std::array sources{pool.allocate(64).value().publish().value()};
  std::array<c::WriteLease, 2> destinations;
  destinations[0] = pool.allocate(64).value();
  destinations[1] = pool.allocate(64).value();
  const std::array generations{destinations[0].generation(), destinations[1].generation()};
  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto episode = uses.begin().value();
  ASSERT_TRUE(episode.write(destinations[1]).ok());
  auto failed = c::Completion::completed(hyle::Status::Internal("intentional slot write failure"));
  ASSERT_TRUE(episode.attach(failed).ok());
  EXPECT_TRUE(failed.wait().IsInternal());
  failed = {};
  auto rejected = s::SlotBindings::create(layout, sources, destinations);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsInternal());
  EXPECT_EQ(destinations[0].generation(), generations[0]);
  EXPECT_EQ(destinations[1].generation(), generations[1]);
  EXPECT_TRUE(destinations[0].valid());
  EXPECT_TRUE(destinations[1].valid());
  EXPECT_TRUE(sources[0].view());
}

TEST(GraphSlots, MetadataFailureDoesNotConsumeAndBudgetTracksOnlySelectedOwners) {
  auto pool = c::LeasePool::create(m::system_resource(), 256 * 1024, {.max_blocks = 2}).value();
  const auto layout = c::LeaseSlotLayout::create(pool, item(), 1024, 256).value();
  const std::array sources{pool.allocate(64).value().publish().value()};
  std::array<c::WriteLease, 1> destinations{pool.allocate(64).value()};
  const auto generation = destinations[0].generation();
  const auto before = pool.stats();
  // Include a late vector allocation failure, after earlier metadata exists.
  for (std::size_t failure = 0; failure < 5; ++failure) {
    hyle_f5_fault_fixture::Allocator allocator;
    allocator.fail = failure;
    auto rejected = s::SlotBindings::create(layout, sources, destinations,
        {.allocator = allocator.ref()});
    ASSERT_FALSE(rejected) << failure;
    EXPECT_TRUE(rejected.error().IsResourceExhausted()) << failure;
    EXPECT_EQ(allocator.live, 0u) << failure;
    EXPECT_EQ(destinations[0].generation(), generation);
    EXPECT_TRUE(destinations[0].valid());
  }
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  auto budget = c::ResourceBudget::create({.limits = limits}).value();
  const auto initial = budget.usage(m::system_resource())->live_bytes;
  {
    auto bindings = s::SlotBindings::create(layout, sources, destinations,
        {.resource_budget = budget}).value();
    EXPECT_FALSE(destinations[0].valid());
    EXPECT_EQ(bindings.source_count(), 1u);
    EXPECT_EQ(bindings.destination_count(), 1u);
    const auto usage = budget.usage(m::system_resource()).value();
    EXPECT_GT(usage.live_bytes, initial);
    EXPECT_LT(usage.live_bytes - initial, 4096u);
    EXPECT_EQ(usage.live_by_category[std::size_t(c::ResourceBudgetCategory::kMetadata)],
        usage.live_bytes);
    EXPECT_EQ(pool.stats().allocated_bytes, before.allocated_bytes);
    EXPECT_EQ(pool.stats().live_blocks, before.live_blocks);
  }
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, initial);
}

TEST(GraphSlots, OwnsExactAddressesAcrossPoolMoveDropAndTransactionMove) {
  s::SlotBindings bindings;
  c::LeaseSlotAddress source_address, destination_address;
  {
    auto pool = c::LeasePool::create(m::system_resource(), 512, {.max_blocks = 2}).value();
    auto layout = c::LeaseSlotLayout::create(pool, item(), 2, 256).value();
    auto initial = pool.allocate(64).value();
    ASSERT_TRUE(initial.view()->fill(std::byte{29}).ok());
    const auto source = initial.publish().value();
    const std::array repeated_sources{source, source};
    std::array<c::WriteLease, 1> destinations{pool.allocate(64).value()};
    source_address = layout.locate(repeated_sources[0]).value();
    destination_address = layout.locate(destinations[0]).value();
    auto moved_pool = std::move(pool);
    auto selected = s::SlotBindings::create(layout, repeated_sources, destinations);
    ASSERT_TRUE(selected) << selected.error().message();
    bindings = std::move(*selected);
    EXPECT_FALSE(selected->valid());
    EXPECT_FALSE(destinations[0].valid());
    EXPECT_EQ(moved_pool.stats().live_blocks, 2u);
  }
  ASSERT_TRUE(bindings.valid());
  EXPECT_EQ(bindings.source_count(), 2u);
  EXPECT_EQ(bindings.destination_count(), 1u);
  EXPECT_EQ(bindings.source_addresses()[0], source_address);
  EXPECT_EQ(bindings.source_addresses()[1], source_address);
  EXPECT_EQ(bindings.destination_addresses()[0], destination_address);
  EXPECT_EQ(bindings.layout().locate(bindings.destinations()[0]).value(), destination_address);
  std::array<std::byte, 64> bytes{};
  ASSERT_TRUE(bindings.sources()[0].view()->read(bytes).ok());
  for (const auto byte : bytes) EXPECT_EQ(byte, std::byte{29});
  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto episode = uses.begin().value();
  EXPECT_TRUE(episode.read(bindings.sources()[0]).IsFailedPrecondition());
  auto moved = std::move(bindings);
  EXPECT_FALSE(bindings.valid());
  EXPECT_EQ(bindings.source_count(), 0u);
  EXPECT_TRUE(bindings.sources().empty());
  EXPECT_EQ(moved.destination_addresses()[0], destination_address);
  EXPECT_EQ(moved.layout().locate(moved.sources()[0]).value(), source_address);
}

} // namespace
