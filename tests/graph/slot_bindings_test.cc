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
    bindings.clear();
    EXPECT_TRUE(bindings.valid());
    EXPECT_EQ(bindings.source_count(), 1u);
    EXPECT_EQ(bindings.destination_count(), 1u);
    EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, usage.live_bytes);
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

TEST(GraphSlots, ClearAndRefillReuseMetadataAndReleaseOldSlotGuardsWithoutAllocation) {
  hyle_f5_fault_fixture::Allocator metadata;
  auto pool_result = c::LeasePool::create(m::system_resource(), 1024, {.max_blocks = 4});
  ASSERT_TRUE(pool_result) << pool_result.error().message();
  auto pool = std::move(*pool_result);
  auto layout_result = c::LeaseSlotLayout::create(pool, item(), 4, 256);
  ASSERT_TRUE(layout_result) << layout_result.error().message();
  const auto layout = *layout_result;
  auto initial_result = pool.allocate(64, 256);
  ASSERT_TRUE(initial_result) << initial_result.error().message();
  auto source_result = initial_result->publish();
  ASSERT_TRUE(source_result) << source_result.error().message();
  auto old_source = std::move(*source_result);
  std::array sources{old_source};
  std::array<c::WriteLease, 2> destinations;
  for (auto &destination : destinations) {
    auto allocated = pool.allocate(64, 256);
    ASSERT_TRUE(allocated) << allocated.error().message();
    destination = std::move(*allocated);
  }
  auto selected = s::SlotBindings::create(layout, sources, destinations,
      {.allocator = metadata.ref()});
  ASSERT_TRUE(selected) << selected.error().message();
  auto returned = std::move(*selected); // Same move used to return the retired owner.
  const auto *source_storage = returned.sources().data();
  const auto *destination_storage = returned.destinations().data();
  const auto *source_addresses = returned.source_addresses().data();
  const auto *destination_addresses = returned.destination_addresses().data();
  const auto chosen_address = returned.destination_addresses()[0];
  const auto discarded_generation = returned.destination_addresses()[1].generation;
  auto chosen_view = returned.destinations()[0].view();
  ASSERT_TRUE(chosen_view) << chosen_view.error().message();
  ASSERT_TRUE(chosen_view->fill(std::byte{37}).ok());
  auto chosen_result = returned.destinations()[0].publish();
  ASSERT_TRUE(chosen_result) << chosen_result.error().message();
  const auto chosen = std::move(*chosen_result);
  sources[0] = {};
  const auto attempts = metadata.attempts, live = metadata.live;
  metadata.fail = attempts;
  returned.clear();
  EXPECT_TRUE(returned.valid());
  EXPECT_TRUE(returned.layout().valid());
  EXPECT_EQ(returned.source_count(), 1u);
  EXPECT_EQ(returned.destination_count(), 2u);
  EXPECT_TRUE(returned.sources().empty());
  EXPECT_TRUE(returned.destinations().empty());
  EXPECT_TRUE(returned.source_addresses().empty());
  EXPECT_TRUE(returned.destination_addresses().empty());
  // clear withdrew the transaction's old source guard, so the external head
  // can regain its original writer instead of remaining pinned by metadata.
  auto recycled = old_source.try_write();
  ASSERT_TRUE(recycled) << recycled.error().message();
  destinations[0] = std::move(*recycled);
  auto next_destination = pool.allocate(64, 256);
  ASSERT_TRUE(next_destination) << next_destination.error().message();
  destinations[1] = std::move(*next_destination);
  const auto next_generation = destinations[1].generation();
  const std::array next_sources{chosen};
  auto moved = std::move(returned);
  returned.clear();
  EXPECT_TRUE(returned.refill(next_sources, destinations).IsFailedPrecondition());
  const auto status = moved.refill(next_sources, destinations);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(destinations[0].valid());
  EXPECT_FALSE(destinations[1].valid());
  EXPECT_EQ(moved.sources().data(), source_storage);
  EXPECT_EQ(moved.destinations().data(), destination_storage);
  EXPECT_EQ(moved.source_addresses().data(), source_addresses);
  EXPECT_EQ(moved.destination_addresses().data(), destination_addresses);
  EXPECT_EQ(moved.source_addresses()[0], chosen_address);
  EXPECT_EQ(moved.destination_addresses()[1].generation, next_generation);
  EXPECT_NE(next_generation, discarded_generation);
  EXPECT_EQ(metadata.attempts, attempts);
  EXPECT_EQ(metadata.live, live);
  std::array<std::byte, 64> bytes{};
  auto current_source = moved.sources()[0].view();
  ASSERT_TRUE(current_source) << current_source.error().message();
  ASSERT_TRUE(current_source->read(bytes).ok());
  for (auto byte : bytes) EXPECT_EQ(byte, std::byte{37});
}

TEST(GraphSlots, RefillRejectsLiveWritersCountsAndLateForeignHandlesWithoutMutation) {
  auto pool_result = c::LeasePool::create(m::system_resource(), 2048, {.max_blocks = 8});
  ASSERT_TRUE(pool_result) << pool_result.error().message();
  auto pool = std::move(*pool_result);
  auto layout_result = c::LeaseSlotLayout::create(pool, item(), 8, 256);
  ASSERT_TRUE(layout_result) << layout_result.error().message();
  const auto layout = *layout_result;
  auto initial = pool.allocate(64, 256);
  ASSERT_TRUE(initial) << initial.error().message();
  auto ready = initial->publish();
  ASSERT_TRUE(ready) << ready.error().message();
  const std::array sources{*ready, *ready};
  std::array<c::WriteLease, 2> original, replacement;
  for (auto *writers : {&original, &replacement}) for (auto &writer : *writers) {
    auto allocated = pool.allocate(64, 256);
    ASSERT_TRUE(allocated) << allocated.error().message();
    writer = std::move(*allocated);
  }
  auto selected = s::SlotBindings::create(layout, sources, original);
  ASSERT_TRUE(selected) << selected.error().message();
  auto &bindings = *selected;
  const std::array saved_sources{bindings.source_addresses()[0], bindings.source_addresses()[1]};
  const std::array saved_destinations{bindings.destination_addresses()[0], bindings.destination_addresses()[1]};
  const auto first_generation = replacement[0].generation();
  EXPECT_TRUE(bindings.refill(sources, replacement).IsFailedPrecondition());
  EXPECT_EQ(bindings.destination_addresses()[0], saved_destinations[0]);
  EXPECT_TRUE(replacement[0].valid());
  EXPECT_TRUE(replacement[1].valid());
  for (auto &writer : bindings.destinations()) writer = {};
  EXPECT_TRUE(bindings.refill({}, replacement).IsInvalidArgument());
  auto foreign_pool = c::LeasePool::create(m::system_resource(), 512, {.max_blocks = 2});
  ASSERT_TRUE(foreign_pool) << foreign_pool.error().message();
  auto foreign_writer = foreign_pool->allocate(64, 256);
  ASSERT_TRUE(foreign_writer) << foreign_writer.error().message();
  auto foreign = foreign_writer->publish();
  ASSERT_TRUE(foreign) << foreign.error().message();
  std::array next_sources{sources[0], *foreign};
  EXPECT_TRUE(bindings.refill(next_sources, replacement).IsInvalidArgument());
  auto wrong_destination = foreign_pool->allocate(64, 256);
  ASSERT_TRUE(wrong_destination) << wrong_destination.error().message();
  replacement[1] = std::move(*wrong_destination);
  const auto foreign_generation = replacement[1].generation();
  next_sources[1] = sources[1];
  EXPECT_TRUE(bindings.refill(next_sources, replacement).IsInvalidArgument());
  EXPECT_EQ(bindings.source_addresses()[0], saved_sources[0]);
  EXPECT_EQ(bindings.source_addresses()[1], saved_sources[1]);
  EXPECT_EQ(bindings.destination_addresses()[0], saved_destinations[0]);
  EXPECT_EQ(bindings.destination_addresses()[1], saved_destinations[1]);
  EXPECT_EQ(bindings.sources()[0].generation(), sources[0].generation());
  EXPECT_EQ(replacement[0].generation(), first_generation);
  EXPECT_EQ(replacement[1].generation(), foreign_generation);
  EXPECT_TRUE(replacement[0].valid());
  EXPECT_TRUE(replacement[1].valid());
  auto valid_destination = pool.allocate(64, 256);
  ASSERT_TRUE(valid_destination) << valid_destination.error().message();
  replacement[1] = std::move(*valid_destination);
  const auto status = bindings.refill(next_sources, replacement);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(replacement[0].valid());
  EXPECT_FALSE(replacement[1].valid());
}

} // namespace
