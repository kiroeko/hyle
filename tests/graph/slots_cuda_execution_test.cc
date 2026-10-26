#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/slots.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;

TEST(GraphSlotsCudaExecution, DeviceStoreUpdatesPrivateIndicesOnTheOriginalQueue) {
  auto device = c::cuda::device(0);
  ASSERT_TRUE(device) << device.error().message();
  auto context = c::Context::create(*device).value();
  auto queue = context.create_queue().value();
  auto memory = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice).value();
  auto staging = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kPinnedHost).value();
  ASSERT_FALSE(memory.ref().properties().has(hyle::mem::MemoryProperty::kHostAccessible));
  constexpr std::size_t item_bytes = 4 * sizeof(float), stride = 256;
  const auto item = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const auto rows_description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 4}).value();
  const auto ids_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  auto pool = c::LeasePool::create(memory.ref(), 3 * stride,
      {.max_blocks = 3}).value();
  const auto layout = c::LeaseSlotLayout::create(pool, item, 3, stride).value();
  std::array<c::WriteLease, 2> destinations;
  destinations[0] = pool.allocate(item_bytes, stride).value();
  auto middle = pool.allocate(item_bytes, stride).value();
  destinations[1] = pool.allocate(item_bytes, stride).value();
  ASSERT_EQ(layout.locate(destinations[0])->index, 0u);
  ASSERT_EQ(layout.locate(middle)->index, 1u);
  ASSERT_EQ(layout.locate(destinations[1])->index, 2u);
  constexpr std::array<float, 4> sentinel_values{101, 103, 107, 109};
  ASSERT_TRUE(middle.view()->write(std::as_bytes(std::span(sentinel_values))).ok());
  const auto sentinel = middle.publish().value();

  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.input(rows_description).value();
  const auto ids = builder.input(ids_description).value();
  const auto store = builder.add_node(c::Operation::custom(s::Store{
      s::Schema::create(item, 3, stride).value()}), std::array{rows, ids}).value();
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  auto input = hyle::mem::Buffer::create(memory.ref(), 2 * item_bytes, 256).value();
  ASSERT_TRUE(invocation.bind(rows, c::ConstTensorView::create(rows_description,
      input.view()).value()).ok());
  ASSERT_TRUE(s::prepare(invocation, queues, {.host_workspace = staging.ref()}).ok());

  constexpr std::array<std::array<float, 8>, 2> incoming{{
      {1, 2, 3, 4, 11, 12, 13, 14},
      {21, 22, 23, 24, 31, 32, 33, 34}}};
  std::array<c::ReadLease, 2> ready;
  c::ConstTensorView first_ids;
  for (std::size_t turn = 0; turn < incoming.size(); ++turn) {
    SCOPED_TRACE(turn);
    if (turn) {
      destinations[0] = ready[1].try_write().value();
      destinations[1] = ready[0].try_write().value();
    }
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(incoming[turn]))).ok());
    auto bindings = s::SlotBindings::create(layout, {}, destinations).value();
    ASSERT_TRUE(s::bind(invocation, graph, store, std::move(bindings)).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    c::ConstTensorView current_ids;
    ASSERT_TRUE(g::detail::RegionAccess::visit_bound(invocation,
        +[](void *opaque, const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
          auto &ids = *static_cast<c::ConstTensorView *>(opaque);
          for (const auto &input : view.inputs)
            if (input.description().data_type() == c::DataType::kInt32 &&
                input.description().rank() == 1)
              ids = input;
          return hyle::Status::Ok();
        }, &current_ids).ok());
    ASSERT_TRUE(current_ids.valid());
    EXPECT_EQ(current_ids.storage().resource(), memory.ref());
    std::array<std::int32_t, 2> actual_ids{};
    ASSERT_TRUE(current_ids.storage().read(
        std::as_writable_bytes(std::span(actual_ids))).ok());
    EXPECT_EQ(actual_ids, (turn == 0 ? std::array<std::int32_t, 2>{0, 2}
                                     : std::array<std::int32_t, 2>{2, 0}));
    if (turn == 0) first_ids = current_ids;
    else {
      EXPECT_EQ(current_ids.storage().native_handle(), first_ids.storage().native_handle());
      EXPECT_EQ(current_ids.storage().offset(), first_ids.storage().offset());
    }
    auto completion = invocation.submit(queues);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    *completion = {};
    auto returned = s::take(invocation, store);
    ASSERT_TRUE(returned) << returned.error().message();
    ASSERT_EQ(returned->destination_count(), 2u);
    for (std::size_t row = 0; row < 2; ++row) {
      ready[row] = returned->destinations()[row].publish().value();
      std::array<float, 4> actual{}, expected{};
      for (std::size_t column = 0; column < 4; ++column)
        expected[column] = incoming[turn][4 * row + column];
      ASSERT_TRUE(ready[row].view()->read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, expected);
    }
    std::array<float, 4> middle_values{};
    ASSERT_TRUE(sentinel.view()->read(std::as_writable_bytes(std::span(middle_values))).ok());
    EXPECT_EQ(middle_values, sentinel_values);
  }
}

TEST(GraphSlotsCudaExecution, TwoStoresShareFixedControlsAndKeepIndependentPoolSlots) {
  auto device = c::cuda::device(0);
  ASSERT_TRUE(device) << device.error().message();
  auto context = c::Context::create(*device).value();
  auto queue = context.create_queue().value();
  auto memory = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice).value();
  auto staging = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kPinnedHost).value();
  constexpr std::size_t item_bytes = 4 * sizeof(float), stride = 256;
  const auto item = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const auto rows_description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 4}).value();
  const auto ids_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  std::array<c::LeasePool, 2> pools;
  std::array<c::LeaseSlotLayout, 2> layouts;
  std::array<std::array<c::WriteLease, 2>, 2> destinations;
  std::array<c::ReadLease, 2> sentinels;
  constexpr std::array<float, 4> sentinel_values{101, 103, 107, 109};
  for (std::size_t i = 0; i < 2; ++i) {
    pools[i] = c::LeasePool::create(memory.ref(), 3 * stride, {.max_blocks = 3}).value();
    layouts[i] = c::LeaseSlotLayout::create(pools[i], item, 3, stride).value();
    destinations[i][0] = pools[i].allocate(item_bytes, stride).value();
    auto middle = pools[i].allocate(item_bytes, stride).value();
    destinations[i][1] = pools[i].allocate(item_bytes, stride).value();
    ASSERT_TRUE(middle.view()->write(std::as_bytes(std::span(sentinel_values))).ok());
    sentinels[i] = middle.publish().value();
  }
  // Each pool derives its own physical IDs, even though the schema is equal.
  std::swap(destinations[1][0], destinations[1][1]);
  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.input(rows_description).value();
  std::array<g::NodeId, 2> stores;
  for (std::size_t i = 0; i < 2; ++i) {
    const auto ids = builder.input(ids_description).value();
    stores[i] = builder.add_node(c::Operation::custom(s::Store{
        s::Schema::create(item, 3, stride).value()}), std::array{rows, ids}).value();
  }
  ASSERT_TRUE(builder.depends_on(stores[1], stores[0]).ok());
  auto logical = std::move(builder).finish();
  ASSERT_TRUE(logical) << logical.error().message();
  const auto graph = std::move(*logical);
  auto compiled = g::cuda::compile(graph, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  auto input = hyle::mem::Buffer::create(memory.ref(), 2 * item_bytes, 256).value();
  ASSERT_TRUE(invocation.bind(rows,
      c::ConstTensorView::create(rows_description, input.view()).value()).ok());
  ASSERT_TRUE(s::prepare(invocation, queues, {.host_workspace = staging.ref()}).ok());
  std::array<s::SlotBindings, 2> transactions;
  std::array<std::array<c::ReadLease, 2>, 2> ready;
  std::array<c::ConstTensorView, 2> first_ids;
  constexpr std::array<std::array<float, 8>, 2> incoming{{
      {1, 2, 3, 4, 11, 12, 13, 14},
      {21, 22, 23, 24, 31, 32, 33, 34}}};
  for (std::size_t turn = 0; turn < 2; ++turn) {
    SCOPED_TRACE(turn);
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(incoming[turn]))).ok());
    for (std::size_t i = 0; i < 2; ++i) {
      if (turn) {
        destinations[i][0] = ready[i][1].try_write().value();
        destinations[i][1] = ready[i][0].try_write().value();
        transactions[i].clear();
        ASSERT_TRUE(transactions[i].refill({}, destinations[i]).ok());
      } else transactions[i] = s::SlotBindings::create(layouts[i], {}, destinations[i]).value();
    }
    const std::array requests{s::BindRequest{stores[0], &transactions[0]},
        s::BindRequest{stores[1], &transactions[1]}};
    // A partial cohort must neither upload nor consume an individual owner.
    EXPECT_TRUE(s::bind(invocation, graph, stores[0], std::move(transactions[0])).IsInvalidArgument());
    ASSERT_TRUE(transactions[0].valid());
    ASSERT_TRUE(transactions[0].destinations()[0].valid());
    auto bound = s::bind(invocation, graph, requests);
    ASSERT_TRUE(bound.ok()) << bound.message();
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    struct Controls final {
      std::array<g::NodeId, 2> stores;
      std::array<c::ConstTensorView, 2> ids;
    } controls{stores};
    ASSERT_TRUE(g::detail::RegionAccess::visit_bound(invocation,
        +[](void *opaque, const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
          auto &controls = *static_cast<Controls *>(opaque);
          for (std::size_t i = 0; i < controls.stores.size(); ++i) {
            if (std::find(view.node_ordinals.begin(), view.node_ordinals.end(),
                controls.stores[i].ordinal()) == view.node_ordinals.end()) continue;
            for (const auto &value : view.inputs)
              if (value.description().data_type() == c::DataType::kInt32)
                controls.ids[i] = value;
          }
          return hyle::Status::Ok();
        }, &controls).ok());
    for (std::size_t i = 0; i < 2; ++i) {
      ASSERT_TRUE(controls.ids[i].valid());
      const std::array<std::int32_t, 2> expected = (i == turn)
          ? std::array<std::int32_t, 2>{0, 2} : std::array<std::int32_t, 2>{2, 0};
      std::array<std::int32_t, 2> actual{};
      ASSERT_TRUE(controls.ids[i].storage().read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, expected);
      if (!turn) first_ids[i] = controls.ids[i];
      else {
        EXPECT_EQ(controls.ids[i].storage().native_handle(), first_ids[i].storage().native_handle());
        EXPECT_EQ(controls.ids[i].storage().offset(), first_ids[i].storage().offset());
      }
    }
    EXPECT_EQ(controls.ids[0].storage().native_handle(), controls.ids[1].storage().native_handle());
    const auto a = controls.ids[0].storage().offset(), b = controls.ids[1].storage().offset();
    EXPECT_EQ(a > b ? a - b : b - a, 2 * sizeof(std::int32_t));
    auto completion = invocation.submit(queues);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    *completion = {};
    for (std::size_t i = 0; i < 2; ++i) {
      auto returned = s::take(invocation, stores[i]);
      ASSERT_TRUE(returned) << returned.error().message();
      transactions[i] = std::move(*returned);
      for (std::size_t row = 0; row < 2; ++row) {
        ready[i][row] = transactions[i].destinations()[row].publish().value();
        std::array<float, 4> actual{}, expected{};
        std::copy_n(incoming[turn].begin() + 4 * row, 4, expected.begin());
        ASSERT_TRUE(ready[i][row].view()->read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, expected);
      }
      std::array<float, 4> actual_sentinel{};
      ASSERT_TRUE(sentinels[i].view()->read(std::as_writable_bytes(std::span(actual_sentinel))).ok());
      EXPECT_EQ(actual_sentinel, sentinel_values);
    }
  }
}

} // namespace
