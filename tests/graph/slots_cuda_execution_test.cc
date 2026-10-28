#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/indexed_copy_backend.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/slots.h"
#include "../../src/graph/plan_state.h"

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
  const auto *state = g::detail::InvocationAccess::state(invocation);
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
    // The second real Store transaction has fresh writer generations and
    // reversed indices, but exactly the same physical capacity/control tuple.
    // This is only a span candidate: the next original prepare must seal the
    // current transaction before any native command can run.
    if (turn) ASSERT_TRUE(g::detail::reusable_configuration_spans(*state));
    EXPECT_FALSE(state->bindings_prepared);
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    EXPECT_FALSE(g::detail::reusable_configuration_spans(*state)); // Hint consumed.
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

// The original Store owns each writer and its private I32 carrier. No raw pool
// view is bound as permission. This retained readonly view is used only for a
// whole-allocation oracle after actual native retirement (or rejected submit).
template<class Word, std::size_t Rows, std::size_t Elements>
void exercise_captured_indexed_store(c::DataType type,
    const std::array<Word, 4> &patterns, bool close_before_submit = false) {
  static_assert(Rows == 1 || Rows == 2);
  constexpr std::size_t slots = 4, stride = 256;
  constexpr std::size_t item_bytes = Elements * sizeof(Word);
  auto device = c::cuda::device(0);
  ASSERT_TRUE(device) << device.error().message();
  auto context = c::Context::create(*device).value();
  auto queue = context.create_queue().value();
  auto memory = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice).value();
  auto staging = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kPinnedHost).value();
  const auto item = c::TensorDesc::dense(type,
      std::array<std::int64_t, 1>{Elements}).value();
  const auto rows_description = c::TensorDesc::dense(type,
      std::array<std::int64_t, 2>{Rows, Elements}).value();
  const auto ids_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{Rows}).value();
  auto backing = hyle::mem::Buffer::create(memory.ref(), slots * stride, stride).value();
  hyle::mem::ConstBufferView inspection = backing.view();
  std::vector<std::byte> expected(slots * stride, std::byte{0xa5}), actual(expected.size());
  ASSERT_TRUE(backing.view().write(expected).ok());
  auto pool = c::LeasePool::adopt(std::move(backing), {.max_blocks = slots}).value();
  const auto layout = c::LeaseSlotLayout::create(pool, item, slots, stride).value();
  std::array<c::WriteLease, slots> original;
  std::array<c::ReadLease, slots> ready;
  for (std::size_t slot = 0; slot < slots; ++slot) {
    original[slot] = pool.allocate(item_bytes, stride).value();
    ASSERT_EQ(layout.locate(original[slot])->index, slot);
  }
  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.input(rows_description).value();
  const auto ids = builder.input(ids_description).value();
  const auto store = builder.add_node(c::Operation::custom(s::Store{
      s::Schema::create(item, slots, stride).value()}), std::array{rows, ids}).value();
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  auto input = hyle::mem::Buffer::create(memory.ref(), Rows * item_bytes, stride).value();
  auto changed_input = hyle::mem::Buffer::create(memory.ref(), Rows * item_bytes, stride).value();
  ASSERT_TRUE(invocation.bind(rows,
      c::ConstTensorView::create(rows_description, input.view()).value()).ok());
  ASSERT_TRUE(s::prepare(invocation, queues, {.host_workspace = staging.ref()}).ok());
  s::SlotBindings transaction;
  c::ConstTensorView first_ids;
  for (std::size_t round = 0; round < (close_before_submit ? 1u : 3u); ++round) {
    SCOPED_TRACE(round);
    std::array<Word, Rows * Elements> payload;
    for (std::size_t i = 0; i < payload.size(); ++i) payload[i] = patterns[(i + round) % patterns.size()];
    auto &source = round == 2 ? changed_input : input;
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(payload))).ok());
    if (round == 2) {
      ASSERT_TRUE(invocation.bind(rows,
          c::ConstTensorView::create(rows_description, source.view()).value()).ok());
      EXPECT_FALSE(invocation.replay_report());
    }
    std::array<c::WriteLease, Rows> destinations;
    std::array<std::int32_t, Rows> selected{};
    for (std::size_t row = 0; row < Rows; ++row) {
      const auto slot = Rows == 1 ? (round == 1 ? 2u : 0u)
          : ((row + round) % 2 ? 2u : 0u);
      selected[row] = static_cast<std::int32_t>(slot);
      destinations[row] = ready[slot].valid() ? ready[slot].try_write().value() : std::move(original[slot]);
    }
    if (round) {
      transaction.clear();
      ASSERT_TRUE(transaction.refill({}, destinations).ok());
    } else transaction = s::SlotBindings::create(layout, {}, destinations).value();
    ASSERT_TRUE(s::bind(invocation, graph, store, std::move(transaction)).ok());
    const auto prepared = invocation.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto report = invocation.replay_report().value();
    EXPECT_EQ(report.route, g::ReplayRoute::kCaptured);
    EXPECT_EQ(report.recorded_capture_range_count, round == 1 ? 0u : 1u);
    EXPECT_EQ(report.reused_capture_range_count, round == 1 ? 1u : 0u);
    c::ConstTensorView controls;
    ASSERT_TRUE(g::detail::RegionAccess::visit_bound(invocation,
        +[](void *opaque, const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
          auto &controls = *static_cast<c::ConstTensorView *>(opaque);
          for (const auto &value : view.inputs)
            if (value.description().data_type() == c::DataType::kInt32) controls = value;
          return hyle::Status::Ok();
        }, &controls).ok());
    ASSERT_TRUE(controls.valid());
    std::array<std::int32_t, Rows> actual_ids{};
    ASSERT_TRUE(controls.storage().read(std::as_writable_bytes(std::span(actual_ids))).ok());
    EXPECT_EQ(actual_ids, selected);
    if (!round) first_ids = controls;
    else {
      EXPECT_EQ(controls.storage().native_handle(), first_ids.storage().native_handle());
      EXPECT_EQ(controls.storage().offset(), first_ids.storage().offset());
      EXPECT_EQ(controls.storage().size(), first_ids.storage().size());
    }
    if (close_before_submit) {
      ASSERT_TRUE(queue.close().ok());
      auto rejected = invocation.submit(queues);
      ASSERT_FALSE(rejected);
      const auto failure = rejected.error();
      EXPECT_TRUE(failure.IsFailedPrecondition()) << failure.message();
      EXPECT_FALSE(failure.has_native_error());
      // This is a real closed-Queue admission refusal, not GPU fault injection.
      auto repeated = invocation.submit(queues);
      ASSERT_FALSE(repeated);
      EXPECT_EQ(repeated.error(), failure);
      ASSERT_FALSE(invocation.in_flight());
      auto returned = s::take(invocation, store);
      ASSERT_TRUE(returned) << returned.error().message();
      transaction = std::move(*returned);
      EXPECT_EQ(rejected.error(), failure);
      ASSERT_TRUE(inspection.read(actual).ok());
      EXPECT_EQ(actual, expected);
      transaction.clear();
      break;
    }
    auto completion = invocation.submit(queues);
    ASSERT_TRUE(completion) << completion.error().message();
    auto premature = s::take(invocation, store);
    ASSERT_FALSE(premature);
    EXPECT_TRUE(premature.error().IsUnavailable());
    ASSERT_TRUE(completion->wait().ok());
    *completion = {};
    auto returned = s::take(invocation, store);
    ASSERT_TRUE(returned) << returned.error().message();
    transaction = std::move(*returned);
    for (std::size_t row = 0; row < Rows; ++row) {
      ready[static_cast<std::size_t>(selected[row])] = transaction.destinations()[row].publish().value();
      const auto bits = std::as_bytes(std::span(payload)).subspan(row * item_bytes, item_bytes);
      std::copy(bits.begin(), bits.end(),
          expected.begin() + static_cast<std::size_t>(selected[row]) * stride);
    }
    ASSERT_TRUE(inspection.read(actual).ok());
    EXPECT_EQ(actual, expected); // Includes all untouched slots and every padding byte.
  }
  transaction.clear(); invocation = {};
  for (auto &lease : ready) lease = {};
  for (auto &writer : original) writer = {};
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_EQ(pool.stats().live_blocks, 0u);
  EXPECT_EQ(pool.stats().allocated_bytes, 0u);
}

TEST(GraphSlotsCudaExecution, CapturedBFloat16M1OddRowsKeepBitsAndPoolPadding) {
  exercise_captured_indexed_store<std::uint16_t, 1, 3>(c::DataType::kBFloat16,
      {0x0000u, 0x8000u, 0x7fc1u, 0xffa3u});
}

TEST(GraphSlotsCudaExecution, CapturedFloat16M2OddRowsKeepBitsAndPoolPadding) {
  exercise_captured_indexed_store<std::uint16_t, 2, 3>(c::DataType::kFloat16,
      {0x0000u, 0x8000u, 0x7e55u, 0xfe81u});
}

TEST(GraphSlotsCudaExecution, CapturedFloat32M2RowsKeepBitsAndPoolPadding) {
  exercise_captured_indexed_store<std::uint32_t, 2, 4>(c::DataType::kFloat32,
      {0x00000000u, 0x80000000u, 0x7fc12345u, 0xffc23456u});
}

TEST(GraphSlotsCudaExecution, CapturedUInt8M2OddBytesKeepPoolPadding) {
  exercise_captured_indexed_store<std::uint8_t, 2, 3>(c::DataType::kUInt8,
      {0x00u, 0x80u, 0xffu, 0x7fu});
}

TEST(GraphSlotsCudaExecution, CapturedStoreClosedQueuePreservesRefusalAndReturnsWriters) {
  exercise_captured_indexed_store<std::uint16_t, 1, 3>(c::DataType::kBFloat16,
      {0x0000u, 0x8000u, 0x7fc1u, 0xffa3u}, true);
}

TEST(GraphSlotsCudaExecution, CapturedIndexedCopyInheritsPrefixFailureWithoutStickyStatus) {
  namespace cd = c::detail;
  auto device = c::cuda::device(0);
  ASSERT_TRUE(device) << device.error().message();
  auto context = c::Context::create(*device).value();
  auto queue = context.create_queue().value();
  auto memory = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice).value();
  constexpr std::size_t slots = 3, rows = 2, stride = 256;
  constexpr std::size_t row_bytes = 3 * sizeof(std::uint16_t);
  const std::array<std::uint16_t, 6> payload{
      0x0000u, 0x8000u, 0x7fc1u, 0xffa3u, 0x1234u, 0x5678u};
  const std::array<std::int32_t, rows> bad_ids{0, slots}, good_ids{2, 0};
  auto source = hyle::mem::Buffer::create(memory.ref(), rows * row_bytes, stride).value();
  auto bad_indices = hyle::mem::Buffer::create(memory.ref(), sizeof(bad_ids), stride).value();
  auto good_indices = hyle::mem::Buffer::create(memory.ref(), sizeof(good_ids), stride).value();
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(payload))).ok());
  ASSERT_TRUE(bad_indices.view().write(std::as_bytes(std::span(bad_ids))).ok());
  ASSERT_TRUE(good_indices.view().write(std::as_bytes(std::span(good_ids))).ok());
  std::array<hyle::mem::Buffer, 2> pools;
  std::vector<std::byte> sentinel(slots * stride, std::byte{0xa5}), actual(sentinel.size());
  for (auto &pool : pools) {
    pool = hyle::mem::Buffer::create(memory.ref(), sentinel.size(), stride).value();
    ASSERT_TRUE(pool.view().write(sentinel).ok());
  }
  auto extension = cd::BackendExtensionAccess::query(context, cd::kIndexedCopyExtensionId);
  ASSERT_TRUE(extension) << extension.error().message();
  // Only the backend's cold native program; the original QueueSequence and
  // Completion remain the sole submission and retirement owners.
  struct Program final {
    const cd::IndexedCopyBackendVTable *table = nullptr;
    void *native = nullptr;
    ~Program() { if (native) table->destroy(native, hyle::mem::system_allocator()); }
  } program;
  struct Prepare final { Program *program; cd::IndexedCopyNativeSpec spec; } prepare{
      &program, {slots, rows, row_bytes, stride}};
  const auto prepared = cd::BackendExtensionAccess::call(*extension,
      +[](void *context, const void *table, void *opaque) noexcept -> hyle::Status {
        auto &call = *static_cast<Prepare *>(opaque);
        call.program->table = static_cast<const cd::IndexedCopyBackendVTable *>(table);
        if (!call.program->table || !call.program->table->prepare ||
            !call.program->table->destroy || !call.program->table->enqueue ||
            !call.program->table->supports_command_capture)
          return hyle::Status::Unimplemented("CUDA indexed copy needs command capture");
        return call.program->table->prepare(context, call.spec,
            hyle::mem::system_allocator(), &call.program->native);
      }, &prepare);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_NE(program.native, nullptr);
  struct Copy final { Program *program; cd::IndexedCopyNativeBindings bindings; } bad{
      &program, {pools[0].view(), source.view(), bad_indices.view()}}, good{
      &program, {pools[1].view(), source.view(), good_indices.view()}};
  constexpr auto enqueue = +[](void *context, const void *, void *queue,
      void *opaque) noexcept -> cd::EnqueueResult {
    const auto &call = *static_cast<Copy *>(opaque);
    return call.program->table->enqueue(context, queue, call.program->native, call.bindings);
  };
  auto capture_extension = cd::CommandCaptureAccess::query(queue);
  ASSERT_TRUE(capture_extension) << capture_extension.error().message();
  auto recording = cd::QueueSequence::begin(queue);
  ASSERT_TRUE(recording) << recording.error().message();
  ASSERT_TRUE(cd::CommandCaptureAccess::begin(*capture_extension, *recording).ok());
  const auto recorded_copy = cd::BackendExtensionAccess::enqueue_admitted(
      *extension, *recording, enqueue, &good);
  ASSERT_TRUE(recorded_copy.accepted);
  ASSERT_TRUE(recorded_copy.status.ok()) << recorded_copy.status.message();
  recording->record(recorded_copy);
  auto captured = cd::CommandCaptureAccess::end(*capture_extension, *recording,
      hyle::mem::system_allocator());
  ASSERT_TRUE(captured) << captured.error().message();
  auto recorded = recording->finish();
  ASSERT_TRUE(recorded) << recorded.error().message();
  ASSERT_TRUE(recorded->wait().ok());
  *recorded = {};
  // A valid first row followed by one invalid ID must prevent ALL writes.
  // The later captured copy must inherit that same original sequence status.
  auto failing = cd::QueueSequence::begin(queue);
  ASSERT_TRUE(failing) << failing.error().message();
  const auto bad_copy = cd::BackendExtensionAccess::enqueue_admitted(
      *extension, *failing, enqueue, &bad);
  ASSERT_TRUE(bad_copy.accepted);
  ASSERT_TRUE(bad_copy.status.ok()) << bad_copy.status.message();
  failing->record(bad_copy);
  const auto failed_replay = captured->enqueue(*failing);
  ASSERT_TRUE(failed_replay.accepted);
  ASSERT_TRUE(failed_replay.status.ok()) << failed_replay.status.message();
  failing->record(failed_replay);
  auto failed = failing->finish();
  ASSERT_TRUE(failed) << failed.error().message();
  const auto failure = failed->wait();
  EXPECT_TRUE(failure.IsOutOfRange()) << failure.message();
  EXPECT_EQ(failed->wait(), failure);
  *failed = {};
  for (const auto &pool : pools) {
    ASSERT_TRUE(pool.view().read(actual).ok());
    EXPECT_EQ(actual, sentinel);
  }
  // Reuse the EXACT same capture in a new admitted sequence. Its independent
  // status word must be reseeded from the new successful prefix.
  auto succeeding = cd::QueueSequence::begin(queue);
  ASSERT_TRUE(succeeding) << succeeding.error().message();
  const auto successful_replay = captured->enqueue(*succeeding);
  ASSERT_TRUE(successful_replay.accepted);
  ASSERT_TRUE(successful_replay.status.ok()) << successful_replay.status.message();
  succeeding->record(successful_replay);
  auto succeeded = succeeding->finish();
  ASSERT_TRUE(succeeded) << succeeded.error().message();
  ASSERT_TRUE(succeeded->wait().ok());
  *succeeded = {};
  ASSERT_TRUE(pools[0].view().read(actual).ok());
  EXPECT_EQ(actual, sentinel);
  auto expected = sentinel;
  const auto payload_bytes = std::as_bytes(std::span(payload));
  for (std::size_t row = 0; row < rows; ++row) {
    const auto bits = payload_bytes.subspan(row * row_bytes, row_bytes);
    std::copy(bits.begin(), bits.end(),
        expected.begin() + static_cast<std::size_t>(good_ids[row]) * stride);
  }
  ASSERT_TRUE(pools[1].view().read(actual).ok());
  EXPECT_EQ(actual, expected);
  captured->reset();
}

} // namespace
