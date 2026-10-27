#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../engine/inference/slot_transactions.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/result.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace i = hyle::inference;
namespace m = hyle::mem;
constexpr std::size_t kBytes = 4 * sizeof(float), kStride = 64;
constexpr std::array<float, 4> kSource{1, 3, 5, 7}, kRows{11, 13, 17, 19}, kSentinel{101, 103, 107, 109};

void require_status(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
c::TensorDesc item() {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{4}).value();
}
struct Pool final {
  c::LeasePool pool = c::LeasePool::create(m::system_resource(), 3 * kStride, {.max_blocks = 3}).value();
  c::LeaseSlotLayout layout = c::LeaseSlotLayout::create(pool, item(), 3, kStride).value();
  c::ReadLease source, sentinel;
  c::WriteLease writer;
  Pool() {
    auto initial = pool.allocate(kBytes, kStride).value();
    writer = pool.allocate(kBytes, kStride).value();
    auto untouched = pool.allocate(kBytes, kStride).value();
    EXPECT_TRUE(initial.view()->write(std::as_bytes(std::span(kSource))).ok());
    EXPECT_TRUE(untouched.view()->write(std::as_bytes(std::span(kSentinel))).ok());
    source = initial.publish().value(); sentinel = untouched.publish().value();
  }
};
std::array<float, 4> bytes(const c::ReadLease &ready) {
  std::array<float, 4> actual{};
  EXPECT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  return actual;
}

// One real CPU Graph owns both a checked Gather/Store bank and an independent
// zero-source Store. The helper supplies only the original lease transactions.
struct Program final {
  c::Context context = c::Context::create(c::cpu::device(), c::native::provider()).value();
  c::Queue queue = context.create_queue().value();
  const std::array<g::QueueBinding, 1> queues{{{g::ExecutionLaneId{0}, &queue}}};
  std::array<g::NodeId, 2> stores;
  g::NodeId gather;
  g::ValueId gathered;
  g::Graph graph;
  g::CompiledGraph compiled;
  m::Buffer rows, gathered_buffer;
  g::Invocation run;
  Pool first, second;
  i::SlotTransactions transactions;
  Program() {
    const auto schema = s::Schema::create(item(), 3, kStride).value();
    const auto row_desc = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{1, 4}).value();
    const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{1}).value();
    auto builder = g::GraphBuilder::create().value();
    const auto pool = builder.input(schema.capacity_description().value()).value();
    const auto source_ids = builder.input(ids_desc).value();
    const auto destination0 = builder.input(ids_desc).value();
    gather = builder.add_node(c::Operation(c::Gather{.axis = 0,
        .bounds = c::IndexBoundsPolicy::kChecked}), std::array{pool, source_ids}).value();
    gathered = builder.result(gather).value();
    stores[0] = builder.add_node(c::Operation::custom(s::Store{schema}),
        std::array{gathered, destination0}).value();
    const auto input = builder.input(row_desc).value();
    const auto destination1 = builder.input(ids_desc).value();
    stores[1] = builder.add_node(c::Operation::custom(s::Store{schema}),
        std::array{input, destination1}).value();
    EXPECT_TRUE(builder.depends_on(stores[1], stores[0]).ok());
    EXPECT_TRUE(builder.publish(gathered).ok());
    graph = std::move(builder).finish().value();
    compiled = g::cpu::compile(graph, context).value();
    run = compiled.executable.create_invocation().value();
    const auto output_description = graph.value_description(gathered).value();
    gathered_buffer = m::Buffer::create(m::system_resource(),
        output_description.required_bytes().value(), kStride).value();
    require_status(run.bind(gathered,
        c::TensorView::create(output_description, gathered_buffer.view()).value()));
    rows = m::Buffer::create(m::system_resource(), kBytes, kStride).value();
    EXPECT_TRUE(rows.view().write(std::as_bytes(std::span(kRows))).ok());
    EXPECT_TRUE(run.bind(input, c::ConstTensorView::create(row_desc, rows.view()).value()).ok());
    const std::array associations{s::SourceBinding{stores[0], gather}};
    EXPECT_TRUE(s::prepare(run, graph, queues, associations).ok());
    const std::array specs{i::SlotTransactionSpec{stores[0], 1, 1},
        i::SlotTransactionSpec{stores[1], 0, 1}};
    EXPECT_TRUE(transactions.prepare(specs).ok());
    EXPECT_TRUE(transactions.set_layout(0, first.layout).ok());
    EXPECT_TRUE(transactions.set_layout(1, second.layout).ok());
  }
  void stage() {
    transactions.sources(0)[0] = first.source;
    transactions.writers(0)[0] = std::move(first.writer);
    transactions.writers(1)[0] = std::move(second.writer);
    EXPECT_TRUE(transactions.stage(0).ok());
    EXPECT_TRUE(transactions.stage(1).ok());
    EXPECT_FALSE(transactions.sources(0)[0].valid());
    EXPECT_TRUE(transactions.sources(1).empty());
  }
  c::Completion submit() {
    require_status(transactions.bind(run, graph));
    require_status(run.prepare_replay(queues));
    auto done = run.submit(queues);
    if (!done) throw std::runtime_error(std::string(done.error().message()));
    return std::move(*done);
  }
};

TEST(InferenceSlotTransactions, SourceAndStandaloneBanksReuseOriginalCohort) {
  Program p;
  for (std::size_t round = 0; round < 2; ++round) {
    p.stage();
    if (!round) {
      auto unprepared = p.compiled.executable.create_invocation().value();
      EXPECT_TRUE(p.transactions.bind(unprepared, p.graph).IsFailedPrecondition());
      EXPECT_FALSE(p.transactions.bound()); // Original cohort is still caller-owned.
    }
    auto done = p.submit(); ASSERT_TRUE(done.wait().ok()); done = {};
    auto first = p.transactions.take(p.run, 0);
    ASSERT_TRUE(first) << first.error().message();
    auto ready0 = (*first)[0].publish().value();
    ASSERT_TRUE(p.transactions.clear(0).ok());
    auto second = p.transactions.take(p.run, 1);
    ASSERT_TRUE(second) << second.error().message();
    auto ready1 = (*second)[0].publish().value();
    ASSERT_TRUE(p.transactions.clear(1).ok());
    EXPECT_FALSE(p.transactions.bound());
    EXPECT_EQ(bytes(ready0), kSource); EXPECT_EQ(bytes(ready1), kRows);
    EXPECT_EQ(bytes(p.first.sentinel), kSentinel); EXPECT_EQ(bytes(p.second.sentinel), kSentinel);
    if (!round) {
      p.first.writer = ready0.try_write().value();
      p.second.writer = ready1.try_write().value();
    }
  }
}

TEST(InferenceSlotTransactions, FixedColdLayoutAndRejectedStagePreserveOriginalHandles) {
  Program p;
  Pool foreign;
  EXPECT_TRUE(p.transactions.set_layout(0, foreign.layout).IsFailedPrecondition());
  p.transactions.sources(0)[0] = p.first.source;
  p.transactions.writers(0)[0] = std::move(foreign.writer);
  const auto generation = p.transactions.writers(0)[0].generation();
  EXPECT_FALSE(p.transactions.stage(0).ok());
  EXPECT_TRUE(p.transactions.sources(0)[0].valid());
  ASSERT_TRUE(p.transactions.writers(0)[0].valid());
  EXPECT_EQ(p.transactions.writers(0)[0].generation(), generation);
  // The refused replacement did not change the layout: its original writer
  // still stages, and the foreign writer remains under the caller's owner.
  foreign.writer = std::move(p.transactions.writers(0)[0]);
  p.transactions.writers(0)[0] = std::move(p.first.writer);
  ASSERT_TRUE(p.transactions.stage(0).ok());
  ASSERT_TRUE(p.transactions.clear(0).ok());
  EXPECT_TRUE(foreign.writer.valid());
  EXPECT_EQ(bytes(foreign.sentinel), kSentinel);
}

TEST(InferenceSlotTransactions, OriginalCompletionAndResultGateRecovery) {
  Program p; p.stage(); auto done = p.submit();
  auto pending = p.transactions.recover(p.run);
  ASSERT_TRUE(pending); EXPECT_FALSE(*pending);
  EXPECT_TRUE(p.transactions.bound());
  EXPECT_TRUE(p.transactions.clear(0).IsUnavailable());
  EXPECT_TRUE(p.transactions.stage(0).IsFailedPrecondition());
  ASSERT_TRUE(done.wait().ok()); done = {};
  {
    auto result = p.run.result(p.gathered).value();
    auto observed = p.transactions.recover(p.run);
    ASSERT_TRUE(observed); EXPECT_FALSE(*observed);
    EXPECT_TRUE(p.transactions.bound());
  }
  auto retired = p.transactions.recover(p.run);
  ASSERT_TRUE(retired); EXPECT_TRUE(*retired); EXPECT_FALSE(p.transactions.bound());
  EXPECT_EQ(bytes(p.first.sentinel), kSentinel); EXPECT_EQ(bytes(p.second.sentinel), kSentinel);
}

TEST(InferenceSlotTransactions, RecoveryRemembersAlreadyReturnedBank) {
  Program p; p.stage(); auto done = p.submit(); ASSERT_TRUE(done.wait().ok()); done = {};
  auto first = p.transactions.take(p.run, 0); ASSERT_TRUE(first);
  auto committed = (*first)[0].publish().value();
  ASSERT_TRUE(p.transactions.clear(0).ok()); EXPECT_TRUE(p.transactions.bound());
  auto duplicate = p.transactions.take(p.run, 0);
  ASSERT_FALSE(duplicate); EXPECT_TRUE(duplicate.error().IsFailedPrecondition());
  auto remaining = p.transactions.recover(p.run);
  ASSERT_TRUE(remaining); EXPECT_TRUE(*remaining); EXPECT_FALSE(p.transactions.bound());
  auto repeated = p.transactions.recover(p.run);
  ASSERT_TRUE(repeated); EXPECT_TRUE(*repeated);
  EXPECT_EQ(bytes(committed), kSource);
  EXPECT_EQ(bytes(p.second.sentinel), kSentinel);
}
} // namespace
