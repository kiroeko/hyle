#include <gtest/gtest.h>

#include <array>
#include <span>

#include "../compute/custom_kernel_fixture.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/graph/result.h"
#include "test_environment.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::Status;

c::Context make_context(custom_test::Probe &probe) {
  const std::array providers{
      c::ProviderBinding{custom_test::provider(probe).value()}};
  return c::Context::create(c::cpu::device(), {.providers = providers}).value();
}

struct Fixture final {
  custom_test::Probe probe;
  c::Context context = make_context(probe);
  c::Queue queue = context.create_queue().value();
  g::test::Environment environment{{&context, hyle::mem::system_resource()}};
  custom_test::Tensors other, constant_storage;
  std::array<g::ValueId, 2> inputs{}, outputs{};
  g::ValueId internal, constant;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  const std::array<g::QueueBinding, 1> queues{{{environment.lane(0), &queue}}};

  Fixture() {
    EXPECT_TRUE(other.fill().ok());
    EXPECT_TRUE(constant_storage.fill().ok());
    auto builder = g::GraphBuilder::create().value();
    for (auto &input : inputs)
      input = builder.input(custom_test::description()).value();
    constant = builder.constant(custom_test::description()).value();
    internal = builder.add(c::Operation::custom(custom_test::Affine{1}),
                           inputs[0]).value();
    outputs[0] = builder.add(c::Operation::custom(custom_test::Affine{1}),
                            internal).value();
    outputs[1] = builder.add(c::Operation::custom(custom_test::Affine{1}),
                            inputs[1]).value();
    for (const auto output : outputs)
      EXPECT_TRUE(builder.publish(output).ok());
    graph = std::move(builder).finish().value();
    const std::array constants{g::ConstantBinding{constant, constant_storage.in}};
    g::CompileOptions options;
    options.constants = constants;
    options.region_selection = g::RegionSelectionPolicy::kDisabled;
    compiled = g::compile(graph, environment.environment(), options).value();
    invocation = compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{}}).value();
    EXPECT_TRUE(invocation.bind(inputs[1], other.in).ok());
    EXPECT_TRUE(invocation.bind(outputs[1], other.out).ok());
  }
};

c::LeasePool pool() {
  return c::LeasePool::create(hyle::mem::system_resource(), 256,
                              {.max_blocks = 1}).value();
}

Status bind_owned(Fixture &fixture, c::LeasePool &reads,
                  c::LeasePool &writes, const std::array<float, 4> &values) {
  HYLE_ASSIGN_OR_RETURN(auto writer, reads.allocate(16, 64));
  HYLE_ASSIGN_OR_RETURN(auto storage, writer.view());
  HYLE_RETURN_IF_ERROR(storage.write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto read, writer.publish());
  HYLE_RETURN_IF_ERROR(fixture.invocation.bind(fixture.inputs[0],
      custom_test::description(), read));
  read.reset();
  HYLE_ASSIGN_OR_RETURN(auto output, writes.allocate(16, 64));
  return fixture.invocation.bind(fixture.outputs[0],
      custom_test::description(), std::move(output));
}

std::array<float, 4> read_output(Fixture &fixture, g::ValueId output) {
  auto result = fixture.invocation.result(output).value();
  auto tensor = result.tensor().value();
  std::array<float, 4> values{};
  EXPECT_TRUE(tensor.storage().read(
      std::as_writable_bytes(std::span(values))).ok());
  return values;
}

TEST(InvocationUnbind, OwnedPortsReleaseTheirActualPoolsAndKeepOtherBindings) {
  Fixture fixture;
  auto reads = pool(), writes = pool();
  ASSERT_TRUE(bind_owned(fixture, reads, writes, {1, 2, 3, 4}).ok());
  ASSERT_EQ(reads.stats().live_blocks, 1u);
  ASSERT_EQ(writes.stats().live_blocks, 1u);
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  {
    auto completion = fixture.invocation.submit(fixture.queues).value();
    ASSERT_TRUE(completion.wait().ok());
  }
  EXPECT_EQ(read_output(fixture, fixture.outputs[0]),
            (std::array<float, 4>{7, 11, 15, 19}));
  EXPECT_EQ(fixture.other.read(), (std::array<float, 4>{3, 5, 7, 9}));
  const std::array selected{fixture.inputs[0], fixture.outputs[0]};
  ASSERT_TRUE(fixture.invocation.unbind(selected).ok());
  EXPECT_EQ(reads.stats().live_blocks, 0u);
  EXPECT_EQ(writes.stats().live_blocks, 0u);

  // Reuse the exact two original pool blocks. The other input/output ports
  // remain bound: no whole-Invocation reset or extra binding is performed.
  ASSERT_TRUE(bind_owned(fixture, reads, writes, {5, 6, 7, 8}).ok());
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  {
    auto completion = fixture.invocation.submit(fixture.queues).value();
    ASSERT_TRUE(completion.wait().ok());
  }
  EXPECT_EQ(read_output(fixture, fixture.outputs[0]),
            (std::array<float, 4>{23, 27, 31, 35}));
  EXPECT_EQ(fixture.other.read(), (std::array<float, 4>{3, 5, 7, 9}));
  ASSERT_TRUE(fixture.invocation.unbind(selected).ok());
  EXPECT_EQ(reads.stats().live_blocks, 0u);
  EXPECT_EQ(writes.stats().live_blocks, 0u);
}

TEST(InvocationUnbind, InvalidTailLeavesEarlierPortsAndPreparationUnchanged) {
  Fixture fixture;
  auto reads = pool(), writes = pool();
  ASSERT_TRUE(bind_owned(fixture, reads, writes, {1, 2, 3, 4}).ok());
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  auto foreign_builder = g::GraphBuilder::create().value();
  const auto foreign = foreign_builder.input(custom_test::description()).value();
  for (const auto rejected : {foreign, fixture.internal, fixture.constant}) {
    const std::array selected{fixture.inputs[0], rejected};
    EXPECT_FALSE(fixture.invocation.unbind(selected).ok());
    EXPECT_EQ(reads.stats().live_blocks, 1u);
    EXPECT_EQ(writes.stats().live_blocks, 1u);
  }
  // The failed span never withdrew the earlier owned input, nor invalidated
  // preparation. Submit without an intervening rebind or prepare_replay.
  {
    auto completion = fixture.invocation.submit(fixture.queues).value();
    ASSERT_TRUE(completion.wait().ok());
  }
  EXPECT_EQ(read_output(fixture, fixture.outputs[0]),
            (std::array<float, 4>{7, 11, 15, 19}));
}

TEST(InvocationUnbind, CompletionAndResultRetirementGatesRemainRequired) {
  Fixture fixture;
  auto reads = pool(), writes = pool();
  ASSERT_TRUE(bind_owned(fixture, reads, writes, {1, 2, 3, 4}).ok());
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  const std::array selected{fixture.inputs[0], fixture.outputs[0]};
  auto completion = fixture.invocation.submit(fixture.queues).value();
  EXPECT_TRUE(fixture.invocation.unbind(selected).IsUnavailable());
  ASSERT_TRUE(completion.wait().ok());
  // Successful observation retires the Completion's original use. A retained
  // GraphOutput is a separate live observer and still prevents withdrawal.
  auto result = fixture.invocation.result(fixture.outputs[0]).value();
  completion = {};
  EXPECT_TRUE(fixture.invocation.unbind(selected).IsUnavailable());
  EXPECT_EQ(reads.stats().live_blocks, 1u);
  result.reset();
  ASSERT_TRUE(fixture.invocation.unbind(selected).ok());
  EXPECT_EQ(reads.stats().live_blocks, 0u);
  EXPECT_EQ(writes.stats().live_blocks, 0u);

  Fixture failed;
  auto failed_reads = pool(), failed_writes = pool();
  ASSERT_TRUE(bind_owned(failed, failed_reads, failed_writes, {1, 2, 3, 4}).ok());
  failed.probe.fault = custom_test::Fault::kAcceptedError;
  ASSERT_TRUE(failed.invocation.prepare_replay(failed.queues).ok());
  auto failed_completion = failed.invocation.submit(failed.queues).value();
  EXPECT_FALSE(failed_completion.wait().ok());
  failed_completion = {};
  const std::array failed_ports{failed.inputs[0], failed.outputs[0]};
  EXPECT_TRUE(failed.invocation.unbind(failed_ports).IsFailedPrecondition());
  EXPECT_EQ(failed_reads.stats().live_blocks, 1u);
}

struct RetiredPayload final {
  int *destroyed = nullptr;
  explicit RetiredPayload(int &value) noexcept : destroyed(&value) {}
  RetiredPayload(RetiredPayload &&other) noexcept
      : destroyed(std::exchange(other.destroyed, nullptr)) {}
  ~RetiredPayload() { if (destroyed) ++*destroyed; }
};

TEST(InvocationUnbind, ResultCleanupDiscountsOnlyTheExactPrivateRetiredOwner) {
  namespace cd = c::detail;
  int destroyed = 0;
  auto slot = cd::ResultSlot<RetiredPayload>::create(
      hyle::mem::system_allocator()).value();
  auto other = cd::ResultSlot<RetiredPayload>::create(
      hyle::mem::system_allocator()).value();
  auto writer = slot.begin().value();
  EXPECT_FALSE(cd::can_detach_retired(slot).ok());
  ASSERT_TRUE(writer.emplace(RetiredPayload(destroyed)).ok());
  auto result = writer.publish(c::Completion::completed()).value();
  auto escaped = result.wait().value();
  auto transfer = cd::ResultAccess::take_storage(std::move(result));
  // Even after native retirement, a genuine independently escaped result
  // lease still pins the payload. The private owner's identity cannot waive it.
  EXPECT_FALSE(cd::can_detach_retired(slot, transfer.owner).ok());
  EXPECT_TRUE(cd::can_detach_retired(other, transfer.owner).IsInvalidArgument());
  EXPECT_EQ(destroyed, 0);
  escaped = {};
  ASSERT_TRUE(cd::can_detach_retired(slot, transfer.owner).ok());
  EXPECT_FALSE(cd::can_detach_retired(slot).ok());
  EXPECT_EQ(destroyed, 0); // successful preflight did not mutate the payload
  transfer.owner.reset();
  cd::discard_retired_payload(slot);
  EXPECT_EQ(destroyed, 1);
  auto fresh = slot.begin();
  ASSERT_TRUE(fresh.has_value());
  EXPECT_EQ(destroyed, 1); // cleanup did not enter a new write generation
}
} // namespace
