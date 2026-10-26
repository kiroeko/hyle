#include <gtest/gtest.h>

#include <array>
#include <span>

#include "../compute/custom_kernel_fixture.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/graph/detail/region_access.h"
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

TEST(InvocationResults, LazyPublicationCoversAllWritersAndRearmsAfterRejectedReuse) {
  Fixture fixture;
  auto reads = pool(), writes = pool(), other_writes = pool();
  ASSERT_TRUE(bind_owned(fixture, reads, writes, {1, 2, 3, 4}).ok());
  auto other = other_writes.allocate(16, 64).value();
  ASSERT_TRUE(fixture.invocation.bind(fixture.outputs[1],
      custom_test::description(), std::move(other)).ok());
  const auto submit = [&]() -> Status {
    HYLE_ASSIGN_OR_RETURN(auto completion, fixture.invocation.submit(fixture.queues));
    return completion.wait();
  };
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  ASSERT_TRUE(submit().ok());
  // Completion alone must not publish: the original writer is reclaimable
  // until the first result, even when another published port will be read first.
  auto reclaimed = g::detail::RegionAccess::take_write(
      fixture.invocation, fixture.outputs[1]);
  ASSERT_TRUE(reclaimed) << reclaimed.error().message();
  ASSERT_TRUE(reclaimed->valid());
  ASSERT_TRUE(fixture.invocation.bind(fixture.outputs[1],
      custom_test::description(), std::move(*reclaimed)).ok());
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  ASSERT_TRUE(submit().ok());

  const auto expect_retained = [&](g::ValueId value,
                                   const std::array<float, 4> &expected) {
    auto output = fixture.invocation.result(value);
    ASSERT_TRUE(output) << output.error().message();
    auto lease = output->storage_lease();
    ASSERT_TRUE(lease) << lease.error().message();
    std::array<float, 4> actual{};
    ASSERT_TRUE(lease->view()->read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, expected);
  };
  expect_retained(fixture.outputs[0], {7, 11, 15, 19});
  auto already_published = g::detail::RegionAccess::take_write(
      fixture.invocation, fixture.outputs[1]);
  ASSERT_TRUE(already_published) << already_published.error().message();
  EXPECT_FALSE(already_published->valid()); // First result covered every writer.
  expect_retained(fixture.outputs[1], {3, 5, 7, 9});
  expect_retained(fixture.outputs[0], {7, 11, 15, 19});

  c::ReadLease retained;
  {
    auto output = fixture.invocation.result(fixture.outputs[1]).value();
    retained = output.storage_lease().value();
  }
  // The next admission can upgrade output 0 before output 1's escaped reader
  // rejects its upgrade. No new native work ran; the completed output must
  // remain exportable, including publication of that upgraded prefix writer.
  auto rejected = fixture.invocation.submit(fixture.queues);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnavailable());
  expect_retained(fixture.outputs[0], {7, 11, 15, 19});
  retained.reset();
  ASSERT_TRUE(submit().ok()); // Same bindings, exclusive reads upgrade again.
  expect_retained(fixture.outputs[0], {7, 11, 15, 19});
  expect_retained(fixture.outputs[1], {3, 5, 7, 9});
}

TEST(InvocationResults, CallerOwnedOutputsRetainExactLogicalRangesAfterRebinding) {
  custom_test::Probe probe;
  auto context = make_context(probe);
  auto queue = context.create_queue().value();
  g::test::Environment environment{{&context, hyle::mem::system_resource()}};
  const std::array queues{g::QueueBinding{environment.lane(0), &queue}};
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(custom_test::description()).value();
  std::array<g::ValueId, 24> outputs;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    outputs[i] = builder.add(c::Operation::custom(custom_test::Affine{
        static_cast<float>(i)}), input).value();
    ASSERT_TRUE(builder.publish(outputs[i]).ok());
  }
  auto graph = std::move(builder).finish().value();
  g::CompileOptions options;
  options.region_selection = g::RegionSelectionPolicy::kDisabled;
  auto compiled = g::compile(graph, environment.environment(), options).value();
  auto invocation = compiled.executable.create_invocation(
      {.resources = g::ResourceOptions{}}).value();
  auto storage = c::LeasePool::create(hyle::mem::system_resource(), 8192,
      {.max_blocks = 26}).value();
  auto input_writer = storage.allocate(64, 64).value();
  const std::array<float, 4> values{1, 2, 3, 4};
  ASSERT_TRUE(input_writer.view()->fill(std::byte{}).ok());
  ASSERT_TRUE(input_writer.view()->subview(0, sizeof(values))->write(
      std::as_bytes(std::span(values))).ok());
  auto input_owner = input_writer.publish().value();
  ASSERT_TRUE(invocation.bind(input, custom_test::description(), input_owner).ok());
  std::array<hyle::mem::ConstBufferView, 24> backing;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto writer = storage.allocate(64, 64).value();
    ASSERT_TRUE(writer.view()->fill(std::byte{0x6d}).ok());
    backing[i] = writer.view().value();
    ASSERT_TRUE(invocation.bind(outputs[i], custom_test::description(), std::move(writer)).ok());
  }
  const auto submit = [&]() -> Status {
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queues));
    return completion.wait();
  };
  ASSERT_TRUE(submit().ok());
  // Read in reverse binding order, including a port near the end of a large
  // binding table. A larger initialized block must still export only its
  // logical tensor, retaining the exact owner rather than another same-shape IO.
  for (std::size_t i = outputs.size(); i != 0; --i) {
    auto result = invocation.result(outputs[i - 1]).value();
    auto lease = result.storage_lease().value();
    const auto view = lease.view().value();
    EXPECT_EQ(lease.size(), 16u);
    EXPECT_EQ(view.native_handle(), backing[i - 1].native_handle());
    EXPECT_EQ(view.offset(), backing[i - 1].offset());
    std::array<float, 4> actual;
    ASSERT_TRUE(view.read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t j = 0; j < actual.size(); ++j)
      EXPECT_EQ(actual[j], 2 * values[j] + static_cast<float>(i - 1));
    std::array<std::byte, 48> suffix;
    ASSERT_TRUE(backing[i - 1].subview(16, suffix.size())->read(suffix).ok());
    for (const auto byte : suffix) EXPECT_EQ(byte, std::byte{0x6d});
  }
  c::ReadLease old;
  {
    auto result = invocation.result(outputs.back()).value();
    old = result.storage_lease().value();
  }
  auto replacement = storage.allocate(64, 64).value();
  ASSERT_TRUE(replacement.view()->fill(std::byte{0x6d}).ok());
  const hyle::mem::ConstBufferView next_backing = replacement.view().value();
  ASSERT_TRUE(invocation.bind(outputs.back(), custom_test::description(), std::move(replacement)).ok());
  ASSERT_TRUE(submit().ok());
  auto result = invocation.result(outputs.back()).value();
  auto fresh = result.storage_lease().value();
  EXPECT_EQ(fresh.view()->native_handle(), next_backing.native_handle());
  EXPECT_EQ(fresh.view()->offset(), next_backing.offset());
  EXPECT_NE(fresh.view()->offset(), old.view()->offset());
  result.reset(); invocation = {};
  std::array<float, 4> old_values;
  ASSERT_TRUE(old.view()->read(std::as_writable_bytes(std::span(old_values))).ok());
  EXPECT_EQ(old_values, (std::array<float, 4>{25, 27, 29, 31}));
}

TEST(InvocationResults, CallerOwnedStorageStillAppliesRetentionBudget) {
  Fixture fixture;
  auto reads = pool(), writes = pool();
  ASSERT_TRUE(bind_owned(fixture, reads, writes, {1, 2, 3, 4}).ok());
  ASSERT_TRUE(fixture.invocation.prepare_replay(fixture.queues).ok());
  {
    auto completion = fixture.invocation.submit(fixture.queues).value();
    ASSERT_TRUE(completion.wait().ok());
  }
  const auto resource = hyle::mem::system_resource();
  const std::array limits{c::ResourceBudgetLimit{resource, 4096}};
  auto prepared_budget = c::ResourceBudget::create({.limits = limits});
  ASSERT_TRUE(prepared_budget) << prepared_budget.error().message();
  auto denied = std::move(*prepared_budget);
  // Budget creation charges its own control metadata. Fill the remaining
  // valid scope here, so rejection exercises output retention, not create().
  auto usage = denied.usage(resource);
  ASSERT_TRUE(usage) << usage.error().message();
  auto occupied = denied.reserve(resource, usage->limit_bytes - usage->live_bytes);
  ASSERT_TRUE(occupied) << occupied.error().message();
  auto result = fixture.invocation.result(fixture.outputs[0]).value();
  auto rejected = result.storage_lease(denied);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  auto retained = result.storage_lease();
  ASSERT_TRUE(retained) << retained.error().message();
  EXPECT_EQ(retained->size(), 16u);
  std::array<float, 4> actual;
  ASSERT_TRUE(retained->view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<float, 4>{7, 11, 15, 19}));
}

TEST(InvocationResults, PublishedInputAndSliceRetainTheirActualSharedOwner) {
  custom_test::Probe probe;
  auto context = make_context(probe);
  g::test::Environment environment{{&context, hyle::mem::system_resource()}};
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(custom_test::description()).value();
  const auto slice = builder.add(c::View::slice(0, 1, 2).value(), input).value();
  ASSERT_TRUE(builder.publish(input).ok());
  ASSERT_TRUE(builder.publish(slice).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = g::compile(graph, environment.environment()).value();
  auto invocation = compiled.executable.create_invocation(
      {.resources = g::ResourceOptions{}}).value();
  auto storage = pool();
  auto writer = storage.allocate(64, 64).value();
  const std::array<float, 4> values{1, 2, 3, 4};
  ASSERT_TRUE(writer.view()->fill(std::byte{}).ok());
  ASSERT_TRUE(writer.view()->subview(0, sizeof(values))->write(
      std::as_bytes(std::span(values))).ok());
  auto owner = writer.publish().value();
  const auto original = owner.view().value();
  const auto generation = owner.generation();
  ASSERT_TRUE(invocation.bind(input, custom_test::description(), owner).ok());
  owner.reset();
  const std::span<const g::QueueBinding> queues{}; // Metadata-only aliases need no Queue.
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  {
    auto completion = invocation.submit(queues).value();
    ASSERT_TRUE(completion.wait().ok());
  }
  c::ReadLease alias;
  {
    auto result = invocation.result(input).value();
    auto exact = result.storage_lease().value();
    EXPECT_EQ(exact.size(), 16u);
    EXPECT_EQ(exact.generation(), generation);
    auto sliced = invocation.result(slice).value();
    alias = sliced.storage_lease().value();
    EXPECT_EQ(alias.generation(), generation);
    EXPECT_EQ(alias.size(), 8u);
    EXPECT_EQ(alias.view()->native_handle(), original.native_handle());
    EXPECT_EQ(alias.view()->offset(), original.offset() + sizeof(float));
  }
  invocation = {};
  EXPECT_EQ(storage.stats().live_blocks, 1u);
  std::array<float, 2> actual;
  ASSERT_TRUE(alias.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<float, 2>{2, 3}));
  alias.reset();
  EXPECT_EQ(storage.stats().live_blocks, 0u);
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
