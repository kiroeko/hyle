#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "../compute/custom_kernel_fixture.h"
#include "hyle/graph/executable.h"

namespace {
using namespace custom_test;

struct PolicyGraph final {
  hyle::graph::Graph graph;
  hyle::graph::ValueId input, output;
};

PolicyGraph policy_graph(std::size_t count = 1) {
  auto builder = hyle::graph::GraphBuilder::create().value();
  const auto input = builder.input(description()).value();
  auto output = input;
  for (std::size_t index = 0; index < count; ++index)
    output = builder.add(Operation::custom(Affine{1}), output).value();
  EXPECT_TRUE(builder.publish(output).ok());
  return {std::move(builder).finish().value(), input, output};
}

TEST(GraphProviderPolicy, GlobalFamilyFiltersPrecedeOperationCallbacks) {
  constexpr ProviderIdentity alternative{
      {0x435553544f4dULL, 3}, "alternative", "1"};
  const auto logical = policy_graph();
  for (int mode = 0; mode < 5; ++mode) {
    Probe first, second;
    const std::array factories{
        provider(first).value(),
        make_custom_provider(alternative, cpu::kBackendId,
                             Factory{std::make_unique<Config>(second)})
            .value()};
    auto context =
        Context::create(cpu::device(), {.providers = factories}).value();
    // Unsorted/repeated lists deliberately exceed inline policy storage.
    const std::array allow{alternative.id, alternative.id, identity.id,
                           alternative.id, identity.id};
    const std::array deny{identity.id};
    const std::array only{alternative.id};
    hyle::graph::CompileOptions options;
    if (mode == 0)
      options.allowed_providers = only;
    else if (mode == 1)
      options.denied_providers = deny;
    else {
      options.allowed_providers = allow;
      options.denied_providers = mode == 2 ? std::span<const ProviderId>(deny)
                                           : std::span<const ProviderId>(allow);
    }
    if (mode == 4)
      options.region_selection = hyle::graph::RegionSelectionPolicy::kDisabled;
    auto compiled = hyle::graph::compile(logical.graph, context, options);
    EXPECT_EQ(first.proposals, 0);
    EXPECT_EQ(first.preparations, 0);
    if (mode >= 3) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsUnimplemented());
      EXPECT_EQ(second.proposals, 0);
      continue;
    }
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.step(0)->provider.id, alternative.id);
    Tensors tensors;
    ASSERT_TRUE(tensors.fill().ok());
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(run.bind(logical.input, tensors.in).ok());
    ASSERT_TRUE(run.bind(logical.output, tensors.out).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(run.submit(queue).value().wait().ok());
    EXPECT_EQ(tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
  }
}

TEST(GraphProviderPolicy, NodesCanSelectDifferentInstancesOfOneFamily) {
  const auto logical = policy_graph(2);
  std::array<Probe, 2> probes;
  const std::array factories{provider(probes[0]).value(),
                             provider(probes[1]).value()};
  auto context =
      Context::create(cpu::device(), {.providers = factories}).value();
  std::array constraints{hyle::graph::NodeProviderConstraint{
                             logical.graph.node_id(0).value(), identity.id,
                             context.provider_instance_id(1).value()},
                         hyle::graph::NodeProviderConstraint{
                             logical.graph.node_id(1).value(), identity.id,
                             context.provider_instance_id(0).value()}};
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       {.node_providers = constraints});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(0)->provider_instance,
            constraints[0].instance);
  EXPECT_EQ(compiled->report.step(1)->provider_instance,
            constraints[1].instance);
  EXPECT_EQ(probes[0].proposals, 1);
  EXPECT_EQ(probes[1].proposals, 1);
  // The policy is borrowed only during compile, not retained for replay.
  constraints = {};
  Tensors tensors;
  ASSERT_TRUE(tensors.fill().ok());
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, tensors.in).ok());
  ASSERT_TRUE(run.bind(logical.output, tensors.out).ok());
  auto queue = context.create_queue().value();
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_TRUE(run.submit(queue).value().wait().ok());
    EXPECT_EQ(tensors.read(), (std::array<float, 4>{7, 11, 15, 19}));
  }
  EXPECT_EQ(probes[0].proposals, 1);
  EXPECT_EQ(probes[1].proposals, 1);
}

TEST(GraphProviderPolicy, RejectsMalformedDeclarationsBeforeProviderWork) {
  const auto logical = policy_graph(2), foreign = policy_graph();
  Probe probe;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  using Constraint = hyle::graph::NodeProviderConstraint;
  const Constraint valid{logical.graph.node_id(0).value(), identity.id};
  const std::array<Constraint, 4> invalid{
      {{},
       {foreign.graph.node_id(0).value(), identity.id},
       {logical.graph.node_id(0).value(), {}},
       valid}};
  for (std::size_t index = 0; index < invalid.size(); ++index) {
    const std::array constraints{valid, invalid[index]};
    auto compiled = hyle::graph::compile(logical.graph, context,
                                         {.node_providers = constraints});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsInvalidArgument());
  }
  const std::array<ProviderId, 2> bad_family{identity.id, {}};
  auto bad_allow = hyle::graph::compile(logical.graph, context,
                                        {.allowed_providers = bad_family});
  auto bad_deny = hyle::graph::compile(logical.graph, context,
                                       {.denied_providers = bad_family});
  ASSERT_FALSE(bad_allow);
  ASSERT_FALSE(bad_deny);
  EXPECT_TRUE(bad_allow.error().IsInvalidArgument());
  EXPECT_TRUE(bad_deny.error().IsInvalidArgument());
  EXPECT_EQ(probe.proposals, 0);
  EXPECT_EQ(probe.preparations, 0);
}

TEST(GraphProviderPolicy, UnavailableOrExcludedRequirementsNeverRelax) {
  const auto logical = policy_graph();
  Probe probe;
  auto factory = provider(probe).value();
  auto context = Context::create(cpu::device(), factory).value();
  ProviderInstanceId old_instance;
  {
    auto old = Context::create(cpu::device(), factory).value();
    old_instance = old.provider_instance_id(0).value();
  }
  constexpr ProviderId unknown{0x12345678, 9};
  const std::array denied{identity.id};
  for (int mode = 0; mode < 4; ++mode) {
    const std::array constraints{hyle::graph::NodeProviderConstraint{
        logical.graph.node_id(0).value(), mode == 0 ? unknown : identity.id,
        mode == 1 ? old_instance : ProviderInstanceId{}}};
    hyle::graph::CompileOptions options{.node_providers = constraints};
    if (mode == 2)
      options.denied_providers = denied;
    if (mode == 3)
      options.allowed_providers = std::span(&unknown, 1);
    auto compiled = hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsUnimplemented());
  }
  EXPECT_EQ(probe.proposals, 0);
  EXPECT_EQ(probe.preparations, 0);
}

TEST(GraphProviderPolicy, LeafResourceDeclineCannotEscapeRequiredInstance) {
  const auto logical = policy_graph();
  for (auto fault : {Fault::kUnsupported, Fault::kResourceDecline}) {
    Probe first, second;
    first.fault = fault;
    const std::array factories{provider(first).value(),
                               provider(second).value()};
    auto context =
        Context::create(cpu::device(), {.providers = factories}).value();
    const std::array constraints{hyle::graph::NodeProviderConstraint{
        logical.graph.node_id(0).value(), identity.id,
        context.provider_instance_id(0).value()}};
    auto constrained = hyle::graph::compile(logical.graph, context,
                                            {.node_providers = constraints});
    ASSERT_FALSE(constrained);
    EXPECT_EQ(second.proposals, 0);
    EXPECT_EQ(first.kernels, 0);
    auto unrestricted = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(unrestricted) << unrestricted.error().message();
    EXPECT_EQ(unrestricted->report.step(0)->provider_instance,
              context.provider_instance_id(1).value());
  }
}

// Keep the raw allocation addressable after logical release, so a stale view
// deterministically sees poison rather than relying on allocator reuse or UB.
struct QuarantinedFactoryAllocator final {
  void *storage = nullptr;
  std::size_t bytes = 0, alignment = 0;
  bool released = false;
  ~QuarantinedFactoryAllocator() {
    if (storage != nullptr)
      hyle::mem::system_allocator().deallocate(storage, bytes, alignment);
  }
  hyle::mem::AllocatorRef ref() noexcept {
    return {&allocate, &deallocate, &reallocate, this};
  }
  static void *allocate(void *state, std::size_t size,
                        std::size_t align) noexcept {
    auto &self = *static_cast<QuarantinedFactoryAllocator *>(state);
    if (self.storage != nullptr)
      return nullptr;
    self.bytes = size;
    self.alignment = align;
    self.storage = hyle::mem::system_allocator().allocate(size, align);
    return self.storage;
  }
  static void deallocate(void *state, void *pointer, std::size_t size,
                         std::size_t) noexcept {
    if (pointer == nullptr)
      return;
    auto &self = *static_cast<QuarantinedFactoryAllocator *>(state);
    self.released = true;
    std::memset(pointer, '!', size);
  }
  static void *reallocate(void *state, void *pointer, std::size_t old_size,
                          std::size_t new_size, std::size_t align) noexcept {
    if (new_size == 0) {
      deallocate(state, pointer, old_size, align);
      return nullptr;
    }
    return pointer == nullptr ? allocate(state, new_size, align) : nullptr;
  }
};

TEST(GraphCustomKernel, ReportOwnsProviderTextWithoutKeepingExecutionAlive) {
  for (const std::string name :
       {std::string{}, std::string("na\0me", 5), std::string(1024, 'n')}) {
    const std::string version(513, 'v');
    Probe probe;
    QuarantinedFactoryAllocator allocator;
    hyle::graph::CompileReport report;
    {
      auto factory =
          make_custom_provider({identity.id, name, version}, cpu::kBackendId,
                               Factory{std::make_unique<Config>(probe)},
                               allocator.ref())
              .value();
      auto context = Context::create(cpu::device(), factory).value();
      auto builder = hyle::graph::GraphBuilder::create().value();
      auto input = builder.input(description()).value();
      auto first = builder.add(Operation::custom(Affine{3}), input).value();
      auto output = builder.add(Operation::custom(Affine{1}), first).value();
      ASSERT_TRUE(builder.publish(output).ok());
      auto graph = std::move(builder).finish().value();
      auto compiled = hyle::graph::compile(graph, context);
      ASSERT_TRUE(compiled) << compiled.error().message();
      report = compiled->report;
      ASSERT_EQ(report.step_count(), 2u);
      EXPECT_EQ(report.step(0)->provider.name.data(),
                report.step(1)->provider.name.data());
      EXPECT_EQ(report.provider().version.data(),
                report.step(0)->provider.version.data());
    }
    EXPECT_TRUE(allocator.released);
    EXPECT_EQ(probe.configurations, 0);
    EXPECT_EQ(probe.kernels, 0);
    auto copy = report;
    report = {};
    auto moved = std::move(copy);
    EXPECT_FALSE(copy);
    EXPECT_EQ(moved.provider().name, name);
    EXPECT_EQ(moved.provider().version, version);
    EXPECT_EQ(moved.step(0)->provider.name, name);
    EXPECT_EQ(moved.step(1)->provider.version, version);
  }
}

TEST(GraphCustomKernel, ReportDistinguishesTextOfSameFamilyInstances) {
  const std::array names{std::string(700, 'a'), std::string(800, 'b')};
  const std::array versions{std::string("v1"), std::string("v2")};
  std::array<Probe, 2> probes;
  std::array<QuarantinedFactoryAllocator, 2> allocators;
  hyle::graph::CompileReport report;
  {
    std::array<Context, 2> contexts;
    for (std::size_t index = 0; index < contexts.size(); ++index) {
      auto factory =
          make_custom_provider({identity.id, names[index], versions[index]},
                               cpu::kBackendId,
                               Factory{std::make_unique<Config>(probes[index])},
                               allocators[index].ref())
              .value();
      contexts[index] = Context::create(cpu::device(), factory).value();
    }
    auto builder = hyle::graph::GraphBuilder::create().value();
    auto input = builder.input(description()).value();
    auto first = builder.add(Operation::custom(Affine{3}), input).value();
    auto output = builder.add(Operation::custom(Affine{1}), first).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    const std::array storage{
        hyle::graph::StorageDomain{hyle::mem::system_resource()}};
    const std::array targets{
        hyle::graph::ExecutionTarget{&contexts[0],
                                     hyle::graph::StorageDomainId{0}},
        hyle::graph::ExecutionTarget{&contexts[1],
                                     hyle::graph::StorageDomainId{0}}};
    const std::array placements{
        hyle::graph::NodePlacement{graph.node_id(0).value(),
                                   hyle::graph::ExecutionTargetId{0},
                                   hyle::graph::PlacementStrength::kRequired},
        hyle::graph::NodePlacement{graph.node_id(1).value(),
                                   hyle::graph::ExecutionTargetId{1},
                                   hyle::graph::PlacementStrength::kRequired}};
    auto compiled = hyle::graph::compile(
        graph, {.storage_domains = storage, .targets = targets},
        {.node_placements = placements});
    ASSERT_TRUE(compiled) << compiled.error().message();
    report = compiled->report;
  }
  ASSERT_EQ(report.step_count(), 2u);
  EXPECT_TRUE(report.mixed_providers());
  EXPECT_FALSE(report.provider().valid());
  EXPECT_NE(report.step(0)->provider_instance,
            report.step(1)->provider_instance);
  for (std::size_t index = 0; index < probes.size(); ++index) {
    EXPECT_TRUE(allocators[index].released);
    EXPECT_EQ(probes[index].configurations, 0);
    EXPECT_EQ(probes[index].kernels, 0);
    EXPECT_EQ(report.step(index)->provider.id, identity.id);
    EXPECT_EQ(report.step(index)->provider.name, names[index]);
    EXPECT_EQ(report.step(index)->provider.version, versions[index]);
  }
}

TEST(GraphCustomKernel, PublicSchemaAndOwningFactoryUseTheSamePreparedRuntime) {
  Probe probe;
  Tensors tensors;
  ASSERT_TRUE(tensors.fill().ok());
  {
    auto factory = provider(probe).value();
    auto context = Context::create(cpu::device(), factory).value();
    auto queue = context.create_queue().value();
    auto builder = hyle::graph::GraphBuilder::create().value();
    auto input = builder.input(description()).value();
    auto first = builder.add(Operation::custom(Affine{3}), input).value();
    auto output = builder.add(Operation::custom(Affine{1}), first).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    auto compiled = hyle::graph::compile(graph, context);
    ASSERT_TRUE(compiled) << compiled.error().message();
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(input, tensors.in).ok());
    ASSERT_TRUE(invocation.bind(output, tensors.out).ok());
    factory = {};
    compiled->executable = {};
    const int prepares = probe.preparations, proposals = probe.proposals;
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto done = invocation.submit(queue);
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(tensors.read(), (std::array<float, 4>{11, 15, 19, 23}));
    }
    EXPECT_EQ(probe.preparations, prepares);
    EXPECT_EQ(probe.proposals, proposals);
    EXPECT_EQ(probe.executions, 6);
    EXPECT_EQ(probe.configurations, 1);
    context = {}; // admission closes; existing objects retain implementation
    EXPECT_EQ(probe.configurations, 1);
  }
  EXPECT_EQ(probe.configurations, 0);
  EXPECT_EQ(probe.kernels, 0);
  EXPECT_EQ(probe.invocations, 0);
}

} // namespace
