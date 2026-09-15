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

TEST(GraphCompilationStorage, TypedPrivateBoundsAreAdmittedBeforePrepare) {
  const auto logical = policy_graph();
  Probe probe;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  hyle::graph::CompileOptions options{.compilation_storage =
                                          hyle::graph::CompilationStorageLimits{
                                              .max_resident_bytes = 1U << 20}};
  // A finite resident cap alone must reject unknown even if peak is unlimited.
  auto unknown = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(unknown);
  EXPECT_EQ(probe.preparations, 0);
  options.compilation_storage->unknown =
      hyle::graph::UnknownStoragePolicy::kAllow;
  auto allowed = hyle::graph::compile(logical.graph, context, options).value();
  EXPECT_TRUE(allowed.report.compilation_storage()->has_unknown_storage);
  allowed = {};
  options.compilation_storage->unknown =
      hyle::graph::UnknownStoragePolicy::kReject;
  probe.compilation_storage = {.resident_bytes = 0, .peak_bytes = 0};
  auto plain = hyle::graph::compile(logical.graph, context, options).value();
  const auto base = plain.report.compilation_storage().value();
  plain = {};
  probe.compilation_storage = {.resident_bytes = 4096, .peak_bytes = 1U << 18};
  auto charged = hyle::graph::compile(logical.graph, context, options).value();
  const auto usage = charged.report.compilation_storage().value();
  EXPECT_EQ(usage.resident_bytes, base.resident_bytes + 4096);
  EXPECT_GE(usage.peak_bytes, 1U << 18);
  EXPECT_FALSE(usage.has_unknown_storage);
  charged = {};
  options.compilation_storage->max_peak_bytes = 1U << 17;
  const auto prepares = probe.preparations.load();
  auto peak = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(peak);
  EXPECT_EQ(probe.preparations, prepares);
  EXPECT_EQ(probe.kernels, 0);
  probe.compilation_storage = {.resident_bytes = SIZE_MAX,
                               .peak_bytes = SIZE_MAX};
  options.compilation_storage->max_peak_bytes = SIZE_MAX;
  auto overflow = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(overflow);
  EXPECT_EQ(probe.preparations, prepares);
  probe.compilation_storage = {.resident_bytes = 2, .peak_bytes = 1};
  auto invalid = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInternal());
  EXPECT_EQ(probe.preparations, prepares);
}

TEST(GraphCompilationStorage,
     RejectedPrivateStorageAndPartialArtifactsAreReleased) {
  constexpr ProviderIdentity second_id{
      {0x435553544f4dULL, 32}, "second-storage", "1"};
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(kind);
    Probe first, second;
    second.compilation_storage = {.resident_bytes = 0, .peak_bytes = 0};
    if (kind != 0) {
      first.compilation_storage = {.resident_bytes = 1U << 18,
                                   .peak_bytes = 1U << 19};
      first.fault = Fault::kResourceDecline;
    }
    if (kind == 2)
      first.compilation_storage.peak_bytes = 1U << 23;
    if (kind == 3)
      first.compilation_storage.peak_bytes.reset();
    const std::array factories{
        provider(first).value(),
        make_custom_provider(second_id, cpu::kBackendId,
                             Factory{std::make_unique<Config>(second)})
            .value()};
    auto context =
        Context::create(cpu::device(), {.providers = factories}).value();
    hyle::graph::CompileOptions options{
        .compilation_storage = hyle::graph::CompilationStorageLimits{
            .max_resident_bytes = 1U << 17, .max_peak_bytes = 1U << 20}};
    if (kind == 3)
      options.compilation_storage->unknown =
          hyle::graph::UnknownStoragePolicy::kAllow;
    auto compiled =
        hyle::graph::compile(policy_graph().graph, context, options);
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.step(0)->provider.id, second_id.id);
    EXPECT_EQ(first.preparations, kind == 1 || kind == 3 ? 1 : 0);
    EXPECT_EQ(first.kernels, 0);
    EXPECT_EQ(compiled->report.compilation_storage()->has_unknown_storage,
              kind == 3);
    if (kind == 1) {
      EXPECT_GE(compiled->report.compilation_storage()->peak_bytes, 1U << 19);
    }
    EXPECT_LT(compiled->report.compilation_storage()->resident_bytes, 1U << 17);
  }
}

TEST(GraphInvocationStorage,
     PrivateStorageIsUnknownUnlessTheProviderDeclaresIt) {
  const auto logical = policy_graph();
  Probe probe;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  hyle::graph::CompileOptions options;
  options.invocation_storage =
      hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 20};
  auto unknown = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(unknown);
  EXPECT_TRUE(unknown.error().IsResourceExhausted());
  EXPECT_EQ(probe.invocations, 0);
  options.invocation_storage->unknown =
      hyle::graph::UnknownStoragePolicy::kAllow;
  auto allowed = hyle::graph::compile(logical.graph, context, options).value();
  {
    auto run = allowed.executable.create_invocation().value();
  }
  EXPECT_TRUE(
      allowed.executable.invocation_storage_usage()->has_unknown_storage);
  EXPECT_EQ(allowed.executable.invocation_storage_usage()->live_bytes, 0U);
  options.invocation_storage->unknown =
      hyle::graph::UnknownStoragePolicy::kReject;
  std::size_t plain_bytes = 0;
  Tensors tensors;
  auto queue = context.create_queue().value();
  ASSERT_TRUE(tensors.fill().ok());
  for (std::size_t extra : {0U, 4096U}) {
    probe.invocation_private_bytes = extra;
    auto compiled =
        hyle::graph::compile(logical.graph, context, options).value();
    {
      auto run = compiled.executable.create_invocation().value();
      const auto usage = compiled.executable.invocation_storage_usage().value();
      if (extra == 0)
        plain_bytes = usage.live_bytes;
      EXPECT_EQ(usage.live_bytes, plain_bytes + extra);
      EXPECT_FALSE(usage.has_unknown_storage);
      ASSERT_TRUE(run.bind(logical.input, tensors.in).ok());
      ASSERT_TRUE(run.bind(logical.output, tensors.out).ok());
      for (int repeat = 0; repeat < 4; ++repeat) {
        ASSERT_TRUE(run.submit(queue)->wait().ok());
        EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes,
                  usage.live_bytes);
      }
      EXPECT_EQ(tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
    }
    EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
  }
  probe.invocation_private_bytes = SIZE_MAX;
  auto overflow = hyle::graph::compile(policy_graph(2).graph, context, options);
  ASSERT_FALSE(overflow);
  EXPECT_TRUE(overflow.error().IsResourceExhausted());
}

TEST(GraphInvocationStorage,
     UnknownPrivateRouteCanBeReplacedByAKnownCandidate) {
  constexpr ProviderIdentity known_id{
      {0x435553544f4dULL, 31}, "known-storage", "1"};
  Probe unknown, known;
  known.invocation_private_bytes = 0;
  const std::array factories{
      provider(unknown).value(),
      make_custom_provider(known_id, cpu::kBackendId,
                           Factory{std::make_unique<Config>(known)})
          .value()};
  auto context =
      Context::create(cpu::device(), {.providers = factories}).value();
  auto compiled = hyle::graph::compile(
      policy_graph().graph, context,
      {.invocation_storage =
           hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 20}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(0)->provider.id, known_id.id);
  EXPECT_GT(compiled->report.search().route_plan_rejections, 0U);
  EXPECT_TRUE(compiled->executable.create_invocation());
  EXPECT_FALSE(
      compiled->executable.invocation_storage_usage()->has_unknown_storage);
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

TEST(GraphDiscovery, CollectsLaterLeafCandidatesWithoutPreparingThem) {
  const auto logical = policy_graph(2);
  Probe first, second;
  const std::array factories{provider(first).value(), provider(second).value()};
  auto context =
      Context::create(cpu::device(), {.providers = factories}).value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       {.max_placement_attempts = 2});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.proposals, 2);
  EXPECT_EQ(second.proposals, 2);
  EXPECT_EQ(first.preparations, 2);
  EXPECT_EQ(second.preparations, 0);
  EXPECT_EQ(compiled->report.search().operation_proposals, 4u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 2u);
}

TEST(GraphDiscovery, ProposalStorageBudgetIsSeparateFromPreparation) {
  const auto logical = policy_graph();
  for (auto policy : {hyle::graph::CompileSearchPolicy::kAllowLimited,
                      hyle::graph::CompileSearchPolicy::kRequireUnexhausted}) {
    Probe first, second;
    const std::array factories{provider(first).value(),
                               provider(second).value()};
    auto context =
        Context::create(cpu::device(), {.providers = factories}).value();
    auto compiled = hyle::graph::compile(logical.graph, context,
                                         {.max_placement_attempts = 1,
                                          .max_operation_proposals = 1,
                                          .search_policy = policy});
    EXPECT_EQ(first.proposals, 1);
    EXPECT_EQ(second.proposals, 0);
    EXPECT_EQ(second.preparations, 0);
    if (policy == hyle::graph::CompileSearchPolicy::kRequireUnexhausted) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsResourceExhausted());
      EXPECT_EQ(first.preparations, 0);
      continue;
    }
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_TRUE(compiled->report.search().operation_proposal_limit_reached);
    EXPECT_TRUE(compiled->report.search().optimization_limited());
    EXPECT_EQ(compiled->report.search().operation_proposals, 1u);
    EXPECT_EQ(compiled->report.search().placement_attempts, 1u);
    Tensors tensors;
    ASSERT_TRUE(tensors.fill().ok());
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(run.bind(logical.input, tensors.in).ok());
    ASSERT_TRUE(run.bind(logical.output, tensors.out).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(run.submit(queue).value().wait().ok());
    EXPECT_EQ(tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
    EXPECT_EQ(first.proposals, 1);
  }
}

TEST(GraphDiscovery, BudgetNeverPermitsAnIncompleteExecutable) {
  const auto logical = policy_graph(2);
  Probe probe;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  auto invalid = hyle::graph::compile(logical.graph, context,
                                      {.max_operation_proposals = 0});
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_EQ(probe.proposals, 0);
  auto limited = hyle::graph::compile(logical.graph, context,
                                      {.max_operation_proposals = 1});
  ASSERT_FALSE(limited);
  EXPECT_TRUE(limited.error().IsUnimplemented());
  EXPECT_EQ(probe.proposals, 1);
  EXPECT_EQ(probe.preparations, 0);
  EXPECT_EQ(probe.kernels, 0);
  auto complete = hyle::graph::compile(logical.graph, context,
                                       {.max_operation_proposals = 2});
  ASSERT_TRUE(complete);
  EXPECT_FALSE(complete->report.search().optimization_limited());
  EXPECT_EQ(complete->report.node_count(), 2u);
}

TEST(GraphDiscovery, MissingLaterOperationDoesNotPrepareEarlierKernels) {
  auto builder = hyle::graph::GraphBuilder::create().value();
  const auto input = builder.input(description()).value();
  const auto affine = builder.add(Operation::custom(Affine{1}), input).value();
  const auto negated = builder.add(Unary{UnaryKind::kNeg}, affine).value();
  ASSERT_TRUE(builder.publish(negated).ok());
  const auto graph = std::move(builder).finish().value();
  Probe probe;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  auto compiled = hyle::graph::compile(graph, context);
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsUnimplemented());
  EXPECT_EQ(probe.proposals, 2);
  EXPECT_EQ(probe.preparations, 0);
  EXPECT_EQ(probe.kernels, 0);
}

TEST(GraphDiscovery, FatalLaterProposalStopsBeforeAnyArtifactPreparation) {
  const auto logical = policy_graph();
  Probe first, second;
  second.fault = Fault::kProposalError;
  const std::array factories{provider(first).value(), provider(second).value()};
  auto context =
      Context::create(cpu::device(), {.providers = factories}).value();
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsInternal());
  EXPECT_EQ(first.proposals, 1);
  EXPECT_EQ(second.proposals, 1);
  EXPECT_EQ(first.preparations, 0);
  EXPECT_EQ(second.preparations, 0);
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
  for (const std::string &name :
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

TEST(GraphCustomKernel, StableKeyDoesNotUseTheTransientPrepareToken) {
  const auto logical = policy_graph();
  for (const auto key : {ProviderRouteKey{}, ProviderRouteKey{42}}) {
    Probe probe;
    probe.profile_key = key;
    auto context =
        Context::create(cpu::device(), provider(probe).value()).value();
    std::array<hyle::graph::CompileReport, 2> reports;
    for (std::size_t index = 0; index < reports.size(); ++index) {
      probe.prepare_token = index == 0 ? 7 : 19;
      auto compiled = hyle::graph::compile(logical.graph, context);
      ASSERT_TRUE(compiled) << compiled.error().message();
      reports[index] = compiled->report;
      EXPECT_EQ(reports[index].step(0)->profile_key, key);
      Tensors tensors;
      ASSERT_TRUE(tensors.fill().ok());
      auto run = compiled->executable.create_invocation().value();
      ASSERT_TRUE(run.bind(logical.input, tensors.in).ok());
      ASSERT_TRUE(run.bind(logical.output, tensors.out).ok());
      auto queue = context.create_queue().value();
      ASSERT_TRUE(run.submit(queue).value().wait().ok());
      EXPECT_EQ(tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
    }
    context = {};
    EXPECT_EQ(probe.configurations, 0);
    EXPECT_EQ(probe.kernels, 0);
    EXPECT_EQ(reports[0].step(0)->profile_key, reports[1].step(0)->profile_key);
  }
}

TEST(GraphCustomKernel, ARejectedRouteKeyDoesNotLeakIntoTheFinalReport) {
  const auto logical = policy_graph();
  Probe first, second;
  first.profile_key = ProviderRouteKey{11};
  second.profile_key = ProviderRouteKey{22};
  first.fault = Fault::kUnsupported;
  const std::array providers{provider(first).value(), provider(second).value()};
  auto context =
      Context::create(cpu::device(), {.providers = providers}).value();
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.preparations, 1);
  EXPECT_EQ(first.kernels, 0);
  EXPECT_EQ(second.preparations, 1);
  EXPECT_EQ(compiled->report.step(0)->profile_key, second.profile_key);
  EXPECT_EQ(compiled->report.step(0)->provider_instance,
            context.provider_instance_id(1).value());
  EXPECT_EQ(compiled->report.search().operation_route_rejections, 1u);
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
