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
