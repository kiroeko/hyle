#include <gtest/gtest.h>

#include <array>

#include "../compute/custom_kernel_fixture.h"
#include "hyle/graph/detail/input_source.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/result.h"
#include "test_environment.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace g = hyle::graph;
namespace gd = g::detail;
using hyle::Result;
using hyle::Status;

struct Source final {
  int references = 0, routes = 0, observations = 0;
};
const gd::InputSourceVTable source_table{
    [](void *p) noexcept { ++static_cast<Source *>(p)->references; },
    [](void *p) noexcept { --static_cast<Source *>(p)->references; },
    [](void *p, const cd::QueueRef *) noexcept -> Result<gd::InputDependencyRoute> {
      ++static_cast<Source *>(p)->routes;
      return gd::InputDependencyRoute::kHostWait;
    },
    [](void *p) noexcept -> Result<c::CompletionPoll> {
      ++static_cast<Source *>(p)->observations;
      return c::CompletionPoll::kComplete;
    },
    [](void *p) noexcept {
      ++static_cast<Source *>(p)->observations;
      return Status::Ok();
    },
    [](void *, cd::QueueSequence &) noexcept {
      return cd::EnqueueResult::rejected(Status::Internal("HostWait source imported"));
    },
    [](void *) noexcept { return Status::Ok(); }};

c::Context context(custom_test::Probe &probe) {
  const std::array providers{c::ProviderBinding{custom_test::provider(probe).value()}};
  return c::Context::create(c::cpu::device(), {.providers = providers}).value();
}
c::ReadLease input_lease(const std::array<float, 4> &values) {
  auto backing = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64).value();
  EXPECT_TRUE(backing.view().write(std::as_bytes(std::span(values))).ok());
  return c::ReadLease::adopt(std::move(backing)).value();
}
struct Fixture final {
  custom_test::Probe probe;
  c::Context cpu = context(probe);
  c::Queue queue = cpu.create_queue().value();
  g::test::Environment environment{{&cpu, hyle::mem::system_resource()}};
  custom_test::Tensors tensors;
  g::Graph graph;
  g::ValueId input, output;
  g::CompiledGraph compiled;
  g::Invocation run;
  const std::array<g::QueueBinding, 1> queues{{{environment.lane(0), &queue}}};

  Fixture() {
    EXPECT_TRUE(tensors.fill().ok());
    auto builder = g::GraphBuilder::create().value();
    input = builder.input(custom_test::description()).value();
    output = builder.add(c::Operation::custom(custom_test::Affine{1}), input).value();
    EXPECT_TRUE(builder.publish(output).ok());
    graph = std::move(builder).finish().value();
    compiled = g::compile(graph, environment.environment(),
        {.region_selection = g::RegionSelectionPolicy::kDisabled}).value();
    run = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    EXPECT_TRUE(run.bind(output, tensors.out).ok());
  }
};
constexpr std::array<float, 4> first{1, 2, 3, 4};
constexpr std::array<float, 4> second{5, 6, 7, 8};

TEST(ConfigurationInput, OrdinaryBindingsCannotReplacePreparedMetadata) {
  Source producer;
  Fixture fixture;
  auto original = input_lease(first);
  auto replacement = input_lease(second);
  auto source = gd::InputSource::acquire(&producer, &source_table, &producer).value();
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), original).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());

  EXPECT_TRUE(fixture.run.bind(fixture.input, fixture.tensors.in).IsFailedPrecondition());
  const auto mutable_input = c::TensorView::create(custom_test::description(),
                                                 fixture.tensors.input.view()).value();
  EXPECT_TRUE(fixture.run.bind(fixture.input, mutable_input).IsFailedPrecondition());
  EXPECT_TRUE(fixture.run.bind(fixture.input, custom_test::description(),
                               replacement).IsFailedPrecondition());
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256,
                                  {.max_blocks = 1}).value();
  auto writer = pool.allocate(16).value();
  EXPECT_TRUE(fixture.run.bind(fixture.input, custom_test::description(),
                               std::move(writer)).IsFailedPrecondition());
  EXPECT_TRUE(writer.valid());
  EXPECT_TRUE(gd::InputAccess::bind(fixture.run, fixture.input,
                                   fixture.tensors.in, source).IsFailedPrecondition());
  EXPECT_EQ(producer.references, 1);
  EXPECT_EQ(producer.routes, 0);
  EXPECT_EQ(producer.observations, 0);

  // None of these rejected replacements invalidated the prepared replay or
  // detached its original immutable owner.
  auto completion = fixture.run.submit(fixture.queues);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
  auto result = fixture.run.result(fixture.output).value();
  ASSERT_TRUE(result.tensor());
  EXPECT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), replacement).IsUnavailable());
}

TEST(ConfigurationInput, FailedColdReplacementPreservesReplayAndResetUnlocksInput) {
  Fixture fixture;
  auto original = input_lease(first);
  auto replacement = input_lease(second);
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), original).ok());
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), replacement).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());

  const auto wrong = c::TensorDesc::dense(c::DataType::kFloat32,
                                         std::array<std::int64_t, 1>{2}).value();
  // This lease covers the supplied tensor. The original Graph validation,
  // after the trusted replacement gate, rejects its mismatched description.
  EXPECT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
                                                   wrong, original).IsInvalidArgument());
  EXPECT_TRUE(fixture.run.bind(fixture.input, fixture.tensors.in).IsFailedPrecondition());
  {
    auto completion = fixture.run.submit(fixture.queues);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{11, 13, 15, 17}));
  }
  ASSERT_TRUE(fixture.run.reset().ok());
  ASSERT_TRUE(fixture.run.bind(fixture.input, fixture.tensors.in).ok());
  ASSERT_TRUE(fixture.run.bind(fixture.output, fixture.tensors.out).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  auto completion = fixture.run.submit(fixture.queues);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
}

TEST(ConfigurationInput, RefreshedConfigurationUsesCurrentLeaseAndReplacementStorage) {
  Fixture fixture;
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256,
                                  {.max_blocks = 1}).value();
  auto writer = pool.allocate(16).value();
  ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(first))).ok());
  auto ready = writer.publish().value();
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), ready).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  {
    auto done = fixture.run.submit(fixture.queues).value();
    ASSERT_TRUE(done.wait().ok());
    auto result = fixture.run.result(fixture.output).value();
    EXPECT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).IsUnavailable());
    EXPECT_FALSE(ready.try_write());
    EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
  }
  ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
  // Reacquire exclusive permission and publish a fresh read permission at the same
  // allocation. The new episode must use its current read permission, while
  // retained physical TensorView metadata may still describe this storage.
  auto refreshed = ready.try_write();
  ASSERT_TRUE(refreshed) << refreshed.error().message();
  ASSERT_TRUE(refreshed->view()->write(std::as_bytes(std::span(second))).ok());
  ready = refreshed->publish().value();
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), ready).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  const auto wrong = c::TensorDesc::dense(c::DataType::kFloat32,
                                         std::array<std::int64_t, 1>{2}).value();
  EXPECT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      wrong, ready).IsInvalidArgument());
  {
    auto done = fixture.run.submit(fixture.queues).value();
    ASSERT_TRUE(done.wait().ok());
    EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{11, 13, 15, 17}));
  }
  // Moving to a different owner/address, then back, must not revive a stale
  // same-span preparation; both replacements retain ordinary validation.
  auto replacement = input_lease(first);
  for (const auto *lease : {&replacement, &ready}) {
    ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
    ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
        custom_test::description(), *lease).ok());
    ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
    auto done = fixture.run.submit(fixture.queues).value();
    ASSERT_TRUE(done.wait().ok());
    EXPECT_EQ(fixture.tensors.read(), lease == &replacement
        ? (std::array<float, 4>{3, 5, 7, 9})
        : (std::array<float, 4>{11, 13, 15, 17}));
  }
  ASSERT_TRUE(fixture.run.reset().ok());
  EXPECT_TRUE(ready.try_write());
  EXPECT_FALSE(fixture.run.prepare_replay(fixture.queues).ok());
}

TEST(ConfigurationInput, CarrierQueryUsesOnlyThisPlansPrivateInput) {
  Fixture fixture;
  auto resource = gd::RegionAccess::input_resource(fixture.compiled.executable, fixture.input);
  ASSERT_TRUE(resource) << resource.error().message();
  EXPECT_EQ(*resource, hyle::mem::system_resource());
  const auto output = gd::RegionAccess::input_resource(fixture.compiled.executable, fixture.output);
  ASSERT_FALSE(output);
  EXPECT_TRUE(output.error().IsInvalidArgument());
  auto builder = g::GraphBuilder::create().value();
  const auto foreign = builder.input(custom_test::description()).value();
  const auto rejected = gd::RegionAccess::input_resource(fixture.compiled.executable, foreign);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
}

TEST(ConfigurationInput, LayoutResetPreservesCallerLeasesAndSourceButExpiresMetadata) {
  Source producer;
  custom_test::Probe probe;
  auto provider = custom_test::provider(probe);
  ASSERT_TRUE(provider) << provider.error().message();
  const std::array providers{c::ProviderBinding{*provider}};
  auto created = c::Context::create(c::cpu::device(), {.providers = providers});
  ASSERT_TRUE(created) << created.error().message();
  auto cpu = std::move(*created);
  auto queue = cpu.create_queue().value();
  g::test::Environment environment{{&cpu, hyle::mem::system_resource()}};
  custom_test::Tensors tensors;
  ASSERT_TRUE(tensors.fill().ok());
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 3> inputs, outputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = builder.input(custom_test::description()).value();
    outputs[i] = builder.add(c::Operation::custom(custom_test::Affine{1}), inputs[i]).value();
    ASSERT_TRUE(builder.publish(outputs[i]).ok());
  }
  auto graph = std::move(builder).finish().value();
  auto prepared = g::compile(graph, environment.environment(),
      {.region_selection = g::RegionSelectionPolicy::kDisabled});
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto compiled = std::move(*prepared);
  auto run = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  const std::array queues{g::QueueBinding{environment.lane(0), &queue}};
  auto stable_pool = c::LeasePool::create(hyle::mem::system_resource(), 256,
                                         {.max_blocks = 1}).value();
  auto stable_writer = stable_pool.allocate(16).value();
  ASSERT_TRUE(stable_writer.view()->write(std::as_bytes(std::span(second))).ok());
  auto stable = stable_writer.publish().value();
  auto source = gd::InputSource::acquire(&producer, &source_table, &producer).value();
  ASSERT_TRUE(run.bind(inputs[1], custom_test::description(), stable).ok());
  ASSERT_TRUE(gd::InputAccess::bind(run, inputs[2], tensors.in, source).ok());
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256,
                                  {.max_blocks = 1}).value();
  auto last_output = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64).value();
  ASSERT_TRUE(run.bind(outputs[0], tensors.out).ok());
  ASSERT_TRUE(run.bind(outputs[1], custom_test::description(), pool.allocate(16).value()).ok());
  ASSERT_TRUE(run.bind(outputs[2], c::TensorView::create(
      custom_test::description(), last_output.view()).value()).ok());
  ASSERT_TRUE(gd::RegionAccess::reserve_reads(run, 4, {}).ok());

  int previous_routes = 0;
  for (int episode = 0; episode < 2; ++episode) {
    if (episode) {
      ASSERT_TRUE(gd::RegionAccess::reset_configuration(run).ok());
      // Binding and this local handle remain; old active/candidate preparation
      // must release their source owners and dependency decisions.
      EXPECT_EQ(producer.references, 2);
      EXPECT_FALSE(run.prepare_replay(queues).ok()); // New metadata is mandatory.
      EXPECT_FALSE(run.result(outputs[0]));
      EXPECT_FALSE(run.submit(queues));
    }
    ASSERT_TRUE(gd::RegionAccess::bind_configuration(run, inputs[0],
        custom_test::description(), input_lease(episode ? second : first)).ok());
    ASSERT_TRUE(run.prepare_replay(queues).ok());
    EXPECT_GT(producer.routes, previous_routes);
    previous_routes = producer.routes;
    {
      auto done = run.submit(queues).value();
      ASSERT_TRUE(done.wait().ok());
      const std::array<std::array<float, 4>, 3> expected{{
          episode ? std::array<float, 4>{11, 13, 15, 17} : std::array<float, 4>{3, 5, 7, 9},
          {11, 13, 15, 17}, {3, 5, 7, 9}}};
      for (std::size_t port = 0; port < outputs.size(); ++port) {
        auto result = run.result(outputs[port]).value();
        const int references = producer.references;
        EXPECT_TRUE(gd::RegionAccess::reset_configuration(run).IsUnavailable());
        EXPECT_EQ(producer.references, references);
        std::array<float, 4> actual;
        ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, expected[port]);
      }
    }
  }
  ASSERT_TRUE(run.reset().ok());
  EXPECT_EQ(producer.references, 1);
  EXPECT_TRUE(stable.try_write());
  EXPECT_FALSE(run.prepare_replay(queues).ok());
}
} // namespace
