#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "../compute/custom_kernel_fixture.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/detail/input_source.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/result.h"
#include "../../src/graph/plan_state.h"
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

c::Context context(custom_test::Probe &probe, bool copy_input = false) {
  const std::array providers{c::ProviderBinding{custom_test::provider(probe).value()},
      c::ProviderBinding{c::native::provider()}};
  return c::Context::create(c::cpu::device(),
      {.providers = std::span(providers).first(copy_input ? 2u : 1u)}).value();
}
c::ReadLease input_lease(const std::array<float, 4> &values) {
  auto backing = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64).value();
  EXPECT_TRUE(backing.view().write(std::as_bytes(std::span(values))).ok());
  return c::ReadLease::adopt(std::move(backing)).value();
}
struct Fixture final {
  custom_test::Probe probe;
  c::Context cpu;
  c::Queue queue = cpu.create_queue().value();
  g::test::Environment environment{{&cpu, hyle::mem::system_resource()}};
  custom_test::Tensors tensors;
  c::TensorDesc input_description = custom_test::description();
  g::Graph graph;
  g::ValueId input, output;
  g::CompiledGraph compiled;
  g::Invocation run;
  const std::array<g::QueueBinding, 1> queues{{{environment.lane(0), &queue}}};

  explicit Fixture(bool copy_input = false, bool token_view = false)
      : cpu(context(probe, copy_input)) {
    EXPECT_TRUE(tensors.fill().ok());
    auto builder = g::GraphBuilder::create().value();
    if (token_view)
      input_description = c::TensorDesc::dense(c::DataType::kFloat32,
          std::array<std::int64_t, 2>{2, 2}).value();
    input = builder.input(input_description).value();
    auto affine_input = input;
    if (token_view)
      affine_input = builder.add(c::View::reshape(c::TensorShape::create(
          std::array<std::int64_t, 1>{4}).value()).value(), affine_input).value();
    // Copy keeps Affine's optional writable alias off the external root.
    // Readonly token Views before Copy still belong to the complete alias
    // table; the original affine math and output remain unchanged.
    if (copy_input) affine_input = builder.add(c::Copy{}, affine_input).value();
    output = builder.add(c::Operation::custom(custom_test::Affine{1}), affine_input).value();
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
  const auto *state = gd::InvocationAccess::state(fixture.run);
  auto original = input_lease(first);
  auto replacement = input_lease(second);
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), original).ok());
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), replacement).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  ASSERT_TRUE(std::binary_search(state->plan->external_input_aliases.begin(),
      state->plan->external_input_aliases.end(), fixture.input.ordinal()));
  ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
  EXPECT_FALSE(gd::reusable_configuration_spans(*state)); // Original optional alias.
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), replacement).ok());
  EXPECT_FALSE(gd::reusable_configuration_spans(*state));
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
  Fixture fixture(true);
  const auto *state = gd::InvocationAccess::state(fixture.run);
  ASSERT_TRUE(state->plan->external_input_aliases.empty());
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 256,
                                  {.max_blocks = 1}).value();
  auto writer = pool.allocate(16).value();
  ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(first))).ok());
  auto ready = writer.publish().value();
  const auto generation = ready.generation();
  const auto original_span = ready.view().value();
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
  ASSERT_TRUE(gd::reusable_configuration_spans(*state));
  EXPECT_FALSE(state->bindings_prepared);
  // try_write proves that the old read guard was released, but preserves the
  // allocation generation. Actually free this block, then allocate again from
  // the max-one-block pool to test a new generation at the same physical tuple.
  auto refreshed = ready.try_write();
  ASSERT_TRUE(refreshed) << refreshed.error().message();
  EXPECT_EQ(refreshed->generation(), generation);
  refreshed->reset();
  ASSERT_EQ(pool.stats().live_blocks, 0u);
  auto replacement_writer = pool.allocate(16);
  ASSERT_TRUE(replacement_writer) << replacement_writer.error().message();
  auto replacement_view = replacement_writer->view();
  ASSERT_TRUE(replacement_view) << replacement_view.error().message();
  ASSERT_TRUE(replacement_view->write(std::as_bytes(std::span(second))).ok());
  ready = replacement_writer->publish().value();
  EXPECT_NE(ready.generation(), generation);
  const auto current_span = ready.view().value();
  EXPECT_EQ(current_span.native_handle(), original_span.native_handle());
  EXPECT_EQ(current_span.offset(), original_span.offset());
  EXPECT_EQ(current_span.size(), original_span.size());
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), ready).ok());
  // Physical reuse does not preserve the previous read permission or replay.
  ASSERT_TRUE(gd::reusable_configuration_spans(*state));
  EXPECT_FALSE(state->replay_prepared);
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  EXPECT_FALSE(gd::reusable_configuration_spans(*state)); // Call-scoped hint.
  EXPECT_FALSE(ready.try_write()); // The new generation is held by this run.
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
    ASSERT_TRUE(gd::reusable_configuration_spans(*state));
    ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
        custom_test::description(), *lease).ok());
    EXPECT_FALSE(gd::reusable_configuration_spans(*state)); // Different owner/address.
    ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
    auto done = fixture.run.submit(fixture.queues).value();
    ASSERT_TRUE(done.wait().ok());
    EXPECT_EQ(fixture.tensors.read(), lease == &replacement
        ? (std::array<float, 4>{3, 5, 7, 9})
        : (std::array<float, 4>{11, 13, 15, 17}));
  }
  // A different actual Queue still follows the original preparation/admission
  // path, even though every physical TensorView is equal to the last episode.
  auto another_queue = fixture.cpu.create_queue().value();
  const std::array another_queues{g::QueueBinding{fixture.environment.lane(0), &another_queue}};
  ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), ready).ok());
  ASSERT_TRUE(gd::reusable_configuration_spans(*state));
  ASSERT_TRUE(fixture.run.prepare_replay(another_queues).ok());
  {
    auto done = fixture.run.submit(another_queues).value();
    ASSERT_TRUE(done.wait().ok());
    EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{11, 13, 15, 17}));
  }
  ASSERT_TRUE(fixture.run.reset().ok());
  EXPECT_FALSE(gd::reusable_configuration_spans(*state));
  EXPECT_TRUE(ready.try_write());
  EXPECT_FALSE(fixture.run.prepare_replay(fixture.queues).ok());
}

TEST(ConfigurationInput, IncompleteConfigurationCannotKeepThePhysicalShortcut) {
  Fixture fixture(true);
  const auto *state = gd::InvocationAccess::state(fixture.run);
  ASSERT_TRUE(state->plan->external_input_aliases.empty());
  const auto ready = input_lease(first);
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), ready).ok());
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
  ASSERT_TRUE(gd::reusable_configuration_spans(*state));
  EXPECT_TRUE(fixture.run.prepare_replay(fixture.queues).IsFailedPrecondition());
  EXPECT_FALSE(gd::reusable_configuration_spans(*state));
  EXPECT_FALSE(state->bindings_prepared);
  EXPECT_FALSE(fixture.run.submit(fixture.queues));
  ASSERT_TRUE(gd::RegionAccess::bind_configuration(fixture.run, fixture.input,
      custom_test::description(), ready).ok());
  EXPECT_FALSE(gd::reusable_configuration_spans(*state));
  ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
  auto done = fixture.run.submit(fixture.queues).value();
  ASSERT_TRUE(done.wait().ok());
  EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
}

TEST(ConfigurationInput, SamePhysicalSourceReplacementRequalifiesItsCurrentProducer) {
  for (const bool token_view : {false, true}) {
    SCOPED_TRACE(token_view ? "two-dimensional token flatten" : "original no-alias input");
    Source original_producer, replacement_producer;
    c::ReadLease moved_input;
    Fixture fixture(true, token_view);
    const auto *state = gd::InvocationAccess::state(fixture.run);
    if (token_view) {
      ASSERT_TRUE(std::binary_search(state->plan->external_input_aliases.begin(),
          state->plan->external_input_aliases.end(), fixture.input.ordinal()));
      const auto step = state->plan->node_steps[fixture.graph.node_id(0)->ordinal()];
      ASSERT_LT(step, state->plan->steps.size());
      EXPECT_EQ(state->plan->steps[step].kind, gd::PlanStepKind::kView);
      EXPECT_FALSE(state->plan->steps[step].materializes);
    } else ASSERT_TRUE(state->plan->external_input_aliases.empty());
    const auto tensor = c::ConstTensorView::create(fixture.input_description,
        fixture.tensors.input.view()).value();
    auto original = gd::InputSource::acquire(&original_producer, &source_table,
        &original_producer).value();
    auto replacement = gd::InputSource::acquire(&replacement_producer, &source_table,
        &replacement_producer).value();
    ASSERT_TRUE(gd::InputAccess::bind(fixture.run, fixture.input, tensor, original).ok());
    ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
    {
      auto done = fixture.run.submit(fixture.queues).value();
      const auto status = done.wait();
      ASSERT_TRUE(status.ok()) << status.message();
      EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
    }
    const auto old_routes = original_producer.routes;
    const auto old_observations = original_producer.observations;
    ASSERT_GT(old_routes, 0);
    ASSERT_GT(old_observations, 0);
    ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
    ASSERT_TRUE(gd::reusable_configuration_spans(*state));
    EXPECT_FALSE(state->bindings_prepared);
    EXPECT_EQ(original_producer.references, 2); // This handle and the retained binding.

    // The physical argument is unchanged. Producer identity and readiness must
    // still be prepared anew; the old Source is not authority for this episode.
    ASSERT_TRUE(gd::InputAccess::bind(fixture.run, fixture.input, tensor, replacement).ok());
    ASSERT_TRUE(gd::reusable_configuration_spans(*state));
    EXPECT_FALSE(state->replay_prepared);
    EXPECT_FALSE(fixture.run.submit(fixture.queues));
    EXPECT_EQ(original_producer.references, 1);
    EXPECT_EQ(replacement_producer.references, 2);
    EXPECT_EQ(replacement_producer.routes, 0);
    EXPECT_EQ(replacement_producer.observations, 0);
    ASSERT_TRUE(fixture.tensors.input.view().write(std::as_bytes(std::span(second))).ok());
    ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
    EXPECT_FALSE(gd::reusable_configuration_spans(*state)); // Preparation consumed the candidate.
    EXPECT_EQ(replacement_producer.routes, 1);
    auto report = fixture.run.input_source_report();
    ASSERT_TRUE(report) << report.error().message();
    EXPECT_EQ(report->source_count, 1u);
    EXPECT_EQ(report->host_wait_count, 1u);
    {
      auto done = fixture.run.submit(fixture.queues).value();
      const auto status = done.wait();
      ASSERT_TRUE(status.ok()) << status.message();
      EXPECT_GT(replacement_producer.observations, 0);
      EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{11, 13, 15, 17}));
    }
    EXPECT_EQ(original_producer.routes, old_routes);
    EXPECT_EQ(original_producer.observations, old_observations);
    if (token_view) {
      ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
      ASSERT_TRUE(gd::reusable_configuration_spans(*state));
      moved_input = input_lease(first);
      const auto storage = moved_input.view().value();
      ASSERT_NE(storage.native_handle(), tensor.storage().native_handle());
      const auto moved_tensor = c::ConstTensorView::create(fixture.input_description, storage).value();
      const auto routes = replacement_producer.routes;
      const auto observations = replacement_producer.observations;
      ASSERT_TRUE(gd::InputAccess::bind(fixture.run, fixture.input, moved_tensor, replacement).ok());
      EXPECT_FALSE(gd::reusable_configuration_spans(*state));
      EXPECT_FALSE(state->replay_prepared);
      EXPECT_FALSE(fixture.run.submit(fixture.queues));
      ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
      EXPECT_EQ(replacement_producer.routes, routes + 1);
      {
        auto done = fixture.run.submit(fixture.queues).value();
        const auto status = done.wait();
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(fixture.tensors.read(), (std::array<float, 4>{3, 5, 7, 9}));
      }
      EXPECT_GT(replacement_producer.observations, observations);
      EXPECT_EQ(original_producer.routes, old_routes);
      EXPECT_EQ(original_producer.observations, old_observations);
    }
    ASSERT_TRUE(fixture.run.reset().ok());
    EXPECT_EQ(original_producer.references, 1);
    EXPECT_EQ(replacement_producer.references, 1);
  }
}

TEST(ConfigurationInput, PartialAndOffsetViewsDiscardTheGeometryCandidateOnRebind) {
  for (const std::int64_t start : {0, 1}) {
    SCOPED_TRACE(start);
    Source producer;
    custom_test::Probe probe;
    auto cpu = context(probe, true);
    auto queue = cpu.create_queue().value();
    g::test::Environment environment{{&cpu, hyle::mem::system_resource()}};
    const std::array queues{g::QueueBinding{environment.lane(0), &queue}};
    custom_test::Tensors tensors;
    ASSERT_TRUE(tensors.fill().ok());
    const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
    auto input = hyle::mem::Buffer::create(hyle::mem::system_resource(),
        sizeof(values), 64).value();
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
    const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 1>{6}).value();
    const auto tensor = c::ConstTensorView::create(description, input.view()).value();
    auto builder = g::GraphBuilder::create().value();
    const auto root = builder.input(description).value();
    const auto view = builder.add(c::View::slice(0, start, 4).value(), root).value();
    const auto copied = builder.add(c::Copy{}, view).value();
    const auto output = builder.add(c::Operation::custom(custom_test::Affine{1}), copied).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    auto compiled = g::compile(graph, environment.environment(),
        {.region_selection = g::RegionSelectionPolicy::kDisabled}).value();
    auto run = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    const auto *state = gd::InvocationAccess::state(run);
    ASSERT_TRUE(std::binary_search(state->plan->external_input_aliases.begin(),
        state->plan->external_input_aliases.end(), root.ordinal()));
    auto source = gd::InputSource::acquire(&producer, &source_table, &producer).value();
    ASSERT_TRUE(run.bind(output, tensors.out).ok());
    std::array<float, 4> expected;
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = values[static_cast<std::size_t>(start) + i] * 2 + 1;
    for (int episode = 0; episode < 2; ++episode) {
      if (episode) ASSERT_TRUE(gd::RegionAccess::reset_configuration(run).ok());
      ASSERT_TRUE(gd::InputAccess::bind(run, root, tensor, source).ok());
      if (episode) EXPECT_FALSE(gd::reusable_configuration_spans(*state));
      EXPECT_FALSE(state->replay_prepared);
      ASSERT_TRUE(run.prepare_replay(queues).ok());
      EXPECT_EQ(producer.routes, episode + 1);
      auto done = run.submit(queues).value();
      const auto status = done.wait();
      ASSERT_TRUE(status.ok()) << status.message();
      EXPECT_EQ(tensors.read(), expected);
    }
    ASSERT_TRUE(run.reset().ok());
    EXPECT_EQ(producer.references, 1);
  }
}

TEST(ConfigurationInput, ActualInPlaceMathAliasDoesNotBecomeAViewOnlyCandidate) {
  Fixture fixture;
  const auto *state = gd::InvocationAccess::state(fixture.run);
  ASSERT_TRUE(std::binary_search(state->plan->external_input_aliases.begin(),
      state->plan->external_input_aliases.end(), fixture.input.ordinal()));
  const auto tensor = c::TensorView::create(custom_test::description(),
      fixture.tensors.input.view()).value();
  ASSERT_TRUE(fixture.run.bind(fixture.input, tensor.as_const()).ok());
  ASSERT_TRUE(fixture.run.bind(fixture.output, tensor).ok());
  const auto input_instance = state->plan->external_instances[fixture.input.ordinal()];
  const auto output_instance = state->plan->external_instances[fixture.output.ordinal()];
  EXPECT_EQ(state->readable[input_instance].storage().native_handle(),
      state->writable[output_instance].storage().native_handle());
  EXPECT_EQ(state->readable[input_instance].storage().offset(),
      state->writable[output_instance].storage().offset());
  EXPECT_EQ(state->readable[input_instance].storage().size(),
      state->writable[output_instance].storage().size());
  for (int episode = 0; episode < 2; ++episode) {
    if (episode) {
      ASSERT_TRUE(gd::RegionAccess::reset_configuration(fixture.run).ok());
      ASSERT_TRUE(fixture.tensors.input.view().write(std::as_bytes(std::span(first))).ok());
      ASSERT_TRUE(fixture.run.bind(fixture.input, tensor.as_const()).ok());
      EXPECT_FALSE(gd::reusable_configuration_spans(*state));
    }
    ASSERT_TRUE(fixture.run.prepare_replay(fixture.queues).ok());
    auto done = fixture.run.submit(fixture.queues).value();
    const auto status = done.wait();
    ASSERT_TRUE(status.ok()) << status.message();
    std::array<float, 4> actual;
    ASSERT_TRUE(fixture.tensors.input.view().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, (std::array<float, 4>{3, 5, 7, 9}));
  }
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
  const auto *state = gd::InvocationAccess::state(run);
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
  ASSERT_TRUE(std::binary_search(state->plan->external_input_aliases.begin(),
      state->plan->external_input_aliases.end(), inputs[0].ordinal()));

  int previous_routes = 0;
  for (int episode = 0; episode < 2; ++episode) {
    if (episode) {
      ASSERT_TRUE(gd::RegionAccess::reset_configuration(run).ok());
      EXPECT_FALSE(gd::reusable_configuration_spans(*state)); // Original configuration alias.
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
