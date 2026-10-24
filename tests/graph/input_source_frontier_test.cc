#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "../../src/graph/submission_internal.h"
#include "../compute/custom_kernel_fixture.h"
#include "hyle/graph/detail/input_source.h"
#include "hyle/graph/result.h"
#include "hyle/mem/stats.h"
#include "test_environment.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace g = hyle::graph;
namespace gd = g::detail;
using hyle::Result;
using hyle::Status;

// No thread, clock or poll-count trigger makes this source ready. Only the
// original wait callback settles it, with either success or its own failure.
struct PendingInput final {
  int references = 0, routes = 0, polls = 0, waits = 0, enqueues = 0;
  bool settled = false;
  Status terminal;
  const custom_test::Probe *independent = nullptr;
  int independent_at_wait = -1;
};
const gd::InputSourceVTable source_table{
    [](void *p) noexcept { ++static_cast<PendingInput *>(p)->references; },
    [](void *p) noexcept { --static_cast<PendingInput *>(p)->references; },
    [](void *p, const cd::QueueRef *) noexcept -> Result<gd::InputDependencyRoute> {
      ++static_cast<PendingInput *>(p)->routes;
      return gd::InputDependencyRoute::kHostWait;
    },
    [](void *p) noexcept -> Result<c::CompletionPoll> {
      auto &source = *static_cast<PendingInput *>(p);
      ++source.polls;
      if (!source.settled) return c::CompletionPoll::kPending;
      if (!source.terminal.ok()) return std::unexpected(source.terminal);
      return c::CompletionPoll::kComplete;
    },
    [](void *p) noexcept -> Status {
      auto &source = *static_cast<PendingInput *>(p);
      ++source.waits;
      source.independent_at_wait = source.independent->executions.load();
      source.settled = true;
      return source.terminal;
    },
    [](void *p, cd::QueueSequence &) noexcept {
      ++static_cast<PendingInput *>(p)->enqueues;
      return cd::EnqueueResult::rejected(Status::Internal("HostWait source was imported"));
    },
    [](void *) noexcept { return Status::Ok(); }};

struct Logical final {
  g::Graph graph;
  std::array<g::ValueId, 3> inputs, outputs;
};
Logical graph() {
  auto builder = g::GraphBuilder::create().value();
  Logical logical;
  for (std::size_t i = 0; i < logical.inputs.size(); ++i) {
    logical.inputs[i] = builder.input(custom_test::description()).value();
    logical.outputs[i] = builder.add(c::Operation::custom(custom_test::Affine{1}),
                                    logical.inputs[i]).value();
    EXPECT_TRUE(builder.publish(logical.outputs[i]).ok());
  }
  logical.graph = std::move(builder).finish().value();
  return logical;
}

c::Context make_context(custom_test::Probe &probe, hyle::mem::AllocatorRef allocator) {
  const std::array providers{c::ProviderBinding{custom_test::provider(probe).value()}};
  return c::Context::create(c::cpu::device(),
      {.metadata_allocator = allocator, .providers = providers}).value();
}

struct Fixture final {
  hyle::mem::AllocatorRef allocator;
  std::array<PendingInput, 2> sources;
  custom_test::Probe blocked, independent;
  c::Context first = make_context(blocked, allocator);
  c::Context second = make_context(independent, allocator);
  c::Queue first_queue = first.create_queue().value();
  c::Queue second_queue = second.create_queue().value();
  g::test::Environment environment{{&first, hyle::mem::system_resource()},
                                   {&second, hyle::mem::system_resource()}};
  std::array<custom_test::Tensors, 3> tensors;
  std::array<g::QueueBinding, 2> queues{
      g::QueueBinding{environment.lane(0), &first_queue},
      g::QueueBinding{environment.lane(1), &second_queue}};

  explicit Fixture(hyle::mem::AllocatorRef owner = hyle::mem::system_allocator())
      : allocator(owner) {
    for (auto &source : sources) source.independent = &independent;
    for (auto &tensor : tensors) EXPECT_TRUE(tensor.fill().ok());
    const std::array<float, 4> sentinel{99, 99, 99, 99};
    for (auto &tensor : tensors)
      EXPECT_TRUE(tensor.output.view().write(std::as_bytes(std::span(sentinel))).ok());
  }
  ~Fixture() {
    for (const auto &source : sources) EXPECT_EQ(source.references, 0);
  }
  Result<g::CompiledGraph> compile(const Logical &logical) {
    const std::array placements{
        g::NodePlacement{logical.graph.node_id(0).value(), environment.target(0), g::PlacementStrength::kRequired},
        g::NodePlacement{logical.graph.node_id(1).value(), environment.target(1), g::PlacementStrength::kRequired},
        g::NodePlacement{logical.graph.node_id(2).value(), environment.target(0), g::PlacementStrength::kRequired}};
    return g::compile(logical.graph, environment.environment(),
        {.node_placements = placements, .region_selection = g::RegionSelectionPolicy::kDisabled});
  }
  Status bind(g::Invocation &run, const Logical &logical) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      if (i == 1) {
        HYLE_RETURN_IF_ERROR(run.bind(logical.inputs[i], tensors[i].in));
      } else {
        auto &source = sources[i / 2];
        HYLE_ASSIGN_OR_RETURN(auto owner, gd::InputSource::acquire(&source, &source_table, &source));
        HYLE_RETURN_IF_ERROR(gd::InputAccess::bind(run, logical.inputs[i], tensors[i].in, std::move(owner)));
      }
      HYLE_RETURN_IF_ERROR(run.bind(logical.outputs[i], tensors[i].out));
    }
    return Status::Ok();
  }
};
constexpr std::array<float, 4> expected{3, 5, 7, 9};
constexpr std::array<float, 4> sentinel{99, 99, 99, 99};

TEST(InputSourceFrontier, OrdinarySubmitSettlesBothSourcesAfterIndependentReadyWork) {
  Fixture fixture;
  const auto logical = graph();
  auto compiled = fixture.compile(logical);
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(fixture.bind(run, logical).ok());
  const auto rejected = run.prepare_replay(fixture.queues, {.host_wait = g::HostWaitPolicy::kReject});
  EXPECT_TRUE(rejected.IsUnimplemented());
  for (const auto &source : fixture.sources) {
    EXPECT_EQ(source.polls, 0);
    EXPECT_EQ(source.waits, 0);
  }
  ASSERT_TRUE(run.prepare_replay(fixture.queues).ok());
  EXPECT_EQ(run.input_source_report()->host_wait_count, 2u);
  auto completion = run.submit(fixture.queues);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(fixture.independent.executions, 1);
  EXPECT_EQ(fixture.blocked.executions, 2);
  EXPECT_GT(fixture.sources[0].polls + fixture.sources[1].polls, 0);
  for (const auto &source : fixture.sources) {
    EXPECT_EQ(source.waits, 1);
    EXPECT_EQ(source.independent_at_wait, 1);
    EXPECT_EQ(source.enqueues, 0);
  }
  for (auto &tensor : fixture.tensors) EXPECT_EQ(tensor.read(), expected);
  for (const auto output : logical.outputs) EXPECT_TRUE(run.output(output));
}

TEST(InputSourceFrontier, NonblockingPendingObservationDoesNotPoisonTheOriginalCursor) {
  Fixture fixture;
  const auto logical = graph();
  auto compiled = fixture.compile(logical);
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(fixture.bind(run, logical).ok());
  ASSERT_TRUE(run.prepare_replay(fixture.queues).ok());
  auto started = gd::SubmissionCursor::begin(run, fixture.queues);
  ASSERT_TRUE(started) << started.error().message();
  auto cursor = std::move(*started);
  auto advance = cursor.advance(false);
  ASSERT_TRUE(advance) << advance.error().message();
  EXPECT_TRUE(*advance); // The independent branch is accepted before a wait.
  EXPECT_EQ(fixture.independent.executions, 1);
  for (int retry = 0; retry < 2; ++retry) {
    advance = cursor.advance(false);
    ASSERT_TRUE(advance) << advance.error().message();
    EXPECT_FALSE(*advance);
    EXPECT_TRUE(cursor.submission_error().ok());
    EXPECT_FALSE(cursor.done());
    EXPECT_EQ(cursor.pending_dependency(), nullptr); // External, not a Plan cell.
    for (const auto &source : fixture.sources) EXPECT_EQ(source.waits, 0);
  }
  for (int remaining = 0; remaining < 4 && !cursor.done(); ++remaining) {
    advance = cursor.advance(true);
    ASSERT_TRUE(advance) << advance.error().message();
  }
  ASSERT_TRUE(cursor.done());
  ASSERT_TRUE(cursor.submission_error().ok());
  auto completion = cursor.finish();
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  for (auto &tensor : fixture.tensors) EXPECT_EQ(tensor.read(), expected);
  for (const auto &source : fixture.sources) EXPECT_EQ(source.waits, 1);
}

TEST(InputSourceFrontier, ProducerWaitFailurePreservesAcceptedIndependentWorkAndItsStatus) {
  Fixture fixture;
  fixture.sources[0].terminal = Status::DataLoss("immutable producer failed during wait");
  const auto logical = graph();
  auto compiled = fixture.compile(logical);
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(fixture.bind(run, logical).ok());
  ASSERT_TRUE(run.prepare_replay(fixture.queues).ok());
  auto completion = run.submit(fixture.queues);
  ASSERT_TRUE(completion) << completion.error().message();
  const auto terminal = completion->wait();
  EXPECT_TRUE(terminal.IsDataLoss());
  EXPECT_EQ(terminal.message(), "immutable producer failed during wait");
  EXPECT_EQ(fixture.sources[0].waits, 1);
  EXPECT_EQ(fixture.sources[0].independent_at_wait, 1);
  EXPECT_EQ(fixture.independent.executions, 1);
  EXPECT_EQ(fixture.tensors[0].read(), sentinel);
  EXPECT_EQ(fixture.tensors[1].read(), expected);
  // Both consumers in this blocked target share its original segment. The
  // failed source prevents either command; the other target already ran.
  EXPECT_EQ(fixture.tensors[2].read(), sentinel);
  EXPECT_FALSE(run.in_flight());
  for (const auto output : logical.outputs) {
    const auto rejected = run.output(output);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  }
}

TEST(InputSourceFrontier, StableSourcesReuseScratchAcrossOrdinaryBindingChanges) {
  auto metadata = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    Fixture fixture(metadata.get());
    const auto logical = graph();
    auto compiled = fixture.compile(logical);
    ASSERT_TRUE(compiled) << compiled.error().message();
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(fixture.bind(run, logical).ok());
    std::array<custom_test::Tensors, 2> rotating;
    const std::array<std::array<float, 4>, 2> inputs{{{1, 2, 3, 4}, {-2, 0, 6, 10}}};
    const std::array<std::array<float, 4>, 2> outputs{{{3, 5, 7, 9}, {-3, 1, 13, 21}}};
    for (std::size_t index = 0; index < rotating.size(); ++index)
      ASSERT_TRUE(rotating[index].input.view().write(std::as_bytes(std::span(inputs[index]))).ok());
    // Both source bindings remain untouched. Two warm rounds fill the active
    // and candidate capacities; measure only each later prepare_replay window.
    for (std::size_t round = 0; round < 6; ++round) {
      SCOPED_TRACE(round);
      const auto index = round % rotating.size();
      ASSERT_TRUE(run.bind(logical.inputs[1], rotating[index].in).ok());
      ASSERT_TRUE(run.bind(logical.outputs[1], rotating[index].out).ok());
      const auto before = metadata.snapshot();
      ASSERT_TRUE(run.prepare_replay(fixture.queues).ok());
      const auto after = metadata.snapshot();
      if (round >= 2) {
        EXPECT_EQ(after.total_allocs, before.total_allocs);
        EXPECT_EQ(after.reallocs, before.reallocs);
      }
      auto report = run.input_source_report();
      ASSERT_TRUE(report) << report.error().message();
      EXPECT_EQ(report->source_count, 2u);
      EXPECT_EQ(report->host_wait_count, 2u);
      // Preparation must still requalify the actual source/Queue route.
      for (const auto &source : fixture.sources)
        EXPECT_EQ(source.routes, static_cast<int>(round + 1));
      auto completion = run.submit(fixture.queues);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      *completion = {};
      EXPECT_EQ(rotating[index].read(), outputs[index]);
      EXPECT_EQ(fixture.tensors[0].read(), expected);
      EXPECT_EQ(fixture.tensors[2].read(), expected);
      auto held = run.result(logical.outputs[1]);
      ASSERT_TRUE(held) << held.error().message();
      EXPECT_TRUE(run.bind(logical.inputs[1], rotating[1 - index].in).IsUnavailable());
    }
    EXPECT_EQ(fixture.independent.executions, 6);
    EXPECT_EQ(fixture.blocked.executions, 12);
    for (const auto &source : fixture.sources) {
      EXPECT_EQ(source.waits, 1);
      EXPECT_EQ(source.enqueues, 0);
    }
  }
  EXPECT_EQ(metadata.snapshot().live_count, 0u);
  EXPECT_EQ(metadata.snapshot().live_bytes, 0u);
}

TEST(InputSourceBundle, InvalidLastTensorPreservesOriginalPreparedBindings) {
  for(const bool existing_sources:{false,true}) {
  SCOPED_TRACE(existing_sources);
  PendingInput replacement;
  Fixture fixture;replacement.independent=&fixture.independent;replacement.settled=true;
  const auto logical=graph();
  auto compiled=fixture.compile(logical).value();
  auto run=compiled.executable.create_invocation().value();
  if(existing_sources) {
    ASSERT_TRUE(fixture.bind(run,logical).ok());
  } else for(std::size_t i=0;i<fixture.tensors.size();++i) {
    ASSERT_TRUE(run.bind(logical.inputs[i],fixture.tensors[i].in).ok());
    ASSERT_TRUE(run.bind(logical.outputs[i],fixture.tensors[i].out).ok());
  }
  ASSERT_TRUE(run.prepare_replay(fixture.queues).ok());
  const auto route=run.replay_route();
  custom_test::Tensors changed;
  const std::array<float,4> values{-2,0,6,10};
  ASSERT_TRUE(changed.input.view().write(std::as_bytes(std::span(values))).ok());
  auto source=gd::InputSource::acquire(&replacement,&source_table,&replacement).value();
  const auto wrong=c::ConstTensorView::create(
      c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{2}).value(),
      changed.input.view()).value();
  const std::array bundle{
      gd::InputSourceBinding{logical.inputs[0],changed.in,source},
      gd::InputSourceBinding{logical.inputs[2],wrong,source}};
  EXPECT_TRUE(gd::InputAccess::bind(run,bundle).IsInvalidArgument());
  EXPECT_EQ(run.replay_route(),route);
  ASSERT_TRUE(run.input_source_report());
  EXPECT_EQ(run.input_source_report()->source_count,existing_sources?2u:0u);
  EXPECT_EQ(replacement.polls+replacement.waits+replacement.enqueues,0);
  auto completion=run.submit(fixture.queues);
  ASSERT_TRUE(completion)<<completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  for(auto &tensor:fixture.tensors) EXPECT_EQ(tensor.read(),expected);
  }
}

TEST(InputSourceBundle, SameProducerReplacesBothInputsAndDeduplicatesReadiness) {
  PendingInput replacement;
  Fixture fixture;replacement.independent=&fixture.independent;replacement.settled=true;
  const auto logical=graph();
  auto compiled=fixture.compile(logical).value();
  auto run=compiled.executable.create_invocation().value();
  ASSERT_TRUE(fixture.bind(run,logical).ok());
  std::array<custom_test::Tensors,2> changed;
  const std::array<std::array<float,4>,2> values{{{-2,0,6,10},{10,20,30,40}}};
  for(std::size_t i=0;i<changed.size();++i)
    ASSERT_TRUE(changed[i].input.view().write(std::as_bytes(std::span(values[i]))).ok());
  auto source=gd::InputSource::acquire(&replacement,&source_table,&replacement).value();
  const std::array bundle{
      gd::InputSourceBinding{logical.inputs[0],changed[0].in,source},
      gd::InputSourceBinding{logical.inputs[2],changed[1].in,source}};
  ASSERT_TRUE(gd::InputAccess::bind(run,bundle).ok());
  EXPECT_EQ(fixture.sources[0].references,0);EXPECT_EQ(fixture.sources[1].references,0);
  ASSERT_TRUE(run.prepare_replay(fixture.queues).ok());
  ASSERT_TRUE(run.input_source_report());
  EXPECT_EQ(run.input_source_report()->source_count,1u);
  EXPECT_EQ(run.input_source_report()->host_wait_count,1u);
  auto completion=run.submit(fixture.queues);
  ASSERT_TRUE(completion)<<completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(fixture.tensors[0].read(),(std::array<float,4>{-3,1,13,21}));
  EXPECT_EQ(fixture.tensors[1].read(),expected);
  EXPECT_EQ(fixture.tensors[2].read(),(std::array<float,4>{21,41,61,81}));
  EXPECT_EQ(replacement.waits,0);EXPECT_EQ(replacement.enqueues,0);
}

TEST(InputSourceBundle, FuturePublishedInputAliasAndDuplicatesRejectAsOneTransaction) {
  PendingInput replacement;
  Fixture fixture;replacement.independent=&fixture.independent;replacement.settled=true;
  auto builder=g::GraphBuilder::create().value();
  const auto left=builder.input(custom_test::description()).value();
  const auto published=builder.input(custom_test::description()).value();
  const auto result=builder.add(c::Operation::custom(custom_test::Affine{1}),left).value();
  ASSERT_TRUE(builder.publish(result).ok());ASSERT_TRUE(builder.publish(published).ok());
  auto logical=std::move(builder).finish().value();
  const std::array placements{g::NodePlacement{logical.node_id(0).value(),
      fixture.environment.target(0),g::PlacementStrength::kRequired}};
  auto compiled=g::compile(logical,fixture.environment.environment(),
      {.node_placements=placements,.region_selection=g::RegionSelectionPolicy::kDisabled}).value();
  const auto queues=std::span(fixture.queues).first(1);
  auto run=compiled.executable.create_invocation().value();
  auto source=gd::InputSource::acquire(&replacement,&source_table,&replacement).value();
  ASSERT_TRUE(gd::InputAccess::bind(run,left,fixture.tensors[0].in,source).ok());
  ASSERT_TRUE(gd::InputAccess::bind(run,published,fixture.tensors[1].in,source).ok());
  ASSERT_TRUE(run.bind(result,fixture.tensors[0].out).ok());
  const auto prepared=run.prepare_replay(queues);
  ASSERT_TRUE(prepared.ok())<<prepared.message();
  const auto route=run.replay_route();
  custom_test::Tensors shared;
  ASSERT_TRUE(shared.fill().ok());
  const std::array future{
      gd::InputSourceBinding{left,shared.in,source},
      gd::InputSourceBinding{published,shared.in,source}};
  EXPECT_TRUE(gd::InputAccess::bind(run,future).IsInvalidArgument());
  EXPECT_EQ(run.replay_route(),route);
  const std::array duplicate{
      gd::InputSourceBinding{left,shared.in,source},
      gd::InputSourceBinding{left,fixture.tensors[1].in,source}};
  EXPECT_TRUE(gd::InputAccess::bind(run,duplicate).IsInvalidArgument());
  EXPECT_EQ(run.replay_route(),route);
  auto completion=run.submit(queues);
  ASSERT_TRUE(completion)<<completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(fixture.tensors[0].read(),expected);
}
} // namespace
