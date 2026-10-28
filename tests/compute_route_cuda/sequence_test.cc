#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "../compute_models/packed_sequence_case.h"
#include "../compute_models/packed_gdn_prefix_case.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/slots.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;
namespace p = packed_sequence_test;

struct CudaCase final {
  c::Context context = c::cuda::create_context().value();
  c::cuda::MemoryResource memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
};

TEST(CudaPackedSequence, GdnFloatingTypesAndRegisterTileMatchReference) {
  CudaCase test;
  p::gdn_math<float>(test.context, test.memory.ref());
  p::gdn_math<float>(test.context, test.memory.ref(), 37, std::nullopt);
  p::gdn_math<float>(test.context, test.memory.ref(), 256);
  p::gdn_math<hyle::simd::float16_t>(test.context, test.memory.ref());
  p::gdn_math<hyle::simd::bfloat16_t>(test.context, test.memory.ref());
}
TEST(CudaPackedSequence, GdnSnapshotsFollowRealSequenceLengths) {
  CudaCase test; p::gdn_snapshots(test.context, test.memory.ref());
}
TEST(CudaPackedSequence, EmptyAndZeroPhysicalRowsPreserveStateBits) {
  CudaCase test; p::gdn_empty(test.context, test.memory.ref());
}
TEST(CudaPackedSequence, ConvFloatingTypesKeepRealHistory) {
  CudaCase test;
  p::conv_math<float>(test.context, test.memory.ref());
  p::conv_math<hyle::simd::float16_t>(test.context, test.memory.ref());
  p::conv_math<hyle::simd::bfloat16_t>(test.context, test.memory.ref());
}
TEST(CudaPackedSequence, WholeOffsetValidationPrecedesEitherOutput) {
  CudaCase test; p::invalid_offsets(test.context, test.memory.ref());
}

TEST(CudaPackedSequence, PrefixesMatchFp64AndContinueFromTheActualDeviceState) {
  CudaCase test;
  packed_gdn_prefix_test::math<float>(test.context,test.memory.ref(),true,std::nullopt);
  packed_gdn_prefix_test::math<hyle::simd::float16_t>(test.context,test.memory.ref(),false);
  packed_gdn_prefix_test::math<hyle::simd::bfloat16_t>(test.context,test.memory.ref(),false);
}

TEST(CudaPackedSequence, PrefixCapacityEmptyMembersAndFailureRemainAtomic) {
  CudaCase test; packed_gdn_prefix_test::empty_and_failure(test.context,test.memory.ref());
}

TEST(CudaPackedSequence, PrefixCaptureReusesOffsetsAndRetiresAnInvalidLastMember) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const packed_gdn_prefix_test::Fixture<float> fixture(true,37);
  const auto operation = c::sequence_ops::packed_gated_delta_net_prefixes(fixture.prefixes,fixture.scale);
  p::Case data(context,memory.ref(),operation,fixture.descriptions,fixture.data());
  EXPECT_EQ(data.prepared.provider().id,c::cuda_native::kProviderId);
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId,7> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port) inputs[port] = builder.input(fixture.descriptions[port]).value();
  const auto node = builder.add_node(operation,inputs).value();
  const std::array outputs{builder.result(node,0).value(),builder.result(node,1).value()};
  for (const auto output : outputs) ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph,context,memory.ref()).value();
  auto run = compiled.executable.create_invocation().value();
  const auto bind = [&] {
    for (std::size_t port = 0; port < inputs.size(); ++port) ASSERT_TRUE(run.bind(inputs[port],data.inputs[port]).ok());
    for (std::size_t port = 0; port < outputs.size(); ++port) ASSERT_TRUE(run.bind(outputs[port],data.outputs[port]).ok());
  };
  bind();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  constexpr std::array<std::array<std::int32_t,3>,4> offsets{{{0,1,4},{0,3,4},{0,1,5},{0,1,4}}};
  for (std::size_t round = 0; round < offsets.size(); ++round) {
    SCOPED_TRACE(round); data.offsets(offsets[round]); data.poison();
    const auto prepared = run.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture,.host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto replay = run.replay_report().value();
    EXPECT_EQ(replay.route,g::ReplayRoute::kCaptured);
    if (round == 1 || round == 2) {
      EXPECT_EQ(replay.recorded_capture_range_count,0U); EXPECT_GE(replay.reused_capture_range_count,1U);
    }
    auto done = run.submit(queues); ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait(); *done = {};
    if (round == 2) {
      EXPECT_TRUE(status.IsInvalidArgument()) << status.message(); data.expect_poison();
      auto denied = run.submit(queues); ASSERT_FALSE(denied); EXPECT_TRUE(denied.error().IsFailedPrecondition());
      run = compiled.executable.create_invocation().value(); bind();
    } else {
      ASSERT_TRUE(status.ok()) << status.message(); fixture.check(data,offsets[round]);
    }
  }
}

TEST(CudaPackedSequence, GdnCapturedGraphKeepsRealSnapshotsAndRetiresShortSequence) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto f32 = c::DataType::kFloat32;
  const std::array descriptions{p::desc(f32,{8,1,37}),p::desc(f32,{8,1,37}),p::desc(f32,{8,2,3}),
      p::desc(f32,{8,2,3}),p::desc(f32,{8,2,1}),p::desc(f32,{2,2,3,37}),p::desc(c::DataType::kInt32,{3})};
  const auto operation = c::sequence_ops::packed_gated_delta_net(2);
  p::Case data(context, memory.ref(), operation, descriptions,
      {p::bytes(std::vector<float>(8*37,.025F)),p::bytes(std::vector<float>(8*37,.05F)),
       p::bytes(std::vector<float>(8*2*3,.5F)),p::bytes(std::vector<float>(8*2*3,-.17F)),
       p::bytes(std::vector<float>(8*2,.4F)),p::bytes(std::vector<float>(2*2*3*37,.1F)),
       p::bytes(std::vector<std::int32_t>{0,2,5})});
  EXPECT_EQ(data.prepared.provider().id, c::cuda_native::kProviderId);
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId,7> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = builder.input(descriptions[i]).value();
  const auto node = builder.add_node(operation, inputs).value();
  const std::array outputs{builder.result(node,0).value(),builder.result(node,1).value()};
  for (const auto output : outputs) ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, memory.ref()).value();
  auto run = compiled.executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  for (std::size_t i = 0; i < inputs.size(); ++i) ASSERT_TRUE(run.bind(inputs[i],data.inputs[i]).ok());
  for (std::size_t i = 0; i < outputs.size(); ++i) ASSERT_TRUE(run.bind(outputs[i],data.outputs[i]).ok());
  constexpr std::array<std::array<std::int32_t,3>,4> offsets{{{0,2,5},{0,3,5},{0,1,5},{0,2,5}}};
  for (std::size_t round = 0; round < offsets.size(); ++round) {
    SCOPED_TRACE(round);
    data.offsets(offsets[round]);
    std::array<std::vector<float>,2> expected;
    if (round != 2) {
      const auto direct = data.submit(); ASSERT_TRUE(direct.ok()) << direct.message();
      for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = data.read<float>(i);
    }
    data.poison();
    const auto prepared = run.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture,.host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto report = run.replay_report().value();
    EXPECT_EQ(report.route, g::ReplayRoute::kCaptured);
    if (round == 1 || round == 2) {
      EXPECT_EQ(report.recorded_capture_range_count,0U);
      EXPECT_GE(report.reused_capture_range_count,1U);
    }
    auto done = run.submit(queues); ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait(); *done = {};
    if (round == 2) {
      EXPECT_TRUE(status.IsInvalidArgument()) << status.message(); data.expect_poison();
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(data.read<float>(i),expected[i]);
      data.expect_tail();
    }
    if (round == 2) {
      // Terminal failure is not a reusable execution carrier. Real retirement
      // permits a new Invocation of the same original executable, not revival.
      auto rejected = run.submit(queues);
      ASSERT_FALSE(rejected);
      EXPECT_TRUE(rejected.error().IsFailedPrecondition());
      run = compiled.executable.create_invocation().value();
      for (std::size_t i = 0; i < inputs.size(); ++i) ASSERT_TRUE(run.bind(inputs[i],data.inputs[i]).ok());
      for (std::size_t i = 0; i < outputs.size(); ++i) ASSERT_TRUE(run.bind(outputs[i],data.outputs[i]).ok());
    }
  }
}

TEST(CudaPackedSequence, CapturedGraphUpdatesOffsetsAndGuardsExternalStoreOnFailure) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto staging = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  auto queue = context.create_queue().value();
  const auto f32 = c::DataType::kFloat32;
  const std::array descriptions{p::desc(f32,{8,2}), p::desc(f32,{2,2}),
      p::desc(f32,{2,1,2}), p::desc(c::DataType::kInt32,{3})};
  auto x = std::vector<float>(16, NAN);
  std::fill_n(x.begin(), 8, 1.0F);
  p::Case data(context, memory.ref(), c::sequence_ops::packed_causal_conv1d(), descriptions,
      {p::bytes(x),p::bytes(std::vector<float>(4,1)),p::bytes(std::vector<float>(4,0)),
       p::bytes(std::vector<std::int32_t>{0,1,4})});
  EXPECT_EQ(data.prepared.provider().id, c::cuda_native::kProviderId);
  EXPECT_EQ(data.prepared.invocation_private_bytes(), 0U);

  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId,4> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = builder.input(descriptions[i]).value();
  const auto conv = builder.add_node(c::sequence_ops::packed_causal_conv1d(), inputs).value();
  const auto y = builder.result(conv, 0).value(), history = builder.result(conv, 1).value();
  const auto ids = builder.input(p::desc(c::DataType::kInt32,{8})).value();
  constexpr std::size_t stride = 256;
  const auto item = p::desc(f32,{2});
  const auto schema = s::Schema::create(item, 9, stride).value();
  const auto store = builder.add_node(c::Operation::custom(s::Store{schema}), std::array{y,ids}).value();
  ASSERT_TRUE(builder.publish(history).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, memory.ref()).value();
  bool found = false;
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step)
    for (std::size_t node = 0; node < compiled.report.step(step)->node_count; ++node)
      if (compiled.report.step_node(step,node).value() == conv) {
        EXPECT_EQ(compiled.report.step(step)->provider.id, c::cuda_native::kProviderId); found = true;
      }
  ASSERT_TRUE(found);
  auto run = compiled.executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  for (std::size_t i = 0; i < inputs.size(); ++i) ASSERT_TRUE(run.bind(inputs[i], data.inputs[i]).ok());
  ASSERT_TRUE(run.bind(history, data.outputs[1]).ok());
  ASSERT_TRUE(s::prepare(run, queues, {.host_workspace = staging.ref()}).ok());

  auto pool = c::LeasePool::create(memory.ref(), 9 * stride, {.max_blocks = 9}).value();
  const auto layout = c::LeaseSlotLayout::create(pool, item, 9, stride).value();
  std::array<c::WriteLease,8> writers;
  for (auto &writer : writers) writer = pool.allocate(2 * sizeof(float), stride).value();
  auto sentinel_writer = pool.allocate(2 * sizeof(float), stride).value();
  const std::array<float,2> sentinel{101,103};
  ASSERT_TRUE(sentinel_writer.view()->write(std::as_bytes(std::span(sentinel))).ok());
  const auto sentinel_read = sentinel_writer.publish().value();
  auto transaction = s::SlotBindings::create(layout, {}, writers).value();
  constexpr std::array<std::array<std::int32_t,3>,4> offsets{{{0,1,4}, {0,0,3}, {0,1,9}, {0,1,4}}};
  constexpr std::array<float,2> untouched{-77,-79};
  for (std::size_t round = 0; round < offsets.size(); ++round) {
    SCOPED_TRACE(round);
    data.offsets(offsets[round]);
    ASSERT_TRUE(data.outputs[1].storage().fill(std::byte{0x5a}).ok());
    std::array<m::BufferView,8> diagnostic_views;
    for (std::size_t row = 0; row < transaction.destination_count(); ++row) {
      auto view = transaction.destinations()[row].view();
      ASSERT_TRUE(view) << view.error().message();
      diagnostic_views[row] = *view;
      ASSERT_TRUE(view->write(std::as_bytes(std::span(untouched))).ok());
    }
    ASSERT_TRUE(s::bind(run, graph, store, std::move(transaction)).ok());
    const auto prepared = run.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto report = run.replay_report().value();
    EXPECT_EQ(report.route, g::ReplayRoute::kCaptured);
    if (round == 1 || round == 2) {
      EXPECT_EQ(report.recorded_capture_range_count, 0U);
      EXPECT_GE(report.reused_capture_range_count, 1U);
    }
    auto done = run.submit(queues);
    ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait();
    if (round == 2) EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
    else EXPECT_TRUE(status.ok()) << status.message();
    *done = {};
    auto returned = s::take(run, store);
    ASSERT_TRUE(returned) << returned.error().message();
    transaction = std::move(*returned);
    for (std::size_t row = 0; row < transaction.destination_count(); ++row) {
      std::array<float,2> actual;
      if (round == 2) {
        // A failed write episode poisons its writer even if the device guard
        // prevented every write. Observe bytes only through this exact prior
        // diagnostic borrow, while its original writer is retained and retired.
        auto denied = transaction.destinations()[row].view();
        ASSERT_FALSE(denied); EXPECT_EQ(denied.error(),status);
        auto publication = transaction.destinations()[row].publish();
        ASSERT_FALSE(publication); EXPECT_EQ(publication.error(),status);
        ASSERT_TRUE(diagnostic_views[row].read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, untouched);
      }
      else {
        auto view = transaction.destinations()[row].view();
        ASSERT_TRUE(view) << view.error().message();
        ASSERT_TRUE(view->read(std::as_writable_bytes(std::span(actual))).ok());
        const float expected = row >= std::size_t(offsets[round].back()) ? 0.0F
            : row == 0 || (round != 1 && row == 1) ? 1.0F : 2.0F;
        EXPECT_EQ(actual, (std::array<float,2>{expected,expected}));
      }
    }
    const auto actual_history = data.read<float>(1);
    if (round == 2) EXPECT_EQ(data.read<std::byte>(1), std::vector<std::byte>(16,std::byte{0x5a}));
    else EXPECT_EQ(actual_history, (round == 1 ? std::vector<float>{0,0,1,1} : std::vector<float>{1,1,1,1}));
    data.expect_tail();
    std::array<float,2> actual_sentinel;
    ASSERT_TRUE(sentinel_read.view()->read(std::as_writable_bytes(std::span(actual_sentinel))).ok());
    EXPECT_EQ(actual_sentinel, sentinel);
    if (round == 2) {
      auto rejected = run.submit(queues);
      ASSERT_FALSE(rejected);
      EXPECT_TRUE(rejected.error().IsFailedPrecondition());
      transaction.clear(); // Failed candidates never become new write authority.
      for (auto &writer : writers) writer = pool.allocate(2 * sizeof(float),stride).value();
      ASSERT_TRUE(transaction.refill({},writers).ok());
      run = compiled.executable.create_invocation().value();
      for (std::size_t i = 0; i < inputs.size(); ++i) ASSERT_TRUE(run.bind(inputs[i],data.inputs[i]).ok());
      ASSERT_TRUE(run.bind(history,data.outputs[1]).ok());
      ASSERT_TRUE(s::prepare(run,queues,{.host_workspace = staging.ref()}).ok());
    }
  }
}
} // namespace
