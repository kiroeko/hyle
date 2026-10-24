#include "../compute_models/fp8_channel_token_qdq_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

TEST(CudaFp8ChannelTokenQdqGraph, RankThreeBfloatReferenceKeepsTypedRoundingAndLifetimes) {
  namespace f = fp8_channel_token_qdq_test;
  auto context = f::take(hyle::graph::cuda::create_context(), "CUDA context");
  auto memory = f::take(f::c::cuda::MemoryResource::create(context, f::c::cuda::MemoryKind::kDevice), "CUDA memory");
  const auto owner = f::take(f::c::StorageOwner::adopt(context, std::move(memory)), "CUDA storage owner");
  auto queue = f::take(context.create_queue(), "CUDA queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  f::run(owner.ref(), queues, [&](const f::g::Graph &graph, f::g::CompileOptions options) {
      return f::g::cuda::compile(graph, context, owner, options); },
      [](f::m::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) {
        const auto status = f::g::cuda::prepare_replay(invocation, *bindings[0].queue,
            {.mode = f::g::ReplayMode::kRequireCapture});
        if (status.ok()) EXPECT_EQ(invocation.replay_route(), f::g::ReplayRoute::kCaptured);
        return status;
      }, owner);
}

TEST(CudaFp8ChannelTokenQdqGraph, BooleanStorageAndBroadcastSelectPreserveIeeeBits) {
  namespace f = fp8_channel_token_qdq_test;
  auto context = f::take(hyle::graph::cuda::create_context(), "CUDA context");
  auto memory = f::take(f::c::cuda::MemoryResource::create(context, f::c::cuda::MemoryKind::kDevice), "CUDA memory");
  const auto owner = f::take(f::c::StorageOwner::adopt(context, std::move(memory)), "CUDA storage owner");
  auto queue = f::take(context.create_queue(), "CUDA queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  f::run_boolean_selection(owner.ref(), queues, [&](const f::g::Graph &graph, f::g::CompileOptions options) {
      return f::g::cuda::compile(graph, context, owner, options); },
      [](f::m::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) {
        const auto status = f::g::cuda::prepare_replay(invocation, *bindings[0].queue,
            {.mode = f::g::ReplayMode::kRequireCapture});
        if (status.ok()) EXPECT_EQ(invocation.replay_route(), f::g::ReplayRoute::kCaptured);
        return status;
      }, owner);
}
