#include "../compute_models/fp8_channel_token_qdq_fixture.h"
#include "hyle/graph/cpu.h"

TEST(CpuFp8ChannelTokenQdqGraph, RankThreeBfloatReferenceKeepsTypedRoundingAndLifetimes) {
  namespace f = fp8_channel_token_qdq_test;
  auto context = f::take(hyle::graph::cpu::create_context(), "CPU context");
  auto queue = f::take(context.create_queue(), "CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  f::run(f::m::system_resource(), queues, [&](const f::g::Graph &graph, f::g::CompileOptions options) {
      return f::g::cpu::compile(graph, context, options); },
      [](f::m::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}

TEST(CpuFp8ChannelTokenQdqGraph, BooleanStorageAndBroadcastSelectPreserveIeeeBits) {
  namespace f = fp8_channel_token_qdq_test;
  auto context = f::take(hyle::graph::cpu::create_context(), "CPU context");
  auto queue = f::take(context.create_queue(), "CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  f::run_boolean_selection(f::m::system_resource(), queues, [&](const f::g::Graph &graph, f::g::CompileOptions options) {
      return f::g::cpu::compile(graph, context, options); },
      [](f::m::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}
