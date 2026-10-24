#include "../compute_models/ct_block128_activation_edges.h"
#include "hyle/graph/cpu.h"

TEST(CpuCtBlock128QdqEdges, FiniteTinyScaleAndOverflowKeepBfloatArithmetic) {
  namespace f=ct_block128_qdq_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU edge context");
  auto queue=f::take(context.create_queue(),"CPU edge queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  f::run_activation_edges(f::m::system_resource(),queues,
    [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
    [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
    [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
    [](f::g::Invocation &run,std::span<const f::g::QueueBinding> bindings) {return run.prepare_replay(bindings);});
}
