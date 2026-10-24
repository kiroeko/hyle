#include "../compute_models/ct_block128_qdq_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
void run_cpu(bool probes,bool checked=false) {
  namespace f=ct_block128_qdq_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU CT block context");
  auto queue=f::take(context.create_queue(),"CPU CT block queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  const auto exercise=[&](auto resource,auto bindings,auto compile,auto write,auto read,auto prepare) {
    if(checked)f::run_checked_ids(resource,bindings,compile,write,read,prepare);
    else f::run(resource,bindings,compile,write,read,prepare,{},probes);
  };
  exercise(f::m::system_resource(),queues,
    [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
    [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
    [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
    [](f::g::Invocation &run,std::span<const f::g::QueueBinding> bindings) {return run.prepare_replay(bindings);});
}
} // namespace
TEST(CpuCtBlock128QdqGraph, DenseAndSelectedExpertsKeepBfloatMathAndResultLifetimes) {run_cpu(false);}
TEST(CpuCtBlock128QdqGraph, PublishedQdqAndSelectedWeightKeepEveryBfloatBoundary) {run_cpu(true);}
TEST(CpuCtBlock128QdqGraph, CompleteExpertIdsFailBeforeSelectionAndFreshSlotRecovers) {run_cpu(false,true);}
