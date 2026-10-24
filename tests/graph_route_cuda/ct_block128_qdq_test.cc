#include "../compute_models/ct_block128_qdq_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
void run_cuda(bool probes,bool checked=false) {
  namespace f=ct_block128_qdq_test;
  auto context=f::take(f::g::cuda::create_context(),"CUDA CT block context");
  auto memory=f::take(f::c::cuda::MemoryResource::create(context,f::c::cuda::MemoryKind::kDevice),"CUDA CT memory");
  auto owner=f::take(f::c::StorageOwner::adopt(context,std::move(memory)),"CUDA CT owner");
  auto queue=f::take(context.create_queue(),"CUDA CT queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  const auto exercise=[&](auto resource,auto bindings,auto compile,auto write,auto read,auto prepare) {
    if(checked)f::run_checked_ids(resource,bindings,compile,write,read,prepare,owner);
    else f::run(resource,bindings,compile,write,read,prepare,owner,probes);
  };
  exercise(owner.ref(),queues,
    [&](const f::g::Graph &graph,f::g::CompileOptions options) {
      auto compiled=f::g::cuda::compile(graph,context,owner,options);
      if(compiled) for(std::size_t i=0;i<compiled->report.step_count();++i) {
        const auto step=f::take(compiled->report.step(i),"cold CUDA step report");
        if(!step.runtime.capture_compatible && step.kind!=f::g::CompileStepKind::kView) {
          const auto node=f::take(compiled->report.node(step.node_begin),"cold CUDA node report");
          ::testing::Test::RecordProperty("uncapturable_step_"+std::to_string(i),
              std::string(step.provider.name)+" node="+std::to_string(step.node_begin)+" kind="+std::to_string(int(node.kind)));
        }
      }
      return compiled;
    },
    [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
    [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
    [](f::g::Invocation &run,std::span<const f::g::QueueBinding> bindings) {
      const auto status=f::g::cuda::prepare_replay(run,*bindings[0].queue,{.mode=f::g::ReplayMode::kRequireCapture});
      if(status.ok()) EXPECT_EQ(run.replay_route(),f::g::ReplayRoute::kCaptured);
      return status;
    });
}
} // namespace
TEST(CudaCtBlock128QdqGraph, DenseAndSelectedExpertsKeepBfloatMathAndResultLifetimes) {run_cuda(false);}
TEST(CudaCtBlock128QdqGraph, PublishedQdqAndSelectedWeightKeepEveryBfloatBoundary) {run_cuda(true);}
TEST(CudaCtBlock128QdqGraph, CompleteExpertIdsFailBeforeSelectionAndFreshSlotRecovers) {run_cuda(false,true);}
