#include "../compute_models/ct_block128_activation_edges.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

TEST(CudaCtBlock128QdqEdges, FiniteTinyScaleAndOverflowKeepBfloatArithmetic) {
  namespace f=ct_block128_qdq_test;
  auto context=f::take(f::g::cuda::create_context(),"CUDA edge context");
  auto memory=f::take(f::c::cuda::MemoryResource::create(context,f::c::cuda::MemoryKind::kDevice),"CUDA edge memory");
  auto owner=f::take(f::c::StorageOwner::adopt(context,std::move(memory)),"CUDA edge owner");
  auto queue=f::take(context.create_queue(),"CUDA edge queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  f::run_activation_edges(owner.ref(),queues,
    [&](const f::g::Graph &graph,f::g::CompileOptions options) {
      auto compiled=f::g::cuda::compile(graph,context,owner,options);
      if(compiled)for(std::size_t i=0;i<compiled->report.step_count();++i) {
        const auto step=f::take(compiled->report.step(i),"edge step");
        ::testing::Test::RecordProperty("step_"+std::to_string(i),std::string(step.provider.name)+" node="+std::to_string(step.node_begin)+" count="+std::to_string(step.node_count));
      }
      return compiled;
    },
    [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
    [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
    [](f::g::Invocation &run,std::span<const f::g::QueueBinding> bindings) {
      const auto status=f::g::cuda::prepare_replay(run,*bindings[0].queue,{.mode=f::g::ReplayMode::kRequireCapture});
      if(status.ok())EXPECT_EQ(run.replay_route(),f::g::ReplayRoute::kCaptured);return status;
    },owner);
}
