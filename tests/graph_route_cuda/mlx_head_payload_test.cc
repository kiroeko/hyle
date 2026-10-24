#include "../compute_models/mlx_head_payload_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f=mlx_head_payload_test;
  auto context=f::take(f::g::cuda::create_context(),"CUDA context");
  auto memory=f::take(f::c::cuda::MemoryResource::create(context,f::c::cuda::MemoryKind::kDevice),"CUDA memory");
  auto owner=f::take(f::c::StorageOwner::adopt(context,std::move(memory)),"CUDA storage owner");
  auto queue=f::take(context.create_queue(),"CUDA queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(owner.ref(),std::span<const f::g::QueueBinding>(queues),
      [&](const f::g::Graph &graph,f::g::CompileOptions options){return f::g::cuda::compile(graph,context,owner,options);},
      [](f::m::BufferView dst,std::span<const std::byte> src){return dst.write(src);},
      [](f::m::ConstBufferView src,std::span<std::byte> dst){return src.read(dst);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings){
        const auto status=f::g::cuda::prepare_replay(invocation,*bindings[0].queue,{.mode=f::g::ReplayMode::kRequireCapture});
        if(status.ok())EXPECT_EQ(invocation.replay_route(),f::g::ReplayRoute::kCaptured);
        return status;
      },owner);
}
}
TEST(CudaMlxHeadPayload, PackedLayer0RolesKeepOriginalBytesAndHeadOrder) {
  run_cuda([](auto... args) { mlx_head_payload_test::run(args...); });
}
