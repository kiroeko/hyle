#include "../compute_models/fp8_scale_policy_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f=fp8_scale_policy_test;
  auto context=f::take(f::g::cuda::create_context(),"CUDA scale-policy context");
  auto memory=f::take(f::c::cuda::MemoryResource::create(context,f::c::cuda::MemoryKind::kDevice),"CUDA device resource");
  auto owner=f::take(f::c::StorageOwner::adopt(context,std::move(memory)),"CUDA original owner");
  auto queue=f::take(context.create_queue(),"CUDA scale-policy queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(owner.ref(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options){return f::g::cuda::compile(graph,context,owner,options);},
      [](f::m::BufferView to,std::span<const std::byte> from){return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to){return from.read(to);},
      [](f::g::Invocation &inv,std::span<const f::g::QueueBinding> q){
        const auto status=f::g::cuda::prepare_replay(inv,*q[0].queue,{.mode=f::g::ReplayMode::kRequireCapture});
        if(status.ok())EXPECT_EQ(inv.replay_route(),f::g::ReplayRoute::kCaptured);
        return status;
      },owner);
}
}
TEST(CudaFp8ScalePolicy, ThreePoliciesKeepDistinctFiniteQuantizers) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner){fp8_scale_policy_test::run_dense(r,q,compile,write,read,prepare,owner);});
}
TEST(CudaFp8ScalePolicy, FixedHeaderRoleKeepsOriginalBlockAxes) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner){fp8_scale_policy_test::run_source_role(r,q,compile,write,read,prepare,owner);});
}
TEST(CudaFp8ScalePolicy, GroupedSharedAndPerSlotInputsKeepReciprocalPolicy) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner){fp8_scale_policy_test::run_grouped(r,q,compile,write,read,prepare,owner);});
}
TEST(CudaFp8ScalePolicy, InvalidIdsFailAndFreshInvocationRecovers) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner){fp8_scale_policy_test::run_checked_ids(r,q,compile,write,read,prepare,owner);});
}
TEST(CudaFp8ScalePolicy, OriginalPowerOfTwoNeighbor) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner){grouped_fp8_power2_test::run(r,q,compile,write,read,prepare,owner);});
}
TEST(CudaFp8ScalePolicy, OriginalDivideNeighbor) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner){grouped_fp8_power2_test::run(r,q,compile,write,read,prepare,owner,true);});
}
