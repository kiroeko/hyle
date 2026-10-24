#include "../compute_models/ga_mhc_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f=ga_mhc_test;
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
        return invocation.prepare_replay(bindings);
      },owner);
}
}
TEST(CudaGaMhc, NativeMhcAndFloatNormFactorKeepTypedResiduals) {
  run_cuda([](auto... args) { ga_mhc_test::run_mhc(args...); });
}
