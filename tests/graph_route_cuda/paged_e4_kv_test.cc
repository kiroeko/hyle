#include "../compute_models/paged_e4_kv_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/flashinfer.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f=paged_e4_kv_test;
  auto context=f::take(f::c::Context::create(f::take(f::c::cuda::device(0), "CUDA device"),
      f::c::flashinfer::provider()), "CUDA paged context");
  auto memory=f::take(f::c::cuda::MemoryResource::create(context,f::c::cuda::MemoryKind::kDevice), "CUDA memory");
  auto staging=f::take(f::c::cuda::MemoryResource::create(context,f::c::cuda::MemoryKind::kPinnedHost), "CUDA metadata staging");
  auto queue=f::take(context.create_queue(), "CUDA paged queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::ExecutionSource::acquire(context), "CUDA source"),f::g::StorageDomainId{0}}};
  run(memory.ref(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::batch::compile(graph,{domains,targets},options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);},staging.ref());
  f::check(context.drain(), "CUDA context retirement");
}
}
TEST(CudaPagedE4Kv, EncoderBytesRespectFixedScale) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto) {
    paged_e4_kv_test::encoder(resource,queues,compile,write,read);
  });
}
TEST(CudaPagedE4Kv, EncoderFailureRetiresAndFreshInvocationRecovers) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto) {
    paged_e4_kv_test::encoder(resource,queues,compile,write,read,true);
  });
}
TEST(CudaPagedE4Kv, OrdinaryPagesKeepSeparateScalesAndOldResults) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    paged_e4_kv_test::paged(resource,queues,compile,write,read,false,workspace);
  });
}
TEST(CudaPagedE4Kv, FixedQ1GrowsHistoryThroughOriginalReader) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    paged_e4_kv_test::paged(resource,queues,compile,write,read,true,workspace);
  });
}
