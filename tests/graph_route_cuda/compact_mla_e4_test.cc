#include "../compute_models/compact_mla_e4_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/flashinfer.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f=compact_mla_e4_test;
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
TEST(CudaCompactMlaE4, PagedRope64SharesScaleAndOldResults) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,576,true,false,workspace,false,false);
  });
}

TEST(CudaCompactMlaE4, RaggedNoPeSharesScaleAndOldResults) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,512,false,false,workspace,false,false);
  });
}

TEST(CudaCompactMlaE4, PagedNoPeQ1GrowthMatchesFreshInvocation) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,512,true,true,workspace,false,false);
  });
}

TEST(CudaCompactMlaE4, RaggedRope64Q1GrowthMatchesFreshInvocation) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,576,false,true,workspace,false,false);
  });
}

TEST(CudaCompactMlaE4, OriginalF32ScalePrecedesBfloatTileRounding) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto workspace) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,512,true,true,workspace,true,false);
  });
}
