#include "../compute_models/ga_encoded_bank_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace f = expert_working_set_test;
namespace cu = f::c::cuda;
struct Backend final {
  using Batch = cu::WeightBatch;
  using Slots = cu::WeightSlots;
  f::c::Context &context;
  cu::MemoryResource &host, &device;
  f::c::Queue &transfer;
  f::m::MemoryResourceRef resource;
  std::span<const f::g::QueueBinding> queues;
  hyle::Result<Slots> create_slots(std::span<const f::BankSpec> banks,
                                  std::size_t capacity) {
    std::vector<cu::WeightSpec> catalog;
    for (const auto &bank : banks) catalog.push_back({bank.id,bank.description});
    return Slots::create(context,host,device,catalog,capacity);
  }
  hyle::Result<Batch> upload(Slots &slots,std::size_t slot,
                             std::span<const f::c::WeightSlice> slices) {
    return slots.upload_slices(slot,slices,transfer);
  }
  hyle::Status bind(f::g::Invocation &invocation,
      std::span<const f::g::WeightBinding> bindings,const Batch &batch) {
    return f::g::cuda::bind(invocation,bindings,batch);
  }
  hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph) {
    return f::g::cuda::compile(graph,context,resource);
  }
  hyle::Status prepare(f::g::Invocation &invocation) {
    auto status=f::g::cuda::prepare_replay(invocation,*queues[0].queue,
        {.mode=f::g::ReplayMode::kRequireCapture,.host_wait=f::g::HostWaitPolicy::kAllow});
    if(status.ok()) EXPECT_EQ(invocation.replay_route(),f::g::ReplayRoute::kCaptured);
    return status;
  }
  hyle::Status write(f::m::BufferView destination,std::span<const std::byte> bytes) {
    return destination.write(bytes);
  }
  hyle::Status read(f::m::ConstBufferView source,std::span<std::byte> bytes) {
    return source.read(bytes);
  }
};
template<class Run> void run_cuda(Run run) {
  auto context=f::take(f::g::cuda::create_context(),"CUDA context");
  auto host=f::take(cu::MemoryResource::create(context,cu::MemoryKind::kPinnedHost),"original pinned staging resource");
  auto device=f::take(cu::MemoryResource::create(context,cu::MemoryKind::kDevice),"original device resource");
  auto compute=f::take(context.create_queue(),"original compute queue");
  auto transfer=f::take(context.create_queue(),"original transfer queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&compute}};
  Backend api{context,host,device,transfer,device.ref(),queues};
  run(api);
}
} // namespace

TEST(CudaGaEncodedBank, NewEncodedOriginalFp4) {
  run_cuda([](auto &api) { ga_encoded_bank_test::run(api); });
}
TEST(CudaGaEncodedBank, PlainNvfp4Neighbor) {
  run_cuda([](auto &api) { f::run_lifetimes(api,f::Family::kNvfp4); });
}
