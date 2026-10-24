#include "../compute_models/ct_packed_grouped_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace f = ct_packed_grouped_test;
void exercise(unsigned journey) {
  auto context = f::take(f::g::cuda::create_context(), "CUDA Context");
  auto resource = f::take(f::c::cuda::MemoryResource::create(context,
      f::c::cuda::MemoryKind::kDevice), "CUDA MemoryResource");
  const auto owner = f::take(f::c::StorageOwner::adopt(context, std::move(resource)), "CUDA StorageOwner");
  auto queue = f::take(context.create_queue(), "CUDA Queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  const auto compile = [&](const f::g::Graph &graph, f::g::CompileOptions options, bool) {
    return f::g::cuda::compile(graph, context, owner, options);
  };
  const auto write = [](f::m::BufferView destination, std::span<const std::byte> bytes) {
    return destination.write(bytes);
  };
  const auto read = [](f::m::ConstBufferView source, std::span<std::byte> bytes) {
    return source.read(bytes);
  };
  const auto prepare = [](f::g::Invocation &slot,
                          std::span<const f::g::QueueBinding> bindings, bool packed) {
    const auto status = f::g::cuda::prepare_replay(slot, *bindings[0].queue,
        {.mode = packed ? f::g::ReplayMode::kRequireCapture : f::g::ReplayMode::kCommands,
         .host_wait = f::g::HostWaitPolicy::kAllow});
    if (status.ok() && packed) EXPECT_EQ(slot.replay_route(), f::g::ReplayRoute::kCaptured);
    return status;
  };
  if (journey == 0) f::run(owner.ref(), queues, compile, write, read, prepare);
  else if (journey == 1) f::run_domain(owner.ref(), queues, compile, write, read, prepare);
  else f::run_escape(owner.ref(), queues, compile, write, read, prepare);
}
TEST(CudaCtPackedGrouped, OriginalBanksKeepRoundingStridesSlotsAndResults) { exercise(0); }
TEST(CudaCtPackedGrouped, CompleteIdsValidateBeforeSelectedBankAccess) { exercise(1); }
TEST(CudaCtPackedGrouped, PublishedDecodeRetainsTheOriginalGraph) { exercise(2); }
} // namespace
