#include "../compute_models/e8m0_weight_import_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f = e8m0_weight_import_test;
  auto context = f::take(f::g::cuda::create_context(), "CUDA context");
  auto memory = f::take(f::c::cuda::MemoryResource::create(context,
      f::c::cuda::MemoryKind::kDevice), "CUDA memory");
  const auto owner = f::take(f::c::StorageOwner::adopt(context, std::move(memory)),
      "CUDA storage owner");
  auto queue = f::take(context.create_queue(), "CUDA queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  run(owner.ref(), queues,
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::cuda::compile(graph, context, owner, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) {
        return destination.write(source);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) {
        return source.read(destination);
      },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) {
        const auto status = f::g::cuda::prepare_replay(invocation, *bindings[0].queue,
            {.mode = f::g::ReplayMode::kRequireCapture});
        if (status.ok()) EXPECT_EQ(invocation.replay_route(), f::g::ReplayRoute::kCaptured);
        return status;
      }, owner);
}
} // namespace

TEST(CudaE8M0WeightImport, ColdCanonicalBankUsesOriginalLinearAndResultLifetime) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare,auto owner) {
    e8m0_weight_import_test::run(resource,queues,compile,write,read,prepare,owner);
  });
}

TEST(CudaE8M0WeightImport, StoredFloat32ScalesKeepDynamicActivationQuantization) {
  run_cuda([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare,auto owner) {
    e8m0_weight_import_test::run(resource,queues,compile,write,read,prepare,owner,true);
  });
}
