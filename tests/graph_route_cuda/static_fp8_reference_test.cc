#include "../compute_models/static_fp8_reference_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/compute/providers/torch_cuda.h"

namespace {
template<class Run> void run_cuda(Run run, bool contracted) {
  namespace f = static_fp8_reference_test;
  auto context = f::take(f::g::cuda::create_context(), "CUDA context");
  auto memory = f::take(f::c::cuda::MemoryResource::create(context,
      f::c::cuda::MemoryKind::kDevice), "CUDA memory");
  const auto owner = f::take(f::c::StorageOwner::adopt(context, std::move(memory)),
      "CUDA storage owner");
  auto queue = f::take(context.create_queue(), "CUDA queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  run(owner.ref(), queues,
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        const std::array providers{f::c::torch_cuda::kProviderId};
        if (contracted) options.allowed_providers = providers;
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

TEST(CudaStaticFp8ReferenceGraph, StoredCodesScalesAndCodeDotSurviveSlotsAndUpdates) {
  run_cuda([](auto resource, auto queues, auto compile, auto write, auto read,
      auto prepare, auto owner) {
    static_fp8_reference_test::run_lifetimes(resource, queues, compile, write, read, prepare, owner);
  }, true);
}
TEST(CudaStaticFp8ReferenceGraph, IeeeScalarUpdatesKeepReciprocalAndSeparateScaleRounds) {
  run_cuda([](auto resource, auto queues, auto compile, auto write, auto read,
      auto prepare, auto) {
    static_fp8_reference_test::run_runtime_domain(resource, queues, compile, write, read, prepare);
  }, true);
}
TEST(CudaStaticFp8ReferenceGraph, PublishedCodeValuesKeepTheOriginalCapturedDecomposition) {
  run_cuda([](auto resource, auto queues, auto compile, auto write, auto read,
      auto prepare, auto) {
    static_fp8_reference_test::run_escape(resource, queues, compile, write, read, prepare);
  }, false);
}
