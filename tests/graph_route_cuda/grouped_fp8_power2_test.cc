#include "../compute_models/grouped_fp8_power2_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
template<class Run> void run_cuda(Run run) {
  namespace f = grouped_fp8_power2_test;
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

TEST(CudaGroupedFp8Power2Graph, OriginalCodesAndPowerScalesSurviveSlotsAndUpdates) {
  run_cuda([](auto resource, auto queues, auto compile, auto write, auto read,
      auto prepare, auto owner) {
    grouped_fp8_power2_test::run(resource, queues, compile, write, read, prepare, owner);
  });
}

TEST(CudaGroupedFp8Power2Graph, CheckedIdsFailBeforeSelectionAndFreshSlotRecovers) {
  run_cuda([](auto resource, auto queues, auto compile, auto write, auto read,
      auto prepare, auto owner) {
    grouped_fp8_power2_test::run_checked_ids(resource, queues, compile, write, read, prepare, owner);
  });
}

TEST(CudaGroupedFp8Power2Graph, Float32PowerShapeInfersButNativePrepareDeclines) {
  namespace f = grouped_fp8_power2_test;
  auto context = f::take(f::c::Context::create(f::take(f::c::cuda::device(0), "CUDA device"),
      f::c::cuda_native::sglang_provider()), "native CUDA context");
  auto memory = f::take(f::c::cuda::MemoryResource::create(context,
      f::c::cuda::MemoryKind::kDevice), "CUDA resource");
  const std::array inputs{
      f::take(f::c::float8_grouped_weight_description(3, 129, 256), "encoded shape"),
      f::dense(f::Type::kFloat32, {2, 1, 256}),
      f::dense(f::Type::kInt32, {2, 3})};
  const auto operation = f::c::model_ops::grouped_float8_linear(
      f::Policy::kPowerOfTwo, 1.e-4F);
  const auto output = f::c::infer_single_output(operation, inputs);
  ASSERT_TRUE(output) << output.error().message();
  EXPECT_EQ(output->data_type(), f::Type::kFloat32);
  auto prepared = f::c::prepare(context, operation, inputs, std::array{*output}, memory.ref());
  ASSERT_FALSE(prepared);
  EXPECT_TRUE(prepared.error().IsUnimplemented()) << prepared.error().message();
  // No payload, Invocation, queue or submit: BF16 AOT never receives F32 data.
  ::testing::Test::RecordProperty("successful_submissions", 0);
}
