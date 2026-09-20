#include "ffn_support.h"
#include "../compute/workload_fixture.h"
#include "hyle/compute/providers/flashinfer.h"
#include "hyle/compute/providers/ggml_cuda.h"

namespace {
using namespace cuda_route_test;

void run_ffn(compute::DataType type, std::int64_t rows, bool first_q4,
             bool second_q4) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        cuda::create_context({.metadata_allocator = stats.get()}).value();
    const std::array expected{
#if defined(__linux__)
        compute::flashinfer::kProviderId,
#endif
        compute::ggml_cuda::kProviderId, compute::cublas::kProviderId,
        compute::marlin::kProviderId, compute::cuda_native::kSglangProviderId,
        compute::cuda_native::kProviderId};
    ASSERT_EQ(owner.provider_count(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(owner.provider_identity(i)->id, expected[i]);
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Linear first(owner, resource.ref(), type, 128, 256, first_q4, 1);
    Linear second(owner, resource.ref(), type, 128, 128, second_q4, 4);
    struct Lane final {
      Tensor input, gate_up, activated, output;
      std::array<mem::Buffer, 3> workspace;
      std::array<compute::OperationInvocation, 3> invocations;
      Lane(compute::DataType dtype, std::int64_t m,
           mem::MemoryResourceRef memory)
          : input(matrix(dtype, m, 128), memory),
            gate_up(matrix(dtype, m, 256), memory),
            activated(matrix(dtype, m, 128), memory),
            output(matrix(dtype, m, 128), memory) {}
    };
    std::array lanes{Lane(type, rows, resource.ref()),
                     Lane(type, rows, resource.ref())};
    const auto a_inputs = first.inputs(lanes[0].input.view.as_const());
    const auto b_inputs = second.inputs(lanes[0].activated.view.as_const());
    auto workload = workload_test::known(rows == 1
                                            ? compute::WorkloadPhase::kDecode
                                            : compute::WorkloadPhase::kPrefill);
    auto a = compute::prepare(owner, first.operation(),
                              std::span(a_inputs).first(first.arity()),
                              std::array{lanes[0].gate_up.view}, resource.ref(),
                              workload);
    ASSERT_TRUE(a.has_value()) << a.error().message();
    auto b =
        compute::prepare(owner, compute::GatedActivation{},
                         lanes[0].gate_up.view.description(), resource.ref(),
                         workload);
    ASSERT_TRUE(b.has_value()) << b.error().message();
    auto c = compute::prepare(owner, second.operation(),
                              std::span(b_inputs).first(second.arity()),
                              std::array{lanes[0].output.view}, resource.ref(),
                              workload);
    ASSERT_TRUE(c.has_value()) << c.error().message();
    std::array programs{std::move(*a), std::move(*b), std::move(*c)};
    for (const auto &program : programs)
      EXPECT_EQ(program.workload(), workload);
    workload = {};
    EXPECT_TRUE(programs[0].workload().model.matches(workload_test::kModel));
    EXPECT_EQ(programs[0].provider().id, first.provider());
    EXPECT_EQ(programs[1].provider().id,
              compute::cuda_native::kSglangProviderId);
    EXPECT_EQ(programs[2].provider().id, second.provider());
    std::array queues{owner.create_queue().value(),
                      owner.create_queue().value()};
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
      ASSERT_TRUE(queues[lane].reserve_completions(3).ok());
      for (std::size_t op = 0; op < programs.size(); ++op) {
        const auto needed = programs[op].workspace_requirement();
        lanes[lane].workspace[op] =
            mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
                .value();
        lanes[lane].invocations[op] =
            programs[op]
                .create_invocation(lanes[lane].workspace[op].view())
                .value();
      }
    }
    const auto before = stats.snapshot();
    for (int seed = 0; seed < 3; ++seed) {
      std::array<std::vector<float>, 2> input;
      std::array<std::array<compute::Completion, 3>, 2> completed;
      for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        auto &run = lanes[lane];
        input[lane] = activations(type, static_cast<std::size_t>(rows) * 128,
                                  seed + static_cast<int>(lane) * 3);
        run.input.write(input[lane]);
      }
      for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        auto &run = lanes[lane];
        const auto first_inputs = first.inputs(run.input.view.as_const());
        const auto second_inputs = second.inputs(run.activated.view.as_const());
        auto done_a = run.invocations[0].submit(
            queues[lane], std::span(first_inputs).first(first.arity()),
            std::array{run.gate_up.view});
        ASSERT_TRUE(done_a.has_value()) << done_a.error().message();
        completed[lane][0] = std::move(*done_a);
        auto done_b = run.invocations[1].submit(
            queues[lane], run.gate_up.view.as_const(), run.activated.view);
        ASSERT_TRUE(done_b.has_value()) << done_b.error().message();
        completed[lane][1] = std::move(*done_b);
        auto done_c = run.invocations[2].submit(
            queues[lane], std::span(second_inputs).first(second.arity()),
            std::array{run.output.view});
        ASSERT_TRUE(done_c.has_value()) << done_c.error().message();
        completed[lane][2] = std::move(*done_c);
      }
      for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        for (auto &done : completed[lane])
          ASSERT_TRUE(done.wait().ok());
        check_ffn(lanes[lane].output, input[lane], first, second);
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CudaDefaultCompute,
     DenseAndCompressedFfnUseFixedRulesAndIndependentSlots) {
  for (const auto rows : {1, 17}) {
    SCOPED_TRACE(rows);
    run_ffn(compute::DataType::kFloat32, rows, false, false);
    run_ffn(compute::DataType::kFloat16, rows, false, false);
    run_ffn(compute::DataType::kFloat16, rows, true, false);
    run_ffn(compute::DataType::kFloat16, rows, true, true);
  }
}

TEST(CudaDefaultCompute, RejectsInvalidDeviceAndUnsupportedOperations) {
  EXPECT_FALSE(cuda::create_context({.device_ordinal = SIZE_MAX}).has_value());
  auto owner = cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto unsupported = compute::prepare(
      owner, compute::Unary{compute::UnaryKind::kNeg},
      matrix(compute::DataType::kFloat64, 1, 128), resource.ref());
  ASSERT_FALSE(unsupported.has_value());
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  ASSERT_TRUE(owner.close().ok());
  EXPECT_FALSE(owner.create_queue().has_value());
}

TEST(CudaDefaultCompute,
     DescriptorPreparationChecksDeviceResourcesBeforePayload) {
  auto owner = cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto description = matrix(compute::DataType::kBFloat16, 1, 4096);
  auto prepared = compute::prepare(owner, compute::GatedActivation{},
                                   description, resource.ref());
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::cuda_native::kSglangProviderId);
  EXPECT_TRUE(prepared->capture_compatible());
  auto wrong_resource = compute::prepare(owner, compute::GatedActivation{},
                                         description, mem::system_resource());
  ASSERT_FALSE(wrong_resource.has_value());
  EXPECT_TRUE(wrong_resource.error().IsUnimplemented());
  ASSERT_TRUE(owner.close().ok());
  auto closed = compute::prepare(owner, compute::GatedActivation{}, description,
                                 resource.ref());
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(CudaDefaultCompute, DefaultRulesRetainSeparateAndPackedTailCapability) {
  auto owner = cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  for (const bool packed : {false, true}) {
    const std::int64_t hidden = packed ? 127 : 128;
    Tensor first(
        matrix(compute::DataType::kFloat32, 1, hidden * (packed ? 2 : 1)),
        resource.ref());
    Tensor second(matrix(compute::DataType::kFloat32, 1, hidden),
                  resource.ref());
    Tensor output(matrix(compute::DataType::kFloat32, 1, hidden),
                  resource.ref());
    first.write(std::vector<float>(
        static_cast<std::size_t>(hidden) * (packed ? 2 : 1), 1.0F));
    second.write(std::vector<float>(static_cast<std::size_t>(hidden), 1.0F));
    const std::array all_inputs{first.view.as_const(), second.view.as_const()};
    const auto inputs = std::span(all_inputs).first(packed ? 1 : 2);
    auto prepared = compute::prepare(
        owner,
        compute::Operation(compute::GatedActivation{
            .inputs = packed ? compute::GatedInputKind::kPackedHalves
                             : compute::GatedInputKind::kSeparate}),
        inputs, std::array{output.view});
    ASSERT_TRUE(prepared.has_value());
    EXPECT_EQ(prepared->provider().id, compute::cuda_native::kProviderId);
    auto lane = prepared->create_invocation().value();
    auto done = lane.submit(queue, inputs, std::array{output.view});
    ASSERT_TRUE(done.has_value());
    ASSERT_TRUE(done->wait().ok());
    for (const auto value : output.read())
      EXPECT_NEAR(value, 1.0 / (1.0 + std::exp(-1.0)), 0.000001);
  }
}

} // namespace
