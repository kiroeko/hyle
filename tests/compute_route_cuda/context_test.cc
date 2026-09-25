#include "../compute/workload_fixture.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "ffn_support.h"
#include "hyle/compute/providers/flashinfer.h"
#include "hyle/compute/providers/ggml_cuda.h"

namespace {
using namespace cuda_route_test;

void run_ffn(compute::DataType type, std::int64_t rows) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        cuda::create_context({.metadata_allocator = stats.get()}).value();
    const std::array expected{
#if defined(__linux__)
        compute::flashinfer::kProviderId,
#endif
        compute::ggml_cuda::kProviderId,
        compute::cublas::kProviderId,
        compute::cuda_native::kSglangProviderId,
        compute::cuda_native::kCcclProviderId,
        compute::cuda_native::kProviderId};
    ASSERT_EQ(owner.provider_count(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(owner.provider_identity(i)->id, expected[i]);
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Linear first(resource.ref(), type, 128, 256, 1);
    Linear second(resource.ref(), type, 128, 128, 4);
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
    auto workload =
        workload_test::known(rows == 1 ? compute::WorkloadPhase::kDecode
                                       : compute::WorkloadPhase::kPrefill);
    auto a = compute::prepare(
        owner, first.operation(), a_inputs,
        std::array{lanes[0].gate_up.view}, resource.ref(), workload);
    ASSERT_TRUE(a.has_value()) << a.error().message();
    auto b = compute::prepare(owner, compute::GatedActivation{},
                              lanes[0].gate_up.view.description(),
                              resource.ref(), workload);
    ASSERT_TRUE(b.has_value()) << b.error().message();
    auto c = compute::prepare(
        owner, second.operation(), b_inputs,
        std::array{lanes[0].output.view}, resource.ref(), workload);
    ASSERT_TRUE(c.has_value()) << c.error().message();
    std::array programs{std::move(*a), std::move(*b), std::move(*c)};
    for (const auto &program : programs)
      EXPECT_EQ(program.workload(), workload);
    workload = {};
    EXPECT_TRUE(programs[0].workload().profile.matches(workload_test::kModel));
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
            queues[lane], first_inputs,
            std::array{run.gate_up.view});
        ASSERT_TRUE(done_a.has_value()) << done_a.error().message();
        completed[lane][0] = std::move(*done_a);
        auto done_b = run.invocations[1].submit(
            queues[lane], run.gate_up.view.as_const(), run.activated.view);
        ASSERT_TRUE(done_b.has_value()) << done_b.error().message();
        completed[lane][1] = std::move(*done_b);
        auto done_c = run.invocations[2].submit(
            queues[lane], second_inputs,
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

TEST(CudaDefaultCompute, DenseFfnUsesFixedRulesAndIndependentSlots) {
  for (const auto rows : {1, 17}) {
    SCOPED_TRACE(rows);
    run_ffn(compute::DataType::kFloat32, rows);
    run_ffn(compute::DataType::kFloat16, rows);
  }
}

TEST(CudaDefaultCompute, D5RealQwenProjectionUsesExactProfileAndReplays) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Projection fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_projection(*fixture_root));

  const auto activation_desc =
      matrix(compute::DataType::kBFloat16, 1,
             d5_qwen35_fixture::kHidden);
  const auto stored_weight_desc =
      matrix(compute::DataType::kBFloat16, d5_qwen35_fixture::kHidden,
             d5_qwen35_fixture::kHidden);
  const auto weight_desc = stored_weight_desc.transposed().value();
  const auto workload = d5_qwen35_fixture::workload(
      compute::WorkloadPhase::kDecode);
  auto owner = cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto prepared = compute::prepare(owner, compute::MatMul{}, activation_desc,
                                   weight_desc, resource.ref(), workload);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::cublas::kProviderId);
  EXPECT_EQ(prepared->workload(), workload);

  auto activation = mem::Buffer::create(resource.ref(), fixture.activation.size(),
                                        256)
                        .value();
  auto weight =
      mem::Buffer::create(resource.ref(), fixture.weight.size(), 256).value();
  const auto output_desc = prepared->output_description().value();
  auto output = mem::Buffer::create(resource.ref(),
                                    output_desc.required_bytes().value(), 256)
                    .value();
  ASSERT_TRUE(weight.view().write(fixture.weight).ok());
  const std::array inputs{
      compute::ConstTensorView::create(activation_desc, activation.view())
          .value(),
      compute::ConstTensorView::create(weight_desc, weight.view()).value()};
  const std::array outputs{
      compute::TensorView::create(output_desc, output.view()).value()};
  const auto requirement = prepared->workspace_requirement();
  auto workspace =
      requirement.size == 0
          ? mem::Buffer{}
          : mem::Buffer::create(resource.ref(), requirement.size,
                                requirement.alignment)
                .value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto queue = owner.create_queue().value();
  for (int run = 0; run < 2; ++run) {
    auto values = fixture.activation;
    if (run != 0) {
      for (std::size_t index = 0; index < d5_qwen35_fixture::kHidden;
           ++index)
        values[index * 2 + 1] ^= std::byte{0x80};
    }
    ASSERT_TRUE(activation.view().write(values).ok());
    auto done = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<hyle::simd::bfloat16_t>(output);
    ASSERT_EQ(actual.size(), fixture.expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
      half_test::expect_near(actual[index],
                             run == 0 ? fixture.expected[index]
                                      : -fixture.expected[index]);
  }
}

class CudaQwen38GdnProfiles
    : public ::testing::TestWithParam<
          const compute::ModelArtifactProfileDescription *> {};

TEST_P(CudaQwen38GdnProfiles, GroupedHeadUsesOrdinaryRouteAndReplays) {
  const d5_qwen38_gdn_fixture::Fixture fixture;
  auto owner = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  std::vector<mem::Buffer> input_storage;
  std::vector<compute::ConstTensorView> inputs;
  input_storage.reserve(fixture.inputs.size());
  inputs.reserve(fixture.inputs.size());
  for (std::size_t port = 0; port < fixture.inputs.size(); ++port) {
    input_storage.push_back(
        mem::Buffer::create(resource,
                            fixture.inputs[port].required_bytes().value(), 256)
            .value());
    ASSERT_TRUE(input_storage.back()
                    .view()
                    .write(std::as_bytes(std::span(fixture.data[port])))
                    .ok());
    inputs.push_back(compute::ConstTensorView::create(
                         fixture.inputs[port], input_storage.back().view())
                         .value());
  }
  std::array output_storage{
      mem::Buffer::create(resource,
                          fixture.outputs[0].required_bytes().value(), 256)
          .value(),
      mem::Buffer::create(resource,
                          fixture.outputs[1].required_bytes().value(), 256)
          .value()};
  const std::array outputs{
      compute::TensorView::create(fixture.outputs[0], output_storage[0].view())
          .value(),
      compute::TensorView::create(fixture.outputs[1], output_storage[1].view())
          .value()};
  const auto workload = d5_qwen38_gdn_fixture::workload(
      *GetParam(), compute::WorkloadPhase::kDecode);
  auto prepared = compute::prepare(owner,
                                   compute::Operation(compute::GatedDeltaNet{}),
                                   inputs, outputs, resource, workload);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->workload(), workload);
  const auto requirement = prepared->workspace_requirement();
  auto workspace =
      requirement.size == 0
          ? mem::Buffer{}
          : mem::Buffer::create(resource, requirement.size,
                                requirement.alignment)
                .value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto queue = owner.create_queue().value();
  const auto expected = fixture.oracle();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      const auto actual = half_test::read<float>(output_storage[port]);
      ASSERT_EQ(actual.size(), expected[port].size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[port][i], 4.e-6) << port << ':' << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, CudaQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kOfficialProfile,
                      &d5_qwen38_gdn_fixture::kPrimaryGgufProfile));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, CudaQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kFidelityGgufProfile));

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
    EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
    auto lane = prepared->create_invocation().value();
    auto done = lane.submit(queue, inputs, std::array{output.view});
    ASSERT_TRUE(done.has_value());
    ASSERT_TRUE(done->wait().ok());
    for (const auto value : output.read())
      EXPECT_NEAR(value, 1.0 / (1.0 + std::exp(-1.0)), 0.000001);
  }
}

} // namespace
