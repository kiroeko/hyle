#include "../compute_provider_cuda/collectives_test_support.h"
#include "../compute_provider_ggml_cuda/softmax_test_support.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_provider_ggml_cuda/test_support.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/low_precision.h"
#include <ggml.h>

#include <algorithm>
#include <vector>

TEST(CudaDefault, OriginalCcclThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  cccl_test::scan<std::int32_t>(context, std::array<std::int64_t, 3>{2, 257, 3},
                                1, 1);
  cccl_test::topk<float>(context, std::array<std::int64_t, 2>{2, 8193}, -1, 0,
                         7);
  cccl_test::gather<std::uint16_t, std::int64_t>(
      context, hyle::compute::DataType::kBFloat16,
      std::array<std::int64_t, 3>{2, 7, 3}, 1,
      std::array<std::int64_t, 2>{2, 3}, 1, 1);
}

TEST(CudaDefault, OriginalGgmlThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_test::exercise_direct(context);
}

TEST(CudaDefault, NativeQ8UsesOrdinaryDirectRouteAndReplays) {
  using namespace hyle::compute;
  constexpr std::int64_t reduction = 32;
  constexpr std::int64_t columns = 2;
  const auto left_desc =
      TensorDesc::dense(DataType::kFloat32,
                        std::array{std::int64_t{1}, reduction})
          .value();
  const auto right_desc = gguf_weight_description(
      GgufQuantizedType::kQ8_0, DataType::kFloat32, reduction, columns);
  ASSERT_TRUE(right_desc) << right_desc.error().message();
  const auto output_desc =
      TensorDesc::dense(DataType::kFloat32,
                        std::array{std::int64_t{1}, columns})
          .value();
  std::vector<float> left_values(static_cast<std::size_t>(reduction), 0.25F);
  std::vector<float> unpacked(static_cast<std::size_t>(reduction * columns));
  std::fill_n(unpacked.begin(), reduction, 1.0F);
  std::fill_n(unpacked.begin() + reduction, reduction, -2.0F);
  std::vector<std::byte> packed(*right_desc->required_bytes());
  ASSERT_EQ(ggml_quantize_chunk(GGML_TYPE_Q8_0, unpacked.data(), packed.data(),
                                0, columns, reduction, nullptr),
            packed.size());

  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto left = ggml_cuda_tensor_test::upload<float>(memory.ref(), left_values);
  auto right = ggml_cuda_tensor_test::upload<std::byte>(memory.ref(), packed);
  auto output = ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc);
  const std::array inputs{
      ConstTensorView::create(left_desc, left.view()).value(),
      ConstTensorView::create(*right_desc, right.view()).value()};
  auto output_view = TensorView::create(output_desc, output.view()).value();
  auto prepared =
      prepare(context, Operation(MatMul{}), inputs, std::span(&output_view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->route_policy().namespace_id, 0x48594c4543554444ULL);
  EXPECT_EQ(prepared->route_preference(), 100u);
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  for (int repeat = 1; repeat <= 2; ++repeat) {
    auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    ASSERT_EQ(actual.size(), 2U);
    EXPECT_NEAR(actual[0], 8.0F * repeat, 0.1F);
    EXPECT_NEAR(actual[1], -16.0F * repeat, 0.1F);
    if (repeat == 1) {
      std::fill(left_values.begin(), left_values.end(), 0.5F);
      ASSERT_TRUE(
          left.view().write(std::as_bytes(std::span(left_values))).ok());
    }
  }
}

TEST(CudaDefault, OriginalTensorBatchThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_tensor_test::norms<float>(context);
  ggml_cuda_tensor_test::check_general_broadcast<float>(context);
}

TEST(CudaDefault, EngineeringPolicyIsColdAndDoesNotLeakToExpertFactory) {
  using namespace hyle::compute;
  const auto description = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 64}).value();
  auto context = cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  for (auto phase : {WorkloadPhase::kUnspecified, WorkloadPhase::kPrefill, WorkloadPhase::kDecode}) {
    const WorkloadContext workload{.profile = {}, .phase = phase, .region = {}};
    auto prepared = prepare(context, Operation(Unary{UnaryKind::kExp}),
        std::span(&description, 1), std::span(&description, 1), memory.ref(), workload);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    EXPECT_EQ(prepared->route_policy().namespace_id, 0x48594c4543554444ULL);
    EXPECT_EQ(prepared->route_preference(), 100u);
    EXPECT_EQ(prepared->workload(), workload);
  }
  auto selected = cuda::device(0).value();
  auto expert = Context::create(selected, cublas::provider()).value();
  const auto right = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 32}).value();
  auto program = prepare(expert, MatMul{}, description, right, memory.ref());
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_TRUE(program->route_policy().empty());
}

TEST(CudaDefault, PlainGemmEngineeringDefaultKeepsCublasResources) {
  using namespace hyle::compute;
  const auto lhs = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 17}).value();
  const auto rhs = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 2>{17, 65}).value();
  auto context = cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto program = prepare(context, MatMul{}, lhs, rhs, memory.ref());
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->provider().id, cublas::kProviderId);
  EXPECT_EQ(program->route_policy().namespace_id, 0x48594c4543554444ULL);
  EXPECT_EQ(program->route_preference(), 100u);
  EXPECT_EQ(program->runtime_properties().workspace_owner, ProviderWorkspaceOwner::kCaller);
  EXPECT_TRUE(program->runtime_properties().capture_compatible);
}

TEST(CudaDefault, OriginalSoftmaxThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_tensor_test::softmax_case<float>(context, {2, 3, 17}, 1, true,
                                             false, true);
}
