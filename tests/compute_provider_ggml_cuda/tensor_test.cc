#include "tensor_test_support.h"

namespace {
using namespace ggml_cuda_tensor_test;

Context context() {
  return Context::create(cuda::device(0).value(), ggml_cuda::provider())
      .value();
}

TEST(GgmlCudaTensor, FloatingNormAffineAndUnaryConversions) {
  auto ctx = context();
  norms<float>(ctx);
  norms<hyle::simd::float16_t>(ctx);
  norms<hyle::simd::bfloat16_t>(ctx);
  unary_and_cast(ctx);
}

TEST(GgmlCudaTensor, LayerNormRetainsSmallVarianceAtLargeOffset) {
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto description = dense(DataType::kFloat32, {2, 32});
  std::array<float, 64> values;
  values.fill(1.e6F);
  for (std::size_t i = 0; i < 32; ++i)
    values[i] += i % 2 == 0 ? -0.25F : 0.25F;
  auto source = upload<float>(memory.ref(), values);
  auto output = output_buffer(memory.ref(), description);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  const auto status =
      execute(ctx, LayerNorm{}, std::span(&input, 1),
              TensorView::create(description, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = half_test::read<float>(output);
  for (std::size_t i = 0; i < 64; ++i)
    EXPECT_NEAR(actual[i],
                (values[i] - 1.e6F) /
                    std::sqrt((i < 32 ? 0.0625F : 0.F) + 1.e-5F),
                2.e-6F);
}

TEST(GgmlCudaTensor, BroadcastUsesLogicalAxesAndBothOperandOrders) {
  auto ctx = context();
  check_general_broadcast<float>(ctx);
  check_general_broadcast<hyle::simd::float16_t>(ctx);
  check_general_broadcast<hyle::simd::bfloat16_t>(ctx);
}

TEST(GgmlCudaTensor, SelectedReductionsUseDenseAndStridedLogicalCoordinates) {
  auto ctx = context();
  const auto check = [&]<class T>() {
    check_reduction_axes<T>(ctx, {2, 3, 2, 2, 3, 5});
    check_reduction_axes<T>(ctx, {2, 3, 5}, true);
    check_reduction_axes<T>(ctx, {});
  };
  check.template operator()<float>();
  check.template operator()<hyle::simd::float16_t>();
  check.template operator()<hyle::simd::bfloat16_t>();
}

TEST(GgmlCudaTensor, StridedCopyAndVariadicConcatPreserveBits) {
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto description =
      TensorDesc::strided_bytes(DataType::kFloat32,
                                std::array<std::int64_t, 3>{2, 3, 7},
                                std::array<std::int64_t, 3>{168, 56, 8})
          .value();
  std::vector<float> values(84);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i) / 8;
  auto source = upload<float>(memory.ref(), values);
  auto output_desc = dense(DataType::kFloat32, {2, 3, 7});
  auto output = output_buffer(memory.ref(), output_desc);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  auto status = execute(ctx, Copy{}, std::span(&input, 1),
                        TensorView::create(output_desc, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto copied = half_test::read<float>(output);
  for (std::size_t i = 0; i < copied.size(); ++i)
    EXPECT_EQ(copied[i], values[i * 2]);
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    const std::array pieces{input, input, input, input};
    std::array<TensorDesc, 4> descriptions;
    descriptions.fill(description);
    const auto joined =
        infer_single_output(Operation(Concat{axis}), descriptions).value();
    auto result = output_buffer(memory.ref(), joined);
    status = execute(ctx, Concat{axis}, pieces,
                     TensorView::create(joined, result.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(result);
    std::size_t inner = 1;
    for (std::size_t d = static_cast<std::size_t>(axis); d < 3; ++d)
      inner *= static_cast<std::size_t>(description.dimensions()[d]);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], copied[(i / (inner * 4)) * inner + i % inner]);
  }
}

TEST(GgmlCudaTensor, UnsupportedNumericDomainsDeclineBeforeNativeExecution) {
  auto ctx = context();
  const auto description = dense(DataType::kFloat32, {2, 31});
  for (const auto kind :
       {UnaryKind::kSilu, UnaryKind::kSigmoid, UnaryKind::kSoftplus})
    EXPECT_FALSE(prepare(ctx, Unary{kind}, description));
  const auto odd = TensorDesc::strided_bytes(DataType::kFloat32,
                                             std::array<std::int64_t, 1>{7},
                                             std::array<std::int64_t, 1>{5})
                       .value();
  EXPECT_FALSE(prepare(ctx, Copy{}, odd));
}
} // namespace
