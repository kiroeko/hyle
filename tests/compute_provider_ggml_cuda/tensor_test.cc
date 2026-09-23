#include "softmax_test_support.h"
#include "tensor_test_support.h"

#include "hyle/compute/model_ops.h"

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

TEST(GgmlCudaTensor, MatureCausalConvUsesNaturalSequenceLayout) {
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();
  constexpr std::size_t channels = 128;
  const auto samples_desc = dense(DataType::kFloat32, {1, 5, channels});
  const auto weights_desc = dense(DataType::kFloat32, {channels, 3});
  const auto output_desc = dense(DataType::kFloat32, {1, 3, channels});
  std::vector<float> samples(5 * channels), weights(3 * channels),
      expected(3 * channels);
  for (std::size_t token = 0; token < 5; ++token)
    for (std::size_t channel = 0; channel < channels; ++channel)
      samples[token * channels + channel] =
          static_cast<float>(token + 1) + static_cast<float>(channel) / 256.F;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    weights[channel * 3] = 1.F;
    weights[channel * 3 + 1] = 2.F;
    weights[channel * 3 + 2] = 3.F;
    for (std::size_t token = 0; token < 3; ++token)
      expected[token * channels + channel] =
          samples[token * channels + channel] +
          2.F * samples[(token + 1) * channels + channel] +
          3.F * samples[(token + 2) * channels + channel];
  }
  auto samples_buffer = upload<float>(resource, samples);
  auto weights_buffer = upload<float>(resource, weights);
  auto output = output_buffer(resource, output_desc);
  const std::array inputs{
      ConstTensorView::create(samples_desc, samples_buffer.view()).value(),
      ConstTensorView::create(weights_desc, weights_buffer.view()).value()};
  const auto status =
      execute(ctx, model_ops::causal_conv1d(), inputs,
              TensorView::create(output_desc, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(half_test::read<float>(output), expected);
}

TEST(GgmlCudaTensor, MatureKdaPublishesScoresAndStateIndependently) {
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();
  const std::array descriptions{dense(DataType::kFloat32, {1, 1, 2, 32}),
                                dense(DataType::kFloat32, {1, 1, 2, 32}),
                                dense(DataType::kFloat32, {1, 1, 2, 32}),
                                dense(DataType::kFloat32, {1, 1, 2, 32}),
                                dense(DataType::kFloat32, {1, 1, 2, 1}),
                                dense(DataType::kFloat32, {1, 2, 32, 32})};
  std::vector<Buffer> storage;
  std::vector<ConstTensorView> inputs;
  storage.reserve(descriptions.size());
  inputs.reserve(descriptions.size());
  std::vector<float> expected_state(*descriptions[5].element_count());
  for (std::size_t i = 0; i < expected_state.size(); ++i)
    expected_state[i] = static_cast<float>(i % 31) / 128.F;
  for (std::size_t i = 0; i < descriptions.size(); ++i) {
    std::vector<float> values(*descriptions[i].element_count(), 0.F);
    if (i == 5)
      values = expected_state;
    storage.push_back(upload<float>(resource, values));
    inputs.push_back(
        ConstTensorView::create(descriptions[i], storage.back().view())
            .value());
  }
  const std::array output_desc{descriptions[2], descriptions[5]};
  std::array output_storage{output_buffer(resource, output_desc[0]),
                            output_buffer(resource, output_desc[1])};
  const std::array outputs{
      TensorView::create(output_desc[0], output_storage[0].view()).value(),
      TensorView::create(output_desc[1], output_storage[1].view()).value()};
  auto prepared = prepare(ctx, Operation(GatedDeltaNet{}), inputs, outputs);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = ctx.create_queue().value();
  auto completion = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(half_test::read<float>(output_storage[0]),
            std::vector<float>(*output_desc[0].element_count(), 0.F));
  EXPECT_EQ(half_test::read<float>(output_storage[1]), expected_state);
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
  EXPECT_FALSE(
      prepare(ctx, Softmax{.kind = SoftmaxKind::kLogSoftmax}, description));
}

TEST(GgmlCudaTensor, MatureLinearRopeAndHadamardUseOrdinaryEntry) {
  using F16 = hyle::simd::float16_t;
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();
  {
    const auto left_desc = dense(DataType::kFloat16, {2, 2, 3});
    const auto right_desc = dense(DataType::kFloat16, {1, 3, 2});
    const auto output_desc = dense(DataType::kFloat16, {2, 2, 2});
    std::array<F16, 12> left_values{};
    std::array<F16, 6> right_values{};
    for (std::size_t i = 0; i < left_values.size(); ++i)
      left_values[i] = half_test::encode<F16>(static_cast<double>(i % 6 + 1));
    for (std::size_t i = 0; i < right_values.size(); ++i)
      right_values[i] = half_test::encode<F16>(static_cast<double>(i + 1));
    auto left = upload<F16>(resource, left_values);
    auto right = upload<F16>(resource, right_values);
    auto output = output_buffer(resource, output_desc);
    const std::array inputs{
        ConstTensorView::create(left_desc, left.view()).value(),
        ConstTensorView::create(right_desc, right.view()).value()};
    ASSERT_TRUE(execute(ctx, MatMul{}, inputs,
                        TensorView::create(output_desc, output.view()).value())
                    .ok());
    const auto actual = half_test::read<F16>(output);
    const std::array expected{22.0, 28.0, 49.0, 64.0, 22.0, 28.0, 49.0, 64.0};
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
  {
    const auto description = dense(DataType::kFloat32, {1, 4});
    auto source = upload<float>(resource, std::array<float, 4>{1, 2, 3, 4});
    auto output = output_buffer(resource, description);
    const auto input =
        ConstTensorView::create(description, source.view()).value();
    for (const bool inverse : {false, true}) {
      ASSERT_TRUE(
          execute(ctx, Hadamard{.inverse = inverse}, std::span(&input, 1),
                  TensorView::create(description, output.view()).value())
              .ok());
      const auto actual = half_test::read<float>(output);
      const std::array expected{5.0F, -1.0F, -2.0F, 0.0F};
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 2.e-6F);
    }
  }
  {
    const auto data_desc = dense(DataType::kFloat16, {1, 2, 1, 4});
    const auto position_desc = dense(DataType::kInt32, {2});
    const auto frequency_desc = dense(DataType::kFloat32, {2});
    std::array<F16, 8> values{};
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = half_test::encode<F16>(static_cast<double>(i + 1));
    auto data = upload<F16>(resource, values);
    auto positions =
        upload<std::int32_t>(resource, std::array<std::int32_t, 2>{0, 2});
    auto frequencies =
        upload<float>(resource, std::array<float, 2>{0.5F, 0.25F});
    auto output = output_buffer(resource, data_desc);
    const std::array inputs{
        ConstTensorView::create(data_desc, data.view()).value(),
        ConstTensorView::create(position_desc, positions.view()).value(),
        ConstTensorView::create(frequency_desc, frequencies.view()).value()};
    ASSERT_TRUE(execute(ctx,
                        RoPE{.rotary_dimensions = 4,
                             .sequence_axis = 1,
                             .layout = RopeLayout::kInterleaved},
                        inputs,
                        TensorView::create(data_desc, output.view()).value())
                    .ok());
    const auto actual = half_test::read<F16>(output);
    const std::array expected{1.0,
                              2.0,
                              3.0,
                              4.0,
                              5.0 * std::cos(1.0) - 6.0 * std::sin(1.0),
                              5.0 * std::sin(1.0) + 6.0 * std::cos(1.0),
                              7.0 * std::cos(0.5) - 8.0 * std::sin(0.5),
                              7.0 * std::sin(0.5) + 8.0 * std::cos(0.5)};
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
}

TEST(GgmlCudaTensor, SoftmaxUsesLogicalAxesMasksSpecialValuesAndExactAlias) {
  auto ctx = context();
  softmax_batch<float>(ctx);
  softmax_batch<hyle::simd::float16_t>(ctx);
  softmax_batch<hyle::simd::bfloat16_t>(ctx);
}

TEST(GgmlCudaTensor, SoftmaxRejectsPartialOverlapAndMaskAlias) {
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto description = dense(DataType::kFloat32, {256});
  auto storage = Buffer::create(memory.ref(), 1280, 256).value();
  const auto first = storage.view().subview(0, 1024).value();
  const auto second = storage.view().subview(256, 1024).value();
  const auto input = ConstTensorView::create(description, first).value();
  EXPECT_FALSE(execute(ctx, Softmax{}, std::span(&input, 1),
                       TensorView::create(description, second).value())
                   .ok());
  auto other = output_buffer(memory.ref(), description);
  const std::array inputs{
      ConstTensorView::create(description, other.view()).value(), input};
  EXPECT_FALSE(execute(ctx, Softmax{.mask = MaskKind::kAdditive}, inputs,
                       TensorView::create(description, first).value())
                   .ok());
}
} // namespace
