#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"
#include "softmax_test_support.h"
#include "tensor_test_support.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace {
using namespace ggml_cuda_tensor_test;

Context context() {
  return Context::create(cuda::device(0).value(), ggml_cuda::provider())
      .value();
}

template <class T> void swiglu(Context &ctx) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t width = 7;
  const auto description = dense(half_test::data_type<T>(), {rows, width});
  const auto packed_description =
      dense(half_test::data_type<T>(), {rows, width * 2});
  std::vector<T> gate(rows * width), value(rows * width),
      packed(rows * width * 2);
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto gate_buffer = upload<T>(memory.ref(), gate);
  auto value_buffer = upload<T>(memory.ref(), value);
  auto packed_buffer = upload<T>(memory.ref(), packed);
  auto separate_output = output_buffer(memory.ref(), description);
  auto packed_output = output_buffer(memory.ref(), description);
  const GatedActivation separate{.activation = GatedActivationKind::kSilu,
                                 .inputs = GatedInputKind::kSeparate};
  const GatedActivation packed_halves{.activation = GatedActivationKind::kSilu,
                                      .inputs = GatedInputKind::kPackedHalves};
  const auto gate_view =
      ConstTensorView::create(description, gate_buffer.view()).value();
  const auto value_view =
      ConstTensorView::create(description, value_buffer.view()).value();
  const auto packed_view =
      ConstTensorView::create(packed_description, packed_buffer.view()).value();
  const auto separate_result =
      TensorView::create(description, separate_output.view()).value();
  const auto packed_result =
      TensorView::create(description, packed_output.view()).value();
  const std::array separate_inputs{gate_view, value_view};
  auto separate_program = prepare(ctx, Operation(separate), separate_inputs,
                                  std::span(&separate_result, 1));
  ASSERT_TRUE(separate_program) << separate_program.error().message();
  EXPECT_EQ(separate_program->provider().id, ggml_cuda::kProviderId);
  auto packed_program =
      prepare(ctx, Operation(packed_halves), std::span(&packed_view, 1),
              std::span(&packed_result, 1));
  ASSERT_TRUE(packed_program) << packed_program.error().message();
  EXPECT_EQ(packed_program->provider().id, ggml_cuda::kProviderId);
  auto separate_invocation = separate_program->create_invocation().value();
  auto packed_invocation = packed_program->create_invocation().value();
  auto queue = ctx.create_queue().value();
  for (int seed : {3, 11}) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < width; ++column) {
        const auto index = row * width + column;
        gate[index] = half_test::encode<T>(
            static_cast<double>(static_cast<int>((index * 7 + seed) % 31) -
                                15) /
            8.0);
        value[index] = half_test::encode<T>(
            static_cast<double>(static_cast<int>((index * 5 + seed) % 23) -
                                11) /
            16.0);
        packed[row * width * 2 + column] = gate[index];
        packed[row * width * 2 + width + column] = value[index];
      }
    }
    ASSERT_TRUE(gate_buffer.view().write(std::as_bytes(std::span(gate))).ok());
    ASSERT_TRUE(
        value_buffer.view().write(std::as_bytes(std::span(value))).ok());
    ASSERT_TRUE(
        packed_buffer.view().write(std::as_bytes(std::span(packed))).ok());
    auto separate_done = separate_invocation.submit(
        queue, separate_inputs, std::span(&separate_result, 1));
    ASSERT_TRUE(separate_done) << separate_done.error().message();
    ASSERT_TRUE(separate_done->wait().ok());
    auto packed_done = packed_invocation.submit(
        queue, std::span(&packed_view, 1), std::span(&packed_result, 1));
    ASSERT_TRUE(packed_done) << packed_done.error().message();
    ASSERT_TRUE(packed_done->wait().ok());
    const auto separate_actual = half_test::read<T>(separate_output);
    const auto packed_actual = half_test::read<T>(packed_output);
    for (std::size_t i = 0; i < gate.size(); ++i) {
      const auto x = half_test::decode(gate[i]);
      const auto expected =
          x / (1.0 + std::exp(-x)) * half_test::decode(value[i]);
      half_test::expect_near(separate_actual[i], expected);
      half_test::expect_near(packed_actual[i], expected);
    }
  }
  auto rejected = prepare(
      ctx,
      Operation(GatedActivation{.activation = GatedActivationKind::kGelu,
                                .inputs = GatedInputKind::kSeparate}),
      separate_inputs, std::span(&separate_result, 1));
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
}

template <class Value>
void rope_type_boundary(Context &ctx, RopeLayout layout) {
  const auto data_desc = dense(half_test::data_type<Value>(), {1, 2, 1, 8});
  const auto position_desc = dense(DataType::kInt32, {2});
  const auto frequency_desc = dense(DataType::kFloat32, {2});
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();
  std::array<Value, 16> values{};
  auto data = upload<Value>(resource, values);
  auto positions =
      upload<std::int32_t>(resource, std::array<std::int32_t, 2>{0, 2});
  auto frequencies = upload<float>(resource, std::array<float, 2>{0.5F, 0.25F});
  auto output = output_buffer(resource, data_desc);
  const std::array inputs{
      ConstTensorView::create(data_desc, data.view()).value(),
      ConstTensorView::create(position_desc, positions.view()).value(),
      ConstTensorView::create(frequency_desc, frequencies.view()).value()};
  const auto output_view = TensorView::create(data_desc, output.view()).value();
  auto prepared = prepare(
      ctx, RoPE{.rotary_dimensions = 4, .sequence_axis = 1, .layout = layout},
      data_desc, position_desc, frequency_desc, resource);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = ctx.create_queue().value();
  for (const int seed : {0, 7}) {
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = half_test::encode<Value>(static_cast<double>(i + seed + 1));
    ASSERT_TRUE(data.view().write(std::as_bytes(std::span(values))).ok());
    auto completion =
        invocation.submit(queue, inputs, std::span(&output_view, 1));
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());

    std::array<double, 16> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = half_test::decode(values[i]);
    if (layout == RopeLayout::kInterleaved) {
      expected[8] = half_test::decode(values[8]) * std::cos(1.0) -
                    half_test::decode(values[9]) * std::sin(1.0);
      expected[9] = half_test::decode(values[8]) * std::sin(1.0) +
                    half_test::decode(values[9]) * std::cos(1.0);
      expected[10] = half_test::decode(values[10]) * std::cos(0.5) -
                     half_test::decode(values[11]) * std::sin(0.5);
      expected[11] = half_test::decode(values[10]) * std::sin(0.5) +
                     half_test::decode(values[11]) * std::cos(0.5);
    } else {
      expected[8] = half_test::decode(values[8]) * std::cos(1.0) -
                    half_test::decode(values[10]) * std::sin(1.0);
      expected[9] = half_test::decode(values[9]) * std::cos(0.5) -
                    half_test::decode(values[11]) * std::sin(0.5);
      expected[10] = half_test::decode(values[8]) * std::sin(1.0) +
                     half_test::decode(values[10]) * std::cos(1.0);
      expected[11] = half_test::decode(values[9]) * std::sin(0.5) +
                     half_test::decode(values[11]) * std::cos(0.5);
    }
    const auto actual = half_test::read<Value>(output);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
}

TEST(GgmlCudaTensor, FloatingNormAffineAndUnaryConversions) {
  auto ctx = context();
  norms<float>(ctx);
  norms<hyle::simd::float16_t>(ctx);
  norms<hyle::simd::bfloat16_t>(ctx);
  unary_and_cast(ctx);
}

TEST(GgmlCudaTensor, MatureSwiGluCoversStorageTypesAndInputLayouts) {
  auto ctx = context();
  swiglu<float>(ctx);
  swiglu<hyle::simd::float16_t>(ctx);
  swiglu<hyle::simd::bfloat16_t>(ctx);
}

TEST(GgmlCudaTensor, MatureRopeCoversQualifiedTypesAndLayouts) {
  auto ctx = context();
  for (const auto layout : {RopeLayout::kInterleaved, RopeLayout::kHalfSplit}) {
    rope_type_boundary<float>(ctx, layout);
    rope_type_boundary<hyle::simd::float16_t>(ctx, layout);
    rope_type_boundary<hyle::simd::bfloat16_t>(ctx, layout);
  }
  const auto data = dense(DataType::kFloat64, {1, 2, 1, 8});
  const auto positions = dense(DataType::kInt32, {2});
  const auto frequencies = dense(DataType::kFloat32, {2});
  auto rejected = prepare(ctx,
                          RoPE{.rotary_dimensions = 4,
                               .sequence_axis = 1,
                               .layout = RopeLayout::kInterleaved},
                          data, positions, frequencies);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
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

TEST(GgmlCudaTensor, D5Qwen38GroupedHeadsUseMatureGatedDeltaNet) {
  const d5_qwen38_gdn_fixture::Fixture fixture;
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  std::vector<Buffer> storage;
  std::vector<ConstTensorView> inputs;
  storage.reserve(fixture.inputs.size());
  inputs.reserve(fixture.inputs.size());
  for (std::size_t i = 0; i < fixture.inputs.size(); ++i) {
    storage.push_back(upload<float>(resource, fixture.data[i]));
    inputs.push_back(
        ConstTensorView::create(fixture.inputs[i], storage.back().view())
            .value());
  }
  std::array output_storage{output_buffer(resource, fixture.outputs[0]),
                            output_buffer(resource, fixture.outputs[1])};
  const std::array outputs{
      TensorView::create(fixture.outputs[0], output_storage[0].view()).value(),
      TensorView::create(fixture.outputs[1], output_storage[1].view()).value()};
  const auto workload = d5_qwen38_gdn_fixture::workload(WorkloadPhase::kDecode);
  auto prepared = prepare(ctx, Operation(GatedDeltaNet{}), inputs, outputs,
                          resource, workload);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->workload(), workload);
  const auto expected = fixture.oracle();
  auto queue = ctx.create_queue().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto invocation = prepared->create_invocation().value();
    auto completion = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      const auto actual = half_test::read<float>(output_storage[port]);
      ASSERT_EQ(actual.size(), expected[port].size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[port][i], 4.e-6) << port << ':' << i;
    }
  }
}

TEST(GgmlCudaTensor, MatureMoeAndHyperConnectionBatchUsesOrdinaryEntry) {
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    std::vector<Buffer> storage;
    std::vector<ConstTensorView> inputs;
    storage.reserve(test.inputs.size());
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i) {
      storage.push_back(upload<std::byte>(resource, test.payloads[i]));
      inputs.push_back(
          ConstTensorView::create(test.inputs[i], storage.back().view())
              .value());
    }
    auto output = output_buffer(resource, test.output);
    const auto status =
        execute(ctx, test.operation, inputs,
                TensorView::create(test.output, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(output);
    ASSERT_EQ(actual.size(), test.expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCudaTensor, MatureLightningIndexerUsesOrdinaryEntryAndReplays) {
  const sparse_fragment_test::Fixture test;
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  std::vector<Buffer> storage;
  std::vector<ConstTensorView> inputs;
  for (std::size_t i = 0; i < test.inputs.size(); ++i) {
    storage.push_back(upload<std::byte>(memory.ref(), test.payloads[i]));
    inputs.push_back(
        ConstTensorView::create(test.inputs[i], storage.back().view()).value());
  }
  auto output = output_buffer(memory.ref(), test.output);
  auto view = TensorView::create(test.output, output.view()).value();
  auto prepared = prepare(ctx, test.operation, inputs, std::span(&view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto invocation = prepared->create_invocation().value();
  auto queue = ctx.create_queue().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue, inputs, std::span(&view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
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
       {UnaryKind::kSilu, UnaryKind::kSoftplus})
    EXPECT_FALSE(prepare(ctx, Unary{kind}, description));
  const auto odd = TensorDesc::strided_bytes(DataType::kFloat32,
                                             std::array<std::int64_t, 1>{7},
                                             std::array<std::int64_t, 1>{5})
                       .value();
  EXPECT_FALSE(prepare(ctx, Copy{}, odd));
  EXPECT_FALSE(
      prepare(ctx, Softmax{.kind = SoftmaxKind::kLogSoftmax}, description));
}

TEST(GgmlCudaTensor, CpuFormatPortDoesNotQualifyGpuMatrices) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto activation = dense(DataType::kFloat32, {2, 256});
  const auto grouped_activation = dense(DataType::kFloat32, {1, 2, 256});
  const auto ids = dense(DataType::kInt32, {1, 2});
  for (const auto type : {GgufQuantizedType::kQ4_0, GgufQuantizedType::kQ4_1,
       GgufQuantizedType::kQ2K, GgufQuantizedType::kIq2Xxs,
       GgufQuantizedType::kIq2Xs, GgufQuantizedType::kIq2S,
       GgufQuantizedType::kIq1S, GgufQuantizedType::kIq1M,
       GgufQuantizedType::kIq1Xs, GgufQuantizedType::kIq1Xxs,
       GgufQuantizedType::kIq1Xxxs}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto weight = gguf_weight_description(type, DataType::kFloat32, 256, 3).value();
    auto rejected = prepare(ctx, MatMul{}, activation, weight, memory.ref());
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
    const auto grouped = gguf_grouped_weight_description(type, DataType::kFloat32, 2, 3, 256).value();
    const std::array inputs{grouped, grouped_activation, ids};
    const auto op = model_ops::grouped_matmul();
    const auto output = infer_single_output(op, inputs).value();
    auto grouped_rejected = prepare(ctx, op, inputs, std::span(&output, 1), memory.ref());
    ASSERT_FALSE(grouped_rejected);
    EXPECT_TRUE(grouped_rejected.error().IsUnimplemented());
  }
}

TEST(GgmlCudaTensor, IntegerConcatPreservesIndexBitsAcrossReplays) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto left = dense(DataType::kInt32, {2, 4});
  const auto right = dense(DataType::kInt32, {2, 3});
  const auto joined = dense(DataType::kInt32, {2, 7});
  std::vector<std::int32_t> a{0, -1, 16777217, INT32_MAX, INT32_MIN, -16777217, 5, 7};
  std::vector<std::int32_t> b{-1, 2053, 2054, INT32_MIN, 16777219, INT32_MAX};
  auto aa = upload<std::int32_t>(memory.ref(), a);
  auto bb = upload<std::int32_t>(memory.ref(), b);
  auto output = output_buffer(memory.ref(), joined);
  const std::array inputs{ConstTensorView::create(left, aa.view()).value(),
                          ConstTensorView::create(right, bb.view()).value()};
  auto view = TensorView::create(joined, output.view()).value();
  auto program = prepare(ctx, Operation(Concat{.axis = 1}), inputs, std::span(&view, 1));
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->provider().id, ggml_cuda::kProviderId);
  auto created = program->create_invocation();
  ASSERT_TRUE(created) << created.error().message();
  auto invocation = std::move(*created);
  auto queue = ctx.create_queue().value();
  for (int replay = 0; replay < 2; ++replay) {
    ASSERT_TRUE(aa.view().write(std::as_bytes(std::span(a))).ok());
    ASSERT_TRUE(bb.view().write(std::as_bytes(std::span(b))).ok());
    auto done = invocation.submit(queue, inputs, std::span(&view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<std::int32_t> expected;
    for (int row = 0; row < 2; ++row) {
      expected.insert(expected.end(), a.begin() + row * 4, a.begin() + (row + 1) * 4);
      expected.insert(expected.end(), b.begin() + row * 3, b.begin() + (row + 1) * 3);
    }
    EXPECT_EQ(half_test::read<std::int32_t>(output), expected);
    std::reverse(a.begin(), a.end());
    std::reverse(b.begin(), b.end());
  }
  auto single_output = output_buffer(memory.ref(), left);
  auto single_status = execute(ctx, Concat{.axis = 1}, std::span(inputs).first(1),
      TensorView::create(left, single_output.view()).value());
  ASSERT_TRUE(single_status.ok()) << single_status.message();
  auto single_expected = a;
  std::reverse(single_expected.begin(), single_expected.end());
  EXPECT_EQ(half_test::read<std::int32_t>(single_output), single_expected);
  const auto strided = TensorDesc::strided_bytes(DataType::kInt32,
      std::array<std::int64_t, 2>{2, 4}, std::array<std::int64_t, 2>{32, 8}).value();
  const std::array invalid_inputs{strided, right};
  const std::array wrong_outputs{dense(DataType::kInt32, {2, 8})};
  EXPECT_FALSE(prepare(ctx, Operation(Concat{.axis = 1}), invalid_inputs, std::array{joined}, memory.ref()));
  EXPECT_FALSE(prepare(ctx, Operation(Concat{.axis = 1}), std::array{left, right}, wrong_outputs, memory.ref()));
  EXPECT_FALSE(prepare(ctx, Unary{UnaryKind::kExp}, left));
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

TEST(GgmlCudaTensor, NativeQ4KMatMulImportsGgufBytesAndReplays) {
  constexpr std::int64_t reduction = 256;
  constexpr std::int64_t columns = 2;
  const auto left_desc = dense(DataType::kFloat32, {1, reduction});
  const auto right_desc = gguf_weight_description(
      GgufQuantizedType::kQ4K, DataType::kFloat32, reduction, columns);
  ASSERT_TRUE(right_desc) << right_desc.error().message();
  const auto output_desc = dense(DataType::kFloat32, {1, columns});
  std::vector<float> left_values(static_cast<std::size_t>(reduction), 0.25F);
  std::vector<float> physical_weights(
      static_cast<std::size_t>(reduction * columns));
  std::fill_n(physical_weights.begin(), reduction, 1.0F);
  std::fill_n(physical_weights.begin() + reduction, reduction, -2.0F);
  std::vector<std::byte> packed(*right_desc->required_bytes());
  ASSERT_EQ(ggml_quantize_chunk(GGML_TYPE_Q4_K, physical_weights.data(),
                                packed.data(), 0, columns, reduction, nullptr),
            packed.size());

  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto left = upload<float>(memory.ref(), left_values);
  auto right = upload<std::byte>(memory.ref(), packed);
  auto output = output_buffer(memory.ref(), output_desc);
  const std::array inputs{
      ConstTensorView::create(left_desc, left.view()).value(),
      ConstTensorView::create(*right_desc, right.view()).value()};
  auto output_view = TensorView::create(output_desc, output.view()).value();
  auto prepared =
      prepare(ctx, Operation(MatMul{}), inputs, std::span(&output_view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = ctx.create_queue().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    const auto scale = repeat == 0 ? 1.0F : 2.0F;
    ASSERT_EQ(actual.size(), 2U);
    EXPECT_NEAR(actual[0], 64.0F * scale, 1.0F);
    EXPECT_NEAR(actual[1], -128.0F * scale, 1.0F);
    if (repeat == 0) {
      std::fill(left_values.begin(), left_values.end(), 0.5F);
      ASSERT_TRUE(
          left.view().write(std::as_bytes(std::span(left_values))).ok());
    }
  }
}

TEST(GgmlCudaTensor, NativeEmbeddingRowsUseTargetGgufFormatsAndCheckBounds) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t rows = 3;
  constexpr std::int64_t width = 256;
  constexpr std::array formats{GgufQuantizedType::kQ4K, GgufQuantizedType::kQ6K,
                               GgufQuantizedType::kQ8_0};
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();

  for (const auto format : formats) {
    SCOPED_TRACE(static_cast<int>(format));
    const auto stored =
        gguf_weight_description(format, DataType::kFloat32, width, rows);
    ASSERT_TRUE(stored) << stored.error().message();
    const auto table_desc = stored->transposed().value();
    const auto ids_desc = dense(DataType::kInt32, {2});
    const auto output_desc = dense(DataType::kFloat32, {2, width});
    const auto physical = d5_gguf_test::values(rows, width, 29);
    const auto packed = d5_gguf_test::pack(format, physical, rows, width,
                                           table_desc.required_bytes().value());
    auto table = upload<std::byte>(memory.ref(), packed.bytes);
    std::array<std::int32_t, 2> selected{2, 0};
    auto ids = upload<std::int32_t>(memory.ref(), selected);
    auto output = output_buffer(memory.ref(), output_desc);
    const std::array inputs{
        ConstTensorView::create(table_desc, table.view()).value(),
        ConstTensorView::create(ids_desc, ids.view()).value()};
    auto output_view = TensorView::create(output_desc, output.view()).value();
    auto prepared =
        prepare(ctx, Operation(Gather{}), inputs, std::span(&output_view, 1));
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    auto invocation = prepared->create_invocation().value();
    auto queue = ctx.create_queue().value();

    for (int replay = 0; replay < 2; ++replay) {
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      auto completion =
          invocation.submit(queue, inputs, std::span(&output_view, 1));
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<float>(output);
      const auto expected =
          d5_gguf_test::selected_rows(packed.decoded, width, selected);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 1.0e-6F) << i;
      selected = {1, 2};
    }

    const auto before_failure = half_test::read<float>(output);
    selected = {0, static_cast<std::int32_t>(rows)};
    ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
    auto failed = invocation.submit(queue, inputs, std::span(&output_view, 1));
    ASSERT_TRUE(failed) << failed.error().message();
    EXPECT_FALSE(failed->wait().ok());
    EXPECT_EQ(half_test::read<float>(output), before_failure);
  }
}

TEST(GgmlCudaTensor, NativeQ8GroupedExpertsUseExactExpertIds) {
  constexpr std::int64_t experts = 3;
  constexpr std::int64_t columns = 2;
  constexpr std::int64_t reduction = 256;
  const auto weights_desc = gguf_grouped_weight_description(
      GgufQuantizedType::kQ8_0, DataType::kFloat32, experts, columns,
      reduction);
  ASSERT_TRUE(weights_desc) << weights_desc.error().message();
  const auto activations_desc = dense(DataType::kFloat32, {1, 2, reduction});
  const auto ids_desc = dense(DataType::kInt32, {1, 2});
  const std::array descriptions{*weights_desc, activations_desc, ids_desc};
  const auto output_desc =
      infer_single_output(model_ops::grouped_matmul(), descriptions).value();
  constexpr std::array<float, experts * columns> row_values{1.0F, 2.0F,  3.0F,
                                                            4.0F, -1.0F, 0.5F};
  std::vector<float> physical_weights(
      static_cast<std::size_t>(experts * columns * reduction));
  for (std::size_t row = 0; row < row_values.size(); ++row)
    std::fill_n(physical_weights.begin() + row * reduction, reduction,
                row_values[row]);
  std::vector<std::byte> packed(*weights_desc->required_bytes());
  ASSERT_EQ(ggml_quantize_chunk(GGML_TYPE_Q8_0, physical_weights.data(),
                                packed.data(), 0, experts * columns, reduction,
                                nullptr),
            packed.size());
  std::vector<float> activations(static_cast<std::size_t>(2 * reduction),
                                 0.25F);
  constexpr std::array<std::int32_t, 2> ids{0, 2};

  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto weights = upload<std::byte>(memory.ref(), packed);
  auto activation_buffer = upload<float>(memory.ref(), activations);
  auto id_buffer = upload<std::int32_t>(memory.ref(), ids);
  auto output = output_buffer(memory.ref(), output_desc);
  const std::array inputs{
      ConstTensorView::create(*weights_desc, weights.view()).value(),
      ConstTensorView::create(activations_desc, activation_buffer.view())
          .value(),
      ConstTensorView::create(ids_desc, id_buffer.view()).value()};
  const auto status =
      execute(ctx, model_ops::grouped_matmul(), inputs,
              TensorView::create(output_desc, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = half_test::read<float>(output);
  ASSERT_EQ(actual.size(), 4U);
  EXPECT_NEAR(actual[0], 64.0F, 0.5F);
  EXPECT_NEAR(actual[1], 128.0F, 0.5F);
  EXPECT_NEAR(actual[2], -64.0F, 0.5F);
  EXPECT_NEAR(actual[3], 32.0F, 0.5F);
}

TEST(GgmlCudaTensor, DenseBfloatGroupedExpertsUseNativeMulMatIdAndReplay) {
  using BF16 = hyle::simd::bfloat16_t;
  constexpr std::int64_t experts = 3;
  constexpr std::int64_t columns = 2;
  constexpr std::int64_t reduction = 32;
  const auto weights_desc =
      dense(DataType::kBFloat16, {experts, columns, reduction});
  const auto activations_desc = dense(DataType::kFloat32, {1, 2, reduction});
  const auto ids_desc = dense(DataType::kInt32, {1, 2});
  const std::array descriptions{weights_desc, activations_desc, ids_desc};
  const auto output_desc =
      infer_single_output(model_ops::grouped_matmul(), descriptions).value();
  constexpr std::array<double, experts * columns> row_values{1.0, 2.0, 3.0,
                                                              4.0, -1.0, 0.5};
  std::vector<BF16> weights(experts * columns * reduction);
  for (std::size_t row = 0; row < row_values.size(); ++row)
    std::fill_n(weights.begin() + row * reduction, reduction,
                half_test::encode<BF16>(row_values[row]));
  std::vector<float> activations(2 * reduction, 0.25F);
  constexpr std::array<std::int32_t, 2> ids{0, 2};
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto weights_buffer = upload<BF16>(memory.ref(), weights);
  auto activation_buffer = upload<float>(memory.ref(), activations);
  auto id_buffer = upload<std::int32_t>(memory.ref(), ids);
  auto output = output_buffer(memory.ref(), output_desc);
  const std::array inputs{
      ConstTensorView::create(weights_desc, weights_buffer.view()).value(),
      ConstTensorView::create(activations_desc, activation_buffer.view()).value(),
      ConstTensorView::create(ids_desc, id_buffer.view()).value()};
  auto output_view = TensorView::create(output_desc, output.view()).value();
  auto prepared = prepare(ctx, model_ops::grouped_matmul(), inputs,
                          std::span(&output_view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = ctx.create_queue().value();
  for (int replay = 1; replay <= 2; ++replay) {
    std::fill(activations.begin(), activations.end(), 0.25F * replay);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(activations)))
                    .ok());
    auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    const std::array expected{8.0F * replay, 16.0F * replay,
                              -8.0F * replay, 4.0F * replay};
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1.0e-5F) << i;
  }
}

TEST(GgmlCudaTensor, BfloatRouterMatMulRetainsNativeF32Score) {
  using BF16 = hyle::simd::bfloat16_t;
  const auto activation_desc = dense(DataType::kBFloat16, {1, 2});
  const auto weight_desc = dense(DataType::kBFloat16, {2, 2});
  const std::array descriptions{activation_desc, weight_desc};
  const auto output_desc =
      infer_single_output(model_ops::bfloat_linear_f32(), descriptions).value();
  const std::array activations{half_test::encode<BF16>(1.0),
                               half_test::encode<BF16>(0.001953125)};
  const std::array weights{
      half_test::encode<BF16>(1.0), half_test::encode<BF16>(1.0),
      half_test::encode<BF16>(1.0), half_test::encode<BF16>(0.0)};
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto activation_buffer = upload<BF16>(memory.ref(), activations);
  auto weight_buffer = upload<BF16>(memory.ref(), weights);
  auto output = output_buffer(memory.ref(), output_desc);
  const std::array inputs{
      ConstTensorView::create(activation_desc, activation_buffer.view()).value(),
      ConstTensorView::create(weight_desc, weight_buffer.view()).value()};
  auto output_view = TensorView::create(output_desc, output.view()).value();
  auto prepared = prepare(ctx, model_ops::bfloat_linear_f32(), inputs,
                          std::span(&output_view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = ctx.create_queue().value();
  auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto actual = half_test::read<float>(output);
  ASSERT_EQ(actual.size(), 2U);
  EXPECT_NEAR(actual[0], 1.001953125F, 1.0e-6F);
  EXPECT_NEAR(actual[1], 1.0F, 1.0e-6F);
}

TEST(GgmlCudaTensor, MalformedNativeGgufFormatIsRejectedAtPrepare) {
  const auto format = StorageFormat::blocked(kGgufQ4K, 256, 143).value();
  constexpr std::array<std::int64_t, 2> shape{256, 2};
  constexpr std::array<std::int64_t, 2> strides{143, 143};
  const auto weights_desc =
      TensorDesc::blocked_strided(DataType::kFloat32, shape, format, 0, strides)
          .value();
  const auto activations_desc = dense(DataType::kFloat32, {1, 256});
  const auto output_desc = dense(DataType::kFloat32, {1, 2});
  auto ctx = context();
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto activations = output_buffer(memory.ref(), activations_desc);
  auto weights = output_buffer(memory.ref(), weights_desc);
  auto output = output_buffer(memory.ref(), output_desc);
  const std::array inputs{
      ConstTensorView::create(activations_desc, activations.view()).value(),
      ConstTensorView::create(weights_desc, weights.view()).value()};
  auto output_view = TensorView::create(output_desc, output.view()).value();
  EXPECT_FALSE(
      prepare(ctx, Operation(MatMul{}), inputs, std::span(&output_view, 1)));
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
