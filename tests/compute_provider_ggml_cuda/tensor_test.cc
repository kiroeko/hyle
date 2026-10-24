#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_models/gdn_snapshot_fixture.h"
#include "../compute_models/batched_rope_fixture.h"
#include "../compute_models/gather_elements_fixture.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"
#include "softmax_test_support.h"
#include "tensor_test_support.h"
#include "bfloat_elementwise_test_support.h"

#include <ggml.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace {
using namespace ggml_cuda_tensor_test;

TEST(GgmlCudaTensor, BfloatTypedElementwiseBroadcastsAndRebindsIndependentSlots) {
  using bfloat_elementwise_test::BF16;
  auto ctx = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto description = dense(DataType::kBFloat16, {3, 17});
  const auto row_description = dense(DataType::kBFloat16, {17});
  std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
  const auto operations = bfloat_elementwise_test::operations();
  for (std::size_t op = 0; op < operations.size(); ++op) {
    SCOPED_TRACE(op);
    const std::array descriptions{description, row_description};
    auto prepared = prepare(ctx, operations[op],
        std::span(descriptions).first(op < 4 ? 2 : 1),
        std::span(&description, 1), memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    std::array workspaces{ggml_cuda_test::workspace_buffer(*prepared, memory.ref()),
                          ggml_cuda_test::workspace_buffer(*prepared, memory.ref())};
    std::array invocations{
        prepared->create_invocation(workspaces[0].view()).value(),
        prepared->create_invocation(workspaces[1].view()).value()};
    Buffer old_output;
    std::vector<Buffer> old_inputs;
    std::vector<BF16> old_values;
    for (std::size_t round = 0; round < 2; ++round) {
      std::array<Buffer, 2> left, right, output;
      std::array<std::vector<BF16>, 2> a, b;
      std::array<Completion, 2> done;
      for (std::size_t slot = 0; slot < 2; ++slot) {
        a[slot].resize(51); b[slot].resize(17);
        constexpr std::array points{0.25, 0.5, 1.0, 2.0, 4.0};
        for (std::size_t i = 0; i < a[slot].size(); ++i)
          a[slot][i] = half_test::encode<BF16>(points[(i + round + slot) % points.size()]);
        for (std::size_t i = 0; i < b[slot].size(); ++i)
          b[slot][i] = half_test::encode<BF16>(points[(i + round + slot + 2) % points.size()]);
        left[slot] = upload<BF16>(memory.ref(), a[slot]);
        right[slot] = upload<BF16>(memory.ref(), b[slot]);
        output[slot] = output_buffer(memory.ref(), description);
        const std::array inputs{
            ConstTensorView::create(description, left[slot].view()).value(),
            ConstTensorView::create(row_description, right[slot].view()).value()};
        const auto target = TensorView::create(description, output[slot].view()).value();
        auto submitted = invocations[slot].submit(queues[slot],
            std::span(inputs).first(op < 4 ? 2 : 1), target);
        ASSERT_TRUE(submitted) << submitted.error().message();
        done[slot] = std::move(*submitted);
      }
      for (std::size_t slot = 0; slot < 2; ++slot) {
        ASSERT_TRUE(done[slot].wait().ok());
        const auto actual = half_test::read<BF16>(output[slot]);
        ASSERT_EQ(actual.size(), 51u);
        for (std::size_t i = 0; i < actual.size(); ++i)
          bfloat_elementwise_test::expect_bits(actual[i],
              bfloat_elementwise_test::reference(op, a[slot][i], b[slot][i % 17]), i);
      }
      if (round == 0) {
        old_values = half_test::read<BF16>(output[0]);
        old_output = std::move(output[0]);
        for (std::size_t slot = 0; slot < 2; ++slot) {
          old_inputs.push_back(std::move(left[slot]));
          old_inputs.push_back(std::move(right[slot]));
        }
      } else {
        const auto preserved = half_test::read<BF16>(old_output);
        ASSERT_EQ(preserved.size(), old_values.size());
        for (std::size_t i = 0; i < preserved.size(); ++i)
          EXPECT_EQ(preserved[i].bits, old_values[i].bits) << i;
      }
    }
  }
}

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
  auto separate_invocation_workspace =
      ggml_cuda_test::workspace_buffer(*separate_program, memory.ref());
  auto separate_invocation =
      separate_program->create_invocation(separate_invocation_workspace.view()).value();
  auto packed_invocation_workspace =
      ggml_cuda_test::workspace_buffer(*packed_program, memory.ref());
  auto packed_invocation =
      packed_program->create_invocation(packed_invocation_workspace.view()).value();
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
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
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

template<class Value> void batched_rope(Context &ctx) {
  namespace f = batched_rope_test;
  const auto data_desc = dense(half_test::data_type<Value>(), {2, 3, 2, 8});
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto queue = ctx.create_queue().value();
  for (bool per_request : {false, true})
    for (bool per_token_frequency : {false, true})
      for (const auto layout : {RopeLayout::kInterleaved, RopeLayout::kHalfSplit}) {
        const auto position_desc = per_request ? dense(DataType::kInt32, {2, 3})
                                               : dense(DataType::kInt32, {3});
        const auto frequency_desc = per_token_frequency ? dense(DataType::kFloat32, {3, 2})
                                                        : dense(DataType::kFloat32, {2});
        auto positions = f::kPositions;
        auto p = upload<std::int32_t>(memory.ref(), std::span(positions).first(per_request ? 6 : 3));
        auto frequencies = upload<float>(memory.ref(), std::span(f::kFrequencies).first(per_token_frequency ? 6 : 2));
        auto output = output_buffer(memory.ref(), data_desc);
        auto prepared = prepare(ctx,
            RoPE{.rotary_dimensions=4, .sequence_axis=-3, .layout=layout},
            data_desc, position_desc, frequency_desc, memory.ref());
        ASSERT_TRUE(prepared) << prepared.error().message();
        auto invocation_workspace =
            ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
        auto invocation =
            prepared->create_invocation(invocation_workspace.view()).value();
        for (int seed : {0, 11}) {
          const auto source = f::data(seed);
          std::array<Value, f::kElements> encoded{};
          for (std::size_t i=0; i<encoded.size(); ++i)
            encoded[i] = half_test::encode<Value>(source[i]);
          auto data = upload<Value>(memory.ref(), encoded);
          positions[0] = seed;
          ASSERT_TRUE(p.view().write(std::as_bytes(std::span(positions).first(per_request ? 6 : 3))).ok());
          const std::array inputs{
              ConstTensorView::create(data_desc, data.view()).value(),
              ConstTensorView::create(position_desc, p.view()).value(),
              ConstTensorView::create(frequency_desc, frequencies.view()).value()};
          const auto result = TensorView::create(data_desc, output.view()).value();
          auto completion = invocation.submit(queue, inputs, std::span(&result, 1));
          ASSERT_TRUE(completion) << completion.error().message();
          ASSERT_TRUE(completion->wait().ok());
          const auto actual = half_test::read<Value>(output);
          const auto expected = f::reference(source, positions, per_request, per_token_frequency, layout);
          for (std::size_t i=0; i<actual.size(); ++i)
            half_test::expect_near(actual[i], expected[i]);
        }
      }
}

TEST(GgmlCudaTensor, BatchedRopePositionsAndPerTokenFrequenciesReplay) {
  auto ctx = context();
  batched_rope<float>(ctx);
  batched_rope<hyle::simd::float16_t>(ctx);
  batched_rope<hyle::simd::bfloat16_t>(ctx);
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
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
  auto queue = ctx.create_queue().value();
  auto completion = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(half_test::read<float>(output_storage[0]),
            std::vector<float>(*output_desc[0].element_count(), 0.F));
  EXPECT_EQ(half_test::read<float>(output_storage[1]), expected_state);
}

TEST(GgmlCudaTensor, GatedDeltaNetSnapshotsKeepChronologicalBatchesOnRebind) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = ctx.create_queue().value();
  for (const std::uint32_t count : {1U, 3U, 5U}) {
    const gdn_snapshot_fixture::Fixture first;
    const auto descriptions = first.outputs(count);
    std::vector<Buffer> storage;
    std::vector<ConstTensorView> inputs;
    for (std::size_t i = 0; i < first.inputs.size(); ++i) {
      storage.push_back(upload<float>(resource, first.data[i]));
      inputs.push_back(
          ConstTensorView::create(first.inputs[i], storage.back().view()).value());
    }
    std::array output_storage{output_buffer(resource, descriptions[0]),
                              output_buffer(resource, descriptions[1])};
    const std::array outputs{
        TensorView::create(descriptions[0], output_storage[0].view()).value(),
        TensorView::create(descriptions[1], output_storage[1].view()).value()};
    auto prepared = prepare(
        ctx, Operation(GatedDeltaNet{.snapshot_count = count}), inputs, outputs);
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto invocation_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto invocation =
        prepared->create_invocation(invocation_workspace.view()).value();
    for (const int seed : {0, 7}) {
      SCOPED_TRACE(::testing::Message() << "snapshots=" << count << " seed=" << seed);
      const gdn_snapshot_fixture::Fixture current(seed);
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        storage[i] = upload<float>(resource, current.data[i]);
        inputs[i] = ConstTensorView::create(current.inputs[i], storage[i].view()).value();
      }
      auto completion = invocation.submit(queue, inputs, outputs);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto expected = current.oracle(count);
      for (std::size_t port = 0; port < outputs.size(); ++port) {
        const auto actual = half_test::read<float>(output_storage[port]);
        ASSERT_EQ(actual.size(), expected[port].size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_NEAR(actual[i], expected[port][i], 3.0e-6) << port << ':' << i;
      }
      EXPECT_EQ(half_test::read<float>(storage[5]), current.data[5]);
    }
  }
}

TEST(GgmlCudaTensor, RecurrentTokenBlockBroadcastsParametersAndPublishesSnapshots) {
  const std::array<std::array<std::int64_t, 4>, 2> shapes{{
      {1, 16, 16, 128}, {2, 2, 6, 32}}};
  for (const auto &shape : shapes) {
    SCOPED_TRACE(shape[0]);
    SCOPED_TRACE(shape[2]);
    SCOPED_TRACE(shape[3]);
    gdn_snapshot_fixture::Fixture fixture(3, shape[0], 3, shape[1], shape[2], shape[3]);
    const auto parameter = dense(DataType::kFloat32, {1, 1, shape[2], 1});
    std::array descriptions{fixture.inputs[0], fixture.inputs[1], fixture.inputs[2],
                             fixture.inputs[3], fixture.inputs[4], parameter,
                             parameter, fixture.inputs[5]};
    std::vector<float> raw_a(fixture.data[3].size()), raw_b(raw_a.size());
    std::vector<float> coefficient(static_cast<std::size_t>(shape[2])), bias(coefficient.size());
    for (std::size_t head = 0; head < coefficient.size(); ++head) {
      coefficient[head] = -0.025F * float(head + 1);
      bias[head] = 0.01F * float(head);
    }
    for (std::size_t i = 0; i < raw_a.size(); ++i) {
      raw_a[i] = float(int(i % 11) - 5) / 16.F;
      raw_b[i] = float(int(i % 7) - 3) / 8.F;
      fixture.data[3][i] = float(
          std::log1p(std::exp(double(raw_a[i]) + bias[i % bias.size()])) *
          coefficient[i % coefficient.size()]);
      fixture.data[4][i] = float(1.0 / (1.0 + std::exp(-double(raw_b[i]))));
    }
    const std::array<std::span<const float>, 8> values{
        fixture.data[0], fixture.data[1], fixture.data[2], raw_a, raw_b,
        coefficient, bias, fixture.data[5]};
    auto ctx = context();
    auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
    const auto resource = memory.ref();
    std::vector<Buffer> storage;
    std::vector<ConstTensorView> inputs;
    for (std::size_t i = 0; i < values.size(); ++i) {
      storage.push_back(upload<float>(resource, values[i]));
      inputs.push_back(
          ConstTensorView::create(descriptions[i], storage.back().view()).value());
    }
    const auto output_desc = fixture.outputs(3);
    std::array output_storage{output_buffer(resource, output_desc[0]),
                              output_buffer(resource, output_desc[1])};
    const std::array outputs{
        TensorView::create(output_desc[0], output_storage[0].view()).value(),
        TensorView::create(output_desc[1], output_storage[1].view()).value()};
    auto prepared = prepare(ctx, model_ops::recurrent_gated_delta_net(3), inputs,
                             outputs);
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto invocation_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto invocation =
        prepared->create_invocation(invocation_workspace.view()).value();
    auto queue = ctx.create_queue().value();
    auto completion = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    const auto expected = fixture.oracle(3);
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      const auto actual = half_test::read<float>(output_storage[port]);
      ASSERT_EQ(actual.size(), expected[port].size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[port][i], 3.0e-6) << port << ':' << i;
    }
  }
}

TEST(GgmlCudaTensor, RecurrentSchemaDoesNotGrantUnavailableNativeDepthOrBatch) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto f32 = DataType::kFloat32;
  // Descriptor preparation must decline before native construction/launch.
  // No tensor payload or large batch backing is allocated by this test.
  for (const auto &shape : std::array<std::array<std::int64_t, 2>, 2>{{
           {1, 24}, {65536, 16}}}) {
    const auto b = shape[0], d = shape[1];
    const auto raw = dense(f32, {b, 1, 1, 1});
    const auto parameter = dense(f32, {1, 1, 1, 1});
    const std::array inputs{dense(f32, {b, 1, 1, d}), dense(f32, {b, 1, 1, d}),
        dense(f32, {b, 1, 1, d}), raw, raw, parameter, parameter,
        dense(f32, {b, 1, d, d})};
    const std::array ordinary{inputs[0], inputs[1], inputs[2],
                               inputs[3], inputs[4], inputs[7]};
    for (const bool composite : {false, true}) {
      const auto operation = composite ? model_ops::recurrent_gated_delta_net()
                                       : Operation(GatedDeltaNet{});
      const auto selected = composite ? std::span<const TensorDesc>(inputs)
                                      : std::span<const TensorDesc>(ordinary);
      std::array<TensorDesc, 2> outputs;
      ASSERT_TRUE(infer_output_descriptions(operation, selected, outputs).ok());
      const auto prepared = prepare(ctx, operation, selected, outputs, memory.ref());
      ASSERT_FALSE(prepared);
      EXPECT_TRUE(prepared.error().IsUnimplemented()) << prepared.error().message();
    }
  }
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
    auto invocation_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto invocation =
        prepared->create_invocation(invocation_workspace.view()).value();
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
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
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

template <class Value> void balanced_concat(Context &ctx) {
  constexpr bool integer = std::is_same_v<Value, std::int32_t>;
  const auto type = integer ? DataType::kInt32 : half_test::data_type<Value>();
  using Bits = std::conditional_t<sizeof(Value) == 2, std::uint16_t,
                                  std::uint32_t>;
  auto memory =
      cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  for (std::size_t selected = 0; selected < 3; ++selected) {
    for (const std::size_t count : {1u, 7u, 8u, 9u}) {
      SCOPED_TRACE(::testing::Message() << selected << '/' << count);
      std::vector<mem::Buffer> storage;
      std::vector<ConstTensorView> inputs;
      std::vector<TensorDesc> descriptions;
      std::vector<std::vector<Value>> logical;
      for (std::size_t port = 0; port < count; ++port) {
        std::array<std::int64_t, 3> shape{2, 3, 7};
        shape[selected] = static_cast<std::int64_t>(port % 3 + 1);
        // Integer Concat's established domain is dense; floating inputs also
        // exercise padded rows and a two-element final-axis stride.
        const std::int64_t last = integer ? 1 : 2;
        const auto row = shape[2] * last + (integer ? 0 : 1);
        const auto plane = shape[1] * row + (integer ? 0 : 3);
        const std::array strides{plane * sizeof(Value), row * sizeof(Value),
                                 last * sizeof(Value)};
        auto description = TensorDesc::strided_bytes(type, shape,
            std::array<std::int64_t, 3>{static_cast<std::int64_t>(strides[0]),
              static_cast<std::int64_t>(strides[1]),
              static_cast<std::int64_t>(strides[2])}).value();
        std::vector<Value> physical(
            description.required_bytes().value() / sizeof(Value));
        std::vector<Value> values;
        for (std::int64_t a = 0; a < shape[0]; ++a)
          for (std::int64_t b = 0; b < shape[1]; ++b)
            for (std::int64_t c = 0; c < shape[2]; ++c) {
              const auto index = values.size();
              // Raw payloads catch accidental floating conversions, including
              // NaN payloads and integer values beyond exact F32 range.
              const auto bits = static_cast<Bits>(
                  (sizeof(Value) == 2 ? 0x7e00u : 0x7fc00000u) |
                  ((port * 97 + index + 1) & 0x1ffu));
              const auto value = std::bit_cast<Value>(bits);
              values.push_back(value);
              physical[static_cast<std::size_t>(a * plane + b * row + c * last)] = value;
            }
        storage.push_back(upload<Value>(memory.ref(), physical));
        inputs.push_back(ConstTensorView::create(description,
                                                 storage.back().view()).value());
        descriptions.push_back(description);
        logical.push_back(std::move(values));
      }
      const auto operation = Concat{static_cast<std::int32_t>(selected) - 3};
      const auto output_desc =
          infer_single_output(Operation(operation), descriptions).value();
      auto output = output_buffer(memory.ref(), output_desc);
      const auto status = execute(ctx, operation, inputs,
          TensorView::create(output_desc, output.view()).value());
      ASSERT_TRUE(status.ok()) << status.message();
      std::size_t outer = 1;
      for (std::size_t axis = 0; axis < selected; ++axis)
        outer *= static_cast<std::size_t>(output_desc.dimensions()[axis]);
      std::vector<Value> expected;
      for (std::size_t row = 0; row < outer; ++row)
        for (const auto &part : logical) {
          const auto width = part.size() / outer;
          expected.insert(expected.end(), part.begin() + row * width,
                          part.begin() + (row + 1) * width);
        }
      const auto actual = half_test::read<Value>(output);
      ASSERT_EQ(actual.size(), expected.size());
      EXPECT_EQ(std::memcmp(actual.data(), expected.data(),
                            expected.size() * sizeof(Value)), 0);
    }
  }
}

TEST(GgmlCudaTensor, BalancedConcatRetainsPortOrderAndStorageBits) {
  auto ctx = context();
  balanced_concat<float>(ctx);
  balanced_concat<hyle::simd::float16_t>(ctx);
  balanced_concat<hyle::simd::bfloat16_t>(ctx);
  balanced_concat<std::int32_t>(ctx);
}

TEST(GgmlCudaTensor, UnsupportedNumericDomainsDeclineBeforeNativeExecution) {
  auto ctx = context();
  const auto description = dense(DataType::kFloat32, {2, 31});
  EXPECT_FALSE(prepare(ctx, Unary{UnaryKind::kSoftplus}, description));
  // Stable SiLU's typed F16 lowering is a separate, unqualified domain.
  EXPECT_FALSE(prepare(ctx, Unary{UnaryKind::kSilu},
                       dense(DataType::kFloat16, {2, 31})));
  const auto odd = TensorDesc::strided_bytes(DataType::kFloat32,
                                             std::array<std::int64_t, 1>{7},
                                             std::array<std::int64_t, 1>{5})
                       .value();
  EXPECT_FALSE(prepare(ctx, Copy{}, odd));
  EXPECT_FALSE(
      prepare(ctx, Softmax{.kind = SoftmaxKind::kLogSoftmax}, description));
}

TEST(GgmlCudaTensor, OriginalNarrowIq1CudaProgramsBindMatricesAndExperts) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  const auto activation = dense(DataType::kFloat32, {2, 256});
  const auto grouped_activation = dense(DataType::kFloat32, {1, 2, 256});
  const auto ids = dense(DataType::kInt32, {1, 2});
  for (const auto type : {GgufQuantizedType::kIq1Xs, GgufQuantizedType::kIq1Xxs,
       GgufQuantizedType::kIq1Xxxs}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto weight = gguf_weight_description(type, DataType::kFloat32, 256, 3).value();
    auto prepared = prepare(ctx, MatMul{}, activation, weight, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    const auto grouped = gguf_grouped_weight_description(type, DataType::kFloat32, 2, 3, 256).value();
    const std::array inputs{grouped, grouped_activation, ids};
    const auto op = model_ops::grouped_matmul();
    const auto output = infer_single_output(op, inputs).value();
    auto grouped_prepared = prepare(ctx, op, inputs, std::span(&output, 1), memory.ref());
    ASSERT_TRUE(grouped_prepared) << grouped_prepared.error().message();
    EXPECT_EQ(grouped_prepared->provider().id, ggml_cuda::kProviderId);
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
  auto created_workspace =
      ggml_cuda_test::workspace_buffer(*program, memory.ref());
  auto created = program->create_invocation(created_workspace.view());
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
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto format : {GgufQuantizedType::kQ4_0,
       GgufQuantizedType::kQ4_1, GgufQuantizedType::kQ2K,
       GgufQuantizedType::kIq2Xxs, GgufQuantizedType::kIq2Xs,
       GgufQuantizedType::kIq2S, GgufQuantizedType::kIq1S,
       GgufQuantizedType::kIq1M, GgufQuantizedType::kQ4K}) {
    SCOPED_TRACE(static_cast<int>(format));
    constexpr std::int64_t reduction = 256;
    constexpr std::int64_t columns = 2;
    const auto left_desc = dense(DataType::kFloat32, {1, reduction});
    const auto right_desc = gguf_weight_description(
        format, DataType::kFloat32, reduction, columns);
    ASSERT_TRUE(right_desc) << right_desc.error().message();
    const auto output_desc = dense(DataType::kFloat32, {1, columns});
    std::vector<float> left_values(static_cast<std::size_t>(reduction), 0.25F);
    std::vector<float> physical_weights(
        static_cast<std::size_t>(reduction * columns));
    std::fill_n(physical_weights.begin(), reduction, 1.0F);
    std::fill_n(physical_weights.begin() + reduction, reduction, -2.0F);
    const auto packed = d5_gguf_test::pack(format, physical_weights, columns,
                                         reduction, *right_desc->required_bytes());

    auto ctx = context();
    auto memory =
        cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
    auto left = upload<float>(memory.ref(), left_values);
    auto right = upload<std::byte>(memory.ref(), packed.bytes);
    auto output = output_buffer(memory.ref(), output_desc);
    const std::array inputs{
        ConstTensorView::create(left_desc, left.view()).value(),
        ConstTensorView::create(*right_desc, right.view()).value()};
    auto output_view = TensorView::create(output_desc, output.view()).value();
    auto prepared =
        prepare(ctx, Operation(MatMul{}), inputs, std::span(&output_view, 1));
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    auto invocation_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto invocation =
        prepared->create_invocation(invocation_workspace.view()).value();
    auto queue = ctx.create_queue().value();
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<float>(output);
      const auto scale = repeat == 0 ? 1.0F : 2.0F;
      ASSERT_EQ(actual.size(), 2U);
      for (std::size_t column = 0; column < actual.size(); ++column) {
        float expected = 0.0F;
        for (std::int64_t k = 0; k < reduction; ++k)
          expected += packed.decoded[column * reduction + k] * 0.25F * scale;
        // Constant activation is checked against the original decoded row;
        // include the original q8 activation scale's half rounding.
        EXPECT_NEAR(actual[column], expected, std::abs(expected) * 0.002F + 1.0e-5F);
      }
      if (repeat == 0) {
        std::fill(left_values.begin(), left_values.end(), 0.5F);
        ASSERT_TRUE(
            left.view().write(std::as_bytes(std::span(left_values))).ok());
      }
    }
  }
}

TEST(GgmlCudaTensor, NativeEmbeddingRowsUseTargetGgufFormatsAndCheckBounds) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t rows = 3;
  constexpr std::int64_t width = 256;
  constexpr std::array formats{
      GgufQuantizedType::kQ4_0, GgufQuantizedType::kQ4_1,
      GgufQuantizedType::kQ2K, GgufQuantizedType::kQ3K,
      GgufQuantizedType::kQ4K, GgufQuantizedType::kQ5K,
      GgufQuantizedType::kQ6K, GgufQuantizedType::kQ8_0,
      GgufQuantizedType::kIq2Xxs, GgufQuantizedType::kIq2Xs,
      GgufQuantizedType::kIq2S, GgufQuantizedType::kIq1S,
      GgufQuantizedType::kIq1M, GgufQuantizedType::kIq3Xxs,
      GgufQuantizedType::kIq3S, GgufQuantizedType::kIq4Nl,
      GgufQuantizedType::kIq4Xs, GgufQuantizedType::kMxfp4};
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
    auto invocation_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto invocation =
        prepared->create_invocation(invocation_workspace.view()).value();
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

  // Valid storage blocks do not imply the CUDA row decoder accepts this width.
  for (const auto format : {GgufQuantizedType::kIq4Nl,
                            GgufQuantizedType::kMxfp4}) {
    SCOPED_TRACE(static_cast<int>(format));
    const auto table_desc =
        gguf_weight_description(format, DataType::kFloat32, 32, rows)
            .value().transposed().value();
    const std::array inputs{table_desc, dense(DataType::kInt32, {2})};
    const auto output_desc = dense(DataType::kFloat32, {2, 32});
    auto rejected = prepare(ctx, Operation(Gather{}), inputs,
                            std::span(&output_desc, 1), memory.ref());
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
}

TEST(GgmlCudaTensor, QuantizedGroupedBanksKeepNativeBlockCapacity) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  for (const auto format : {GgufQuantizedType::kIq4Xs,
                            GgufQuantizedType::kQ8_0}) {
    const auto weight = gguf_grouped_weight_description(
        format, DataType::kFloat32, 512, 2048, 8192).value();
    const std::array inputs{weight, dense(DataType::kFloat32, {1, 2, 8192}),
                            dense(DataType::kInt32, {1, 2})};
    const auto output = infer_single_output(model_ops::grouped_matmul(), inputs).value();
    auto prepared = prepare(ctx, model_ops::grouped_matmul(), inputs,
                            std::span(&output, 1), memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  }
}

TEST(GgmlCudaTensor, FullIq4BankReadsLastExpertBeyondFourGiB) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t experts = 512, columns = 2048, reduction = 8192;
  const auto weight_desc = gguf_grouped_weight_description(
      GgufQuantizedType::kIq4Xs, DataType::kFloat32, experts, columns,
      reduction).value();
  const auto activation_desc = dense(DataType::kFloat32, {1, 2, reduction});
  const auto ids_desc = dense(DataType::kInt32, {1, 2});
  const std::array descriptions{weight_desc, activation_desc, ids_desc};
  const auto output_desc =
      infer_single_output(model_ops::grouped_matmul(), descriptions).value();
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto prepared = prepare(ctx, model_ops::grouped_matmul(), descriptions,
                          std::span(&output_desc, 1), memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  std::size_t free = 0, total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
  const auto bytes = *weight_desc.required_bytes();
  if (free < bytes + (std::size_t{1} << 30))
    GTEST_SKIP() << "Full IQ4 bank requires its bytes plus 1 GiB headroom";
  auto weights = output_buffer(memory.ref(), weight_desc);
  // Only the selected final expert is initialized. No full bank host copy,
  // fill, or F32 conversion is needed by the native indexed quantized kernel.
  const auto row_desc = gguf_weight_description(
      GgufQuantizedType::kIq4Xs, DataType::kFloat32, reduction, 1).value();
  const std::vector<float> row(reduction, 0.75F);
  const auto packed = d5_gguf_test::pack(GgufQuantizedType::kIq4Xs, row, 1,
                                       reduction, *row_desc.required_bytes());
  std::vector<std::byte> selected(static_cast<std::size_t>(columns) * packed.bytes.size());
  for (std::int64_t c = 0; c < columns; ++c)
    std::copy(packed.bytes.begin(), packed.bytes.end(),
              selected.begin() + static_cast<std::size_t>(c) * packed.bytes.size());
  const auto offset = static_cast<std::size_t>(experts - 1) * selected.size();
  ASSERT_GT(offset, UINT32_MAX);
  ASSERT_TRUE(weights.view().subview(offset, selected.size())->write(selected).ok());
  auto activation = output_buffer(memory.ref(), activation_desc);
  const std::array<std::int32_t, 2> ids{experts - 1, experts - 1};
  auto index = upload<std::int32_t>(memory.ref(), ids);
  auto output = output_buffer(memory.ref(), output_desc);
  const std::array inputs{ConstTensorView::create(weight_desc, weights.view()).value(),
      ConstTensorView::create(activation_desc, activation.view()).value(),
      ConstTensorView::create(ids_desc, index.view()).value()};
  const auto result = TensorView::create(output_desc, output.view()).value();
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
  auto queue = ctx.create_queue().value();
  for (const float value : {0.25F, 0.5F}) {
    const std::vector<float> x(2 * reduction, value);
    ASSERT_TRUE(activation.view().write(std::as_bytes(std::span(x))).ok());
    auto done = invocation.submit(queue, inputs, std::span(&result, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    double expected = 0;
    for (const float w : packed.decoded)
      expected += static_cast<double>(w) * value;
    const auto actual = half_test::read<float>(output);
    for (const float y : actual)
      EXPECT_NEAR(y, expected, std::abs(expected) * 0.002 + 1.0e-5);
  }
}

TEST(GgmlCudaTensor, NativeQ8GroupedExpertsUseExactExpertIds) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto format : {GgufQuantizedType::kQ4_0,
       GgufQuantizedType::kQ4_1, GgufQuantizedType::kQ2K,
       GgufQuantizedType::kIq2Xxs, GgufQuantizedType::kIq2Xs,
       GgufQuantizedType::kIq2S, GgufQuantizedType::kIq1S,
       GgufQuantizedType::kIq1M, GgufQuantizedType::kQ8_0}) {
    SCOPED_TRACE(static_cast<int>(format));
    constexpr std::int64_t experts = 3;
    constexpr std::int64_t columns = 2;
    constexpr std::int64_t reduction = 256;
    const auto weights_desc = gguf_grouped_weight_description(
        format, DataType::kFloat32, experts, columns,
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
    const auto packed = d5_gguf_test::pack(format, physical_weights,
        experts * columns, reduction, *weights_desc->required_bytes());
    std::vector<float> activations(static_cast<std::size_t>(2 * reduction),
                                   0.25F);
    constexpr std::array<std::int32_t, 2> ids{0, 2};

    auto ctx = context();
    auto memory =
        cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
    auto weights = upload<std::byte>(memory.ref(), packed.bytes);
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
    for (std::size_t slot = 0; slot < ids.size(); ++slot)
      for (std::int64_t column = 0; column < columns; ++column) {
        float expected = 0.0F;
        const auto row = ids[slot] * columns + column;
        for (std::int64_t k = 0; k < reduction; ++k)
          expected += packed.decoded[row * reduction + k] * 0.25F;
        EXPECT_NEAR(actual[slot * columns + column], expected,
                    std::abs(expected) * 0.002F + 1.0e-5F);
      }
  }
}

TEST(GgmlCudaTensor, DenseBfloatGroupedExpertsUseNativeMulMatIdAndReplay) {
  using BF16 = hyle::simd::bfloat16_t;
  constexpr std::int64_t experts = 3;
  constexpr std::int64_t columns = 2;
  constexpr std::int64_t reduction = 32;
  const auto weights_desc =
      dense(DataType::kBFloat16, {experts, columns, reduction});
  constexpr std::int64_t slots = 3;
  const auto activations_desc = dense(DataType::kFloat32, {1, slots, reduction});
  const auto ids_desc = dense(DataType::kInt32, {1, slots});
  const std::array descriptions{weights_desc, activations_desc, ids_desc};
  const auto output_desc =
      infer_single_output(model_ops::grouped_matmul(), descriptions).value();
  constexpr std::array<double, experts * columns> row_values{1.0, 2.0, 3.0,
                                                              4.0, -1.0, 0.5};
  std::vector<BF16> weights(experts * columns * reduction);
  for (std::size_t row = 0; row < row_values.size(); ++row)
    std::fill_n(weights.begin() + row * reduction, reduction,
                half_test::encode<BF16>(row_values[row]));
  std::vector<float> activations(slots * reduction, 0.25F);
  std::array<std::int32_t, slots> ids{0, 2, 2};
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
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
  auto queue = ctx.create_queue().value();
  for (int replay = 1; replay <= 2; ++replay) {
    ids = replay == 1 ? std::array<std::int32_t, slots>{0, 2, 2}
                      : std::array<std::int32_t, slots>{2, 0, 0};
    for (std::int64_t slot = 0; slot < slots; ++slot)
      std::fill_n(activations.begin() + slot * reduction, reduction,
                  0.25F * replay + 0.03125F * slot + 0.00137F);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(activations)))
                    .ok());
    ASSERT_TRUE(id_buffer.view().write(std::as_bytes(std::span(ids))).ok());
    auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    ASSERT_EQ(actual.size(), std::size_t(slots * columns));
    for (std::int64_t slot = 0; slot < slots; ++slot)
      for (std::int64_t column = 0; column < columns; ++column) {
        const auto expected = float(row_values[ids[slot] * columns + column]) *
                              activations[slot * reduction] * reduction;
        EXPECT_NEAR(actual[slot * columns + column], expected, 1.0e-5F)
            << "slot=" << slot << " column=" << column;
      }
  }
}

TEST(GgmlCudaTensor, LargeGroupedF32RowsRetainNonTf32OperandsAndRepeatedIds) {
  constexpr std::int64_t experts=3,n=512,k=128,slots=3;
  const auto wdesc=dense(DataType::kFloat32,{experts,n,k});
  const auto adesc=dense(DataType::kFloat32,{1,slots,k});
  const auto idesc=dense(DataType::kInt32,{1,slots});
  const std::array descs{wdesc,adesc,idesc};
  const auto ydesc=infer_single_output(model_ops::grouped_matmul(),descs).value();
  std::vector<float> weights(experts*n*k);
  for(std::int64_t e=0;e<experts;++e)for(std::int64_t row=0;row<n;++row) {
    weights[(e*n+row)*k+row%k]=0.5F+float(e)/8;
    weights[(e*n+row)*k+(row+17)%k]=-0.125F;
  }
  auto ctx=context();auto memory=cuda::MemoryResource::create(ctx,cuda::MemoryKind::kDevice).value();
  auto wb=upload<float>(memory.ref(),weights),ab=output_buffer(memory.ref(),adesc);
  auto ib=output_buffer(memory.ref(),idesc),yb=output_buffer(memory.ref(),ydesc);
  const std::array in{ConstTensorView::create(wdesc,wb.view()).value(),
    ConstTensorView::create(adesc,ab.view()).value(),ConstTensorView::create(idesc,ib.view()).value()};
  const auto out=TensorView::create(ydesc,yb.view()).value();
  auto prepared=prepare(ctx,model_ops::grouped_matmul(),in,std::span(&out,1));
  ASSERT_TRUE(prepared)<<prepared.error().message();
  auto call_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto call =
      prepared->create_invocation(call_workspace.view()).value();
  auto queue = ctx.create_queue().value();
  for(int replay=0;replay<2;++replay) {
    const std::array<std::int32_t,slots> ids=replay?std::array<std::int32_t,slots>{0,2,0}:
                                                    std::array<std::int32_t,slots>{2,0,2};
    std::vector<float> activation(slots*k);
    for(std::int64_t i=0;i<slots*k;++i)activation[i]=float((i*7+replay*13)%41-20)*0.01337F+0.00013F;
    ASSERT_TRUE(ab.view().write(std::as_bytes(std::span(activation))).ok());
    ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(ids))).ok());
    auto done=call.submit(queue,in,std::span(&out,1));ASSERT_TRUE(done)<<done.error().message();
    const auto status=done->wait();ASSERT_TRUE(status.ok())<<status.message();
    const auto actual=half_test::read<float>(yb);ASSERT_EQ(actual.size(),std::size_t(slots*n));
    for(std::int64_t slot=0;slot<slots;++slot)for(std::int64_t row=0;row<n;++row) {
      const auto expected=double(0.5F+float(ids[slot])/8)*activation[slot*k+row%k]
                          -0.125*activation[slot*k+(row+17)%k];
      EXPECT_NEAR(actual[slot*n+row],expected,1.0e-6)<<slot<<" "<<row;
    }
  }
}

TEST(GgmlCudaTensor, BfloatRouterMatMulRetainsNativeF32Score) {
  using BF16 = hyle::simd::bfloat16_t;
  const auto activation_desc = dense(DataType::kBFloat16, {1, 2});
  const auto weight_desc = dense(DataType::kBFloat16, {2, 2});
  const std::array descriptions{activation_desc, weight_desc};
  const auto output_desc =
      infer_single_output(model_ops::dense_linear_f32(), descriptions).value();
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
  auto prepared = prepare(ctx, model_ops::dense_linear_f32(), inputs,
                          std::span(&output_view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
  auto queue = ctx.create_queue().value();
  auto done = invocation.submit(queue, inputs, std::span(&output_view, 1));
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto actual = half_test::read<float>(output);
  ASSERT_EQ(actual.size(), 2U);
  EXPECT_NEAR(actual[0], 1.001953125F, 1.0e-6F);
  EXPECT_NEAR(actual[1], 1.0F, 1.0e-6F);
}

TEST(GgmlCudaTensor, CheckedExpertIdsKeepOutputAndAllowRecreatedInvocationRetry) {
  constexpr std::int64_t experts = 3, columns = 2, reduction = 32;
  const auto wdesc = dense(DataType::kFloat32, {experts, columns, reduction});
  const auto adesc = dense(DataType::kFloat32, {1, 2, reduction});
  const auto idesc = dense(DataType::kInt32, {1, 2});
  const std::array descriptions{wdesc, adesc, idesc};
  const auto ydesc = infer_single_output(model_ops::grouped_matmul(), descriptions).value();
  std::vector<float> weights(experts * columns * reduction);
  for (std::size_t row = 0; row < std::size_t(experts * columns); ++row)
    std::fill_n(weights.begin() + row * reduction, reduction, float(row + 1));
  const std::vector<float> activation(2 * reduction, 0.25F);
  const std::vector<float> sentinel(4, -917.25F);
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  auto w = upload<float>(memory.ref(), weights);
  auto a = upload<float>(memory.ref(), activation);
  auto ids = output_buffer(memory.ref(), idesc);
  auto y = upload<float>(memory.ref(), sentinel);
  const std::array inputs{ConstTensorView::create(wdesc, w.view()).value(),
      ConstTensorView::create(adesc, a.view()).value(),
      ConstTensorView::create(idesc, ids.view()).value()};
  const auto output = TensorView::create(ydesc, y.view()).value();
  auto prepared = prepare(ctx, model_ops::grouped_matmul(), inputs, std::span(&output, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto queue = ctx.create_queue().value();
  for (const auto bad : {-1, INT32_MAX}) {
    auto invocation_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto invocation =
        prepared->create_invocation(invocation_workspace.view()).value();
    const std::array<std::int32_t, 2> selected{2, bad};
    ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = invocation.submit(queue, inputs, std::span(&output, 1));
    ASSERT_TRUE(done) << done.error().message();
    EXPECT_TRUE(done->wait().IsOutOfRange());
    EXPECT_EQ(half_test::read<float>(y), sentinel);
    EXPECT_EQ(half_test::read<std::int32_t>(ids),
              std::vector<std::int32_t>(selected.begin(), selected.end()));
  }
  const std::array<std::int32_t, 2> selected{2, 0};
  ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation =
      prepared->create_invocation(invocation_workspace.view()).value();
  auto done = invocation.submit(queue, inputs, std::span(&output, 1));
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  EXPECT_EQ(half_test::read<float>(y), (std::vector<float>{40, 48, 8, 16}));
}

namespace {
template <class T> void gather_elements_rows() {
  namespace fixture = gather_elements_test;
  const auto ddesc = dense(half_test::data_type<T>(),
      {std::int64_t(fixture::kRows), std::int64_t(fixture::kColumns)});
  const auto idesc = dense(DataType::kInt32,
      {std::int64_t(fixture::kRows), std::int64_t(fixture::kSelected)});
  const auto ydesc = TensorDesc::dense(half_test::data_type<T>(), idesc.dimensions()).value();
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  for (const auto bounds : {IndexBoundsPolicy::kChecked, IndexBoundsPolicy::kUnchecked}) {
    SCOPED_TRACE(static_cast<int>(bounds));
    auto data = output_buffer(memory.ref(), ddesc);
    auto ids = output_buffer(memory.ref(), idesc);
    std::array outputs{output_buffer(memory.ref(), ydesc), output_buffer(memory.ref(), ydesc)};
    const std::array descs{ddesc, idesc};
    auto prepared = prepare(ctx, model_ops::gather_elements(-1, bounds),
                            descs, std::span(&ydesc, 1), memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    std::array workspaces{
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref()),
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref())};
    std::array calls{prepared->create_invocation(workspaces[0].view()).value(),
                     prepared->create_invocation(workspaces[1].view()).value()};
    std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
    for (int phase = 0; phase < 2; ++phase) {
      const auto source = fixture::data(phase);
      const auto selected = fixture::indices(phase);
      std::vector<T> stored;
      for (const auto value : source) stored.push_back(half_test::encode<T>(value));
      ASSERT_TRUE(data.view().write(std::as_bytes(std::span(stored))).ok());
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      const std::array inputs{ConstTensorView::create(ddesc, data.view()).value(),
          ConstTensorView::create(idesc, ids.view()).value()};
      auto out = TensorView::create(ydesc, outputs[phase].view()).value();
      auto done = calls[phase].submit(queues[phase], inputs, std::span(&out, 1));
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<T>(outputs[phase]);
      const auto expected = fixture::reference(source, selected);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(half_test::decode(actual[i]), expected[i]) << i;
      if (phase == 1) {
        const auto old = half_test::read<T>(outputs[0]);
        const auto reference = fixture::reference(fixture::data(), fixture::indices());
        for (std::size_t i = 0; i < old.size(); ++i)
          EXPECT_EQ(half_test::decode(old[i]), reference[i]) << i;
      }
    }
    if (bounds == IndexBoundsPolicy::kChecked) {
      const auto before = half_test::read<T>(outputs[0]);
      auto invalid = fixture::indices();
      invalid.back() = std::int32_t(fixture::kColumns);
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(invalid))).ok());
      const std::array inputs{ConstTensorView::create(ddesc, data.view()).value(),
          ConstTensorView::create(idesc, ids.view()).value()};
      const auto out = TensorView::create(ydesc, outputs[0].view()).value();
      auto done = calls[0].submit(queues[0], inputs, std::span(&out, 1));
      ASSERT_TRUE(done) << done.error().message();
      EXPECT_TRUE(done->wait().IsOutOfRange());
      const auto after = half_test::read<T>(outputs[0]);
      EXPECT_TRUE(std::ranges::equal(std::as_bytes(std::span(before)),
                                    std::as_bytes(std::span(after))));
    }
  }
}
} // namespace

TEST(GgmlCudaTensor, GatherElementsUsesPerRowIdsAndPreservesDtypeAndBounds) {
  gather_elements_rows<float>();
  gather_elements_rows<hyle::simd::float16_t>();
  gather_elements_rows<hyle::simd::bfloat16_t>();
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

TEST(GgmlCudaTensor, BatchedF32MatMulRebindsIndependentSlotsWithoutNarrowing) {
  auto ctx = context();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t m = 17, k = 37, n = 19;
  for (const int rank : {3, 4}) {
    SCOPED_TRACE(rank);
    const auto left_desc = rank == 3 ? dense(DataType::kFloat32, {2, m, k})
        : dense(DataType::kFloat32, {2, 3, m, k});
    const auto right_desc = rank == 3 ? dense(DataType::kFloat32, {1, k, n})
        : dense(DataType::kFloat32, {1, 3, k, n});
    const std::array descriptions{left_desc, right_desc};
    const auto output_desc = infer_single_output(Operation(MatMul{}), descriptions).value();
    auto prepared = prepare(ctx, Operation(MatMul{}), descriptions,
                            std::span(&output_desc, 1), memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
    auto first_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto first =
        prepared->create_invocation(first_workspace.view()).value();
    auto second_workspace =
        ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
    auto second =
        prepared->create_invocation(second_workspace.view()).value();
    auto first_queue = ctx.create_queue().value();
    auto second_queue = ctx.create_queue().value();
    Buffer old_output;
    std::vector<float> old_values;
    const auto batches = *left_desc.element_count() / (m * k);
    const auto right_batches = *right_desc.element_count() / (k * n);
    for (const int phase : {0, 1}) {
      SCOPED_TRACE(phase);
      std::array<Buffer, 2> left, right, output;
      std::array<std::vector<float>, 2> a, b;
      for (std::size_t slot = 0; slot < 2; ++slot) {
        a[slot].resize(*left_desc.element_count());
        b[slot].resize(*right_desc.element_count());
        // Actual F32 values deliberately retain bits that TF32 would discard.
        for (std::size_t i = 0; i < a[slot].size(); ++i)
          a[slot][i] = float(0.15 + double(i % 19) * 0.037 + phase * 0.013 + slot * 0.019);
        for (std::size_t i = 0; i < b[slot].size(); ++i)
          b[slot][i] = float(0.11 + double(i % 23) * 0.029 - phase * 0.007 + slot * 0.031);
        left[slot] = upload<float>(memory.ref(), a[slot]);
        right[slot] = upload<float>(memory.ref(), b[slot]);
        output[slot] = output_buffer(memory.ref(), output_desc);
      }
      const auto inputs = [&](std::size_t slot) {
        return std::array{ConstTensorView::create(left_desc, left[slot].view()).value(),
                          ConstTensorView::create(right_desc, right[slot].view()).value()};
      };
      const auto destination = [&](std::size_t slot) {
        return TensorView::create(output_desc, output[slot].view()).value();
      };
      auto one = first.submit(first_queue, inputs(0), destination(0));
      ASSERT_TRUE(one) << one.error().message();
      auto two = second.submit(second_queue, inputs(1), destination(1));
      ASSERT_TRUE(two) << two.error().message();
      ASSERT_TRUE(two->wait().ok());
      ASSERT_TRUE(one->wait().ok());
      for (std::size_t slot = 0; slot < 2; ++slot) {
        const auto actual = half_test::read<float>(output[slot]);
        for (std::size_t batch = 0; batch < batches; ++batch)
          for (std::size_t row = 0; row < m; ++row)
            for (std::size_t column = 0; column < n; ++column) {
              double expected = 0;
              for (std::size_t inner = 0; inner < k; ++inner)
                expected += double(a[slot][(batch * m + row) * k + inner]) *
                    b[slot][((batch % right_batches) * k + inner) * n + column];
              EXPECT_NEAR(actual[(batch * m + row) * n + column], expected,
                          2e-6 + 6e-6 * std::abs(expected));
            }
      }
      if (phase == 0) {
        old_values = half_test::read<float>(output[0]);
        old_output = std::move(output[0]);
      } else {
        EXPECT_EQ(half_test::read<float>(old_output), old_values);
      }
    }
  }
}
} // namespace
