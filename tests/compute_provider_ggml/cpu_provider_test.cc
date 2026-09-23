#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/quota_allocator.h"
#include "../compute_ops/qwen35_recurrent_contract_test_support.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
using hyle::mem::Buffer;

TensorDesc dense(DataType type, std::initializer_list<std::int64_t> shape) {
  return TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
Buffer output_buffer(const TensorDesc &description) {
  return Buffer::create(hyle::mem::system_resource(),
                        description.required_bytes().value(), 64)
      .value();
}
template <class Op>
hyle::Status execute(Context &context, Op operation,
                     std::span<const ConstTensorView> inputs,
                     TensorView output) {
  auto prepared =
      prepare(context, Operation(operation), inputs, std::span(&output, 1));
  if (!prepared)
    return std::move(prepared).error();
  EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
  auto invocation = prepared->create_invocation();
  if (!invocation)
    return std::move(invocation).error();
  auto queue = context.create_queue();
  if (!queue)
    return std::move(queue).error();
  auto completion = invocation->submit(*queue, inputs, std::span(&output, 1));
  if (!completion)
    return std::move(completion).error();
  return completion->wait();
}

template <class T> void norms() {
  const auto description = dense(half_test::data_type<T>(), {2, 17, 31});
  const auto parameter = dense(DataType::kFloat32, {31});
  std::vector<T> values(description.element_count().value());
  std::vector<float> scales(31), biases(31);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = half_test::encode<T>((static_cast<int>(i % 29) - 14) / 8.0);
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = 0.5F + static_cast<float>(i % 3) / 8.F;
    biases[i] = static_cast<float>(i % 5) / 16.F;
  }
  auto source = half_test::buffer<T>(values);
  auto scale = half_test::buffer<float>(scales);
  auto bias = half_test::buffer<float>(biases);
  auto output = output_buffer(description);
  const std::array inputs{
      ConstTensorView::create(description, source.view()).value(),
      ConstTensorView::create(parameter, scale.view()).value(),
      ConstTensorView::create(parameter, bias.view()).value()};
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  for (bool rms : {true, false}) {
    const auto operation = rms ? Operation(RmsNorm{}) : Operation(LayerNorm{});
    auto status =
        execute(context, operation, std::span(inputs).first(rms ? 2 : 3),
                TensorView::create(description, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<T>(output);
    for (std::size_t row = 0; row < 34; ++row) {
      double mean = 0, squares = 0;
      if (!rms) {
        for (std::size_t col = 0; col < 31; ++col)
          mean += half_test::decode(values[row * 31 + col]);
        mean /= 31;
      }
      for (std::size_t col = 0; col < 31; ++col) {
        const double centered =
            half_test::decode(values[row * 31 + col]) - mean;
        squares += centered * centered;
      }
      for (std::size_t col = 0; col < 31; ++col) {
        const auto index = row * 31 + col;
        const double expected = (half_test::decode(values[index]) - mean) /
                                    std::sqrt(squares / 31 + 1.e-5) *
                                    scales[col] +
                                (rms ? 0 : biases[col]);
        half_test::expect_near(actual[index], expected);
      }
    }
  }
}
TEST(GgmlCpuProvider, NormsUseIndependentOracleAcrossStorageTypes) {
  norms<float>();
  norms<hyle::simd::float16_t>();
  norms<hyle::simd::bfloat16_t>();
}

TEST(GgmlCpuProvider, MatureCausalConvUsesNaturalSequenceLayout) {
  const auto samples_desc = dense(DataType::kFloat32, {1, 5, 2});
  const auto weights_desc = dense(DataType::kFloat32, {2, 3});
  const auto output_desc = dense(DataType::kFloat32, {1, 3, 2});
  const std::array<float, 10> samples{1, 10, 2, 20, 3, 30, 4, 40, 5, 50};
  const std::array<float, 6> weights{1, 2, 3, -1, 0, 1};
  auto samples_buffer = half_test::buffer<float>(samples);
  auto weights_buffer = half_test::buffer<float>(weights);
  auto output = output_buffer(output_desc);
  const std::array inputs{
      ConstTensorView::create(samples_desc, samples_buffer.view()).value(),
      ConstTensorView::create(weights_desc, weights_buffer.view()).value()};
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto status =
      execute(context, model_ops::causal_conv1d(), inputs,
              TensorView::create(output_desc, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(half_test::read<float>(output),
            (std::vector<float>{14, 20, 20, 20, 26, 20}));
}

TEST(GgmlCpuProvider, MatureKdaPublishesScoresAndStateIndependently) {
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
    storage.push_back(half_test::buffer<float>(values));
    inputs.push_back(
        ConstTensorView::create(descriptions[i], storage.back().view())
            .value());
  }
  const std::array output_desc{descriptions[2], descriptions[5]};
  std::array output_storage{output_buffer(output_desc[0]),
                            output_buffer(output_desc[1])};
  const std::array outputs{
      TensorView::create(output_desc[0], output_storage[0].view()).value(),
      TensorView::create(output_desc[1], output_storage[1].view()).value()};
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto prepared = prepare(context, Operation(GatedDeltaNet{}), inputs, outputs);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  auto completion = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(half_test::read<float>(output_storage[0]),
            std::vector<float>(*output_desc[0].element_count(), 0.F));
  EXPECT_EQ(half_test::read<float>(output_storage[1]), expected_state);
}

TEST(GgmlCpuProvider,
     Qwen35RealShapeAcceptsF32AndRejectsBfloatAtPreparation) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  for (const std::int64_t tokens : {1, 17, 128}) {
    SCOPED_TRACE(tokens);
    const auto f32 = qwen35_recurrent_contract_test::inputs(
        DataType::kFloat32, tokens);
    const auto f32_outputs = qwen35_recurrent_contract_test::outputs(f32);
    auto accepted = prepare(context, Operation(GatedDeltaNet{}), f32,
                            f32_outputs, hyle::mem::system_resource());
    ASSERT_TRUE(accepted) << accepted.error().message();
    EXPECT_EQ(accepted->provider().id, ggml_cpu::kProviderId);

    const auto b16 = qwen35_recurrent_contract_test::inputs(
        DataType::kBFloat16, tokens);
    const auto b16_outputs = qwen35_recurrent_contract_test::outputs(b16);
    auto rejected = prepare(context, Operation(GatedDeltaNet{}), b16,
                            b16_outputs, hyle::mem::system_resource());
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
}

TEST(GgmlCpuProvider, MatureMoeAndHyperConnectionBatchUsesOrdinaryEntry) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    std::vector<Buffer> storage;
    std::vector<ConstTensorView> inputs;
    storage.reserve(test.inputs.size());
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i) {
      storage.push_back(output_buffer(test.inputs[i]));
      ASSERT_TRUE(storage.back().view().write(test.payloads[i]).ok());
      inputs.push_back(
          ConstTensorView::create(test.inputs[i], storage.back().view()).value());
    }
    auto output = output_buffer(test.output);
    const auto status = execute(
        context, test.operation, inputs,
        TensorView::create(test.output, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(output);
    ASSERT_EQ(actual.size(), test.expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCpuProvider, MatureLightningIndexerUsesOrdinaryEntryAndReplays) {
  const sparse_fragment_test::Fixture test;
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  std::vector<Buffer> storage;
  std::vector<ConstTensorView> inputs;
  for (std::size_t i = 0; i < test.inputs.size(); ++i) {
    storage.push_back(output_buffer(test.inputs[i]));
    ASSERT_TRUE(storage.back().view().write(test.payloads[i]).ok());
    inputs.push_back(
        ConstTensorView::create(test.inputs[i], storage.back().view()).value());
  }
  auto output = output_buffer(test.output);
  auto view = TensorView::create(test.output, output.view()).value();
  auto prepared = prepare(context, test.operation, inputs, std::span(&view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue, inputs, std::span(&view, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCpuProvider, AdmittedUnarySpecialValuesAndHalfCastRounding) {
  const std::array values{-std::numeric_limits<float>::infinity(),
                          -4.0F,
                          -0.0F,
                          0.0F,
                          0.125F,
                          2.0F,
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()};
  const auto description = dense(DataType::kFloat32, {8});
  auto source = half_test::buffer<float>(values);
  auto output = output_buffer(description);
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  for (auto kind :
       {UnaryKind::kAbs, UnaryKind::kNeg, UnaryKind::kExp, UnaryKind::kLog,
        UnaryKind::kSqrt, UnaryKind::kRsqrt, UnaryKind::kTanh}) {
    auto status =
        execute(context, Unary{kind}, std::span(&input, 1),
                TensorView::create(description, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(output);
    for (std::size_t i = 0; i < values.size(); ++i) {
      double expected = 0;
      switch (kind) {
      case UnaryKind::kAbs:
        expected = std::abs(values[i]);
        break;
      case UnaryKind::kNeg:
        expected = -values[i];
        break;
      case UnaryKind::kExp:
        expected = std::exp(values[i]);
        break;
      case UnaryKind::kLog:
        expected = std::log(values[i]);
        break;
      case UnaryKind::kSqrt:
        expected = std::sqrt(values[i]);
        break;
      case UnaryKind::kRsqrt:
        expected = 1.0F / std::sqrt(values[i]);
        break;
      case UnaryKind::kTanh:
        expected = std::tanh(values[i]);
        break;
      default:
        FAIL();
      }
      half_test::expect_near(actual[i], expected);
    }
  }
  // RNE ties, signed zero, subnormals and infinities use an independent
  // encoder.
  const std::array rounding{
      -0.0F,          1.00048828125F,
      1.00146484375F, 1.00390625F,
      1.01171875F,    0x1p-24F,
      65504.F,        std::numeric_limits<float>::infinity()};
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(rounding))).ok());
  const auto check = [&]<class T>() {
    const auto converted = dense(half_test::data_type<T>(), {8});
    auto cast_output = output_buffer(converted);
    auto status =
        execute(context, Cast{half_test::data_type<T>()}, std::span(&input, 1),
                TensorView::create(converted, cast_output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<T>(cast_output);
    for (std::size_t i = 0; i < rounding.size(); ++i)
      EXPECT_EQ(actual[i].bits, half_test::encode<T>(rounding[i]).bits);
  };
  check.template operator()<hyle::simd::float16_t>();
  check.template operator()<hyle::simd::bfloat16_t>();
}

TEST(GgmlCpuProvider, DenseRankSixBroadcastStridedCopyConcatAndGather) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description = dense(DataType::kFloat32, {1, 2, 1, 3, 1, 7});
  const auto row = dense(DataType::kFloat32, {7});
  std::vector<float> values(42), scale_values{1, 2, 3, 4, 5, 6, 7};
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i) / 8;
  auto source = half_test::buffer<float>(values);
  auto scale = half_test::buffer<float>(scale_values);
  auto output = output_buffer(description);
  const std::array inputs{
      ConstTensorView::create(row, scale.view()).value(),
      ConstTensorView::create(description, source.view()).value()};
  auto status = execute(context, Binary{BinaryKind::kMultiply}, inputs,
                        TensorView::create(description, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = half_test::read<float>(output);
  for (std::size_t i = 0; i < values.size(); ++i)
    EXPECT_FLOAT_EQ(actual[i], values[i] * scale_values[i % 7]);

  const auto strided =
      TensorDesc::strided_bytes(DataType::kFloat32,
                                std::array<std::int64_t, 2>{3, 7},
                                std::array<std::int64_t, 2>{56, 8})
          .value();
  const auto copied = dense(DataType::kFloat32, {3, 7});
  auto copy = output_buffer(copied);
  const auto view = ConstTensorView::create(strided, source.view()).value();
  status = execute(context, Copy{}, std::span(&view, 1),
                   TensorView::create(copied, copy.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto copied_values = half_test::read<float>(copy);
  for (std::size_t i = 0; i < copied_values.size(); ++i)
    EXPECT_FLOAT_EQ(copied_values[i], values[i * 2]);

  const std::array pieces{ConstTensorView::create(copied, copy.view()).value(),
                          ConstTensorView::create(copied, copy.view()).value()};
  const auto joined = dense(DataType::kFloat32, {3, 14});
  auto joined_buffer = output_buffer(joined);
  status = execute(context, Concat{.axis = -1}, pieces,
                   TensorView::create(joined, joined_buffer.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto joined_values = half_test::read<float>(joined_buffer);
  for (std::size_t i = 0; i < joined_values.size(); ++i)
    EXPECT_FLOAT_EQ(joined_values[i], copied_values[(i / 14) * 7 + i % 7]);

  const auto indices_desc = dense(DataType::kInt32, {2});
  auto indices =
      half_test::buffer<std::int32_t>(std::array<std::int32_t, 2>{2, 0});
  const auto selected = dense(DataType::kFloat32, {2, 14});
  auto gathered = output_buffer(selected);
  const std::array indexed{
      ConstTensorView::create(joined, joined_buffer.view()).value(),
      ConstTensorView::create(indices_desc, indices.view()).value()};
  status = execute(context, Gather{.axis = 0}, indexed,
                   TensorView::create(selected, gathered.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto gathered_values = half_test::read<float>(gathered);
  for (std::size_t i = 0; i < gathered_values.size(); ++i)
    EXPECT_FLOAT_EQ(gathered_values[i],
                    joined_values[(i < 14 ? 28 : 0) + i % 14]);
  const std::array<std::int32_t, 2> invalid{3, -1};
  ASSERT_TRUE(indices.view().write(std::as_bytes(std::span(invalid))).ok());
  status = execute(context, Gather{.axis = 0}, indexed,
                   TensorView::create(selected, gathered.view()).value());
  EXPECT_TRUE(status.IsOutOfRange()) << status.message();
  EXPECT_EQ(half_test::read<float>(gathered), gathered_values);
}

template <class Value> void check_gather_axes() {
  const auto type = std::is_same_v<Value, std::int32_t>
                        ? DataType::kInt32
                        : half_test::data_type<Value>();
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto queue = context.create_queue().value();
  for (const bool strided_source : {false, true}) {
    const auto item = static_cast<std::int64_t>(sizeof(Value));
    const auto description =
        strided_source ? TensorDesc::strided_bytes(
                             type, std::array<std::int64_t, 3>{2, 3, 5},
                             std::array{30 * item, 10 * item, 2 * item})
                             .value()
                       : dense(type, {2, 1, 3, 2, 1, 5});
    const auto count = description.element_count().value();
    const std::size_t spacing = strided_source ? 2 : 1;
    std::vector<Value> logical(count), physical(count * spacing);
    for (std::size_t i = 0; i < count; ++i) {
      if constexpr (std::is_same_v<Value, std::int32_t>)
        logical[i] = 16777217 + static_cast<std::int32_t>(i * 3);
      else
        logical[i] = half_test::encode<Value>(static_cast<double>(i) / 8 - 3);
      physical[i * spacing] = logical[i];
    }
    auto source = half_test::buffer<Value>(physical);
    for (std::size_t selected = 0; selected < description.rank(); ++selected) {
      const auto extent = description.dimensions()[selected];
      std::size_t inner = 1, outer = 1;
      for (std::size_t i = 0; i < selected; ++i)
        outer *= static_cast<std::size_t>(description.dimensions()[i]);
      for (std::size_t i = selected + 1; i < description.rank(); ++i)
        inner *= static_cast<std::size_t>(description.dimensions()[i]);
      for (int layout = 0; layout < 3; ++layout) {
        SCOPED_TRACE(::testing::Message()
                     << "axis=" << selected << " indices=" << layout
                     << " strided source=" << strided_source);
        const auto index_desc =
            layout == 0 ? dense(DataType::kInt32, {})
            : layout == 1
                ? dense(DataType::kInt32, {2, 2})
                : TensorDesc::strided_bytes(DataType::kInt32,
                                            std::array<std::int64_t, 2>{2, 2},
                                            std::array<std::int64_t, 2>{24, 8})
                      .value();
        const std::size_t index_count = layout == 0 ? 1 : 4;
        const std::array<std::size_t, 4> offsets =
            layout == 2 ? std::array<std::size_t, 4>{0, 2, 6, 8}
                        : std::array<std::size_t, 4>{0, 1, 2, 3};
        std::vector<std::int32_t> index_data(layout == 2 ? 9 : index_count,
                                             -99);
        std::array<std::int32_t, 4> ids{static_cast<std::int32_t>(extent - 1),
                                        0,
                                        static_cast<std::int32_t>(extent - 1),
                                        static_cast<std::int32_t>(extent / 2)};
        auto indices = half_test::buffer<std::int32_t>(index_data);
        const Gather operation{
            .axis = static_cast<std::int32_t>(selected) -
                    static_cast<std::int32_t>(description.rank())};
        const std::array descriptors{description, index_desc};
        const auto out_desc =
            infer_single_output(Operation(operation), descriptors).value();
        auto output = output_buffer(out_desc);
        const std::array inputs{
            ConstTensorView::create(description, source.view()).value(),
            ConstTensorView::create(index_desc, indices.view()).value()};
        auto out_view = TensorView::create(out_desc, output.view()).value();
        auto prepared = prepare(context, Operation(operation), inputs,
                                std::span(&out_view, 1));
        ASSERT_TRUE(prepared) << prepared.error().message();
        EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
        auto invocation = prepared->create_invocation();
        ASSERT_TRUE(invocation) << invocation.error().message();
        const auto submit = [&]() -> hyle::Status {
          auto done =
              invocation->submit(queue, inputs, std::span(&out_view, 1));
          if (!done)
            return std::move(done).error();
          return done->wait();
        };
        std::vector<Value> expected(outer * index_count * inner);
        for (int run = 0; run < 2; ++run) {
          for (std::size_t i = 0; i < index_count; ++i)
            index_data[offsets[i]] = ids[i];
          ASSERT_TRUE(
              indices.view().write(std::as_bytes(std::span(index_data))).ok());
          const auto status = submit();
          ASSERT_TRUE(status.ok()) << status.message();
          for (std::size_t row = 0; row < outer; ++row)
            for (std::size_t i = 0; i < index_count; ++i)
              for (std::size_t col = 0; col < inner; ++col)
                expected[(row * index_count + i) * inner + col] =
                    logical[(row * static_cast<std::size_t>(extent) +
                             static_cast<std::size_t>(ids[i])) *
                                inner +
                            col];
          const auto actual = half_test::read<Value>(output);
          ASSERT_EQ(actual.size(), expected.size());
          EXPECT_EQ(std::memcmp(actual.data(), expected.data(),
                                actual.size() * sizeof(Value)),
                    0);
          for (auto &id : ids)
            id = static_cast<std::int32_t>(extent - 1) - id;
        }
        index_data[offsets[index_count - 1]] =
            static_cast<std::int32_t>(extent);
        ASSERT_TRUE(
            indices.view().write(std::as_bytes(std::span(index_data))).ok());
        EXPECT_TRUE(submit().IsOutOfRange());
        const auto unchanged = half_test::read<Value>(output);
        EXPECT_EQ(std::memcmp(unchanged.data(), expected.data(),
                              unchanged.size() * sizeof(Value)),
                  0);
      }
    }
  }
}

TEST(GgmlCpuProvider, GatherAxesPreserveTypesAndValidateCurrentStridedIndices) {
  check_gather_axes<float>();
  check_gather_axes<hyle::simd::float16_t>();
  check_gather_axes<hyle::simd::bfloat16_t>();
  check_gather_axes<std::int32_t>();
}

TEST(GgmlCpuProvider, GatherEmptyIndicesNeedNoPayload) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description = dense(DataType::kFloat32, {3, 5});
  const auto index_desc = dense(DataType::kInt32, {0});
  const auto out_desc = dense(DataType::kFloat32, {3, 0});
  auto source = output_buffer(description);
  auto indices = output_buffer(index_desc);
  auto output = output_buffer(out_desc);
  const std::array inputs{
      ConstTensorView::create(description, source.view()).value(),
      ConstTensorView::create(index_desc, indices.view()).value()};
  const auto status =
      execute(context, Gather{.axis = -1}, inputs,
              TensorView::create(out_desc, output.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
}

template <class Value> void check_scatter_add() {
  const auto type = half_test::data_type<Value>();
  const auto base_desc = dense(type, {2, 3, 2});
  const auto index_desc = dense(DataType::kInt32, {2, 2});
  const auto update_desc = dense(type, {2, 2, 2, 2});
  std::vector<Value> base(12), updates(16);
  for (std::size_t i = 0; i < base.size(); ++i)
    base[i] = half_test::encode<Value>(10.0 + static_cast<double>(i));
  for (std::size_t i = 0; i < updates.size(); ++i)
    updates[i] = half_test::encode<Value>(1.0 + static_cast<double>(i % 4));
  std::array<std::int32_t, 4> ids{2, 0, 2, 1};
  auto base_buffer = half_test::buffer<Value>(base);
  auto index_buffer = half_test::buffer<std::int32_t>(ids);
  auto update_buffer = half_test::buffer<Value>(updates);
  auto output = output_buffer(base_desc);
  const std::array inputs{
      ConstTensorView::create(base_desc, base_buffer.view()).value(),
      ConstTensorView::create(index_desc, index_buffer.view()).value(),
      ConstTensorView::create(update_desc, update_buffer.view()).value()};
  auto output_view = TensorView::create(base_desc, output.view()).value();
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto prepared =
      prepare(context, Operation(Scatter{.axis = 1, .kind = ScatterKind::kAdd}),
              inputs, std::span(&output_view, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
  auto invocation = prepared->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  const auto submit = [&]() -> hyle::Status {
    auto completion =
        invocation->submit(queue, inputs, std::span(&output_view, 1));
    if (!completion)
      return std::move(completion).error();
    return completion->wait();
  };
  ASSERT_TRUE(submit().ok());
  auto expected = base;
  for (std::size_t outer = 0; outer < 2; ++outer) {
    std::array<double, 6> sums{};
    for (std::size_t row = 0; row < 3; ++row)
      for (std::size_t inner = 0; inner < 2; ++inner)
        sums[row * 2 + inner] =
            half_test::decode(base[(outer * 3 + row) * 2 + inner]);
    for (std::size_t update = 0; update < ids.size(); ++update)
      for (std::size_t inner = 0; inner < 2; ++inner)
        sums[static_cast<std::size_t>(ids[update]) * 2 + inner] +=
            half_test::decode(
                updates[(outer * ids.size() + update) * 2 + inner]);
    for (std::size_t row = 0; row < 3; ++row)
      for (std::size_t inner = 0; inner < 2; ++inner)
        expected[(outer * 3 + row) * 2 + inner] =
            half_test::encode<Value>(sums[row * 2 + inner]);
  }
  const auto actual = half_test::read<Value>(output);
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(std::memcmp(actual.data(), expected.data(),
                        actual.size() * sizeof(Value)),
            0);

  ids.back() = 3;
  ASSERT_TRUE(index_buffer.view().write(std::as_bytes(std::span(ids))).ok());
  EXPECT_TRUE(submit().IsOutOfRange());
  const auto unchanged = half_test::read<Value>(output);
  EXPECT_EQ(std::memcmp(unchanged.data(), expected.data(),
                        unchanged.size() * sizeof(Value)),
            0);
}

TEST(GgmlCpuProvider, ScatterAddUsesMatureDuplicateReductionAndChecksIndices) {
  check_scatter_add<float>();
  check_scatter_add<hyle::simd::float16_t>();
  check_scatter_add<hyle::simd::bfloat16_t>();
}

TEST(GgmlCpuProvider, ScatterAddWithEmptyIndicesCopiesTheBase) {
  const auto base_desc = dense(DataType::kFloat32, {2, 3});
  const auto index_desc = dense(DataType::kInt32, {0});
  const auto update_desc = dense(DataType::kFloat32, {2, 0});
  const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  auto base = half_test::buffer<float>(values);
  auto indices = output_buffer(index_desc);
  auto updates = output_buffer(update_desc);
  auto output = output_buffer(base_desc);
  const std::array inputs{
      ConstTensorView::create(base_desc, base.view()).value(),
      ConstTensorView::create(index_desc, indices.view()).value(),
      ConstTensorView::create(update_desc, updates.view()).value()};
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto status =
      execute(context, Scatter{.axis = -1, .kind = ScatterKind::kAdd}, inputs,
              TensorView::create(base_desc, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(half_test::read<float>(output),
            (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(GgmlCpuProvider, ColumnBroadcastKeepsOperandOrder) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description = dense(DataType::kFloat32, {2, 1, 3, 2, 1, 7});
  const auto column = dense(DataType::kFloat32, {2, 1, 3, 2, 1, 1});
  std::vector<float> values(84), scalars(12);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i + 4);
  for (std::size_t i = 0; i < scalars.size(); ++i)
    scalars[i] = static_cast<float>(i + 2);
  auto source = half_test::buffer<float>(values);
  auto scales = half_test::buffer<float>(scalars);
  auto output = output_buffer(description);
  for (const auto kind : {BinaryKind::kAdd, BinaryKind::kSubtract,
                          BinaryKind::kMultiply, BinaryKind::kDivide}) {
    for (const bool column_first : {false, true}) {
      const auto whole =
          ConstTensorView::create(description, source.view()).value();
      const auto part = ConstTensorView::create(column, scales.view()).value();
      const std::array inputs{column_first ? part : whole,
                              column_first ? whole : part};
      const auto status =
          execute(context, Binary{kind}, inputs,
                  TensorView::create(description, output.view()).value());
      ASSERT_TRUE(status.ok()) << status.message();
      const auto actual = half_test::read<float>(output);
      for (std::size_t i = 0; i < values.size(); ++i) {
        const float a = column_first ? scalars[i / 7] : values[i];
        const float b = column_first ? values[i] : scalars[i / 7];
        const float expected = kind == BinaryKind::kAdd        ? a + b
                               : kind == BinaryKind::kSubtract ? a - b
                               : kind == BinaryKind::kMultiply ? a * b
                                                               : a / b;
        EXPECT_FLOAT_EQ(actual[i], expected);
      }
    }
  }
}

template <class Value> void check_general_broadcast() {
  using Shape = std::vector<std::int64_t>;
  const std::array<std::pair<Shape, Shape>, 3> shapes{
      {{{2, 1, 3, 1, 2, 1}, {1, 3, 1, 2, 1, 5}},
       {{2, 1, 2, 1, 2, 1, 2, 1}, {1, 2, 1, 2, 1, 2, 1, 3}},
       {{3, 1, 2, 1}, {2, 1, 1, 4, 1, 5}}}};
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  for (const auto &[left_shape, right_shape] : shapes) {
    const auto left_desc =
        TensorDesc::dense(half_test::data_type<Value>(), left_shape).value();
    const auto right_desc =
        TensorDesc::dense(half_test::data_type<Value>(), right_shape).value();
    std::vector<Value> left(left_desc.element_count().value()),
        right(right_desc.element_count().value());
    for (std::size_t i = 0; i < left.size(); ++i)
      left[i] = half_test::encode<Value>(0.5 + static_cast<double>(i % 11) / 8);
    for (std::size_t i = 0; i < right.size(); ++i)
      right[i] =
          half_test::encode<Value>(0.75 + static_cast<double>(i % 7) / 8);
    auto left_buffer = half_test::buffer<Value>(left);
    auto right_buffer = half_test::buffer<Value>(right);
    const auto left_view =
        ConstTensorView::create(left_desc, left_buffer.view()).value();
    const auto right_view =
        ConstTensorView::create(right_desc, right_buffer.view()).value();
    for (const auto kind : {BinaryKind::kAdd, BinaryKind::kSubtract,
                            BinaryKind::kMultiply, BinaryKind::kDivide}) {
      for (const bool reverse : {false, true}) {
        const std::array descriptions{left_desc, right_desc};
        const auto description =
            infer_single_output(Operation(Binary{kind}), descriptions).value();
        const std::array inputs{reverse ? right_view : left_view,
                                reverse ? left_view : right_view};
        auto output = output_buffer(description);
        const auto status =
            execute(context, Binary{kind}, inputs,
                    TensorView::create(description, output.view()).value());
        ASSERT_TRUE(status.ok()) << status.message();
        const auto actual = half_test::read<Value>(output);
        for (std::size_t i = 0; i < actual.size(); ++i) {
          auto remainder = i;
          std::array<std::size_t, kMaxTensorRank> coordinates{};
          for (std::size_t d = description.rank(); d-- > 0;) {
            const auto extent =
                static_cast<std::size_t>(description.dimensions()[d]);
            coordinates[d] = remainder % extent;
            remainder /= extent;
          }
          const auto offset = [&](const TensorDesc &operand) {
            std::size_t index = 0;
            for (std::size_t d = 0; d < operand.rank(); ++d) {
              const auto extent =
                  static_cast<std::size_t>(operand.dimensions()[d]);
              index =
                  index * extent +
                  (extent == 1
                       ? 0
                       : coordinates[description.rank() - operand.rank() + d]);
            }
            return index;
          };
          const float l =
              static_cast<float>(half_test::decode(left[offset(left_desc)]));
          const float r =
              static_cast<float>(half_test::decode(right[offset(right_desc)]));
          const float a = reverse ? r : l, b = reverse ? l : r;
          const float expected = kind == BinaryKind::kAdd        ? a + b
                                 : kind == BinaryKind::kSubtract ? a - b
                                 : kind == BinaryKind::kMultiply ? a * b
                                                                 : a / b;
          if constexpr (std::is_same_v<Value, float>)
            EXPECT_NEAR(actual[i], expected, 2.e-6F);
          else
            half_test::expect_near(actual[i], expected);
        }
      }
    }
    const auto preserved_left = half_test::read<Value>(left_buffer);
    const auto preserved_right = half_test::read<Value>(right_buffer);
    EXPECT_EQ(std::memcmp(left.data(), preserved_left.data(),
                          left.size() * sizeof(Value)),
              0);
    EXPECT_EQ(std::memcmp(right.data(), preserved_right.data(),
                          right.size() * sizeof(Value)),
              0);
  }
}

TEST(GgmlCpuProvider, GeneralBroadcastPreservesHeadBatchAndOperandAxes) {
  check_general_broadcast<float>();
  check_general_broadcast<hyle::simd::float16_t>();
  check_general_broadcast<hyle::simd::bfloat16_t>();
}

template <class Value>
void check_reduction_axes(std::initializer_list<std::int64_t> shape,
                          bool strided = false) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto description = dense(half_test::data_type<Value>(), shape);
  if (strided) {
    std::vector<std::int64_t> strides(description.byte_strides().begin(),
                                      description.byte_strides().end());
    for (auto &stride : strides)
      stride *= 2;
    description = TensorDesc::strided_bytes(half_test::data_type<Value>(),
                                            description.dimensions(), strides)
                      .value();
  }
  const auto size = description.element_count().value();
  const std::size_t spacing = strided ? 2 : 1;
  std::vector<Value> values(size), physical(size * spacing);
  for (std::size_t i = 0; i < size; ++i) {
    values[i] = half_test::encode<Value>(
        static_cast<double>(static_cast<int>(i % 23) - 11) / 32);
    physical[i * spacing] = values[i];
  }
  auto source = half_test::buffer<Value>(physical);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  const auto end = std::size_t{1} << description.rank();
  for (std::size_t mask = description.rank() == 0 ? 0 : 1; mask < end; ++mask) {
    for (const bool keep : {false, true}) {
      std::vector<std::int32_t> requested;
      for (std::size_t d = description.rank(); d-- > 0;)
        if ((mask & (std::size_t{1} << d)) != 0)
          requested.push_back(
              static_cast<std::int32_t>(d) -
              (keep ? 0 : static_cast<std::int32_t>(description.rank())));
      const auto axes = requested.empty() || (mask == end - 1 && keep)
                            ? Axes::all()
                            : Axes::from(requested).value();
      for (const auto kind : {ReductionKind::kSum, ReductionKind::kMean}) {
        SCOPED_TRACE(::testing::Message()
                     << "rank=" << description.rank() << " axes=" << mask
                     << " keep=" << keep);
        const Reduce operation{.kind = kind,
                               .axes = axes,
                               .keep_dimensions = keep,
                               .accumulation_type = DataType::kFloat32};
        const auto out_desc = infer_single_output(Operation(operation),
                                                  std::span(&description, 1))
                                  .value();
        auto output = output_buffer(out_desc);
        const auto status =
            execute(context, operation, std::span(&input, 1),
                    TensorView::create(out_desc, output.view()).value());
        ASSERT_TRUE(status.ok()) << status.message();
        const auto actual = half_test::read<Value>(output);
        std::vector<double> sums(actual.size());
        std::vector<std::size_t> counts(actual.size());
        for (std::size_t i = 0; i < size; ++i) {
          auto remainder = i;
          std::array<std::size_t, kMaxTensorRank> coordinates{};
          for (std::size_t d = description.rank(); d-- > 0;) {
            const auto extent =
                static_cast<std::size_t>(description.dimensions()[d]);
            coordinates[d] = remainder % extent;
            remainder /= extent;
          }
          std::size_t output_index = 0;
          for (std::size_t d = 0; d < description.rank(); ++d)
            if ((mask & (std::size_t{1} << d)) == 0)
              output_index = output_index * static_cast<std::size_t>(
                                                description.dimensions()[d]) +
                             coordinates[d];
          sums[output_index] += half_test::decode(values[i]);
          ++counts[output_index];
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
          const auto expected = kind == ReductionKind::kSum ? sums[i]
                                : counts[i] == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : sums[i] / static_cast<double>(counts[i]);
          if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(half_test::decode(actual[i])));
          else if constexpr (std::is_same_v<Value, float>)
            EXPECT_NEAR(actual[i], expected, 2.e-6);
          else
            half_test::expect_near(actual[i], expected);
        }
      }
    }
  }
}

template <class Value> void check_all_reduction_axes() {
  check_reduction_axes<Value>({2, 3, 2, 2, 3, 5});
  check_reduction_axes<Value>({2, 3, 5}, true);
}
TEST(GgmlCpuProvider, SelectedReductionsCoverAllRankSixAxisSets) {
  check_all_reduction_axes<float>();
  check_all_reduction_axes<hyle::simd::float16_t>();
  check_all_reduction_axes<hyle::simd::bfloat16_t>();
}

template <class Value> void check_empty_reductions() {
  check_reduction_axes<Value>({});
  check_reduction_axes<Value>({0});
  check_reduction_axes<Value>({2, 0, 3});
  check_reduction_axes<Value>({0, 2, 3});
}
TEST(GgmlCpuProvider, EmptyReductionsPublishIdentitiesAndScalarValues) {
  check_empty_reductions<float>();
  check_empty_reductions<hyle::simd::float16_t>();
  check_empty_reductions<hyle::simd::bfloat16_t>();
}

TEST(GgmlCpuProvider, PersistentIndependentSlotsHonorExecutorAndClose) {
  struct Executor final {
    std::size_t calls = 0;
    std::thread::id thread;
  } executor;
  const ParallelExecutorRef ref{&executor, 7,
                                [](void *state, std::size_t count,
                                   ParallelTaskFn task, void *data) noexcept {
                                  auto &record =
                                      *static_cast<Executor *>(state);
                                  EXPECT_EQ(count, 1U);
                                  ++record.calls;
                                  record.thread = std::this_thread::get_id();
                                  task(data, 0, count);
                                }};
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const std::array providers{
        ggml_cpu::provider({.executor = ref, .native_threads = 2})};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                        .providers = providers})
            .value();
    const auto description = dense(DataType::kFloat32, {17, 31});
    auto prepared = prepare(context, Unary{UnaryKind::kNeg}, description);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_TRUE(prepared->runtime_properties().uses_private_threads);
    EXPECT_TRUE(prepared->runtime_properties().supports_concurrent_invocations);
    auto first = prepared->create_invocation();
    auto second = prepared->create_invocation();
    ASSERT_TRUE(first) << first.error().message();
    ASSERT_TRUE(second) << second.error().message();
    auto source = output_buffer(description);
    std::array outputs{output_buffer(description), output_buffer(description)};
    const auto input =
        ConstTensorView::create(description, source.view()).value();
    auto queue = context.create_queue().value();
    const auto allocation_count = stats.snapshot().total_allocs;
    for (std::size_t run = 0; run < 4; ++run) {
      std::vector<float> values(527, static_cast<float>(run + 1));
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
      auto output =
          TensorView::create(description, outputs[run % 2].view()).value();
      auto completion =
          (run % 2 == 0 ? *first : *second)
              .submit(queue, std::span(&input, 1), std::span(&output, 1));
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      for (float value : half_test::read<float>(outputs[run % 2]))
        EXPECT_FLOAT_EQ(value, -static_cast<float>(run + 1));
      if (run != 0)
        for (float value : half_test::read<float>(outputs[(run + 1) % 2]))
          EXPECT_FLOAT_EQ(value, -static_cast<float>(run));
    }
    EXPECT_EQ(stats.snapshot().total_allocs, allocation_count);
    EXPECT_EQ(executor.calls, 4U);
    EXPECT_EQ(executor.thread, std::this_thread::get_id());
    ASSERT_TRUE(context.close().ok());
    auto output = TensorView::create(description, outputs[0].view()).value();
    EXPECT_FALSE(
        first->submit(queue, std::span(&input, 1), std::span(&output, 1)));
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuProvider, DeclinesUnqualifiedDomainsAndInvalidThreadBudget) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description = dense(DataType::kFloat32, {2, 7});
  for (auto kind :
       {UnaryKind::kRelu, UnaryKind::kSigmoid, UnaryKind::kSilu,
        UnaryKind::kGelu, UnaryKind::kGeluTanh, UnaryKind::kSoftplus}) {
    auto result = prepare(context, Unary{kind}, description);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
  }
  auto softmax = prepare(context, Softmax{}, description);
  ASSERT_FALSE(softmax);
  EXPECT_TRUE(softmax.error().IsUnimplemented());
  for (auto threads : {0U, std::numeric_limits<std::uint32_t>::max()}) {
    auto invalid = Context::create(
        cpu::device(), ggml_cpu::provider({.native_threads = threads}));
    ASSERT_FALSE(invalid);
    EXPECT_TRUE(invalid.error().IsInvalidArgument());
  }
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    hyle::mem::QuotaAllocator quota(stats.get(), 0);
    const std::array providers{ggml_cpu::provider()};
    auto denied =
        Context::create(cpu::device(), {.metadata_allocator = quota.get(),
                                        .providers = providers});
    ASSERT_FALSE(denied);
    EXPECT_TRUE(denied.error().IsResourceExhausted());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuProvider, InvocationAllocationFailureRollsBackAndCanRecover) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  hyle::mem::QuotaAllocator quota(stats.get(), 4 * 1024 * 1024);
  {
    const std::array providers{ggml_cpu::provider({.native_threads = 2})};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = quota.get(),
                                        .providers = providers})
            .value();
    const auto description = dense(DataType::kFloat16, {17, 31});
    auto prepared = prepare(context, RmsNorm{}, description).value();
    const auto before = quota.used();
    std::size_t peak = 0;
    {
      auto invocation = prepared.create_invocation();
      ASSERT_TRUE(invocation) << invocation.error().message();
      peak = quota.peak() - before;
    }
    ASSERT_EQ(quota.used(), before);
    for (const auto available : {std::size_t{0}, peak / 2, peak - 1}) {
      SCOPED_TRACE(available);
      auto reservation = quota.try_reserve_bytes(quota.remaining() - available);
      ASSERT_TRUE(reservation);
      auto denied = prepared.create_invocation();
      ASSERT_FALSE(denied);
      EXPECT_TRUE(denied.error().IsResourceExhausted());
      reservation->reset();
      EXPECT_EQ(quota.used(), before);
    }
    EXPECT_TRUE(prepared.create_invocation());
  }
  EXPECT_TRUE(stats.balanced());
}
TEST(GgmlCpuProvider, StandardScanHandlesAxisStridesAndCheckedOverflow) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description =
      TensorDesc::strided_bytes(DataType::kInt32,
                                std::array<std::int64_t, 3>{2, 7, 3},
                                std::array<std::int64_t, 3>{140, 19, 5})
          .value();
  std::vector<std::byte> bytes(description.required_bytes().value());
  std::array<std::int32_t, 42> logical{};
  for (std::size_t a = 0; a < 2; ++a)
    for (std::size_t b = 0; b < 7; ++b)
      for (std::size_t c = 0; c < 3; ++c) {
        const auto value =
            static_cast<std::int32_t>(b) - 2 + static_cast<std::int32_t>(c);
        logical[(a * 7 + b) * 3 + c] = value;
        std::memcpy(bytes.data() + a * 140 + b * 19 + c * 5, &value,
                    sizeof(value));
      }
  auto source =
      Buffer::create(hyle::mem::system_resource(), bytes.size(), 64).value();
  ASSERT_TRUE(source.view().write(bytes).ok());
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  const auto output_desc = dense(DataType::kInt32, {2, 7, 3});
  auto output = output_buffer(output_desc);
  for (bool exclusive : {false, true}) {
    auto status = execute(
        context, Scan{.axis = 1, .exclusive = exclusive}, std::span(&input, 1),
        TensorView::create(output_desc, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<std::int32_t>(output);
    for (std::size_t a = 0; a < 2; ++a)
      for (std::size_t c = 0; c < 3; ++c) {
        std::int64_t sum = 0;
        for (std::size_t b = 0; b < 7; ++b) {
          const auto index = (a * 7 + b) * 3 + c;
          if (exclusive)
            EXPECT_EQ(actual[index], sum);
          sum += logical[index];
          if (!exclusive)
            EXPECT_EQ(actual[index], sum);
        }
      }
  }
  const auto small = dense(DataType::kInt32, {2, 3});
  const std::array<std::array<std::int32_t, 6>, 2> overflows{
      {{1, 2, 3, INT32_MAX, 1, -1}, {1, 2, 3, INT32_MAX - 1, 0, 2}}};
  const std::array<std::int32_t, 6> sentinel{91, 92, 93, 94, 95, 96};
  auto preserved = half_test::buffer<std::int32_t>(sentinel);
  for (const auto &values : overflows) {
    auto bad = half_test::buffer<std::int32_t>(values);
    const auto bad_input = ConstTensorView::create(small, bad.view()).value();
    for (bool exclusive : {false, true}) {
      auto status = execute(
          context, Scan{.exclusive = exclusive}, std::span(&bad_input, 1),
          TensorView::create(small, preserved.view()).value());
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_EQ(half_test::read<std::int32_t>(preserved),
                std::vector<std::int32_t>(sentinel.begin(), sentinel.end()));
    }
  }
}

template <class T> void standard_topk() {
  const auto type = half_test::data_type<T>();
  const auto description = dense(type, {1, 2, 1, 9, 1, 3});
  const std::array values{std::numeric_limits<double>::quiet_NaN(),
                          -0.0,
                          0.0,
                          3.0,
                          3.0,
                          -std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN(),
                          -5.0};
  const std::array<std::int32_t, 9> descending{6, 3, 4, 1, 2, 8, 5, 0, 7};
  const std::array<std::int32_t, 9> ascending{5, 8, 1, 2, 3, 4, 6, 0, 7};
  std::vector<T> payload(54);
  for (std::size_t outer = 0; outer < 2; ++outer)
    for (std::size_t i = 0; i < 9; ++i)
      for (std::size_t inner = 0; inner < 3; ++inner)
        payload[(outer * 9 + i) * 3 + inner] = half_test::encode<T>(values[i]);
  auto source = half_test::buffer<T>(payload);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  auto context =
      Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 2}))
          .value();
  auto queue = context.create_queue().value();
  for (std::size_t count : {1U, 3U, 9U})
    for (bool largest : {false, true}) {
      auto prepared =
          prepare(context, TopK{.count = count, .axis = 3, .largest = largest},
                  description);
      ASSERT_TRUE(prepared) << prepared.error().message();
      EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
      EXPECT_FALSE(prepared->runtime_properties().uses_private_threads);
      auto first = output_buffer(prepared->output_description(0).value());
      auto second = output_buffer(prepared->output_description(1).value());
      const std::array outputs{
          TensorView::create(prepared->output_description(0).value(),
                             first.view())
              .value(),
          TensorView::create(prepared->output_description(1).value(),
                             second.view())
              .value()};
      auto invocation = prepared->create_invocation().value();
      auto completion = invocation.submit(queue, std::span(&input, 1), outputs);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual_values = half_test::read<T>(first);
      const auto actual_indices = half_test::read<std::int32_t>(second);
      const auto &expected = largest ? descending : ascending;
      for (std::size_t outer = 0; outer < 2; ++outer)
        for (std::size_t i = 0; i < count; ++i)
          for (std::size_t inner = 0; inner < 3; ++inner) {
            const auto index = (outer * count + i) * 3 + inner;
            EXPECT_EQ(actual_indices[index], expected[i]);
            const auto &original =
                payload[(outer * 9 + expected[i]) * 3 + inner];
            EXPECT_EQ(std::memcmp(&actual_values[index], &original, sizeof(T)),
                      0);
          }
    }
}
TEST(GgmlCpuProvider, StandardSelectionPreservesStableOrderAndStorageBits) {
  standard_topk<float>();
  standard_topk<hyle::simd::float16_t>();
  standard_topk<hyle::simd::bfloat16_t>();
}

TEST(GgmlCpuProvider, StandardScanReplaysWithoutAllocationsAndHonorsQuota) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  hyle::mem::QuotaAllocator quota(stats.get(), 1 << 20);
  {
    const std::array providers{ggml_cpu::provider({.native_threads = 2})};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = quota.get(),
                                        .providers = providers})
            .value();
    const auto description = dense(DataType::kFloat32, {2, 7});
    auto prepared =
        prepare(context, Scan{.exclusive = true}, description).value();
    EXPECT_EQ(prepared.provider().id, ggml_cpu::kProviderId);
    EXPECT_FALSE(prepared.runtime_properties().uses_private_threads);
    EXPECT_EQ(prepared.invocation_private_bytes(), 0U);
    {
      auto reserved = quota.try_reserve_bytes(quota.remaining());
      ASSERT_TRUE(reserved);
      auto denied = prepared.create_invocation();
      ASSERT_FALSE(denied);
      EXPECT_TRUE(denied.error().IsResourceExhausted());
    }
    auto first = prepared.create_invocation().value();
    auto second = prepared.create_invocation().value();
    auto source = output_buffer(description);
    std::array outputs{output_buffer(description), output_buffer(description)};
    const auto input =
        ConstTensorView::create(description, source.view()).value();
    auto queue = context.create_queue().value();
    const auto before = stats.snapshot();
    for (std::size_t run = 0; run < 4; ++run) {
      const std::vector<float> values(14, static_cast<float>(run + 1));
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
      const auto output =
          TensorView::create(description, outputs[run % 2].view()).value();
      auto completion =
          (run % 2 == 0 ? first : second)
              .submit(queue, std::span(&input, 1), std::span(&output, 1));
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<float>(outputs[run % 2]);
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_FLOAT_EQ(actual[i], static_cast<float>((i % 7) * (run + 1)));
      if (run > 0) {
        const auto retained = half_test::read<float>(outputs[(run + 1) % 2]);
        for (std::size_t i = 0; i < retained.size(); ++i)
          EXPECT_FLOAT_EQ(retained[i], static_cast<float>((i % 7) * run));
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuProvider, StandardSelectionKeepsAdjacentLargeIntegersDistinct) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description = dense(DataType::kInt32, {6});
  const std::array<std::int32_t, 6> payload{
      INT32_MAX, INT32_MAX - 1, INT32_MIN, 0, 0, INT32_MAX};
  const std::array<std::int32_t, 3> expected{0, 5, 1};
  auto source = half_test::buffer<std::int32_t>(payload);
  auto prepared = prepare(context, TopK{.count = 3}, description).value();
  auto values = output_buffer(prepared.output_description(0).value());
  auto indices = output_buffer(prepared.output_description(1).value());
  auto invocation = prepared.create_invocation().value();
  auto queue = context.create_queue().value();
  auto done = invocation.submit(
      queue, ConstTensorView::create(description, source.view()).value(),
      TensorView::create(prepared.output_description(0).value(), values.view())
          .value(),
      TensorView::create(prepared.output_description(1).value(), indices.view())
          .value());
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  EXPECT_EQ(half_test::read<std::int32_t>(indices),
            std::vector<std::int32_t>(expected.begin(), expected.end()));
  EXPECT_EQ(half_test::read<std::int32_t>(values),
            (std::vector<std::int32_t>{INT32_MAX, INT32_MAX, INT32_MAX - 1}));
}

TEST(GgmlCpuProvider, EmptyScanHasNoNativeEpisodeOrPayloadAccess) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto description = dense(DataType::kInt32, {2, 0, 3});
  auto source = output_buffer(description), output = output_buffer(description);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  auto status = execute(context, Scan{.axis = 1}, std::span(&input, 1),
                        TensorView::create(description, output.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST(GgmlCpuProvider, MatureLinearRopeAndHadamardUseOrdinaryEntry) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  {
    using F16 = hyle::simd::float16_t;
    const auto left_desc = dense(DataType::kFloat16, {2, 3});
    const auto right_desc = dense(DataType::kFloat16, {3, 2});
    const auto output_desc = dense(DataType::kFloat16, {2, 2});
    const std::array<float, 6> left_values{1, 2, 3, 4, 5, 6};
    const std::array<float, 6> right_values{1, 2, 3, 4, 5, 6};
    std::array<F16, 6> left_half{}, right_half{};
    for (std::size_t i = 0; i < left_half.size(); ++i) {
      left_half[i] = half_test::encode<F16>(left_values[i]);
      right_half[i] = half_test::encode<F16>(right_values[i]);
    }
    auto left = half_test::buffer<F16>(left_half);
    auto right = half_test::buffer<F16>(right_half);
    auto output = output_buffer(output_desc);
    const std::array inputs{
        ConstTensorView::create(left_desc, left.view()).value(),
        ConstTensorView::create(right_desc, right.view()).value()};
    ASSERT_TRUE(execute(context, MatMul{}, inputs,
                        TensorView::create(output_desc, output.view()).value())
                    .ok());
    const auto actual = half_test::read<F16>(output);
    const std::array expected{22.0F, 28.0F, 49.0F, 64.0F};
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
  {
    using B16 = hyle::simd::bfloat16_t;
    const auto left_desc = dense(DataType::kBFloat16, {2, 3});
    const auto right_desc = dense(DataType::kBFloat16, {3, 2});
    const auto output_desc = dense(DataType::kBFloat16, {2, 2});
    std::array<B16, 6> left_values{}, right_values{};
    for (std::size_t i = 0; i < left_values.size(); ++i) {
      left_values[i] = half_test::encode<B16>(static_cast<double>(i + 1));
      right_values[i] = half_test::encode<B16>(static_cast<double>(i + 1));
    }
    auto left = half_test::buffer<B16>(left_values);
    auto right = half_test::buffer<B16>(right_values);
    auto output = output_buffer(output_desc);
    const std::array inputs{
        ConstTensorView::create(left_desc, left.view()).value(),
        ConstTensorView::create(right_desc, right.view()).value()};
    ASSERT_TRUE(execute(context, MatMul{}, inputs,
                        TensorView::create(output_desc, output.view()).value())
                    .ok());
    const auto actual = half_test::read<B16>(output);
    const std::array expected{22.0F, 28.0F, 49.0F, 64.0F};
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
  {
    const auto description = dense(DataType::kFloat32, {1, 4});
    auto source = half_test::buffer<float>(std::array<float, 4>{1, 2, 3, 4});
    auto output = output_buffer(description);
    const auto input =
        ConstTensorView::create(description, source.view()).value();
    for (const bool inverse : {false, true}) {
      ASSERT_TRUE(
          execute(context, Hadamard{.inverse = inverse}, std::span(&input, 1),
                  TensorView::create(description, output.view()).value())
              .ok());
      EXPECT_EQ(half_test::read<float>(output),
                (std::vector<float>{5, -1, -2, 0}));
    }
  }
  {
    using F16 = hyle::simd::float16_t;
    const auto data_desc = dense(DataType::kFloat16, {1, 2, 1, 4});
    const auto position_desc = dense(DataType::kInt32, {2});
    const auto frequency_desc = dense(DataType::kFloat32, {2});
    std::array<F16, 8> half_data{};
    for (std::size_t i = 0; i < half_data.size(); ++i)
      half_data[i] = half_test::encode<F16>(static_cast<float>(i + 1));
    auto data = half_test::buffer<F16>(half_data);
    auto positions =
        half_test::buffer<std::int32_t>(std::array<std::int32_t, 2>{0, 2});
    auto frequencies =
        half_test::buffer<float>(std::array<float, 2>{0.5F, 0.25F});
    auto output = output_buffer(data_desc);
    const std::array inputs{
        ConstTensorView::create(data_desc, data.view()).value(),
        ConstTensorView::create(position_desc, positions.view()).value(),
        ConstTensorView::create(frequency_desc, frequencies.view()).value()};
    ASSERT_TRUE(execute(context,
                        RoPE{.rotary_dimensions = 4,
                             .sequence_axis = 1,
                             .layout = RopeLayout::kInterleaved},
                        inputs,
                        TensorView::create(data_desc, output.view()).value())
                    .ok());
    const auto actual = half_test::read<F16>(output);
    const std::array expected{1.0F,
                              2.0F,
                              3.0F,
                              4.0F,
                              5.0F * std::cos(1.0F) - 6.0F * std::sin(1.0F),
                              5.0F * std::sin(1.0F) + 6.0F * std::cos(1.0F),
                              7.0F * std::cos(0.5F) - 8.0F * std::sin(0.5F),
                              7.0F * std::sin(0.5F) + 8.0F * std::cos(0.5F)};
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
  {
    using F16 = hyle::simd::float16_t;
    const auto data_desc = dense(DataType::kFloat16, {1, 2, 1, 8});
    const auto position_desc = dense(DataType::kInt32, {2});
    const auto frequency_desc = dense(DataType::kFloat32, {2});
    std::array<F16, 16> half_data{};
    for (std::size_t i = 0; i < half_data.size(); ++i)
      half_data[i] = half_test::encode<F16>(static_cast<float>(i + 1));
    auto data = half_test::buffer<F16>(half_data);
    auto positions =
        half_test::buffer<std::int32_t>(std::array<std::int32_t, 2>{0, 2});
    auto frequencies =
        half_test::buffer<float>(std::array<float, 2>{0.5F, 0.25F});
    auto output = output_buffer(data_desc);
    const std::array inputs{
        ConstTensorView::create(data_desc, data.view()).value(),
        ConstTensorView::create(position_desc, positions.view()).value(),
        ConstTensorView::create(frequency_desc, frequencies.view()).value()};
    ASSERT_TRUE(execute(context,
                        RoPE{.rotary_dimensions = 4,
                             .sequence_axis = 1,
                             .layout = RopeLayout::kHalfSplit},
                        inputs,
                        TensorView::create(data_desc, output.view()).value())
                    .ok());
    const auto actual = half_test::read<F16>(output);
    std::array<double, 16> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = static_cast<double>(i + 1);
    expected[8] = 9.0 * std::cos(1.0) - 11.0 * std::sin(1.0);
    expected[9] = 10.0 * std::cos(0.5) - 12.0 * std::sin(0.5);
    expected[10] = 9.0 * std::sin(1.0) + 11.0 * std::cos(1.0);
    expected[11] = 10.0 * std::sin(0.5) + 12.0 * std::cos(0.5);
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
}

TEST(GgmlCpuProvider,
     BatchedMatMulBroadcastsNativeTransposedWeightsWithoutCopy) {
  using F16 = hyle::simd::float16_t;
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto left_desc = dense(DataType::kFloat16, {2, 2, 3});
  std::array<F16, 12> left_values{};
  for (std::size_t i = 0; i < left_values.size(); ++i)
    left_values[i] = half_test::encode<F16>(static_cast<double>(i % 6 + 1));
  auto left = half_test::buffer<F16>(left_values);
  for (const bool batched_weights : {false, true}) {
    SCOPED_TRACE(batched_weights);
    const auto physical_desc = batched_weights
                                   ? dense(DataType::kFloat16, {2, 2, 3})
                                   : dense(DataType::kFloat16, {2, 3});
    const auto right_desc = physical_desc.transposed().value();
    std::vector<F16> weights(physical_desc.element_count().value());
    const std::size_t batches = batched_weights ? 2 : 1;
    for (std::size_t batch = 0; batch < batches; ++batch)
      for (std::size_t column = 0; column < 2; ++column)
        for (std::size_t reduction = 0; reduction < 3; ++reduction) {
          const double value =
              static_cast<double>(1 + batch * 6 + reduction * 2 + column);
          weights[(batch * 2 + column) * 3 + reduction] =
              half_test::encode<F16>(value);
        }
    auto right = half_test::buffer<F16>(weights);
    const std::array descriptions{left_desc, right_desc};
    const auto output_desc =
        infer_single_output(Operation(MatMul{}), descriptions).value();
    auto output = output_buffer(output_desc);
    const std::array inputs{
        ConstTensorView::create(left_desc, left.view()).value(),
        ConstTensorView::create(right_desc, right.view()).value()};
    auto status =
        execute(context, MatMul{}, inputs,
                TensorView::create(output_desc, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<F16>(output);
    for (std::size_t batch = 0; batch < 2; ++batch)
      for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t column = 0; column < 2; ++column) {
          double expected = 0;
          for (std::size_t reduction = 0; reduction < 3; ++reduction) {
            const auto activation = half_test::decode(
                left_values[(batch * 2 + row) * 3 + reduction]);
            const auto weight_batch = batched_weights ? batch : 0;
            const auto weight = half_test::decode(
                weights[(weight_batch * 2 + column) * 3 + reduction]);
            expected += activation * weight;
          }
          half_test::expect_near(actual[(batch * 2 + row) * 2 + column],
                                 expected);
        }
  }
}

TEST(GgmlCpuProvider, HalfMatMulCoversDecodeTailAndRankFourBroadcast) {
  using F16 = hyle::simd::float16_t;
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  for (const std::int64_t rows : {1, 17, 128}) {
    SCOPED_TRACE(rows);
    const auto left_desc = dense(DataType::kFloat16, {rows, 5});
    const auto right_desc = dense(DataType::kFloat16, {5, 7});
    const auto output_desc = dense(DataType::kFloat16, {rows, 7});
    std::vector<F16> left_values(static_cast<std::size_t>(rows * 5),
                                 half_test::encode<F16>(1));
    std::vector<F16> right_values(35, half_test::encode<F16>(1));
    auto left = half_test::buffer<F16>(left_values);
    auto right = half_test::buffer<F16>(right_values);
    auto output = output_buffer(output_desc);
    const std::array inputs{
        ConstTensorView::create(left_desc, left.view()).value(),
        ConstTensorView::create(right_desc, right.view()).value()};
    ASSERT_TRUE(execute(context, MatMul{}, inputs,
                        TensorView::create(output_desc, output.view()).value())
                    .ok());
    for (const auto value : half_test::read<F16>(output))
      half_test::expect_near(value, 5);
  }

  const auto left_desc = dense(DataType::kFloat16, {2, 3, 2, 3});
  const auto right_desc = dense(DataType::kFloat16, {1, 3, 3, 2});
  const auto output_desc = dense(DataType::kFloat16, {2, 3, 2, 2});
  std::vector<F16> left_values(left_desc.element_count().value(),
                               half_test::encode<F16>(1));
  std::vector<F16> right_values(right_desc.element_count().value(),
                                half_test::encode<F16>(1));
  auto left = half_test::buffer<F16>(left_values);
  auto right = half_test::buffer<F16>(right_values);
  auto output = output_buffer(output_desc);
  const std::array inputs{
      ConstTensorView::create(left_desc, left.view()).value(),
      ConstTensorView::create(right_desc, right.view()).value()};
  ASSERT_TRUE(execute(context, MatMul{}, inputs,
                      TensorView::create(output_desc, output.view()).value())
                  .ok());
  for (const auto value : half_test::read<F16>(output))
    half_test::expect_near(value, 3);
}

TEST(GgmlCpuProvider, MatureAttentionHandlesHead256GqaDecode) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto query_desc = dense(DataType::kFloat32, {1, 1, 8, 256});
  const auto kv_desc = dense(DataType::kFloat32, {1, 1, 2, 256});
  std::vector<float> query(query_desc.element_count().value(), 0.03125F);
  std::vector<float> key(kv_desc.element_count().value(), 0.0625F);
  std::vector<float> value(kv_desc.element_count().value());
  for (std::size_t head = 0; head < 2; ++head)
    for (std::size_t depth = 0; depth < 256; ++depth)
      value[head * 256 + depth] =
          static_cast<float>(head * 1000 + depth) / 256.0F;
  auto q = half_test::buffer<float>(query);
  auto k = half_test::buffer<float>(key);
  auto v = half_test::buffer<float>(value);
  auto output = output_buffer(query_desc);
  const std::array inputs{ConstTensorView::create(query_desc, q.view()).value(),
                          ConstTensorView::create(kv_desc, k.view()).value(),
                          ConstTensorView::create(kv_desc, v.view()).value()};
  const ScaledDotProductAttention attention{
      .layout = AttentionLayout::kBatchSequenceHeadDepth,
      .scale = 0.0625F,
      .causal = true,
      .accumulation_type = DataType::kFloat32};
  ASSERT_TRUE(execute(context, attention, inputs,
                      TensorView::create(query_desc, output.view()).value())
                  .ok());
  const auto actual = half_test::read<float>(output);
  for (std::size_t head = 0; head < 8; ++head)
    for (std::size_t depth = 0; depth < 256; ++depth)
      EXPECT_NEAR(actual[head * 256 + depth], value[(head / 4) * 256 + depth],
                  1.0e-5F)
          << head << ':' << depth;
}

template <class T> void causal_gqa_prefill() {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  constexpr std::size_t query_length = 2, key_length = 3;
  const auto query_desc =
      dense(half_test::data_type<T>(), {1, query_length, 8, 256});
  const auto kv_desc =
      dense(half_test::data_type<T>(), {1, key_length, 2, 256});
  std::vector<T> query(query_desc.element_count().value(),
                       half_test::encode<T>(0));
  std::vector<T> key(kv_desc.element_count().value(), half_test::encode<T>(0));
  std::vector<T> value(kv_desc.element_count().value());
  for (std::size_t position = 0; position < key_length; ++position)
    for (std::size_t head = 0; head < 2; ++head)
      for (std::size_t depth = 0; depth < 256; ++depth)
        value[(position * 2 + head) * 256 + depth] = half_test::encode<T>(
            static_cast<double>(position * 4 + head * 2 + depth % 2));
  auto q = half_test::buffer<T>(query);
  auto k = half_test::buffer<T>(key);
  auto v = half_test::buffer<T>(value);
  auto output = output_buffer(query_desc);
  const std::array inputs{ConstTensorView::create(query_desc, q.view()).value(),
                          ConstTensorView::create(kv_desc, k.view()).value(),
                          ConstTensorView::create(kv_desc, v.view()).value()};
  const ScaledDotProductAttention attention{
      .layout = AttentionLayout::kBatchSequenceHeadDepth,
      .scale = 0.0625F,
      .causal = true,
      .accumulation_type = DataType::kFloat32};
  ASSERT_TRUE(execute(context, attention, inputs,
                      TensorView::create(query_desc, output.view()).value())
                  .ok());
  const auto actual = half_test::read<T>(output);
  for (std::size_t position = 0; position < query_length; ++position)
    for (std::size_t head = 0; head < 8; ++head)
      for (std::size_t depth = 0; depth < 256; ++depth) {
        const auto kv_head = head / 4;
        double expected = 0;
        for (std::size_t key_position = 0;
             key_position <= position + key_length - query_length;
             ++key_position)
          expected +=
              static_cast<double>(key_position * 4 + kv_head * 2 + depth % 2);
        expected /=
            static_cast<double>(position + key_length - query_length + 1);
        half_test::expect_near(actual[(position * 8 + head) * 256 + depth],
                               expected);
      }
}

TEST(GgmlCpuProvider, MatureAttentionAppliesCausalMaskForHalfPrefill) {
  causal_gqa_prefill<hyle::simd::float16_t>();
  causal_gqa_prefill<hyle::simd::bfloat16_t>();
}
} // namespace
