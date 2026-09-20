#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <thread>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/quota_allocator.h"
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
} // namespace
