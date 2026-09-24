#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>
#include <type_traits>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/stats.h"
#include "hyle/simd/dtype.h"
#include "provider_internal.h"

namespace {
using namespace hyle::compute;
using hyle::mem::Buffer;

template <class T> constexpr DataType type() {
  if constexpr (std::is_same_v<T, float>)
    return DataType::kFloat32;
  else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
    return DataType::kFloat16;
  else
    return DataType::kBFloat16;
}
template <class T> T encode(float value) {
  if constexpr (std::is_same_v<T, float>)
    return value;
  else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
    return hyle::simd::f32_to_f16(value);
  else
    return hyle::simd::f32_to_bf16(value);
}
template <class T> float decode(T value) {
  if constexpr (std::is_same_v<T, float>)
    return value;
  else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
    return hyle::simd::f16_to_f32(value);
  else
    return hyle::simd::bf16_to_f32(value);
}
struct Tensor final {
  Buffer buffer;
  TensorView view;
};
Tensor tensor(TensorDesc description) {
  auto buffer = Buffer::create(hyle::mem::system_resource(),
                               description.required_bytes().value(), 64)
                    .value();
  auto view = TensorView::create(description, buffer.view()).value();
  return {std::move(buffer), view};
}
Buffer workspace(const PreparedOperation &prepared) {
  const auto requirement = prepared.workspace_requirement();
  if (requirement.size == 0)
    return {};
  return Buffer::create(hyle::mem::system_resource(), requirement.size,
                        requirement.alignment)
      .value();
}

template <class T, class Parameter>
void check_norm(bool rms, std::int64_t rows, std::int64_t width,
                std::size_t parameters, bool alias = false,
                ParallelExecutorRef executor = {}) {
  SCOPED_TRACE(testing::Message()
               << rms << ":" << rows << "x" << width
               << " parameters=" << parameters << " alias=" << alias);
  const auto description =
      TensorDesc::dense(type<T>(), std::array{rows, width}).value();
  const auto parameter =
      TensorDesc::dense(type<Parameter>(), std::array{width}).value();
  std::vector<T> values(static_cast<std::size_t>(rows * width));
  std::vector<Parameter> scales(static_cast<std::size_t>(width));
  std::vector<Parameter> biases(static_cast<std::size_t>(width));
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] = encode<T>(
        static_cast<float>(static_cast<int>(index % 37) - 18) * 0.125F);
  for (std::size_t index = 0; index < scales.size(); ++index) {
    scales[index] =
        encode<Parameter>(0.5F + static_cast<float>(index % 5) * 0.125F);
    biases[index] = encode<Parameter>(static_cast<float>(index % 7) * 0.0625F);
  }
  auto input = tensor(description), output = tensor(description);
  auto scale = tensor(parameter), bias = tensor(parameter);
  ASSERT_TRUE(input.buffer.view().write(std::as_bytes(std::span(values))).ok());
  ASSERT_TRUE(scale.buffer.view().write(std::as_bytes(std::span(scales))).ok());
  ASSERT_TRUE(bias.buffer.view().write(std::as_bytes(std::span(biases))).ok());
  const std::array inputs{input.view.as_const(), scale.view.as_const(),
                          bias.view.as_const()};
  const std::array outputs{alias ? input.view : output.view};
  auto context =
      Context::create(cpu::device(), onednn::provider({.executor = executor}))
          .value();
  const Operation operation =
      rms ? Operation(RmsNorm{}) : Operation(LayerNorm{});
  auto result = prepare(context, operation,
                        std::span(inputs).first(parameters + 1), outputs);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(result->provider().id, onednn::kProviderId);
  EXPECT_EQ(result->runtime_properties().replay_class,
            ProviderReplayClass::kManaged);
  EXPECT_FALSE(result->invocation_private_bytes().has_value());
  auto scratch = workspace(*result);
  auto invocation = result->create_invocation(scratch.view()).value();
  auto queue = context.create_queue().value();
  auto completion = invocation.submit(
      queue, std::span(inputs).first(parameters + 1), outputs);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  std::vector<T> actual(values.size());
  ASSERT_TRUE(outputs[0]
                  .storage()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  for (std::int64_t row = 0; row < rows; ++row) {
    double mean = 0;
    if (!rms) {
      for (std::int64_t col = 0; col < width; ++col)
        mean += decode(values[static_cast<std::size_t>(row * width + col)]);
      mean /= static_cast<double>(width);
    }
    double variance = 0;
    for (std::int64_t col = 0; col < width; ++col) {
      const double centered =
          decode(values[static_cast<std::size_t>(row * width + col)]) - mean;
      variance += centered * centered;
    }
    const double inverse =
        1 / std::sqrt(variance / static_cast<double>(width) + 1.e-5);
    for (std::int64_t col = 0; col < width; ++col) {
      const auto index = static_cast<std::size_t>(row * width + col);
      double expected = (decode(values[index]) - mean) * inverse;
      if (parameters > 0)
        expected *= decode(scales[static_cast<std::size_t>(col)]);
      if (parameters > 1)
        expected += decode(biases[static_cast<std::size_t>(col)]);
      const float rounded = decode(encode<T>(static_cast<float>(expected)));
      const float tolerance =
          std::is_same_v<T, float>
              ? 3.e-5F
              : (std::is_same_v<T, hyle::simd::float16_t> ? 0.002F : 0.016F);
      EXPECT_NEAR(decode(actual[index]), rounded, tolerance) << index;
    }
  }
}

TEST(OneDnnNormalization, Float32NormsUseIndependentDoubleOracle) {
  for (const auto rows : {1, 17, 128})
    for (const auto width : {7, 2048})
      for (const bool rms : {true, false})
        for (std::size_t parameters = 0; parameters <= (rms ? 1U : 2U);
             ++parameters)
          check_norm<float, float>(rms, rows, width, parameters);
}
template <class T> bool native_half_available() {
  bool supported = false;
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const auto description =
        TensorDesc::dense(type<T>(), std::array<std::int64_t, 2>{2, 7}).value();
    const std::array providers{
        hyle::compute::ProviderBinding{onednn::provider(), 100}};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                        .providers = providers})
            .value();
    auto result = prepare(context, RmsNorm{}, description);
    supported = result.has_value();
    if (!supported) {
      EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
    }
  }
  EXPECT_TRUE(stats.balanced());
  testing::Test::RecordProperty(
      type<T>() == DataType::kFloat16 ? "native_f16" : "native_bf16",
      supported ? "EXECUTED_BY_ORACLE_CASES"
                : "ISA_UNSUPPORTED_NUMERICS_UNPROVEN");
  return supported;
}

TEST(OneDnnNormalization, HalfIsaAdmissionAndAvailableNumerics) {
  const bool f16 = native_half_available<hyle::simd::float16_t>();
  const bool bf16 = native_half_available<hyle::simd::bfloat16_t>();
  for (const bool rms : {true, false}) {
    if (f16) {
      check_norm<hyle::simd::float16_t, float>(rms, 17, 2048, rms ? 1 : 2);
      check_norm<hyle::simd::float16_t, hyle::simd::float16_t>(rms, 1, 7,
                                                               rms ? 1 : 2);
    }
    if (bf16) {
      check_norm<hyle::simd::bfloat16_t, float>(rms, 17, 2048, rms ? 1 : 2);
      check_norm<hyle::simd::bfloat16_t, hyle::simd::bfloat16_t>(rms, 1, 7,
                                                                 rms ? 1 : 2);
      check_norm<hyle::simd::bfloat16_t, float>(rms, 17, 31, rms ? 1 : 2, true);
    }
  }
}
TEST(OneDnnNormalization, ExactInputAliasRetainsOriginalReductionInput) {
  for (const bool rms : {true, false}) {
    check_norm<float, float>(rms, 17, 31, rms ? 1 : 2, true);
  }
}
TEST(OneDnnNormalization, BorrowedExecutorAndConcurrentArtifacts) {
  std::atomic<std::size_t> episodes{0};
  const ParallelExecutorRef executor(
      &episodes, 2,
      [](void *opaque, std::size_t count, ParallelTaskFn task,
         void *argument) noexcept {
        ++*static_cast<std::atomic<std::size_t> *>(opaque);
        std::thread worker([&] {
          for (std::size_t index = 1; index < count; index += 2)
            task(argument, index, count);
        });
        for (std::size_t index = 0; index < count; index += 2)
          task(argument, index, count);
        worker.join();
      });
  std::thread first(
      [&] { check_norm<float, float>(true, 128, 2048, 1, false, executor); });
  std::thread second(
      [&] { check_norm<float, float>(false, 17, 2048, 2, false, executor); });
  first.join();
  second.join();
  EXPECT_GT(episodes.load(), 0U);
}

TEST(OneDnnNormalization, RejectsUnsupportedPrecisionAtBothColdStages) {
  const auto data =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{7})
          .value();
  auto context = Context::create(cpu::device(), onednn::provider()).value();
  const std::array inputs{data}, outputs{data};
  std::array<detail::ResourceBindingRequirement, 1> in{}, out{};
  auto proposal =
      detail::OperationPreparationAccess::propose(
          context, 0, Operation(RmsNorm{}), inputs, outputs, in, out)
          .value();
  const std::array resources{hyle::mem::system_resource()};
  for (const auto precision : {DataType::kUnknown, DataType::kFloat64,
                               DataType::kFloat16, DataType::kInt32}) {
    const Operation operation(RmsNorm{.accumulation_type = precision});
    auto refused = prepare(context, operation, inputs, outputs,
                           hyle::mem::system_resource());
    ASSERT_FALSE(refused);
    EXPECT_TRUE(refused.error().IsUnimplemented());
    auto exact = detail::OperationPreparationAccess::prepare_candidate(
        context, proposal, operation, inputs, outputs, in, out,
        {resources, resources});
    ASSERT_TRUE(exact);
    EXPECT_FALSE(exact->succeeded());
    EXPECT_EQ(exact->decline, detail::ProviderPrepareDecline::kUnsupported);
  }
}

TEST(OneDnnNormalization, IndependentInvocationsShareCodeAndRebindData) {
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{17, 31})
          .value();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const std::array providers{
        hyle::compute::ProviderBinding{onednn::provider(), 100}};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                        .providers = providers})
            .value();
    auto prepared = prepare(context, RmsNorm{}, description).value();
    auto run = [&](float seed) {
      auto scratch = workspace(prepared);
      auto invocation = prepared.create_invocation(scratch.view()).value();
      auto queue = context.create_queue().value();
      auto first = tensor(description), second = tensor(description),
           output = tensor(description);
      std::vector<float> values(17 * 31, seed);
      ASSERT_TRUE(
          first.buffer.view().write(std::as_bytes(std::span(values))).ok());
      std::fill(values.begin(), values.end(), -seed);
      ASSERT_TRUE(
          second.buffer.view().write(std::as_bytes(std::span(values))).ok());
      for (const bool alternate : {false, true, false}) {
        const std::array inputs{alternate ? second.view.as_const()
                                          : first.view.as_const()};
        const std::array outputs{output.view};
        auto done = invocation.submit(queue, inputs, outputs);
        ASSERT_TRUE(done) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        ASSERT_TRUE(output.buffer.view()
                        .read(std::as_writable_bytes(std::span(values)))
                        .ok());
        const float expected =
            (alternate ? -seed : seed) / std::sqrt(seed * seed + 1.e-5F);
        for (const auto value : values)
          EXPECT_NEAR(value, expected, 1.e-5F);
      }
    };
    std::thread first(run, 1.F), second(run, 2.F);
    first.join();
    second.join();
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(OneDnnNormalization, NestedBorrowedExecutorMakesProgress) {
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{17, 31})
          .value();
  auto inner_context =
      Context::create(cpu::device(), onednn::provider()).value();
  auto inner_prepared = prepare(inner_context, RmsNorm{}, description).value();
  auto scratch = workspace(inner_prepared);
  auto inner = inner_prepared.create_invocation(scratch.view()).value();
  auto queue = inner_context.create_queue().value();
  auto input = tensor(description), output = tensor(description);
  ASSERT_TRUE(input.buffer.view().fill(std::byte{0}).ok());
  struct Nested final {
    OperationInvocation *invocation;
    Queue *queue;
    ConstTensorView input;
    TensorView output;
    std::size_t calls = 0;
    hyle::Status failure = hyle::Status::Ok();
  } nested{&inner, &queue, input.view.as_const(), output.view};
  const ParallelExecutorRef executor(
      &nested, 2,
      [](void *opaque, std::size_t count, ParallelTaskFn task,
         void *context) noexcept {
        auto &nested = *static_cast<Nested *>(opaque);
        ++nested.calls;
        auto completion = nested.invocation->submit(*nested.queue, nested.input,
                                                    nested.output);
        if (!completion)
          nested.failure = std::move(completion).error();
        else {
          auto status = completion->wait();
          if (!status.ok())
            nested.failure = std::move(status);
        }
        // oneDNN THREADPOOL loops used here have no cross-task barrier. The
        // borrowed executor is explicitly synchronous and supports nesting.
        for (std::size_t index = 0; index < count; ++index)
          task(context, index, count);
      });
  check_norm<float, float>(false, 17, 2048, 2, false, executor);
  check_norm<float, float>(true, 17, 2048, 1, false, executor);
  EXPECT_GT(nested.calls, 0U);
  EXPECT_TRUE(nested.failure.ok()) << nested.failure.message();
}

TEST(OneDnnNormalization, StableCenteredVarianceAndSpecialValues) {
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{4, 4})
          .value();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const std::array values{10000.F,  10001.F, 9999.F, 10000.F, 0.F, 0.F,
                          0.F,      0.F,     nan,    1.F,     2.F, 3.F,
                          infinity, 1.F,     2.F,    3.F};
  auto input = tensor(description), output = tensor(description);
  ASSERT_TRUE(input.buffer.view().write(std::as_bytes(std::span(values))).ok());
  auto context = Context::create(cpu::device(), onednn::provider()).value();
  const std::array inputs{input.view.as_const()};
  const std::array outputs{output.view};
  auto prepared =
      prepare(context, Operation(LayerNorm{}), inputs, outputs).value();
  auto scratch = workspace(prepared);
  auto invocation = prepared.create_invocation(scratch.view()).value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  std::array<float, 16> actual{};
  ASSERT_TRUE(output.buffer.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  EXPECT_NEAR(actual[0], 0.F, 1.e-5F);
  EXPECT_NEAR(actual[1], 1.F / std::sqrt(0.5F + 1.e-5F), 1.e-4F);
  EXPECT_NEAR(actual[2], -actual[1], 1.e-4F);
  for (std::size_t index = 4; index < 8; ++index)
    EXPECT_EQ(actual[index], 0.F);
  for (std::size_t index = 8; index < 16; ++index)
    EXPECT_TRUE(std::isnan(actual[index])) << index;
}
} // namespace
