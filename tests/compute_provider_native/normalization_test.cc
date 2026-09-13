#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::LayerNorm;
using hyle::compute::ParallelTaskFn;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;

template <std::size_t Size>
Buffer make_buffer(const std::array<float, Size> &values) {
  Buffer result = Buffer::create(hyle::mem::system_resource(), sizeof(values),
                                 alignof(float))
                      .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(std::span(values))).ok());
  return result;
}

template <std::size_t Size>
std::array<float, Size> read_buffer(const Buffer &buffer) {
  std::array<float, Size> result{};
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_NEAR(actual[index], expected[index], 2.0e-5F) << index;
}

struct ParallelProbe final {
  std::size_t episodes = 0;
  std::size_t tasks = 0;
  bool nested = false;
};

void probe_parallel_for(void *context, std::size_t count, ParallelTaskFn task,
                        void *task_context) noexcept {
  auto &probe = *static_cast<ParallelProbe *>(context);
  ++probe.episodes;
  probe.tasks += count;
  for (std::size_t index = 0; index < count; ++index)
    task(task_context, index, count);
}

bool probe_in_parallel(void *context) noexcept {
  return static_cast<ParallelProbe *>(context)->nested;
}

TEST(NativeLayerNorm, ExecutesNonAffineScaleOnlyAndAffineRows) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int64_t, 1> parameter_shape{3};
  constexpr std::array<float, 6> values{1, 2, 3, 2, 4, 6};
  constexpr std::array<float, 3> scale{2, 3, 4};
  constexpr std::array<float, 3> bias{10, 20, 30};
  constexpr float epsilon = 1.0e-5F;
  const float first_root = 1.0F / std::sqrt(2.0F / 3.0F + epsilon);
  const float second_root = 2.0F / std::sqrt(8.0F / 3.0F + epsilon);
  const std::array<float, 6> normalized{-first_root,  0, first_root,
                                        -second_root, 0, second_root};
  const std::array<float, 6> scaled{-2 * first_root,  0, 4 * first_root,
                                    -2 * second_root, 0, 4 * second_root};
  const std::array<float, 6> affine{
      10 - 2 * first_root,  20, 30 + 4 * first_root,
      10 - 2 * second_root, 20, 30 + 4 * second_root};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc parameter_desc =
      TensorDesc::dense(DataType::kFloat32, parameter_shape).value();
  Buffer input_storage = make_buffer(values);
  Buffer scale_storage = make_buffer(scale);
  Buffer bias_storage = make_buffer(bias);
  Buffer output_storage = make_buffer(std::array<float, 6>{});
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const ConstTensorView scale_view =
      ConstTensorView::create(parameter_desc, scale_storage.view()).value();
  const ConstTensorView bias_view =
      ConstTensorView::create(parameter_desc, bias_storage.view()).value();
  const TensorView output =
      TensorView::create(input_desc, output_storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const LayerNorm operation{.epsilon = epsilon};

  auto prepared =
      hyle::compute::prepare(context, operation, input_desc).value();
  ASSERT_EQ(prepared.output_description().value(), input_desc);
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
  expect_near(read_buffer<6>(output_storage), normalized);

  prepared =
      hyle::compute::prepare(context, operation, input_desc, parameter_desc)
          .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, scale_view, output)->wait().ok());
  expect_near(read_buffer<6>(output_storage), scaled);

  prepared = hyle::compute::prepare(context, operation, input_desc,
                                    parameter_desc, parameter_desc)
                 .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, scale_view, bias_view, output)
                  ->wait()
                  .ok());
  expect_near(read_buffer<6>(output_storage), affine);
}

TEST(NativeLayerNorm, SupportsExactInputOutputAlias) {
  constexpr std::array<std::int64_t, 2> shape{1, 3};
  constexpr std::array<float, 3> values{1, 2, 3};
  constexpr float root = 1.2247357F;
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer storage = make_buffer(values);
  const ConstTensorView input =
      ConstTensorView::create(description, storage.view()).value();
  const TensorView output =
      TensorView::create(description, storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, LayerNorm{}, description).value();
  auto invocation = prepared.create_invocation().value();

  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
  expect_near(read_buffer<3>(storage), std::array<float, 3>{-root, 0, root});
}

TEST(NativeLayerNorm, AcceptsAnEmptyLastDimension) {
  constexpr std::array<std::int64_t, 2> shape{3, 0};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer storage = Buffer::create(hyle::mem::system_resource(), 0).value();
  const ConstTensorView input =
      ConstTensorView::create(description, storage.view()).value();
  const TensorView output =
      TensorView::create(description, storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, LayerNorm{}, description).value();
  auto invocation = prepared.create_invocation().value();

  EXPECT_TRUE(invocation.submit(queue, input, output)->wait().ok());
}

TEST(NativeLayerNorm, RejectsUnsupportedStorageAndAccumulation) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  const TensorDesc f16 = TensorDesc::dense(DataType::kFloat16, shape).value();
  const TensorDesc f32 = TensorDesc::dense(DataType::kFloat32, shape).value();
  auto context = hyle::compute::cpu::create_context().value();

  auto storage = hyle::compute::prepare(context, LayerNorm{}, f16);
  ASSERT_FALSE(storage.has_value());
  EXPECT_TRUE(storage.error().IsUnimplemented());

  auto accumulation = hyle::compute::prepare(
      context, LayerNorm{.accumulation_type = DataType::kFloat64}, f32);
  ASSERT_FALSE(accumulation.has_value());
  EXPECT_TRUE(accumulation.error().IsUnimplemented());
}

TEST(NativeLayerNorm, PinsExecutorAndUsesOneBoundedSynchronousFanOut) {
  constexpr std::array<std::int64_t, 2> shape{8, 8192};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const std::size_t bytes =
      static_cast<std::size_t>(shape[0] * shape[1]) * sizeof(float);
  Buffer input =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  Buffer output =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  ASSERT_TRUE(input.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(output.view().fill(std::byte{0}).ok());

  ParallelProbe probe;
  const hyle::compute::ParallelExecutorRef executor(
      &probe, 4, &probe_parallel_for, &probe_in_parallel);
  auto context =
      hyle::compute::cpu::create_context({.executor = executor}).value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, LayerNorm{}, description).value();
  auto invocation = prepared.create_invocation().value();
  const ConstTensorView input_view =
      ConstTensorView::create(description, input.view()).value();
  const TensorView output_view =
      TensorView::create(description, output.view()).value();

  ASSERT_TRUE(invocation.submit(queue, input_view, output_view)->wait().ok());
  EXPECT_EQ(probe.episodes, 1U);
  EXPECT_EQ(probe.tasks, 4U);

  probe.nested = true;
  ASSERT_TRUE(invocation.submit(queue, input_view, output_view)->wait().ok());
  EXPECT_EQ(probe.episodes, 1U);
  EXPECT_EQ(probe.tasks, 4U);
}

} // namespace
