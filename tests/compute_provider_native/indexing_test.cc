#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "../compute/native_context_fixture.h"
#include "half_test_support.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::Concat;
using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::Gather;
using hyle::compute::Operation;
using hyle::compute::Scatter;
using hyle::compute::ScatterKind;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;

template <class Value, std::size_t Size>
Buffer make_buffer(const std::array<Value, Size> &values) {
  Buffer result = Buffer::create(hyle::mem::system_resource(), sizeof(values),
                                 alignof(Value))
                      .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(std::span(values))).ok());
  return result;
}

template <class Value, std::size_t Size>
std::array<Value, Size> read_buffer(const Buffer &buffer) {
  std::array<Value, Size> result{};
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

TEST(NativeIndexing, GatherCopiesDenseSuffixBlocksAndChecksIndices) {
  constexpr std::array<std::int64_t, 3> source_shape{2, 3, 2};
  constexpr std::array<std::int64_t, 1> index_shape{2};
  constexpr std::array<std::int64_t, 3> output_shape{2, 2, 2};
  constexpr std::array<std::int32_t, 12> source_values{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  constexpr std::array<std::int32_t, 2> indices{2, 0};
  constexpr std::array<std::int32_t, 8> expected{5, 6, 1, 2,
                                                 11, 12, 7, 8};
  std::array<std::int32_t, expected.size()> poison{};
  poison.fill(-7);

  const TensorDesc source_desc =
      TensorDesc::dense(DataType::kInt32, source_shape).value();
  const TensorDesc index_desc =
      TensorDesc::dense(DataType::kInt32, index_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kInt32, output_shape).value();
  Buffer source_storage = make_buffer(source_values);
  Buffer index_storage = make_buffer(indices);
  Buffer output_storage = make_buffer(poison);
  const std::array inputs{
      ConstTensorView::create(source_desc, source_storage.view()).value(),
      ConstTensorView::create(index_desc, index_storage.view()).value()};
  const std::array outputs{
      TensorView::create(output_desc, output_storage.view()).value()};

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Gather{.axis = 1}, source_desc,
                             index_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs[0])->wait().ok());
  EXPECT_EQ((read_buffer<std::int32_t, expected.size()>(output_storage)),
            expected);

  constexpr std::array<std::int32_t, 2> invalid_indices{3, 0};
  ASSERT_TRUE(index_storage.view()
                  .write(std::as_bytes(std::span(invalid_indices)))
                  .ok());
  ASSERT_TRUE(
      output_storage.view().write(std::as_bytes(std::span(poison))).ok());
  auto failed = invocation.submit(queue, inputs, outputs[0]);
  ASSERT_TRUE(failed.has_value());
  EXPECT_TRUE(failed->wait().IsOutOfRange());
  EXPECT_EQ((read_buffer<std::int32_t, poison.size()>(output_storage)), poison);
}

TEST(NativeIndexing, ScatterAssignsOrAddsAndSupportsExactSourceAlias) {
  constexpr std::array<std::int64_t, 2> source_shape{2, 3};
  constexpr std::array<std::int64_t, 1> index_shape{2};
  constexpr std::array<std::int64_t, 2> update_shape{2, 2};
  constexpr std::array<float, 6> source_values{1, 2, 3, 4, 5, 6};
  constexpr std::array<std::int32_t, 2> indices{2, 0};
  constexpr std::array<float, 4> updates{10, 20, 30, 40};
  constexpr std::array<float, 6> assigned{20, 2, 10, 40, 5, 30};
  constexpr std::array<float, 6> added{21, 2, 13, 44, 5, 36};
  std::array<float, source_values.size()> zero{};

  const TensorDesc source_desc =
      TensorDesc::dense(DataType::kFloat32, source_shape).value();
  const TensorDesc index_desc =
      TensorDesc::dense(DataType::kInt32, index_shape).value();
  const TensorDesc update_desc =
      TensorDesc::dense(DataType::kFloat32, update_shape).value();
  Buffer source_storage = make_buffer(source_values);
  Buffer index_storage = make_buffer(indices);
  Buffer update_storage = make_buffer(updates);
  Buffer output_storage = make_buffer(zero);
  const std::array inputs{
      ConstTensorView::create(source_desc, source_storage.view()).value(),
      ConstTensorView::create(index_desc, index_storage.view()).value(),
      ConstTensorView::create(update_desc, update_storage.view()).value()};
  const std::array outputs{
      TensorView::create(source_desc, output_storage.view()).value()};

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto assign =
      hyle::compute::prepare(
          context,
          Scatter{.axis = 1, .kind = ScatterKind::kAssign}, source_desc,
          index_desc, update_desc)
          .value();
  auto assign_invocation = assign.create_invocation().value();
  ASSERT_TRUE(assign_invocation.submit(queue, inputs, outputs[0])->wait().ok());
  EXPECT_EQ((read_buffer<float, assigned.size()>(output_storage)), assigned);

  const TensorView alias_output =
      TensorView::create(source_desc, source_storage.view()).value();
  const std::array alias_outputs{alias_output};
  auto add =
      hyle::compute::prepare(
          context, Scatter{.axis = 1, .kind = ScatterKind::kAdd}, source_desc,
          index_desc, update_desc)
          .value();
  EXPECT_TRUE(add.alias_support(0, 0)->supported);
  auto add_invocation = add.create_invocation().value();
  ASSERT_TRUE(add_invocation.submit(queue, inputs, alias_outputs[0])
                  ->wait()
                  .ok());
  EXPECT_EQ((read_buffer<float, added.size()>(source_storage)), added);
}

TEST(NativeIndexing, ConcatenatesVariadicDenseInputsAlongMiddleAxis) {
  constexpr std::array<std::int64_t, 2> first_shape{2, 2};
  constexpr std::array<std::int64_t, 2> second_shape{2, 1};
  constexpr std::array<std::int64_t, 2> output_shape{2, 3};
  constexpr std::array<std::int16_t, 4> first_values{1, 2, 3, 4};
  constexpr std::array<std::int16_t, 2> second_values{5, 6};
  constexpr std::array<std::int16_t, 6> expected{1, 2, 5, 3, 4, 6};
  std::array<std::int16_t, expected.size()> zero{};

  const TensorDesc first_desc =
      TensorDesc::dense(DataType::kInt16, first_shape).value();
  const TensorDesc second_desc =
      TensorDesc::dense(DataType::kInt16, second_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kInt16, output_shape).value();
  Buffer first_storage = make_buffer(first_values);
  Buffer second_storage = make_buffer(second_values);
  Buffer output_storage = make_buffer(zero);
  const std::array inputs{
      ConstTensorView::create(first_desc, first_storage.view()).value(),
      ConstTensorView::create(second_desc, second_storage.view()).value()};
  const std::array outputs{
      TensorView::create(output_desc, output_storage.view()).value()};

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  const std::array descriptions{first_desc, second_desc};
  auto prepared =
      hyle::compute::prepare(context, Concat{.axis = 1}, descriptions).value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs[0])->wait().ok());
  EXPECT_EQ((read_buffer<std::int16_t, expected.size()>(output_storage)),
            expected);
}

template <class Value> void expect_half_scatter() {
  const auto type = half_test::data_type<Value>();
  std::array<Value, 8> source{};
  source.fill(half_test::encode<Value>(1.0));
  constexpr bool bf16 = std::is_same_v<Value, hyle::simd::bfloat16_t>;
  const double half_ulp = bf16 ? 1.0 / 256 : 1.0 / 2048;
  const std::array updates{
      half_test::encode<Value>(half_ulp), half_test::encode<Value>(half_ulp),
      half_test::encode<Value>(half_ulp), half_test::encode<Value>(4.0)};
  std::array<std::int32_t, 4> indices{2, 2, 2, 0};
  const auto desc =
      TensorDesc::dense(type, std::array<std::int64_t, 1>{8}).value();
  const auto update_desc =
      TensorDesc::dense(type, std::array<std::int64_t, 1>{4}).value();
  const auto index_desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{4})
          .value();
  Buffer storage = make_buffer(source);
  Buffer update_storage = make_buffer(updates);
  Buffer index_storage = make_buffer(indices);
  const auto output = TensorView::create(desc, storage.view()).value();
  const auto update_view =
      ConstTensorView::create(update_desc, update_storage.view()).value();
  const auto index_view =
      ConstTensorView::create(index_desc, index_storage.view()).value();
  auto context =
      hyle::compute::Context::create(hyle::compute::cpu::device(),
                                     hyle::compute::aten_cpu::provider())
          .value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Scatter{.kind = ScatterKind::kAdd}, desc,
                             index_desc, update_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(
      invocation
          .submit(queue, output.as_const(), index_view, update_view, output)
          ->wait()
          .ok());
  const auto actual = half_test::read<Value>(storage);
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_EQ(actual[index].bits,
              half_test::encode<Value>(index == 0   ? 5.0
                                       : index == 2 ? 1.0 + 3 * half_ulp
                                                    : 1.0)
                  .bits);

  indices.back() = 8;
  ASSERT_TRUE(
      index_storage.view().write(std::as_bytes(std::span(indices))).ok());
  auto failed = invocation.submit(queue, output.as_const(), index_view,
                                  update_view, output);
  ASSERT_TRUE(failed.has_value());
  EXPECT_TRUE(failed->wait().IsOutOfRange());
  const auto unchanged = half_test::read<Value>(storage);
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_EQ(unchanged[index].bits, actual[index].bits);
}

TEST(AtenCpuIndexing,
     HalfScatterAccumulatesBeforeStorageAndChecksBeforeWriting) {
  if (!hyle::compute::aten_cpu::provider().valid())
    GTEST_SKIP() << "ATen CPU SDK is not enabled";
  expect_half_scatter<hyle::simd::float16_t>();
  expect_half_scatter<hyle::simd::bfloat16_t>();
}

} // namespace
