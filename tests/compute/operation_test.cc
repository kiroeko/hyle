#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::GatedActivation;
using hyle::compute::MatMul;
using hyle::compute::OperationKind;
using hyle::compute::PreparedOperation;
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;
using hyle::mem::BufferView;
using hyle::mem::ConstBufferView;
using hyle::mem::HostAccess;
using hyle::mem::MemoryDomainId;
using hyle::mem::MemoryProperties;
using hyle::mem::MemoryProperty;
using hyle::mem::MemoryResource;
using hyle::mem::MemoryResourceRef;
using hyle::mem::ResourceAllocation;
using hyle::mem::ResourceHostMapping;

TensorDesc row_description(std::size_t count) {
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(count)};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TensorDesc matrix_description(std::size_t rows, std::size_t columns) {
  const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(rows),
                                          static_cast<std::int64_t>(columns)};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

void write_floats(BufferView view, std::span<const float> values) {
  ASSERT_TRUE(view.write(std::as_bytes(values)).ok());
}

std::vector<float> read_floats(ConstBufferView view, std::size_t count) {
  std::vector<float> values(count);
  EXPECT_TRUE(view.read(std::as_writable_bytes(std::span(values))).ok());
  return values;
}

void expect_rms_norm(std::span<const float> input,
                     std::span<const float> weight,
                     std::span<const float> actual, float epsilon) {
  ASSERT_EQ(input.size(), actual.size());
  ASSERT_TRUE(weight.empty() || weight.size() == input.size());
  float sum = 0.0F;
  for (float value : input)
    sum += value * value;
  const float inverse =
      1.0F / std::sqrt(sum / static_cast<float>(input.size()) + epsilon);
  for (std::size_t index = 0; index < input.size(); ++index) {
    float expected = input[index] * inverse;
    if (!weight.empty())
      expected *= weight[index];
    EXPECT_NEAR(actual[index], expected,
                std::abs(expected) * 3.0e-5F + 3.0e-6F);
  }
}

class InstrumentedResource final : public MemoryResource {
public:
  std::string_view name() const noexcept override { return "compute.test"; }

  MemoryProperties properties() const noexcept override {
    MemoryProperty flags = MemoryProperty::kHostAccessible;
    if (coherent_)
      flags |= MemoryProperty::kHostDeviceCoherent;
    if (!host_accessible_)
      flags = MemoryProperty::kDeviceAccessible;
    return MemoryProperties{
        .flags = flags,
        .minimum_alignment = 64,
        .maximum_allocation = storage_.size(),
        .domain = MemoryDomainId{this, 1},
    };
  }

  Result<ResourceAllocation> allocate(std::size_t size,
                                      std::size_t alignment) noexcept override {
    if (live_ || size > storage_.size() || alignment > 64) {
      return std::unexpected(
          Status::ResourceExhausted("test allocation failed"));
    }
    live_ = true;
    return ResourceAllocation{size == 0 ? nullptr : storage_.data(), size,
                              alignment};
  }

  void deallocate(ResourceAllocation) noexcept override { live_ = false; }

  Result<ResourceHostMapping> map_host(void *native_handle, std::size_t offset,
                                       std::size_t size,
                                       HostAccess) noexcept override {
    ++maps_;
    if (fail_mapping_) {
      return std::unexpected(Status::Unavailable("test mapping failed"));
    }
    std::byte *data = nullptr;
    if (size != 0) {
      data = static_cast<std::byte *>(native_handle) + offset;
      if (misaligned_mapping_)
        ++data;
    }
    return ResourceHostMapping{data, nullptr};
  }

  Status unmap_host(void *, void *, std::size_t, std::size_t,
                    HostAccess) noexcept override {
    ++unmaps_;
    return fail_unmap_ ? Status::Unavailable("test unmap failed")
                       : Status::Ok();
  }

  Status flush(void *, std::size_t, std::size_t) noexcept override {
    ++flushes_;
    return Status::Ok();
  }

  Status invalidate(void *, std::size_t, std::size_t) noexcept override {
    ++invalidates_;
    return Status::Ok();
  }

  void reset_counts() noexcept {
    maps_ = 0;
    unmaps_ = 0;
    flushes_ = 0;
    invalidates_ = 0;
  }

  void fail_mapping(bool value) noexcept { fail_mapping_ = value; }
  void fail_unmap(bool value) noexcept { fail_unmap_ = value; }
  void host_accessible(bool value) noexcept { host_accessible_ = value; }
  void coherent(bool value) noexcept { coherent_ = value; }
  void misaligned_mapping(bool value) noexcept { misaligned_mapping_ = value; }
  std::span<std::byte> storage() noexcept { return storage_; }
  int maps() const noexcept { return maps_; }
  int unmaps() const noexcept { return unmaps_; }
  int flushes() const noexcept { return flushes_; }
  int invalidates() const noexcept { return invalidates_; }

private:
  alignas(64) std::array<std::byte, 1024> storage_{};
  bool live_ = false;
  bool fail_mapping_ = false;
  bool fail_unmap_ = false;
  bool host_accessible_ = true;
  bool coherent_ = true;
  bool misaligned_mapping_ = false;
  int maps_ = 0;
  int unmaps_ = 0;
  int flushes_ = 0;
  int invalidates_ = 0;
};

TEST(PreparedOperation, ExposesColdRouteAndShapeDiagnostics) {
  PreparedOperation invalid;
  EXPECT_EQ(invalid.kind(), OperationKind::kUnknown);
  EXPECT_FALSE(invalid.provider().valid());
  EXPECT_EQ(hyle::compute::operation_name(invalid.kind()), "unknown");

  Context context = hyle::compute::cpu::create_context().value();
  const TensorDesc description = row_description(8);
  PreparedOperation prepared =
      hyle::compute::prepare(context, RmsNorm{}, description)
          .value();

  EXPECT_EQ(prepared.kind(), OperationKind::kRmsNorm);
  EXPECT_EQ(prepared.provider().id, hyle::compute::native::kProviderId);
  EXPECT_EQ(prepared.input_count(), 1u);
  EXPECT_EQ(prepared.output_count(), 1u);
  EXPECT_EQ(prepared.input_description().value(), description);
  EXPECT_EQ(prepared.output_description().value(), description);
  EXPECT_EQ(hyle::compute::operation_name(prepared.kind()), "rms-norm");
  EXPECT_EQ(prepared.provider().name, "hyle-native");
  EXPECT_EQ(prepared.runtime_properties().replay_class,
            hyle::compute::ProviderReplayClass::kPrepared);

  auto bad_input = prepared.input_description(1);
  ASSERT_FALSE(bad_input.has_value());
  EXPECT_TRUE(bad_input.error().IsOutOfRange());
  auto bad_output = prepared.output_description(1);
  ASSERT_FALSE(bad_output.has_value());
  EXPECT_TRUE(bad_output.error().IsOutOfRange());
}

TEST(PreparedOperation, RejectsUnsupportedOrInconsistentPreparation) {
  Context context = hyle::compute::cpu::create_context().value();
  const TensorDesc row = row_description(8);
  const TensorDesc other = row_description(7);

  auto negative =
      hyle::compute::prepare(context, RmsNorm{.epsilon = -1.0F}, row);
  ASSERT_FALSE(negative.has_value());
  EXPECT_TRUE(negative.error().IsInvalidArgument());

  auto nan = hyle::compute::prepare(
      context, RmsNorm{.epsilon = (std::numeric_limits<float>::quiet_NaN)()},
      row);
  ASSERT_FALSE(nan.has_value());
  EXPECT_TRUE(nan.error().IsInvalidArgument());

  auto shape = hyle::compute::prepare(context, RmsNorm{}, row, other);
  ASSERT_FALSE(shape.has_value());
  EXPECT_TRUE(shape.error().IsInvalidArgument());

  const TensorDesc scalar = TensorDesc::dense(DataType::kFloat32, {}).value();
  auto rank = hyle::compute::prepare(context, RmsNorm{}, scalar, scalar);
  ASSERT_FALSE(rank.has_value());
  EXPECT_TRUE(rank.error().IsInvalidArgument());

  const std::array<std::int64_t, 1> shape8{8};
  const std::array<std::int64_t, 1> broadcast_stride{0};
  const TensorDesc broadcast =
      TensorDesc::strided_bytes(DataType::kFloat32, shape8, broadcast_stride)
          .value();
  auto layout = hyle::compute::prepare(context, RmsNorm{}, broadcast, row);
  ASSERT_FALSE(layout.has_value());
  EXPECT_TRUE(layout.error().IsUnimplemented());

  ASSERT_TRUE(context.close().ok());
  auto closed = hyle::compute::prepare(context, RmsNorm{}, row);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(OperationShape, InfersNaturalInferenceShapesAndRejectsMismatches) {
  const TensorDesc rows = matrix_description(3, 8);
  const TensorDesc weight = row_description(8);
  EXPECT_EQ(hyle::compute::infer_single_output(
                hyle::compute::Operation(RmsNorm{}), std::array{rows, weight})
                .value(),
            rows);

  const TensorDesc left = matrix_description(3, 8);
  const TensorDesc right = matrix_description(8, 10);
  const TensorDesc expected = matrix_description(3, 10);
  EXPECT_EQ(hyle::compute::infer_single_output(
                hyle::compute::Operation(MatMul{}), std::array{left, right})
                .value(),
            expected);

  const TensorDesc stored_weight = matrix_description(10, 8);
  const TensorDesc logical_weight = stored_weight.transposed().value();
  EXPECT_EQ(
      hyle::compute::infer_single_output(hyle::compute::Operation(MatMul{}),
                                         std::array{left, logical_weight})
          .value(),
      expected);

  const TensorDesc packed = matrix_description(3, 20);
  EXPECT_EQ(hyle::compute::infer_single_output(
                hyle::compute::Operation(GatedActivation{}), std::array{packed})
                .value(),
            expected);

  EXPECT_TRUE(hyle::compute::infer_single_output(
                  hyle::compute::Operation(MatMul{}),
                  std::array{left, matrix_description(7, 10)})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(hyle::compute::infer_single_output(
                  hyle::compute::Operation(GatedActivation{}),
                  std::array{matrix_description(3, 9)})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(
      hyle::compute::infer_single_output(hyle::compute::Operation(RmsNorm{}),
                                         std::array{rows, row_description(7)})
          .error()
          .IsInvalidArgument());
}

TEST(PreparedOperation, LeavesValidEncodedWeightsForAQualifiedProvider) {
  constexpr StorageFormatId codec{0x48594c45U, 31U};
  const StorageFormat blocked =
      StorageFormat::blocked(codec, 32, 18, 2).value();
  const TensorDesc left = matrix_description(2, 64);
  const std::array<std::int64_t, 2> stored_shape{4, 64};
  const TensorDesc right =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, blocked)
          ->transposed()
          .value();
  const TensorDesc output = matrix_description(2, 4);

  auto inferred = hyle::compute::infer_single_output(
      hyle::compute::Operation(MatMul{}), std::array{left, right});
  ASSERT_TRUE(inferred.has_value()) << inferred.error().message();
  EXPECT_EQ(*inferred, output);

  Context context = hyle::compute::cpu::create_context().value();
  auto native = hyle::compute::prepare(context, MatMul{}, left, right, output);
  ASSERT_FALSE(native.has_value());
  EXPECT_TRUE(native.error().IsUnimplemented());
}

TEST(PreparedOperation, NativeCpuResultLeaseRetainsComputedOutput) {
  hyle::compute::ResultLease<Buffer> retained;
  const std::array<float, 4> values{1.0F, -2.0F, 3.0F, -4.0F};
  constexpr float epsilon = 1.0e-5F;
  {
    auto context = hyle::compute::cpu::create_context().value();
    auto queue = context.create_queue().value();
    const auto description = row_description(values.size());
    auto input =
        Buffer::create(hyle::mem::system_resource(), sizeof(values)).value();
    auto output =
        Buffer::create(hyle::mem::system_resource(), sizeof(values)).value();
    write_floats(input.view(), values);
    const auto input_view =
        ConstTensorView::create(description, input.view()).value();
    const auto output_view =
        TensorView::create(description, output.view()).value();
    auto prepared =
        hyle::compute::prepare(context, RmsNorm{epsilon}, description).value();
    auto invocation = prepared.create_invocation().value();
    auto slot = hyle::compute::detail::ResultSlot<Buffer>::create(
                    hyle::mem::system_allocator())
                    .value();
    auto write = slot.begin().value();
    ASSERT_TRUE(write.emplace(std::move(output)).ok());
    auto completion = invocation.submit(queue, input_view, output_view);
    ASSERT_TRUE(completion.has_value());
    auto submitted = write.publish(std::move(*completion));
    ASSERT_TRUE(submitted.has_value());
    auto ready = submitted->wait();
    ASSERT_TRUE(ready.has_value());
    retained = *ready;
  }
  expect_rms_norm(values, {}, read_floats(retained->view(), values.size()),
                  epsilon);
}

TEST(PreparedOperation, RunsRowWiseWeightedRmsNorm) {
  constexpr std::size_t row_count = 2;
  constexpr std::size_t hidden = 4;
  const std::vector<float> input{1, 2, 3, 4, -2, 1, 0.5F, 3};
  const std::vector<float> weight{0.5F, 1, 1.5F, -2};
  const TensorDesc input_desc = matrix_description(row_count, hidden);
  const TensorDesc weight_desc = row_description(hidden);
  Buffer input_buffer =
      Buffer::create(hyle::mem::system_resource(), input.size() * sizeof(float))
          .value();
  Buffer weight_buffer = Buffer::create(hyle::mem::system_resource(),
                                        weight.size() * sizeof(float))
                             .value();
  Buffer output_buffer =
      Buffer::create(hyle::mem::system_resource(), input.size() * sizeof(float))
          .value();
  write_floats(input_buffer.view(), input);
  write_floats(weight_buffer.view(), weight);

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(context, RmsNorm{}, input_desc, weight_desc)
          .value();
  auto input_view =
      ConstTensorView::create(input_desc, input_buffer.view()).value();
  auto weight_view =
      ConstTensorView::create(weight_desc, weight_buffer.view()).value();
  auto output_view =
      TensorView::create(input_desc, output_buffer.view()).value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input_view, weight_view, output_view)
                  ->wait()
                  .ok());

  const auto actual = read_floats(output_buffer.view(), input.size());
  for (std::size_t row = 0; row < row_count; ++row) {
    expect_rms_norm(std::span(input).subspan(row * hidden, hidden), weight,
                    std::span(actual).subspan(row * hidden, hidden),
                    RmsNorm{}.epsilon);
  }
}

TEST(PreparedOperation, RunsRowMajorAndTransposedWeightMatMul) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t inner = 3;
  constexpr std::size_t columns = 4;
  const std::vector<float> left{1, 2, 3, -1, 0.5F, 4};
  const std::vector<float> right_row_major{1, 2, 3, 4,  5,  6,
                                           7, 8, 9, 10, 11, 12};
  std::vector<float> right_transposed(columns * inner);
  for (std::size_t k = 0; k < inner; ++k) {
    for (std::size_t column = 0; column < columns; ++column) {
      right_transposed[column * inner + k] =
          right_row_major[k * columns + column];
    }
  }
  std::vector<float> expected(rows * columns, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t k = 0; k < inner; ++k) {
        expected[row * columns + column] +=
            left[row * inner + k] * right_row_major[k * columns + column];
      }
    }
  }

  const TensorDesc left_desc = matrix_description(rows, inner);
  const TensorDesc right_desc = matrix_description(inner, columns);
  const TensorDesc physical_desc = matrix_description(columns, inner);
  const TensorDesc transposed_desc = physical_desc.transposed().value();
  const TensorDesc output_desc = matrix_description(rows, columns);
  Buffer left_buffer =
      Buffer::create(hyle::mem::system_resource(), left.size() * sizeof(float))
          .value();
  Buffer right_buffer = Buffer::create(hyle::mem::system_resource(),
                                       right_row_major.size() * sizeof(float))
                            .value();
  Buffer output_buffer = Buffer::create(hyle::mem::system_resource(),
                                        expected.size() * sizeof(float))
                             .value();
  write_floats(left_buffer.view(), left);

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  auto left_view =
      ConstTensorView::create(left_desc, left_buffer.view()).value();
  auto output_view =
      TensorView::create(output_desc, output_buffer.view()).value();

  write_floats(right_buffer.view(), right_row_major);
  auto right_view =
      ConstTensorView::create(right_desc, right_buffer.view()).value();
  auto row_major = hyle::compute::prepare(context, MatMul{}, left_desc,
                                          right_desc, output_desc)
                       .value();
  auto row_major_invocation = row_major.create_invocation().value();
  ASSERT_TRUE(
      row_major_invocation.submit(queue, left_view, right_view, output_view)
          ->wait()
          .ok());
  EXPECT_EQ(read_floats(output_buffer.view(), expected.size()), expected);

  write_floats(right_buffer.view(), right_transposed);
  auto transposed_view =
      ConstTensorView::create(transposed_desc, right_buffer.view()).value();
  auto transposed = hyle::compute::prepare(context, MatMul{}, left_desc,
                                           transposed_desc, output_desc)
                        .value();
  auto transposed_invocation = transposed.create_invocation().value();
  ASSERT_TRUE(transposed_invocation
                  .submit(queue, left_view, transposed_view, output_view)
                  ->wait()
                  .ok());
  const auto actual = read_floats(output_buffer.view(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 2.0e-6F + 1.0e-6F);
  }
}

TEST(PreparedOperation, RowMajorMatMulCoversEveryNativeTileAndTail) {
  struct Shape final {
    std::size_t rows;
    std::size_t inner;
    std::size_t columns;
  };
  constexpr std::array shapes{
      Shape{1, 5, 37},   Shape{4, 7, 19},   Shape{6, 9, 23},   Shape{7, 11, 29},
      Shape{10, 13, 31}, Shape{17, 15, 35}, Shape{32, 17, 67}, Shape{9, 0, 21},
  };

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  for (const Shape shape : shapes) {
    SCOPED_TRACE(::testing::Message() << "M=" << shape.rows << " K="
                                      << shape.inner << " N=" << shape.columns);
    std::vector<float> left(shape.rows * shape.inner);
    std::vector<float> right(shape.inner * shape.columns);
    for (std::size_t index = 0; index < left.size(); ++index) {
      left[index] =
          static_cast<float>(static_cast<int>(index % 19) - 9) / 32.0F;
    }
    for (std::size_t index = 0; index < right.size(); ++index) {
      right[index] =
          static_cast<float>(static_cast<int>(index % 23) - 11) / 64.0F;
    }
    std::vector<float> expected(shape.rows * shape.columns, 0.0F);
    for (std::size_t row = 0; row < shape.rows; ++row) {
      for (std::size_t column = 0; column < shape.columns; ++column) {
        for (std::size_t k = 0; k < shape.inner; ++k) {
          expected[row * shape.columns + column] +=
              left[row * shape.inner + k] * right[k * shape.columns + column];
        }
      }
    }

    const TensorDesc left_desc = matrix_description(shape.rows, shape.inner);
    const TensorDesc right_desc =
        matrix_description(shape.inner, shape.columns);
    const TensorDesc output_desc =
        matrix_description(shape.rows, shape.columns);
    Buffer left_buffer = Buffer::create(hyle::mem::system_resource(),
                                        left.size() * sizeof(float), 64)
                             .value();
    Buffer right_buffer = Buffer::create(hyle::mem::system_resource(),
                                         right.size() * sizeof(float), 64)
                              .value();
    Buffer output_buffer = Buffer::create(hyle::mem::system_resource(),
                                          expected.size() * sizeof(float), 64)
                               .value();
    write_floats(left_buffer.view(), left);
    write_floats(right_buffer.view(), right);
    std::vector<float> poison(expected.size(),
                              std::numeric_limits<float>::quiet_NaN());
    write_floats(output_buffer.view(), poison);

    const ConstTensorView left_view =
        ConstTensorView::create(left_desc, left_buffer.view()).value();
    const ConstTensorView right_view =
        ConstTensorView::create(right_desc, right_buffer.view()).value();
    const TensorView output_view =
        TensorView::create(output_desc, output_buffer.view()).value();
    PreparedOperation prepared =
        hyle::compute::prepare(context, MatMul{}, left_desc, right_desc,
                               output_desc)
            .value();
    auto invocation = prepared.create_invocation().value();
    auto completion =
        invocation.submit(queue, left_view, right_view, output_view);
    ASSERT_TRUE(completion.has_value());
    ASSERT_TRUE(completion->wait().ok());

    const auto actual = read_floats(output_buffer.view(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_NEAR(actual[index], expected[index],
                  std::abs(expected[index]) * 3.0e-5F + 3.0e-6F);
    }
  }
}

TEST(PreparedOperation, RunsPackedSiluGatedActivationAcrossRows) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t hidden = 3;
  const std::vector<float> packed{-1,   0,  2, 3,  4, -2,
                                  0.5F, -3, 1, -1, 2, 0.25F};
  const TensorDesc input_desc = matrix_description(rows, hidden * 2);
  const TensorDesc output_desc = matrix_description(rows, hidden);
  Buffer input_buffer = Buffer::create(hyle::mem::system_resource(),
                                       packed.size() * sizeof(float))
                            .value();
  Buffer output_buffer = Buffer::create(hyle::mem::system_resource(),
                                        rows * hidden * sizeof(float))
                             .value();
  write_floats(input_buffer.view(), packed);
  auto input_view =
      ConstTensorView::create(input_desc, input_buffer.view()).value();
  auto output_view =
      TensorView::create(output_desc, output_buffer.view()).value();
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, GatedActivation{}, input_desc)
                      .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input_view, output_view)->wait().ok());
  const auto actual = read_floats(output_buffer.view(), rows * hidden);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t index = 0; index < hidden; ++index) {
      const float gate = packed[row * hidden * 2 + index];
      const float value = packed[row * hidden * 2 + hidden + index];
      const float expected = gate / (1.0F + std::exp(-gate)) * value;
      EXPECT_NEAR(actual[row * hidden + index], expected,
                  std::abs(expected) * 5.0e-5F + 2.0e-6F);
    }
  }
}

TEST(PreparedOperation, RunsUnweightedWeightedAndExactInPlaceRmsNorm) {
  constexpr float epsilon = 1.0e-5F;
  const std::vector<float> input{1.0F, -2.0F, 3.0F, -4.0F, 0.5F, 6.0F};
  const std::vector<float> weight{0.5F, 1.0F, 1.5F, 2.0F, -1.0F, 0.25F};
  const std::size_t bytes = input.size() * sizeof(float);
  const TensorDesc description = row_description(input.size());

  Buffer input_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  Buffer weight_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  Buffer output_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  write_floats(input_buffer.view(), input);
  write_floats(weight_buffer.view(), weight);

  ConstTensorView input_view =
      ConstTensorView::create(description, input_buffer.view()).value();
  ConstTensorView weight_view =
      ConstTensorView::create(description, weight_buffer.view()).value();
  TensorView output_view =
      TensorView::create(description, output_buffer.view()).value();

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  PreparedOperation plain = hyle::compute::prepare(context, RmsNorm{epsilon},
                                                   description)
                                .value();
  auto plain_invocation = plain.create_invocation().value();
  auto completion = plain_invocation.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const auto plain_output = read_floats(output_buffer.view(), input.size());
  expect_rms_norm(input, {}, plain_output, epsilon);

  PreparedOperation weighted =
      hyle::compute::prepare(context, RmsNorm{epsilon}, description,
                             description)
          .value();
  auto weighted_invocation = weighted.create_invocation().value();
  completion =
      weighted_invocation.submit(queue, input_view, weight_view, output_view);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const auto weighted_output = read_floats(output_buffer.view(), input.size());
  expect_rms_norm(input, weight, weighted_output, epsilon);

  write_floats(input_buffer.view(), input);
  TensorView in_place_mutable =
      TensorView::create(description, input_buffer.view()).value();
  completion = plain_invocation.submit(queue, in_place_mutable.as_const(),
                                       in_place_mutable);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const auto in_place = read_floats(input_buffer.view(), input.size());
  expect_rms_norm(input, {}, in_place, epsilon);
}

TEST(PreparedOperation, RejectsWrongBindingContextAndUnsafeAlias) {
  constexpr std::size_t count = 8;
  const std::size_t bytes = count * sizeof(float);
  const TensorDesc description = row_description(count);
  Buffer storage = Buffer::create(hyle::mem::system_resource(),
                                  bytes + sizeof(float), alignof(float))
                       .value();
  Buffer separate =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  ConstTensorView input =
      ConstTensorView::create(description,
                              storage.view().subview(0, bytes).value())
          .value();
  TensorView partial =
      TensorView::create(description,
                         storage.view().subview(sizeof(float), bytes).value())
          .value();
  TensorView output = TensorView::create(description, separate.view()).value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  Queue first_queue = first.create_queue().value();
  Queue wrong_queue = second.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(first, RmsNorm{}, description)
          .value();
  auto invocation = prepared.create_invocation().value();

  auto wrong_context = invocation.submit(wrong_queue, input, output);
  ASSERT_FALSE(wrong_context.has_value());
  EXPECT_TRUE(wrong_context.error().IsInvalidArgument());

  auto partial_alias = invocation.submit(first_queue, input, partial);
  ASSERT_FALSE(partial_alias.has_value());
  EXPECT_TRUE(partial_alias.error().IsInvalidArgument());

  PreparedOperation weighted =
      hyle::compute::prepare(first, RmsNorm{}, description, description)
          .value();
  auto weighted_invocation = weighted.create_invocation().value();
  auto weight_alias =
      weighted_invocation.submit(first_queue, input, output.as_const(), output);
  ASSERT_FALSE(weight_alias.has_value());
  EXPECT_TRUE(weight_alias.error().IsInvalidArgument());

  ASSERT_TRUE(first.close().ok());
  auto closed = invocation.submit(first_queue, input, output);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(PreparedOperation,
     UsesExplicitCoherenceAndReportsProviderFailureInCompletion) {
  constexpr std::size_t count = 8;
  const std::size_t bytes = count * sizeof(float);
  const TensorDesc description = row_description(count);
  const std::array<float, count> input{1, 2, 3, 4, 5, 6, 7, 8};

  InstrumentedResource resource;
  resource.coherent(false);
  Buffer storage =
      Buffer::create(MemoryResourceRef(resource), bytes * 2, 64).value();
  std::memcpy(resource.storage().data(), input.data(), bytes);
  ConstTensorView input_view =
      ConstTensorView::create(description,
                              storage.view().subview(0, bytes).value())
          .value();
  TensorView output_view =
      TensorView::create(description,
                         storage.view().subview(bytes, bytes).value())
          .value();

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(context, RmsNorm{}, description,
                             MemoryResourceRef(resource))
          .value();
  auto invocation = prepared.create_invocation().value();

  resource.reset_counts();
  auto completion = invocation.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(resource.invalidates(), 1);
  EXPECT_EQ(resource.flushes(), 1);
  EXPECT_EQ(resource.maps(), 2);
  EXPECT_EQ(resource.unmaps(), 2);

  InstrumentedResource other_resource;
  Buffer other_storage =
      Buffer::create(MemoryResourceRef(other_resource), bytes * 2, 64).value();
  ConstTensorView other_input =
      ConstTensorView::create(
          description, other_storage.view().subview(0, bytes).value())
          .value();
  TensorView other_output =
      TensorView::create(
          description, other_storage.view().subview(bytes, bytes).value())
          .value();
  auto wrong_exact_route =
      invocation.submit(queue, other_input, other_output);
  ASSERT_FALSE(wrong_exact_route.has_value());
  EXPECT_TRUE(wrong_exact_route.error().IsInvalidArgument());
  EXPECT_EQ(other_resource.maps(), 0);

  resource.fail_mapping(true);
  completion = invocation.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsUnavailable());

  resource.fail_mapping(false);
  resource.fail_unmap(true);
  completion = invocation.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsUnavailable());
}

TEST(PreparedOperation, RejectsAnActuallyMisalignedHostMapping) {
  constexpr std::size_t count = 8;
  const std::size_t bytes = count * sizeof(float);
  const TensorDesc description = row_description(count);
  InstrumentedResource resource;
  resource.misaligned_mapping(true);
  Buffer storage =
      Buffer::create(MemoryResourceRef(resource), bytes * 2, 64).value();
  ConstTensorView input =
      ConstTensorView::create(description,
                              storage.view().subview(0, bytes).value())
          .value();
  TensorView output =
      TensorView::create(description,
                         storage.view().subview(bytes, bytes).value())
          .value();
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  auto invocation =
      hyle::compute::prepare(context, RmsNorm{}, description,
                             MemoryResourceRef(resource))
          ->create_invocation()
          .value();

  auto completion = invocation.submit(queue, input, output);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsFailedPrecondition());
}

TEST(PreparedOperation, RejectsStorageThatCpuCannotMapDuringPrepare) {
  constexpr std::size_t count = 4;
  const std::size_t bytes = count * sizeof(float);
  const TensorDesc description = row_description(count);
  InstrumentedResource resource;
  resource.host_accessible(false);
  Buffer buffer =
      Buffer::create(MemoryResourceRef(resource), bytes * 2, 64).value();
  ConstTensorView input =
      ConstTensorView::create(description,
                              buffer.view().subview(0, bytes).value())
          .value();
  TensorView output =
      TensorView::create(description,
                         buffer.view().subview(bytes, bytes).value())
          .value();

  Context context = hyle::compute::cpu::create_context().value();
  const std::array inputs{input};
  const std::array outputs{output};
  auto rejected = hyle::compute::prepare(
      context, hyle::compute::Operation(RmsNorm{}), inputs, outputs);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  EXPECT_EQ(resource.maps(), 0);
}

TEST(PreparedOperation,
     ContextDestructionRejectsRetainedPreparedAndQueueHandles) {
  const TensorDesc description = row_description(0);
  Buffer empty = Buffer::create(hyle::mem::system_resource(), 0).value();
  ConstTensorView input =
      ConstTensorView::create(description, empty.view()).value();
  TensorView output = TensorView::create(description, empty.view()).value();
  Queue queue;
  PreparedOperation prepared;
  hyle::compute::OperationInvocation invocation;
  {
    Context context = hyle::compute::cpu::create_context().value();
    queue = context.create_queue().value();
    prepared =
        hyle::compute::prepare(context, RmsNorm{}, description)
            .value();
    invocation = prepared.create_invocation().value();
  }
  EXPECT_TRUE(prepared.valid());
  EXPECT_FALSE(queue.is_open());
  auto rejected = invocation.submit(queue, input, output);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
}

} // namespace
