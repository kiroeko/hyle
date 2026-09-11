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

#include "hyle/compute/operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::Device;
using hyle::compute::OperationKind;
using hyle::compute::PreparedOperation;
using hyle::compute::ProviderKind;
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
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
  const std::array<std::int64_t, 1> shape{
      static_cast<std::int64_t>(count)};
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
  for (float value : input) sum += value * value;
  const float inverse =
      1.0F / std::sqrt(sum / static_cast<float>(input.size()) + epsilon);
  for (std::size_t index = 0; index < input.size(); ++index) {
    float expected = input[index] * inverse;
    if (!weight.empty()) expected *= weight[index];
    EXPECT_NEAR(actual[index], expected,
                std::abs(expected) * 3.0e-5F + 3.0e-6F);
  }
}

class InstrumentedResource final : public MemoryResource {
 public:
  std::string_view name() const noexcept override { return "compute.test"; }

  MemoryProperties properties() const noexcept override {
    MemoryProperty flags = MemoryProperty::kHostAccessible;
    if (coherent_) flags |= MemoryProperty::kHostDeviceCoherent;
    if (!host_accessible_) flags = MemoryProperty::kDeviceAccessible;
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
      return std::unexpected(Status::ResourceExhausted("test allocation failed"));
    }
    live_ = true;
    return ResourceAllocation{size == 0 ? nullptr : storage_.data(), size,
                              alignment};
  }

  void deallocate(ResourceAllocation) noexcept override { live_ = false; }

  Result<ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      HostAccess) noexcept override {
    ++maps_;
    if (fail_mapping_) {
      return std::unexpected(Status::Unavailable("test mapping failed"));
    }
    return ResourceHostMapping{
        size == 0 ? nullptr : static_cast<std::byte*>(native_handle) + offset,
        nullptr};
  }

  Status unmap_host(void*, void*, std::size_t, std::size_t,
                    HostAccess) noexcept override {
    ++unmaps_;
    return fail_unmap_ ? Status::Unavailable("test unmap failed")
                       : Status::Ok();
  }

  Status flush(void*, std::size_t, std::size_t) noexcept override {
    ++flushes_;
    return Status::Ok();
  }

  Status invalidate(void*, std::size_t, std::size_t) noexcept override {
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
  int maps_ = 0;
  int unmaps_ = 0;
  int flushes_ = 0;
  int invalidates_ = 0;
};

TEST(PreparedOperation, ExposesColdRouteAndShapeDiagnostics) {
  Context context = Context::create(Device::cpu()).value();
  const TensorDesc description = row_description(8);
  PreparedOperation prepared =
      hyle::compute::prepare(context, RmsNorm{}, description, description)
          .value();

  EXPECT_EQ(prepared.kind(), OperationKind::kRmsNorm);
  EXPECT_EQ(prepared.provider(), ProviderKind::kHyleNative);
  EXPECT_EQ(prepared.input_count(), 1u);
  EXPECT_EQ(prepared.output_count(), 1u);
  EXPECT_EQ(prepared.input_description().value(), description);
  EXPECT_EQ(prepared.output_description().value(), description);
  EXPECT_EQ(hyle::compute::operation_name(prepared.kind()), "rms-norm");
  EXPECT_EQ(hyle::compute::provider_name(prepared.provider()), "hyle-native");

  auto bad_input = prepared.input_description(1);
  ASSERT_FALSE(bad_input.has_value());
  EXPECT_TRUE(bad_input.error().IsOutOfRange());
  auto bad_output = prepared.output_description(1);
  ASSERT_FALSE(bad_output.has_value());
  EXPECT_TRUE(bad_output.error().IsOutOfRange());
}

TEST(PreparedOperation, RejectsUnsupportedOrInconsistentPreparation) {
  Context context = Context::create(Device::cpu()).value();
  const TensorDesc row = row_description(8);
  const TensorDesc other = row_description(7);

  auto negative = hyle::compute::prepare(
      context, RmsNorm{.epsilon = -1.0F}, row, row);
  ASSERT_FALSE(negative.has_value());
  EXPECT_TRUE(negative.error().IsInvalidArgument());

  auto nan = hyle::compute::prepare(
      context,
      RmsNorm{.epsilon = (std::numeric_limits<float>::quiet_NaN)()}, row, row);
  ASSERT_FALSE(nan.has_value());
  EXPECT_TRUE(nan.error().IsInvalidArgument());

  auto shape = hyle::compute::prepare(context, RmsNorm{}, row, other);
  ASSERT_FALSE(shape.has_value());
  EXPECT_TRUE(shape.error().IsInvalidArgument());

  const std::array<std::int64_t, 2> matrix_shape{2, 4};
  const TensorDesc matrix =
      TensorDesc::dense(DataType::kFloat32, matrix_shape).value();
  auto rank = hyle::compute::prepare(context, RmsNorm{}, matrix, matrix);
  ASSERT_FALSE(rank.has_value());
  EXPECT_TRUE(rank.error().IsUnimplemented());

  const std::array<std::int64_t, 1> shape8{8};
  const std::array<std::int64_t, 1> broadcast_stride{0};
  const TensorDesc broadcast =
      TensorDesc::strided(DataType::kFloat32, shape8, broadcast_stride).value();
  auto layout =
      hyle::compute::prepare(context, RmsNorm{}, broadcast, broadcast);
  ASSERT_FALSE(layout.has_value());
  EXPECT_TRUE(layout.error().IsUnimplemented());

  ASSERT_TRUE(context.close().ok());
  auto closed = hyle::compute::prepare(context, RmsNorm{}, row, row);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(PreparedOperation, RunsUnweightedWeightedAndExactInPlaceRmsNorm) {
  constexpr float epsilon = 1.0e-5F;
  const std::vector<float> input{1.0F, -2.0F, 3.0F, -4.0F, 0.5F, 6.0F};
  const std::vector<float> weight{0.5F, 1.0F, 1.5F, 2.0F, -1.0F, 0.25F};
  const std::size_t bytes = input.size() * sizeof(float);
  const TensorDesc description = row_description(input.size());

  Buffer input_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float)).value();
  Buffer weight_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float)).value();
  Buffer output_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float)).value();
  write_floats(input_buffer.view(), input);
  write_floats(weight_buffer.view(), weight);

  ConstTensorView input_view =
      ConstTensorView::create(description, input_buffer.view()).value();
  ConstTensorView weight_view =
      ConstTensorView::create(description, weight_buffer.view()).value();
  TensorView output_view =
      TensorView::create(description, output_buffer.view()).value();

  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  PreparedOperation plain =
      hyle::compute::prepare(context, RmsNorm{epsilon}, description, description)
          .value();
  auto completion = plain.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const auto plain_output = read_floats(output_buffer.view(), input.size());
  expect_rms_norm(input, {}, plain_output, epsilon);

  PreparedOperation weighted = hyle::compute::prepare(
      context, RmsNorm{epsilon}, description, description, description)
                                   .value();
  completion = weighted.submit(queue, input_view, weight_view, output_view);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const auto weighted_output = read_floats(output_buffer.view(), input.size());
  expect_rms_norm(input, weight, weighted_output, epsilon);

  write_floats(input_buffer.view(), input);
  TensorView in_place_mutable =
      TensorView::create(description, input_buffer.view()).value();
  completion = plain.submit(queue, in_place_mutable.as_const(), in_place_mutable);
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
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float)).value();
  ConstTensorView input = ConstTensorView::create(
                              description,
                              storage.view().subview(0, bytes).value())
                              .value();
  TensorView partial =
      TensorView::create(description,
                         storage.view().subview(sizeof(float), bytes).value())
          .value();
  TensorView output =
      TensorView::create(description, separate.view()).value();

  Context first = Context::create(Device::cpu()).value();
  Context second = Context::create(Device::cpu()).value();
  Queue first_queue = first.create_queue().value();
  Queue wrong_queue = second.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(first, RmsNorm{}, description, description).value();

  auto wrong_context = prepared.submit(wrong_queue, input, output);
  ASSERT_FALSE(wrong_context.has_value());
  EXPECT_TRUE(wrong_context.error().IsInvalidArgument());

  auto partial_alias = prepared.submit(first_queue, input, partial);
  ASSERT_FALSE(partial_alias.has_value());
  EXPECT_TRUE(partial_alias.error().IsInvalidArgument());

  PreparedOperation weighted = hyle::compute::prepare(
      first, RmsNorm{}, description, description, description).value();
  auto weight_alias = weighted.submit(first_queue, input, output.as_const(), output);
  ASSERT_FALSE(weight_alias.has_value());
  EXPECT_TRUE(weight_alias.error().IsInvalidArgument());

  ASSERT_TRUE(first.close().ok());
  auto closed = prepared.submit(first_queue, input, output);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(PreparedOperation, UsesExplicitCoherenceAndReportsProviderFailureInCompletion) {
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
      ConstTensorView::create(
          description, storage.view().subview(0, bytes).value()).value();
  TensorView output_view =
      TensorView::create(
          description, storage.view().subview(bytes, bytes).value()).value();

  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(context, RmsNorm{}, description, description)
          .value();

  resource.reset_counts();
  auto completion = prepared.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(resource.invalidates(), 1);
  EXPECT_EQ(resource.flushes(), 1);
  EXPECT_EQ(resource.maps(), 2);
  EXPECT_EQ(resource.unmaps(), 2);

  resource.fail_mapping(true);
  completion = prepared.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsUnavailable());

  resource.fail_mapping(false);
  resource.fail_unmap(true);
  completion = prepared.submit(queue, input_view, output_view);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsUnavailable());
}

TEST(PreparedOperation, RejectsStorageThatCpuCannotMap) {
  constexpr std::size_t count = 4;
  const std::size_t bytes = count * sizeof(float);
  const TensorDesc description = row_description(count);
  InstrumentedResource resource;
  resource.host_accessible(false);
  Buffer buffer =
      Buffer::create(MemoryResourceRef(resource), bytes * 2, 64).value();
  ConstTensorView input =
      ConstTensorView::create(
          description, buffer.view().subview(0, bytes).value()).value();
  TensorView output =
      TensorView::create(
          description, buffer.view().subview(bytes, bytes).value()).value();

  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(context, RmsNorm{}, description, description)
          .value();
  auto rejected = prepared.submit(queue, input, output);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(resource.maps(), 0);
}

TEST(PreparedOperation, ContextDestructionRejectsRetainedPreparedAndQueueHandles) {
  const TensorDesc description = row_description(0);
  Buffer empty = Buffer::create(hyle::mem::system_resource(), 0).value();
  ConstTensorView input =
      ConstTensorView::create(description, empty.view()).value();
  TensorView output = TensorView::create(description, empty.view()).value();
  Queue queue;
  PreparedOperation prepared;
  {
    Context context = Context::create(Device::cpu()).value();
    queue = context.create_queue().value();
    prepared =
        hyle::compute::prepare(context, RmsNorm{}, description, description)
            .value();
  }
  EXPECT_TRUE(prepared.valid());
  EXPECT_FALSE(queue.is_open());
  auto rejected = prepared.submit(queue, input, output);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
}

}  // namespace
