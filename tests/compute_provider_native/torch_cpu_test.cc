#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include <c10/core/CPUAllocator.h>

#include "half_test_support.h"
#include "../compute/native_context_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace mem = hyle::mem;

// Isolated test process only. Preserve native allocation and deleter ownership;
// fail one allocation through the real pinned CPU allocator's failure path.
class AllocationFailure final : public at::Allocator {
public:
  at::Allocator *original = c10::GetCPUAllocator();
  std::atomic<bool> fail{false};
  at::DataPtr allocate(std::size_t bytes) override {
    return original->allocate(fail.exchange(false) ? std::size_t{1} << 62 : bytes);
  }
  void copy_data(void *dest, const void *src, std::size_t bytes) const override {
    original->copy_data(dest, src, bytes);
  }
};

TEST(NativeTorchCpu, ManagedReplayReportsRealAllocationFailureAndReleasesProgram) {
  static AllocationFailure allocation;
  c10::SetCPUAllocator(&allocation, 255);
  mem::QuotaAllocator metadata(mem::system_allocator(), 4 * 1024 * 1024);
  const auto initial = metadata.remaining();
  {
    auto context = c::test::native_context({.metadata_allocator = metadata.get()}).value();
    const auto desc = c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array<std::int64_t, 2>{1, 128}).value();
    const std::array descriptors{desc};
    const auto op = c::model_ops::block_quantization_roundtrip(
        c::model_ops::BlockQuantizationFormat::kFloat8E4M3Block128);
    auto prepared = c::prepare(context, op, descriptors, descriptors, mem::system_resource()).value();
    EXPECT_EQ(prepared.provider().id, c::native::kProviderId);
    EXPECT_EQ(prepared.runtime_properties().replay_class, c::ProviderReplayClass::kManaged);
    EXPECT_TRUE(prepared.runtime_properties().uses_private_threads);
    EXPECT_TRUE(prepared.runtime_properties().blocks_host);
    EXPECT_FALSE(prepared.invocation_private_bytes());
    EXPECT_FALSE(prepared.capture_compatible());
    auto input = half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(128, 0x3f80));
    auto output = half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(128));
    const auto x = c::ConstTensorView::create(desc, input.view()).value();
    const auto y = c::TensorView::create(desc, output.view()).value();
    auto queue = context.create_queue().value();
    auto lane = prepared.create_invocation().value();
    allocation.fail = true;
    {
      auto done = lane.submit(queue, x, y);
      ASSERT_TRUE(done) << done.error().message();
      EXPECT_TRUE(done->wait().IsResourceExhausted());
    }
    EXPECT_FALSE(allocation.fail.load());
    EXPECT_TRUE(lane.submit(queue, x, y).error().IsFailedPrecondition());
    lane = prepared.create_invocation().value();
    for (int replay = 0; replay < 3; ++replay) {
      auto done = lane.submit(queue, x, y);
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().ok());
      for (auto value : half_test::read<std::uint16_t>(output)) EXPECT_EQ(value, 0x3f80);
    }
    {
      auto reserved = metadata.try_reserve_bytes(metadata.remaining()).value();
      auto failed = c::prepare(context, op, descriptors, descriptors, mem::system_resource());
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
    }
    EXPECT_TRUE(c::prepare(context, op, descriptors, descriptors, mem::system_resource()));
  }
  EXPECT_EQ(metadata.remaining(), initial);
}

TEST(NativeTorchCpu, DuplicateExpertIdsAndFailedInvocationRecovery) {
  auto context = c::test::native_context().value();
  auto queue = context.create_queue().value();
  const auto w = c::float8_grouped_weight_description(2, 128, 128).value();
  const auto a = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 1, 128}).value();
  const auto ids = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{1, 2}).value();
  const std::array inputs{w, a, ids};
  std::array<c::TensorDesc, 1> outputs;
  const auto op = c::model_ops::grouped_float8_linear(448.0F);
  ASSERT_TRUE(c::infer_output_descriptions(op, inputs, outputs).ok());
  auto prepared = c::prepare(context, op, inputs, outputs, mem::system_resource()).value();
  std::vector<std::byte> weights(w.required_bytes().value(), std::byte{0x38});
  const float scale = 1;
  std::memcpy(weights.data() + 2 * 128 * 128, &scale, 4);
  std::memcpy(weights.data() + 2 * 128 * 128 + 4, &scale, 4);
  auto wb = half_test::buffer<std::byte>(weights);
  auto ab = half_test::buffer<float>(std::vector<float>(128, 1));
  auto ib = half_test::buffer<std::int32_t>(std::vector<std::int32_t>{1, 1});
  auto ob = half_test::buffer<float>(std::vector<float>(256));
  const std::array x{c::ConstTensorView::create(w, wb.view()).value(),
      c::ConstTensorView::create(a, ab.view()).value(),
      c::ConstTensorView::create(ids, ib.view()).value()};
  const auto y = c::TensorView::create(outputs[0], ob.view()).value();
  auto lane = prepared.create_invocation().value();
  for (auto selected : {std::array<std::int32_t, 2>{1, 1}, {0, 1}, {0, 0}}) {
    ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = lane.submit(queue, x, std::span(&y, 1));
    ASSERT_TRUE(done);
    ASSERT_TRUE(done->wait().ok());
    for (auto value : half_test::read<float>(ob)) EXPECT_EQ(value, 128);
  }
  auto second = prepared.create_invocation().value();
  auto second_queue = context.create_queue().value();
  auto second_output = half_test::buffer<float>(std::vector<float>(256));
  const auto second_y = c::TensorView::create(outputs[0], second_output.view()).value();
  bool first_ok = false, second_ok = false;
  {
    std::jthread first_thread([&] {
      auto done = lane.submit(queue, x, std::span(&y, 1));
      first_ok = done && done->wait().ok();
    });
    std::jthread second_thread([&] {
      auto done = second.submit(second_queue, x, std::span(&second_y, 1));
      second_ok = done && done->wait().ok();
    });
  }
  EXPECT_TRUE(first_ok);
  EXPECT_TRUE(second_ok);
  for (auto value : half_test::read<float>(second_output)) EXPECT_EQ(value, 128);
  for (auto selected : {std::array<std::int32_t, 2>{-1, 0}, {0, 2}}) {
    ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = lane.submit(queue, x, std::span(&y, 1));
    ASSERT_TRUE(done);
    EXPECT_TRUE(done->wait().IsOutOfRange());
    lane = prepared.create_invocation().value();
  }
  const std::array<std::int32_t, 2> selected{0, 1};
  ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(selected))).ok());
  auto nonfinite = std::vector<float>(128, 1);
  nonfinite[127] = std::numeric_limits<float>::infinity();
  ASSERT_TRUE(ab.view().write(std::as_bytes(std::span(nonfinite))).ok());
  auto done = lane.submit(queue, x, std::span(&y, 1));
  ASSERT_TRUE(done);
  EXPECT_TRUE(done->wait().IsInvalidArgument());
}
} // namespace
