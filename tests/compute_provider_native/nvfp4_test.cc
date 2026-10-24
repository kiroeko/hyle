#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;

std::vector<std::byte> read_bytes(const m::Buffer &buffer) {
  std::vector<std::byte> result(buffer.size());
  EXPECT_TRUE(buffer.view().read(result).ok());
  return result;
}

TEST(NativeNvfp4, RejectsCallerWorkspaceAliasingReadInputsBeforeAnyWrite) {
  const auto matrix = [](c::DataType type, std::int64_t rows,
                          std::int64_t columns) {
    return c::TensorDesc::dense(type, std::array{rows, columns}).value();
  };
  const auto scalar = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 0>{}).value();
  const std::array descriptions{
      matrix(c::DataType::kBFloat16, 1, 16),
      matrix(c::DataType::kUInt8, 1, 8),
      matrix(c::DataType::kFloat8E4M3FN, 1, 1), scalar, scalar};
  const std::array output_descriptions{matrix(c::DataType::kBFloat16, 1, 1)};
  auto context = c::test::native_context().value();
  auto prepared = c::prepare(context, c::model_ops::nvfp4_linear(),
      descriptions, output_descriptions, m::system_resource()).value();
  EXPECT_EQ(prepared.provider().id, c::native::kProviderId);
  const auto requirement = prepared.workspace_requirement();
  ASSERT_GT(requirement.size, 0U);

  std::array<m::Buffer, 5> storage;
  std::array<c::ConstTensorView, 5> inputs;
  for (std::size_t i = 0; i < storage.size(); ++i) {
    const auto bytes = descriptions[i].required_bytes().value();
    // Both alias candidates satisfy the prepared workspace size/alignment.
    // Rejection must detect overlap rather than an undersized scratch span.
    storage[i] = m::Buffer::create(m::system_resource(),
        i < 2 ? (std::max)(bytes, requirement.size) : bytes, 64).value();
    std::vector<std::byte> initial(storage[i].size(), std::byte{0x5a});
    ASSERT_TRUE(storage[i].view().write(initial).ok());
    inputs[i] = c::ConstTensorView::create(descriptions[i], storage[i].view()).value();
  }
  std::array<std::uint16_t, 16> activation;
  activation.fill(0x3fa0); // BF16 1.25; this QDQ boundary changes its bits.
  std::array<std::uint8_t, 8> weights;
  weights.fill(0x22); // Both adjacent low-first E2M1 values are +1.
  const std::array<std::uint8_t, 1> scales{0x38}; // E4M3 +1.
  const std::array<float, 1> weight_global{1.F}, input_global{1.F / 448.F};
  ASSERT_TRUE(storage[0].view().subview(0, sizeof(activation)).value()
      .write(std::as_bytes(std::span(activation))).ok());
  ASSERT_TRUE(storage[1].view().subview(0, sizeof(weights)).value()
      .write(std::as_bytes(std::span(weights))).ok());
  ASSERT_TRUE(storage[2].view().write(std::as_bytes(std::span(scales))).ok());
  ASSERT_TRUE(storage[3].view().write(std::as_bytes(std::span(weight_global))).ok());
  ASSERT_TRUE(storage[4].view().write(std::as_bytes(std::span(input_global))).ok());
  std::array<std::vector<std::byte>, 5> original;
  for (std::size_t i = 0; i < storage.size(); ++i) original[i] = read_bytes(storage[i]);
  auto destination = m::Buffer::create(m::system_resource(), 2, 64).value();
  const std::array outputs{
      c::TensorView::create(output_descriptions[0], destination.view()).value()};

  for (const std::size_t alias : {0U, 1U}) {
    SCOPED_TRACE(alias == 0 ? "activation workspace" : "packed-weight workspace");
    auto queue = context.create_queue().value();
    auto invocation = prepared.create_invocation(storage[alias].view()).value();
    auto done = invocation.submit(queue, inputs, outputs);
    if (done) {
      const auto status = done->wait();
      EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
    } else {
      EXPECT_TRUE(done.error().IsInvalidArgument()) << done.error().message();
    }
    for (std::size_t i = 0; i < storage.size(); ++i)
      EXPECT_EQ(read_bytes(storage[i]), original[i]) << "input " << i;
  }
  auto scratch = m::Buffer::create(m::system_resource(), requirement.size, 64).value();
  auto invocation = prepared.create_invocation(scratch.view()).value();
  auto queue = context.create_queue().value();
  auto done = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  // E4M3 scale96 gives BF16(6*96/448)=1.2890625, then16 ones sum20.625.
  std::array<std::uint16_t, 1> actual{};
  ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual[0], 0x41a5U);
  for (std::size_t i = 0; i < storage.size(); ++i)
    EXPECT_EQ(read_bytes(storage[i]), original[i]) << "input " << i;
}
} // namespace
