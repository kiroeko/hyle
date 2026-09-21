#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
using hyle::mem::Buffer;

struct Tensor final {
  Buffer buffer;
  TensorView view;
};
Tensor tensor(TensorDesc description) {
  auto bytes = description.required_bytes().value();
  auto buffer = Buffer::create(hyle::mem::system_resource(),
                               std::max<std::size_t>(bytes, 1), 64)
                    .value();
  auto view = TensorView::create(description, buffer.view()).value();
  return {std::move(buffer), view};
}
Buffer scratch(const PreparedOperation &program) {
  auto requirement = program.workspace_requirement();
  return requirement.size == 0
             ? Buffer{}
             : Buffer::create(hyle::mem::system_resource(), requirement.size,
                              requirement.alignment)
                   .value();
}

TEST(OneDnnTensor, CopyPreservesEveryStorageBitWithOddByteStrides) {
  auto context = Context::create(cpu::device(), onednn::provider()).value();
  auto queue = context.create_queue().value();
  for (int value = static_cast<int>(DataType::kBool);
       value <= static_cast<int>(DataType::kUInt64); ++value) {
    const auto type = static_cast<DataType>(value);
    SCOPED_TRACE(value);
    const auto width = data_type_size(type);
    const std::int64_t count = width == 2 ? 65536 : 257;
    const auto step = static_cast<std::int64_t>(width + 3);
    const auto input_description =
        TensorDesc::strided_bytes(type, std::array{count}, std::array{step})
            .value();
    const auto output_description =
        TensorDesc::dense(type, std::array{count}).value();
    auto input = tensor(input_description), output = tensor(output_description);
    std::vector<std::uint8_t> source(
        input_description.required_bytes().value());
    std::vector<std::uint8_t> expected(static_cast<std::size_t>(count) * width);
    for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
      const auto bits = static_cast<std::uint64_t>(i) *
                        (width == 2 ? 1 : 0x9e3779b97f4a7c15ULL);
      std::memcpy(source.data() + i * static_cast<std::size_t>(step), &bits,
                  width);
      std::memcpy(expected.data() + i * width, &bits, width);
    }
    ASSERT_TRUE(
        input.buffer.view().write(std::as_bytes(std::span(source))).ok());
    const std::array inputs{input.view.as_const()};
    const std::array outputs{output.view};
    auto program = prepare(context, Copy{}, input_description);
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(program->provider().id, onednn::kProviderId);
    auto workspace = scratch(*program);
    auto invocation = program->create_invocation(workspace.view()).value();
    auto done = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<std::uint8_t> actual(expected.size());
    ASSERT_TRUE(output.buffer.view()
                    .read(std::as_writable_bytes(std::span(actual)))
                    .ok());
    EXPECT_EQ(actual, expected);
  }
}

TEST(OneDnnTensor, SelectBroadcastsStorageBitsAndReusesIndependentRequests) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  for (int value = static_cast<int>(DataType::kBool);
       value <= static_cast<int>(DataType::kUInt64); ++value) {
    SCOPED_TRACE(value);
    const auto type = static_cast<DataType>(value);
    const auto width = data_type_size(type);
    const auto step = static_cast<std::int64_t>(width + 3);
    {
      const std::array providers{onednn::provider()};
      auto context =
          Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                          .providers = providers})
              .value();
      auto queue = context.create_queue().value();
      const auto mask_description =
          TensorDesc::strided_bytes(DataType::kBool,
                                    std::array<std::int64_t, 2>{2, 3},
                                    std::array<std::int64_t, 2>{7, 2})
              .value();
      const auto a_description =
          TensorDesc::strided_bytes(type, std::array<std::int64_t, 2>{2, 1},
                                    std::array{step, step})
              .value();
      const auto b_description =
          TensorDesc::strided_bytes(type, std::array<std::int64_t, 2>{1, 3},
                                    std::array{3 * step, step})
              .value();
      const auto output_description =
          TensorDesc::dense(type, std::array<std::int64_t, 2>{2, 3}).value();
      auto mask = tensor(mask_description), a = tensor(a_description),
           b = tensor(b_description);
      std::array outputs{tensor(output_description),
                         tensor(output_description)};
      std::vector<std::uint8_t> a_bytes(a_description.required_bytes().value(),
                                        0xfa);
      std::vector<std::uint8_t> b_bytes(b_description.required_bytes().value(),
                                        0x81);
      const std::array<std::uint64_t, 3> patterns{
          0xfff123457fa12345ULL, 0x8000000080008000ULL, 0x123456783f800000ULL};
      for (std::size_t i = 0; i < 2; ++i)
        std::memcpy(a_bytes.data() + i * static_cast<std::size_t>(step),
                    &patterns[i], width);
      for (std::size_t i = 0; i < 3; ++i)
        std::memcpy(b_bytes.data() + i * static_cast<std::size_t>(step),
                    &patterns[2 - i], width);
      ASSERT_TRUE(
          a.buffer.view().write(std::as_bytes(std::span(a_bytes))).ok());
      ASSERT_TRUE(
          b.buffer.view().write(std::as_bytes(std::span(b_bytes))).ok());
      const std::array inputs{mask.view.as_const(), a.view.as_const(),
                              b.view.as_const()};
      auto program = prepare(context, Select{}, mask_description, a_description,
                             b_description);
      ASSERT_TRUE(program) << program.error().message();
      EXPECT_EQ(program->provider().id, onednn::kProviderId);
      std::array workspaces{scratch(*program), scratch(*program)};
      std::array invocations{
          program->create_invocation(workspaces[0].view()).value(),
          program->create_invocation(workspaces[1].view()).value()};
      const auto allocations = stats.snapshot().total_allocs;
      std::array<std::vector<std::uint8_t>, 2> previous;
      for (std::size_t run = 0; run < 4; ++run) {
        std::vector<std::uint8_t> conditions(
            mask_description.required_bytes().value(), 0);
        std::vector<std::uint8_t> expected(6 * width);
        for (std::size_t row = 0; row < 2; ++row)
          for (std::size_t col = 0; col < 3; ++col) {
            bool choose = (row + col + run) % 2 != 0;
            conditions[row * 7 + col * 2] = static_cast<std::uint8_t>(choose);
            const auto *source =
                choose ? a_bytes.data() + row * static_cast<std::size_t>(step)
                       : b_bytes.data() + col * static_cast<std::size_t>(step);
            std::memcpy(expected.data() + (row * 3 + col) * width, source,
                        width);
          }
        ASSERT_TRUE(mask.buffer.view()
                        .write(std::as_bytes(std::span(conditions)))
                        .ok());
        auto done = invocations[run % 2].submit(
            queue, inputs, std::array{outputs[run % 2].view});
        ASSERT_TRUE(done) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        std::vector<std::uint8_t> actual(expected.size());
        ASSERT_TRUE(outputs[run % 2]
                        .buffer.view()
                        .read(std::as_writable_bytes(std::span(actual)))
                        .ok());
        EXPECT_EQ(actual, expected);
        previous[run % 2] = actual;
        if (run != 0) {
          ASSERT_TRUE(outputs[(run + 1) % 2]
                          .buffer.view()
                          .read(std::as_writable_bytes(std::span(actual)))
                          .ok());
          EXPECT_EQ(actual, previous[(run + 1) % 2]);
        }
      }
      EXPECT_EQ(stats.snapshot().total_allocs, allocations);
    }
    EXPECT_TRUE(stats.balanced());
  }
}

TEST(OneDnnTensor, ScalarBranchesRankEightAndEmptyShapes) {
  auto context = Context::create(cpu::device(), onednn::provider()).value();
  auto queue = context.create_queue().value();
  for (bool empty : {false, true}) {
    SCOPED_TRACE(empty);
    const std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, empty ? 0 : 2, 3};
    auto mask_description = TensorDesc::dense(DataType::kBool, shape).value();
    auto scalar = TensorDesc::dense(DataType::kUInt64, {}).value();
    auto output_description =
        TensorDesc::dense(DataType::kUInt64, shape).value();
    auto mask = tensor(mask_description), a = tensor(scalar),
         b = tensor(scalar);
    auto output = tensor(output_description);
    const std::array<std::uint64_t, 1> left{0xfedcba9876543210ULL},
        right{0xabcdef0123456789ULL};
    const std::array<std::uint8_t, 6> conditions{0, 1, 1, 0, 1, 0};
    if (!empty)
      ASSERT_TRUE(
          mask.buffer.view().write(std::as_bytes(std::span(conditions))).ok());
    ASSERT_TRUE(a.buffer.view().write(std::as_bytes(std::span(left))).ok());
    ASSERT_TRUE(b.buffer.view().write(std::as_bytes(std::span(right))).ok());
    auto program = prepare(context, Select{}, mask_description, scalar, scalar);
    ASSERT_TRUE(program) << program.error().message();
    auto workspace = scratch(*program);
    auto invocation = program->create_invocation(workspace.view()).value();
    const std::array inputs{mask.view.as_const(), a.view.as_const(),
                            b.view.as_const()};
    auto done = invocation.submit(queue, inputs, std::array{output.view});
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    if (!empty) {
      std::array<std::uint64_t, 6> actual{};
      ASSERT_TRUE(output.buffer.view()
                      .read(std::as_writable_bytes(std::span(actual)))
                      .ok());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(actual[i], conditions[i] ? left[0] : right[0]);
    }
    auto copy = prepare(context, Copy{}, output_description);
    ASSERT_TRUE(copy) << copy.error().message();
    auto copied = tensor(output_description);
    auto copy_scratch = scratch(*copy);
    auto copy_invocation = copy->create_invocation(copy_scratch.view()).value();
    auto copied_done =
        copy_invocation.submit(queue, output.view.as_const(), copied.view);
    ASSERT_TRUE(copied_done) << copied_done.error().message();
    ASSERT_TRUE(copied_done->wait().ok());
  }
}

TEST(OneDnnTensor, OverlapIsRejectedBeforeOutputChanges) {
  auto context = Context::create(cpu::device(), onednn::provider()).value();
  auto queue = context.create_queue().value();
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4})
          .value();
  auto value = tensor(description);
  const std::array<std::uint32_t, 4> bits{0x7fa12345, 0xff800000, 0x80000000,
                                          0x01234567};
  ASSERT_TRUE(value.buffer.view().write(std::as_bytes(std::span(bits))).ok());
  auto copy = prepare(context, Copy{}, description).value();
  auto workspace = scratch(copy);
  auto invocation = copy.create_invocation(workspace.view()).value();
  auto rejected = invocation.submit(queue, value.view.as_const(), value.view);
  ASSERT_FALSE(rejected);
  std::array<std::uint32_t, 4> after{};
  ASSERT_TRUE(
      value.buffer.view().read(std::as_writable_bytes(std::span(after))).ok());
  EXPECT_EQ(after, bits);
}
} // namespace
