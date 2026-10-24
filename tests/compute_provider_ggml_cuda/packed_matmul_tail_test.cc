#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/mem/buffer.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;

void check_q8_tail(std::int64_t columns, std::int64_t rows) {
  auto context = c::Context::create(c::cuda::device(0).value(),
                                    c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t width = 512;
  constexpr std::size_t encoded_row_bytes = 16 * 34;
  const auto a_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array{rows, width}).value();
  const auto w_desc = c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32, width, columns).value();
  const auto y_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array{rows, columns}).value();
  ASSERT_EQ(w_desc.required_bytes().value(), std::size_t(columns) * encoded_row_bytes);
  std::vector<std::byte> encoded(w_desc.required_bytes().value(), std::byte{0});
  for (std::int64_t column = 0; column < columns; ++column) {
    for (std::size_t block = 0; block < 16; ++block) {
      const auto offset = std::size_t(column) * encoded_row_bytes + block * 34;
      encoded[offset + 1] = std::byte{0x3c}; // original F16 scale1
      std::fill_n(encoded.begin() + offset + 2, 32, std::byte(column + 1));
    }
  }
  std::vector<float> input(std::size_t(rows * width));
  for (std::int64_t row = 0; row < rows; ++row)
    std::fill_n(input.begin() + row * width, std::size_t(width), row == 0 ? 127.F : -127.F);
  auto weights = m::Buffer::create(memory.ref(), encoded.size(), 256).value();
  auto activation = m::Buffer::create(memory.ref(), a_desc.required_bytes().value(), 256).value();
  auto output = m::Buffer::create(memory.ref(), y_desc.required_bytes().value(), 256).value();
  ASSERT_TRUE(weights.write(0, encoded).ok());
  ASSERT_TRUE(activation.write(0, std::as_bytes(std::span(input))).ok());
  ASSERT_TRUE(output.fill(0, output.size(), std::byte{0x7f}).ok());
  auto prepared = c::prepare(context, c::Operation(c::MatMul{}),
      std::array{a_desc, w_desc}, std::array{y_desc}, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cuda::kProviderId);
  const auto requirement = prepared->workspace_requirement();
  ASSERT_LT(requirement.size, 1024U * 1024U);
  m::Buffer workspace;
  if (requirement.size != 0)
    workspace = m::Buffer::create(memory.ref(), requirement.size,
        std::max(std::size_t{256}, requirement.alignment)).value();
  auto invocation = prepared->create_invocation(workspace.view());
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  const auto a = c::ConstTensorView::create(a_desc, activation.view()).value();
  const auto w = c::ConstTensorView::create(w_desc, weights.view()).value();
  const auto y = c::TensorView::create(y_desc, output.view()).value();
  auto completion = invocation->submit(queue, a, w, y);
  ASSERT_TRUE(completion) << completion.error().message();
  const auto status = completion->wait();
  ASSERT_TRUE(status.ok()) << status.message();
  std::vector<float> actual(std::size_t(rows * columns));
  ASSERT_TRUE(output.read(0, std::as_writable_bytes(std::span(actual))).ok());
  // Uniform +/-127 produces original AQ8_1 scale1 and codes+/-127.
  // Every dot is an exact integer below2^24; no tolerance is needed.
  for (std::int64_t row = 0; row < rows; ++row)
    for (std::int64_t column = 0; column < columns; ++column)
      EXPECT_EQ(actual[std::size_t(row * columns + column)],
                (row == 0 ? 1.F : -1.F) * 65024.F * float(column + 1));
  ::testing::Test::RecordProperty("generated_original_weight_bytes", std::to_string(encoded.size()));
  ::testing::Test::RecordProperty("workspace_bytes", std::to_string(requirement.size));
  ::testing::Test::RecordProperty("scope", "Public MatMul numeric tail regression; allocation slack is not a memcheck proof");
}

TEST(GgmlCudaPackedTail, Q8K512N1M1) { check_q8_tail(1, 1); }
TEST(GgmlCudaPackedTail, Q8K512N3M1) { check_q8_tail(3, 1); }
TEST(GgmlCudaPackedTail, Q8K512N3M2) { check_q8_tail(3, 2); }
} // namespace
