#include "hyle/compute/providers/onednn.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::ContextOptions;
using hyle::compute::DataType;
using hyle::compute::MatMul;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::graph::GraphBuilder;
using hyle::mem::Buffer;

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array dimensions{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, dimensions).value();
}

TEST(OneDnnGraph, PlansProviderScratchpadAndExecutesWithoutHiddenFallback) {
  constexpr std::size_t rows = 16;
  constexpr std::size_t inner = 3;
  constexpr std::size_t columns = 4;
  const TensorDesc left_description = matrix(rows, inner);
  const TensorDesc right_description = matrix(inner, columns);
  const TensorDesc output_description = matrix(rows, columns);

  GraphBuilder builder = GraphBuilder::create().value();
  const auto left = builder.input(left_description).value();
  const auto right = builder.input(right_description).value();
  const auto output = builder.add(MatMul{}, left, right).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();

  const std::array providers{
      hyle::compute::ProviderBinding{hyle::compute::onednn::provider(), 100}};
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.provider().id, hyle::compute::onednn::kProviderId);
  EXPECT_FALSE(compiled.report.mixed_providers());
  EXPECT_EQ(compiled.report.node_count(), 1U);
  EXPECT_EQ(compiled.report.copy_bytes(), 0U);
  EXPECT_EQ(compiled.report.synchronization_count(), 0U);

  std::array<float, rows * inner> left_values{};
  std::array<float, inner * columns> right_values{};
  for (std::size_t index = 0; index < left_values.size(); ++index)
    left_values[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.25F;
  for (std::size_t index = 0; index < right_values.size(); ++index)
    right_values[index] =
        static_cast<float>(static_cast<int>(index % 5) - 2) * 0.5F;
  std::array<float, rows * columns> expected{};
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t k = 0; k < inner; ++k) {
        expected[row * columns + column] +=
            left_values[row * inner + k] * right_values[k * columns + column];
      }
    }
  }

  Buffer left_buffer =
      Buffer::create(hyle::mem::system_resource(), sizeof(left_values), 64)
          .value();
  Buffer right_buffer =
      Buffer::create(hyle::mem::system_resource(), sizeof(right_values), 64)
          .value();
  Buffer output_buffer =
      Buffer::create(hyle::mem::system_resource(), sizeof(expected), 64)
          .value();
  ASSERT_TRUE(
      left_buffer.view().write(std::as_bytes(std::span(left_values))).ok());
  ASSERT_TRUE(
      right_buffer.view().write(std::as_bytes(std::span(right_values))).ok());
  auto left_view =
      ConstTensorView::create(left_description, left_buffer.view()).value();
  auto right_view =
      ConstTensorView::create(right_description, right_buffer.view()).value();
  auto output_view =
      TensorView::create(output_description, output_buffer.view()).value();

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(left, left_view).ok());
  ASSERT_TRUE(invocation.bind(right, right_view).ok());
  ASSERT_TRUE(invocation.bind(output, output_view).ok());
  auto queue = context.create_queue().value();
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());

  std::array<float, rows * columns> actual{};
  ASSERT_TRUE(output_buffer.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 2.0e-5F + 2.0e-5F);
  }
}

} // namespace
