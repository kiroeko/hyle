#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;

template <class T> void relu(std::uint32_t threads) {
  constexpr std::int64_t rows = 3, width = 17;
  const auto dtype = half_test::data_type<T>();
  const auto output_desc = TensorDesc::dense(dtype, std::array{rows, width}).value();
  auto stats = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const std::array providers{ProviderBinding{
        ggml_cpu::provider({.native_threads = threads}), 100}};
    auto context = Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                                   .providers = providers}).value();
    auto queue = context.create_queue().value();
    for (const bool strided : {false, true}) {
      const auto source_desc = strided
          ? TensorDesc::strided_bytes(dtype, std::array{rows, width},
                                      std::array<std::int64_t, 2>{(width + 3) * sizeof(T), sizeof(T)}).value()
          : output_desc;
      std::vector<T> values(source_desc.required_bytes().value() / sizeof(T));
      auto input = half_test::buffer<T>(values);
      auto output = half_test::buffer<T>(std::vector<T>(rows * width));
      auto prepared = prepare(context, Unary{UnaryKind::kRelu}, source_desc);
      ASSERT_TRUE(prepared) << prepared.error().message();
      EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
      auto invocation = prepared->create_invocation().value();
      const auto source = ConstTensorView::create(source_desc, input.view()).value();
      for (const bool alias : {false, true}) {
        if (alias && strided)
          continue;
        auto destination = TensorView::create(output_desc, alias ? input.view() : output.view()).value();
        const auto allocations = stats.snapshot().total_allocs;
        for (const int seed : {0, 5, 0}) {
          for (std::int64_t row = 0; row < rows; ++row)
            for (std::int64_t col = 0; col < width; ++col) {
              const auto offset = static_cast<std::size_t>(row * (strided ? width + 3 : width) + col);
              constexpr auto inf = std::numeric_limits<double>::infinity();
              constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
              constexpr std::array special{nan, -inf, inf, -0., 0., -10000., 10000.};
              const auto x = col < 7 && seed == 0 ? special[col]
                  : static_cast<double>((row * 17 + col * 7 + seed) % 23 - 11) / 8.;
              values[offset] = half_test::encode<T>(x);
            }
          ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
          auto done = invocation.submit(queue, source, destination);
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          const auto actual = half_test::read<T>(alias ? input : output);
          for (std::int64_t row = 0; row < rows; ++row)
            for (std::int64_t col = 0; col < width; ++col) {
              const auto x = half_test::decode(values[row * (strided ? width + 3 : width) + col]);
              const auto y = half_test::decode(actual[row * width + col]);
              if (std::isnan(x)) {
                EXPECT_TRUE(std::isnan(y));
              } else {
                EXPECT_EQ(y, x > 0 ? x : 0);
                if (y == 0) {
                  EXPECT_FALSE(std::signbit(y));
                }
              }
            }
        }
        EXPECT_EQ(stats.snapshot().total_allocs, allocations);
      }
    }
    ASSERT_TRUE(context.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuProvider, MatureReluPropagatesNanAcrossStorageStrideAliasAndReplay) {
  relu<float>(1);
  relu<hyle::simd::float16_t>(1);
  relu<hyle::simd::bfloat16_t>(1);
}
TEST(GgmlCpuProvider, MatureReluKeepsOddParallelChunksAndReplay) {
  relu<float>(4);
  relu<hyle::simd::float16_t>(4);
  relu<hyle::simd::bfloat16_t>(4);
}

void batched_matmul(bool rank_four, bool broadcast_outer, bool broadcast_inner,
                    std::int64_t rows) {
  constexpr std::int64_t outer = 2, inner = 3, k = 7, n = 5;
  const auto lhs = rank_four ? TensorDesc::dense(DataType::kFloat32, std::array{outer, inner, rows, k}).value()
      : TensorDesc::dense(DataType::kFloat32, std::array{outer, rows, k}).value();
  const auto physical = rank_four ? TensorDesc::dense(DataType::kFloat32,
      std::array{broadcast_outer ? 1 : outer, broadcast_inner ? 1 : inner, n, k}).value()
      : broadcast_outer ? TensorDesc::dense(DataType::kFloat32, std::array{n, k}).value()
      : TensorDesc::dense(DataType::kFloat32, std::array{outer, n, k}).value();
  const auto rhs = physical.transposed().value();
  const std::array descriptions{lhs, rhs};
  const auto result = infer_single_output(Operation(MatMul{}), descriptions).value();
  std::vector<float> a(lhs.element_count().value()), b(physical.element_count().value());
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 16.F;
  auto left = half_test::buffer<float>(a), right = half_test::buffer<float>(b);
  auto output = half_test::buffer<float>(std::vector<float>(result.element_count().value()));
  auto context = Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 3})).value();
  auto queue = context.create_queue().value();
  auto prepared = prepare(context, Operation(MatMul{}), descriptions, std::span(&result, 1), hyle::mem::system_resource());
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto invocation = prepared->create_invocation().value();
  const std::array sources{ConstTensorView::create(lhs, left.view()).value(), ConstTensorView::create(rhs, right.view()).value()};
  const auto destination = TensorView::create(result, output.view()).value();
  const auto batch_inner = rank_four ? inner : 1;
  for (const int seed : {1, 9}) {
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = static_cast<float>((static_cast<int>(i) * 3 + seed) % 17 - 8) / 8.F;
    ASSERT_TRUE(left.view().write(std::as_bytes(std::span(a))).ok());
    auto done = invocation.submit(queue, sources, std::span(&destination, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    for (std::int64_t batch = 0; batch < outer * batch_inner; ++batch) {
      const auto b_outer = broadcast_outer ? 0 : batch / batch_inner;
      const auto b_inner = broadcast_inner ? 0 : batch % batch_inner;
      const auto weight_batch = b_outer * (rank_four && !broadcast_inner ? inner : 1) + b_inner;
      for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t col = 0; col < n; ++col) {
          double expected = 0;
          for (std::int64_t reduction = 0; reduction < k; ++reduction)
            expected += double(a[(batch * rows + row) * k + reduction]) * b[(weight_batch * n + col) * k + reduction];
          EXPECT_EQ(actual[(batch * rows + row) * n + col], expected);
        }
    }
  }
}

TEST(GgmlCpuProvider, F32BatchMatMulKeepsBothLeadingBroadcastAxesAndDecodeTail) {
  for (const std::int64_t rows : {1, 3}) {
    for (const bool broadcast_outer : {false, true}) {
      batched_matmul(false, broadcast_outer, true, rows);
      for (const bool broadcast_inner : {false, true})
        batched_matmul(true, broadcast_outer, broadcast_inner, rows);
    }
  }
}

TEST(GgmlCpuProvider, F32BatchMatMulDeclinesLeftBroadcastUnsupportedByOriginal) {
  const auto lhs = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 3>{1, 3, 7}).value();
  const auto rhs = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 3>{2, 5, 7}).value().transposed().value();
  const std::array sources{lhs, rhs};
  const auto output = infer_single_output(Operation(MatMul{}), sources).value();
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto prepared = prepare(context, Operation(MatMul{}), sources, std::span(&output, 1), hyle::mem::system_resource());
  ASSERT_FALSE(prepared);
  EXPECT_TRUE(prepared.error().IsUnimplemented());
}
} // namespace
