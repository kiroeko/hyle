#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#if defined(HYLE_COMPUTE_ONEDNN)
#include "hyle/compute/providers/onednn.h"
#endif
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;

auto create_context(hyle::mem::AllocatorRef allocator = {}) {
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  return hyle::graph::cpu::create_context({.metadata_allocator = allocator});
#else
  return hyle::compute::cpu::create_context({.metadata_allocator = allocator});
#endif
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  return TensorDesc::dense(DataType::kFloat32, std::array{rows, columns})
      .value();
}

TEST(CpuDefaultContext, ColdFailureReleasesPartialProviderAssembly) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context = create_context(stats.get());
    ASSERT_TRUE(context.has_value()) << context.error().message();
  }
  ASSERT_TRUE(stats.balanced());
  const auto required = stats.snapshot().peak_live_bytes;
  ASSERT_GT(required, 1U);
  for (const auto limit : {std::uint64_t{0}, required / 2, required - 1}) {
    SCOPED_TRACE(limit);
    hyle::mem::QuotaAllocator quota(stats.get(), limit);
    auto denied = create_context(quota.get());
    ASSERT_FALSE(denied.has_value());
    EXPECT_TRUE(denied.error().IsResourceExhausted());
    EXPECT_TRUE(stats.balanced());
  }
  {
    hyle::mem::QuotaAllocator quota(stats.get(), required);
    auto recovered = create_context(quota.get());
    ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CpuDefaultContext, MatMulRulesAndChangedInputReplay) {
  constexpr std::int64_t inner = 129, columns = 65;
  for (const std::int64_t rows : {1, 15, 16, 17}) {
    for (const bool transposed : {false, true}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(transposed);
      auto stats =
          hyle::mem::StatsAllocator::create(hyle::mem::system_allocator())
              .value();
      {
        auto context = create_context(stats.get()).value();
        auto queue = context.create_queue().value();
        const auto left_desc = matrix(rows, inner);
        const auto right_desc =
            transposed ? matrix(columns, inner).transposed().value()
                       : matrix(inner, columns);
        auto prepared = prepare(context, MatMul{}, left_desc, right_desc);
        ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
        ProviderId expected = native::kProviderId;
#if defined(HYLE_COMPUTE_ONEDNN)
        if (rows >= 16 && !transposed)
          expected = onednn::kProviderId;
#endif
        EXPECT_EQ(prepared->provider().id, expected);
        auto left =
            hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      left_desc.required_bytes().value(), 64)
                .value();
        auto right =
            hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      right_desc.required_bytes().value(), 64)
                .value();
        const auto output_desc = prepared->output_description().value();
        auto output =
            hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      output_desc.required_bytes().value(), 64)
                .value();
        hyle::mem::Buffer workspace;
        const auto requirement = prepared->workspace_requirement();
        if (requirement.size != 0) {
          workspace =
              hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                        requirement.size, requirement.alignment)
                  .value();
        }
        auto invocation = prepared->create_invocation(workspace.view()).value();
        const auto lhs =
            ConstTensorView::create(left_desc, left.view()).value();
        const auto rhs =
            ConstTensorView::create(right_desc, right.view()).value();
        const auto out = TensorView::create(output_desc, output.view()).value();
        std::vector<float> a(static_cast<std::size_t>(rows * inner));
        std::vector<float> b(static_cast<std::size_t>(inner * columns));
        std::vector<float> actual(static_cast<std::size_t>(rows * columns));
        for (std::size_t i = 0; i < b.size(); ++i)
          b[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 64.0F;
        ASSERT_TRUE(right.view().write(std::as_bytes(std::span(b))).ok());
        const auto before = stats.snapshot();
        for (int seed : {1, 7, 13}) {
          for (std::size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<float>((static_cast<int>(i % 23) + seed) % 23 -
                                      11) /
                   32.0F;
          ASSERT_TRUE(left.view().write(std::as_bytes(std::span(a))).ok());
          auto done = invocation.submit(queue, lhs, rhs, out);
          ASSERT_TRUE(done.has_value()) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          ASSERT_TRUE(output.view()
                          .read(std::as_writable_bytes(std::span(actual)))
                          .ok());
          for (std::int64_t row = 0; row < rows; ++row) {
            for (std::int64_t col = 0; col < columns; ++col) {
              double oracle = 0;
              for (std::int64_t k = 0; k < inner; ++k)
                oracle += static_cast<double>(a[row * inner + k]) *
                          b[transposed ? col * inner + k : k * columns + col];
              EXPECT_NEAR(actual[row * columns + col], oracle,
                          3.0e-5 * (1.0 + std::abs(oracle)));
            }
          }
        }
        EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
        EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
        ASSERT_TRUE(context.close().ok());
        auto closed = invocation.submit(queue, lhs, rhs, out);
        ASSERT_FALSE(closed.has_value());
        EXPECT_TRUE(closed.error().IsFailedPrecondition());
      }
      EXPECT_TRUE(stats.balanced());
    }
  }
}

TEST(CpuDefaultContext, InvalidAndUnsupportedInputsRemainErrors) {
  auto context = create_context().value();
  auto malformed = prepare(context, MatMul{}, matrix(16, 7), matrix(8, 9));
  ASSERT_FALSE(malformed.has_value());
  EXPECT_TRUE(malformed.error().IsInvalidArgument());
  const auto integer =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{16, 16})
          .value();
  auto unsupported = prepare(context, MatMul{}, integer, integer);
  ASSERT_FALSE(unsupported.has_value());
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  EXPECT_NE(unsupported.error().message().find("hyle-native"),
            std::string_view::npos);
}

} // namespace
