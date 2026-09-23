#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "../compute/workload_fixture.h"

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/ggml_cpu.h"
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

#if defined(HYLE_COMPUTE_GGML_CPU)
TEST(CpuDefaultContext, GgmlBindsThroughOrdinaryFactory) {
  auto context = create_context().value();
  const auto description = matrix(17, 31);
  auto prepared = prepare(context, Unary{UnaryKind::kNeg}, description);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().replay_class,
            ProviderReplayClass::kPrepared);
  EXPECT_FALSE(prepared->runtime_properties().uses_private_threads);
}

TEST(CpuDefaultContext, StandardIndexProgramsBindThroughOrdinaryFactory) {
  auto context = create_context().value();
  const auto scores = matrix(2, 9);
  auto topk = prepare(context, TopK{.count = 3}, scores);
  ASSERT_TRUE(topk) << topk.error().message();
  EXPECT_EQ(topk->provider().id, ggml_cpu::kProviderId);
  EXPECT_EQ(topk->output_count(), 2U);
  const auto counts = TensorDesc::dense(DataType::kInt32,
      std::array<std::int64_t, 1>{9}).value();
  auto scan = prepare(context, Scan{.exclusive = true}, counts);
  ASSERT_TRUE(scan) << scan.error().message();
  EXPECT_EQ(scan->provider().id, ggml_cpu::kProviderId);
  EXPECT_FALSE(scan->runtime_properties().uses_private_threads);
}

TEST(CpuDefaultContext, MatureScatterAddBindsThroughOrdinaryFactory) {
  auto context = create_context().value();
  const auto base = TensorDesc::dense(
                        DataType::kFloat16,
                        std::array<std::int64_t, 3>{2, 4, 8})
                        .value();
  const auto indices = TensorDesc::dense(
                           DataType::kInt32,
                           std::array<std::int64_t, 1>{3})
                           .value();
  const auto updates = TensorDesc::dense(
                           DataType::kFloat16,
                           std::array<std::int64_t, 3>{2, 3, 8})
                           .value();
  const std::array inputs{base, indices, updates};
  const std::array outputs{base};
  auto prepared = prepare(
      context, Operation(Scatter{.axis = 1, .kind = ScatterKind::kAdd}),
      inputs, outputs, hyle::mem::system_resource());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().replay_class,
            ProviderReplayClass::kPrepared);
}
#endif

#if defined(HYLE_COMPUTE_ONEDNN)
TEST(CpuDefaultContext, MatureNormalizationBindsThroughOrdinaryFactory) {
  auto context = create_context().value();
  for (const auto type :
       {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16}) {
    const auto description =
        TensorDesc::dense(type, std::array<std::int64_t, 3>{2, 17, 2048})
            .value();
    auto prepared = prepare(context, RmsNorm{}, description);
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto vendor_context =
        Context::create(cpu::device(), onednn::provider()).value();
    auto vendor = prepare(vendor_context, RmsNorm{}, description);
    if (type == DataType::kFloat32) {
      ASSERT_TRUE(vendor) << vendor.error().message();
    }
    if (vendor) {
      EXPECT_EQ(prepared->provider().id, onednn::kProviderId);
      EXPECT_EQ(prepared->runtime_properties().replay_class,
                ProviderReplayClass::kManaged);
    } else {
      ASSERT_TRUE(vendor.error().IsUnimplemented());
#if defined(HYLE_COMPUTE_GGML_CPU)
      EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
#else
      EXPECT_EQ(prepared->provider().id, native::kProviderId);
#endif
      EXPECT_EQ(prepared->runtime_properties().replay_class,
                ProviderReplayClass::kPrepared);
    }
    EXPECT_FALSE(prepared->runtime_properties().uses_private_threads);
  }
}
#endif

TEST(CpuDefaultContext, MatMulNeverFallsBackToLowerAccumulationPrecision) {
  auto context = create_context().value();
  for (const std::int64_t rows : {1, 16}) {
    for (const bool transposed : {false, true}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(transposed);
      const auto left = matrix(rows, 5);
      const auto right =
          transposed ? matrix(4, 5).transposed().value() : matrix(5, 4);
      auto result = prepare(context, MatMul{DataType::kFloat64}, left, right);
      ASSERT_FALSE(result.has_value());
      EXPECT_TRUE(result.error().IsUnimplemented());
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
      auto builder = hyle::graph::GraphBuilder::create().value();
      const auto a = builder.input(left).value();
      const auto b = builder.input(right).value();
      const auto output = builder.add(MatMul{DataType::kFloat64}, a, b).value();
      ASSERT_TRUE(builder.publish(output).ok());
      auto logical = std::move(builder).finish().value();
      auto compiled = hyle::graph::compile(logical, context);
      ASSERT_FALSE(compiled.has_value());
      EXPECT_TRUE(compiled.error().IsUnimplemented());
#endif
    }
  }
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
        auto workload = rows < 16 ? WorkloadContext{}
                                 : workload_test::known(WorkloadPhase::kPrefill);
        auto prepared = prepare(context, MatMul{}, left_desc, right_desc,
                                hyle::mem::system_resource(), workload);
        ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
        EXPECT_EQ(prepared->workload(), workload);
        workload = {};
        if (rows >= 16) {
          EXPECT_TRUE(prepared->workload().model.matches(workload_test::kModel));
        }
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

#if defined(HYLE_COMPUTE_GGML_CPU)
TEST(CpuDefaultContext, B2MatureDomainsBindThroughDefaultRuleSet) {
  auto context = create_context().value();
  const auto f16_left =
      TensorDesc::dense(DataType::kFloat16,
                        std::array<std::int64_t, 3>{2, 17, 64})
          .value();
  const auto f16_weight =
      TensorDesc::dense(DataType::kFloat16,
                        std::array<std::int64_t, 2>{64, 96})
          .value();
  auto linear = prepare(context, MatMul{}, f16_left, f16_weight);
  ASSERT_TRUE(linear) << linear.error().message();
  EXPECT_EQ(linear->provider().id, ggml_cpu::kProviderId);

  const auto rope_data =
      TensorDesc::dense(DataType::kFloat16,
                        std::array<std::int64_t, 4>{1, 17, 8, 256})
          .value();
  const auto positions =
      TensorDesc::dense(DataType::kInt32,
                        std::array<std::int64_t, 1>{17})
          .value();
  const auto frequencies =
      TensorDesc::dense(DataType::kFloat32,
                        std::array<std::int64_t, 1>{128})
          .value();
  auto rope = prepare(context,
                      RoPE{.rotary_dimensions = 256,
                           .sequence_axis = 1,
                           .layout = RopeLayout::kHalfSplit},
                      rope_data, positions, frequencies);
  ASSERT_TRUE(rope) << rope.error().message();
  EXPECT_EQ(rope->provider().id, ggml_cpu::kProviderId);

  const auto hadamard_input =
      TensorDesc::dense(DataType::kFloat32,
                        std::array<std::int64_t, 3>{2, 17, 256})
          .value();
  auto hadamard = prepare(context, Hadamard{}, hadamard_input);
  ASSERT_TRUE(hadamard) << hadamard.error().message();
  EXPECT_EQ(hadamard->provider().id, ggml_cpu::kProviderId);

  const auto query =
      TensorDesc::dense(DataType::kFloat32,
                        std::array<std::int64_t, 4>{1, 1, 8, 256})
          .value();
  const auto key_value =
      TensorDesc::dense(DataType::kFloat32,
                        std::array<std::int64_t, 4>{1, 31, 2, 256})
          .value();
  auto attention = prepare(
      context,
      ScaledDotProductAttention{
          .layout = AttentionLayout::kBatchSequenceHeadDepth,
          .scale = 0.0625F,
          .causal = true,
          .accumulation_type = DataType::kFloat32},
      query, key_value, key_value);
  ASSERT_TRUE(attention) << attention.error().message();
  EXPECT_EQ(attention->provider().id, ggml_cpu::kProviderId);
}
#endif

} // namespace

#if defined(HYLE_COMPUTE_ONEDNN)
namespace {
TEST(CpuDefaultContext, StableSoftplusUsesMatureBindingAcrossReplay) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context = create_context(stats.get()).value();
    auto queue = context.create_queue().value();
    const auto description = matrix(2, 4);
    auto input =
        hyle::mem::Buffer::create(hyle::mem::system_resource(), 32, 64).value();
    auto output =
        hyle::mem::Buffer::create(hyle::mem::system_resource(), 32, 64).value();
    auto program = prepare(context, Unary{UnaryKind::kSoftplus}, description);
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(program->provider().id, onednn::kProviderId);
    const auto requirement = program->workspace_requirement();
    auto workspace =
        requirement.size == 0
            ? hyle::mem::Buffer{}
            : hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                        requirement.size, requirement.alignment)
                  .value();
    auto invocation = program->create_invocation(workspace.view()).value();
    const auto source =
        ConstTensorView::create(description, input.view()).value();
    const auto result = TensorView::create(description, output.view()).value();
    const auto before = stats.snapshot();
    std::array<float, 8> values{-90, -40, -20, -1, -0.F, 1, 20, 100};
    for (int run = 0; run < 2; ++run) {
      ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
      auto done = invocation.submit(queue, source, result);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      std::array<float, 8> actual{};
      ASSERT_TRUE(
          output.view().read(std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const double x = values[i];
        const double expected =
            x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
        EXPECT_NEAR(actual[i], expected, 5e-6 * std::abs(expected) + 1e-43);
        values[i] += 0.25F;
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    ASSERT_TRUE(context.close().ok());
    auto closed = invocation.submit(queue, source, result);
    ASSERT_FALSE(closed);
    EXPECT_TRUE(closed.error().IsFailedPrecondition());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CpuDefaultContext, SelectUsesMatureProviderWithoutManualOrdering) {
  auto context = create_context().value();
  auto queue = context.create_queue().value();
  const auto condition =
      TensorDesc::dense(DataType::kBool, std::array<std::int64_t, 1>{4})
          .value();
  const auto values =
      TensorDesc::dense(DataType::kInt64, std::array<std::int64_t, 1>{4})
          .value();
  auto mask =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4, 64).value();
  auto a =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 32, 64).value();
  auto b =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 32, 64).value();
  auto out =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 32, 64).value();
  const std::array<std::uint8_t, 4> conditions{0, 1, 1, 0};
  const std::array<std::int64_t, 4> left{-1, 0x123456789abcdefLL, -7, 4};
  const std::array<std::int64_t, 4> right{9, 8, 7, -0x123456789abcdefLL};
  ASSERT_TRUE(mask.view().write(std::as_bytes(std::span(conditions))).ok());
  ASSERT_TRUE(a.view().write(std::as_bytes(std::span(left))).ok());
  ASSERT_TRUE(b.view().write(std::as_bytes(std::span(right))).ok());
  const std::array inputs{
      ConstTensorView::create(condition, mask.view()).value(),
      ConstTensorView::create(values, a.view()).value(),
      ConstTensorView::create(values, b.view()).value()};
  const std::array outputs{TensorView::create(values, out.view()).value()};
  auto prepared = prepare(context, Select{}, condition, values, values);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, onednn::kProviderId);
  const auto requirement = prepared->workspace_requirement();
  auto workspace =
      requirement.size == 0
          ? hyle::mem::Buffer{}
          : hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      requirement.size, requirement.alignment)
                .value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto done = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::array<std::int64_t, 4> actual{};
  ASSERT_TRUE(out.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array{right[0], left[1], left[2], right[3]}));
}
} // namespace
#endif
