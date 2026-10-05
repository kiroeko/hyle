#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
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
template <class T> constexpr double tolerance(double f32, double f16, double bf16) {
  if constexpr (std::is_same_v<T, float>) return f32;
  else if constexpr (std::is_same_v<T, hyle::simd::float16_t>) return f16;
  else return bf16;
}

template <class T> void softmax_batch() {
  const auto type = half_test::data_type<T>();
  auto stats = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const std::array providers{ProviderBinding{ggml_cpu::provider({.native_threads = 4}), 100}};
    auto context = Context::create(cpu::device(), {.metadata_allocator = stats.get(), .providers = providers}).value();
    auto queue = context.create_queue().value();
    for (const auto &shape : {std::vector<std::int64_t>{7}, {3, 7}, {2, 2, 3, 7}}) {
      const auto desc = TensorDesc::dense(type, shape).value();
      const auto count = desc.element_count().value();
      for (const bool masked : {false, true}) {
        // Nontrivial leading and row broadcasts use the original layout lowering.
        const auto mask_desc = shape.size() == 4
            ? TensorDesc::dense(type, std::array<std::int64_t, 4>{2, 1, 3, 1}).value()
            : TensorDesc::dense(type, std::array<std::int64_t, 1>{7}).value();
        std::vector<T> values(count), masks(mask_desc.element_count().value());
        auto input = half_test::buffer<T>(values), mask = half_test::buffer<T>(masks);
        auto output = half_test::buffer<T>(std::vector<T>(count));
        const std::array descriptions{desc, mask_desc};
        const auto operands = std::span(descriptions).first(masked ? 2 : 1);
        const Softmax operation{.axis = static_cast<std::int32_t>(shape.size() - 1),
                                .scale = 0.5F, .mask = masked ? MaskKind::kAdditive : MaskKind::kNone};
        auto prepared = prepare(context, Operation(operation), operands, std::span(&desc, 1), hyle::mem::system_resource());
        ASSERT_TRUE(prepared) << prepared.error().message();
        EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
        auto invocation = prepared->create_invocation().value();
        const std::array sources{ConstTensorView::create(desc, input.view()).value(),
                                 ConstTensorView::create(mask_desc, mask.view()).value()};
        const auto destination = TensorView::create(desc, output.view()).value();
        const auto allocs = stats.snapshot().total_allocs;
        for (const int seed : {0, 1, 2, 3, 4, 0}) {
          for (std::size_t i = 0; i < count; ++i)
            values[i] = half_test::encode<T>(static_cast<double>(static_cast<int>(i % 11) - 5) / 8.);
          for (std::size_t i = 0; i < masks.size(); ++i)
            masks[i] = half_test::encode<T>(shape.size() == 4 ? double(i % 3) / 4. : (i == 0 ? -2. : 0.));
          if (seed == 1) values[0] = half_test::encode<T>(std::numeric_limits<double>::quiet_NaN());
          if (seed == 2) values[0] = half_test::encode<T>(INFINITY);
          if (seed == 3) std::fill_n(values.begin(), 7, half_test::encode<T>(-INFINITY));
          if (seed == 4 && masked) std::fill(masks.begin(), masks.end(), half_test::encode<T>(-INFINITY));
          ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
          ASSERT_TRUE(mask.view().write(std::as_bytes(std::span(masks))).ok());
          auto done = invocation.submit(queue, std::span(sources).first(masked ? 2 : 1), std::span(&destination, 1));
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          const auto actual = half_test::read<T>(output);
          for (std::size_t row = 0; row < count / 7; ++row) {
            if ((row == 0 && seed >= 1 && seed <= 3) || (seed == 4 && masked)) {
              for (std::size_t col = 0; col < 7; ++col) EXPECT_TRUE(std::isnan(half_test::decode(actual[row * 7 + col])));
              continue;
            }
            std::array<double, 7> scores{};
            double sum = 0;
            for (std::size_t col = 0; col < 7; ++col) {
              const auto mi = shape.size() == 4 ? (row / 6) * 3 + row % 3 : col;
              scores[col] = std::exp(half_test::decode(values[row * 7 + col]) * 0.5 +
                                      (masked ? half_test::decode(masks[mi]) : 0.));
              sum += scores[col];
            }
            for (std::size_t col = 0; col < 7; ++col)
              EXPECT_NEAR(half_test::decode(actual[row * 7 + col]), scores[col] / sum, tolerance<T>(2.e-6, 2.e-4, 2.e-3));
          }
        }
        EXPECT_EQ(stats.snapshot().total_allocs, allocs);
        const auto aliased = TensorView::create(desc, input.view()).value();
        auto denied = invocation.submit(queue, std::span(sources).first(masked ? 2 : 1), std::span(&aliased, 1));
        ASSERT_FALSE(denied);
        EXPECT_TRUE(denied.error().IsInvalidArgument());
      }
    }
    ASSERT_TRUE(context.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuProvider, TransformerSoftmaxBroadcastScaleStorageUndefinedRowsAndReplay) {
  softmax_batch<float>();
  softmax_batch<hyle::simd::float16_t>();
  softmax_batch<hyle::simd::bfloat16_t>();
}

TEST(GgmlCpuProvider, SoftmaxZeroScaleAndResidualContractsRemainExplicit) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const auto desc = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{2, 7}).value();
  std::vector<float> values(14, 3.F);
  values[0] = INFINITY; // Zero times infinity is still undefined, not uniform.
  auto input = half_test::buffer<float>(values), output = half_test::buffer<float>(std::vector<float>(14));
  auto prepared = prepare(context, Softmax{.scale = 0.F}, desc).value();
  auto invocation = prepared.create_invocation().value();
  auto source = ConstTensorView::create(desc, input.view()).value();
  auto destination = TensorView::create(desc, output.view()).value();
  auto done = invocation.submit(queue, source, destination).value();
  ASSERT_TRUE(done.wait().ok());
  const auto result = half_test::read<float>(output);
  for (std::size_t i = 0; i < 7; ++i) EXPECT_TRUE(std::isnan(result[i]));
  for (std::size_t i = 7; i < 14; ++i) EXPECT_NEAR(result[i], 1.F / 7.F, 2.e-7);
  for (const auto operation : {Softmax{.axis = 0}, Softmax{.kind = SoftmaxKind::kLogSoftmax}}) {
    auto rejected = prepare(context, operation, desc);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
  const auto bd = TensorDesc::dense(DataType::kBool, std::array<std::int64_t, 1>{7}).value();
  auto boolean = prepare(context, Softmax{.mask = MaskKind::kBoolean}, desc, bd);
  ASSERT_FALSE(boolean);
  EXPECT_TRUE(boolean.error().IsUnimplemented());
}

template <class T> void fill_batch() {
  const auto type = half_test::data_type<T>();
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto queue = context.create_queue().value();
  for (const auto &shape : {std::vector<std::int64_t>{}, {0, 7}, {2, 3, 7}}) {
    const auto desc = TensorDesc::dense(type, shape).value();
    for (const std::uint32_t bits : {0U, 0x80000000U, 0x7f800000U, 0x7fc12345U, 0x3fa00000U}) {
      // Include native quiet NaN payloads; no numeric roundtrip is permitted.
      const auto payload = type == DataType::kFloat32 ? bits : type == DataType::kBFloat16
          ? bits >> 16 : bits == 0x80000000U ? 0x8000U : bits == 0x7f800000U ? 0x7c00U
          : bits == 0x7fc12345U ? 0x7e25U : bits == 0x3fa00000U ? 0x3d00U : 0U;
      const Fill operation{.output_shape = TensorShape::create(shape).value(), .output_type = type,
                            .value = ScalarValue::from_bits(type, payload)};
      auto prepared = prepare(context, operation, hyle::mem::system_resource());
      ASSERT_TRUE(prepared) << prepared.error().message();
      EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
      auto invocation = prepared->create_invocation().value();
      auto buffer = half_test::buffer<T>(std::vector<T>(desc.element_count().value()));
      const auto destination = TensorView::create(desc, buffer.view()).value();
      for (int repeat = 0; repeat < 2; ++repeat) {
        auto done = invocation.submit(queue, {}, std::span(&destination, 1));
        ASSERT_TRUE(done) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        for (const auto value : half_test::read<T>(buffer)) {
          if constexpr (std::is_same_v<T, float>) EXPECT_EQ(std::bit_cast<std::uint32_t>(value), payload);
          else EXPECT_EQ(value.bits, payload);
        }
      }
    }
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(GgmlCpuProvider, MatureFillHasNoInputsAndPreservesExactStorageBits) {
  fill_batch<float>();
  fill_batch<hyle::simd::float16_t>();
  fill_batch<hyle::simd::bfloat16_t>();
}

template <class T> void noncausal_attention() {
  constexpr std::int64_t batch = 2, nq = 3, nk = 2, heads = 4, kv_heads = 2, depth = 7;
  const auto type = half_test::data_type<T>();
  const auto qd = TensorDesc::dense(type, std::array{batch, nq, heads, depth}).value();
  const auto kd = TensorDesc::dense(type, std::array{batch, nk, kv_heads, depth}).value();
  auto context = Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 3})).value();
  auto queue = context.create_queue().value();
  const std::array descriptions{qd, kd, kd};
  const ScaledDotProductAttention operation{.scale = 0.5F};
  auto prepared = prepare(context, Operation(operation), descriptions, std::span(&qd, 1), hyle::mem::system_resource());
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto invocation = prepared->create_invocation().value();
  std::vector<T> q(qd.element_count().value()), k(kd.element_count().value()), v(k.size());
  auto qb = half_test::buffer<T>(q), kb = half_test::buffer<T>(k), vb = half_test::buffer<T>(v);
  auto result = half_test::buffer<T>(std::vector<T>(q.size()));
  const std::array sources{ConstTensorView::create(qd, qb.view()).value(), ConstTensorView::create(kd, kb.view()).value(), ConstTensorView::create(kd, vb.view()).value()};
  const auto destination = TensorView::create(qd, result.view()).value();
  for (int seed : {1, 3}) {
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = half_test::encode<T>(double((int(i) + seed) % 9 - 4) / 8.);
    for (std::size_t i = 0; i < k.size(); ++i) {
      k[i] = half_test::encode<T>(double((int(i) * 3 + seed) % 11 - 5) / 8.);
      v[i] = half_test::encode<T>(double((int(i) * 7 + seed) % 13 - 6) / 4.);
    }
    ASSERT_TRUE(qb.view().write(std::as_bytes(std::span(q))).ok());
    ASSERT_TRUE(kb.view().write(std::as_bytes(std::span(k))).ok());
    ASSERT_TRUE(vb.view().write(std::as_bytes(std::span(v))).ok());
    auto done = invocation.submit(queue, sources, std::span(&destination, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<T>(result);
    for (std::int64_t b = 0; b < batch; ++b) for (std::int64_t row = 0; row < nq; ++row)
      for (std::int64_t h = 0; h < heads; ++h) {
        std::array<double, nk> weights{};
        double sum = 0;
        const auto kh = h / (heads / kv_heads);
        for (std::int64_t key = 0; key < nk; ++key) {
          double score = 0;
          for (std::int64_t d = 0; d < depth; ++d)
            score += half_test::decode(q[((b * nq + row) * heads + h) * depth + d]) *
                     half_test::decode(k[((b * nk + key) * kv_heads + kh) * depth + d]);
          weights[key] = std::exp(score * 0.5); sum += weights[key];
        }
        for (std::int64_t d = 0; d < depth; ++d) {
          double expected = 0;
          for (std::int64_t key = 0; key < nk; ++key)
            expected += weights[key] / sum * half_test::decode(v[((b * nk + key) * kv_heads + kh) * depth + d]);
          EXPECT_NEAR(half_test::decode(actual[((b * nq + row) * heads + h) * depth + d]), expected,
                       tolerance<T>(2.e-6, 7.e-4, 6.e-3));
        }
      }
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(GgmlCpuProvider, NoncausalAttentionUsesOriginalRegionForGroupedHeadsAndLongerQuery) {
  noncausal_attention<float>();
  noncausal_attention<hyle::simd::float16_t>();
  noncausal_attention<hyle::simd::bfloat16_t>();
}
} // namespace
