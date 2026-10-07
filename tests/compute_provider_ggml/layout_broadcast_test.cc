#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
template <class T> constexpr double tolerance() {
  if constexpr (std::is_same_v<T, float>)
    return 3.e-6;
  else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
    return 1.e-3;
  else
    return 8.e-3;
}

TEST(GgmlCpuProvider, LayoutMatMulReplaysRowMajorAndBothBroadcastDirections) {
  auto context =
      Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 3}))
          .value();
  auto queue = context.create_queue().value();
  for (int rank : {2, 3, 4})
    for (const auto &batch : {std::array<std::int64_t, 4>{1, 1, 1, 1},
                              {1, 1, 1, 3},
                              {1, 3, 1, 1},
                              {1, 1, 2, 3},
                              {2, 1, 1, 3},
                              {2, 3, 1, 1},
                              {2, 3, 2, 3}}) {
      if (rank < 4 && (batch[0] != 1 || batch[2] != 1))
        continue;
      if (rank < 3 && (batch[1] != 1 || batch[3] != 1))
        continue;
      for (const bool transposed : {false, true}) {
        constexpr std::int64_t m = 3, k = 7, n = 5;
        const std::array left_shape{batch[0], batch[1], m, k};
        const std::array right_shape{batch[2], batch[3], k, n};
        const std::array transposed_shape{batch[2], batch[3], n, k};
        const auto lhs = TensorDesc::dense(DataType::kFloat32,
                                           std::span(left_shape).last(rank))
                             .value();
        const auto rhs =
            transposed
                ? TensorDesc::dense(DataType::kFloat32,
                                    std::span(transposed_shape).last(rank))
                      .value()
                      .transposed()
                      .value()
                : TensorDesc::dense(DataType::kFloat32,
                                    std::span(right_shape).last(rank))
                      .value();
        const std::array descriptions{lhs, rhs};
        const auto output =
            infer_single_output(Operation(MatMul{}), descriptions).value();
        auto prepared =
            prepare(context, Operation(MatMul{}), descriptions,
                    std::span(&output, 1), hyle::mem::system_resource());
        ASSERT_TRUE(prepared) << prepared.error().message();
        auto invocation = prepared->create_invocation().value();
        std::vector<float> a(lhs.element_count().value()),
            b(rhs.element_count().value());
        auto ab = half_test::buffer<float>(a), bb = half_test::buffer<float>(b);
        auto out = half_test::buffer<float>(
            std::vector<float>(output.element_count().value()));
        const std::array sources{
            ConstTensorView::create(lhs, ab.view()).value(),
            ConstTensorView::create(rhs, bb.view()).value()};
        const auto destination = TensorView::create(output, out.view()).value();
        for (int seed : {1, 3}) {
          for (std::size_t i = 0; i < a.size(); ++i)
            a[i] = float((int(i) + seed) % 11 - 5) / 8.F;
          for (std::size_t i = 0; i < b.size(); ++i)
            b[i] = float((int(i) * 3 + seed) % 13 - 6) / 8.F;
          ASSERT_TRUE(ab.view().write(std::as_bytes(std::span(a))).ok());
          ASSERT_TRUE(bb.view().write(std::as_bytes(std::span(b))).ok());
          auto done =
              invocation.submit(queue, sources, std::span(&destination, 1));
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          const auto actual = half_test::read<float>(out);
          const auto outer = (std::max)(batch[0], batch[2]),
                     inner = (std::max)(batch[1], batch[3]);
          for (std::int64_t x = 0; x < outer; ++x)
            for (std::int64_t y = 0; y < inner; ++y)
              for (std::int64_t row = 0; row < m; ++row)
                for (std::int64_t col = 0; col < n; ++col) {
                  const auto ai = ((batch[0] == 1 ? 0 : x) * batch[1] +
                                   (batch[1] == 1 ? 0 : y)) *
                                  m * k;
                  const auto bi = ((batch[2] == 1 ? 0 : x) * batch[3] +
                                   (batch[3] == 1 ? 0 : y)) *
                                  k * n;
                  double expected = 0;
                  for (std::int64_t d = 0; d < k; ++d)
                    expected +=
                        double(a[ai + row * k + d]) *
                        b[bi + (transposed ? col * k + d : d * n + col)];
                  EXPECT_NEAR(actual[((x * inner + y) * m + row) * n + col],
                              expected, 2.e-6);
                }
        }
      }
    }
  ASSERT_TRUE(context.close().ok());
}

TEST(GgmlCpuProvider, ZeroContractedMatMulProducesIndependentZeros) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const auto lhs = TensorDesc::dense(DataType::kFloat32,
                                     std::array<std::int64_t, 3>{1, 3, 0})
                       .value();
  const auto rhs = TensorDesc::dense(DataType::kFloat32,
                                     std::array<std::int64_t, 3>{2, 0, 5})
                       .value();
  auto a = half_test::buffer<float>({}), b = half_test::buffer<float>({});
  const std::array sources{ConstTensorView::create(lhs, a.view()).value(),
                           ConstTensorView::create(rhs, b.view()).value()};
  auto output = half_test::buffer<float>(std::vector<float>(30, 42.F));
  const auto desc =
      infer_single_output(Operation(MatMul{}), std::array{lhs, rhs}).value();
  auto prepared = prepare(context, MatMul{}, lhs, rhs).value();
  auto invocation = prepared.create_invocation().value();
  const auto destination = TensorView::create(desc, output.view()).value();
  for (int i = 0; i < 2; ++i) {
    auto done = invocation.submit(queue, sources, std::span(&destination, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    for (auto value : half_test::read<float>(output))
      EXPECT_EQ(value, 0.F);
  }
}

template <class T> void axis_softmax() {
  const auto type = half_test::data_type<T>();
  const auto desc =
      TensorDesc::dense(type, std::array<std::int64_t, 3>{2, 3, 7}).value();
  const auto md =
      TensorDesc::dense(type, std::array<std::int64_t, 3>{1, 3, 1}).value();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const std::array bindings{
        ProviderBinding{ggml_cpu::provider({.native_threads = 3}), 100}};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                        .providers = bindings})
            .value();
    auto queue = context.create_queue().value();
    for (int selected : {0, 1, -2}) {
      const Softmax op{
          .axis = selected, .scale = .5F, .mask = MaskKind::kAdditive};
      auto prepared = prepare(context, op, desc, md).value();
      auto invocation = prepared.create_invocation().value();
      std::vector<T> values(42), masks(3);
      auto a = half_test::buffer<T>(values), mb = half_test::buffer<T>(masks),
           out = half_test::buffer<T>(values);
      const std::array sources{ConstTensorView::create(desc, a.view()).value(),
                               ConstTensorView::create(md, mb.view()).value()};
      const auto destination = TensorView::create(desc, out.view()).value();
      const auto allocs = stats.snapshot().total_allocs;
      for (int seed : {1, -1, -2, -3, 3}) {
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] =
              half_test::encode<T>(double((int(i) + seed) % 11 - 5) / 8.);
        for (std::size_t i = 0; i < masks.size(); ++i)
          masks[i] = half_test::encode<T>(double(i) / 4.);
        if (seed == -1)
          values[0] = half_test::encode<T>(NAN);
        if (seed == -2)
          values[0] = half_test::encode<T>(INFINITY);
        if (seed == -3) {
          if (selected == 0)
            for (int b = 0; b < 2; ++b)
              values[b * 21] = half_test::encode<T>(-INFINITY);
          else
            for (int h = 0; h < 3; ++h)
              values[h * 7] = half_test::encode<T>(-INFINITY);
        }
        ASSERT_TRUE(a.view().write(std::as_bytes(std::span(values))).ok());
        ASSERT_TRUE(mb.view().write(std::as_bytes(std::span(masks))).ok());
        auto done =
            invocation.submit(queue, sources, std::span(&destination, 1));
        ASSERT_TRUE(done);
        ASSERT_TRUE(done->wait().ok());
        const auto actual = half_test::read<T>(out);
        for (int b = 0; b < 2; ++b)
          for (int h = 0; h < 3; ++h)
            for (int d = 0; d < 7; ++d) {
              const bool invalid =
                  seed < 0 && d == 0 && (selected == 0 ? h == 0 : b == 0);
              const auto index = (b * 3 + h) * 7 + d;
              if (invalid) {
                EXPECT_TRUE(std::isnan(half_test::decode(actual[index])));
                continue;
              }
              double sum = 0;
              for (int i = 0; i < (selected == 0 ? 2 : 3); ++i) {
                const auto ri =
                    selected == 0 ? (i * 3 + h) * 7 + d : (b * 3 + i) * 7 + d;
                sum +=
                    std::exp(half_test::decode(values[ri]) * .5 +
                             half_test::decode(masks[selected == 0 ? h : i]));
              }
              const auto expected =
                  std::exp(half_test::decode(values[index]) * .5 +
                           half_test::decode(masks[h])) /
                  sum;
              EXPECT_NEAR(half_test::decode(actual[index]), expected,
                          tolerance<T>());
            }
      }
      EXPECT_EQ(stats.snapshot().total_allocs, allocs);
    }
    ASSERT_TRUE(context.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
TEST(GgmlCpuProvider, SoftmaxRestoresArbitraryAxesMasksAndUndefinedRows) {
  axis_softmax<float>();
  axis_softmax<hyle::simd::float16_t>();
  axis_softmax<hyle::simd::bfloat16_t>();
}

template <class T>
void attention_layouts(std::int64_t nq = 3, std::int64_t dk = 7,
                        std::int64_t dv = 5, std::int64_t keys = 0,
                        std::int64_t nb = 2) {
  constexpr std::int64_t hq = 4, hk = 2;
  const auto type = half_test::data_type<T>();
  auto context =
      Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 3}))
          .value();
  auto queue = context.create_queue().value();
  for (bool bshd : {false, true})
    for (bool causal : {false, true}) {
      const std::int64_t nk = keys != 0 ? keys : causal ? 4 : 2;
      const auto shape = [=](std::int64_t length, std::int64_t heads,
                             std::int64_t depth) {
        return bshd ? std::array{nb, length, heads, depth}
                    : std::array{nb, heads, length, depth};
      };
      const auto qd = TensorDesc::dense(type, shape(nq, hq, dk)).value();
      const auto kd = TensorDesc::dense(type, shape(nk, hk, dk)).value();
      const auto vd = TensorDesc::dense(type, shape(nk, hk, dv)).value();
      const auto od = TensorDesc::dense(type, shape(nq, hq, dv)).value();
      const auto index = [=](std::int64_t b, std::int64_t t, std::int64_t h,
                             std::int64_t d, std::int64_t length,
                             std::int64_t heads, std::int64_t depth) {
        return (bshd ? (b * length + t) * heads + h
                     : (b * heads + h) * length + t) *
                   depth +
               d;
      };
      const ScaledDotProductAttention op{
          .layout = bshd ? AttentionLayout::kBatchSequenceHeadDepth
                         : AttentionLayout::kBatchHeadSequenceDepth,
          .scale = .5F,
          .causal = causal};
      auto prepared = prepare(context, op, qd, kd, vd);
      ASSERT_TRUE(prepared) << prepared.error().message();
      auto invocation = prepared->create_invocation();
      ASSERT_TRUE(invocation) << invocation.error().message();
      std::vector<T> q(qd.element_count().value()),
          k(kd.element_count().value()), v(vd.element_count().value());
      auto qb = half_test::buffer<T>(q), kb = half_test::buffer<T>(k),
           vb = half_test::buffer<T>(v),
           out =
               half_test::buffer<T>(std::vector<T>(od.element_count().value()));
      const std::array sources{ConstTensorView::create(qd, qb.view()).value(),
                               ConstTensorView::create(kd, kb.view()).value(),
                               ConstTensorView::create(vd, vb.view()).value()};
      const auto destination = TensorView::create(od, out.view()).value();
      for (int seed : {1, 3}) {
        for (std::size_t i = 0; i < q.size(); ++i)
          q[i] = half_test::encode<T>(double((int(i) + seed) % 9 - 4) / 8.);
        for (std::size_t i = 0; i < k.size(); ++i)
          k[i] =
              half_test::encode<T>(double((int(i) * 3 + seed) % 11 - 5) / 8.);
        for (std::size_t i = 0; i < v.size(); ++i)
          v[i] =
              half_test::encode<T>(double((int(i) * 7 + seed) % 13 - 6) / 4.);
        ASSERT_TRUE(qb.view().write(std::as_bytes(std::span(q))).ok());
        ASSERT_TRUE(kb.view().write(std::as_bytes(std::span(k))).ok());
        ASSERT_TRUE(vb.view().write(std::as_bytes(std::span(v))).ok());
        auto done =
            invocation->submit(queue, sources, std::span(&destination, 1));
        ASSERT_TRUE(done);
        ASSERT_TRUE(done->wait().ok());
        const auto actual = half_test::read<T>(out);
        for (std::int64_t b = 0; b < nb; ++b)
          for (std::int64_t row = 0; row < nq; ++row)
            for (std::int64_t h = 0; h < hq; ++h) {
              std::vector<double> weights(nk);
              double sum = 0;
              for (std::int64_t key = 0; key < nk; ++key) {
                if (causal && key > nk - nq + row)
                  continue;
                double dot = 0;
                for (std::int64_t d = 0; d < dk; ++d)
                  dot +=
                      half_test::decode(q[index(b, row, h, d, nq, hq, dk)]) *
                      half_test::decode(k[index(b, key, h / 2, d, nk, hk, dk)]);
                weights[key] = std::exp(dot * .5);
                sum += weights[key];
              }
              for (std::int64_t d = 0; d < dv; ++d) {
                double expected = 0;
                for (std::int64_t key = 0; key < nk; ++key)
                  expected +=
                      weights[key] / sum *
                      half_test::decode(v[index(b, key, h / 2, d, nk, hk, dv)]);
                EXPECT_NEAR(
                    half_test::decode(actual[index(b, row, h, d, nq, hq, dv)]),
                    expected, tolerance<T>());
              }
            }
      }
    }
  ASSERT_TRUE(context.close().ok());
}
TEST(GgmlCpuProvider, AttentionPreservesBothLayoutsAndIndependentValueDepth) {
  attention_layouts<float>();
  attention_layouts<hyle::simd::float16_t>();
  attention_layouts<hyle::simd::bfloat16_t>();
}
TEST(GgmlCpuProvider, F32AttentionActualWidthsReplayAndLongKv) {
  for (const auto rows : {1, 32}) {
    attention_layouts<float>(rows, 64, 64, 128);
    attention_layouts<float>(rows, 256, 256, 128);
  }
  attention_layouts<float>(1, 256, 256, 1024, 1);
}

TEST(GgmlCpuProvider, PlainStorageCopyConcatPreservesAllScalarBitsAndStrides) {
  auto context =
      Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 3}))
          .value();
  auto queue = context.create_queue().value();
  for (int raw = int(DataType::kBool); raw <= int(DataType::kUInt64); ++raw) {
    const auto type = static_cast<DataType>(raw);
    const auto size = data_type_size(type);
    for (int axis : {0, 1, 2}) {
      std::array<std::int64_t, 3> shape{2, 3, 7};
      shape[axis] = 1;
      auto outshape = shape;
      outshape[axis] = 2;
      auto emptyshape = shape;
      emptyshape[axis] = 0;
      const std::array<std::int64_t, 3> strides{
          shape[1] * shape[2] * std::int64_t(size) * 2,
          shape[2] * std::int64_t(size) * 2, std::int64_t(size) * 2};
      const auto dense = TensorDesc::dense(type, shape).value(),
                 output = TensorDesc::dense(type, outshape).value();
      const auto strided =
                     TensorDesc::strided_bytes(type, shape, strides).value(),
                 empty = TensorDesc::dense(type, emptyshape).value();
      auto a = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                         strided.required_bytes().value(), 64)
                   .value();
      auto b = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                         dense.required_bytes().value(), 64)
                   .value();
      auto z = hyle::mem::Buffer::create(hyle::mem::system_resource(), 0, 64)
                   .value();
      auto out = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                           output.required_bytes().value(), 64)
                     .value();
      auto cp = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                          dense.required_bytes().value(), 64)
                    .value();
      auto copy = prepare(context, Copy{}, strided).value();
      auto ci = copy.create_invocation().value();
      const std::array descriptions{strided, empty, dense};
      auto concat =
          prepare(context, Operation(Concat{.axis = axis}), descriptions,
                  std::span(&output, 1), hyle::mem::system_resource());
      ASSERT_TRUE(concat) << concat.error().message();
      auto invocation = concat->create_invocation();
      ASSERT_TRUE(invocation) << invocation.error().message();
      const std::array sources{
          ConstTensorView::create(strided, a.view()).value(),
          ConstTensorView::create(empty, z.view()).value(),
          ConstTensorView::create(dense, b.view()).value()};
      const auto destination = TensorView::create(output, out.view()).value(),
                 copied = TensorView::create(dense, cp.view()).value();
      for (int seed : {1, 7}) {
        std::vector<std::byte> av(a.size()), bv(b.size());
        for (std::size_t i = 0; i < bv.size(); ++i)
          bv[i] = std::byte((i * 17 + seed) & 255);
        const std::array<std::uint64_t, 4> patterns{
            0xffffffffffffffffULL, 0x8000000080008000ULL, 0x7ff812347fc17e25ULL,
            0x0020000000000001ULL};
        for (std::size_t i = 0;
             i < (std::min)(patterns.size(), dense.element_count().value());
             ++i) {
          const auto bits = patterns[i];
          std::memcpy(bv.data() + i * size, &bits, size);
        }
        // Full 64-bit values/NaN payloads are deliberately opaque, never
        // float-converted.
        for (std::size_t i = 0; i < dense.element_count().value(); ++i)
          for (std::size_t j = 0; j < size; ++j)
            av[i * size * 2 + j] = bv[i * size + j];
        ASSERT_TRUE(a.view().write(av).ok());
        ASSERT_TRUE(b.view().write(bv).ok());
        auto done = ci.submit(queue, sources[0], copied);
        ASSERT_TRUE(done);
        ASSERT_TRUE(done->wait().ok());
        std::vector<std::byte> actual(cp.size());
        ASSERT_TRUE(cp.view().read(actual).ok());
        EXPECT_EQ(actual, bv);
        done = invocation->submit(queue, sources, std::span(&destination, 1));
        ASSERT_TRUE(done);
        ASSERT_TRUE(done->wait().ok());
        actual.resize(out.size());
        ASSERT_TRUE(out.view().read(actual).ok());
        std::size_t inner = 1;
        for (int i = axis + 1; i < 3; ++i)
          inner *= std::size_t(shape[i]);
        for (std::size_t i = 0; i < output.element_count().value(); ++i) {
          const auto source = (i / (2 * inner)) * inner + i % inner;
          EXPECT_EQ(std::memcmp(actual.data() + i * size,
                                bv.data() + source * size, size),
                    0);
        }
      }
    }
  }
}
TEST(GgmlCpuProvider, StorageCarrierDoesNotAdmitUnprovidedWideMath) {
  auto context = Context::create(cpu::device(), ggml_cpu::provider()).value();
  const auto ids =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{2})
          .value();
  for (auto type : {DataType::kInt8, DataType::kInt16, DataType::kUInt32,
                    DataType::kInt64, DataType::kUInt64, DataType::kFloat64}) {
    const auto desc =
        TensorDesc::dense(type, std::array<std::int64_t, 2>{3, 7}).value();
    auto gathered = prepare(context, Gather{.axis = 0}, desc, ids);
    ASSERT_FALSE(gathered);
    EXPECT_TRUE(gathered.error().IsUnimplemented());
    auto cast = prepare(context, Cast{.output_type = DataType::kFloat32}, desc);
    ASSERT_FALSE(cast);
    EXPECT_TRUE(cast.error().IsUnimplemented());
  }
}
} // namespace
