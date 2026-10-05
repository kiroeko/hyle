#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <thread>
#include <vector>

#include "half_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
using hyle::mem::Buffer;

TensorDesc description(DataType type, std::span<const std::int64_t> shape,
                       bool odd = false) {
  if (!odd)
    return TensorDesc::dense(type, shape).value();
  std::vector<std::int64_t> strides(shape.size());
  auto stride = static_cast<std::int64_t>(data_type_size(type)) + 1;
  for (std::size_t i = shape.size(); i-- != 0;) {
    strides[i] = stride;
    stride = stride * shape[i] + 1;
  }
  return TensorDesc::strided_bytes(type, shape, strides).value();
}
std::size_t offset(const TensorDesc &desc, std::size_t linear) {
  std::size_t result = 0;
  for (std::size_t i = desc.rank(); i-- != 0;) {
    const auto dim = static_cast<std::size_t>(desc.dimensions()[i]);
    result += linear % dim * static_cast<std::size_t>(desc.byte_strides()[i]);
    linear /= dim;
  }
  return result;
}
Buffer storage(const TensorDesc &desc) {
  return Buffer::create(hyle::mem::system_resource(),
                        desc.required_bytes().value(), 64)
      .value();
}
template <class T>
void write(Buffer &buffer, const TensorDesc &desc, std::span<const T> values) {
  std::vector<std::byte> bytes(buffer.size(), std::byte{0xc7});
  for (std::size_t i = 0; i < values.size(); ++i)
    std::memcpy(bytes.data() + offset(desc, i), &values[i], sizeof(T));
  ASSERT_TRUE(buffer.view().write(bytes).ok());
}
template <class T> std::vector<T> read(const Buffer &buffer) {
  std::vector<T> result(buffer.size() / sizeof(T));
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}
Context
context(hyle::mem::AllocatorRef allocator = hyle::mem::system_allocator()) {
  const std::array bindings{ProviderBinding{aten_cpu::provider(), 100}};
  return Context::create(cpu::device(), {.metadata_allocator = allocator,
                                         .providers = bindings})
      .value();
}

#if defined(HYLE_TEST_TORCH_CPU)
template <class T> void near_value(T actual, double expected) {
  const double value = half_test::decode(actual);
  const double rounded = half_test::decode(half_test::encode<T>(expected));
  if (std::isnan(expected))
    EXPECT_TRUE(std::isnan(value));
  else if (std::isinf(expected))
    EXPECT_EQ(value, expected);
  else
    EXPECT_NEAR(value, rounded, 1e-6 + std::abs(rounded) * 8e-5);
}
template <class T>
std::vector<T> run(Context &ctx, Operation op,
                   std::span<const TensorDesc> descriptors,
                   std::span<Buffer> buffers) {
  std::array<TensorDesc, 1> outputs{};
  EXPECT_TRUE(infer_output_descriptions(op, descriptors, outputs).ok());
  auto prepared =
      prepare(ctx, op, descriptors, outputs, hyle::mem::system_resource())
          .value();
  EXPECT_EQ(prepared.provider().id, aten_cpu::kProviderId);
  auto output = storage(outputs[0]);
  auto invocation = prepared.create_invocation().value();
  auto queue = ctx.create_queue().value();
  std::vector<ConstTensorView> inputs;
  for (std::size_t i = 0; i < descriptors.size(); ++i)
    inputs.push_back(
        ConstTensorView::create(descriptors[i], buffers[i].view()).value());
  const std::array destinations{
      TensorView::create(outputs[0], output.view()).value()};
  auto done = invocation.submit(queue, inputs, destinations);
  EXPECT_TRUE(done);
  EXPECT_TRUE(done->wait().ok());
  return read<T>(output);
}
template <class T> void matrix_batch() {
  auto ctx = context();
  const auto type = half_test::data_type<T>();
  for (bool odd : {false, true}) {
    const std::array<std::int64_t, 8> lhs_shape{1, 1, 1, 1, 1, 1, 2, 3},
        rhs_shape{2, 1, 1, 1, 1, 1, 3, 4};
    const std::array descriptors{description(type, lhs_shape, odd),
                                 description(type, rhs_shape, odd)};
    std::array buffers{storage(descriptors[0]), storage(descriptors[1])};
    std::vector<T> lhs(6), rhs(24);
    for (int i = 0; i < 6; ++i)
      lhs[i] = half_test::encode<T>(i);
    for (int i = 0; i < 24; ++i)
      rhs[i] = half_test::encode<T>(i);
    write<T>(buffers[0], descriptors[0], lhs);
    write<T>(buffers[1], descriptors[1], rhs);
    auto result =
        run<T>(ctx, Operation(MatMul{.accumulation_type = DataType::kFloat32}),
               descriptors, buffers);
    ASSERT_EQ(result.size(), 16U);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) {
          float sum = 0;
          for (int k = 0; k < 3; ++k)
            sum += float(i * 3 + k) * float(b * 12 + k * 4 + j);
          near_value(result[b * 8 + i * 4 + j], sum);
        }
    auto transposed =
        TensorDesc::dense(type, std::array<std::int64_t, 2>{4, 3}).value();
    auto logical = transposed.transposed().value();
    auto ld = description(type, std::array<std::int64_t, 2>{2, 3}, odd);
    const std::array td{ld, logical};
    std::array tb{storage(ld), storage(logical)};
    write<T>(tb[0], ld, lhs);
    std::vector<T> rw(12);
    for (int i = 0; i < 12; ++i)
      rw[i] = half_test::encode<T>(i * .125);
    write<T>(tb[1], transposed, rw);
    auto tr =
        run<T>(ctx, Operation(MatMul{.accumulation_type = DataType::kFloat32}),
               td, tb);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 4; ++j) {
        float expected = 0;
        for (int k = 0; k < 3; ++k)
          expected += float(i * 3 + k) * float(j * 3 + k) * .125F;
        near_value(tr[i * 4 + j], expected);
      }
  }
  const std::array zero{description(type, std::array<std::int64_t, 2>{2, 0}),
                        description(type, std::array<std::int64_t, 2>{0, 3})};
  std::array zbuf{storage(zero[0]), storage(zero[1])};
  auto z =
      run<T>(ctx, Operation(MatMul{.accumulation_type = DataType::kFloat32}),
             zero, zbuf);
  for (auto x : z)
    near_value(x, 0);
}
TEST(AtenCpuTransformer, MatrixRankEightBroadcastTransposeOddAndZeroK) {
  matrix_batch<float>();
  matrix_batch<hyle::simd::float16_t>();
  matrix_batch<hyle::simd::bfloat16_t>();
}
template <class T> void softmax_batch() {
  auto ctx = context();
  const auto type = half_test::data_type<T>();
  for (bool odd : {false, true})
    for (auto kind : {SoftmaxKind::kSoftmax, SoftmaxKind::kLogSoftmax})
      for (auto maskkind :
           {MaskKind::kNone, MaskKind::kBoolean, MaskKind::kAdditive}) {
        const auto d =
            description(type, std::array<std::int64_t, 2>{4, 2}, odd);
        const auto md =
            description(maskkind == MaskKind::kBoolean ? DataType::kBool : type,
                        std::array<std::int64_t, 2>{4, 1}, odd);
        std::vector<T> input;
        for (double x : {0., -2., 1., 2., -1., 3., 2., -4.})
          input.push_back(half_test::encode<T>(x));
        std::vector<T> addition;
        for (double x : {0., -.5, 1., -2.})
          addition.push_back(half_test::encode<T>(x));
        const std::array<std::uint8_t, 4> bits{2, 0, 255, 1};
        std::vector<TensorDesc> descriptors{d};
        std::vector<Buffer> buffers;
        buffers.push_back(storage(d));
        write<T>(buffers[0], d, input);
        if (maskkind != MaskKind::kNone) {
          descriptors.push_back(md);
          buffers.push_back(storage(md));
          if (maskkind == MaskKind::kBoolean)
            write<std::uint8_t>(buffers[1], md, bits);
          else
            write<T>(buffers[1], md, addition);
        }
        auto result = run<T>(
            ctx,
            Operation(Softmax{
                .axis = 0, .scale = .5F, .kind = kind, .mask = maskkind}),
            descriptors, buffers);
        for (int col = 0; col < 2; ++col) {
          double sum = 0, maximum = -INFINITY;
          std::array<double, 4> scores{};
          for (int row = 0; row < 4; ++row) {
            scores[row] = half_test::decode(input[row * 2 + col]) * .5;
            if (maskkind == MaskKind::kBoolean && bits[row] == 0)
              scores[row] = -INFINITY;
            if (maskkind == MaskKind::kAdditive)
              scores[row] += half_test::decode(addition[row]);
            maximum = std::max(maximum, scores[row]);
          }
          for (double x : scores)
            sum += std::exp(x - maximum);
          for (int row = 0; row < 4; ++row) {
            double expected = kind == SoftmaxKind::kLogSoftmax
                                  ? scores[row] - maximum - std::log(sum)
                                  : std::exp(scores[row] - maximum) / sum;
            near_value(result[row * 2 + col], expected);
          }
        }
      }
  const auto d = description(type, std::array<std::int64_t, 2>{2, 3});
  std::vector<T> special;
  for (double x : {-INFINITY, -INFINITY, -INFINITY, 0., INFINITY, 1.})
    special.push_back(half_test::encode<T>(x));
  std::array bd{d};
  std::array bb{storage(d)};
  write<T>(bb[0], d, special);
  for (auto kind : {SoftmaxKind::kSoftmax, SoftmaxKind::kLogSoftmax}) {
    auto result = run<T>(ctx, Operation(Softmax{.kind = kind}), bd, bb);
    for (auto x : result)
      EXPECT_TRUE(std::isnan(half_test::decode(x)));
  }
}
TEST(AtenCpuTransformer, BooleanAdditiveLogNonLastAxisAndInvalidRows) {
  softmax_batch<float>();
  softmax_batch<hyle::simd::float16_t>();
  softmax_batch<hyle::simd::bfloat16_t>();
}
TEST(AtenCpuTransformer, RopeLongIndicesLayoutsPrefixAndTailBits) {
  auto ctx = context();
  for (bool odd : {false, true})
    for (auto index_type : {DataType::kInt32, DataType::kInt64})
      for (auto layout : {RopeLayout::kInterleaved, RopeLayout::kHalfSplit}) {
        const auto d = description(
            DataType::kFloat32, std::array<std::int64_t, 4>{2, 2, 2, 5}, odd);
        const auto p =
            description(index_type, std::array<std::int64_t, 2>{2, 2}, odd);
        const auto f = description(DataType::kFloat32,
                                   std::array<std::int64_t, 2>{2, 2}, odd);
        const std::array ds{d, p, f};
        std::array bs{storage(d), storage(p), storage(f)};
        std::vector<float> x(40);
        for (int i = 0; i < 40; ++i)
          x[i] = float(i) / 8.F;
        for (int row = 0; row < 8; ++row)
          x[row * 5 + 4] = std::bit_cast<float>(0xffa12345U);
        std::array<std::int64_t, 4> ps{1, -2, 2147483649LL, -2147483651LL};
        if (index_type == DataType::kInt32) {
          const std::array<std::int32_t, 4> short_ps{1, -2, 3, -4};
          for (int i = 0; i < 4; ++i)
            ps[i] = short_ps[i];
          write<std::int32_t>(bs[1], p, short_ps);
        } else
          write<std::int64_t>(bs[1], p, ps);
        const std::array<float, 4> freq{.1F, .003F, .2F, .007F};
        write<float>(bs[0], d, x);
        write<float>(bs[2], f, freq);
        auto result = run<float>(ctx,
                                 Operation(RoPE{.rotary_dimensions = 4,
                                                .sequence_axis = 1,
                                                .layout = layout}),
                                 ds, bs);
        for (int batch = 0; batch < 2; ++batch)
          for (int t = 0; t < 2; ++t)
            for (int h = 0; h < 2; ++h) {
              int base = ((batch * 2 + t) * 2 + h) * 5;
              for (int j = 0; j < 2; ++j) {
                int left = layout == RopeLayout::kHalfSplit ? j : j * 2,
                    right =
                        layout == RopeLayout::kHalfSplit ? j + 2 : j * 2 + 1;
                double angle = double(ps[batch * 2 + t]) * freq[t * 2 + j];
                near_value(result[base + left],
                           float(double(x[base + left]) * std::cos(angle) -
                                 double(x[base + right]) * std::sin(angle)));
                near_value(result[base + right],
                           float(double(x[base + left]) * std::sin(angle) +
                                 double(x[base + right]) * std::cos(angle)));
              }
              EXPECT_EQ(std::bit_cast<std::uint32_t>(result[base + 4]),
                        0xffa12345U);
            }
      }
}
template <class T> void attention_batch() {
  auto ctx = context();
  auto type = half_test::data_type<T>();
  for (bool odd : {false, true})
    for (auto layout : {AttentionLayout::kBatchHeadSequenceDepth,
                        AttentionLayout::kBatchSequenceHeadDepth})
      for (bool causal : {false, true})
        for (auto maskkind : {MaskKind::kBoolean, MaskKind::kAdditive}) {
          const bool bshd = layout == AttentionLayout::kBatchSequenceHeadDepth;
          const auto q =
              description(type,
                          bshd ? std::array<std::int64_t, 4>{1, 3, 2, 2}
                               : std::array<std::int64_t, 4>{1, 2, 3, 2},
                          odd);
          const auto k =
              description(type,
                          bshd ? std::array<std::int64_t, 4>{1, 2, 1, 2}
                               : std::array<std::int64_t, 4>{1, 1, 2, 2},
                          odd);
          const auto v =
              description(type,
                          bshd ? std::array<std::int64_t, 4>{1, 2, 1, 3}
                               : std::array<std::int64_t, 4>{1, 1, 2, 3},
                          odd);
          const auto mask = description(
              maskkind == MaskKind::kBoolean ? DataType::kBool : type,
              std::array<std::int64_t, 2>{3, 2}, odd);
          const std::array ds{q, k, v, mask};
          std::array bs{storage(q), storage(k), storage(v), storage(mask)};
          const std::vector<T> qs(12, half_test::encode<T>(1)),
              ks(4, half_test::encode<T>(1));
          std::vector<T> vs;
          for (double x : {1., 2., 3., 4., 5., 6.})
            vs.push_back(half_test::encode<T>(x));
          write<T>(bs[0], q, qs);
          write<T>(bs[1], k, ks);
          write<T>(bs[2], v, vs);
          const std::array<std::uint8_t, 6> bits{0, 0, 2, 0, 255, 1};
          std::vector<T> add;
          for (auto x : bits)
            add.push_back(half_test::encode<T>(x == 0 ? -INFINITY : 0));
          if (maskkind == MaskKind::kBoolean)
            write<std::uint8_t>(bs[3], mask, bits);
          else
            write<T>(bs[3], mask, add);
          auto result =
              run<T>(ctx,
                     Operation(ScaledDotProductAttention{.layout = layout,
                                                         .mask = maskkind,
                                                         .scale = .5F,
                                                         .causal = causal}),
                     ds, bs);
          for (int head = 0; head < 2; ++head)
            for (int t = 0; t < 3; ++t)
              for (int depth = 0; depth < 3; ++depth) {
                std::size_t index = bshd ? (t * 2 + head) * 3 + depth
                                         : (head * 3 + t) * 3 + depth;
                double expected = t == 0   ? NAN
                                  : t == 1 ? depth + 1
                                           : depth + 2.5;
                near_value(result[index], expected);
              }
        }
}
TEST(AtenCpuTransformer, MaskedGqaBothLayoutsCausalDifferentValueDepth) {
  attention_batch<float>();
  attention_batch<hyle::simd::float16_t>();
  attention_batch<hyle::simd::bfloat16_t>();
}
template <class T> void norm_batch() {
  auto ctx = context();
  auto type = half_test::data_type<T>();
  for (bool odd : {false, true})
    for (bool layer : {false, true}) {
      const auto d = description(type, std::array<std::int64_t, 2>{2, 4}, odd);
      const auto param =
          description(DataType::kFloat32, std::array<std::int64_t, 1>{4}, odd);
      std::vector<T> x;
      for (double v : {1., 2., 3., 4., -1., -2., -3., -4.})
        x.push_back(half_test::encode<T>(v));
      const std::array<float, 4> gamma{1, .5F, 2, 1.5F},
          beta{.125F, -.25F, .5F, 1};
      std::vector<TensorDesc> ds{d, param};
      std::vector<Buffer> bs;
      bs.push_back(storage(d));
      bs.push_back(storage(param));
      write<T>(bs[0], d, x);
      write<float>(bs[1], param, gamma);
      if (layer) {
        ds.push_back(param);
        bs.push_back(storage(param));
        write<float>(bs[2], param, beta);
      }
      auto result = run<T>(
          ctx, layer ? Operation(LayerNorm{}) : Operation(RmsNorm{}), ds, bs);
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 4; ++col) {
          float value = float(half_test::decode(x[row * 4 + col]));
          float normalized =
              layer
                  ? float((double(value) - (row == 0 ? 2.5 : -2.5)) /
                          std::sqrt(1.25 + double(float(1e-5))))
                  : float(double(value) / std::sqrt(7.5 + double(float(1e-5))));
          float expected = normalized * gamma[col] + (layer ? beta[col] : 0);
          near_value(result[row * 4 + col], expected);
        }
    }
}
TEST(AtenCpuTransformer, NormOddStorageAndFloatingAffineParameters) {
  norm_batch<float>();
  norm_batch<hyle::simd::float16_t>();
  norm_batch<hyle::simd::bfloat16_t>();
}

TEST(AtenCpuTransformer, DenseAliasPrivateSlotsAndCloseRetainCorrectSoftmax) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto ctx = context(stats.get());
    auto queue = ctx.create_queue().value();
    const auto d =
        description(DataType::kFloat32, std::array<std::int64_t, 2>{1, 4});
    const auto mask =
        description(DataType::kBool, std::array<std::int64_t, 2>{1, 4});
    const std::array ds{d, mask};
    const std::array out{d};
    auto prepared = prepare(ctx, Operation(Softmax{.mask = MaskKind::kBoolean}),
                            ds, out, hyle::mem::system_resource())
                        .value();
    EXPECT_TRUE(prepared.alias_support(0, 0)->supported);
    EXPECT_FALSE(prepared.alias_support(0, 1)->supported);
    std::array inv{prepared.create_invocation().value(),
                   prepared.create_invocation().value()};
    std::array data{storage(d), storage(d)};
    std::array masks{storage(mask), storage(mask)};
    const std::array<std::uint8_t, 4> bits{2, 0, 255, 1};
    for (auto &m : masks)
      write<std::uint8_t>(m, mask, bits);
    std::array<std::array<float, 4>, 2> prior{};
    for (int replay = 0; replay < 4; ++replay) {
      const auto slot = std::size_t(replay % 2);
      std::array<float, 4> x{float(replay), -1, float(replay) * .5F, 2};
      write<float>(data[slot], d, x);
      const std::array views{
          ConstTensorView::create(d, data[slot].view()).value(),
          ConstTensorView::create(mask, masks[slot].view()).value()};
      const std::array output{TensorView::create(d, data[slot].view()).value()};
      auto done = inv[slot].submit(queue, views, output);
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      auto values = read<float>(data[slot]);
      double sum = std::exp(double(x[0])) + std::exp(double(x[2])) +
                   std::exp(double(x[3]));
      for (int i = 0; i < 4; ++i) {
        double expected = bits[i] ? std::exp(double(x[i])) / sum : 0;
        near_value(values[i], expected);
        prior[slot][i] = values[i];
      }
      if (replay != 0) {
        auto other = read<float>(data[1 - slot]);
        for (int i = 0; i < 4; ++i)
          EXPECT_EQ(other[i], prior[1 - slot][i]);
      }
    }
    ASSERT_TRUE(ctx.close().ok());
    const std::array views{
        ConstTensorView::create(d, data[0].view()).value(),
        ConstTensorView::create(mask, masks[0].view()).value()};
    const std::array output{TensorView::create(d, data[0].view()).value()};
    EXPECT_FALSE(inv[0].submit(queue, views, output));
  }
  EXPECT_TRUE(stats.balanced());
}
#endif
} // namespace
