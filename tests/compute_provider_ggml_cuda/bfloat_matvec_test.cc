#include "tensor_test_support.h"
#include "hyle/compute/model_ops.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace ggml_cuda_tensor_test;
using BF16 = hyle::simd::bfloat16_t;
constexpr std::int64_t kColumns = 17;

template <class Value> struct MatvecInputs {
  std::vector<Value> left, right, expected;
};

template <class Value>
MatvecInputs<Value> inputs(std::int64_t inner, bool transposed, int seed) {
  MatvecInputs<Value> result;
  result.left.resize(static_cast<std::size_t>(inner));
  result.right.resize(static_cast<std::size_t>(inner * kColumns));
  const auto offset = [=](std::int64_t k, std::int64_t n) {
    return static_cast<std::size_t>(transposed ? n * inner + k
                                             : k * kColumns + n);
  };
  for (std::size_t i = 0; i < result.left.size(); ++i) {
    const double value = std::is_same_v<Value, BF16>
        ? (static_cast<int>(i % 19) - 9) / 16.0 + seed / 128.0
        : 0.15 + static_cast<double>(i % 19) * 0.037 + seed * 0.013;
    result.left[i] = half_test::encode<Value>(value);
  }
  for (std::int64_t k = 0; k < inner; ++k)
    for (std::int64_t n = 0; n < kColumns; ++n) {
      const auto i = k * kColumns + n;
      const double value = std::is_same_v<Value, BF16>
          ? (static_cast<int>(i % 23) - 11) / 32.0 - seed / 256.0
          : 0.11 + static_cast<double>(i % 23) * 0.029 - seed * 0.007;
      result.right[offset(k, n)] = half_test::encode<Value>(value);
    }
  if constexpr (std::is_same_v<Value, BF16>) {
    result.left[0] = result.left[1] = half_test::encode<Value>(1);
    // Exact halfway dots exercise both directions of BF16 ties-to-even.
    for (std::int64_t k = 0; k < inner; ++k)
      for (std::int64_t n = 0; n < 2; ++n)
        result.right[offset(k, n)] = half_test::encode<Value>(
            k == 0 ? 1.0 + n / 128.0 : k == 1 ? 1.0 / 256.0 : 0.0);
  }
  for (std::int64_t n = 0; n < kColumns; ++n) {
    double sum = 0;
    for (std::int64_t k = 0; k < inner; ++k)
      sum += half_test::decode(result.left[static_cast<std::size_t>(k)]) *
             half_test::decode(result.right[offset(k, n)]);
    // BF16 operands are binary rationals with common product denominator
    // 2^15 and bounded numerators: all products and partial sums fit F32
    // exactly for K<=128. The FP64 dot therefore isolates public BF16 RNE.
    if constexpr (std::is_same_v<Value, BF16>)
      EXPECT_EQ(static_cast<double>(static_cast<float>(sum)), sum);
    result.expected.push_back(half_test::encode<Value>(sum));
  }
  if constexpr (std::is_same_v<Value, BF16>) {
    EXPECT_EQ(result.expected[0].bits, 0x3f80U);
    EXPECT_EQ(result.expected[1].bits, 0x3f82U);
  }
  return result;
}

template <class Value>
void check(const Buffer &output, const std::vector<Value> &expected,
           bool unchanged = false) {
  const auto actual = half_test::read<Value>(output);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_same_v<Value, BF16>) {
      EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
    } else if (unchanged) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                std::bit_cast<std::uint32_t>(expected[i])) << i;
    } else {
      EXPECT_NEAR(actual[i], expected[i],
                  2e-6 + 6e-6 * std::abs(expected[i])) << i;
    }
  }
}

template <class Value> void matvec(std::int64_t inner, bool transposed) {
  SCOPED_TRACE(::testing::Message()
               << "type=" << int(half_test::data_type<Value>())
               << " K=" << inner << " transposed=" << transposed);
  auto context = Context::create(cuda::device(0).value(), ggml_cuda::provider())
                     .value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice)
                    .value();
  const auto type = half_test::data_type<Value>();
  const auto left_desc = dense(type, {1, inner});
  const auto right_desc = transposed
      ? dense(type, {kColumns, inner}).transposed().value()
      : dense(type, {inner, kColumns});
  const auto output_desc = dense(type, {1, kColumns});
  const std::array descriptions{left_desc, right_desc};
  auto prepared = prepare(context,
      Operation(MatMul{.accumulation_type = DataType::kFloat32}), descriptions,
      std::span(&output_desc, 1), memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  // Owners outlive both invocations and every old binding; native_handle is
  // the CUDA allocation pointer, so retained buffers also prevent address reuse.
  std::vector<Buffer> retained;
  std::array workspaces{
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref()),
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref())};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array runs{prepared->create_invocation(workspaces[0].view()).value(),
                  prepared->create_invocation(workspaces[1].view()).value()};
  std::array<std::vector<Value>, 2> old;
  for (int round = 0; round < 2; ++round) {
    SCOPED_TRACE(round);
    std::array<Buffer, 2> left, right, output;
    std::array<MatvecInputs<Value>, 2> host;
    std::array<Completion, 2> done;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      host[slot] = inputs<Value>(inner, transposed,
                                round * 3 + static_cast<int>(slot) + 1);
      left[slot] = upload<Value>(memory.ref(), host[slot].left);
      right[slot] = upload<Value>(memory.ref(), host[slot].right);
      output[slot] = output_buffer(memory.ref(), output_desc);
      if (round == 1) {
        EXPECT_NE(left[slot].native_handle(), retained[slot * 3].native_handle());
        EXPECT_NE(right[slot].native_handle(), retained[slot * 3 + 1].native_handle());
        EXPECT_NE(output[slot].native_handle(), retained[slot * 3 + 2].native_handle());
      }
      const std::array bindings{
          ConstTensorView::create(left_desc, left[slot].view()).value(),
          ConstTensorView::create(right_desc, right[slot].view()).value()};
      const auto target = TensorView::create(output_desc, output[slot].view()).value();
      auto submitted = runs[slot].submit(queues[slot], bindings, target);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok());
    ASSERT_TRUE(done[0].wait().ok());
    for (std::size_t slot = 0; slot < 2; ++slot) {
      check(output[slot], host[slot].expected);
      if (round == 0)
        old[slot] = half_test::read<Value>(output[slot]);
      retained.push_back(std::move(left[slot]));
      retained.push_back(std::move(right[slot]));
      retained.push_back(std::move(output[slot]));
    }
    if (round == 1)
      for (std::size_t slot = 0; slot < 2; ++slot)
        check(retained[slot * 3 + 2], old[slot], true);
  }
}

TEST(GgmlCudaBfloatMatvec, ExactRoundingForAlignedAndOddInnerWithBothRhsLayouts) {
  for (const auto inner : {std::int64_t{128}, std::int64_t{127}})
    for (const bool transposed : {true, false})
      matvec<BF16>(inner, transposed);
}

TEST(GgmlCudaBfloatMatvec, BfloatWeightsPreserveFloat32ActivationCancellation) {
  constexpr std::int64_t width = 66, columns = 3;
  auto context = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  for (const auto rows : {std::int64_t{1}, std::int64_t{2}, std::int64_t{17}}) {
    SCOPED_TRACE(rows);
    const auto ad = dense(DataType::kFloat32, {rows, width});
    const auto wd = dense(DataType::kBFloat16, {columns, width});
    const auto yd = dense(DataType::kFloat32, {rows, columns});
    std::vector<float> a(std::size_t(rows * width), 0.0F);
    std::vector<BF16> w(std::size_t(columns * width), half_test::encode<BF16>(0));
    for (std::int64_t row = 0; row < rows; ++row) {
      a[std::size_t(row * width)] = 1.0F + float(row % 3 + 1) / 262144.0F;
      a[std::size_t(row * width + 1)] = -1.0F;
    }
    for (std::int64_t column = 0; column < columns; ++column) {
      const float coefficient = column == 2 ? -1.0F : float(column + 1);
      w[std::size_t(column * width)] = w[std::size_t(column * width + 1)] =
          half_test::encode<BF16>(coefficient);
    }
    EXPECT_EQ(half_test::decode(half_test::encode<BF16>(a[0])), 1.0);
    auto activation = upload<float>(memory.ref(), a), weights = upload<BF16>(memory.ref(), w);
    auto output = output_buffer(memory.ref(), yd);
    const std::array inputs{ConstTensorView::create(ad, activation.view()).value(),
                            ConstTensorView::create(wd, weights.view()).value()};
    const auto status = execute(context, model_ops::dense_linear_f32(), inputs,
                               TensorView::create(yd, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(output);
    for (std::int64_t row = 0; row < rows; ++row)
      for (std::int64_t column = 0; column < columns; ++column) {
        const float coefficient = column == 2 ? -1.0F : float(column + 1);
        EXPECT_EQ(actual[std::size_t(row * columns + column)],
                  coefficient * float(row % 3 + 1) / 262144.0F);
      }
  }
  // Cold-only proof of the original full-vocabulary bank's native domain.
  // No weight/output allocation, Invocation or submit for this large shape.
  const auto head = dense(DataType::kBFloat16, {154880, 4096});
  for (const auto rows : {std::int64_t{2}, std::int64_t{17}}) {
    const auto ad = dense(DataType::kFloat32, {rows, 4096});
    const auto yd = dense(DataType::kFloat32, {rows, 154880});
    const auto prepared = prepare(context, model_ops::dense_linear_f32(),
        std::array{ad, head}, std::array{yd}, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_LT(prepared->workspace_requirement().size, head.required_bytes().value() / 4);
  }
  for (const auto shape : std::array{std::array<std::int64_t,2>{2,65},
                                    std::array<std::int64_t,2>{65536,66}}) {
    const auto ad = dense(DataType::kFloat32, {shape[0], shape[1]});
    const auto wd = dense(DataType::kBFloat16, {3, shape[1]});
    const auto yd = dense(DataType::kFloat32, {shape[0], 3});
    const auto rejected = prepare(context, model_ops::dense_linear_f32(),
        std::array{ad, wd}, std::array{yd}, memory.ref());
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
}

TEST(GgmlCudaBfloatMatvec, BatchedBfloatWeightsPreserveOriginalFloat32Rows) {
  using BF16 = hyle::simd::bfloat16_t;
  constexpr std::int64_t batches=3,width=66,columns=4;
  auto context=Context::create(cuda::device(0).value(),ggml_cuda::provider()).value();
  auto memory=cuda::MemoryResource::create(context,cuda::MemoryKind::kDevice).value();
  for(const auto rows:{std::int64_t{1},std::int64_t{2},std::int64_t{17}}) {
    SCOPED_TRACE(rows);
    const auto ad=dense(DataType::kFloat32,{batches,rows,width});
    const auto wd=dense(DataType::kBFloat16,{batches,columns,width});
    const auto yd=dense(DataType::kFloat32,{batches,rows,columns});
    std::vector<float> a(std::size_t(batches*rows*width),0.F);
    std::vector<BF16> w(std::size_t(batches*columns*width),half_test::encode<BF16>(0));
    for(std::int64_t b=0;b<batches;++b) {
      for(std::int64_t m=0;m<rows;++m) {
        const auto start=std::size_t((b*rows+m)*width);
        a[start]=1.F+float((b+m)%3+1)/262144.F;a[start+1]=-1.F;
      }
      for(std::int64_t n=0;n<columns;++n) {
        const auto first=std::size_t((b*columns+n)*width);
        w[first]=w[first+1]=half_test::encode<BF16>(float((b+1)*(n+1)));
      }
    }
    auto activation=upload<float>(memory.ref(),a),weights=upload<BF16>(memory.ref(),w);
    auto output=output_buffer(memory.ref(),yd);
    const std::array inputs{ConstTensorView::create(ad,activation.view()).value(),
                            ConstTensorView::create(wd,weights.view()).value()};
    const auto status=execute(context,model_ops::dense_linear_f32(),inputs,
        TensorView::create(yd,output.view()).value());
    ASSERT_TRUE(status.ok())<<status.message();
    const auto actual=half_test::read<float>(output);
    for(std::int64_t b=0;b<batches;++b)for(std::int64_t m=0;m<rows;++m)
      for(std::int64_t n=0;n<columns;++n)
        EXPECT_EQ(actual[std::size_t((b*rows+m)*columns+n)],
            float((b+1)*(n+1))*float((b+m)%3+1)/262144.F);
  }
  // Actual Q32 key/value cold shapes: 16MiB BF16 per original compact bank.
  // Descriptor/workspace qualification only; this does not allocate either W.
  for(const auto nk:std::array{std::array<std::int64_t,2>{512,256},std::array<std::int64_t,2>{256,512}}) {
    const auto ad=dense(DataType::kFloat32,{64,1,nk[1]});
    const auto wd=dense(DataType::kBFloat16,{64,nk[0],nk[1]});
    const auto yd=dense(DataType::kFloat32,{64,1,nk[0]});
    const auto prepared=prepare(context,model_ops::dense_linear_f32(),
        std::array{ad,wd},std::array{yd},memory.ref());
    ASSERT_TRUE(prepared)<<prepared.error().message();
    EXPECT_LT(prepared->workspace_requirement().size,wd.required_bytes().value()/4);
  }
  // Multiple original W batches with N>512 fail before native MMF/cuBLAS.
  const auto rejected=prepare(context,model_ops::dense_linear_f32(),
      std::array{dense(DataType::kFloat32,{3,1,66}),dense(DataType::kBFloat16,{3,513,66})},
      std::array{dense(DataType::kFloat32,{3,1,513})},memory.ref());
  ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsUnimplemented());
}

TEST(GgmlCudaBfloatMatvec, Float32SourceAndOutputKeepTheirPrecision) {
  // The F32 control retains input bits that BF16/TF32 would discard.
  matvec<float>(128, true);
}

TEST(GgmlCudaDenseLinear, HalfWeightsPreserveFloat32ActivationAndOutput) {
  using Half = hyle::simd::float16_t;
  auto context = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t width = 66, columns = 4;
  for (const auto rows : {std::int64_t{1}, std::int64_t{2}, std::int64_t{17}}) {
    const auto ad = dense(DataType::kFloat32, {rows, width});
    const auto wd = dense(DataType::kFloat16, {columns, width});
    const auto yd = dense(DataType::kFloat32, {rows, columns});
    std::vector<float> a(std::size_t(rows * width), 0);
    std::vector<Half> w(std::size_t(columns * width), half_test::encode<Half>(0));
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto first = std::size_t(row * width);
      a[first] = 1 + float(row % 3 + 1) / 262144; a[first + 1] = -1;
      a[first + 2] = 1; a[first + 3] = 1.F / 128;
    }
    for (std::int64_t column = 0; column < 3; ++column) {
      const auto first = std::size_t(column * width);
      w[first] = w[first + 1] = half_test::encode<Half>(column == 2 ? -1. : double(column + 1));
    }
    w[std::size_t(3 * width + 2)] = half_test::encode<Half>(1);
    w[std::size_t(3 * width + 3)] = half_test::encode<Half>(1. / 64);
    EXPECT_EQ(half_test::decode(half_test::encode<Half>(a[0])), 1);
    auto activation = upload<float>(memory.ref(), a), weights = upload<Half>(memory.ref(), w);
    auto output = output_buffer(memory.ref(), yd);
    const auto status = execute(context, model_ops::dense_linear_f32(),
        std::array{ConstTensorView::create(ad, activation.view()).value(), ConstTensorView::create(wd, weights.view()).value()},
        TensorView::create(yd, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(output);
    for (std::int64_t row = 0; row < rows; ++row) {
      for (std::int64_t column = 0; column < 3; ++column)
        EXPECT_EQ(actual[std::size_t(row * columns + column)],
            float(column == 2 ? -1 : column + 1) * float(row % 3 + 1) / 262144);
      EXPECT_EQ(actual[std::size_t(row * columns + 3)], 1.F + 1.F / 8192);
      EXPECT_EQ(half_test::decode(half_test::encode<Half>(actual[std::size_t(row * columns + 3)])), 1);
    }
  }
  // Original full vocabulary F16 bank is imported without a full F32 copy.
  // Cold qualification only: no allocation or execution of these large tensors.
  const auto head = dense(DataType::kFloat16, {248320, 2048});
  const auto prepared = prepare(context, model_ops::dense_linear_f32(),
      std::array{dense(DataType::kFloat32, {17, 2048}), head},
      std::array{dense(DataType::kFloat32, {17, 248320})}, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_LT(prepared->workspace_requirement().size, head.required_bytes().value() / 4);
  const auto odd = prepare(context, model_ops::dense_linear_f32(),
      std::array{dense(DataType::kFloat32, {2, 65}), dense(DataType::kFloat16, {4, 65})},
      std::array{dense(DataType::kFloat32, {2, 4})}, memory.ref());
  ASSERT_FALSE(odd); EXPECT_TRUE(odd.error().IsUnimplemented());
}

TEST(GgmlCudaDenseLinear, BatchedBfloatActivationDoesNotNarrowToHalfWeightRange) {
  using Half = hyle::simd::float16_t;
  auto context = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t batches = 3, rows = 2, width = 66, columns = 3;
  const auto ad = dense(DataType::kBFloat16, {batches, rows, width});
  const auto wd = dense(DataType::kFloat16, {batches, columns, width});
  const auto yd = dense(DataType::kFloat32, {batches, rows, columns});
  std::vector<BF16> a(std::size_t(batches * rows * width), half_test::encode<BF16>(0));
  std::vector<Half> w(std::size_t(batches * columns * width), half_test::encode<Half>(0));
  for (std::int64_t batch = 0; batch < batches; ++batch) {
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto first = std::size_t((batch * rows + row) * width);
      a[first] = half_test::encode<BF16>(65536. * double(batch + 1));
      a[first + 1] = half_test::encode<BF16>(-65536. * double(batch + 1));
      a[first + 2] = half_test::encode<BF16>(1); a[first + 3] = half_test::encode<BF16>(.25);
    }
    const auto first = std::size_t(batch * columns * width);
    w[first] = half_test::encode<Half>(1. / 2048);
    w[first + std::size_t(width)] = w[first + std::size_t(width) + 1] = half_test::encode<Half>(1. / 2048);
    w[first + std::size_t(2 * width) + 2] = half_test::encode<Half>(1);
    w[first + std::size_t(2 * width) + 3] = half_test::encode<Half>(1. / 2048);
  }
  auto activation = upload<BF16>(memory.ref(), a), weights = upload<Half>(memory.ref(), w);
  auto output = output_buffer(memory.ref(), yd);
  const auto status = execute(context, model_ops::dense_linear_f32(),
      std::array{ConstTensorView::create(ad, activation.view()).value(), ConstTensorView::create(wd, weights.view()).value()},
      TensorView::create(yd, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = half_test::read<float>(output);
  for (std::int64_t batch = 0; batch < batches; ++batch) for (std::int64_t row = 0; row < rows; ++row) {
    const auto first = std::size_t((batch * rows + row) * columns);
    EXPECT_EQ(actual[first], float(32 * (batch + 1))); EXPECT_EQ(actual[first + 1], 0);
    EXPECT_EQ(actual[first + 2], 1.F + 1.F / 8192);
  }
}

TEST(GgmlCudaDenseLinear, OriginalFloat32WeightAndActivationBitsSurvive) {
  auto context = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto ad = dense(DataType::kFloat32, {2, 66}), wd = dense(DataType::kFloat32, {3, 66});
  const auto yd = dense(DataType::kFloat32, {2, 3});
  std::vector<float> a(132, 0), w(198, 0);
  for (std::size_t row = 0; row < 2; ++row) { a[row * 66] = 1.F + float(row + 1) / 262144; a[row * 66 + 1] = -1; }
  for (std::size_t column = 0; column < 3; ++column) w[column * 66] = w[column * 66 + 1] = float(column + 1);
  auto activation = upload<float>(memory.ref(), a), weights = upload<float>(memory.ref(), w);
  auto output = output_buffer(memory.ref(), yd);
  const auto status = execute(context, model_ops::dense_linear_f32(),
      std::array{ConstTensorView::create(ad, activation.view()).value(), ConstTensorView::create(wd, weights.view()).value()},
      TensorView::create(yd, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message(); const auto actual = half_test::read<float>(output);
  for (std::size_t row = 0; row < 2; ++row) for (std::size_t column = 0; column < 3; ++column)
    EXPECT_EQ(actual[row * 3 + column], float((row + 1) * (column + 1)) / 262144);
  const auto odd = prepare(context, model_ops::dense_linear_f32(),
      std::array{dense(DataType::kFloat32, {2, 65}), dense(DataType::kFloat32, {3, 65})},
      std::array{dense(DataType::kFloat32, {2, 3})}, memory.ref());
  ASSERT_FALSE(odd); EXPECT_TRUE(odd.error().IsUnimplemented());
}

template<class A, class W> void dense_pair(Context &context, mem::MemoryResourceRef resource) {
  const auto ad = dense(half_test::data_type<A>(), {2, 66});
  const auto wd = dense(half_test::data_type<W>(), {3, 66}), yd = dense(DataType::kFloat32, {2, 3});
  std::vector<A> a(132, half_test::encode<A>(0)); std::vector<W> w(198, half_test::encode<W>(0));
  for (std::size_t row = 0; row < 2; ++row) {
    a[row * 66] = half_test::encode<A>(1 + double(row + 1) / 262144);
    a[row * 66 + 1] = half_test::encode<A>(-1); a[row * 66 + 2] = half_test::encode<A>(.25);
  }
  w[0] = w[1] = half_test::encode<W>(1); w[66 + 2] = half_test::encode<W>(1 + 1. / 262144);
  w[132 + 2] = half_test::encode<W>(1. / 2048);
  auto activation = upload<A>(resource, a), weights = upload<W>(resource, w); auto output = output_buffer(resource, yd);
  const auto status = execute(context, model_ops::dense_linear_f32(),
      std::array{ConstTensorView::create(ad, activation.view()).value(), ConstTensorView::create(wd, weights.view()).value()},
      TensorView::create(yd, output.view()).value());
  ASSERT_TRUE(status.ok()) << status.message(); const auto actual = half_test::read<float>(output);
  for (std::size_t row = 0; row < 2; ++row) for (std::size_t column = 0; column < 3; ++column) {
    double expected = 0;
    for (std::size_t k = 0; k < 66; ++k)
      expected += half_test::decode(a[row * 66 + k]) * half_test::decode(w[column * 66 + k]);
    EXPECT_EQ(actual[row * 3 + column], expected);
  }
}
TEST(GgmlCudaDenseLinear, AllDenseFloatingPairsUseOriginalOperands) {
  using Half = hyle::simd::float16_t;
  auto context = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  dense_pair<Half, Half>(context, memory.ref()); dense_pair<Half, BF16>(context, memory.ref());
  dense_pair<Half, float>(context, memory.ref()); dense_pair<BF16, Half>(context, memory.ref());
  dense_pair<BF16, BF16>(context, memory.ref()); dense_pair<BF16, float>(context, memory.ref());
  dense_pair<float, Half>(context, memory.ref()); dense_pair<float, BF16>(context, memory.ref());
  dense_pair<float, float>(context, memory.ref());
}
} // namespace
