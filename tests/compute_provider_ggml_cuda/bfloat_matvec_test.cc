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
    const auto status = execute(context, model_ops::bfloat_linear_f32(), inputs,
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
    const auto prepared = prepare(context, model_ops::bfloat_linear_f32(),
        std::array{ad, head}, std::array{yd}, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_LT(prepared->workspace_requirement().size, head.required_bytes().value() / 4);
  }
  for (const auto shape : std::array{std::array<std::int64_t,2>{2,65},
                                    std::array<std::int64_t,2>{65536,66}}) {
    const auto ad = dense(DataType::kFloat32, {shape[0], shape[1]});
    const auto wd = dense(DataType::kBFloat16, {3, shape[1]});
    const auto yd = dense(DataType::kFloat32, {shape[0], 3});
    const auto rejected = prepare(context, model_ops::bfloat_linear_f32(),
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
    const auto status=execute(context,model_ops::bfloat_linear_f32(),inputs,
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
    const auto prepared=prepare(context,model_ops::bfloat_linear_f32(),
        std::array{ad,wd},std::array{yd},memory.ref());
    ASSERT_TRUE(prepared)<<prepared.error().message();
    EXPECT_LT(prepared->workspace_requirement().size,wd.required_bytes().value()/4);
  }
  // Multiple original W batches with N>512 fail before native MMF/cuBLAS.
  const auto rejected=prepare(context,model_ops::bfloat_linear_f32(),
      std::array{dense(DataType::kFloat32,{3,1,66}),dense(DataType::kBFloat16,{3,513,66})},
      std::array{dense(DataType::kFloat32,{3,1,513})},memory.ref());
  ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsUnimplemented());
}

TEST(GgmlCudaBfloatMatvec, Float32SourceAndOutputKeepTheirPrecision) {
  // The F32 control retains input bits that BF16/TF32 would discard.
  matvec<float>(128, true);
}
} // namespace
