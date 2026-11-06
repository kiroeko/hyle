#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_models/d5_gguf_test_support.h"
#include "../../src/compute_provider_ggml_cuda/packed_matmul_native.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
constexpr c::RoutePolicyId kPolicy{0x48594c4547474355ULL, 8, 1};

TEST(GgmlCudaF32GdnNorm, WeightedNormMatchesFloat64AndStandaloneProduct) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  constexpr std::size_t width=128, rows=32;
  const auto desc=dense(c::DataType::kFloat32,{rows,width});
  const auto weight_desc=dense(c::DataType::kFloat32,{width});
  auto builder=g::GraphBuilder::create().value();
  const auto x=builder.input(desc).value(), w=builder.input(weight_desc).value();
  const auto node=builder.add_node(c::Operation(c::RmsNorm{.epsilon=1.e-6F}),std::array{x,w}).value();
  const auto y=builder.result(node).value(); ASSERT_TRUE(builder.publish(y).ok());
  auto compiled=g::cuda::compile(std::move(builder).finish().value(),context,owner).value();
  auto slot=compiled.executable.create_invocation().value();
  for (int round=0;round<2;++round) {
    const auto values=d5_gguf_test::values(rows,width,19+round);
    std::vector<float> scale(width);
    for (std::size_t i=0;i<width;++i) scale[i]=float(int((i+round)%17)-8)*.25F;
    auto input=upload<float>(owner.ref(),values), weight=upload<float>(owner.ref(),scale);
    auto normalized=output_buffer(owner.ref(),desc), staged=output_buffer(owner.ref(),desc);
    auto output=output_buffer(owner.ref(),desc);
    const auto xv=c::ConstTensorView::create(desc,input.view()).value();
    const auto wv=c::ConstTensorView::create(weight_desc,weight.view()).value();
    ASSERT_TRUE(ggml_cuda_tensor_test::execute(context,c::Operation(c::RmsNorm{.epsilon=1.e-6F}),
        std::span(&xv,1),c::TensorView::create(desc,normalized.view()).value()).ok());
    const std::array inputs{c::ConstTensorView::create(desc,normalized.view()).value(),wv};
    ASSERT_TRUE(ggml_cuda_tensor_test::execute(context,c::Operation(c::Binary{c::BinaryKind::kMultiply}),
        inputs,c::TensorView::create(desc,staged.view()).value()).ok());
    ASSERT_TRUE(slot.bind(x,xv).ok()); ASSERT_TRUE(slot.bind(w,wv).ok());
    ASSERT_TRUE(slot.bind(y,c::TensorView::create(desc,output.view()).value()).ok());
    ASSERT_TRUE(g::cuda::prepare_replay(slot,queue,{.mode=g::ReplayMode::kRequireCapture}).ok());
    auto done=slot.submit(queue).value(); ASSERT_TRUE(done.wait().ok());
    const auto got=half_test::read<float>(output), previous=half_test::read<float>(staged);
    EXPECT_EQ(got,previous);
    for (std::size_t row=0;row<rows;++row) {
      double square=0;
      for (std::size_t i=0;i<width;++i) square+=double(values[row*width+i])*values[row*width+i];
      for (std::size_t i=0;i<width;++i) {
        const double reference=double(values[row*width+i])/std::sqrt(square/width+1.e-6)*scale[i];
        EXPECT_NEAR(double(got[row*width+i]),reference,2.e-7+std::abs(reference)*2.e-6);
      }
    }
    auto held=slot.result(y).value(); EXPECT_TRUE(slot.reset().IsUnavailable());
    held.reset(); ASSERT_TRUE(slot.reset().ok());
  }
}

TEST(GgmlCudaF32L2, AdditiveEpsilonMatchesFloat64AcrossMagnitudesAndCapturedRebinding) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  for (const std::int64_t width : {64,96,128,256}) {
    const auto desc=dense(c::DataType::kFloat32,{7,width});
    const auto weights=dense(c::DataType::kFloat32,{width});
    auto builder=g::GraphBuilder::create().value();
    const auto x=builder.input(desc).value(), w=builder.input(weights).value();
    const auto y=builder.add_node(c::Operation(c::RmsNorm{.epsilon=1.e-6F/float(width)}),
        std::array{x,w}).value();
    const auto result=builder.result(y).value(); ASSERT_TRUE(builder.publish(result).ok());
    auto compiled=g::cuda::compile(std::move(builder).finish().value(),context,owner).value();
    auto slot=compiled.executable.create_invocation().value();
    for (int round=0;round<2;++round) {
      std::vector<float> values(std::size_t(7*width));
      constexpr std::array magnitudes{0.F,1.e-8F,1.e-5F,1.e-3F,1.F,1.e4F,1.e10F};
      for (std::size_t i=0;i<values.size();++i)
        values[i]=float(int((i+std::size_t(round)*7)%19)-9)*magnitudes[i/std::size_t(width)]/16.F;
      const std::vector<float> scale(std::size_t(width),1.F/std::sqrt(float(width)));
      auto source=upload<float>(owner.ref(),values), weight=upload<float>(owner.ref(),scale);
      auto output=output_buffer(owner.ref(),desc);
      ASSERT_TRUE(slot.bind(x,c::ConstTensorView::create(desc,source.view()).value()).ok());
      ASSERT_TRUE(slot.bind(w,c::ConstTensorView::create(weights,weight.view()).value()).ok());
      ASSERT_TRUE(slot.bind(result,c::TensorView::create(desc,output.view()).value()).ok());
      ASSERT_TRUE(g::cuda::prepare_replay(slot,queue,{.mode=g::ReplayMode::kRequireCapture}).ok());
      auto done=slot.submit(queue).value(); ASSERT_TRUE(done.wait().ok());
      const auto got=half_test::read<float>(output);
      for (std::size_t row=0;row<7;++row) {
        double square=0;
        for (std::size_t col=0;col<std::size_t(width);++col) {
          const double v=values[row*std::size_t(width)+col]; square+=v*v;
        }
        for (std::size_t col=0;col<std::size_t(width);++col) {
          const auto i=row*std::size_t(width)+col;
          const double expected=double(values[i])/std::sqrt(square+1.e-6);
          EXPECT_NEAR(double(got[i]),expected,2.e-7+std::abs(expected)*2.e-6)<<row<<","<<col;
        }
      }
      ASSERT_TRUE(slot.reset().ok());
    }
  }
}

void check_quantized_glu(std::int64_t k) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  constexpr std::int64_t n = 257;
  const auto a_desc = dense(c::DataType::kFloat32, {1,k});
  const auto y_desc = dense(c::DataType::kFloat32, {1,n});
  std::vector formats{c::GgufQuantizedType::kQ8_0,
      c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K};
  if (k == 512) {
    const std::array additional{c::GgufQuantizedType::kQ4_0, c::GgufQuantizedType::kQ4_1,
        c::GgufQuantizedType::kQ2K, c::GgufQuantizedType::kQ3K, c::GgufQuantizedType::kQ5K,
        c::GgufQuantizedType::kIq1S, c::GgufQuantizedType::kIq1M,
        c::GgufQuantizedType::kIq2Xxs, c::GgufQuantizedType::kIq2Xs,
        c::GgufQuantizedType::kIq2S, c::GgufQuantizedType::kIq3Xxs,
        c::GgufQuantizedType::kIq3S, c::GgufQuantizedType::kIq4Nl,
        c::GgufQuantizedType::kIq4Xs, c::GgufQuantizedType::kMxfp4,
        c::GgufQuantizedType::kNvfp4};
    formats.insert(formats.end(), additional.begin(), additional.end());
  }
  for (const auto format : formats) {
    const auto w_desc = c::gguf_weight_description(format,c::DataType::kFloat32,k,n).value();
    for (int escape = 0; escape < 4; ++escape) {
      SCOPED_TRACE(static_cast<int>(format));
      SCOPED_TRACE(escape);
      auto builder = g::GraphBuilder::create().value();
      const auto a = builder.input(a_desc).value();
      const auto other_a = escape == 2 ? builder.input(a_desc).value() : a;
      const std::array weights{builder.input(w_desc).value(),builder.input(w_desc).value()};
      const auto first = builder.add_node(c::Operation(c::MatMul{}),std::array{a,weights[0]}).value();
      const auto gate = builder.result(first).value();
      g::NodeId control;
      if (escape == 3) {
        control = builder.add_node(c::Operation(c::Copy{}),std::array{a}).value();
        ASSERT_TRUE(builder.depends_on(control,first).ok());
      }
      const auto second = builder.add_node(c::Operation(c::MatMul{}),std::array{other_a,weights[1]}).value();
      if (escape == 3) ASSERT_TRUE(builder.depends_on(second,control).ok());
      const auto up = builder.result(second).value();
      const auto result = builder.add_node(c::Operation(c::GatedActivation{
          .activation=c::GatedActivationKind::kSilu,.inputs=c::GatedInputKind::kSeparate}),
          std::array{gate,up}).value();
      const auto y = builder.result(result).value();
      ASSERT_TRUE(builder.publish(y).ok());
      if (escape == 1) ASSERT_TRUE(builder.publish(gate).ok());
      const auto graph = std::move(builder).finish().value();
      auto compiled = g::cuda::compile(graph,context,owner).value();
      std::size_t paired = 0;
      for (std::size_t i=0;i<compiled.report.step_count();++i)
        paired += compiled.report.step(i).value().policy == kPolicy;
      const bool legacy = format == c::GgufQuantizedType::kQ8_0 ||
          format == c::GgufQuantizedType::kQ4K || format == c::GgufQuantizedType::kQ6K;
      namespace native = c::ggml_cuda::internal;
      const bool admitted = legacy || native::packed_matmul_plan(
          0, d5_gguf_test::native_type(format), k, n, 1).route == native::PackedMatmulRoute::kMmvq;
      EXPECT_EQ(paired, escape == 0 && admitted ? 1U : 0U);
      auto slot = compiled.executable.create_invocation().value();
      for (int round=0;round<2;++round) {
        const auto values=d5_gguf_test::values(1,k,11+round);
        const auto other_values=d5_gguf_test::values(1,k,31+round);
        auto ab = upload<float>(owner.ref(),values);
        auto other = upload<float>(owner.ref(),other_values);
        const auto p0 = d5_gguf_test::pack(format,d5_gguf_test::values(n,k,3+round),n,k,w_desc.required_bytes().value());
        const auto p1 = d5_gguf_test::pack(format,d5_gguf_test::values(n,k,7+round),n,k,w_desc.required_bytes().value());
        auto w0 = upload<std::byte>(owner.ref(),p0.bytes), w1 = upload<std::byte>(owner.ref(),p1.bytes);
        auto expected_gate=output_buffer(owner.ref(),y_desc), expected_up=output_buffer(owner.ref(),y_desc);
        auto expected=output_buffer(owner.ref(),y_desc), actual=output_buffer(owner.ref(),y_desc);
        auto observed=output_buffer(owner.ref(),y_desc);
        const auto av=c::ConstTensorView::create(a_desc,ab.view()).value();
        const auto bv=c::ConstTensorView::create(a_desc,(escape == 2 ? other : ab).view()).value();
        const auto wv0=c::ConstTensorView::create(w_desc,w0.view()).value();
        const auto wv1=c::ConstTensorView::create(w_desc,w1.view()).value();
        const std::array inputs0{av,wv0}, inputs1{bv,wv1};
        ASSERT_TRUE(ggml_cuda_tensor_test::execute(context,c::Operation(c::MatMul{}),inputs0,
            c::TensorView::create(y_desc,expected_gate.view()).value()).ok());
        ASSERT_TRUE(ggml_cuda_tensor_test::execute(context,c::Operation(c::MatMul{}),inputs1,
            c::TensorView::create(y_desc,expected_up.view()).value()).ok());
        const std::array products{c::ConstTensorView::create(y_desc,expected_gate.view()).value(),
            c::ConstTensorView::create(y_desc,expected_up.view()).value()};
        ASSERT_TRUE(ggml_cuda_tensor_test::execute(context,c::Operation(c::GatedActivation{
            .activation=c::GatedActivationKind::kSilu,.inputs=c::GatedInputKind::kSeparate}),products,
            c::TensorView::create(y_desc,expected.view()).value()).ok());
        ASSERT_TRUE(slot.bind(a,av).ok());
        if (escape == 2) ASSERT_TRUE(slot.bind(other_a,bv).ok());
        ASSERT_TRUE(slot.bind(weights[0],wv0).ok()); ASSERT_TRUE(slot.bind(weights[1],wv1).ok());
        ASSERT_TRUE(slot.bind(y,c::TensorView::create(y_desc,actual.view()).value()).ok());
        if (escape == 1) ASSERT_TRUE(slot.bind(gate,c::TensorView::create(y_desc,observed.view()).value()).ok());
        ASSERT_TRUE(g::cuda::prepare_replay(slot,queue,{.mode=g::ReplayMode::kRequireCapture}).ok());
        auto done=slot.submit(queue).value(); ASSERT_TRUE(done.wait().ok());
        const auto got=half_test::read<float>(actual), want=half_test::read<float>(expected);
        for (std::size_t i=0;i<got.size();++i) EXPECT_NEAR(got[i],want[i],2.e-5F+std::abs(want[i])*2.e-5F)<<i;
        if (escape == 0) {
          // The singleton decode route may fuse these projections, whereas a
          // two-row verifier cannot. Compare the same token across both shapes.
          const auto a2_desc=dense(c::DataType::kFloat32,{2,k});
          const auto y2_desc=dense(c::DataType::kFloat32,{2,n});
          auto b2=g::GraphBuilder::create().value();
          const auto a2=b2.input(a2_desc).value();
          const auto w20=b2.input(w_desc).value(), w21=b2.input(w_desc).value();
          const auto g2=b2.result(b2.add_node(c::Operation(c::MatMul{}),std::array{a2,w20}).value()).value();
          const auto u2=b2.result(b2.add_node(c::Operation(c::MatMul{}),std::array{a2,w21}).value()).value();
          const auto y2=b2.result(b2.add_node(c::Operation(c::GatedActivation{
              .activation=c::GatedActivationKind::kSilu,.inputs=c::GatedInputKind::kSeparate}),
              std::array{g2,u2}).value()).value();
          ASSERT_TRUE(b2.publish(y2).ok());
          auto compiled2=g::cuda::compile(std::move(b2).finish().value(),context,owner).value();
          std::size_t paired2=0;
          for (std::size_t i=0;i<compiled2.report.step_count();++i)
            paired2 += compiled2.report.step(i).value().policy == kPolicy;
          EXPECT_EQ(paired2,0U);
          auto slot2=compiled2.executable.create_invocation().value();
          auto values2=other_values; values2.insert(values2.end(),values.begin(),values.end());
          auto input2=upload<float>(owner.ref(),values2), output2=output_buffer(owner.ref(),y2_desc);
          ASSERT_TRUE(slot2.bind(a2,c::ConstTensorView::create(a2_desc,input2.view()).value()).ok());
          ASSERT_TRUE(slot2.bind(w20,wv0).ok()); ASSERT_TRUE(slot2.bind(w21,wv1).ok());
          ASSERT_TRUE(slot2.bind(y2,c::TensorView::create(y2_desc,output2.view()).value()).ok());
          ASSERT_TRUE(g::cuda::prepare_replay(slot2,queue,{.mode=g::ReplayMode::kRequireCapture}).ok());
          auto done2=slot2.submit(queue).value(); ASSERT_TRUE(done2.wait().ok());
          const auto two=half_test::read<float>(output2);
          for (std::size_t i=0;i<got.size();++i)
            EXPECT_NEAR(two[std::size_t(n)+i],got[i],2.e-5F+std::abs(got[i])*2.e-5F)<<i;
          ASSERT_TRUE(slot2.reset().ok());
        }
        // Independent float64 decoded-weight oracle. Bound the original Q8_1
        // activation quantization by half a step per 32-element block, then
        // propagate that bound through SiLU (|derivative| < 1.1) and product.
        const auto dot = [&](const auto &x,const auto &w,std::size_t row) {
          std::array<double,2> sum{};
          for (std::size_t block=0;block<std::size_t(k);block+=32) {
            double maximum=0;
            for (std::size_t j=block;j<block+32;++j) maximum=std::max(maximum,std::abs(double(x[j])));
            for (std::size_t j=block;j<block+32;++j) {
              sum[0]+=double(x[j])*w[row*std::size_t(k)+j];
              sum[1]+=maximum/254.*std::abs(double(w[row*std::size_t(k)+j]));
            }
          }
          return sum;
        };
        for (std::size_t i=0;i<got.size();++i) {
          const auto gate_dot=dot(values,p0.decoded,i);
          const auto up_dot=dot(escape == 2 ? other_values : values,p1.decoded,i);
          const double silu=gate_dot[0]/(1.+std::exp(-gate_dot[0]));
          const double bound=1.1*gate_dot[1]*(std::abs(up_dot[0])+up_dot[1])+std::abs(silu)*up_dot[1]+2.e-5;
          EXPECT_NEAR(double(got[i]),silu*up_dot[0],bound)<<i;
        }
        if (escape == 1) EXPECT_EQ(half_test::read<float>(observed),half_test::read<float>(expected_gate));
        auto held=slot.result(y).value(); EXPECT_TRUE(slot.reset().IsUnavailable());
        held.reset(); ASSERT_TRUE(slot.reset().ok());
      }
    }
  }
}

TEST(GgmlCudaQuantizedGlu, OriginalWidthPreservesCapturedRebinding) { check_quantized_glu(256); }
TEST(GgmlCudaQuantizedGlu, ProviderQualifiedFormatsPreserveCapturedRebinding) { check_quantized_glu(512); }
} // namespace
