#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_models/d5_gguf_test_support.h"

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

TEST(GgmlCudaQuantizedGlu, OriginalProjectionsAndGluMatchCapturedRebindingWithPrivateBoundaries) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  constexpr std::int64_t k = 256, n = 257;
  const auto a_desc = dense(c::DataType::kFloat32, {1,k});
  const auto y_desc = dense(c::DataType::kFloat32, {1,n});
  for (const auto format : {c::GgufQuantizedType::kQ8_0,
       c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K}) {
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
      EXPECT_EQ(paired,escape == 0 ? 1U : 0U);
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
} // namespace
