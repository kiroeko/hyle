#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>
#include "hyle/compute/model_ops.h"

#if defined(HYLE_GDN_SCALE_CPU) || defined(HYLE_GDN_SCALE_CUDA)
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "../../src/compute_ggml_runtime/gated_delta_scale.h"
#if defined(HYLE_GDN_SCALE_CPU)
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/ggml_cpu.h"
#else
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#endif
#elif defined(HYLE_GDN_SCALE_APPLE)
#include "hyle/compute/apple.h"
#elif defined(HYLE_GDN_SCALE_VULKAN)
#include "../compute_route_vulkan/tensor_test_support.h"
#else
#error Select the focused GDN scale backend
#endif

namespace {
namespace c = hyle::compute;
struct Fixture final {
  static constexpr std::int64_t depth = 128;
  static constexpr float delta = 1.F / 1048576.F;
  bool recurrent;
  std::vector<c::TensorDesc> inputs;
  std::vector<std::vector<float>> data;
  Fixture(bool kda, bool raw_gates) : recurrent(raw_gates) {
    const auto add = [&](std::array<std::int64_t,4> shape) {
      auto d = c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
      inputs.push_back(d); data.emplace_back(d.element_count().value(), 0.F);
    };
    for (int port = 0; port != 3; ++port) add({1,2,1,depth});
    add({1,2,1,kda ? depth : 1}); add({1,2,1,1});
    if (raw_gates) { add({1,1,1,1}); add({1,1,1,1}); }
    add({1,1,depth,depth});
    for (int t = 0; t != 2; ++t) {
      data[0][std::size_t(t*depth)] = 1.F + delta;
      data[0][std::size_t(t*depth+1)] = -1.F;
    }
    for (std::int64_t row = 0; row < depth; ++row)
      data.back()[std::size_t(row*depth)] =
          data.back()[std::size_t(row*depth+1)] = float(row%3+1);
  }
  c::Operation operation(std::optional<float> scale) const {
    return recurrent ? c::model_ops::recurrent_gated_delta_net(2, scale)
                     : c::Operation(c::GatedDeltaNet{.snapshot_count=2, .output_scale=scale});
  }
  std::array<c::TensorDesc,2> outputs(std::optional<float> scale) const {
    std::array<c::TensorDesc,2> result;
    EXPECT_TRUE(c::infer_output_descriptions(operation(scale), inputs, result).ok());
    return result;
  }
  void check(std::span<const float> scores, std::span<const float> states,
             std::optional<float> scale) const {
    ASSERT_EQ(scores.size(), 2*std::size_t(depth));
    ASSERT_EQ(states.size(), 2*data.back().size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
      const float raw = float((i%std::size_t(depth))%3+1)*delta;
      if (scale) EXPECT_EQ(scores[i], raw* *scale);
      else EXPECT_NEAR(scores[i], raw*(1.F/std::sqrt(float(depth))), 1.e-12F);
    }
    for (std::size_t i = 0; i < states.size(); ++i)
      EXPECT_EQ(states[i], data.back()[i%data.back().size()]);
  }
};

#if defined(HYLE_GDN_SCALE_CPU) || defined(HYLE_GDN_SCALE_CUDA)
void exercise(c::Context &context, hyle::mem::MemoryResourceRef resource) {
  auto queue = context.create_queue().value();
  for (int mode = 0; mode != 3; ++mode) {
    const Fixture f(mode == 1, mode == 2);
    for (const std::optional<float> scale : {std::optional<float>{}, std::optional<float>{1.F}}) {
      SCOPED_TRACE(::testing::Message() << "mode=" << mode << " explicit=" << scale.has_value());
      std::vector<hyle::mem::Buffer> owners;
      std::vector<c::ConstTensorView> inputs;
      owners.reserve(f.inputs.size());
      for (std::size_t port = 0; port < f.inputs.size(); ++port) {
        owners.push_back(hyle::mem::Buffer::create(resource, f.inputs[port].required_bytes().value(),256).value());
        ASSERT_TRUE(owners.back().write(0, std::as_bytes(std::span(f.data[port]))).ok());
        inputs.push_back(c::ConstTensorView::create(f.inputs[port],owners.back().view()).value());
      }
      const auto desc = f.outputs(scale);
      std::array<hyle::mem::Buffer,2> output_owners;
      std::array<c::TensorView,2> outputs;
      for (std::size_t port=0;port<2;++port) {
        output_owners[port]=hyle::mem::Buffer::create(resource,desc[port].required_bytes().value(),256).value();
        outputs[port]=c::TensorView::create(desc[port],output_owners[port].view()).value();
      }
      auto prepared=c::prepare(context,f.operation(scale),inputs,outputs,resource);
      ASSERT_TRUE(prepared)<<prepared.error().message();
      const auto w=prepared->workspace_requirement();
      auto workspace=w.size?hyle::mem::Buffer::create(resource,w.size,w.alignment).value():hyle::mem::Buffer{};
      auto invocation=prepared->create_invocation(workspace.view());
      ASSERT_TRUE(invocation)<<invocation.error().message();
      auto completion=invocation->submit(queue,inputs,outputs);
      ASSERT_TRUE(completion)<<completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      std::array<std::vector<float>,2> actual;
      for(std::size_t port=0;port<2;++port) {
        actual[port].resize(desc[port].element_count().value());
        ASSERT_TRUE(output_owners[port].read(0,std::as_writable_bytes(std::span(actual[port]))).ok());
      }
      f.check(actual[0],actual[1],scale);
      std::vector<float> retained(f.data.back().size());
      ASSERT_TRUE(owners.back().read(0,std::as_writable_bytes(std::span(retained))).ok());
      EXPECT_EQ(retained,f.data.back());
    }
  }
}
TEST(GdnOutputScale, RawDefaultAndFusedGatesKeepStateSnapshots) {
#if defined(HYLE_GDN_SCALE_CPU)
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider()).value();
  exercise(context,hyle::mem::system_resource());
#else
  auto context=c::Context::create(c::cuda::device(0).value(),c::ggml_cuda::provider()).value();
  auto resource=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  exercise(context,resource.ref());
#endif
}
TEST(GdnOutputScale, ColdParameterIdentityRetainsPresenceAndSignedFloatBits) {
  ggml_tensor node{};
  node.op_params[0]=2;node.op_params[1]=0x48594731;
  c::ggml_runtime::set_gdn_output_scale(&node,{});
  EXPECT_FALSE(c::ggml_runtime::has_gdn_output_scale(&node));
  for(float value:{1.F,0.F,-0.F,-2.F}) {
    c::ggml_runtime::set_gdn_output_scale(&node,value);
    EXPECT_TRUE(c::ggml_runtime::has_gdn_output_scale(&node));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(c::ggml_runtime::gdn_output_scale(&node,17.F)),
              std::bit_cast<std::uint32_t>(value));
  }
  EXPECT_EQ(node.op_params[0],2);EXPECT_EQ(node.op_params[1],0x48594731);
}
#else
TEST(GdnOutputScale, RawDefaultAndFusedGatesKeepStateSnapshots) {
#if defined(HYLE_GDN_SCALE_APPLE)
  namespace mx=mlx::core;namespace a=c::apple;
  auto context=a::Context::create();ASSERT_TRUE(context)<<context.error().message();
#else
  namespace v=c::vulkan;
  auto context=vulkan_test::context().value();
#endif
  for(int mode=0;mode!=3;++mode) {
    const Fixture f(mode==1,mode==2);
    for(const std::optional<float> scale:{std::optional<float>{},std::optional<float>{1.F}}) {
      SCOPED_TRACE(::testing::Message()<<"mode="<<mode<<" explicit="<<scale.has_value());
#if defined(HYLE_GDN_SCALE_APPLE)
      std::vector<mx::array> inputs;std::vector<a::InputSpec> specs;
      for(std::size_t port=0;port<f.inputs.size();++port) {
        mx::Shape shape;for(auto d:f.inputs[port].dimensions())shape.push_back(int(d));
        inputs.emplace_back(f.data[port].data(),shape);
        specs.push_back({shape,inputs.back().dtype()});
      }
      auto prepared=context->prepare(f.operation(scale),specs);
      ASSERT_TRUE(prepared)<<prepared.error().message();
      auto invocation=prepared->create_invocation();ASSERT_TRUE(invocation);
      ASSERT_TRUE(invocation->bind(inputs).ok());
      auto work=invocation->submit();ASSERT_TRUE(work)<<work.error().message();
      auto done=work->wait();ASSERT_TRUE(done)<<done.error().message();
      const auto result=done->get()->arrays();ASSERT_EQ(result.size(),2U);
      f.check({result[0].data<float>(),result[0].size()},{result[1].data<float>(),result[1].size()},scale);
      EXPECT_EQ(std::memcmp(inputs.back().data<float>(),f.data.back().data(),f.data.back().size()*sizeof(float)),0);
#else
      std::vector<v::Argument> specs;std::vector<v::Tensor> inputs;
      for(std::size_t port=0;port<f.inputs.size();++port) {
        specs.emplace_back(f.inputs[port]);inputs.push_back(vulkan_test::upload(context,f.inputs[port],f.data[port]).value());
      }
      auto prepared=v::prepare(context,f.operation(scale),specs);
      ASSERT_TRUE(prepared)<<prepared.error().message();
      auto invocation=prepared->create_invocation();ASSERT_TRUE(invocation);
      ASSERT_TRUE(invocation->bind(inputs).ok());
      auto work=invocation->submit();ASSERT_TRUE(work)<<work.error().message();
      auto done=work->wait();ASSERT_TRUE(done)<<done.error().message();
      const auto result=done->get()->tensors();ASSERT_EQ(result.size(),2U);
      auto y=vulkan_test::read(context,result[0]).value(),s=vulkan_test::read(context,result[1]).value();
      f.check(y,s,scale);EXPECT_EQ(vulkan_test::read(context,inputs.back()).value(),f.data.back());
#endif
    }
  }
}
#endif

TEST(GdnOutputScale, PublicInferenceRejectsNonfiniteAndRetainsExplicitDomain) {
  for(bool recurrent:{false,true}) {
    const Fixture f(false,recurrent);
    for(float scale:{0.F,-0.F,-2.F,1.F}) {
      std::array<c::TensorDesc,2> outputs;
      EXPECT_TRUE(c::infer_output_descriptions(f.operation(scale),f.inputs,outputs).ok());
    }
    for(float scale:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN()}) {
      std::array<c::TensorDesc,2> outputs;
      EXPECT_TRUE(c::infer_output_descriptions(f.operation(scale),f.inputs,outputs).IsInvalidArgument());
      EXPECT_FALSE(outputs[0].valid());EXPECT_FALSE(outputs[1].valid());
    }
  }
}
} // namespace
