#include "tensor_test_support.h"
#include "hyle/compute/sequence_ops.h"

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;

struct Fixture final {
  static constexpr std::size_t rows = 8, batches = 2, hk = 2, hv = 4, dv = 2;
  std::size_t dk, gate_width;
  std::array<c::TensorDesc, 7> descriptions;
  std::array<std::vector<float>, 6> data;

  Fixture(std::size_t depth, std::size_t gates) : dk(depth), gate_width(gates) {
    const auto f32 = c::DataType::kFloat32;
    const auto dense = [](c::DataType type, std::initializer_list<std::int64_t> shape) {
      return vulkan_tensor_test::dense(type, shape);
    };
    descriptions = {dense(f32, {rows,hk,std::int64_t(dk)}), dense(f32, {rows,hk,std::int64_t(dk)}),
        dense(f32, {rows,hv,dv}), dense(f32, {rows,hv,std::int64_t(gate_width)}),
        dense(f32, {rows,hv,1}), dense(f32, {batches,hv,dv,std::int64_t(dk)}),
        dense(c::DataType::kInt32, {batches+1})};
    for (std::size_t port = 0; port < data.size(); ++port) {
      data[port].resize(*descriptions[port].element_count());
      for (std::size_t i = 0; i < data[port].size(); ++i)
        data[port][i] = float(int((i*7+port*11)%19)-9)/64.0F;
    }
    for (auto &x : data[3]) x = -0.17F + x / 4;
    for (auto &x : data[4]) x = 0.4F + x / 8;
    for (std::size_t port = 0; port < 5; ++port) {
      const auto per_row = data[port].size()/rows;
      for (std::size_t i = 4*per_row; i < data[port].size(); ++i)
        data[port][i] = std::numeric_limits<float>::quiet_NaN();
    }
  }

  hyle::Result<v::Program> prepare(const v::Context &context, std::uint32_t snapshots = 1,
                                  std::optional<float> scale = {}) const {
    return v::prepare(context, c::sequence_ops::packed_gated_delta_net(snapshots, scale),
        descriptions[0], descriptions[1], descriptions[2], descriptions[3],
        descriptions[4], descriptions[5], descriptions[6]);
  }

  std::array<v::Tensor, 7> upload(const v::Context &context,
      const std::array<std::int32_t,3> &offsets, bool borrowed = false,
      std::vector<std::pair<v::Tensor,std::vector<std::byte>>> *guards = nullptr) const {
    std::array<v::Tensor, 7> tensors;
    for (std::size_t port = 0; port < tensors.size(); ++port) {
      const auto bytes = port == 6 ? std::as_bytes(std::span(offsets))
          : std::as_bytes(std::span(data[port]));
      if (!borrowed) tensors[port] = context.upload(descriptions[port], bytes).value();
      else {
        const auto prefix = (port+1)*4;
        std::vector<std::byte> raw(prefix + bytes.size() + 16, std::byte{0x5a});
        std::ranges::copy(bytes, raw.begin()+prefix);
        const auto owner_desc = c::TensorDesc::dense(descriptions[port].data_type(),
            std::array{static_cast<std::int64_t>(raw.size()/4)}).value();
        auto owner = context.upload(owner_desc, raw).value();
        const auto slice = c::View::slice(0, static_cast<std::int64_t>(prefix/4),
            static_cast<std::int64_t>(bytes.size()/4)).value();
        const auto shape = c::TensorShape::create(descriptions[port].dimensions()).value();
        tensors[port] = owner.view(slice).value().view(c::View::reshape(shape).value()).value();
        if (guards) guards->push_back({std::move(owner),std::move(raw)});
      }
    }
    return tensors;
  }

  void expect(const v::Context &context, std::span<const v::Tensor> actual,
              const std::array<std::int32_t,3> &offsets, std::optional<float> scale = {}) const {
    std::vector<double> state(data[5].begin(),data[5].end()), output(data[2].size(),0.0);
    for (std::size_t batch=0; batch<batches; ++batch)
      for (auto row=std::size_t(offsets[batch]); row<std::size_t(offsets[batch+1]); ++row)
        for (std::size_t head=0; head<hv; ++head)
          for (std::size_t channel=0; channel<dv; ++channel) {
            const auto base=((batch*hv+head)*dv+channel)*dk;
            const auto qk=(row*hk+head%hk)*dk;
            const auto gate=(row*hv+head)*gate_width+(gate_width==1?0:channel);
            double prediction=0;
            for (std::size_t key=0; key<dk; ++key) {
              state[base+key] *= std::exp(double(data[3][gate]));
              prediction += state[base+key]*data[1][qk+key];
            }
            const double delta=data[4][row*hv+head]*(data[2][(row*hv+head)*dv+channel]-prediction);
            for (std::size_t key=0; key<dk; ++key) {
              state[base+key] += delta*data[1][qk+key];
              output[(row*hv+head)*dv+channel] += state[base+key]*data[0][qk+key];
            }
          }
    const double factor=scale ? *scale : 1.0/std::sqrt(double(dk));
    const auto y=vulkan_tensor_test::values(context,actual[0]);
    const auto s=vulkan_tensor_test::values(context,actual[1]);
    ASSERT_EQ(y.size(),output.size()); ASSERT_EQ(s.size(),state.size());
    for (std::size_t i=0;i<y.size();++i) EXPECT_NEAR(y[i],output[i]*factor,0.000003) << i;
    for (std::size_t i=0;i<s.size();++i) EXPECT_NEAR(s[i],state[i],0.000003) << i;
    const auto bits=vulkan_tensor_test::bits(context,actual[0]);
    for (std::size_t i=std::size_t(offsets.back())*hv*dv;i<bits.size();++i) EXPECT_EQ(bits[i],0u);
  }
};

TEST(VulkanSequence, PackedGdnRaggedMathAndChangedLengths) {
  auto context=vulkan_test::context().value();
  for (auto depth : {3U,128U,256U}) for (auto gates : {1U,2U}) {
    SCOPED_TRACE(::testing::Message()<<"depth="<<depth<<" gates="<<gates);
    Fixture fixture(depth,gates);
    auto program=fixture.prepare(context); ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value();
    for (const auto offsets : {std::array<std::int32_t,3>{0,1,4}, {0,0,3}}) {
      const auto inputs=fixture.upload(context,offsets);
      ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
      auto work=slot.submit().value(); auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();
      fixture.expect(context,result->get()->tensors(),offsets);
    }
  }
}

TEST(VulkanSequence, PackedGdnBorrowedInputsRetainEveryByteOrigin) {
  auto context=vulkan_test::context().value();
  Fixture fixture(3,2);
  constexpr std::array<std::int32_t,3> offsets{0,1,4};
  std::vector<std::pair<v::Tensor,std::vector<std::byte>>> guards;
  const auto inputs=fixture.upload(context,offsets,true,&guards);
  auto program=fixture.prepare(context,1,0.25F).value();
  auto slot=program.create_invocation().value();
  ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
  auto work=slot.submit().value(); auto result=work.wait();
  ASSERT_TRUE(result)<<result.error().message();
  fixture.expect(context,result->get()->tensors(),offsets,0.25F);
  for (const auto &[owner,raw] : guards) EXPECT_EQ(vulkan_tensor_test::bytes(context,owner),raw);
}

TEST(VulkanSequence, PackedGdnEmptyMembersCopyStateBitsAndZeroOnlyTheTail) {
  auto context=vulkan_test::context().value(); Fixture fixture(3,1);
  auto inputs=fixture.upload(context,{0,0,0});
  std::vector<std::uint32_t> bits(fixture.data[5].size());
  constexpr std::array<std::uint32_t,4> pattern{0x80000000u,0x7f812345u,0x7fc12345u,0xff800000u};
  for (std::size_t i=0;i<bits.size();++i) bits[i]=pattern[i%pattern.size()];
  inputs[5]=context.upload(fixture.descriptions[5],std::as_bytes(std::span(bits))).value();
  auto program=fixture.prepare(context).value(); auto slot=program.create_invocation().value();
  ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
  auto work=slot.submit().value(); auto result=work.wait(); ASSERT_TRUE(result)<<result.error().message();
  EXPECT_EQ(vulkan_tensor_test::bits(context,result->get()->tensors()[0]),std::vector<std::uint64_t>(fixture.data[2].size(),0));
  EXPECT_EQ(vulkan_tensor_test::bits(context,result->get()->tensors()[1]),std::vector<std::uint64_t>(bits.begin(),bits.end()));
}

TEST(VulkanSequence, PackedGdnSnapshotsValidateTheWholeInputBeforeEitherWrite) {
  auto context=vulkan_test::context().value();
  const auto f32=c::DataType::kFloat32;
  const auto desc=[](c::DataType type,std::initializer_list<std::int64_t> shape){return vulkan_tensor_test::dense(type,shape);};
  const std::array descriptions{desc(f32,{8,1,1}),desc(f32,{8,1,1}),desc(f32,{8,1,1}),
      desc(f32,{8,1,1}),desc(f32,{8,1,1}),desc(f32,{2,1,1,1}),desc(c::DataType::kInt32,{3})};
  const std::array data{std::vector<float>(8,1),std::vector<float>(8,1),
      std::vector<float>{2,4,6,8,10,NAN,NAN,NAN},std::vector<float>(8,0),
      std::vector<float>(8,0.5F),std::vector<float>{0,1}};
  auto program=v::prepare(context,c::sequence_ops::packed_gated_delta_net(2,1.0F),
      descriptions[0],descriptions[1],descriptions[2],descriptions[3],descriptions[4],descriptions[5],descriptions[6]).value();
  std::array<v::Tensor,7> inputs;
  for (std::size_t i=0;i<data.size();++i) inputs[i]=context.upload(descriptions[i],std::as_bytes(std::span(data[i]))).value();
  for (const auto offsets : {std::array<std::int32_t,3>{1,2,5}, {0,-1,4}, {0,4,3}, {0,2,9}, {0,1,4}, {0,2,5}}) {
    const bool valid=offsets==std::array<std::int32_t,3>{0,2,5};
    inputs[6]=context.upload(descriptions[6],std::as_bytes(std::span(offsets))).value();
    std::array<c::TensorDesc,2> inferred;
    ASSERT_TRUE(c::infer_output_descriptions(c::sequence_ops::packed_gated_delta_net(2,1.0F),descriptions,inferred).ok());
    std::array<v::Tensor,2> outputs;
    std::array<std::vector<std::byte>,2> poison;
    for (std::size_t i=0;i<outputs.size();++i) {
      poison[i].assign(*inferred[i].required_bytes(),std::byte{0x5a});
      outputs[i]=context.upload(inferred[i],poison[i]).value();
    }
    auto recorder=v::Recorder::create(context).value();
    ASSERT_TRUE(recorder.append(program,inputs,outputs).ok());
    auto recorded=recorder.finish().value(); auto completion=recorded.submit().value();
    const auto status=completion.wait();
    if (!valid) {
      EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
      for (std::size_t i=0;i<outputs.size();++i) EXPECT_EQ(vulkan_tensor_test::bytes(context,outputs[i]),poison[i]);
    } else {
      ASSERT_TRUE(status.ok())<<status.message();
      EXPECT_EQ(vulkan_tensor_test::values(context,outputs[0]),(std::vector<double>{1,2.5,3.5,5.75,7.875,0,0,0}));
      EXPECT_EQ(vulkan_tensor_test::values(context,outputs[1]),(std::vector<double>{1,2.5,5.75,7.875}));
    }
  }
}

TEST(VulkanSequence, PackedGdnDeclinesUnimplementedNumericalDomainsCold) {
  auto context=vulkan_test::context().value();
  Fixture fixture(257,1);
  auto result=fixture.prepare(context); EXPECT_FALSE(result);
  if (!result) EXPECT_TRUE(result.error().IsUnimplemented());
  for (auto type : {c::DataType::kFloat16,c::DataType::kBFloat16}) {
    Fixture low(3,1);
    for (std::size_t i=0;i<3;++i)
      low.descriptions[i]=c::TensorDesc::dense(type,low.descriptions[i].dimensions()).value();
    auto prepared=low.prepare(context); EXPECT_FALSE(prepared);
    if (!prepared) EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
}
} // namespace
