#include "tensor_test_support.h"
#include "hyle/compute/sequence_ops.h"

#include <algorithm>
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
                                  std::optional<float> scale = {}, bool prefixes = false) const {
    return v::prepare(context, prefixes
        ? c::sequence_ops::packed_gated_delta_net_prefixes(snapshots, scale)
        : c::sequence_ops::packed_gated_delta_net(snapshots, scale),
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
              const std::array<std::int32_t,3> &offsets, std::optional<float> scale = {},
              std::uint32_t prefix_capacity = 0) const {
    std::vector<double> state(data[5].begin(),data[5].end()), output(data[2].size(),0.0);
    std::vector<double> prefixes(prefix_capacity * state.size(), 0.0);
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
              if (prefix_capacity) {
                const auto prefix=((batch*prefix_capacity+row-offsets[batch])*hv+head)*dv+channel;
                prefixes[prefix*dk+key]=state[base+key];
              }
            }
          }
    const double factor=scale ? *scale : 1.0/std::sqrt(double(dk));
    const auto y=vulkan_tensor_test::values(context,actual[0]);
    const auto s=vulkan_tensor_test::values(context,actual[1]);
    const auto &expected_state=prefix_capacity?prefixes:state;
    ASSERT_EQ(y.size(),output.size()); ASSERT_EQ(s.size(),expected_state.size());
    for (std::size_t i=0;i<y.size();++i) EXPECT_NEAR(y[i],output[i]*factor,0.000003) << i;
    for (std::size_t i=0;i<s.size();++i) EXPECT_NEAR(s[i],expected_state[i],0.000003) << i;
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

TEST(VulkanSequence, PackedGdnPrefixesKeepRealPrefixesAndBorrowedOrigins) {
  auto context=vulkan_test::context().value();
  for (auto depth:{3U,128U}) {
    Fixture fixture(depth,2);
    auto program=fixture.prepare(context,4,.25F,true).value();
    auto slot=program.create_invocation().value();
    for (const auto offsets:{std::array<std::int32_t,3>{0,1,4},{0,0,3},{0,2,4}}) {
      std::vector<std::pair<v::Tensor,std::vector<std::byte>>> guards;
      const auto inputs=fixture.upload(context,offsets,true,&guards);
      ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
      auto work=slot.submit().value(); auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();
      const auto actual=result->get()->tensors();
      EXPECT_EQ(actual[1].description(),vulkan_tensor_test::dense(c::DataType::kFloat32,
          {2,4,4,2,std::int64_t(depth)}));
      fixture.expect(context,actual,offsets,.25F,4);
      const auto states=vulkan_tensor_test::bits(context,actual[1]);
      const auto width=Fixture::hv*Fixture::dv*depth;
      for (std::size_t b=0;b<Fixture::batches;++b)
        for (auto prefix=offsets[b+1]-offsets[b];prefix<4;++prefix)
          for (std::size_t lane=0;lane<width;++lane)
            EXPECT_EQ(states[(b*4+prefix)*width+lane],0u);
      for (const auto &[owner,raw]:guards) EXPECT_EQ(vulkan_tensor_test::bytes(context,owner),raw);
    }
  }
}

TEST(VulkanSequence, PackedGdnPrefixCapacityOneValidatesBeforeEitherWrite) {
  auto context=vulkan_test::context().value(); Fixture fixture(3,1);
  const auto operation=c::sequence_ops::packed_gated_delta_net_prefixes(1,.25F);
  auto program=fixture.prepare(context,1,.25F,true).value();
  std::array<c::TensorDesc,2> inferred;
  ASSERT_TRUE(c::infer_output_descriptions(operation,fixture.descriptions,inferred).ok());
  EXPECT_EQ(inferred[1],vulkan_tensor_test::dense(c::DataType::kFloat32,{2,1,4,2,3}));
  for (const auto offsets:{std::array<std::int32_t,3>{1,1,2},{0,-1,2},{0,2,1},
                          {0,1,9},{0,2,3},{0,1,2}}) {
    const auto inputs=fixture.upload(context,offsets);
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
    if (offsets!=std::array<std::int32_t,3>{0,1,2}) {
      EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
      for (std::size_t i=0;i<outputs.size();++i)
        EXPECT_EQ(vulkan_tensor_test::bytes(context,outputs[i]),poison[i]);
    } else {
      ASSERT_TRUE(status.ok())<<status.message();
      fixture.expect(context,outputs,offsets,.25F,1);
    }
  }
}

TEST(VulkanSequence, PackedGdnEmptyPrefixesArePositiveZeroWithoutPublishingInitialState) {
  auto context=vulkan_test::context().value();
  for (bool empty_rows:{false,true}) {
    Fixture fixture(3,1);
    std::ranges::fill(fixture.data[5],std::numeric_limits<float>::quiet_NaN());
    if (empty_rows) for (std::size_t i=0;i<5;++i) {
      auto shape=std::vector<std::int64_t>(fixture.descriptions[i].dimensions().begin(),
          fixture.descriptions[i].dimensions().end());
      shape[0]=0;
      fixture.descriptions[i]=c::TensorDesc::dense(c::DataType::kFloat32,shape).value();
      fixture.data[i].clear();
    }
    const auto inputs=fixture.upload(context,{0,0,0});
    auto program=fixture.prepare(context,3,{},true).value();
    auto slot=program.create_invocation().value();
    ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    ASSERT_TRUE(result)<<result.error().message();
    EXPECT_EQ(vulkan_tensor_test::bits(context,result->get()->tensors()[0]),
        std::vector<std::uint64_t>(fixture.data[2].size(),0));
    EXPECT_EQ(vulkan_tensor_test::bits(context,result->get()->tensors()[1]),
        std::vector<std::uint64_t>(3*fixture.data[5].size(),0));
  }
}

struct ConvFixture final {
  static constexpr int batches=2;
  int rows,channels,taps;
  c::DataType type;
  std::array<c::TensorDesc,4> descriptions;
  std::array<std::vector<double>,3> data;
  ConvFixture(c::DataType dtype,int kernel=4,int capacity=8,int width=3)
      : rows(capacity),channels(width),taps(kernel),type(dtype),descriptions{
          vulkan_tensor_test::dense(type,{rows,channels}),
          vulkan_tensor_test::dense(type,{channels,taps}),
          vulkan_tensor_test::dense(type,{batches,taps-1,channels}),
          vulkan_tensor_test::dense(c::DataType::kInt32,{batches+1})} {
    for (std::size_t port=0;port<data.size();++port) {
      data[port].resize(*descriptions[port].element_count());
      for (std::size_t i=0;i<data[port].size();++i)
        data[port][i]=vulkan_tensor_test::rounded(double(int((i*7+port*11)%19)-9)/64,type);
    }
    for (std::size_t i=4*channels;i<data[0].size();++i)
      data[0][i]=std::numeric_limits<double>::quiet_NaN();
  }

  v::Program prepare(const v::Context &context) const {
    return v::prepare(context,c::sequence_ops::packed_causal_conv1d(),descriptions[0],
        descriptions[1],descriptions[2],descriptions[3]).value();
  }

  std::array<v::Tensor,4> upload(const v::Context &context,
      std::array<std::int32_t,3> offsets,bool borrowed=false,
      std::vector<std::pair<v::Tensor,std::vector<std::byte>>> *guards=nullptr) const {
    std::array<v::Tensor,4> inputs;
    for (std::size_t port=0;port<inputs.size();++port) {
      std::vector<std::byte> bytes;
      if (port<3) {
        std::vector<std::uint64_t> bits;
        for (double value:data[port]) bits.push_back(vulkan_tensor_test::encode(value,type));
        bytes=vulkan_tensor_test::payload(descriptions[port],bits);
      } else {
        const auto raw=std::as_bytes(std::span(offsets));
        bytes.assign(raw.begin(),raw.end());
      }
      if (!borrowed) inputs[port]=context.upload(descriptions[port],bytes).value();
      else {
        const auto lane=c::data_type_size(descriptions[port].data_type());
        const auto prefix=(2*port+1)*lane;
        std::vector<std::byte> raw(prefix+bytes.size()+16,std::byte{0x5a});
        std::ranges::copy(bytes,raw.begin()+prefix);
        auto owner=context.upload(c::TensorDesc::dense(descriptions[port].data_type(),
            std::array{std::int64_t(raw.size()/lane)}).value(),raw).value();
        inputs[port]=owner.view(c::View::slice(0,std::int64_t(prefix/lane),
            std::int64_t(bytes.size()/lane)).value()).value().view(c::View::reshape(
                c::TensorShape::create(descriptions[port].dimensions()).value()).value()).value();
        if (guards) guards->push_back({std::move(owner),std::move(raw)});
      }
    }
    return inputs;
  }

  void expect(const v::Context &context,std::span<const v::Tensor> actual,
              std::array<std::int32_t,3> offsets) const {
    std::vector<double> y(data[0].size(),0),history(data[2].size());
    for (int batch=0;batch<batches;++batch) for (int channel=0;channel<channels;++channel) {
      // Form only this member's real history and rows, independently of shader addresses.
      std::vector<float> sequence;
      for (int row=0;row<taps-1;++row)
        sequence.push_back(float(data[2][(batch*(taps-1)+row)*channels+channel]));
      for (int row=offsets[batch];row<offsets[batch+1];++row)
        sequence.push_back(float(data[0][row*channels+channel]));
      for (int row=offsets[batch];row<offsets[batch+1];++row) {
        float sum=0;
        for (int tap=0;tap<taps;++tap)
          sum=std::fma(sequence[row-offsets[batch]+tap],float(data[1][channel*taps+tap]),sum);
        y[row*channels+channel]=vulkan_tensor_test::rounded(sum,type);
      }
      for (int row=0;row<taps-1;++row)
        history[(batch*(taps-1)+row)*channels+channel]=sequence[sequence.size()-(taps-1)+row];
    }
    EXPECT_EQ(vulkan_tensor_test::values(context,actual[0]),y);
    std::vector<std::uint64_t> expected;
    for (double value:history) expected.push_back(vulkan_tensor_test::encode(value,type));
    EXPECT_EQ(vulkan_tensor_test::bits(context,actual[1]),expected);
    const auto raw=vulkan_tensor_test::bits(context,actual[0]);
    for (std::size_t i=std::size_t(offsets.back())*channels;i<raw.size();++i) EXPECT_EQ(raw[i],0u);
  }
};

TEST(VulkanSequence, PackedConvolutionRaggedTypesAndChangedLengths) {
  auto context=vulkan_test::context().value();
  for (auto type:{c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16})
    for (int taps:{1,4}) {
      SCOPED_TRACE(::testing::Message()<<"type="<<int(type)<<" taps="<<taps);
      ConvFixture fixture(type,taps);
      auto program=fixture.prepare(context); auto slot=program.create_invocation().value();
      for (const auto offsets:{std::array<std::int32_t,3>{0,1,4},{0,0,3},{0,4,4}}) {
        const auto inputs=fixture.upload(context,offsets);
        ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
        auto work=slot.submit().value(); auto result=work.wait();
        ASSERT_TRUE(result)<<result.error().message();
        fixture.expect(context,result->get()->tensors(),offsets);
      }
    }
}

TEST(VulkanSequence, PackedConvolutionBorrowedHalfwordOriginsAndRetainedResults) {
  auto context=vulkan_test::context().value();
  for (auto type:{c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16}) {
    ConvFixture fixture(type,4,8,65);
    std::vector<std::pair<v::Tensor,std::vector<std::byte>>> guards;
    const auto inputs=fixture.upload(context,{0,1,4},true,&guards);
    auto program=fixture.prepare(context); auto slot=program.create_invocation().value();
    ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
    std::vector<v::Tensor> retained;
    {
      auto work=slot.submit().value(); auto result=work.wait(); ASSERT_TRUE(result)<<result.error().message();
      retained.assign(result->get()->tensors().begin(),result->get()->tensors().end());
    }
    const auto changed=fixture.upload(context,{0,0,3});
    ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(changed)).ok());
    // A retained result pins this slot's storage. Another invocation may
    // progress, but the pinned slot must refuse to overwrite the old bytes.
    auto blocked=slot.submit();
    ASSERT_FALSE(blocked);
    EXPECT_TRUE(blocked.error().IsFailedPrecondition())<<blocked.error().message();
    auto other=program.create_invocation().value();
    ASSERT_TRUE(other.bind(std::span<const v::Tensor>(changed)).ok());
    {
      auto work=other.submit().value(); auto result=work.wait(); ASSERT_TRUE(result);
      fixture.expect(context,result->get()->tensors(),{0,0,3});
    }
    slot={}; program={};
    fixture.expect(context,retained,{0,1,4});
    for (const auto &[owner,raw]:guards) EXPECT_EQ(vulkan_tensor_test::bytes(context,owner),raw);
  }
}

TEST(VulkanSequence, PackedConvolutionEmptyMembersAndPortsPreserveRawHistoryBits) {
  auto context=vulkan_test::context().value();
  constexpr std::array<std::uint32_t,4> pattern32{0x80000000u,0x7f812345u,0x7fc12345u,0xff800000u};
  constexpr std::array<std::uint16_t,4> pattern16{0x8000u,0x7f81u,0x7e45u,0xfc00u};
  for (auto type:{c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16})
    for (int rows:{0,8}) for (int taps:{1,4}) {
      ConvFixture fixture(type,taps,rows);
      auto inputs=fixture.upload(context,{0,0,0});
      std::vector<std::uint64_t> expected(fixture.data[2].size());
      for (std::size_t i=0;i<expected.size();++i)
        expected[i]=type==c::DataType::kFloat32?pattern32[i%4]:pattern16[i%4];
      inputs[2]=context.upload(fixture.descriptions[2],
          vulkan_tensor_test::payload(fixture.descriptions[2],expected)).value();
      auto program=fixture.prepare(context); auto slot=program.create_invocation().value();
      ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
      auto work=slot.submit().value(); auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();
      EXPECT_EQ(vulkan_tensor_test::bits(context,result->get()->tensors()[0]),
          std::vector<std::uint64_t>(fixture.data[0].size(),0));
      EXPECT_EQ(vulkan_tensor_test::bits(context,result->get()->tensors()[1]),expected);
    }
}

TEST(VulkanSequence, PackedConvolutionOddHalfwordExtent) {
  auto context=vulkan_test::context().value();
  for (auto type:{c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16}) {
    // Nine output lanes put the final halfword at an odd physical extent.
    ConvFixture fixture(type,2,3,3);
    std::vector<std::pair<v::Tensor,std::vector<std::byte>>> guards;
    const auto inputs=fixture.upload(context,{0,1,3},true,&guards);
    auto program=fixture.prepare(context);
    auto slot=program.create_invocation().value();
    ASSERT_TRUE(slot.bind(std::span<const v::Tensor>(inputs)).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    ASSERT_TRUE(result)<<result.error().message();
    fixture.expect(context,result->get()->tensors(),{0,1,3});
    for (const auto &[owner,raw]:guards) EXPECT_EQ(vulkan_tensor_test::bytes(context,owner),raw);
  }
}

TEST(VulkanSequence, PackedConvolutionInvalidOffsetsWriteNeitherOutputAndRecover) {
  auto context=vulkan_test::context().value();
  for (auto type:{c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16}) {
    ConvFixture fixture(type); auto program=fixture.prepare(context);
    for (const auto offsets:{std::array<std::int32_t,3>{1,2,4},{0,-1,4},{0,4,3},{0,2,9},{0,1,4}}) {
      const auto inputs=fixture.upload(context,offsets);
      std::array<v::Tensor,2> outputs;
      std::array<std::vector<std::byte>,2> poison;
      for (std::size_t i=0;i<outputs.size();++i) {
        const auto &desc=fixture.descriptions[i==0?0:2];
        poison[i].assign(*desc.required_bytes(),std::byte{0x5a});
        outputs[i]=context.upload(desc,poison[i]).value();
      }
      auto recorder=v::Recorder::create(context).value();
      ASSERT_TRUE(recorder.append(program,inputs,outputs).ok());
      auto recorded=recorder.finish().value(); auto completion=recorded.submit().value();
      const auto status=completion.wait();
      if (offsets!=std::array<std::int32_t,3>{0,1,4}) {
        EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
        for (std::size_t i=0;i<outputs.size();++i)
          EXPECT_EQ(vulkan_tensor_test::bytes(context,outputs[i]),poison[i]);
      } else {
        ASSERT_TRUE(status.ok())<<status.message(); fixture.expect(context,outputs,offsets);
      }
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
