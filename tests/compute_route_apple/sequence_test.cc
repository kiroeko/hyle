#include <gtest/gtest.h>
#include "tensor_test_support.h"
#include "hyle/compute/sequence_ops.h"

namespace {
using namespace apple_tensor_test;

struct Fixture {
  static constexpr int rows=8, batches=2, hk=2, hv=4, dv=2;
  int dk, gates;
  mx::Dtype type;
  std::array<mx::Shape, 7> shapes;
  std::array<std::vector<double>, 6> data;
  Fixture(int depth, int gate_width, mx::Dtype dtype=mx::float32)
      : dk(depth), gates(gate_width), type(dtype),
        shapes{mx::Shape{rows,hk,dk}, {rows,hk,dk}, {rows,hv,dv},
               {rows,hv,gates}, {rows,hv,1}, {batches,hv,dv,dk}, {batches+1}} {
    for (std::size_t port=0; port<data.size(); ++port) {
      std::size_t count=1;
      for (auto d:shapes[port]) count*=std::size_t(d);
      data[port].resize(count);
      for (std::size_t i=0;i<count;++i) {
        double value=double(int((i*7+port*11)%19)-9)/64;
        if (port==3) value=-.17+value/4;
        if (port==4) value=.4+value/8;
        data[port][i]=rounded(value,port<3?type:mx::float32);
      }
      if (port<5)
        for (std::size_t i=4*count/rows;i<count;++i)
          data[port][i]=std::numeric_limits<double>::quiet_NaN();
    }
  }
  std::vector<mx::array> inputs(std::array<std::int32_t,3> offsets,
      std::vector<mx::array> *owners=nullptr) const {
    std::vector<mx::array> result;
    for (std::size_t i=0;i<6;++i)
      result.push_back(make(data[i],shapes[i],i<3?type:mx::float32));
    result.push_back(mx::array(offsets.data(),{3},mx::int32));
    for (auto &value:result) {
      if (owners) {
        auto owner=ready(mx::concatenate({mx::zeros({3},value.dtype(),stream()),
            mx::reshape(value,{-1},stream()),mx::zeros({5},value.dtype(),stream())},0,stream()));
        owners->push_back(owner);
        value=mx::reshape(mx::slice(owner,{3},{3+int(value.size())},stream()),value.shape(),stream());
      }
      value=ready(value);
    }
    return result;
  }
  void expect(std::span<const mx::array> actual, std::array<std::int32_t,3> offsets,
      double scale, unsigned prefixes=0) const {
    auto state=data[5];
    std::vector<double> y(data[2].size(),0);
    std::vector<double> snapshots(prefixes*state.size(),0);
    for (int b=0;b<batches;++b) for (int row=offsets[b];row<offsets[b+1];++row)
      for (int h=0;h<hv;++h) for (int v=0;v<dv;++v) {
        const auto base=((b*hv+h)*dv+v)*dk, qk=(row*hk+h%hk)*dk;
        const auto gate=(row*hv+h)*gates+(gates==1?0:v);
        double prediction=0;
        for (int k=0;k<dk;++k) {
          state[base+k]*=std::exp(data[3][gate]);
          prediction+=state[base+k]*data[1][qk+k];
        }
        const double delta=data[4][row*hv+h]*(data[2][(row*hv+h)*dv+v]-prediction);
        for (int k=0;k<dk;++k) {
          state[base+k]+=delta*data[1][qk+k];
          y[(row*hv+h)*dv+v]+=state[base+k]*data[0][qk+k]*scale;
          if (prefixes)
            snapshots[(((b*prefixes+row-offsets[b])*hv+h)*dv+v)*dk+k]=state[base+k];
        }
      }
    apple_tensor_test::expect(actual[0],y);
    apple_tensor_test::expect(actual[1],prefixes?snapshots:state);
    if (prefixes) {
      const auto state_bits=bits(actual[1]);
      for (int b=0;b<batches;++b)
        for (unsigned row=unsigned(offsets[b+1]-offsets[b]);row<prefixes;++row)
          for (int lane=0;lane<hv*dv*dk;++lane)
            EXPECT_EQ(state_bits[(b*prefixes+row)*hv*dv*dk+lane],0u);
    }
    const auto raw=bits(actual[0]);
    for (std::size_t i=std::size_t(offsets.back())*hv*dv;i<raw.size();++i) EXPECT_EQ(raw[i],0u);
  }
};

TEST(AppleSequence, RaggedGdnUsesActualLengthsAndF32Accumulation) {
  auto context=a::Context::create().value();
  for (auto depth:{3,128,256}) for (auto gates:{1,2}) {
    Fixture fixture(depth,gates);
    auto inputs=fixture.inputs({0,1,4});
    auto program=context.prepare(c::sequence_ops::packed_gated_delta_net(),inputs);
    ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value();
    for (const auto offsets:{std::array<std::int32_t,3>{0,1,4},{0,0,3}}) {
      inputs=fixture.inputs(offsets);
      ASSERT_TRUE(slot.bind(inputs).ok());
      auto work=slot.submit().value(); auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();
      fixture.expect(result->get()->arrays(),offsets,1/std::sqrt(double(depth)));
    }
  }
  for (auto type:{mx::float16,mx::bfloat16}) {
    Fixture fixture(3,2,type);
    auto inputs=fixture.inputs({0,1,4});
    auto program=context.prepare(c::sequence_ops::packed_gated_delta_net(1,.25F),inputs);
    ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit().value(); auto result=work.wait(); ASSERT_TRUE(result)<<result.error().message();
    fixture.expect(result->get()->arrays(),{0,1,4},.25);
  }
}

TEST(AppleSequence, EveryBorrowedOriginAndRetainedResultSurvivesReuse) {
  auto context=a::Context::create().value(); Fixture fixture(3,2);
  std::vector<mx::array> owners;
  auto inputs=fixture.inputs({0,1,4},&owners);
  std::vector<std::vector<std::uint64_t>> before;
  for (const auto &owner:owners) before.push_back(bits(owner));
  auto prepared=context.prepare(c::sequence_ops::packed_gated_delta_net(1,.25F),inputs);
  ASSERT_TRUE(prepared)<<prepared.error().message();
  auto program=std::move(*prepared);
  auto invocation=program.create_invocation(); ASSERT_TRUE(invocation)<<invocation.error().message();
  auto slot=std::move(*invocation); ASSERT_TRUE(slot.bind(inputs).ok());
  std::vector<mx::array> retained;
  {
    auto submitted=slot.submit(); ASSERT_TRUE(submitted)<<submitted.error().message();
    auto result=submitted->wait(); ASSERT_TRUE(result)<<result.error().message();
    retained.assign(result->get()->arrays().begin(),result->get()->arrays().end());
  }
  for (std::size_t i=0;i<owners.size();++i) EXPECT_EQ(bits(owners[i]),before[i]);
  inputs=fixture.inputs({0,0,3}); ASSERT_TRUE(slot.bind(inputs).ok());
  {
    auto second=slot.submit(); ASSERT_TRUE(second)<<second.error().message();
    auto result=second->wait(); ASSERT_TRUE(result)<<result.error().message();
  }
  slot={}; program={}; ASSERT_TRUE(context.close().ok());
  fixture.expect(retained,{0,1,4},.25);
}

TEST(AppleSequence, EmptyMembersPreserveAllStateBits) {
  auto context=a::Context::create().value(); Fixture fixture(3,1);
  auto inputs=fixture.inputs({0,0,0});
  std::vector<std::uint32_t> raw(fixture.data[5].size());
  constexpr std::array<std::uint32_t,4> pattern{0x80000000u,0x7f812345u,0x7fc12345u,0xff800000u};
  for (std::size_t i=0;i<raw.size();++i) raw[i]=pattern[i%pattern.size()];
  inputs[5]=ready(mx::view(mx::array(raw.data(),fixture.shapes[5],mx::uint32),mx::float32,stream()));
  auto program=context.prepare(c::sequence_ops::packed_gated_delta_net(),inputs).value();
  auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
  auto work=slot.submit().value(); auto result=work.wait(); ASSERT_TRUE(result)<<result.error().message();
  EXPECT_EQ(bits(result->get()->arrays()[0]),std::vector<std::uint64_t>(fixture.data[2].size(),0));
  EXPECT_EQ(bits(result->get()->arrays()[1]),std::vector<std::uint64_t>(raw.begin(),raw.end()));
  // Zero physical row capacity still has a real, bit-preserving state output.
  for (std::size_t port=0;port<5;++port) {
    auto shape=fixture.shapes[port]; shape[0]=0;
    inputs[port]=ready(mx::zeros(shape,mx::float32,stream()));
  }
  auto empty=context.prepare(c::sequence_ops::packed_gated_delta_net(),inputs);
  ASSERT_TRUE(empty)<<empty.error().message();
  auto empty_slot=empty->create_invocation().value(); ASSERT_TRUE(empty_slot.bind(inputs).ok());
  auto pending=empty_slot.submit().value(); auto preserved=pending.wait();
  ASSERT_TRUE(preserved)<<preserved.error().message();
  EXPECT_EQ(preserved->get()->arrays()[0].size(),0);
  EXPECT_EQ(bits(preserved->get()->arrays()[1]),std::vector<std::uint64_t>(raw.begin(),raw.end()));
  inputs[6]=ready(mx::array({0,0,1}));
  auto invalid_slot=empty->create_invocation().value(); ASSERT_TRUE(invalid_slot.bind(inputs).ok());
  auto invalid_work=invalid_slot.submit().value(); auto invalid=invalid_work.wait();
  EXPECT_FALSE(invalid);
  if (!invalid) EXPECT_TRUE(invalid.error().IsInvalidArgument())<<invalid.error().message();
}

TEST(AppleSequence, SnapshotsRejectTheWholeInvalidOffsetDomain) {
  auto context=a::Context::create().value();
  std::vector inputs{mx::ones({8,1,1},mx::float32,stream()),mx::ones({8,1,1},mx::float32,stream()),
    mx::array({2.F,4.F,6.F,8.F,10.F,NAN,NAN,NAN},{8,1,1}),mx::zeros({8,1,1},mx::float32,stream()),
    mx::full({8,1,1},.5F,mx::float32,stream()),mx::array({0.F,1.F},{2,1,1,1}),mx::array({0,2,5})};
  for (auto &input:inputs) input=ready(input);
  auto program=context.prepare(c::sequence_ops::packed_gated_delta_net(2,1.F),inputs).value();
  for (const auto offsets:{std::array<std::int32_t,3>{1,2,5},{0,-1,4},{0,4,3},{0,2,9},{0,1,4},{0,2,5}}) {
    inputs[6]=ready(mx::array(offsets.data(),{3},mx::int32));
    auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    if (offsets!=std::array<std::int32_t,3>{0,2,5}) {
      EXPECT_FALSE(result);
      if (!result) EXPECT_TRUE(result.error().IsInvalidArgument())<<result.error().message();
    } else {
      ASSERT_TRUE(result)<<result.error().message();
      EXPECT_EQ(values(result->get()->arrays()[0]),(std::vector<double>{1,2.5,3.5,5.75,7.875,0,0,0}));
      EXPECT_EQ(values(result->get()->arrays()[1]),(std::vector<double>{1,2.5,5.75,7.875}));
    }
  }
}

TEST(AppleSequence, NativePrivateStateLimitIsRejectedCold) {
  auto context=a::Context::create().value(); Fixture fixture(257,1);
  auto result=context.prepare(c::sequence_ops::packed_gated_delta_net(),fixture.inputs({0,1,4}));
  EXPECT_FALSE(result);
  if (!result) EXPECT_TRUE(result.error().IsUnimplemented())<<result.error().message();
}

TEST(AppleSequence, PackedGdnPrefixesRetainEveryRealPrefixAndZeroUnusedSlots) {
  auto context=a::Context::create().value();
  for (auto type:{mx::float32,mx::float16,mx::bfloat16}) {
    Fixture fixture(3,2,type);
    std::vector<mx::array> owners;
    auto inputs=fixture.inputs({0,1,4},&owners);
    std::vector<std::vector<std::uint64_t>> before;
    for (const auto &owner:owners) before.push_back(bits(owner));
    auto program=context.prepare(c::sequence_ops::packed_gated_delta_net_prefixes(4,.25F),inputs);
    ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value();
    for (auto offsets:{std::array<std::int32_t,3>{0,1,4},{0,0,3},{0,2,4}}) {
      if (offsets!=std::array<std::int32_t,3>{0,1,4}) inputs=fixture.inputs(offsets);
      ASSERT_TRUE(slot.bind(inputs).ok());
      auto work=slot.submit().value(); auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();
      EXPECT_EQ(result->get()->arrays()[1].shape(),(mx::Shape{2,4,4,2,3}));
      fixture.expect(result->get()->arrays(),offsets,.25,4);
    }
    for (std::size_t i=0;i<owners.size();++i) EXPECT_EQ(bits(owners[i]),before[i]);
  }
}

TEST(AppleSequence, PackedGdnPrefixCapacityOneStaysRankFiveAndRejectsOverlongMembers) {
  auto context=a::Context::create().value(); Fixture fixture(3,1);
  auto inputs=fixture.inputs({0,1,2});
  auto program=context.prepare(c::sequence_ops::packed_gated_delta_net_prefixes(1,.25F),inputs).value();
  for (auto offsets:{std::array<std::int32_t,3>{1,1,2},{0,-1,2},{0,2,1},{0,1,9},{0,2,3},{0,1,2}}) {
    inputs[6]=ready(mx::array(offsets.data(),{3},mx::int32));
    auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    if (offsets!=std::array<std::int32_t,3>{0,1,2}) {
      EXPECT_FALSE(result);
      if (!result) EXPECT_TRUE(result.error().IsInvalidArgument())<<result.error().message();
    } else {
      ASSERT_TRUE(result)<<result.error().message();
      EXPECT_EQ(result->get()->arrays()[1].shape(),(mx::Shape{2,1,4,2,3}));
      fixture.expect(result->get()->arrays(),offsets,.25,1);
    }
  }
}

TEST(AppleSequence, PackedGdnEmptyPrefixesDoNotPublishInitialOrPaddingState) {
  auto context=a::Context::create().value(); Fixture fixture(3,1);
  auto inputs=fixture.inputs({0,0,0});
  inputs[5]=ready(mx::full(fixture.shapes[5],NAN,mx::float32,stream()));
  for (bool empty_rows:{false,true}) {
    if (empty_rows) for (std::size_t port=0;port<5;++port) {
      auto shape=fixture.shapes[port]; shape[0]=0;
      inputs[port]=ready(mx::zeros(shape,inputs[port].dtype(),stream()));
    }
    auto program=context.prepare(c::sequence_ops::packed_gated_delta_net_prefixes(3),inputs).value();
    auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    ASSERT_TRUE(result)<<result.error().message();
    EXPECT_EQ(bits(result->get()->arrays()[0]),
        std::vector<std::uint64_t>(empty_rows?0:fixture.data[2].size(),0));
    EXPECT_EQ(bits(result->get()->arrays()[1]),std::vector<std::uint64_t>(3*fixture.data[5].size(),0));
    if (empty_rows) {
      inputs[6]=ready(mx::array({0,0,1}));
      auto invalid_slot=program.create_invocation().value(); ASSERT_TRUE(invalid_slot.bind(inputs).ok());
      auto invalid_work=invalid_slot.submit().value(); auto invalid=invalid_work.wait();
      EXPECT_FALSE(invalid);
      if (!invalid) EXPECT_TRUE(invalid.error().IsInvalidArgument())<<invalid.error().message();
    }
  }
}

struct ConvFixture {
  static constexpr int batches=2;
  int rows, channels, taps;
  mx::Dtype type;
  std::array<mx::Shape,3> shapes;
  std::array<std::vector<double>,3> data;
  ConvFixture(mx::Dtype dtype, int kernel=4, int capacity=8, int width=3)
      : rows(capacity), channels(width), taps(kernel), type(dtype),
        shapes{mx::Shape{rows,channels},{channels,taps},{batches,taps-1,channels}} {
    for (std::size_t port=0;port<data.size();++port) {
      std::size_t count=1;
      for (int size:shapes[port]) count*=std::size_t(size);
      data[port].resize(count);
      for (std::size_t i=0;i<count;++i)
        data[port][i]=rounded(double(int((i*7+port*11)%19)-9)/64,type);
    }
    for (std::size_t i=4*channels;i<data[0].size();++i)
      data[0][i]=std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<mx::array> inputs(std::array<std::int32_t,3> offsets,
      std::vector<mx::array> *owners=nullptr) const {
    std::vector<mx::array> result;
    for (std::size_t i=0;i<data.size();++i) result.push_back(make(data[i],shapes[i],type));
    result.push_back(mx::array(offsets.data(),{3},mx::int32));
    for (auto &value:result) {
      if (owners) {
        auto owner=ready(mx::concatenate({mx::zeros({3},value.dtype(),stream()),
            mx::reshape(value,{-1},stream()),mx::zeros({5},value.dtype(),stream())},0,stream()));
        owners->push_back(owner);
        value=mx::reshape(mx::slice(owner,{3},{3+int(value.size())},stream()),value.shape(),stream());
      }
      value=ready(value);
    }
    return result;
  }
  void expect(std::span<const mx::array> actual, std::array<std::int32_t,3> offsets) const {
    std::vector<double> y(data[0].size(),0), history(data[2].size());
    for (int b=0;b<batches;++b) for (int channel=0;channel<channels;++channel) {
      // An independent concatenation oracle includes only this member's real rows.
      std::vector<float> sequence;
      for (int row=0;row<taps-1;++row)
        sequence.push_back(float(data[2][(b*(taps-1)+row)*channels+channel]));
      for (int row=offsets[b];row<offsets[b+1];++row)
        sequence.push_back(float(data[0][row*channels+channel]));
      for (int row=offsets[b];row<offsets[b+1];++row) {
        float sum=0;
        for (int tap=0;tap<taps;++tap)
          sum=std::fma(sequence[row-offsets[b]+tap],float(data[1][channel*taps+tap]),sum);
        y[row*channels+channel]=rounded(sum,type);
      }
      for (int row=0;row<taps-1;++row)
        history[(b*(taps-1)+row)*channels+channel]=sequence[sequence.size()-(taps-1)+row];
    }
    EXPECT_EQ(values(actual[0]),y);
    EXPECT_EQ(bits(actual[1]),bits(ready(make(history,shapes[2],type))));
    const auto raw=bits(actual[0]);
    for (std::size_t i=std::size_t(offsets.back())*channels;i<raw.size();++i) EXPECT_EQ(raw[i],0u);
  }
};

TEST(AppleSequence, PackedConvolutionRaggedLengthsTypesAndInvocationReuse) {
  auto context=a::Context::create().value();
  for (auto type:{mx::float32,mx::float16,mx::bfloat16}) for (int taps:{1,4}) {
    ConvFixture fixture(type,taps);
    auto inputs=fixture.inputs({0,1,4});
    auto program=context.prepare(c::sequence_ops::packed_causal_conv1d(),inputs);
    ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value();
    for (auto offsets:{std::array<std::int32_t,3>{0,1,4},{0,0,3},{0,4,4}}) {
      inputs=fixture.inputs(offsets); ASSERT_TRUE(slot.bind(inputs).ok());
      auto work=slot.submit().value(); auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();
      fixture.expect(result->get()->arrays(),offsets);
    }
  }
}

TEST(AppleSequence, PackedConvolutionBorrowedOriginsAndRetainedResults) {
  auto context=a::Context::create().value(); ConvFixture fixture(mx::float32,4,8,65);
  std::vector<mx::array> owners;
  auto inputs=fixture.inputs({0,1,4},&owners);
  std::vector<std::vector<std::uint64_t>> before;
  for (const auto &owner:owners) before.push_back(bits(owner));
  auto program=context.prepare(c::sequence_ops::packed_causal_conv1d(),inputs).value();
  auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
  std::vector<mx::array> retained;
  {
    auto work=slot.submit().value(); auto result=work.wait();
    ASSERT_TRUE(result)<<result.error().message();
    retained.assign(result->get()->arrays().begin(),result->get()->arrays().end());
  }
  for (std::size_t i=0;i<owners.size();++i) EXPECT_EQ(bits(owners[i]),before[i]);
  inputs=fixture.inputs({0,0,3}); ASSERT_TRUE(slot.bind(inputs).ok());
  {
    auto work=slot.submit().value(); ASSERT_TRUE(work.wait());
  }
  slot={}; program={}; ASSERT_TRUE(context.close().ok());
  fixture.expect(retained,{0,1,4});
}

TEST(AppleSequence, PackedConvolutionEmptyMembersPreserveRawHistoryBits) {
  auto context=a::Context::create().value();
  for (auto type:{mx::float32,mx::float16,mx::bfloat16}) for (int rows:{0,8}) {
    ConvFixture fixture(type,4,rows);
    auto inputs=fixture.inputs({0,0,0});
    constexpr std::array<std::uint32_t,4> pattern32{0x80000000u,0x7f812345u,0x7fc12345u,0xff800000u};
    constexpr std::array<std::uint16_t,4> pattern16{0x8000u,0x7f81u,0x7e45u,0xfc00u};
    std::vector<std::uint64_t> expected(fixture.data[2].size());
    if (type==mx::float32) {
      std::vector<std::uint32_t> raw(expected.size());
      for (std::size_t i=0;i<raw.size();++i) expected[i]=raw[i]=pattern32[i%pattern32.size()];
      inputs[2]=ready(mx::view(mx::array(raw.data(),fixture.shapes[2],mx::uint32),type,stream()));
    } else {
      std::vector<std::uint16_t> raw(expected.size());
      for (std::size_t i=0;i<raw.size();++i) expected[i]=raw[i]=pattern16[i%pattern16.size()];
      inputs[2]=ready(mx::view(mx::array(raw.data(),fixture.shapes[2],mx::uint16),type,stream()));
    }
    auto program=context.prepare(c::sequence_ops::packed_causal_conv1d(),inputs).value();
    auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    ASSERT_TRUE(result)<<result.error().message();
    EXPECT_EQ(bits(result->get()->arrays()[0]),std::vector<std::uint64_t>(fixture.data[0].size(),0));
    EXPECT_EQ(bits(result->get()->arrays()[1]),expected);
  }
  ConvFixture fixture(mx::float32,1,0);
  auto inputs=fixture.inputs({0,0,0});
  auto program=context.prepare(c::sequence_ops::packed_causal_conv1d(),inputs).value();
  auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
  auto work=slot.submit().value(); auto result=work.wait();
  ASSERT_TRUE(result)<<result.error().message();
  for (const auto &output:result->get()->arrays()) EXPECT_EQ(output.size(),0);
}

TEST(AppleSequence, PackedConvolutionRejectsWholeInvalidOffsetDomainAndRecovers) {
  auto context=a::Context::create().value(); ConvFixture fixture(mx::float32);
  auto inputs=fixture.inputs({0,1,4});
  auto program=context.prepare(c::sequence_ops::packed_causal_conv1d(),inputs).value();
  for (auto offsets:{std::array<std::int32_t,3>{1,2,4},{0,-1,4},{0,4,3},{0,2,9},{0,1,4}}) {
    inputs[3]=ready(mx::array(offsets.data(),{3},mx::int32));
    auto slot=program.create_invocation().value(); ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit().value(); auto result=work.wait();
    if (offsets!=std::array<std::int32_t,3>{0,1,4}) {
      EXPECT_FALSE(result);
      if (!result) EXPECT_TRUE(result.error().IsInvalidArgument())<<result.error().message();
    } else {
      ASSERT_TRUE(result)<<result.error().message(); fixture.expect(result->get()->arrays(),offsets);
    }
  }
}
} // namespace
