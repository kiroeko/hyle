#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "tensor_test_support.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace {
using namespace apple_tensor_test;

float fp8(std::uint8_t code) {
  const unsigned exponent=(code>>3)&15, fraction=code&7;
  if(exponent==15&&fraction==7)return std::numeric_limits<float>::quiet_NaN();
  const float value=exponent?std::ldexp(1.F+float(fraction)/8,int(exponent)-7)
                            :std::ldexp(float(fraction),-9);
  return code&128?-value:value;
}
mx::array carrier(const std::vector<std::uint8_t>& codes,const std::vector<float>& scales) {
  std::vector<std::uint8_t> storage(codes.size()+scales.size()*sizeof(float));
  std::memcpy(storage.data(),codes.data(),codes.size());
  std::memcpy(storage.data()+codes.size(),scales.data(),scales.size()*sizeof(float));
  return ready(mx::array(storage.data(),{int(storage.size())},mx::uint8));
}
a::Program program(const a::Context& context,int m,int n,int k) {
  const std::array descriptions{
    c::float8_activation_description(m,k,c::Float8ScaleMode::kFloat32Block128).value(),
    c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128).value()};
  const std::array specs{a::input_spec(descriptions[0]).value(),a::input_spec(descriptions[1]).value()};
  return context.prepare(c::MatMul{},specs).value();
}

TEST(ApplePackedFp8, EdgeTiles64RebindingsPreserveBlockScaleOrderAndPriorResults) {
  constexpr int m=17,n=19,k=256,blocks=k/128;
  auto context=a::Context::create().value();
  auto prepared=program(context,m,n,k);
  auto first=prepared.create_invocation().value(),second=prepared.create_invocation().value();
  std::vector<std::uint8_t> qa(m*k),qw(n*k);
  std::vector<float> qs(m*blocks),ws(blocks);
  std::vector<std::uint64_t> original;
  std::vector<mx::array> retained;
  for(int step=0;step<64;++step) {
    SCOPED_TRACE(step);
    for(int i=0;i<m*k;++i)qa[i]=std::uint8_t(0x28+(i+step)%24)|((i%3==0)?128:0);
    for(int i=0;i<n*k;++i)qw[i]=std::uint8_t(0x30+(i*3+step)%24)|((i%5==0)?128:0);
    for(int i=0;i<m*blocks;++i)qs[i]=.07F+float((i+step)%7)*.013F;
    ws={.103F+float(step%3)*.017F,.271F};
    const std::array inputs{carrier(qa,qs),carrier(qw,ws)};
    auto& slot=step%2?second:first;
    ASSERT_TRUE(slot.bind(inputs).ok());
    auto work=slot.submit();ASSERT_TRUE(work)<<work.error().message();
    auto result=work->wait();ASSERT_TRUE(result)<<result.error().message();
    auto actual=result->get()->arrays()[0];
    ASSERT_EQ(actual.dtype(),mx::float32);
    const auto observed=values(actual);
    ASSERT_EQ(observed.size(),std::size_t(m*n));
    for(int row=0;row<m;++row)for(int column=0;column<n;++column) {
      double expected=0,magnitude=0;
      for(int block=0;block<blocks;++block) {
        double dot=0,absolute=0;
        for(int j=0;j<128;++j) {
          const double term=double(fp8(qa[row*k+block*128+j]))*fp8(qw[column*k+block*128+j]);
          dot+=term;absolute+=std::abs(term);
        }
        float scaled=float(dot)*qs[row*blocks+block];
        scaled=scaled*ws[block];
        expected=float(float(expected)+scaled);
        magnitude+=absolute*std::abs(qs[row*blocks+block]*ws[block]);
      }
      EXPECT_NEAR(observed[row*n+column],expected,
          16*std::numeric_limits<float>::epsilon()*magnitude) << row << ',' << column;
    }
    if(step==0) {retained.push_back(actual);original=bits(actual);}
  }
  EXPECT_EQ(bits(retained[0]),original);
}

TEST(ApplePackedFp8, AllCodesPreserveFiniteValuesAndReservedNan) {
  constexpr int n=256,k=128;
  auto context=a::Context::create().value();
  auto prepared=program(context,1,n,k);
  auto slot=prepared.create_invocation().value();
  std::vector<std::uint8_t> qa(k),qw(n*k);
  qa[0]=0x38;
  for(int code=0;code<256;++code)qw[code*k]=std::uint8_t(code);
  const std::array inputs{carrier(qa,{1.F}),carrier(qw,{1.F,1.F})};
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto result=slot.submit().value().wait();ASSERT_TRUE(result)<<result.error().message();
  const auto actual=values(result->get()->arrays()[0]);
  ASSERT_EQ(actual.size(),256u);
  for(int code=0;code<256;++code) {
    SCOPED_TRACE(code);const auto expected=fp8(std::uint8_t(code));
    if(std::isnan(expected))EXPECT_TRUE(std::isnan(actual[code]));
    else EXPECT_EQ(actual[code],expected);
  }
}

TEST(ApplePackedFp8, EncodedDotAndExplicitDecodeKeepMacroblockScaleAxes) {
  constexpr int n=129,k=256,blocks=k/128;
  auto context=a::Context::create().value();
  std::vector<std::uint8_t> qa(k),qw(n*k);
  qa[0]=0x38;qa[128]=0x30; // Exactly1 and0.5 in separate K groups.
  for(int column=0;column<n;++column) {
    qw[column*k]=0x38;
    qw[column*k+128]=column%2?0xb8:0x38;
  }
  const std::array<float,4> scales{1.F,2.F,4.F,8.F};
  const auto bank=carrier(qw,std::vector<float>(scales.begin(),scales.end()));
  auto prepared=program(context,1,n,k);
  auto slot=prepared.create_invocation().value();
  const std::array inputs{carrier(qa,{1.F,1.F}),bank};
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto work=slot.submit();ASSERT_TRUE(work)<<work.error().message();
  auto result=work->wait();ASSERT_TRUE(result)<<result.error().message();
  const auto observed=values(result->get()->arrays()[0]);
  ASSERT_EQ(observed.size(),std::size_t(n));
  for(int column=0;column<n;++column) {
    const auto base=(column/128)*blocks;
    EXPECT_EQ(observed[column],scales[base]+(column%2?-.5F:.5F)*scales[base+1]);
  }

  // Explicit full-weight decoding still expands each shared scale over its
  //128 rows. The packed-dot compact layout must not change this public op.
  const auto desc=c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128).value();
  const std::array specs{a::input_spec(desc).value()};
  auto decoder=context.prepare(c::model_ops::float8_weight_decode(c::DataType::kFloat32),specs).value();
  auto decode_slot=decoder.create_invocation().value();
  const std::array decode_inputs{bank};
  ASSERT_TRUE(decode_slot.bind(decode_inputs).ok());
  auto decoded_work=decode_slot.submit();ASSERT_TRUE(decoded_work)<<decoded_work.error().message();
  auto decoded=decoded_work->wait();ASSERT_TRUE(decoded)<<decoded.error().message();
  const auto weights=values(decoded->get()->arrays()[0]);
  ASSERT_EQ(weights.size(),std::size_t(n*k));
  for(int column=0;column<n;++column)for(int depth=0;depth<k;++depth) {
    const auto expected=fp8(qw[column*k+depth])*scales[(column/128)*blocks+depth/128];
    EXPECT_EQ(weights[column*k+depth],expected)<<column<<','<<depth;
  }
  ::testing::Test::RecordProperty("successful_submissions",2);
}
} // namespace
