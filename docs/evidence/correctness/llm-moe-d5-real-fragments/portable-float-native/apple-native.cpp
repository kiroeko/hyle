#include <mlx/mlx.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
namespace mx = mlx::core;

float f8(unsigned code) {
  auto v=code&127u;
  float x=v<8 ? std::ldexp(float(v),-9) : std::ldexp(1.0f+float(v%8)/8,int(v/8)-7);
  return code&128 ? -x:x;
}
float bf(float x) {
  uint32_t b=std::bit_cast<uint32_t>(x); b+=0x7fff+((b>>16)&1);
  return std::bit_cast<float>(b&0xffff0000u);
}
unsigned nearest(float x, bool four=false) {
  const float f4[]{0,.5f,1,1.5f,2,3,4,6};
  unsigned best=0; double distance=INFINITY;
  for (unsigned i=0;i<(four?8u:127u);++i) {
    double d=std::abs(double(std::abs(x))-(four?f4[i]:f8(i)));
    if(d<distance||(d==distance&&(i&1)==0)) {best=i;distance=d;}
  }
  return best|(std::signbit(x)?(four?8u:128u):0u);
}
mx::array up_scale(mx::array a) {
  // Original DeepSeek fast_log2_ceil / fast_pow2 expressed with MLX ops.
  auto bits=mx::view(a,mx::uint32);
  auto exponent=mx::right_shift(bits,mx::array(uint32_t{23}));
  auto nonzero=mx::not_equal(mx::bitwise_and(bits,mx::array(uint32_t{0x7fffff})),mx::array(uint32_t{0}));
  return mx::view(mx::left_shift(mx::add(exponent,mx::astype(nonzero,mx::uint32)),mx::array(uint32_t{23})),mx::float32);
}
int main() {
 try {
  mx::set_default_device(mx::Device::gpu);
  unsigned failed=0;
  std::vector<uint8_t> codes;
  for(unsigned i=0;i<256;++i) if((i&127)!=127) codes.push_back(i);
  auto decoded=mx::from_fp8(mx::array(codes.data(),{int(codes.size())},mx::uint8),mx::float32);
  mx::eval(decoded); unsigned wrong=0;
  for(size_t i=0;i<codes.size();++i) if(std::bit_cast<uint32_t>(decoded.data<float>()[i])!=std::bit_cast<uint32_t>(f8(codes[i]))) ++wrong;
  std::cout << "fp8_decode samples="<<codes.size()<<" mismatches="<<wrong<<'\n'; failed+=wrong;
  std::vector<float> x;
  for(unsigned i=0;i<126;++i) {
    float a=f8(i),b=f8(i+1);
    for(float v:{a,(a+b)/2,std::nextafter((a+b)/2,a),std::nextafter((a+b)/2,b)}) {x.push_back(v);x.push_back(-v);}
  }
  for(float v:{448.0f,450.0f,500.0f,1e9f}) {x.push_back(v);x.push_back(-v);}
  auto encoded=mx::to_fp8(mx::array(x.data(),{int(x.size())},mx::float32)); mx::eval(encoded); wrong=0;
  for(size_t i=0;i<x.size();++i) if(encoded.data<uint8_t>()[i]!=nearest(x[i])) ++wrong;
  std::cout<<"fp8_encode samples="<<x.size()<<" mismatches="<<wrong<<'\n';failed+=wrong;
  // Keep the native MLX quantizer's small-scale boundary visible separately.
  for(bool tiny:{false,true}) {
    std::vector<float> values(32*8);
    for(size_t i=0;i<values.size();++i)
      values[i]=bf(i<32?0.0f:float(int(i%23)-11)*(tiny?std::ldexp(1.0f,-128):std::ldexp(1.0f,int(i/32)-5)));
    auto a=mx::astype(mx::array(values.data(),{8,32},mx::float32),mx::bfloat16);
    auto q=mx::quantize(a,32,4,"mxfp4");
    auto y=mx::astype(mx::dequantize(q[0],q[1],std::nullopt,32,4,"mxfp4",std::nullopt,mx::bfloat16),mx::float32);
    mx::eval(y);wrong=0;
    const float f4[]{0,.5f,1,1.5f,2,3,4,6};
    for(int g=0;g<8;++g) {
      float amax=6*std::ldexp(1.0f,-126);
      for(int j=0;j<32;++j) amax=std::max(amax,std::abs(values[g*32+j]));
      float scale=std::exp2(std::ceil(std::log2(amax*(1.0f/6))));
      for(int j=0;j<32;++j) {auto i=g*32+j;auto code=nearest(values[i]/scale,true);
        float expected=bf(std::copysign(f4[code&7],code&8?-1.0f:1.0f)*scale);
        if(y.data<float>()[i]!=expected) {if(wrong<3) std::cout<<"fp4 mismatch index="<<i<<" got="<<y.data<float>()[i]<<" expected="<<expected<<'\n';++wrong;}}
    }
    std::cout<<"fp4_roundtrip tiny="<<tiny<<" samples="<<values.size()<<" mismatches="<<wrong<<'\n';failed+=wrong;
  }
  for(int group:{64,128}) for(bool power:{false,true}) for(bool bfloat:{false,true}) {
    std::vector<float> values(3*1024),expected(values.size());
    for(size_t i=0;i<values.size();++i) {
      float x=(i<group?0.0f:std::sin(float(i)*.017f)*std::ldexp(1.0f,int(i/group)%9-5));
      values[i]=bfloat?bf(x):x;
    }
    auto a=mx::array(values.data(),{3,1024},mx::float32);
    if(bfloat) a=mx::astype(a,mx::bfloat16);
    auto grouped=mx::reshape(mx::astype(a,mx::float32),{3,1024/group,group});
    float floor=power?1e-4f:1e-10f;
    auto scale=mx::divide(mx::maximum(mx::max(mx::abs(grouped),-1,true),mx::array(floor)),mx::array(448.0f));
    if(power) scale=up_scale(scale);
    auto q=mx::to_fp8(mx::divide(grouped,scale));
    auto y=mx::multiply(mx::from_fp8(q,mx::float32),scale);
    if(bfloat) y=mx::astype(y,mx::bfloat16);
    auto result=mx::reshape(mx::astype(y,mx::float32),{int(values.size())});
    mx::eval(result,q,scale);wrong=0;unsigned codes_wrong=0;float max_scale_error=0;
    for(size_t start=0;start<values.size();start+=group) {
      float amax=floor; for(int j=0;j<group;++j) amax=std::max(amax,std::abs(values[start+j]));
      float s=amax/448.0f; if(power) s=std::exp2(std::ceil(std::log2(s)));
      max_scale_error=std::max(max_scale_error,std::abs(scale.data<float>()[start/group]/s-1));
      for(int j=0;j<group;++j) {size_t i=start+j;auto c=nearest(values[i]/s);
        if(q.data<uint8_t>()[i]!=c) ++codes_wrong;
        float expected=f8(c)*s;if(bfloat)expected=bf(expected);
        if(bfloat?result.data<float>()[i]!=expected:std::abs(result.data<float>()[i]-expected)>2e-6f*(1+std::abs(expected))) ++wrong;
      }
    }
    std::cout<<"fp8_groups group="<<group<<" power="<<power<<" bf16="<<bfloat
      <<" samples="<<values.size()<<" codes_mismatches="<<codes_wrong<<" output_mismatches="<<wrong
      <<" scale_relative_error="<<max_scale_error<<'\n';
    failed+=wrong+codes_wrong+(max_scale_error>2e-6f);
  }
  std::cout<<"total_mismatches="<<failed<<'\n';
  return failed?1:0;
 } catch(const std::exception& e) {std::cerr<<"native exception: "<<e.what()<<'\n';return 2;}
}
