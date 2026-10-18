#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
namespace mx=mlx::core;
constexpr std::array<float,8> kValues{0,.5f,1,1.5f,2,3,4,6};
float decode(unsigned c){return std::copysign(kValues[c&7],(c&8)?-1.f:1.f);}
uint32_t bits(float x){return std::bit_cast<uint32_t>(x);}
float bf(float x){auto b=bits(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);}
unsigned nearest(float x){unsigned best=0;double delta=INFINITY;for(unsigned c=0;c<8;++c){double d=std::abs(double(std::abs(x))-kValues[c]);if(d<delta||(d==delta&&(c&1)==0)){best=c;delta=d;}}return best|(std::signbit(x)?8:0);}
struct Oracle{std::vector<uint8_t> codes,scales;std::vector<float> output;};
Oracle reference(const std::vector<uint16_t>& input){
  Oracle out;out.codes.resize(input.size());out.scales.resize(input.size()/32);out.output.resize(input.size());
  for(size_t base=0;base<input.size();base+=32){
    float maximum=6*std::ldexp(1.f,-126);
    for(size_t i=0;i<32;++i)maximum=std::max(maximum,std::abs(std::bit_cast<float>(uint32_t(input[base+i])<<16)));
    float scaled=maximum*(1.f/6.f);
    float scale=std::exp2(std::ceil(std::log2(double(scaled))));
    out.scales[base/32]=uint8_t((bits(scale)>>23)&255);
    for(size_t i=0;i<32;++i){auto x=std::bit_cast<float>(uint32_t(input[base+i])<<16);auto c=nearest(x/scale);out.codes[base+i]=c;out.output[base+i]=bf(decode(c)*scale);}
  }return out;
}
std::vector<mx::array> quantize(const std::vector<mx::array>& in){
  auto q=mx::quantize(in[0],32,4,"mxfp4");
  auto y=mx::astype(mx::dequantize(q[0],q[1],std::nullopt,32,4,"mxfp4",std::nullopt,mx::bfloat16),mx::float32);
  return {q[0],q[1],y};
}
int main(){
  try{
    mx::set_default_device(mx::Device::gpu);
    unsigned failures=0;
    // Force every E2M1 code under each finite nonzero E8M0 exponent. There
    // is no quantizer here, so code order and decode arithmetic are isolated.
    for(bool bf16:{false,true}){
      std::vector<uint32_t> packed(254*4);std::vector<uint8_t> scales(254);
      for(unsigned r=0;r<254;++r){scales[r]=uint8_t(r+1);for(unsigned j=0;j<4;++j){uint32_t word=0;for(unsigned k=0;k<8;++k)word|=((j*8+k)%16)<<(4*k);packed[r*4+j]=word;}}
      auto y=mx::dequantize(mx::array(packed.data(),{254,4},mx::uint32),mx::array(scales.data(),{254,1},mx::uint8),std::nullopt,32,4,"mxfp4",std::nullopt,bf16?mx::bfloat16:mx::float32);
      y=mx::astype(y,mx::float32);mx::eval(y);
      unsigned wrong=0,unit_wrong=0,ordinary_wrong=0;
      for(unsigned r=0;r<254;++r)for(unsigned j=0;j<32;++j){float expected=decode(j%16)*std::ldexp(1.f,int(r+1)-127);if(bf16)expected=bf(expected);float got=y.data<float>()[r*32+j];if(bits(expected)!=bits(got)){if(wrong<4)std::cout<<"decode mismatch bf16="<<bf16<<" exponent="<<r+1<<" code="<<j%16<<" got="<<got<<" expected="<<expected<<'\n';++wrong;if(r+1==127)++unit_wrong;if(r+1>=16&&r+1<=238)++ordinary_wrong;}}
      std::cout<<"decode bf16="<<bf16<<" samples="<<packed.size()*8<<" mismatches="<<wrong<<" unit_scale_mismatches="<<unit_wrong<<" exponents16_238_mismatches="<<ordinary_wrong<<'\n';failures+=wrong;
    }
    auto compiled=mx::compile(quantize);
    auto fused=mx::compile([](const std::vector<mx::array>& in){return std::vector{quantize(in)[2]};});
    for(int domain:{0,1}){
      std::vector<uint16_t> values;
      if(domain==0){for(unsigned i=0;i<65536;++i)if((i&0x7f80)!=0x7f80)values.push_back(uint16_t(i));}
      else{
        for(int sign:{1,-1})for(unsigned c=0;c<7;++c){
          uint16_t center=uint16_t(bits(float(sign)*(kValues[c]+kValues[c+1])/2)>>16);
          for(int delta:{-1,0,1}){for(int j=0;j<32;++j)values.push_back(j==0?uint16_t(bits(6.f)>>16):uint16_t(center+delta));}
        }
      }
      auto expected=reference(values);
      auto input=mx::view(mx::array(values.data(),{int(values.size()/32),32},mx::uint16),mx::bfloat16);
      for(int mode:{0,1,2}){
        auto result=mode==0?quantize({input}):(mode==1?compiled({input}):fused({input}));mx::eval(result);
        auto &y=result[mode==2?0:2];unsigned code_wrong=0,scale_wrong=0,output_wrong=0,tiny_wrong=0;
        if(mode!=2){auto packed=reinterpret_cast<const uint8_t*>(result[0].data<uint32_t>());for(size_t i=0;i<values.size();++i){unsigned c=(packed[i/2]>>((i%2)*4))&15;code_wrong+=c!=expected.codes[i];}for(size_t i=0;i<expected.scales.size();++i)scale_wrong+=result[1].data<uint8_t>()[i]!=expected.scales[i];}
        for(size_t i=0;i<values.size();++i){float got=y.data<float>()[i];if(bits(got)!=bits(expected.output[i])){if(output_wrong<4)std::cout<<std::setprecision(10)<<"quant mismatch domain="<<domain<<" mode="<<mode<<" index="<<i<<" input_bits="<<values[i]<<" got="<<got<<" expected="<<expected.output[i]<<" expected_scale="<<unsigned(expected.scales[i/32])<<'\n';++output_wrong;if(expected.scales[i/32]<=8)++tiny_wrong;}}
        std::cout<<"quant domain="<<domain<<" mode="<<mode<<" samples="<<values.size()<<" code_mismatches="<<code_wrong<<" scale_mismatches="<<scale_wrong<<" output_mismatches="<<output_wrong<<" tiny_output_mismatches="<<tiny_wrong<<'\n';failures+=code_wrong+scale_wrong+output_wrong;
      }
    }
    std::cout<<"total_mismatches="<<failures<<'\n';return failures?1:0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
