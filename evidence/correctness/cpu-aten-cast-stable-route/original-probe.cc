#include "original-program.h"
#include <ATen/core/grad_mode.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static std::size_t assertions = 0;
void check(bool result, const char *what) {
  ++assertions;
  if (!result) { std::fprintf(stderr,"%s at check %zu\n",what,assertions); std::exit(2); }
}
struct Format { at::ScalarType type; int exponent, mantissa, bias, width; bool infinity; };
constexpr std::array formats{
  Format{at::kFloat,8,23,127,4,true}, Format{at::kHalf,5,10,15,2,true},
  Format{at::kBFloat16,8,7,127,2,true}, Format{at::kFloat8_e4m3fn,4,3,7,1,false},
  Format{at::kFloat8_e5m2,5,2,15,1,true}};
double decode(std::uint32_t bits, const Format &f) {
  if (f.width==4) return std::bit_cast<float>(bits);
  const int sign = bits >> (f.exponent+f.mantissa);
  const auto mantissa = bits & ((1U<<f.mantissa)-1);
  const auto exponent = (bits>>f.mantissa)&((1U<<f.exponent)-1);
  if (exponent==((1U<<f.exponent)-1) && (f.infinity || mantissa==((1U<<f.mantissa)-1)))
    return mantissa==0 && f.infinity ? (sign ? -INFINITY : INFINITY) : std::numeric_limits<double>::quiet_NaN();
  const auto magnitude = exponent==0 ? std::ldexp(double(mantissa),1-f.bias-f.mantissa)
      : std::ldexp(1+std::ldexp(double(mantissa),-f.mantissa),int(exponent)-f.bias);
  return sign ? -magnitude : magnitude;
}
std::uint32_t encode(float value, const Format &f, bool negative_nan) {
  const auto sign = std::uint32_t(std::signbit(value) || (std::isnan(value)&&negative_nan))<<(f.exponent+f.mantissa);
  const std::uint32_t nan = f.width==1 ? (f.infinity ? 0x7e : 0x7f) : f.mantissa==10 ? 0x7fff : 0x7fc0;
  if (std::isnan(value)) return sign|nan;
  const auto infinity = ((1U<<f.exponent)-1)<<f.mantissa;
  const auto maximum = f.infinity ? infinity-1 : 0x7e;
  if (std::isinf(value)) return sign|(f.infinity ? infinity : maximum);
  const double magnitude = std::abs(double(value));
  if (f.width==2 && magnitude>=std::ldexp(1.,((1<<f.exponent)-2)-f.bias+1)-std::ldexp(1.,((1<<f.exponent)-2)-f.bias-f.mantissa-1)) return sign|infinity;
  // Independent nearest-representable search; ties select an even encoding.
  unsigned lo=0, hi=maximum;
  while(lo<hi) {const auto mid=(lo+hi)/2;if(decode(mid,f)<magnitude)lo=mid+1;else hi=mid;}
  if(lo>0) {
    const auto lower=decode(lo-1,f), upper=decode(lo,f);
    if(magnitude-lower<upper-magnitude || (magnitude-lower==upper-magnitude && ((lo-1)&1)==0)) --lo;
  }
  return sign|lo;
}
std::uint32_t float_bits(std::uint32_t source, const Format &f) {
  if (f.width==4) return source;
  const auto value = static_cast<float>(decode(source,f));
  if (!std::isnan(value)) return std::bit_cast<std::uint32_t>(value);
  const auto sign = (source>>(f.exponent+f.mantissa))<<31;
  if(f.width==2 && f.mantissa==10) return sign|0x7f800000|((source&0x3ff)<<13);
  if(f.width==2) return source<<16;
  return f.infinity ? sign|0x7f800000|((source&3)<<21) : sign|0x7fc00000;
}
int main() {
  at::NoGradGuard guard;
  std::vector<std::uint32_t> floats{
    0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7f800001,0xff800001,
    0x7fc12345,0xffc12345,0x7f7fffff,0xff7fffff};
  std::uint32_t state=0x8372916d;
  for(int i=0;i<65536;++i){state=1664525U*state+1013904223U;floats.push_back(state);}
  // Explicit values just below/at/above every FP8 RNE midpoint and overflow.
  for(const auto &f : formats) if(f.width==1) {
    const unsigned maximum=f.infinity ? 0x7b : 0x7e;
    for(unsigned code=0;code<maximum;++code){
      const float midpoint=static_cast<float>((decode(code,f)+decode(code+1,f))/2);
      for(float x : {std::nextafter(midpoint,-INFINITY),midpoint,std::nextafter(midpoint,INFINITY)})
        for(float y : {x,-x}) floats.push_back(std::bit_cast<std::uint32_t>(y));
    }
    for(float x : {448.F,449.F,464.F,57344.F,61440.F,65504.F,65520.F,INFINITY,-INFINITY})
      floats.push_back(std::bit_cast<std::uint32_t>(x));
  }
  for(const auto &source : formats) {
    const std::size_t count=source.width==4 ? floats.size() : std::size_t(1)<<(8*source.width);
    std::vector<std::uint8_t> input(count*source.width);
    for(std::size_t i=0;i<count;++i){const std::uint32_t bits=source.width==4?floats[i]:std::uint32_t(i);std::memcpy(input.data()+i*source.width,&bits,source.width);}
    const auto native=at::from_blob(input.data(),{std::int64_t(input.size())},at::kByte).view(source.type);
    for(const auto &destination : formats) {
      const auto result=original_cast(native,destination.type).contiguous().view(at::kByte);
      check(result.numel()==std::int64_t(count*destination.width),"native output byte capacity differs");
      const auto *bytes=result.data_ptr<std::uint8_t>();
      for(std::size_t i=0;i<count;++i){
        const auto bits=source.width==4 ? floats[i] : std::uint32_t(i);
        std::uint32_t actual=0;std::memcpy(&actual,bytes+i*destination.width,destination.width);
        const auto widened=float_bits(bits,source);
        const auto expected=source.type==destination.type ? bits : destination.width==4 ? widened
          : encode(std::bit_cast<float>(widened),destination,(widened>>31)!=0);
        if(actual!=expected) {std::fprintf(stderr,"src=%d dst=%d input=%08x actual=%08x expected=%08x\n",int(source.type),int(destination.type),bits,actual,expected);return 2;}
        check(true,"numeric Cast mismatch");
      }
      std::printf("original Cast src=%d dst=%d %zu bit results PASS\n",int(source.type),int(destination.type),count);
    }
  }
  for(const auto &f : formats) if(f.width>=2) {
    std::vector<float> values{-INFINITY,-100.F,-90.F,-80.F,-0.F,0.F,INFINITY,std::numeric_limits<float>::quiet_NaN()};
    for(int i=-2000;i<=2000;++i)values.push_back(float(i)/20);
    const auto input=at::from_blob(values.data(),{std::int64_t(values.size())},at::kFloat).to(f.type).to(at::kFloat);
    const auto output=at::sigmoid(input.to(at::kDouble)).to(at::kFloat).to(f.type).to(at::kFloat);
    for(std::size_t i=0;i<values.size();++i){
      const double x=input.data_ptr<float>()[i],e=std::exp(-std::abs(x));
      float expected=static_cast<float>(x<0?e/(1+e):1/(1+e));
      if(f.type!=at::kFloat) expected=static_cast<float>(decode(encode(expected,f,false),f));
      const float actual=output.data_ptr<float>()[i];
      check(std::isnan(expected)?std::isnan(actual):f.type==at::kFloat?std::abs(double(actual)-expected)<=3e-6*std::abs(double(expected))+4*std::numeric_limits<float>::denorm_min():std::abs(double(actual)-expected)<=std::abs(double(expected))*(f.type==at::kHalf?.001:.008),"stable Sigmoid mismatch");
    }
  }
  std::printf("original ATen Cast/Sigmoid %zu independent assertions PASS\n",assertions);
}
