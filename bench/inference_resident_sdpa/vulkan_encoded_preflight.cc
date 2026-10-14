#include "reference.h"
#include "vulkan_program_internal.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace {
namespace c=hyle::compute;namespace n=c::vulkan;
void check(hyle::Status status){if(!status.ok()){
  std::fprintf(stderr,"original Q8 preflight: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}}
template<class T>T take(hyle::Result<T> result){if(!result)check(result.error());return std::move(*result);}
double half(std::uint16_t bits){
  const auto e=(bits>>10)&31u,f=bits&1023u;const double sign=bits&0x8000u?-1.:1.;
  return sign*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));
}
struct Cache final {
  std::vector<std::byte> bytes;std::vector<double> decoded;
  Cache(unsigned rows,unsigned depth,unsigned seed){
    decoded.resize(std::size_t(rows)*2*depth);bytes.resize(decoded.size()/32*34);
    for(std::size_t block=0;block<decoded.size()/32;++block){
      const auto bits=std::uint16_t(0x1800+((block+seed)%5)*0x400);
      std::memcpy(bytes.data()+block*34,&bits,2);
      for(unsigned i=0;i<32;++i){
        const auto code=std::int8_t(int((block*13+i*7+seed)%31)-15);
        bytes[block*34+2+i]=std::byte(std::uint8_t(code));decoded[block*32+i]=half(bits)*code;
      }
    }
  }
};
void run(n::Context&context,unsigned rows,unsigned keys,unsigned dk,unsigned dv,bool causal,int window){
  const auto dense=[](c::DataType type,std::array<std::int64_t,4> shape){return take(c::TensorDesc::dense(type,shape));};
  const auto qd=dense(c::DataType::kFloat16,{1,rows,8,dk});
  const auto kd=dense(c::DataType::kUInt8,{1,keys,2,dk/32*34});
  const auto vd=dense(c::DataType::kUInt8,{1,keys,2,dv/32*34});
  const auto od=dense(c::DataType::kFloat16,{1,rows,8,dv});
  std::vector<std::uint16_t> query(std::size_t(rows)*8*dk);
  constexpr std::array<std::uint16_t,7> codes{0,0x2400,0xa400,0x2800,0xa800,0x2c00,0xac00};
  for(std::size_t i=0;i<query.size();++i)query[i]=codes[(i*13)%codes.size()];
  Cache k(keys,dk,17),v(keys,dv,29);const auto scale=1.F/std::sqrt(float(dk));
  const std::array inputs{take(context.upload(qd,std::as_bytes(std::span(query)))),
    take(context.upload(kd,k.bytes)),take(context.upload(vd,v.bytes))};
  auto program=hyle::bench::resident::encoded_reference_program(context,qd,kd,vd,od,causal,window,scale);
  const auto dispatches=n::detail::Access::program(program)->dispatches.size();
  // This finite case must include the original split reduction, in addition
  // to Q conversion, flash and output conversion; math agreement alone is not
  // evidence that the split path was instantiated.
  if(rows==1 && keys==2048 && dispatches!=4)
    check(hyle::Status::Internal("original long-KV split path was not instantiated"));
  std::array slots{take(program.create_invocation()),take(program.create_invocation())};
  std::vector<double> expected(std::size_t(rows)*8*dv),weights(keys);
  for(unsigned row=0;row<rows;++row)for(unsigned h=0;h<8;++h){
    const int position=int(keys)-int(rows)+int(row);double sum=0;
    for(unsigned j=0;j<keys;++j){
      if(causal&&(int(j)>position||(window>=0&&int(j)<std::max(0,position-window)))){weights[j]=0;continue;}
      double dot=0;for(unsigned d=0;d<dk;++d)dot+=half(query[(std::size_t(row)*8+h)*dk+d])*k.decoded[(std::size_t(j)*2+h/4)*dk+d];
      weights[j]=std::exp(dot*scale);sum+=weights[j];
    }
    for(unsigned d=0;d<dv;++d)for(unsigned j=0;j<keys;++j)
      expected[(std::size_t(row)*8+h)*dv+d]+=weights[j]/sum*v.decoded[(std::size_t(j)*2+h/4)*dv+d];
  }
  double worst=0;
  for(unsigned replay=0;replay<4;++replay){
    auto&slot=slots[replay%2];check(slot.bind(std::span(inputs)));auto submitted=take(slot.submit());auto result=take(submitted.wait());
    std::vector<std::uint16_t> output(expected.size());check(context.download(result->tensors()[0],std::as_writable_bytes(std::span(output))));
    for(std::size_t i=0;i<output.size();++i){
      if((output[i]&0x7c00)==0x7c00)check(hyle::Status::Internal("original Q8 shader produced nonfinite output"));
      worst=std::max(worst,std::abs(half(output[i])-expected[i]));
    }
  }
  for(unsigned port=1;port<3;++port){const auto&original=port==1?k.bytes:v.bytes;
    std::vector<std::byte> after(original.size());check(context.download(inputs[port],after));
    if(after!=original)check(hyle::Status::Internal("original Q8 shader modified readonly cache"));
  }
  if(worst>2e-3)check(hyle::Status::Internal("original Q8 shader differs from independent stored-byte double oracle"));
  std::printf("ORIGINAL_Q8_PREFLIGHT Q=%u KV=%u Dqk=%u Dv=%u causal=%d window=%d replays=4 dispatches=%zu max_abs_error=%.9g PASS\n",
              rows,keys,dk,dv,int(causal),window,dispatches,worst);
}
void plain_control(n::Context&context,bool f32){
  const auto type=f32?c::DataType::kFloat32:c::DataType::kFloat16;
  const auto qd=take(c::TensorDesc::dense(type,std::array<std::int64_t,4>{1,2,8,64}));
  const auto kd=take(c::TensorDesc::dense(type,std::array<std::int64_t,4>{1,7,2,64}));
  std::vector<std::byte> query(take(qd.required_bytes()),std::byte{0}),cache(take(kd.required_bytes()));
  for(std::size_t i=0;i<7*2*64;++i){
    if(f32){const float value=.25F;std::memcpy(cache.data()+i*4,&value,4);}
    else {const std::uint16_t value=0x3400;std::memcpy(cache.data()+i*2,&value,2);}
  }
  const std::array inputs{take(context.upload(qd,query)),take(context.upload(kd,cache)),take(context.upload(kd,cache))};
  auto program=hyle::bench::resident::reference_program(context,qd,kd,true);
  auto slot=take(program.create_invocation());check(slot.bind(std::span(inputs)));
  for(unsigned replay=0;replay<2;++replay){
    auto done=take(slot.submit());auto result=take(done.wait());std::vector<std::byte> output(query.size());
    check(context.download(result->tensors()[0],output));
    for(std::size_t i=0;i<2*8*64;++i){double actual;
      if(f32){float value;std::memcpy(&value,output.data()+i*4,4);actual=value;}
      else {std::uint16_t value;std::memcpy(&value,output.data()+i*2,2);actual=half(value);}
      if(!std::isfinite(actual)||std::abs(actual-.25)>1e-4)check(hyle::Status::Internal("original plain reference regression"));
    }
  }
  std::printf("ORIGINAL_PLAIN_CONTROL type=%s PASS\n",f32?"F32":"F16");
}
}
int main(){
  auto context=take(n::Context::create());const auto&env=context.environment();
  if(env.software_device)check(hyle::Status::Unimplemented("original Q8 preflight requires real GPU"));
  std::printf("ORIGINAL_Q8_ENV device=%s driver=%s revision=%s\n",env.device_name.c_str(),env.driver_name.c_str(),env.ggml_revision.c_str());
  for(const auto dims:std::array<std::array<unsigned,2>,4>{{{64,64},{128,128},{256,256},{64,128}}})
    for(int window:{-1,0,16})run(context,dims[0]==256?65u:17u,dims[0]==256?129u:47u,dims[0],dims[1],true,window);
  run(context,17,7,64,128,false,-1);run(context,1,2048,64,128,true,16);
  plain_control(context,false);plain_control(context,true);
  check(context.close());
}
