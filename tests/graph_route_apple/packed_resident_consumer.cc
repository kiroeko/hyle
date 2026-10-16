#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <hyle/compute/low_precision.h>
#include <hyle/graph/apple.h>

namespace {
namespace c=hyle::compute;
namespace a=c::apple;
namespace g=hyle::graph;
namespace mx=mlx::core;
template<class T> T take(hyle::Result<T> result) {
  if(!result)throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if(!status.ok())throw std::runtime_error(std::string(status.message()));
}
mx::array carrier(const std::vector<std::uint8_t>& codes,const std::vector<float>& scales) {
  std::vector<std::uint8_t> bytes(codes.size()+scales.size()*sizeof(float));
  std::memcpy(bytes.data(),codes.data(),codes.size());
  std::memcpy(bytes.data()+codes.size(),scales.data(),scales.size()*sizeof(float));
  auto value=mx::array(bytes.data(),{int(bytes.size())},mx::uint8);
  mx::async_eval({value});mx::synchronize(mx::default_stream(mx::Device::gpu));value.wait();return value;
}
float decode(std::uint8_t code) {
  const auto e=(code>>3)&15,m=code&7;
  const auto value=e?std::ldexp(1.F+float(m)/8,int(e)-7):std::ldexp(float(m),-9);
  return code&128?-value:value;
}
void run(int rows) {
  constexpr int n=19,k=256,blocks=k/128;
  auto context=take(a::Context::create());
  auto builder=take(g::GraphBuilder::create());
  const auto x=take(builder.input(take(c::float8_activation_description(rows,k,c::Float8ScaleMode::kFloat32Block128))));
  const auto w=take(builder.input(take(c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128))));
  const auto y=take(builder.add(c::MatMul{},x,w));require(builder.publish(y));
  auto logical=take(std::move(builder).finish());
  auto program=take(g::apple::compile(context,logical));logical={};
  auto first=take(program.create_invocation()),second=take(program.create_invocation());
  std::vector<std::uint8_t> qa(rows*k),qw(n*k);
  std::vector<float> qs(rows*blocks),ws(blocks);
  std::optional<mx::array> old;
  std::vector<float> retained;
  for(int step=0;step<65;++step) {
    for(int r=0;r<rows;++r) {
      qa[r*k]=std::uint8_t(0x30+(r+step)%8);
      qa[r*k+129]=std::uint8_t(0x38+(r+step)%8);
      qs[r*blocks]=.125F*float(1+(r+step)%4);qs[r*blocks+1]=.25F;
    }
    for(int column=0;column<n;++column)for(int j=0;j<k;++j)
      qw[column*k+j]=std::uint8_t(0x30+(column+j+step)%8)|((j%3==0)?128:0);
    ws={.25F,.125F*float(1+step%4)};
    const std::array inputs{carrier(qa,qs),carrier(qw,ws)};
    auto& slot=step%2?second:first;require(slot.bind(inputs));
    auto work=take(slot.submit());auto result=take(work.wait());
    const auto output=result.get()->arrays()[0];
    if(output.dtype()!=mx::float32||output.size()!=std::size_t(rows*n))
      throw std::runtime_error("Graph FP8 output description differs");
    for(int r=0;r<rows;++r)for(int column=0;column<n;++column) {
      float expected=0;
      for(int block=0;block<blocks;++block) {
        const int j=block==0?0:129;
        const double dot=double(decode(qa[r*k+j]))*decode(qw[column*k+j]);
        float value=float(dot)*qs[r*blocks+block];value=value*ws[block];expected=expected+value;
      }
      if(output.data<float>()[r*n+column]!=expected)
        throw std::runtime_error("Graph FP8 stored-byte/ordered-scale oracle differs");
    }
    if(step==0){old.emplace(output);retained.assign(output.data<float>(),output.data<float>()+rows*n);}
  }
  first={};second={};program={};require(context.close());
  if(std::vector<float>(old->data<float>(),old->data<float>()+rows*n)!=retained)
    throw std::runtime_error("Graph FP8 old array changed after program/context close");
  std::printf("PASS Apple Graph packed M=%d:65 steps/two slots/changed codes/scales/ordered oracle/old array after close\n",rows);
}
} // namespace
int main() {
  try {for(int rows:{1,17,64})run(rows);return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
