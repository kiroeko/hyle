#include <array>
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
#include <hyle/compute/apple.h>
#include <hyle/compute/low_precision.h>
#include <hyle/compute/operation.h>

namespace {
namespace c=hyle::compute;
namespace a=c::apple;
namespace mx=mlx::core;
template<class T> T take(hyle::Result<T> result) {
  if(!result)throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if(!status.ok())throw std::runtime_error(std::string(status.message()));
}
mx::array carrier(std::vector<std::uint8_t> codes,float scale) {
  const auto size=codes.size();codes.resize(size+sizeof(float));
  std::memcpy(codes.data()+size,&scale,sizeof(scale));
  auto value=mx::array(codes.data(),{int(codes.size())},mx::uint8);
  mx::async_eval({value});mx::synchronize(mx::default_stream(mx::Device::gpu));value.wait();return value;
}
void run() {
  constexpr int n=9,k=128;
  auto context=take(a::Context::create());
  const std::array specs{
    take(a::input_spec(take(c::float8_activation_description(1,k,c::Float8ScaleMode::kFloat32Block128)))),
    take(a::input_spec(take(c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128))))};
  auto program=take(context.prepare(c::MatMul{},specs));
  auto first=take(program.create_invocation()),second=take(program.create_invocation());
  std::vector<std::uint8_t> qa(k),qw(n*k);
  qa[0]=0x38;
  for(int row=0;row<n;++row)for(int j=0;j<k;++j)qw[row*k+j]=std::uint8_t(0x38+row%8);
  auto weight=carrier(qw,.25F);
  std::optional<mx::array> old;
  std::vector<float> retained;
  for(int step=0;step<64;++step) {
    const float scale=.5F+float(step)*.125F;
    const std::array inputs{carrier(qa,scale),weight};
    auto& slot=step%2?second:first;require(slot.bind(inputs));
    auto work=take(slot.submit());auto result=take(work.wait());
    const auto output=result.get()->arrays()[0];
    if(output.dtype()!=mx::float32||output.size()!=n)throw std::runtime_error("packed SDK output description differs");
    for(int row=0;row<n;++row)if(output.data<float>()[row]!=(1.F+float(row%8)*.125F)*scale*.25F)
      throw std::runtime_error("packed SDK independent oracle differs");
    if(step==0){old.emplace(output);retained.assign(output.data<float>(),output.data<float>()+n);}
  }
  first={};second={};program={};require(context.close());
  if(std::vector<float>(old->data<float>(),old->data<float>()+n)!=retained)
    throw std::runtime_error("packed SDK old native array changed after close");
}
} // namespace
int main() {
  try {run();std::puts("PASS Apple public resident packed consumer: 64 steps, two slots, retained array after close");return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
