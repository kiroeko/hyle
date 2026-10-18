#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <utility>
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/buffer.h"
#include "brgemm_internal.h"
namespace c=hyle::compute;
template<class T>T take(hyle::Result<T> r){if(!r){std::fprintf(stderr,"%.*s\n",int(r.error().message().size()),r.error().message().data());std::exit(2);}return std::move(*r);}
void require(hyle::Status s){if(!s.ok()){std::fprintf(stderr,"%.*s\n",int(s.message().size()),s.message().data());std::exit(2);}}
void gate(c::Provider provider,const char* name){
  auto d=take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{1,8}));
  auto a=take(hyle::mem::Buffer::create(hyle::mem::system_resource(),32,64));
  auto b=take(hyle::mem::Buffer::create(hyle::mem::system_resource(),32,64));
  auto y=take(hyle::mem::Buffer::create(hyle::mem::system_resource(),32,64));
  const float inf=std::numeric_limits<float>::infinity(),nan=std::numeric_limits<float>::quiet_NaN();
  std::array<float,8>x{-inf,nan,inf,-0.0f,0.0f,-1.0e4f,-20.0f,2.0f},v{};v.fill(1);
  require(a.view().write(std::as_bytes(std::span(x))));require(b.view().write(std::as_bytes(std::span(v))));
  auto context=take(c::Context::create(c::cpu::device(),provider));auto queue=take(context.create_queue());
  std::array inputs{take(c::ConstTensorView::create(d,a.view())),take(c::ConstTensorView::create(d,b.view()))};
  auto output=take(c::TensorView::create(d,y.view()));
  auto p=take(c::prepare(context,c::Operation{c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}},inputs,std::span(&output,1)));
  auto lane=take(p.create_invocation());auto done=take(lane.submit(queue,inputs,std::span(&output,1)));require(done.wait());
  std::array<float,8>actual{};require(y.view().read(std::as_writable_bytes(std::span(actual))));
  std::printf("%s separate SiLU gate:",name);for(float f:actual)std::printf(" %+.9g",f);std::puts("");
}
int main(){
  gate(c::native::provider(),"Native");gate(c::ggml_cpu::provider(),"ggml optimized");
  auto a=take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{1,64}));
  auto b=take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{64,64}));
  auto y=take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{1,64}));
  auto context=take(c::Context::create(c::cpu::device(),c::onednn::provider()));
  std::array descriptions{a,b};auto public_plan=c::prepare(context,c::Operation{c::MatMul{}},descriptions,std::span(&y,1),hyle::mem::system_resource());
  if(public_plan)std::puts("oneDNN ordinary M1: admitted");
  else std::printf("oneDNN ordinary M1 rejected: %.*s\n",int(public_plan.error().message().size()),public_plan.error().message().data());
  namespace o=c::onednn::internal;
  o::BrgemmPlan* raw=nullptr;auto status=o::create_brgemm_plan({1,64,64},1,hyle::mem::system_allocator(),&raw);
  if(!status.ok()){std::printf("oneDNN original M1: %.*s\n",int(status.message().size()),status.message().data());return 0;}
  auto scratch=take(hyle::mem::Buffer::create(hyle::mem::system_resource(),o::brgemm_scratchpad_size(raw),64));
  auto mapping=take(scratch.view().map_host(hyle::mem::HostAccess::kReadWrite));auto bytes=take(mapping.writable_bytes());
  require(o::validate_brgemm_scratchpad(raw,bytes.data(),bytes.size()));
  std::array<float,64>left{},output{};std::array<float,4096>right{};left.fill(1);right.fill(1);
  require(o::execute_brgemm(raw,{},left.data(),right.data(),output.data(),bytes.data()));
  for(float f:output)if(f!=64){std::fprintf(stderr,"M1 original numeric mismatch: %g\n",f);return 3;}
  o::destroy_brgemm_plan(raw,hyle::mem::system_allocator());require(mapping.close());
  std::puts("oneDNN original M1 numeric: 64/64 outputs equal 64 PASS (not a timing or ordinary admission claim)");
}
