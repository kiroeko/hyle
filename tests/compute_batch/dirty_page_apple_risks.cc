#define HYLE_G01_PAGE_RISK_APPLE 1
#include "dirty_page_native_risk_fixture.h"
#include "native_apple_failure_fixture.h"
namespace {
namespace r=hyle_g03_dirty_risk;namespace n=r::n;namespace c=r::c;namespace mx=mlx::core;
using hyle::Status;
Status episode(unsigned mode){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto state,r::prepare(context,device,staging));
  auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Release{decltype(gate) value;n::Context& context;~Release(){value->release();context.close().IgnoreError();}}release{gate,context};
  const std::array specs{n::InputSpec{{4},mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto& inputs,mx::Stream stream){
    auto prefix=mx::square(inputs[0],stream);
    return std::vector{mx::array(prefix.shape(),prefix.dtype(),std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{prefix})};
  },specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());
  const std::array input{mx::array({1.F,2.F,3.F,4.F})};HYLE_RETURN_IF_ERROR(blocking.bind(input));
  {std::lock_guard lock(gate->mutex);gate->armed=true;gate->fail=mode==2;}
  HYLE_ASSIGN_OR_RETURN(auto prefix,blocking.submit());
  HYLE_RETURN_IF_ERROR(r::require(gate->await(),"dirty copy did not enter actual GPU checkpoint"));
  HYLE_ASSIGN_OR_RETURN(auto submitted,state.copy.submit());
  HYLE_RETURN_IF_ERROR(r::require(state.copy.in_flight() && state.row_pool.stats().active_uses==1 &&
      !state.copy.unbind_mutable({1}) && !submitted.output({1},{1},256,256),"queued native copy fabricated output or withdrew writer"));
  if(mode==1)HYLE_RETURN_IF_ERROR(submitted.request_cancel());
  if(mode==3){
    submitted={};
    HYLE_RETURN_IF_ERROR(r::require(!state.copy.unbind_mutable({1}) && state.row_pool.stats().active_uses==1,
        "abandoned queued copy released actual write/read uses"));
    gate->release();HYLE_RETURN_IF_ERROR(context.close());
    HYLE_RETURN_IF_ERROR(r::success(state));
  }else{
    if(mode==4)state.copy.close().IgnoreError();
    gate->release();const auto terminal=submitted.wait();
    HYLE_RETURN_IF_ERROR(r::require(mode==1?terminal.IsCancelled():mode==2?terminal.IsInternal():terminal.ok(),
        "dirty native cancellation/prefix failure lost terminal"));
    HYLE_RETURN_IF_ERROR(r::require(state.row_pool.stats().active_uses==0,"dirty copy actual retirement kept source use"));
    if(mode==1 || mode==2)HYLE_RETURN_IF_ERROR(r::require(!submitted.output({1},{1},256,256),"failed dirty copy published ready output"));
    submitted={};
    if(mode==1 || mode==2 || mode==4)HYLE_RETURN_IF_ERROR(r::require(!state.copy.unbind_mutable({1}),"failed/cancelled/closed copy withdrew mutable page"));
    else HYLE_RETURN_IF_ERROR(r::success(state));
  }
  std::fprintf(stderr,"Apple dirty copy risk mode=%u queued native admission, successful/failed/abandoned/cancelled/closed permission return PASS\n",mode);
  return Status::Ok();
}
Status run(){for(unsigned mode=0;mode<5;++mode)HYLE_RETURN_IF_ERROR(episode(mode));return Status::Ok();}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"Apple dirty risk: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
