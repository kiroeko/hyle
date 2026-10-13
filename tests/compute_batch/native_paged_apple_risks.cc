#define HYLE_G01_PAGE_RISK_APPLE 1
#include "native_paged_risk_fixture.h"
#include "native_apple_failure_fixture.h"
namespace {
namespace r=hyle_g01_page_risk;namespace n=r::n;namespace c=r::c;namespace mx=mlx::core;
using hyle::Status;
Status episode(unsigned mode){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
    HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
    HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
    HYLE_ASSIGN_OR_RETURN(auto data,r::inputs(context,device,staging));
    if(mode==4)return r::cold_failures(context,device,staging,data);
    if(mode==5)return r::domain_checks(context,device,staging,data);
    HYLE_ASSIGN_OR_RETURN(auto slot,r::create(context,device,staging,data,metadata.get()));
    if(mode){HYLE_RETURN_IF_ERROR(slot.prepare_batch(data.layout));HYLE_RETURN_IF_ERROR(slot.bind(data.query,data.key_frame,data.value_frame));}
    auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
    struct Release{std::shared_ptr<hyle_f5_apple_failure_fixture::Gate> gate;n::Context& context;~Release(){gate->release();context.close().IgnoreError();}}release{gate,context};
    const std::array specs{n::InputSpec{{4},mx::float32}};
    HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto& inputs,mx::Stream stream){
      auto prefix=mx::square(inputs[0],stream);
      return std::vector{mx::array(prefix.shape(),prefix.dtype(),std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{prefix})};
    },specs));
    HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());
    const std::array values{mx::array({1.F,2.F,3.F,4.F})};HYLE_RETURN_IF_ERROR(blocking.bind(values));
    {std::lock_guard lock(gate->mutex);gate->armed=true;gate->fail=mode==0 || mode==3;}
    HYLE_ASSIGN_OR_RETURN(auto prefix,blocking.submit());
    HYLE_RETURN_IF_ERROR(r::require(gate->await(),"page risk did not enter actual GPU checkpoint"));
    if(!mode){
      std::promise<void> started;auto entered=started.get_future();
      auto prepared=std::async(std::launch::async,[&]{started.set_value();return slot.prepare_batch(data.layout);});entered.get();
      const bool waited=prepared.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout;
      gate->release();const auto terminal=prepared.get();
      HYLE_RETURN_IF_ERROR(r::require(waited && terminal.IsInternal() && !slot.submit() &&
          !slot.prepare_batch(data.next).ok() && slot.report().layout_generation==0,"page failed metadata queued prefix published/reused generation"));
      slot.drain().IgnoreError();slot={};data.drop_external();HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
    }else{
      HYLE_ASSIGN_OR_RETURN(auto queued,slot.submit());
      const auto observed=queued.poll();
      HYLE_RETURN_IF_ERROR(r::require(observed && *observed==c::CompletionPoll::kPending &&
          data.query_pool.stats().active_uses==1 && !queued.result(),"page queued native guard/result differs"));
      {HYLE_ASSIGN_OR_RETURN(auto free,data.keys.acquire(1));}
      if(mode==1)HYLE_RETURN_IF_ERROR(queued.request_cancel());
      if(mode==2){
        queued={};slot={};data.drop_external();
        HYLE_RETURN_IF_ERROR(r::require(data.query_pool.stats().active_uses==1 &&
            !data.keys.acquire(0) && !data.keys.acquire(2) && !data.values.acquire(0) && !data.values.acquire(2),"page abandonment released actual admitted guards"));
        gate->release();HYLE_RETURN_IF_ERROR(context.close());
      }else{
        gate->release();auto terminal=queued.wait();
        HYLE_RETURN_IF_ERROR(r::require(!terminal && (mode==1?terminal.error().IsCancelled():terminal.error().IsInternal()),"page cancellation/native prefix error lost first terminal"));
        queued={};HYLE_RETURN_IF_ERROR(r::require(!slot.prepare_batch(data.next).ok() && !slot.submit(),"page failed native slot reused after retirement"));
        slot.drain().IgnoreError();slot={};data.drop_external();
      }
      HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
    }
    prefix={};
    std::printf("Apple native page risk mode=%u actual queued native work, query guard and all exact pages retired PASS\n",mode);
  }
  return r::require(metadata.used()==0,"Apple page risk metadata leaked");
}
Status run(int selected){for(unsigned mode=0;mode<6;++mode)if(selected<0 || int(mode)==selected)HYLE_RETURN_IF_ERROR(episode(mode));return Status::Ok();}
}
int main(int argc,char** argv){const auto status=run(argc>1?std::atoi(argv[1]):-1);if(!status.ok()){std::fprintf(stderr,"FAIL: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
