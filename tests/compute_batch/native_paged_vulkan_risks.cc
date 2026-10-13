#include "native_paged_risk_fixture.h"
#include "native_vulkan_failure_fixture.h"
namespace {
namespace r=hyle_g01_page_risk;namespace n=r::n;namespace c=r::c;
using hyle::Status;
std::atomic<bool> fail_wait{false};
std::atomic<unsigned> wait_faults{0},native_drains{0};
Status episode(unsigned mode){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
    HYLE_RETURN_IF_ERROR(r::require(!context.environment().software_device,"Vulkan risk requires actual GPU"));
    HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
    HYLE_ASSIGN_OR_RETURN(auto data,r::inputs(context,device,staging));
    if(mode==4)return r::cold_failures(context,device,staging,data);
    if(mode==5)return r::domain_checks(context,device,staging,data);
    HYLE_ASSIGN_OR_RETURN(auto slot,r::create(context,device,staging,data,metadata.get()));
    if(mode){HYLE_RETURN_IF_ERROR(slot.prepare_batch(data.layout));HYLE_RETURN_IF_ERROR(slot.bind(data.query,data.key_frame,data.value_frame));}
    hyle_f5_vk_failure_fixture::Boundary boundary(context);
    fail_wait=false;wait_faults=0;native_drains=0;
    boundary.device.fn.QueueWaitIdle=[](VkQueue queue)->VkResult{
      ++native_drains;return hyle_f5_vk_failure_fixture::Boundary::active->original.QueueWaitIdle(queue);
    };
    boundary.device.fn.WaitForFences=[](VkDevice device,std::uint32_t count,const VkFence* fences,VkBool32 all,std::uint64_t timeout)->VkResult{
      auto& hook=*hyle_f5_vk_failure_fixture::Boundary::active;++hook.waits;
      if(fail_wait.exchange(false)){++wait_faults;return VK_ERROR_OUT_OF_HOST_MEMORY;}
      return hook.original.WaitForFences(device,count,fences,all,timeout);
    };
    if(!mode){
      boundary.fail_at=1;const auto denied=slot.prepare_batch(data.layout);
      HYLE_RETURN_IF_ERROR(r::require(denied.IsResourceExhausted() && boundary.accepted==0 && !slot.submit(),"page metadata rejection published new generation"));
      HYLE_RETURN_IF_ERROR(r::require(slot.report().layout_generation==0,"failed upload published layout generation"));
      slot.drain().IgnoreError();slot={};data.drop_external();HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
    }else{
      // This page recipe uses one original fused native submission, unlike
      // the reference's multiple stages. Test rejection before acceptance
      // (mode 1), then a real blocking observation after acceptance (mode 2).
      boundary.fail_at=mode==1?1:0;
      auto submitted=slot.submit();
      if(mode==2){
        HYLE_RETURN_IF_ERROR(r::require(submitted.has_value() && boundary.accepted>0 &&
            data.query_pool.stats().active_uses==1,"page wait-error episode had no admitted work"));
        fail_wait=true;
      }
      if(mode==3){
        HYLE_RETURN_IF_ERROR(r::require(submitted.has_value() && boundary.accepted>0,"page unknown episode had no accepted native work"));
        boundary.fail_poll=true;const auto observed=submitted->poll();
        HYLE_RETURN_IF_ERROR(r::require(observed && *observed==c::CompletionPoll::kPending &&
            !boundary.fail_poll.load() && data.query_pool.stats().active_uses==1 && !submitted->result(),"page unknown poll released guard or fabricated result"));
        HYLE_RETURN_IF_ERROR(r::require(!data.keys.acquire(0) && !data.keys.acquire(2),"page unknown work allowed shared page write"));
        {HYLE_ASSIGN_OR_RETURN(auto free,data.keys.acquire(1));}
        auto terminal=submitted->wait();
        HYLE_RETURN_IF_ERROR(r::require(!terminal && terminal.error().IsResourceExhausted() && boundary.waits>0 &&
            data.query_pool.stats().active_uses==0,"page actual fence retirement lost first error/guard"));
        *submitted={};
        HYLE_RETURN_IF_ERROR(r::require(!slot.prepare_batch(data.next).ok() && !slot.submit(),"page unknown-failed slot reused"));
        const auto drained=slot.drain();HYLE_RETURN_IF_ERROR(r::require(drained.IsResourceExhausted(),"page drain replaced original unknown error"));
      }else{
        Status terminal;
        if(submitted){auto result=submitted->wait();if(!result)terminal=result.error();}else terminal=submitted.error();
        std::printf("Vulkan page fault diagnostic mode=%u submitted=%d submits=%u accepted=%u waits=%u drains=%u injected-waits=%u terminal=%.*s\n",
            mode,submitted.has_value(),boundary.submits,boundary.accepted,boundary.waits.load(),native_drains.load(),wait_faults.load(),int(terminal.message().size()),terminal.message().data());
        HYLE_RETURN_IF_ERROR(r::require(terminal.IsResourceExhausted() &&
            (mode==1?!submitted && boundary.submits==1 && boundary.accepted==0:
                     submitted.has_value() && native_drains>0 && wait_faults==1 && boundary.accepted>0),
            "page rejection/accepted observation error lacked actual boundary proof"));
        if(submitted){HYLE_RETURN_IF_ERROR(r::require(!submitted->result(),"failed page math published output"));*submitted={};}
        HYLE_RETURN_IF_ERROR(r::require(!slot.prepare_batch(data.next).ok() && !slot.submit(),"failed page slot reused"));
        slot.drain().IgnoreError();
      }
      slot={};data.drop_external();HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
    }
    std::printf("Vulkan native page risk mode=%u actual submits=%u accepted=%u, all exact pages reusable after retirement PASS\n",mode,boundary.submits,boundary.accepted);
  }
  return r::require(metadata.used()==0,"Vulkan page risk metadata leaked");
}
Status run(int selected){for(unsigned mode=0;mode<6;++mode)if(selected<0 || int(mode)==selected)HYLE_RETURN_IF_ERROR(episode(mode));return Status::Ok();}
}
int main(int argc,char** argv){const auto status=run(argc>1?std::atoi(argv[1]):-1);if(!status.ok()){std::fprintf(stderr,"FAIL: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
