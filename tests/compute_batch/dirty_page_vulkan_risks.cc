#include "dirty_page_native_risk_fixture.h"
#include "native_vulkan_failure_fixture.h"
namespace {
namespace r=hyle_g03_dirty_risk;namespace n=r::n;namespace c=r::c;
using hyle::Status;
std::atomic<unsigned> confirmed_fences{0};
Status episode(unsigned mode){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
  HYLE_RETURN_IF_ERROR(r::require(!context.environment().software_device,"dirty copy requires actual Vulkan GPU"));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
  // Resources are borrowed by accepted work. Drain even when a test assertion
  // returns early, before the device/staging wrappers can be destroyed.
  struct Close{n::Context& context;~Close(){context.close().IgnoreError();}}close{context};
  HYLE_ASSIGN_OR_RETURN(auto state,r::prepare(context,device,staging));
  hyle_f5_vk_failure_fixture::Boundary boundary(context);boundary.fail_at=mode==1?1:0;
  confirmed_fences=0;
  boundary.device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{
    auto& hook=*hyle_f5_vk_failure_fixture::Boundary::active;++hook.polls;
    if(hook.fail_poll.exchange(false))return VK_ERROR_OUT_OF_HOST_MEMORY;
    const auto result=hook.original.GetFenceStatus(device,fence);
    if(result==VK_SUCCESS)++confirmed_fences;
    return result;
  };
  auto submitted=state.copy.submit();
  if(mode==1){
    // The empty first logical phase was accepted even though the real native
    // copy phase was rejected; observe that accepted prefix through its token.
    HYLE_RETURN_IF_ERROR(r::require(submitted.has_value() && submitted->wait().IsResourceExhausted() &&
        !submitted->output({1},{1},256,256) && boundary.submits==1 && boundary.accepted==0 &&
        !state.copy.unbind_mutable({1}),"rejected Vulkan copy returned mutable writer"));
  }else{
    HYLE_RETURN_IF_ERROR(r::require(submitted.has_value() && boundary.accepted==1 && state.row_pool.stats().active_uses==1,
        "Vulkan dirty risk lacks actual native copy admission"));
    if(mode==2){
      boundary.fail_poll=true;const auto output=submitted->output({1},{1},256,256);
      HYLE_RETURN_IF_ERROR(r::require(!output && !boundary.fail_poll.load() && state.row_pool.stats().active_uses==1 &&
          !state.copy.unbind_mutable({1}),"unknown Vulkan dirty output released use/returned writer"));
    }
    if(mode==3)state.copy.close().IgnoreError();
    const auto terminal=submitted->wait();
    HYLE_RETURN_IF_ERROR(r::require(mode==2?terminal.IsResourceExhausted():terminal.ok(),"Vulkan dirty actual wait lost first terminal"));
    HYLE_RETURN_IF_ERROR(r::require(state.row_pool.stats().active_uses==0 && (boundary.waits>0 || confirmed_fences>0),"Vulkan dirty copy lacks actual fence retirement"));
    if(mode==2)HYLE_RETURN_IF_ERROR(r::require(!submitted->output({1},{1},256,256),"unknown-failed Vulkan dirty copy published output"));
    *submitted={};
    if(mode==2 || mode==3)HYLE_RETURN_IF_ERROR(r::require(!state.copy.unbind_mutable({1}),"failed/closed Vulkan dirty copy withdrew writer"));
    else HYLE_RETURN_IF_ERROR(r::success(state));
  }
  std::fprintf(stderr,"Vulkan dirty copy risk mode=%u actual submits=%u accepted=%u waits=%u confirmed-fences=%u successful/failed/unknown/closed permission return PASS\n",mode,boundary.submits,boundary.accepted,boundary.waits.load(),confirmed_fences.load());
  state.copy.drain().IgnoreError();return Status::Ok();
}
Status run(){for(unsigned mode=0;mode<4;++mode)HYLE_RETURN_IF_ERROR(episode(mode));return Status::Ok();}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"Vulkan dirty risk: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
