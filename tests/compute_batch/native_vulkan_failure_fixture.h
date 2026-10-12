#pragma once
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/simd/dtype.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <utility>

namespace hyle_f5_vk_failure_fixture {
namespace c=hyle::compute;namespace v=c::vulkan;namespace d=v::detail;
using hyle::Status;
inline Status check(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Boundary final {
  d::Device&device;d::DeviceFunctions original;
  unsigned submits=0,accepted=0,fail_at=0;std::atomic<bool> fail_poll{false};std::atomic<unsigned> polls{0},waits{0};
  static inline Boundary*active=nullptr;
  explicit Boundary(const v::Context&context):device(*d::Access::runtime(context)->device),original(device.fn){
    active=this;
    device.fn.QueueSubmit=[](VkQueue queue,std::uint32_t count,const VkSubmitInfo*info,VkFence fence)->VkResult{
      auto&s=*active;if(++s.submits==s.fail_at)return VK_ERROR_OUT_OF_HOST_MEMORY;
      const auto status=s.original.QueueSubmit(queue,count,info,fence);s.accepted+=status==VK_SUCCESS;return status;
    };
    device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{auto&s=*active;++s.polls;if(s.fail_poll.exchange(false))return VK_ERROR_OUT_OF_HOST_MEMORY;return s.original.GetFenceStatus(device,fence);};
    device.fn.WaitForFences=[](VkDevice device,std::uint32_t count,const VkFence*fences,VkBool32 all,std::uint64_t timeout)->VkResult{auto&s=*active;++s.waits;return s.original.WaitForFences(device,count,fences,all,timeout);};
  }
  ~Boundary(){device.fn=original;active=nullptr;}
};
inline hyle::Result<std::array<c::ReadLease,3>> inputs(const v::Context&context,c::LeasePool&device,c::LeasePool&host,hyle::mem::MemoryResourceRef gpu,hyle::mem::MemoryResourceRef staging){
  const std::array specs{c::CopyResourceSpec{staging,256},c::CopyResourceSpec{gpu,256}};std::array<c::ReadLease,3> result;
  unsigned port=0;for(auto&read:result){
    std::array<std::uint16_t,128> words{};for(unsigned i=0;i<words.size();++i)words[i]=hyle::simd::f32_to_f16(port==2?float(i+1)/256.f:float(i%7)/32.f).bits;++port;
    HYLE_ASSIGN_OR_RETURN(auto source,host.allocate(256));HYLE_RETURN_IF_ERROR(source.view()->write(std::as_bytes(std::span(words))));HYLE_ASSIGN_OR_RETURN(auto ready,source.publish());
    HYLE_ASSIGN_OR_RETURN(auto target,device.allocate(256));HYLE_ASSIGN_OR_RETURN(auto program,v::CopyBatchProgram::prepare(context,specs));HYLE_ASSIGN_OR_RETURN(auto invocation,v::CopyBatchInvocation::create(program));
    HYLE_RETURN_IF_ERROR(invocation.bind_source({0},ready));HYLE_RETURN_IF_ERROR(invocation.bind_mutable({1},std::move(target)));const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,256}}};
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto copied,invocation.submit());HYLE_RETURN_IF_ERROR(copied.wait());HYLE_ASSIGN_OR_RETURN(read,copied.output({0},{1},0,256));
  }return result;
}
inline Status episode(unsigned mode,bool graph){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto context,v::Context::create());HYLE_RETURN_IF_ERROR(check(!context.environment().software_device,"Vulkan fault requires actual GPU"));
    HYLE_ASSIGN_OR_RETURN(auto memory,v::MemoryResource::create(context,v::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,v::MemoryResource::create(context,v::MemoryKind::kStaging));
    struct Close{v::Context&context;~Close(){context.close().IgnoreError();}}close{context};
    c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,.query_heads=2,.kv_heads=2,.head_dim=64,.page_size=1};
    HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));const std::array<std::int32_t,2> ptr{0,1};HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,{},{}}));
    HYLE_ASSIGN_OR_RETURN(auto recipe,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=1,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=1u<<20,.scale=.125f}));
    HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),4096));HYLE_ASSIGN_OR_RETURN(auto host,c::LeasePool::create(staging.ref(),4096));HYLE_ASSIGN_OR_RETURN(auto ready,inputs(context,pool,host,memory.ref(),staging.ref()));
    const c::MaterializedNativeBatchOptions options{.max_metadata_backing_bytes=1u<<20,.allocator=metadata.get()};
    hyle::Result<v::MaterializedBatchInvocation> made=graph?hyle::graph::vulkan::prepare_materialized_batch(context,recipe,memory,staging,options):v::MaterializedBatchInvocation::create(context,recipe,memory,staging,options);
    if(!made)return made.error();auto controller=std::move(*made);
    if(mode){HYLE_RETURN_IF_ERROR(controller.prepare_batch(layout));HYLE_RETURN_IF_ERROR(controller.bind(ready[0],ready[1],ready[2]));}
    Boundary boundary(context);Status terminal;
    if(!mode){boundary.fail_at=1;terminal=controller.prepare_batch(layout);HYLE_RETURN_IF_ERROR(check(terminal.IsResourceExhausted() && boundary.accepted==0 && !controller.submit(),"Vulkan metadata rejection published runnable mirror"));}
    else if(mode==4){
      HYLE_ASSIGN_OR_RETURN(auto submitted,controller.submit());HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());std::array<std::uint16_t,128> words{};
      HYLE_RETURN_IF_ERROR(context.download(result->tensors()[0],std::as_writable_bytes(std::span(words))));
      for(unsigned i=0;i<words.size();++i)HYLE_RETURN_IF_ERROR(check(words[i]==hyle::simd::f32_to_f16(float(i+1)/256.f).bits,"Vulkan unit KV nonuniform value/stride oracle differs"));
      std::printf("Vulkan %s ragged query-cap=1 kv-cap=1 nonuniform 128-value independent oracle PASS\n",graph?"Graph":"Direct");
    }else{
      boundary.fail_at=mode==1?1:mode==2?2:0;auto submitted=controller.submit();
      if(mode==3){
        HYLE_RETURN_IF_ERROR(check(submitted.has_value() && boundary.accepted>0,"Vulkan unknown episode never accepted actual work"));boundary.fail_poll=true;const auto observed=submitted->poll();
        std::printf("Vulkan unknown observed=%d injected=%d active_uses=%llu\n",observed.has_value(),!boundary.fail_poll.load(),static_cast<unsigned long long>(pool.stats().active_uses));
        HYLE_RETURN_IF_ERROR(check(observed && *observed==c::CompletionPoll::kPending && !boundary.fail_poll.load() && pool.stats().active_uses==3,"Vulkan unknown observation fabricated retirement/result"));
        auto failed=submitted->wait();HYLE_RETURN_IF_ERROR(check(!failed && failed.error().IsResourceExhausted() && boundary.waits.load()>0 && pool.stats().active_uses==0,"Vulkan wait lost original error/actual fence proof"));*submitted={};
        HYLE_RETURN_IF_ERROR(check(!controller.prepare_batch(layout).ok() && !controller.submit(),"Vulkan open unknown-failed slot reused after actual retirement"));terminal=controller.drain();
        HYLE_RETURN_IF_ERROR(check(terminal.IsResourceExhausted(),"Vulkan drain replaced original observation error"));
        std::printf("Vulkan %s unknown actual-wait/open-failed-no-reuse PASS\n",graph?"Graph":"Direct");
      }else{
        if(submitted){auto result=submitted->wait();if(!result)terminal=result.error();}else terminal=submitted.error();
        HYLE_RETURN_IF_ERROR(check(terminal.IsResourceExhausted() && boundary.submits==mode && boundary.accepted==mode-1,"Vulkan rejected math prefix/suffix contract differs"));
        if(submitted){HYLE_RETURN_IF_ERROR(check(!submitted->result(),"Vulkan failed math published output"));*submitted={};}
        HYLE_RETURN_IF_ERROR(check(!controller.prepare_batch(layout).ok() && !controller.submit(),"Vulkan open rejected-math slot reused"));controller.drain().IgnoreError();
      }
    }
    controller.drain().IgnoreError();HYLE_RETURN_IF_ERROR(check(pool.stats().active_uses==0 && !controller.prepare_batch(layout).ok() && !controller.submit(),"Vulkan failed/closed controller reused or retained input guards"));
    controller={};context.close().IgnoreError();
    if(mode<4)std::printf("Vulkan %s actual fault mode=%u submits=%u accepted=%u guards=0 no-publication/no-reuse PASS\n",graph?"Graph":"Direct",mode,boundary.submits,boundary.accepted);
    else std::printf("Vulkan %s unit shape actual completion/retirement PASS\n",graph?"Graph":"Direct");
  }return check(metadata.used()==0,"Vulkan fault controller metadata leaked");
}
inline Status run(){for(bool graph:{false,true})for(unsigned mode=0;mode<5;++mode)HYLE_RETURN_IF_ERROR(episode(mode,graph));return Status::Ok();}
} // namespace hyle_f5_vk_failure_fixture
