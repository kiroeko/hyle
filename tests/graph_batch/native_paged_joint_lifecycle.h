#pragma once
#include "native_paged_joint_risk_fixture.h"
#include <future>
#include <thread>
#if defined(HYLE_JOINT_PAGE_APPLE)
#include "../compute_batch/native_apple_failure_fixture.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include <thread>
#else
#include "../compute_batch/native_vulkan_failure_fixture.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#endif

namespace hyle_joint_page_risk {
#if !defined(HYLE_JOINT_PAGE_APPLE)
struct NativeGate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
struct JointBoundary final {
  hyle_f5_vk_failure_fixture::Boundary boundary;
  NativeGate gate;
  unsigned mode;
  std::atomic<unsigned> math_polls{0},unknowns{0};
  std::atomic<VkFence> math_fence{VK_NULL_HANDLE};
  static inline JointBoundary* active=nullptr;
  JointBoundary(n::Context& context,unsigned selected):boundary(context),mode(selected){
    active=this;
    boundary.fail_at=mode==3?2:0; // Real CPU/upload prefix, then page admission refusal.
    boundary.device.fn.QueueSubmit=[](VkQueue queue,std::uint32_t count,const VkSubmitInfo* info,VkFence fence)->VkResult{
      auto& self=*active;auto& base=self.boundary;
      if(++base.submits==base.fail_at)return VK_ERROR_OUT_OF_HOST_MEMORY;
      const auto status=base.original.QueueSubmit(queue,count,info,fence);base.accepted+=status==VK_SUCCESS;
      if(self.mode==4)std::fprintf(stderr,"unknown native submit=%u accepted=%d nonnull_fence=%d\n",
          base.submits,status==VK_SUCCESS,fence!=VK_NULL_HANDLE);
      // A native compute submit can signal a semaphore and carry no fence;
      // its following transfer submit carries the real completion proof.
      if(base.submits>=2 && status==VK_SUCCESS && fence!=VK_NULL_HANDLE){
        auto empty=VkFence{VK_NULL_HANDLE};self.math_fence.compare_exchange_strong(empty,fence);
      }
      // The page math really reached the original driver. Stop only its host
      // admission return, so the public parent still owns all actual uses.
      if(base.submits==2 && status==VK_SUCCESS && (self.mode==1 || self.mode==2 || self.mode==5))self.gate.wait();
      return status;
    };
    boundary.device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{
      auto& self=*active;auto& base=self.boundary;++base.polls;
      if(self.mode==4 && fence==self.math_fence.load(std::memory_order_acquire)){
        const auto observed=++self.math_polls;
        if(observed==1){++self.unknowns;return VK_ERROR_OUT_OF_HOST_MEMORY;}
        if(observed==2)self.gate.wait(); // No later actual fence proof yet.
      }
      return base.original.GetFenceStatus(device,fence);
    };
  }
  ~JointBoundary(){active=nullptr;}
};
#endif
// mode0: actual closed CPU queue; mode1: native pending cancellation;
// mode2: both public owners abandoned; mode3: native accepted-prefix error;
// Vulkan mode4: unknown observation held until a later real fence proof.
inline Status lifecycle(unsigned mode){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto native,n::Context::create());
#if defined(HYLE_JOINT_PAGE_APPLE)
    HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native));
#else
    HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native,n::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native,n::MemoryKind::kStaging));
    HYLE_RETURN_IF_ERROR(require(!native.environment().software_device,"joint risk requires actual Vulkan GPU"));
#endif
    HYLE_ASSIGN_OR_RETURN(auto data,r::inputs(native,device,staging));HYLE_ASSIGN_OR_RETURN(auto shape,graph(data));
    HYLE_ASSIGN_OR_RETURN(auto env,environment(cpu,native,device,staging));
    HYLE_ASSIGN_OR_RETURN(auto program,compile(shape,env,metadata.get()));
    hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
    hyle::sched::BlockingPool workers(hyle::mem::system_allocator(),1);
    Drain drain{scheduler};
    HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());HYLE_ASSIGN_OR_RETURN(auto source,query(shape));
    HYLE_ASSIGN_OR_RETURN(auto lanes,queues(program.report,queue));HYLE_ASSIGN_OR_RETURN(auto outputs,output_pools(shape));
    HYLE_ASSIGN_OR_RETURN(auto slot,create(program,scheduler,workers,metadata.get()));
    HYLE_RETURN_IF_ERROR(prepare(slot,lanes,shape,data,source,outputs,metadata.get()));
    std::future<Status> closed;
#if defined(HYLE_JOINT_PAGE_APPLE)
    auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
    struct Release {decltype(gate) held;n::Context& native;~Release(){held->release();native.close().IgnoreError();}} release{gate,native};
    n::Invocation blocking;hyle::compute::OwnedResult<n::Results> prefix;
    if(mode){
      namespace mx=mlx::core;
      const std::array specs{n::InputSpec{{4},mx::float32}};
      HYLE_ASSIGN_OR_RETURN(auto blocker,native.compile([gate](const auto& inputs,mx::Stream stream){
        auto first=mx::square(inputs[0],stream);
        return std::vector{mx::array(first.shape(),first.dtype(),std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{first})};
      },specs));
      HYLE_ASSIGN_OR_RETURN(blocking,blocker.create_invocation());
      const std::array values{mx::array({1.F,2.F,3.F,4.F})};HYLE_RETURN_IF_ERROR(blocking.bind(values));
      {std::lock_guard lock(gate->mutex);gate->armed=true;gate->fail=mode==3 || mode==6;}
      HYLE_ASSIGN_OR_RETURN(prefix,blocking.submit());HYLE_RETURN_IF_ERROR(require(gate->await(),"joint original native checkpoint not reached"));
    }
#else
    JointBoundary boundary(native,mode);
    struct Release {JointBoundary& boundary;n::Context& native;std::future<Status> observation;
      ~Release(){boundary.gate.release();if(observation.valid())observation.wait();native.close().IgnoreError();}} release{boundary,native,{}};
#endif
    if(mode==6){
#if defined(HYLE_JOINT_PAGE_APPLE)
      const std::array layouts{g::batch::NodeLayout{shape.layouts[0].node,data.next}};
      std::promise<void> started;auto entered=started.get_future();
      auto pending=std::async(std::launch::async,[&]{started.set_value();return g::batch::prepare(slot,lanes,layouts,{.allocator=metadata.get()});});entered.get();
      const bool blocked=pending.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout;
      gate->release();const auto terminal=pending.get();
      HYLE_RETURN_IF_ERROR(require(blocked && terminal.IsInternal() && !slot.submit(lanes) && !slot.prepare_replay(lanes).ok() &&
          source.pool.stats().active_uses==0,"joint failed metadata preparation admitted math/input uses"));
      slot.close().IgnoreError();slot={};data.drop_external();source.read={};program={};prefix={};
      HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
      std::printf("Apple joint queued metadata preparation waited, original prefix error, no page admission PASS\n");
      return Status::Ok();
#else
      return Status::InvalidArgument("metadata queued checkpoint is an Apple fixture");
#endif
    }
    if(!mode)HYLE_RETURN_IF_ERROR(queue.close());
    auto submitted=slot.submit(lanes);c::Completion work;Status admission;
    if(submitted)work=std::move(*submitted);
    else{admission=submitted.error();HYLE_RETURN_IF_ERROR(require(!mode && admission.IsFailedPrecondition(),"unexpected pre-acceptance Graph rejection"));}
    if(mode){
#if defined(HYLE_JOINT_PAGE_APPLE)
      // The external checkpoint alone does not prove the joint native job
      // was admitted. Observe its real original FIFO under the queue lock;
      // all setup/table copies already finished before the blocker entered.
      const auto& runtime=n::detail::ContextAccess::runtime(native);
      bool admitted=false;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
      while(!admitted && std::chrono::steady_clock::now()<deadline){
        {std::lock_guard lock(runtime->queue_mutex);admitted=runtime->head!=nullptr;}
        if(!admitted)std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      HYLE_RETURN_IF_ERROR(require(admitted,"joint original FIFO has no actual admitted native job"));
#else
      if(mode==4){
        // Common progress hands off the real prefix; it need not keep polling
        // afterwards. Collect only its original Runtime owners to exercise the
        // page-prefix completion fence, independently of later copy fences.
        release.observation=std::async(std::launch::async,[&]{
          const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
          while(boundary.math_polls.load()<2 && std::chrono::steady_clock::now()<deadline){
            native.collect().IgnoreError();std::this_thread::yield();
          }
          return require(boundary.math_polls.load()>=2,"original page fence was not observed twice");
        });
      }
      if(mode!=3)HYLE_RETURN_IF_ERROR(require(boundary.gate.await(),"joint actual page admission/unknown boundary not reached"));
#endif
      if(mode!=3){
#if !defined(HYLE_JOINT_PAGE_APPLE)
        if(mode==4)HYLE_RETURN_IF_ERROR(require(release.observation.wait_for(std::chrono::milliseconds(0))==std::future_status::timeout,
            "unknown page observer obtained a premature native proof"));
        else
#endif
        {HYLE_ASSIGN_OR_RETURN(auto observed,work.poll());
          HYLE_RETURN_IF_ERROR(require(observed==c::CompletionPoll::kPending,"actual native prefix was not pending"));}
        HYLE_RETURN_IF_ERROR(require(!slot.result(shape.outputs[0]) && source.pool.stats().active_uses==1,
            "joint pending/unknown released query use or fabricated output"));
        {HYLE_ASSIGN_OR_RETURN(auto unreferenced,data.keys.acquire(1));}
        HYLE_RETURN_IF_ERROR(require(!data.keys.acquire(0) && !data.keys.acquire(2) && !data.values.acquire(0) && !data.values.acquire(2),
            "joint pending exact page guards became writable"));
      }
      if(mode==1)HYLE_RETURN_IF_ERROR(work.request_cancel());
      if(mode==2){
        work={};slot={};data.drop_external();source.read={};
        HYLE_RETURN_IF_ERROR(require(source.pool.stats().active_uses==1 && !data.keys.acquire(0) && !data.keys.acquire(2),
            "joint abandoned public handles released actual prefix permissions"));
      }
      if(mode==5){
        HYLE_RETURN_IF_ERROR(work.request_cancel());
        std::promise<void> started;auto entered=started.get_future();
        closed=std::async(std::launch::async,[&slot,started=std::move(started)]()mutable{started.set_value();return slot.drain();});entered.get();
        const bool held=closed.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout && source.pool.stats().active_uses==1;
        if(!held){
#if defined(HYLE_JOINT_PAGE_APPLE)
          gate->release();
#else
          boundary.gate.release();
#endif
        }
        HYLE_RETURN_IF_ERROR(require(held,"Graph drain returned or released uses before native prefix proof"));
      }
#if defined(HYLE_JOINT_PAGE_APPLE)
      gate->release();
#else
      boundary.gate.release();
      if(mode==4)HYLE_RETURN_IF_ERROR(release.observation.get());
#endif
    }
    if(mode==2){
      scheduler.drain();
      // The progress task has handed off its actual native prefix. Scheduler
      // idleness does not prove device retirement after both public owners go.
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
      while(source.pool.stats().active_uses && std::chrono::steady_clock::now()<deadline){
#if !defined(HYLE_JOINT_PAGE_APPLE)
        HYLE_RETURN_IF_ERROR(native.collect());
#endif
        std::this_thread::yield();
      }
    }else{
      auto terminal=work.valid()?work.wait():admission;
      const bool expected=!terminal.ok() && (!mode?terminal.IsFailedPrecondition():
          (mode==1 || mode==5)?terminal.IsCancelled():
#if defined(HYLE_JOINT_PAGE_APPLE)
          terminal.IsInternal());
#else
          terminal.IsResourceExhausted());
#endif
      HYLE_RETURN_IF_ERROR(require(expected && !slot.result(shape.outputs[0]),"joint prefix failed/cancelled/unknown diagnosis or publication differs"));
      if(mode==5)HYLE_RETURN_IF_ERROR(require(closed.get().IsCancelled(),"Graph pending drain lost original cancellation terminal"));
      work={};
      if(mode)HYLE_RETURN_IF_ERROR(require(!slot.prepare_replay(lanes).ok() && !slot.submit(lanes),"failed joint slot accepted reuse"));
      slot.close().IgnoreError();slot={};data.drop_external();source.read={};
    }
    HYLE_RETURN_IF_ERROR(require(source.pool.stats().active_uses==0,"joint actual prefix query use did not retire"));
    HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
#if !defined(HYLE_JOINT_PAGE_APPLE)
    if(mode==3)HYLE_RETURN_IF_ERROR(require(boundary.boundary.submits==2 && boundary.boundary.accepted==1,
        "joint native rejection lacked real completed upload prefix"));
    if(mode==4)HYLE_RETURN_IF_ERROR(require(boundary.unknowns==1 && boundary.math_polls>=2 &&
        boundary.math_fence.load()!=VK_NULL_HANDLE && boundary.boundary.accepted>=2 &&
        boundary.boundary.accepted==boundary.boundary.submits,
        "joint unknown lacked actual page work and later original fence proof"));
    std::printf("Vulkan joint lifecycle mode=%u submits=%u accepted=%u unknowns=%u mathpolls=%u actual guards retired PASS\n",
        mode,boundary.boundary.submits,boundary.boundary.accepted,boundary.unknowns.load(),boundary.math_polls.load());
#else
    prefix={};std::printf("Apple joint lifecycle mode=%u actual native checkpoint, query and all exact guards retired PASS\n",mode);
#endif
    program={};
  }
  return require(metadata.used()==0,"joint lifecycle explicit metadata leaked");
}
} // namespace hyle_joint_page_risk
