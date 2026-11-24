#pragma once
#include "native_paged_joint_risk_fixture.h"
#include <future>
#include <thread>
#include "../compute_batch/native_apple_failure_fixture.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include <thread>

namespace hyle_joint_page_risk {
// mode0: actual closed CPU queue; mode1: native pending cancellation;
// mode2: both public owners abandoned; mode3: native accepted-prefix error.
inline Status lifecycle(unsigned mode){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto native,n::Context::create());
    HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native));
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
    if(mode==6){
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
    }
    if(!mode)HYLE_RETURN_IF_ERROR(queue.close());
    auto submitted=slot.submit(lanes);c::Completion work;Status admission;
    if(submitted)work=std::move(*submitted);
    else{admission=submitted.error();HYLE_RETURN_IF_ERROR(require(!mode && admission.IsFailedPrecondition(),"unexpected pre-acceptance Graph rejection"));}
    if(mode){
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
      if(mode!=3){
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
          gate->release();
        }
        HYLE_RETURN_IF_ERROR(require(held,"Graph drain returned or released uses before native prefix proof"));
      }
      gate->release();
    }
    if(mode==2){
      scheduler.drain();
      // The progress task has handed off its actual native prefix. Scheduler
      // idleness does not prove device retirement after both public owners go.
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
      while(source.pool.stats().active_uses && std::chrono::steady_clock::now()<deadline){
        std::this_thread::yield();
      }
    }else{
      auto terminal=work.valid()?work.wait():admission;
      const bool expected=!terminal.ok() && (!mode?terminal.IsFailedPrecondition():
          (mode==1 || mode==5)?terminal.IsCancelled():
          terminal.IsInternal());
      HYLE_RETURN_IF_ERROR(require(expected && !slot.result(shape.outputs[0]),"joint prefix failed/cancelled/unknown diagnosis or publication differs"));
      if(mode==5)HYLE_RETURN_IF_ERROR(require(closed.get().IsCancelled(),"Graph pending drain lost original cancellation terminal"));
      work={};
      if(mode)HYLE_RETURN_IF_ERROR(require(!slot.prepare_replay(lanes).ok() && !slot.submit(lanes),"failed joint slot accepted reuse"));
      slot.close().IgnoreError();slot={};data.drop_external();source.read={};
    }
    HYLE_RETURN_IF_ERROR(require(source.pool.stats().active_uses==0,"joint actual prefix query use did not retire"));
    HYLE_RETURN_IF_ERROR(r::reusable_pages(data));
    prefix={};std::printf("Apple joint lifecycle mode=%u actual native checkpoint, query and all exact guards retired PASS\n",mode);
    program={};
  }
  return require(metadata.used()==0,"joint lifecycle explicit metadata leaked");
}
} // namespace hyle_joint_page_risk
