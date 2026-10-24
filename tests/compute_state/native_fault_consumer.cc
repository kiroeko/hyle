#include <hyle/compute/state_clone.h>
#include <hyle/mem/quota_allocator.h>
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/compute/apple_memory.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/backends/cuda.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_memory.h>
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>

namespace {
namespace c=hyle::compute;namespace m=hyle::mem;
using hyle::Status;template<class T>using Result=hyle::Result<T>;
Status require(bool value,const char *message){return value?Status::Ok():Status::Internal(message);}
Result<c::StateSchema> schema(){HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{64}));return c::StateSchema::create(std::array{c::StateEntrySpec{.description=desc}});}

// The same explicit-observer boundary as the CPU StateClone regression. The
// notification happens inside original wait forwarding, after observer admission.
// No copy, publication or native retirement is replaced by this boundary.
struct CopyObservation {
  std::atomic<bool> block_wait{false},entered{false},released{false},active{false},fail_next_wait{false};
  std::atomic<unsigned> forbidden{0},waits{0},cancellations{0};
  std::function<Status()> retire_current;
  bool reject() noexcept {
    if(!active.load(std::memory_order_acquire))return false;
    forbidden.fetch_add(1,std::memory_order_relaxed);return true;
  }
  void enter() noexcept {
    waits.fetch_add(1,std::memory_order_relaxed);
    if(!block_wait.exchange(false))return;
    active.store(true,std::memory_order_release);
    entered.store(true,std::memory_order_release);entered.notify_all();
    released.wait(false,std::memory_order_acquire);
  }
  void release() noexcept {released.store(true,std::memory_order_release);released.notify_all();}
};
template<class Submission>struct ObservedSubmission {
  Submission original;CopyObservation *boundary=nullptr;
  bool valid()const noexcept{return boundary->reject()?true:original.valid();}
  Status wait()noexcept {
    boundary->enter();const auto terminal=original.wait();boundary->active.store(false,std::memory_order_release);
    if(terminal.ok()&&boundary->fail_next_wait.exchange(false))return Status::DataLoss("test original native wait observation failed");
    return terminal;
  }
  Status request_cancel()noexcept {
    boundary->cancellations.fetch_add(1,std::memory_order_relaxed);
    return boundary->reject()?Status::Unavailable("test explicit observer owns original submission"):original.request_cancel();
  }
  Result<c::ReadLease> output(c::CopyPhaseId phase,c::CopyResourceId resource,std::size_t offset,std::size_t bytes)noexcept {
    if(boundary->reject())return std::unexpected(Status::Unavailable("test explicit observer owns original output"));
    return original.output(phase,resource,offset,bytes);
  }
};
template<class Invocation>struct ObservedInvocation {
  Invocation original;CopyObservation *boundary=nullptr;
  Status bind_source(c::CopyResourceId id,const c::ReadLease &source)noexcept {
    return boundary->reject()?Status::Unavailable("test explicit observer owns original binding"):original.bind_source(id,source);
  }
  Status bind_mutable(c::CopyResourceId id,c::WriteLease &&writer)noexcept {
    if(boundary->reject())return Status::Unavailable("test explicit observer owns original binding");
    return original.bind_mutable(id,std::move(writer));
  }
  Status prepare_replay(std::span<const c::CopyPhase> phases,std::span<const c::CopyRange> ranges)noexcept {
    return boundary->reject()?Status::Unavailable("test explicit observer owns original preparation"):original.prepare_replay(phases,ranges);
  }
  auto submit()noexcept -> Result<ObservedSubmission<typename decltype(original.submit())::value_type>> {
    if(boundary->reject())return std::unexpected(Status::Unavailable("test explicit observer owns original submission"));
    HYLE_ASSIGN_OR_RETURN(auto submitted,original.submit());
    // submit runs in the route's already-reserved fixed slot. This pointer is
    // used only before cold rearm and explicitly discarded before replacement.
    boundary->retire_current=[this]()noexcept {
      const auto terminal=original.drain();
      if(original.in_flight())return Status::Unavailable("original copy retirement remains unproven");
      return terminal.IsCancelled()?Status::Ok():terminal;
    };
    return ObservedSubmission<typename decltype(original.submit())::value_type>{std::move(submitted),boundary};
  }
  Status reset()noexcept{return boundary->reject()?Status::Unavailable("test explicit observer owns reset"):original.reset();}
  Status close()noexcept{return boundary->reject()?Status::Unavailable("test explicit observer owns close"):original.close();}
  Status drain()noexcept{return boundary->reject()?Status::Unavailable("test explicit observer owns drain"):original.drain();}
  bool in_flight()const noexcept{return boundary->reject()?false:original.in_flight();}
  c::CopyBatchReport report()const noexcept{return boundary->reject()?c::CopyBatchReport{}:original.report();}
};
template<class Factory>auto observed_factory(Factory factory,CopyObservation &boundary) {
  return [factory,&boundary](const auto &program)noexcept
      -> Result<ObservedInvocation<typename decltype(factory(program))::value_type>> {
    HYLE_ASSIGN_OR_RETURN(auto invocation,factory(program));
    return ObservedInvocation<typename decltype(factory(program))::value_type>{std::move(invocation),&boundary};
  };
}

template<class Program,class Factory,class Write,class Read>
Status clone_reuse(m::MemoryResourceRef resource,const Program &program,Factory factory,
    const c::ReadLease &source,Write write,Read read,const char *name) {
  // Two terminal-control witnesses, one actual geometry. No failing GPU kernel
  // is manufactured: DataLoss is reported only after the original wait succeeds.
  for(const bool failed_observation:{false,true}) {
    CopyObservation boundary;
    HYLE_ASSIGN_OR_RETURN(auto contract,schema());HYLE_ASSIGN_OR_RETURN(auto id,contract.entry(0));
    HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(contract));
    HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{id,source}}));
    HYLE_ASSIGN_OR_RETURN(auto route,c::StateCloneRoute::prepare(store,{.resource=resource,.capacity_bytes=4096},program,observed_factory(factory,boundary)));
    HYLE_ASSIGN_OR_RETURN(auto old_draft,store.begin(base));HYLE_ASSIGN_OR_RETURN(auto old_ticket,old_draft.clone_for_write(id,route));
    if(failed_observation)boundary.fail_next_wait=true;else HYLE_RETURN_IF_ERROR(old_draft.cancel());
    auto failure=old_ticket.wait();
    HYLE_RETURN_IF_ERROR(require(!failure&&(failed_observation?failure.error().IsDataLoss():failure.error().IsCancelled()),"old native clone did not retain its terminal status"));
    HYLE_RETURN_IF_ERROR(require(bool(boundary.retire_current),"original copy retirement endpoint missing"));
    HYLE_RETURN_IF_ERROR(boundary.retire_current());boundary.retire_current={};
    HYLE_RETURN_IF_ERROR(store.collect());HYLE_RETURN_IF_ERROR(require(store.stats().active_clones==0,"old native clone not retired"));
    HYLE_RETURN_IF_ERROR(route.rearm());
    HYLE_ASSIGN_OR_RETURN(auto current,store.begin(base));HYLE_ASSIGN_OR_RETURN(auto ticket,current.clone_for_write(id,route));
    const auto waits=boundary.waits.load(),cancellations=boundary.cancellations.load();
    auto old_wait=old_ticket.wait();auto old_poll=old_ticket.poll();
    HYLE_RETURN_IF_ERROR(require(!old_wait&&!old_poll&&(failed_observation?old_wait.error().IsDataLoss():old_wait.error().IsCancelled())&&(failed_observation?old_poll.error().IsDataLoss():old_poll.error().IsCancelled()),"old control observed a different native episode"));
    HYLE_RETURN_IF_ERROR(old_ticket.request_cancel());HYLE_RETURN_IF_ERROR(old_draft.cancel());old_ticket={};old_draft={};
    HYLE_RETURN_IF_ERROR(require(boundary.waits.load()==waits&&boundary.cancellations.load()==cancellations,"old controls touched the reused native slot"));
    HYLE_RETURN_IF_ERROR(store.collect());HYLE_RETURN_IF_ERROR(require(store.stats().active_clones==1,"old controls retired the new clone"));
    HYLE_ASSIGN_OR_RETURN(auto writer,ticket.wait());HYLE_RETURN_IF_ERROR(write(writer,std::byte{47}));writer={};
    HYLE_ASSIGN_OR_RETURN(auto candidate,current.finalize());HYLE_ASSIGN_OR_RETURN(auto old_reader,base.member(id));HYLE_ASSIGN_OR_RETURN(auto new_reader,candidate.member(id));
    HYLE_RETURN_IF_ERROR(read(old_reader,std::byte{37}));HYLE_RETURN_IF_ERROR(read(new_reader,std::byte{47}));
    HYLE_ASSIGN_OR_RETURN(auto report,route.report());HYLE_RETURN_IF_ERROR(require(report.physical_bytes==256,"clone report does not describe the real 256-byte copy"));
    boundary.retire_current={};ticket={};current={};base.reset();candidate.reset();route={};store={};contract={};
    HYLE_RETURN_IF_ERROR(read(old_reader,std::byte{37}));HYLE_RETURN_IF_ERROR(read(new_reader,std::byte{47}));
    HYLE_RETURN_IF_ERROR(require(boundary.forbidden.load()==0,"native observer overlap in clone reuse"));
    std::printf("F4 %s original clone %s/rearm/success/escaped-readers PASS\n",name,failed_observation?"failed-observation":"cancelled");
  }
  return Status::Ok();
}

#if defined(HYLE_COMPUTE_CUDA)
namespace cu=c::cuda;
struct Gate {
  std::atomic<bool> open{false};
  static void CUDART_CB wait(void *opaque){auto &gate=*static_cast<Gate *>(opaque);while(!gate.open.load(std::memory_order_acquire))gate.open.wait(false,std::memory_order_acquire);}
  void release()noexcept{open.store(true,std::memory_order_release);open.notify_all();}
};
Status cuda_clone_lifetime() {
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(cu::device(0).value()));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto memory,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),4096));HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(256));
  HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{37}));HYLE_ASSIGN_OR_RETURN(auto source,writer.publish());
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));auto *endpoint=&queue;
  return clone_reuse(memory.ref(),program,[endpoint](const c::CopyBatchProgram &p)noexcept{return c::CopyBatchInvocation::create(p,*endpoint);},source,
    [](c::StateWriter &output,std::byte value)->Status{HYLE_RETURN_IF_ERROR(output.lease().view()->fill(value));return output.publish();},
    [](const c::ReadLease &value,std::byte expected)->Status{std::array<std::byte,256> bytes;HYLE_RETURN_IF_ERROR(value.view()->read(bytes));for(auto b:bytes)HYLE_RETURN_IF_ERROR(require(b==expected,"CUDA clone reader bytes differ"));return Status::Ok();},"CUDA");
}
Status cuda_pending(bool abandon,bool concurrent) {
  // The injected allocator outlives the original native endpoints and their
  // asynchronous retirement cleanup, including every failure return.
  m::QuotaAllocator metadata(m::system_allocator());
  CopyObservation observation;
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(cu::device(0).value()));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto memory,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(memory.ref(),4096));
  HYLE_ASSIGN_OR_RETURN(auto source_writer,source_pool.allocate(256));HYLE_RETURN_IF_ERROR(source_writer.view()->fill(std::byte{37}));HYLE_ASSIGN_OR_RETURN(auto source,source_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto contract,schema());HYLE_ASSIGN_OR_RETURN(auto id,contract.entry(0));
  HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(contract,{.allocator=metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{id,source}}));
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));auto *endpoint=&queue;
  HYLE_ASSIGN_OR_RETURN(auto route,c::StateCloneRoute::prepare(store,{.resource=memory.ref(),.capacity_bytes=4096},program,
      observed_factory([endpoint](const c::CopyBatchProgram &p)noexcept{return c::CopyBatchInvocation::create(p,*endpoint);},observation)));
  HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
  Gate gate;struct Release{Gate &gate;c::Context &context;~Release(){gate.release();context.drain().IgnoreError();}} release{gate,context};
  HYLE_ASSIGN_OR_RETURN(auto blocked,cu::submit(queue,[&](cudaStream_t stream)noexcept{return cudaLaunchHostFunc(stream,&Gate::wait,&gate);}));
  HYLE_ASSIGN_OR_RETURN(auto ticket,draft.clone_for_write(id,route));
  HYLE_RETURN_IF_ERROR(require(ticket.poll().error().IsUnavailable(),"pending CUDA clone became writable"));
  HYLE_RETURN_IF_ERROR(require(route.report().error().IsUnavailable(),"pending native report was read"));
  HYLE_RETURN_IF_ERROR(require(!draft.finalize(),"pending CUDA candidate finalized"));
  if(concurrent) {
    Status observed;observation.block_wait=true;
    std::thread waiter([&]{auto value=ticket.wait();if(value)observed=Status::Internal("closed candidate got writer");else observed=value.error();});
    struct Join{std::thread &thread;CopyObservation &observation;Gate &gate;~Join(){observation.release();gate.release();if(thread.joinable())thread.join();}}join{waiter,observation,gate};
    observation.entered.wait(false,std::memory_order_acquire);
    HYLE_RETURN_IF_ERROR(require(store.close().IsUnavailable(),"observed CUDA close claimed retirement"));
    HYLE_RETURN_IF_ERROR(require(store.collect().IsUnavailable()&&!route.report(),"observed CUDA clone admitted collection/report"));
    HYLE_RETURN_IF_ERROR(require(route.rearm().IsFailedPrecondition(),"closed CUDA clone admitted cold rearm"));
    HYLE_RETURN_IF_ERROR(require(observation.forbidden.load()==0,"CUDA observer overlapped native controls"));
    observation.release();gate.release();waiter.join();
    HYLE_RETURN_IF_ERROR(require(!observed.ok(),"concurrent close published CUDA clone"));
    store.drain().IgnoreError();
  } else {
    const auto cancellation=ticket.request_cancel();HYLE_RETURN_IF_ERROR(require(cancellation.ok()||cancellation.IsUnimplemented(),"CUDA native cancel contract"));
    HYLE_RETURN_IF_ERROR(require(!store.close().ok(),"pending close claimed actual retirement"));
    HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses>0,"pending source was released before CUDA retirement"));
    if(abandon){ticket={};draft={};base.reset();route={};store={};HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses>0,"abandon released accepted native source use"));}
    gate.release();HYLE_RETURN_IF_ERROR(context.drain());
    if(!abandon){auto result=ticket.wait();HYLE_RETURN_IF_ERROR(require(!result,"cancelled clone got writer"));store.drain().IgnoreError();}
  }
  HYLE_RETURN_IF_ERROR(blocked.wait());blocked={};
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(source_pool.stats().active_uses && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses==0,"CUDA clone last-use did not retire"));
  HYLE_ASSIGN_OR_RETURN(auto view,source.view());std::array<std::byte,256> bytes;HYLE_RETURN_IF_ERROR(view.read(bytes));
  for(auto value:bytes)HYLE_RETURN_IF_ERROR(require(value==std::byte{37},"CUDA cancelled/abandoned clone modified source"));
  if(!abandon){ticket={};draft={};base.reset();route={};store={};}
  // LeaseUse retirement is observable before the original CUDA reaper drops
  // the final native descriptor/pool metadata. Observe that cleanup explicitly.
  const auto cleanup_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(metadata.used()&&std::chrono::steady_clock::now()<cleanup_deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  HYLE_RETURN_IF_ERROR(require(metadata.used()==0,"F4 store metadata retained after controls/actual copy retirement"));
  std::printf("F4 CUDA native pending %s/%s PASS\n",abandon?"abandon":"cancel",concurrent?"concurrent-close":"serial-close");return Status::Ok();
}
#endif
#if defined(HYLE_COMPUTE_VULKAN)
namespace v=c::vulkan;namespace vd=v::detail;
Status vulkan_clone_lifetime() {
  HYLE_ASSIGN_OR_RETURN(auto context,v::Context::create());HYLE_RETURN_IF_ERROR(require(!context.environment().software_device,"Vulkan clone requires actual GPU"));
  HYLE_ASSIGN_OR_RETURN(auto memory,v::MemoryResource::create(context,v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,v::MemoryResource::create(context,v::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{64}));
  std::array<std::byte,256> initial;initial.fill(std::byte{37});HYLE_ASSIGN_OR_RETURN(auto tensor,context.upload(desc,initial));
  HYLE_ASSIGN_OR_RETURN(auto source,memory.import_readonly(tensor));
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,v::CopyBatchProgram::prepare(context,specs));
  return clone_reuse(memory.ref(),program,[](const v::CopyBatchProgram &p)noexcept{return v::CopyBatchInvocation::create(p);},source,
    [&](c::StateWriter &output,std::byte pattern)->Status{
      HYLE_ASSIGN_OR_RETURN(auto buffer,m::Buffer::create(staging.ref(),256,256));HYLE_RETURN_IF_ERROR(buffer.view().fill(pattern));HYLE_ASSIGN_OR_RETURN(auto host,c::ReadLease::adopt(std::move(buffer)));
      const std::array transfer_specs{c::CopyResourceSpec{staging.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
      HYLE_ASSIGN_OR_RETURN(auto transfer,v::CopyBatchProgram::prepare(context,transfer_specs));HYLE_ASSIGN_OR_RETURN(auto invocation,v::CopyBatchInvocation::create(transfer));
      HYLE_RETURN_IF_ERROR(invocation.bind_source({0},host));HYLE_RETURN_IF_ERROR(invocation.bind_mutable({1},std::move(output.lease())));
      const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,256}}};
      HYLE_RETURN_IF_ERROR(invocation.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,invocation.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_ASSIGN_OR_RETURN(auto ready,submitted.output({0},{1},0,256));submitted={};HYLE_RETURN_IF_ERROR(invocation.reset());return output.complete(ready);
    },
    [&](const c::ReadLease &value,std::byte expected)->Status{HYLE_ASSIGN_OR_RETURN(auto view,memory.binding(value,desc));std::array<std::byte,256> bytes;HYLE_RETURN_IF_ERROR(context.download(view,bytes));for(auto b:bytes)HYLE_RETURN_IF_ERROR(require(b==expected,"Vulkan clone reader bytes differ"));return Status::Ok();},"Vulkan");
}
struct UnknownObservation {
  vd::Device &device;vd::DeviceFunctions original;std::atomic<bool> armed{true};
  static inline UnknownObservation *active=nullptr;
  explicit UnknownObservation(const v::Context &context):device(*vd::Access::runtime(context)->device),original(device.fn){active=this;device.fn.GetFenceStatus=poll;device.fn.WaitForFences=wait;device.fn.QueueWaitIdle=drain;}
  ~UnknownObservation(){armed=false;device.fn=original;active=nullptr;}
  static VKAPI_ATTR VkResult VKAPI_CALL poll(VkDevice device,VkFence fence){return active->armed?VK_ERROR_OUT_OF_HOST_MEMORY:active->original.GetFenceStatus(device,fence);}
  static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice device,std::uint32_t n,const VkFence *f,VkBool32 all,std::uint64_t timeout){return active->armed?VK_ERROR_OUT_OF_HOST_MEMORY:active->original.WaitForFences(device,n,f,all,timeout);}
  static VKAPI_ATTR VkResult VKAPI_CALL drain(VkQueue queue){return active->armed?VK_ERROR_OUT_OF_HOST_MEMORY:active->original.QueueWaitIdle(queue);}
};
Status vulkan_unknown(bool abandon) {
  HYLE_ASSIGN_OR_RETURN(auto context,v::Context::create());HYLE_RETURN_IF_ERROR(require(!context.environment().software_device,"Vulkan fault requires actual GPU"));
  HYLE_ASSIGN_OR_RETURN(auto memory,v::MemoryResource::create(context,v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto contract,schema());HYLE_ASSIGN_OR_RETURN(auto id,contract.entry(0));HYLE_ASSIGN_OR_RETURN(auto desc,contract.specification(id));
  std::array<float,64> values;values.fill(3.5f);HYLE_ASSIGN_OR_RETURN(auto tensor,context.upload(desc.description,std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto source,memory.import_readonly(tensor));HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(contract));
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{id,source}}));
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,v::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto route,c::StateCloneRoute::prepare(store,{.resource=memory.ref(),.capacity_bytes=4096},program,
      [](const v::CopyBatchProgram &p)noexcept{return v::CopyBatchInvocation::create(p);}));
  HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));UnknownObservation boundary(context);
  struct Restore{UnknownObservation &boundary;v::Context &context;~Restore(){boundary.armed=false;context.close().IgnoreError();}} restore{boundary,context};
  HYLE_ASSIGN_OR_RETURN(auto ticket,draft.clone_for_write(id,route));
  auto first=ticket.poll();HYLE_RETURN_IF_ERROR(require(!first && first.error().IsUnavailable(),"unknown Vulkan copy returned a writer"));
  HYLE_RETURN_IF_ERROR(require(!draft.finalize(),"unknown Vulkan candidate published"));
  ticket.request_cancel().IgnoreError();HYLE_RETURN_IF_ERROR(require(route.rearm().IsUnavailable(),"unproven Vulkan retirement admitted cold rearm"));
  HYLE_RETURN_IF_ERROR(require(!store.close().ok(),"unknown Vulkan close claimed retirement"));
  HYLE_RETURN_IF_ERROR(require(store.stats().active_clones>0,"unknown Vulkan clone record was reclaimed"));
  HYLE_RETURN_IF_ERROR(require(route.rearm().IsFailedPrecondition(),"closed Vulkan route admitted cold rearm"));
  if(abandon){ticket={};draft={};base.reset();route={};store={};}
  boundary.armed=false;
  if(!abandon){ticket.wait().error().IgnoreError();store.drain().IgnoreError();ticket={};draft={};base.reset();route={};store={};}
  std::array<float,64> output;HYLE_RETURN_IF_ERROR(context.download(tensor,std::as_writable_bytes(std::span(output))));
  HYLE_RETURN_IF_ERROR(require(output==values,"unknown/abandoned Vulkan clone changed source"));
  std::printf("F4 Vulkan unknown actual retirement/%s PASS\n",abandon?"abandon":"cancel");return Status::Ok();
}
#endif
#if defined(HYLE_COMPUTE_APPLE)
namespace a=c::apple;namespace mx=mlx::core;
Status apple_clone_lifetime() {
  HYLE_ASSIGN_OR_RETURN(auto context,a::Context::create());HYLE_ASSIGN_OR_RETURN(auto memory,a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),4096));HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(256));
  HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{37}));HYLE_ASSIGN_OR_RETURN(auto source,writer.publish());
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,a::CopyBatchProgram::prepare(context,specs));
  return clone_reuse(memory.ref(),program,[](const a::CopyBatchProgram &p)noexcept{return a::CopyBatchInvocation::create(p);},source,
    [](c::StateWriter &output,std::byte value)->Status{HYLE_RETURN_IF_ERROR(output.lease().view()->fill(value));return output.publish();},
    [](const c::ReadLease &value,std::byte expected)->Status{std::array<std::byte,256> bytes;HYLE_RETURN_IF_ERROR(value.view()->read(bytes));for(auto b:bytes)HYLE_RETURN_IF_ERROR(require(b==expected,"Apple clone reader bytes differ"));return Status::Ok();},"Apple");
}
struct AppleGate {
  std::mutex mutex;std::condition_variable condition;
  bool armed=false,entered=false,released=false,fail=false;
  void wait(){std::unique_lock lock(mutex);if(!armed)return;entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});if(fail)throw std::runtime_error("F4 original MLX prefix failure");}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
class AppleCheckpoint final:public mx::Primitive {
public:
  AppleCheckpoint(mx::Stream stream,std::shared_ptr<AppleGate> gate):Primitive(stream),gate_(std::move(gate)){}
  void eval_cpu(const std::vector<mx::array>&,std::vector<mx::array>&)override{throw std::runtime_error("F4 checkpoint requires native GPU");}
  void eval_gpu(const std::vector<mx::array>&inputs,std::vector<mx::array>&outputs)override{gate_->wait();outputs[0].copy_shared_buffer(inputs[0]);}
  const char *name()const override{return "HyleF4NativeCheckpoint";}
private:std::shared_ptr<AppleGate> gate_;
};
Status apple_pending(bool abandon,bool concurrent,bool fail) {
  CopyObservation observation;
  HYLE_ASSIGN_OR_RETURN(auto context,a::Context::create());HYLE_ASSIGN_OR_RETURN(auto memory,a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(memory.ref(),4096));
  HYLE_ASSIGN_OR_RETURN(auto source_writer,source_pool.allocate(256));HYLE_RETURN_IF_ERROR(source_writer.view()->fill(std::byte{37}));HYLE_ASSIGN_OR_RETURN(auto source,source_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto contract,schema());HYLE_ASSIGN_OR_RETURN(auto id,contract.entry(0));
  m::QuotaAllocator metadata(m::system_allocator());HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(contract,{.allocator=metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{id,source}}));
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,a::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto route,c::StateCloneRoute::prepare(store,{.resource=memory.ref(),.capacity_bytes=4096},program,
      observed_factory([](const a::CopyBatchProgram&p)noexcept{return a::CopyBatchInvocation::create(p);},observation)));
  HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
  auto gate=std::make_shared<AppleGate>();struct Release{std::shared_ptr<AppleGate> gate;a::Context&context;~Release(){gate->release();context.close().IgnoreError();}}release{gate,context};
  const std::array inputs{a::InputSpec{{4},mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto&args,mx::Stream stream){auto value=mx::square(args[0],stream);return std::vector{mx::array(value.shape(),value.dtype(),std::make_shared<AppleCheckpoint>(stream,gate),{value})};},inputs));
  HYLE_ASSIGN_OR_RETURN(auto invocation,blocker.create_invocation());const std::array values{mx::array({1.f,2.f,3.f,4.f})};HYLE_RETURN_IF_ERROR(invocation.bind(values));
  {std::lock_guard lock(gate->mutex);gate->armed=true;gate->fail=fail;}
  HYLE_ASSIGN_OR_RETURN(auto blocked,invocation.submit());
  {std::unique_lock lock(gate->mutex);HYLE_RETURN_IF_ERROR(require(gate->condition.wait_for(lock,std::chrono::seconds(3),[&]{return gate->entered;}),"MLX native gate not entered"));}
  HYLE_ASSIGN_OR_RETURN(auto ticket,draft.clone_for_write(id,route));
  HYLE_RETURN_IF_ERROR(require(ticket.poll().error().IsUnavailable(),"pending Apple clone became writable"));
  HYLE_RETURN_IF_ERROR(require(route.report().error().IsUnavailable(),"pending Apple report was read"));
  HYLE_RETURN_IF_ERROR(require(!draft.finalize(),"pending Apple candidate finalized"));
  HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses>0,"Apple accepted source use missing"));
  if(concurrent){
    Status observed;observation.block_wait=true;std::thread waiter([&]{auto result=ticket.wait();if(result)observed=Status::Internal("closed Apple candidate got writer");else observed=result.error();});
    struct Join{std::thread &thread;CopyObservation &observation;std::shared_ptr<AppleGate> gate;~Join(){observation.release();gate->release();if(thread.joinable())thread.join();}}join{waiter,observation,gate};
    observation.entered.wait(false,std::memory_order_acquire);
    HYLE_RETURN_IF_ERROR(require(store.close().IsUnavailable(),"observed Apple close claimed retirement"));
    HYLE_RETURN_IF_ERROR(require(store.collect().IsUnavailable()&&!route.report(),"observed Apple clone admitted collection/report"));
    HYLE_RETURN_IF_ERROR(require(route.rearm().IsFailedPrecondition(),"closed Apple clone admitted cold rearm"));
    HYLE_RETURN_IF_ERROR(require(observation.forbidden.load()==0,"Apple observer overlapped native controls"));
    observation.release();gate->release();waiter.join();
    HYLE_RETURN_IF_ERROR(require(!observed.ok(),"concurrent Apple close published clone"));store.drain().IgnoreError();
  }else{
    if(!fail)HYLE_RETURN_IF_ERROR(ticket.request_cancel());
    if(!fail)HYLE_RETURN_IF_ERROR(require(!store.close().ok(),"pending Apple close claimed retirement"));
    if(abandon){ticket={};draft={};base.reset();route={};store={};HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses>0,"Apple abandon released accepted native use"));}
    gate->release();auto prefix=blocked.wait();
    if(fail)HYLE_RETURN_IF_ERROR(require(!prefix&&prefix.error().IsInternal(),"original MLX prefix failure missing"));else HYLE_RETURN_IF_ERROR(require(bool(prefix),"original MLX prefix failed"));
    if(!abandon){auto result=ticket.wait();HYLE_RETURN_IF_ERROR(require(!result,"cancelled/failed Apple clone got writer"));if(fail)HYLE_RETURN_IF_ERROR(require(result.error().IsInternal(),"Apple prefix error was not propagated"));HYLE_RETURN_IF_ERROR(require(!draft.finalize(),"failed Apple draft finalized"));store.drain().IgnoreError();}
  }
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(source_pool.stats().active_uses&&std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses==0,"Apple clone last use did not retire"));
  HYLE_ASSIGN_OR_RETURN(auto view,source.view());std::array<std::byte,256> bytes;HYLE_RETURN_IF_ERROR(view.read(bytes));for(auto value:bytes)HYLE_RETURN_IF_ERROR(require(value==std::byte{37},"Apple failed clone changed source"));
  if(!abandon){ticket={};draft={};base.reset();route={};store={};}
  // The native owner may still hold a retired Job after its LeaseUse callback.
  // Join that original owner explicitly before judging allocator balance.
  context.close().IgnoreError();
  HYLE_RETURN_IF_ERROR(require(metadata.used()==0,"Apple F4 metadata not released"));
  std::printf("F4 Apple native pending %s/%s/%s PASS\n",abandon?"abandon":"cancel",concurrent?"concurrent-close":"serial-close",fail?"prefix-failure":"success-prefix");return Status::Ok();
}
#endif
Status run() {
#if defined(HYLE_COMPUTE_CUDA)
  HYLE_RETURN_IF_ERROR(cuda_clone_lifetime());
  HYLE_RETURN_IF_ERROR(cuda_pending(false,false));HYLE_RETURN_IF_ERROR(cuda_pending(true,false));HYLE_RETURN_IF_ERROR(cuda_pending(false,true));
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_RETURN_IF_ERROR(vulkan_clone_lifetime());
  HYLE_RETURN_IF_ERROR(vulkan_unknown(false));HYLE_RETURN_IF_ERROR(vulkan_unknown(true));
#endif
#if defined(HYLE_COMPUTE_APPLE)
  HYLE_RETURN_IF_ERROR(apple_clone_lifetime());
  HYLE_RETURN_IF_ERROR(apple_pending(false,false,false));HYLE_RETURN_IF_ERROR(apple_pending(true,false,false));HYLE_RETURN_IF_ERROR(apple_pending(false,true,false));HYLE_RETURN_IF_ERROR(apple_pending(false,false,true));
#endif
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F4 native fault: %s\n",text);return 1;}
