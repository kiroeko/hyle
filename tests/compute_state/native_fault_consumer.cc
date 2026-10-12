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
#include <thread>

namespace {
namespace c=hyle::compute;namespace m=hyle::mem;
using hyle::Status;template<class T>using Result=hyle::Result<T>;
Status require(bool value,const char *message){return value?Status::Ok():Status::Internal(message);}
Result<c::StateSchema> schema(){HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{64}));return c::StateSchema::create(std::array{c::StateEntrySpec{.description=desc}});}

#if defined(HYLE_COMPUTE_CUDA)
namespace cu=c::cuda;
struct Gate {
  std::atomic<bool> open{false};
  static void CUDART_CB wait(void *opaque){auto &gate=*static_cast<Gate *>(opaque);while(!gate.open.load(std::memory_order_acquire))gate.open.wait(false,std::memory_order_acquire);}
  void release()noexcept{open.store(true,std::memory_order_release);open.notify_all();}
};
Status cuda_pending(bool abandon,bool concurrent) {
  // The injected allocator outlives the original native endpoints and their
  // asynchronous retirement cleanup, including every failure return.
  m::QuotaAllocator metadata(m::system_allocator());
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
      [endpoint](const c::CopyBatchProgram &p)noexcept{return c::CopyBatchInvocation::create(p,*endpoint);}));
  HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
  Gate gate;struct Release{Gate &gate;c::Context &context;~Release(){gate.release();context.drain().IgnoreError();}} release{gate,context};
  HYLE_ASSIGN_OR_RETURN(auto blocked,cu::submit(queue,[&](cudaStream_t stream)noexcept{return cudaLaunchHostFunc(stream,&Gate::wait,&gate);}));
  HYLE_ASSIGN_OR_RETURN(auto ticket,draft.clone_for_write(id,route));
  HYLE_RETURN_IF_ERROR(require(ticket.poll().error().IsUnavailable(),"pending CUDA clone became writable"));
  HYLE_RETURN_IF_ERROR(require(route.report().error().IsUnavailable(),"pending native report was read"));
  HYLE_RETURN_IF_ERROR(require(!draft.finalize(),"pending CUDA candidate finalized"));
  if(concurrent) {
    Status observed;std::atomic<bool> entered{false};
    std::thread waiter([&]{entered=true;auto value=ticket.wait();if(value)observed=Status::Internal("closed candidate got writer");else observed=value.error();});
    while(!entered.load())std::this_thread::yield();
    store.close().IgnoreError();gate.release();waiter.join();
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
  HYLE_ASSIGN_OR_RETURN(auto context,a::Context::create());HYLE_ASSIGN_OR_RETURN(auto memory,a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(memory.ref(),4096));
  HYLE_ASSIGN_OR_RETURN(auto source_writer,source_pool.allocate(256));HYLE_RETURN_IF_ERROR(source_writer.view()->fill(std::byte{37}));HYLE_ASSIGN_OR_RETURN(auto source,source_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto contract,schema());HYLE_ASSIGN_OR_RETURN(auto id,contract.entry(0));
  m::QuotaAllocator metadata(m::system_allocator());HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(contract,{.allocator=metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{id,source}}));
  const std::array specs{c::CopyResourceSpec{memory.ref(),256},c::CopyResourceSpec{memory.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto program,a::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto route,c::StateCloneRoute::prepare(store,{.resource=memory.ref(),.capacity_bytes=4096},program,
      [](const a::CopyBatchProgram&p)noexcept{return a::CopyBatchInvocation::create(p);}));
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
    Status observed;std::atomic<bool> entered{false};std::thread waiter([&]{entered=true;auto result=ticket.wait();if(result)observed=Status::Internal("closed Apple candidate got writer");else observed=result.error();});
    while(!entered.load())std::this_thread::yield();store.close().IgnoreError();gate->release();waiter.join();
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
  HYLE_RETURN_IF_ERROR(cuda_pending(false,false));HYLE_RETURN_IF_ERROR(cuda_pending(true,false));HYLE_RETURN_IF_ERROR(cuda_pending(false,true));
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_RETURN_IF_ERROR(vulkan_unknown(false));HYLE_RETURN_IF_ERROR(vulkan_unknown(true));
#endif
#if defined(HYLE_COMPUTE_APPLE)
  HYLE_RETURN_IF_ERROR(apple_pending(false,false,false));HYLE_RETURN_IF_ERROR(apple_pending(true,false,false));HYLE_RETURN_IF_ERROR(apple_pending(false,true,false));HYLE_RETURN_IF_ERROR(apple_pending(false,false,true));
#endif
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F4 native fault: %s\n",text);return 1;}
