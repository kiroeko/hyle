#include <hyle/compute/batch_execution.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/compute/detail/completion_access.h>
#include <hyle/core/assert.h>
#include "../../src/compute/context_state.h"
#include "../../src/compute/completion_internal.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>

namespace {
namespace c=hyle::compute;namespace cd=c::detail;namespace cu=c::cuda;
using hyle::Status;
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Gate final {
  cudaStream_t stream=nullptr;std::atomic<bool> released{false};
  static void CUDART_CB wait(void*p){auto&gate=*static_cast<Gate*>(p);gate.released.wait(false,std::memory_order_acquire);}
  void release(){released.store(true,std::memory_order_release);released.notify_all();}
  ~Gate(){release();if(stream)cudaStreamSynchronize(stream);}
};
struct UnknownObservation final {c::Completion native;cd::CompletionRetention retention{};cd::CompletionLifetime forwarded{};};
const cd::CompletionVTable observation{
  .destroy=[](void*p,cd::CompletionLifetime lifetime,Status hint)noexcept{
    auto*state=static_cast<UnknownObservation*>(p);state->forwarded=lifetime;
    state->retention.own={state,[](void*p,const Status&terminal)noexcept{const auto&forwarded=static_cast<UnknownObservation*>(p)->forwarded;if(forwarded.valid())forwarded.terminal(forwarded.state,terminal);},[](void*p)noexcept{auto*state=static_cast<UnknownObservation*>(p);const auto forwarded=state->forwarded;if(forwarded.valid())forwarded.release(forwarded.state);delete state;}};
    auto native=std::move(state->native);cd::CompletionAccess::append_terminal(native,std::move(hint));HYLE_FATAL_ASSERT(cd::CompletionAccess::attach_retention(native,state->retention).ok());
  },
  .poll=[](void*)noexcept->hyle::Result<c::CompletionPoll>{return std::unexpected(Status::Unavailable("F5 test observation lacks retirement proof"));},
  .wait=[](void*)noexcept{return Status::Unavailable("F5 test observation lacks retirement proof");}
};
struct Decorator final {
  cd::ContextState*context;const cd::BackendVTable*original;cd::BackendVTable table;bool armed=false;
  static inline Decorator*active=nullptr;
  explicit Decorator(c::Context&owner):context(cd::ContextAccess::state(owner)),original(context->backend_vtable),table(*original){
    table.finish_sequence=[](void*q,void*sequence,bool accepted,Status terminal)noexcept{auto&self=*active;auto completed=self.original->finish_sequence(q,sequence,accepted,std::move(terminal));if(!self.armed)return completed;self.armed=false;auto*state=new UnknownObservation{std::move(completed)};return cd::CompletionAccess::make(&observation,state);};active=this;context->backend_vtable=&table;
  }
  ~Decorator(){context->backend_vtable=original;active=nullptr;}
};
Status run(){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto device_id,cu::device(0));HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id,c::flashinfer::provider()));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());HYLE_ASSIGN_OR_RETURN(auto device,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto host,cu::MemoryResource::create(context,cu::MemoryKind::kPinnedHost));struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
    c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,.query_heads=2,.kv_heads=2,.head_dim=64,.page_size=1};HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap,{.allocator=metadata.get()}));const std::array<std::int32_t,2> ptr{0,1};HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,{},{}}));HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),4096));std::array<c::ReadLease,3> inputs;for(auto&input:inputs){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(256));HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{0}));HYLE_ASSIGN_OR_RETURN(input,writer.publish());}
    HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,{.resource=device.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10,.scale=.125f,.allocator=metadata.get()}));HYLE_ASSIGN_OR_RETURN(auto invocation,c::BatchInvocation::create(program,queue));HYLE_ASSIGN_OR_RETURN(auto prepared,invocation.prepare_batch(layout));
    Decorator decorate(context);Gate gate;HYLE_ASSIGN_OR_RETURN(gate.stream,cu::stream(queue));HYLE_RETURN_IF_ERROR(cu::status(cudaLaunchHostFunc(gate.stream,&Gate::wait,&gate)));decorate.armed=true;
    HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(inputs));HYLE_RETURN_IF_ERROR(require(!done.poll() && invocation.in_flight() && !invocation.output(),"unknown observation lost F5 native guards/output gate"));done={};prepared={};
    auto drained=std::async(std::launch::async,[&]{return invocation.drain();});
    const bool waited=drained.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout;
    const bool held=pool.stats().active_uses==3;gate.release();const auto terminal=drained.get();HYLE_RETURN_IF_ERROR(context.drain());
    HYLE_RETURN_IF_ERROR(require(waited,"F5 drain returned before real CUDA prefix retirement"));HYLE_RETURN_IF_ERROR(require(held && !terminal.ok() && !invocation.in_flight() && pool.stats().active_uses==0,"F5 actual retirement/first error contract differs"));HYLE_RETURN_IF_ERROR(require(!invocation.output() && !invocation.prepare_batch(layout),"failed/closed F5 slot published or reused"));
    std::printf("F5 actual CUDA unknown observation/drain/guards/no-publication PASS\n");
  }
  return require(metadata.used()==0,"F5 fault native ownership metadata leaked");
}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F5 fault: %s\n",text);return 1;}
