#define HYLE_TESTING 1
#include "../../src/compute_backend_cuda/cuda_internal.h"
#include <hyle/compute/detail/copy_stream_access.h>
#include <hyle/compute/detail/queue_sequence.h>
#include <hyle/mem/quota_allocator.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include <hyle/compute/cpu.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace {
namespace c=hyle::compute;
namespace n=c::cuda;
using hyle::Status;
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
enum class Mode { known_source_error,unknown_before_acceptance,late_source_error,abandon,cpu_foreign_queue };
thread_local bool recording=false,fail_source=false;
thread_local std::vector<cudaEvent_t> source_events;
bool is_source(cudaEvent_t event){return std::find(source_events.begin(),source_events.end(),event)!=source_events.end();}
cudaError_t query_event(cudaEvent_t event)noexcept{
  if(recording && !is_source(event))source_events.push_back(event);
  if(fail_source && is_source(event)){fail_source=false;return cudaErrorMemoryAllocation;}
  return cudaEventQuery(event);
}
cudaError_t synchronize(cudaEvent_t event)noexcept{
  if(fail_source && is_source(event)){fail_source=false;return cudaErrorMemoryAllocation;}
  return cudaEventSynchronize(event);
}
Status run(Mode mode){
  HYLE_ASSIGN_OR_RETURN(auto id,n::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(id,c::flashinfer::provider()));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,n::MemoryResource::create(context,n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  const auto resource=device.ref();
  Gate gate;
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator()),source_metadata(hyle::mem::system_allocator());
  const auto status=[&]()->Status {
  const c::BatchCapacity cap{.data_type=c::DataType::kFloat16,.max_requests=1,.max_query_rows=1,.max_kv_rows=2,
      .max_pages=2,.max_page_refs=1,.query_heads=2,.kv_heads=1,.head_dim=64,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> pages{1},tail{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,pages,tail}));
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,
      {.resource=resource,.host_workspace=host.ref(),.max_float_workspace_bytes=8u<<20,
       .max_int_workspace_bytes=64u<<10,.max_invocations=1,.scale=.125F,.allocator=metadata.get()}));
  const auto program_metadata=metadata.used();
  HYLE_ASSIGN_OR_RETURN(auto native,c::BatchInvocation::create(program,queue));
  // Explicitly warm this same original math/controller before stopping its
  // stream. CUDA's first cudaFuncSetAttribute can load a module and synchronize
  // the device; holding a host callback then would deadlock the fixture.
  {
    HYLE_ASSIGN_OR_RETURN(auto warm,c::LeasePool::create(resource,8704));
    HYLE_ASSIGN_OR_RETURN(auto wq,warm.allocate(256));
    HYLE_ASSIGN_OR_RETURN(auto wk,warm.allocate(4096));
    HYLE_ASSIGN_OR_RETURN(auto wv,warm.allocate(4096));
    HYLE_RETURN_IF_ERROR(wq.view()->fill(std::byte{0}));
    HYLE_RETURN_IF_ERROR(wk.view()->fill(std::byte{0}));
    HYLE_RETURN_IF_ERROR(wv.view()->fill(std::byte{0}));
    HYLE_ASSIGN_OR_RETURN(auto rq,wq.publish());
    HYLE_ASSIGN_OR_RETURN(auto rk,wk.publish());
    HYLE_ASSIGN_OR_RETURN(auto rv,wv.publish());
    HYLE_ASSIGN_OR_RETURN(auto prepared,native.prepare_batch(layout));
    const std::array inputs{rq,rk,rv};
    HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(inputs));HYLE_RETURN_IF_ERROR(done.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto receiver,c::PagedBatchReceiver::create(std::move(native),4));
  HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto kw,keys.acquire(1));HYLE_ASSIGN_OR_RETURN(auto vw,values.acquire(1));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(resource,1024));
  HYLE_ASSIGN_OR_RETURN(auto q,pool.allocate(256));HYLE_RETURN_IF_ERROR(q.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto query,q.publish());
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(device.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto source,source_pool.allocate(512));
  HYLE_ASSIGN_OR_RETURN(auto hp,c::LeasePool::create(host.ref(),1024));HYLE_ASSIGN_OR_RETURN(auto h,hp.allocate(512));
  std::array<std::uint16_t,256> words{};for(std::size_t i=128;i<256;++i)words[i]=0x3400;
  HYLE_RETURN_IF_ERROR(h.view()->write(std::as_bytes(std::span(words))));HYLE_ASSIGN_OR_RETURN(auto packed,h.publish());
  HYLE_ASSIGN_OR_RETURN(auto src,n::pointer<std::byte>(source.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::pointer<std::byte>(packed.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto guards,c::LeaseUsePool::create());HYLE_ASSIGN_OR_RETURN(auto writing,guards.begin());
  HYLE_RETURN_IF_ERROR(writing.write(source));HYLE_RETURN_IF_ERROR(writing.read(packed));
  const std::array specs{c::CopyResourceSpec{device.ref(),512},c::CopyResourceSpec{resource,keys.page_bytes()},c::CopyResourceSpec{resource,values.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto copies,c::CopyBatchProgram::prepare(context,specs,{.allocator=source_metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(copies,queue));
  struct ReleaseGate final {Gate&gate;~ReleaseGate(){gate.release();}}release_gate{gate};
  HYLE_ASSIGN_OR_RETURN(auto producer,n::submit(queue,[&](cudaStream_t stream)noexcept->Status{
    HYLE_RETURN_IF_ERROR(n::status(cudaLaunchHostFunc(stream,[](void*p){static_cast<Gate*>(p)->wait();},&gate)));
    return n::status(cudaMemcpyAsync(src,staging,512,cudaMemcpyHostToDevice,stream));
  }));
  HYLE_RETURN_IF_ERROR(writing.attach(producer));HYLE_RETURN_IF_ERROR(require(gate.await(),"actual CUDA producer missed stream checkpoint"));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({0},std::move(source)));HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(producer)));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(kw.lease())));HYLE_RETURN_IF_ERROR(copy.bind_mutable({2},std::move(vw.lease())));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,2> ranges{{{{0},{1},{1},0,0,256},{{0},{2},{1},256,0,256}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto copied,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,c::CopyBatchUpdate::take(std::move(copy),std::move(copied)));
  const std::array<c::PagedPageUpdate,1> ku{{{&kw,{1}}}},vu{{{&vw,{2}}}};
  HYLE_ASSIGN_OR_RETURN(auto k,keys.pending_read(layout,{},ku));HYLE_ASSIGN_OR_RETURN(auto v,values.pending_read(layout,{},vu));
  source_events.clear();recording=true;fail_source=false;
  const auto previous=n::detail::set_completion_observation_hooks({&query_event,&synchronize});
  struct Reset final {n::detail::CompletionObservationHooks previous;~Reset(){n::detail::set_completion_observation_hooks(previous);}}reset{previous};
  const auto observed=update.observe();
  auto prefix=c::detail::CopyStreamAccess::dependency(update);recording=false;
  HYLE_RETURN_IF_ERROR(require(observed.IsUnavailable() && prefix.has_value() && prefix->pending() && !source_events.empty(),"risk did not capture actual original source event"));
  if(mode==Mode::cpu_foreign_queue){
    HYLE_ASSIGN_OR_RETURN(auto cpu,c::Context::create(c::cpu::device()));
    HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto sequence,c::detail::QueueSequence::begin(cpu_queue));
    const auto rejected=sequence.enqueue_resolved_wait(*prefix);
    HYLE_RETURN_IF_ERROR(require(!rejected.accepted && rejected.status.IsUnimplemented() && update.valid() &&
        guards.active_batches()==1 && !gate.released,"CPU queue imported actual pending CUDA prefix or hid a host wait"));
    sequence.record(rejected);auto finished=sequence.finish();
    HYLE_RETURN_IF_ERROR(require(!finished && finished.error().IsUnimplemented(),"rejected CPU prefix fabricated a completion"));
    gate.release();HYLE_RETURN_IF_ERROR(update.wait());
    std::fprintf(stderr,"original CPU QueueSequence rejects real pending CUDA prefix without host wait/accepted prefix/control loss PASS\n");return Status::Ok();
  }
  if(mode==Mode::known_source_error || mode==Mode::unknown_before_acceptance){
    fail_source=true;HYLE_RETURN_IF_ERROR(require(update.observe().IsUnavailable() && !fail_source && guards.active_batches()==1,"known source observation fabricated retirement"));
    auto failed_prefix=c::detail::CopyStreamAccess::dependency(update);
    HYLE_RETURN_IF_ERROR(require(failed_prefix.has_value() && failed_prefix->terminal_hint && failed_prefix->terminal_hint->ok(),"unknown observation fabricated public source terminal before retirement"));
  }
  if(mode==Mode::known_source_error){
    gate.release();HYLE_RETURN_IF_ERROR(require(update.wait().IsResourceExhausted(),"known source lost actual terminal error"));
    auto rejected=receiver.submit(query,k,v,std::move(update));
    HYLE_RETURN_IF_ERROR(require(!rejected && update.valid() && !receiver.output(),"common CUDA receiver accepted known failed source or lost original Update"));
    HYLE_RETURN_IF_ERROR(require(!update.take_mutable({1}),"failed source fabricated successful writer"));
    std::fprintf(stderr,"common CUDA receiver known-source failure preserves controls and actual retirement PASS\n");return Status::Ok();
  }
  HYLE_ASSIGN_OR_RETURN(auto submitted,receiver.submit(query,k,v,std::move(update)));
  HYLE_RETURN_IF_ERROR(require(!update.valid() && !receiver.output() && !receiver.take_update() && guards.active_batches()==1,
      "CUDA receiver released pending producer/output/source early"));
  HYLE_RETURN_IF_ERROR(require(receiver.prepare_batch(layout).IsUnavailable(),"CUDA pending native slot reused"));
  if(mode==Mode::abandon){
    submitted={};receiver={};
    HYLE_RETURN_IF_ERROR(require(guards.active_batches()==1 && !c::BatchInvocation::create(program,queue),"abandoned common receiver released native work/slot before retirement"));
    gate.release();HYLE_RETURN_IF_ERROR(context.drain());
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(metadata.used()!=program_metadata && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    HYLE_RETURN_IF_ERROR(require(metadata.used()==program_metadata && guards.active_batches()==0,"abandoned common receiver did not retire original owner metadata/guards"));
    HYLE_ASSIGN_OR_RETURN(auto next,c::BatchInvocation::create(program,queue));HYLE_RETURN_IF_ERROR(next.drain());
    std::fprintf(stderr,"common CUDA receiver pending aggregate/owner abandon retains then retires original slot/metadata/guards PASS\n");return Status::Ok();
  }
  fail_source=mode==Mode::late_source_error;gate.release();const auto terminal=submitted.wait();
  HYLE_RETURN_IF_ERROR(require(!fail_source && terminal.IsResourceExhausted() && !receiver.output(),"source failure was hidden by successful original native suffix"));
  HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
  HYLE_RETURN_IF_ERROR(require(!update.take_mutable({1}) && receiver.prepare_batch(layout).IsFailedPrecondition(),"late failed source/epoch fabricated ready output or reuse"));
  std::fprintf(stderr,"common CUDA receiver source risk=%u unknown/late source failure merges original terminal and refuses output/reuse PASS\n",unsigned(mode));return Status::Ok();


  }();
  if(!status.ok())std::fprintf(stderr,"risk observed before cleanup: %.*s\n",int(status.message().size()),status.message().data());
  gate.release();context.drain().IgnoreError();
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while((metadata.used() || source_metadata.used()) && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  HYLE_RETURN_IF_ERROR(require(metadata.used()==0 && source_metadata.used()==0,"risk teardown did not retire original metadata"));
  return status;

}
}
int main(){bool failed=false;for(auto mode:{Mode::known_source_error,Mode::unknown_before_acceptance,Mode::late_source_error,Mode::abandon,Mode::cpu_foreign_queue}){const auto status=run(mode);if(!status.ok()){
  std::fprintf(stderr,"CUDA receiver risk=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());failed=true;
}}return failed?1:0;}
