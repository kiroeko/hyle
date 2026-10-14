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
Status run(bool host_receiver){
  HYLE_ASSIGN_OR_RETURN(auto id,n::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(id,c::flashinfer::provider()));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,n::MemoryResource::create(context,n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto cpu,c::cpu::create_context({.native_threads=2}));
  const auto resource=device.ref();
  Gate gate;
  struct Drain final {Gate&gate;c::Context&context;~Drain(){gate.release();context.drain().IgnoreError();}}drain{gate,context};
  const c::BatchCapacity cap{.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.max_requests=1,.max_query_rows=1,.max_kv_rows=2,
      .max_pages=2,.max_page_refs=1,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> pages{1},tail{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,pages,tail}));
  if(host_receiver){
    auto rejected=c::BatchProgram::prepare(cpu,schema,
        {.resource=host.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=8u<<20,
         .max_int_workspace_bytes=64u<<10,.max_invocations=1,.scale=.125F});
    HYLE_RETURN_IF_ERROR(require(!rejected && rejected.error().IsUnimplemented(),
        "CPU native batch admitted CUDA pinned memory without coherent-host qualification"));
    std::fprintf(stderr,"CPU native batch preserves coherent-host qualification; CUDA pinned resource rejected PASS\n");
    return Status::Ok();
  }
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,
      {.resource=resource,.host_workspace=host.ref(),.max_float_workspace_bytes=8u<<20,
       .max_int_workspace_bytes=64u<<10,.max_invocations=1,.scale=.125F}));
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
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,resource,{.max_backing_bytes=8192}));
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
  HYLE_ASSIGN_OR_RETURN(auto copies,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(copies,queue));
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
  HYLE_ASSIGN_OR_RETURN(auto submitted,receiver.submit(query,k,v,std::move(update)));
  HYLE_RETURN_IF_ERROR(require(!update.valid() && !receiver.output() && !receiver.take_update() && guards.active_batches()==1,
      "CUDA receiver released pending producer/output/source early"));
  HYLE_RETURN_IF_ERROR(require(receiver.prepare_batch(layout).IsUnavailable(),"CUDA pending native slot reused"));
  gate.release();HYLE_RETURN_IF_ERROR(submitted.wait());HYLE_ASSIGN_OR_RETURN(auto output,receiver.output());
  std::array<std::uint16_t,128> actual{};HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
  for(auto word:actual)HYLE_RETURN_IF_ERROR(require(word==0x3400,"CUDA original paged native mean oracle differs"));
  HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
  HYLE_RETURN_IF_ERROR(require(receiver.prepare_batch(layout).IsUnavailable(),"CUDA old output did not prevent overwrite"));
  HYLE_ASSIGN_OR_RETURN(kw.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(vw.lease(),update.take_mutable({2}));
  HYLE_ASSIGN_OR_RETURN(auto ready_k,kw.publish(2));HYLE_ASSIGN_OR_RETURN(auto ready_v,vw.publish(2));
  HYLE_RETURN_IF_ERROR(require(keys.read(layout,std::span(&ready_k,1)).has_value() && values.read(layout,std::span(&ready_v,1)).has_value(),
      "CUDA common receiver did not recover original initialized ready frames"));
  HYLE_ASSIGN_OR_RETURN(copy,update.release_invocation());HYLE_RETURN_IF_ERROR(copy.reset());
  output.reset();HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));HYLE_RETURN_IF_ERROR(receiver.drain());
  HYLE_RETURN_IF_ERROR(require(guards.active_batches()==0,"CUDA producer native guards did not retire"));
  std::fprintf(stderr,"CUDA paged FlashInfer final-only aggregate common receiver 512-byte gated native producer/F2, .25 mean oracle, old-output/source/writer/ready recovery PASS\n");
  return Status::Ok();
}
}
int main(){for(bool host_receiver:{false,true}){const auto status=run(host_receiver);if(!status.ok()){
  std::fprintf(stderr,"CUDA receiver host=%u: %.*s\n",unsigned(host_receiver),int(status.message().size()),status.message().data());return 1;
}}return 0;}
