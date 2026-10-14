#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/transfer.h>
#include <hyle/compute/detail/paged_arena_access.h>
#include <hyle/compute/detail/copy_outcome_source.h>
#include <hyle/compute/detail/dependent_completion.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace {
namespace c=hyle::compute;namespace n=c::cuda;namespace m=hyle::mem;
using hyle::Status;using Access=c::detail::PagedArenaAccess;using CopyAccess=c::detail::CopyStreamAccess;
Status require(bool ok,const char*message){return ok?Status::Ok():Status::Internal(message);}
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto id,n::device(0));HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(id));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,n::MemoryResource::create(context,n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());Gate gate;
  struct Drain final {Gate&gate;c::Queue&queue;~Drain(){gate.release();queue.drain().IgnoreError();}}drain{gate,queue};
  const c::BatchCapacity cap{.max_requests=1,.max_query_rows=1,.max_kv_rows=2,.max_pages=2,.max_page_refs=1,
    .query_heads=2,.kv_heads=2,.head_dim=64,.page_size=2};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=2048}));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(host.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto incoming,pool.allocate(256));HYLE_RETURN_IF_ERROR(incoming.view()->fill(std::byte{17}));
  HYLE_ASSIGN_OR_RETURN(auto source,incoming.publish());HYLE_ASSIGN_OR_RETURN(auto target,pool.allocate(256));
  HYLE_ASSIGN_OR_RETURN(auto target_view,target.view());
  const std::array specs{c::CopyResourceSpec{host.ref(),256},c::CopyResourceSpec{device.ref(),512}};
  HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(program,queue));
  HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(0));
  HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer.lease())));
  using Source=c::detail::CopyOutcomeSource<c::CopyBatchUpdate,CopyAccess>;
  HYLE_ASSIGN_OR_RETURN(auto cold,Source::create(4));
  HYLE_ASSIGN_OR_RETURN(auto observer,c::detail::DependentCompletion::create(m::system_allocator()));
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> pages{0},tail{1},too_long{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,pages,tail}));
  HYLE_ASSIGN_OR_RETURN(auto longer,schema.version({ptr,ptr,pages,too_long}));
  const std::array<c::PagedPageUpdate,1> updates{{{&writer,{1}}}};
  HYLE_ASSIGN_OR_RETURN(auto frame,arena.pending_read(layout,{},updates));
  HYLE_ASSIGN_OR_RETURN(auto excessive,arena.pending_read(longer,{},updates));
  HYLE_ASSIGN_OR_RETURN(auto producer,n::submit(queue,[&](cudaStream_t stream)noexcept{
    return cudaLaunchHostFunc(stream,[](void*p){static_cast<Gate*>(p)->wait();},&gate);
  }));
  HYLE_RETURN_IF_ERROR(require(gate.await(),"CUDA pending source did not reach actual stream checkpoint"));
  HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(producer)));
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,256}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,c::CopyBatchUpdate::take(std::move(copy),std::move(submitted)));
  HYLE_RETURN_IF_ERROR((Access::validate<c::CopyBatchUpdate,CopyAccess>(frame,update)));
  HYLE_RETURN_IF_ERROR(require((Access::validate<c::CopyBatchUpdate,CopyAccess>(excessive,update)).IsOutOfRange() && update.valid(),
    "CUDA accepted row copy fabricated full-page initialized coverage"));
  HYLE_ASSIGN_OR_RETURN(auto protocol,cold.take(std::move(update)));
  struct Release final {c::detail::OutcomeSources*protocol;~Release(){if(protocol->valid())protocol->release(protocol->state);}}release{&protocol};
  HYLE_ASSIGN_OR_RETURN(auto pending,CopyAccess::input(cold.update(),{1}));
  HYLE_ASSIGN_OR_RETURN(auto reading,uses.begin());
  // Source and suffix use the same original CUDA queue/stream. Enqueue the
  // original transfer after its copy prefix; no intermediate host observation.
  HYLE_RETURN_IF_ERROR((Access::admit_read<c::CopyBatchUpdate,CopyAccess>(reading,frame,cold.update())));
  HYLE_RETURN_IF_ERROR(reading.write(target));HYLE_ASSIGN_OR_RETURN(auto row,pending.view.subview(0,256));
  HYLE_ASSIGN_OR_RETURN(auto suffix,c::copy_async(queue,target_view,row));HYLE_RETURN_IF_ERROR(reading.attach(suffix));
  auto completion=observer.publish(std::move(suffix),protocol);protocol={};
  HYLE_RETURN_IF_ERROR(require(uses.active_batches()==1 && !cold.update().take_mutable({1}),
    "CUDA pending suffix released page permission before stream completion"));
  gate.release();HYLE_RETURN_IF_ERROR(completion.wait());
  HYLE_RETURN_IF_ERROR(require(uses.active_batches()==0 && cold.available(),"CUDA original aggregate missed actual retirement"));
  std::array<std::byte,256> actual{};HYLE_RETURN_IF_ERROR(target.view()->read(actual));
  unsigned sum=0;for(auto value:actual){HYLE_RETURN_IF_ERROR(require(value==std::byte{17},"CUDA initialized row byte oracle differs"));sum+=std::to_integer<unsigned>(value);}
  HYLE_RETURN_IF_ERROR(require(sum==17*256,"CUDA-to-CPU row reduction differs"));
  HYLE_ASSIGN_OR_RETURN(writer.lease(),cold.update().take_mutable({1}));
  HYLE_ASSIGN_OR_RETURN(auto ready,writer.publish(1));
  HYLE_RETURN_IF_ERROR(require(arena.read(layout,std::span(&ready,1)).has_value() && !arena.read(longer,std::span(&ready,1)),
    "CUDA ready frame ignored committed initialized row count"));
  HYLE_ASSIGN_OR_RETURN(copy,cold.update().release_invocation());HYLE_RETURN_IF_ERROR(copy.reset());
  std::fprintf(stderr,"CUDA pending arena: 256-byte real gated producer/F2/queued download, original cold source and final-only aggregate, byte/reduction oracle, tail rejection, guards and original control recovery PASS\n");
  return Status::Ok();
}
} // namespace
int main(){auto status=run();if(!status.ok())std::fprintf(stderr,"CUDA pending:%.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
