#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/paged_arena.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {
namespace c=hyle::compute;namespace n=c::cuda;
using hyle::Status;
Status require(bool ok,const char* message){return ok?Status::Ok():Status::Internal(message);}
struct Gate final {
  std::mutex mutex;std::condition_variable condition;
  bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto device_id,n::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,n::MemoryResource::create(context,n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Drain{c::Queue& queue;~Drain(){queue.drain().IgnoreError();}}drain{queue};
  HYLE_ASSIGN_OR_RETURN(auto environment,n::environment(context));
  const c::BatchCapacity cap{.max_requests=2,.max_query_rows=2,.max_kv_rows=32,
      .max_pages=2,.max_page_refs=2,.query_heads=8,.kv_heads=2,.head_dim=256};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=128*1024}));
  const auto page_bytes=arena.page_bytes();constexpr std::size_t row_bytes=1024;
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(host.ref(),2*page_bytes+4096));
  HYLE_ASSIGN_OR_RETURN(auto destination_pool,c::LeasePool::create(host.ref(),page_bytes+512));
  HYLE_ASSIGN_OR_RETURN(auto source_writer,source_pool.allocate(page_bytes));
  std::vector<std::byte> expected(page_bytes,std::byte{17}),actual(page_bytes);
  HYLE_RETURN_IF_ERROR(source_writer.view()->write(expected));
  HYLE_ASSIGN_OR_RETURN(auto source,source_writer.publish(cap.page_size));
  const std::array specs{c::CopyResourceSpec{host.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(program,queue));
  const std::array<c::CopyPhase,1> phases{{{}}};
  HYLE_ASSIGN_OR_RETURN(auto first,arena.acquire(0));
  HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(first.lease())));
  const std::array<c::CopyRange,1> full{{{{0},{1},{0},0,0,page_bytes}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,full));
  c::ReadLease ready;
  {HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
    HYLE_ASSIGN_OR_RETURN(ready,submitted.output({0},{1},0,page_bytes));HYLE_RETURN_IF_ERROR(first.complete(ready,cap.page_size));}
  HYLE_RETURN_IF_ERROR(copy.reset());const auto generation=ready.generation();
  const std::array down_specs{c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{host.ref(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto down_program,c::CopyBatchProgram::prepare(context,down_specs));
  HYLE_ASSIGN_OR_RETURN(auto down,c::CopyBatchInvocation::create(down_program,queue));
  for(unsigned step=0;step<64;++step){
    HYLE_ASSIGN_OR_RETURN(source_writer,source.try_write());
    HYLE_RETURN_IF_ERROR(source_writer.view()->fill(std::byte(23+step)));
    HYLE_ASSIGN_OR_RETURN(source,source_writer.publish(cap.page_size));
    {auto alias=ready;HYLE_RETURN_IF_ERROR(require(!arena.acquire(ready) && ready.valid(),"CUDA shared page permitted mutation or consumed reader"));}
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(ready));
    HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer.lease())));
    const auto offset=(step%16)*row_bytes;
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,offset,row_bytes}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    Gate gate;struct Release{Gate& gate;c::Queue& queue;~Release(){gate.release();queue.drain().IgnoreError();}}release{gate,queue};
    if(step==0){
      HYLE_ASSIGN_OR_RETURN(auto producer,n::submit(queue,[&](cudaStream_t stream)noexcept{
        return cudaLaunchHostFunc(stream,[](void* p){static_cast<Gate*>(p)->wait();},&gate);
      }));
      HYLE_RETURN_IF_ERROR(require(gate.await(),"CUDA native stream did not enter checkpoint"));
      HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(producer)));
      HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    }
    {HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
      HYLE_RETURN_IF_ERROR(require(!copy.unbind_mutable({1}),"CUDA retained submission returned writer"));
      if(step==0){
        HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses==1 && !submitted.output({0},{1},offset,row_bytes),"CUDA pending producer released guard or published dirty data"));
        gate.release();
      }
      HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_RETURN_IF_ERROR(require(submitted.report().logical_bytes==row_bytes,"CUDA update copied history"));
    }
    HYLE_ASSIGN_OR_RETURN(writer.lease(),copy.unbind_mutable({1}));
    HYLE_ASSIGN_OR_RETURN(ready,writer.publish(cap.page_size));
    HYLE_RETURN_IF_ERROR(require(ready.generation()==generation,"CUDA in-place update changed generation"));
    HYLE_RETURN_IF_ERROR(copy.reset());
    std::fill_n(expected.begin()+offset,row_bytes,std::byte(23+step));
    HYLE_ASSIGN_OR_RETURN(auto destination,destination_pool.allocate(page_bytes));
    HYLE_RETURN_IF_ERROR(down.bind_source({0},ready));HYLE_RETURN_IF_ERROR(down.bind_mutable({1},std::move(destination)));
    HYLE_RETURN_IF_ERROR(down.prepare_replay(phases,full));
    {HYLE_ASSIGN_OR_RETURN(auto submitted,down.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_ASSIGN_OR_RETURN(auto output,submitted.output({0},{1},0,page_bytes));
      HYLE_RETURN_IF_ERROR(output.view()->read(actual));HYLE_RETURN_IF_ERROR(require(actual==expected,"CUDA dirty row or preserved history differs"));}
    HYLE_RETURN_IF_ERROR(down.reset());
  }
  HYLE_RETURN_IF_ERROR(copy.close());HYLE_RETURN_IF_ERROR(down.close());
  std::fprintf(stderr,"CUDA public dirty page PASS64 device=%s SM%d%d, payload<128KiB, dirty1024/page16384, preserved history/generation, exact public F2 permission return, actual gated stream producer/guard; download only oracle\n",environment.device_name.data(),environment.sm_major,environment.sm_minor);
  return Status::Ok();
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"CUDA dirty page: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
