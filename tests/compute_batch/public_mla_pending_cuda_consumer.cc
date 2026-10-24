#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include "mla_test_support.h"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {
namespace c=hyle::compute;
namespace n=c::cuda;
using hyle::Status;
using mla_test::check;
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
Status exercise(bool bf){
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(n::device(0).value(),c::flashinfer::provider()));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,n::MemoryResource::create(context,n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  Gate gate;
  struct Drain final {Gate&gate;c::Context&context;~Drain(){gate.release();context.drain().IgnoreError();}}drain{gate,context};
  const c::MlaBatchCapacity mc{.query_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .output_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .cache_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),
      .max_query_rows=1,.max_kv_rows=2,.max_pages=1,.max_page_refs=1,.query_heads=64};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create_mla(mc));
  const auto cap=schema.capacity();
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> pages{0},tail{2},short_tail{1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,pages,tail}));
  HYLE_ASSIGN_OR_RETURN(auto different,schema.version({ptr,ptr,pages,short_tail}));
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,{
      .resource=device.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=16u<<20,
      .max_int_workspace_bytes=1u<<20,.max_invocations=1,.scale=1.F/16}));
  HYLE_ASSIGN_OR_RETURN(auto native,c::BatchInvocation::create(program,queue));
  std::vector<std::uint16_t> q(cap.query_heads*cap.qk_head_dim,mla_test::encode(0,bf));
  std::vector<std::uint16_t> cache(cap.page_size*cap.qk_head_dim,mla_test::encode(0,bf));
  for(unsigned row=0;row<2;++row)for(unsigned d=0;d<cap.qk_head_dim;++d)
    cache[row*cap.qk_head_dim+d]=mla_test::encode(d<512?(row?.75F:.25F)+float(d%7)/128:.0625F,bf);
  const auto row_bytes=std::size_t(cap.qk_head_dim)*2,payload_bytes=2*row_bytes;
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),q.size()*2+cache.size()*2+payload_bytes+1024));
  HYLE_ASSIGN_OR_RETURN(auto qw,pool.allocate(q.size()*2));HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query,qw.publish());
  // Warm this same native specialization before holding its stream: the first
  // CUDA module/attribute setup may synchronize the device.
  {
    HYLE_ASSIGN_OR_RETURN(auto cw,pool.allocate(cache.size()*2));HYLE_RETURN_IF_ERROR(cw.view()->write(std::as_bytes(std::span(cache))));
    HYLE_ASSIGN_OR_RETURN(auto ready,cw.publish());
    HYLE_ASSIGN_OR_RETURN(auto prepared,native.prepare_batch(layout));
    HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(c::MlaBatchInputs{query,ready}));HYLE_RETURN_IF_ERROR(done.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,c::BatchCacheRole::kMla,device.ref(),{.max_backing_bytes=cache.size()*2+1}));
  HYLE_ASSIGN_OR_RETURN(auto page,arena.acquire(0));
  HYLE_ASSIGN_OR_RETURN(auto receiver,c::PagedBatchReceiver::create(std::move(native),3));
  HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
  std::puts("MLA pending native specialization and receiver warmed");
  HYLE_ASSIGN_OR_RETURN(auto source,pool.allocate(payload_bytes));
  HYLE_ASSIGN_OR_RETURN(auto hp,c::LeasePool::create(host.ref(),payload_bytes+64));
  HYLE_ASSIGN_OR_RETURN(auto hw,hp.allocate(payload_bytes));HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(std::span(cache)).first(payload_bytes)));
  HYLE_ASSIGN_OR_RETURN(auto staging,hw.publish());
  HYLE_ASSIGN_OR_RETURN(auto destination,n::pointer<std::byte>(source.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto source_pointer,n::pointer<std::byte>(staging.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto guards,c::LeaseUsePool::create());HYLE_ASSIGN_OR_RETURN(auto writing,guards.begin());
  HYLE_RETURN_IF_ERROR(writing.write(source));HYLE_RETURN_IF_ERROR(writing.read(staging));
  const std::array specs{c::CopyResourceSpec{device.ref(),payload_bytes},c::CopyResourceSpec{device.ref(),arena.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto copies,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(copies,queue));
  HYLE_ASSIGN_OR_RETURN(auto producer,n::submit(queue,[&](cudaStream_t stream)noexcept->Status{
    HYLE_RETURN_IF_ERROR(n::status(cudaLaunchHostFunc(stream,[](void*p){static_cast<Gate*>(p)->wait();},&gate)));
    return n::status(cudaMemcpyAsync(destination,source_pointer,payload_bytes,cudaMemcpyHostToDevice,stream));
  }));
  HYLE_RETURN_IF_ERROR(writing.attach(producer));HYLE_RETURN_IF_ERROR(check(gate.await(),"MLA native producer missed its checkpoint"));
  // Release before older CUDA allocation owners on every later early return.
  struct Release final {Gate&gate;~Release(){gate.release();}}release{gate};
  std::puts("MLA pending producer held at native checkpoint");
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({0},std::move(source)));HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(producer)));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(page.lease())));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,1> ranges{{{{0},{1},{1},0,0,payload_bytes}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto copied,copy.submit());
  std::puts("MLA pending copy submitted without producer host wait");
  HYLE_ASSIGN_OR_RETURN(auto update,c::CopyBatchUpdate::take(std::move(copy),std::move(copied)));
  const std::array<c::PagedPageUpdate,1> updates{{{&page,{1}}}};
  HYLE_ASSIGN_OR_RETURN(auto frame,arena.pending_read(layout,{},updates));
  HYLE_ASSIGN_OR_RETURN(auto wrong,arena.pending_read(different,{},updates));
  HYLE_RETURN_IF_ERROR(check(!receiver.submit(query,frame,frame,std::move(update)) && update.valid(),"MLA accepted independent pending K/V or consumed rejected Update"));
  HYLE_RETURN_IF_ERROR(check(!receiver.submit(query,wrong,std::move(update)) && update.valid(),"MLA accepted mismatched pending layout or consumed rejected Update"));
  HYLE_ASSIGN_OR_RETURN(auto done,receiver.submit(query,frame,std::move(update)));
  std::puts("MLA pending attention submitted without producer host wait");
  HYLE_RETURN_IF_ERROR(check(!update.valid() && !receiver.output() && !receiver.take_update() && guards.active_batches()==1,
      "MLA pending source/output/guards retired before the actual producer"));
  HYLE_RETURN_IF_ERROR(check(receiver.prepare_batch(layout).IsUnavailable(),"MLA pending slot allowed reuse"));
  gate.release();HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto output,receiver.output());HYLE_RETURN_IF_ERROR(mla_test::oracle(cap,layout,q,cache,output,bf));
  HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
  HYLE_RETURN_IF_ERROR(check(receiver.prepare_batch(layout).IsUnavailable(),"MLA old output allowed overwrite"));
  HYLE_ASSIGN_OR_RETURN(page.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(auto ready,page.publish(2));
  HYLE_RETURN_IF_ERROR(check(arena.read(layout,std::array{ready}).has_value(),"MLA copy writer did not recover as the same ready cache page"));
  HYLE_ASSIGN_OR_RETURN(copy,update.release_invocation());HYLE_RETURN_IF_ERROR(copy.reset());
  output.reset();done={};frame={};wrong={};HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
  HYLE_RETURN_IF_ERROR(receiver.drain());HYLE_RETURN_IF_ERROR(check(guards.active_batches()==0,"MLA producer native guards leaked"));
  std::printf("MLA %s one pending CKV cache: gated native producer/F2/attention, rejected binding preserves Update, full oracle, retained output and writer recovery PASS\n",bf?"BF16":"F16");
  return Status::Ok();
}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);for(bool bf:{false,true}){const auto status=exercise(bf);if(!status.ok()){
  char message[768]{};status.ToString(message,sizeof(message));std::fprintf(stderr,"public pending CUDA MLA: %s\n",message);return 1;
}}return 0;}
