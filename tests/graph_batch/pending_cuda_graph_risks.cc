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
#include <hyle/graph/batch.h>
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
#include <hyle/graph/cpu_cuda.h>
#include <hyle/compute/providers/ggml_cpu.h>
#endif
#include <hyle/compute/paged_arena.h>
#include <hyle/compute/cpu.h>
#include <hyle/compute/providers/native.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <cmath>

namespace {
namespace c=hyle::compute;
namespace n=c::cuda;namespace g=hyle::graph;namespace b=g::batch;
using hyle::Status;
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
enum class Mode { known_source_error,unknown_before_acceptance,late_source_error,abandon,cpu_foreign_queue,success,cancel_before,cancel_after };
thread_local bool recording=false,fail_source=false;
thread_local std::vector<cudaEvent_t> source_events;
bool is_source(cudaEvent_t event){return std::find(source_events.begin(),source_events.end(),event)!=source_events.end();}
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
// The original GPU-to-CPU replay boundary waits inside submit. Release the
// fixture stream gate at that actual event observation, after acceptance;
// no helper thread, new execution owner or product host wait is introduced.
struct JointWaitCheckpoint final {
  Gate *gate; c::LeaseUsePool *guards; c::Completion *cancel_source; Mode mode;
  bool seen=false; Status status;
};
thread_local JointWaitCheckpoint *joint_checkpoint=nullptr;
#endif
cudaError_t query_event(cudaEvent_t event)noexcept{
  if(recording && !is_source(event))source_events.push_back(event);
  if(fail_source && is_source(event)){fail_source=false;return cudaErrorMemoryAllocation;}
  return cudaEventQuery(event);
}
cudaError_t synchronize(cudaEvent_t event)noexcept{
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  if(joint_checkpoint && !is_source(event) && !joint_checkpoint->seen){
    auto &checkpoint=*joint_checkpoint;checkpoint.seen=true;
    checkpoint.status.Update(require(checkpoint.guards->active_batches()==1 && !checkpoint.gate->released,
        "joint checked wait lost original gated producer guards before actual retirement"));
    if(checkpoint.mode==Mode::cancel_after){
      checkpoint.status.Update(require(checkpoint.cancel_source->request_cancel().IsUnimplemented() &&
          checkpoint.guards->active_batches()==1,
          "joint accepted native cancel refusal lost work or fabricated retirement"));
    }
    fail_source=checkpoint.mode==Mode::late_source_error;
    checkpoint.gate->release();
  }
#endif
  if(fail_source && is_source(event)){fail_source=false;return cudaErrorMemoryAllocation;}
  return cudaEventSynchronize(event);
}
Status run(Mode mode){
  HYLE_ASSIGN_OR_RETURN(auto id,n::device(0));
  const c::BatchCapacity cap{.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.max_requests=1,.max_query_rows=1,.max_kv_rows=2,
      .max_pages=2,.max_page_refs=1,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> pages{1},tail{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,pages,tail}));
  HYLE_ASSIGN_OR_RETURN(auto wrapped,b::provider(c::flashinfer::provider(),schema,
      {.resource={},.host_workspace=hyle::mem::system_resource(),.max_float_workspace_bytes=8u<<20,
       .max_int_workspace_bytes=64u<<10,.max_invocations=1,.scale=.125F}));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(id,wrapped));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,n::MemoryResource::create(context,n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  const auto resource=device.ref();
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  HYLE_ASSIGN_OR_RETURN(auto cpu_context,c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=2})));
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu_context.create_queue());
#endif
  Gate gate;
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator()),source_metadata(hyle::mem::system_allocator());
  const auto status=[&]()->Status {
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
  native={}; // Warmed original native math; Graph has its own original slot.
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{1,2,64}));
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  HYLE_ASSIGN_OR_RETURN(auto host_desc,c::TensorDesc::dense(c::DataType::kFloat32,qdesc.dimensions()));
#else
  const auto host_desc=qdesc;
#endif
  HYLE_ASSIGN_OR_RETURN(auto kvdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,4>{2,16,1,64}));
  HYLE_ASSIGN_OR_RETURN(auto metadata_desc,schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  const std::array descriptions{host_desc,kvdesc,kvdesc,metadata_desc};std::array<g::ValueId,4> inputs;
  for(unsigned i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
  auto ports=inputs;
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  HYLE_ASSIGN_OR_RETURN(ports[0],builder.add(c::Unary{c::UnaryKind::kNeg},ports[0]));
  HYLE_ASSIGN_OR_RETURN(ports[0],builder.add(c::Cast{c::DataType::kFloat16},ports[0]));
#endif
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::PagedAttention{cap,.125F}),ports));
  HYLE_ASSIGN_OR_RETURN(auto output_id,builder.result(node));
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  // Publish only the actual CPU suffix, not the intermediate GPU result.
  HYLE_ASSIGN_OR_RETURN(auto suffix,builder.add(c::Cast{c::DataType::kFloat32},output_id));
  HYLE_ASSIGN_OR_RETURN(auto final,builder.add(c::Unary{c::UnaryKind::kNeg},suffix));
  output_id=final;
#endif
  HYLE_RETURN_IF_ERROR(builder.publish(output_id));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  namespace bridge=g::cpu_cuda;
  std::vector<g::NodePlacement> placements;
  for(std::size_t i=0;i<graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto id,graph.node_id(i));
    placements.push_back({id,id==node?bridge::kCudaTarget:bridge::kCpuTarget,g::PlacementStrength::kRequired});}
  const std::array external{g::ExternalResourceContract{inputs[0],bridge::kHostStorage},g::ExternalResourceContract{output_id,bridge::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,bridge::compile(graph,{&cpu_context,&context,hyle::mem::system_resource(),&host,&device},
      {.node_placements=placements,.external_resources=external}));
  const std::array lanes{b::LaneBinding{g::ExecutionLaneId{0},&context,&queue,host.ref()},
      b::LaneBinding{g::ExecutionLaneId{1},&cpu_context,&cpu_queue,hyle::mem::system_resource()}};
#else
  const std::array storage{g::StorageDomain{resource}};
  const std::array targets{g::ExecutionTarget{&context,g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::compile(graph,{.storage_domains=storage,.targets=targets},
      {.operation_fallback=g::OperationFallbackPolicy::kReject}));
  const std::array lanes{b::LaneBinding{g::ExecutionLaneId{0},&context,&queue,host.ref()}};
#endif
  const g::LeasedInvocationOptions options{.allocator=metadata.get()};
  HYLE_ASSIGN_OR_RETURN(auto original,b::Invocation::create(compiled.executable,lanes,options));
  HYLE_ASSIGN_OR_RETURN(auto receiver,b::PendingInvocation::create(std::move(original),4,options));
  const std::array versions{b::NodeLayout{node,layout}};
  HYLE_RETURN_IF_ERROR(receiver.prepare_batch(versions));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto kw,keys.acquire(1));HYLE_ASSIGN_OR_RETURN(auto vw,values.acquire(1));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(resource,1024));
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  HYLE_ASSIGN_OR_RETURN(auto cpu_pool,c::LeasePool::create(hyle::mem::system_resource(),2048));
  auto&external_pool=cpu_pool;constexpr std::size_t external_bytes=512;
#else
  auto&external_pool=pool;constexpr std::size_t external_bytes=256;
#endif
  HYLE_ASSIGN_OR_RETURN(auto q,external_pool.allocate(external_bytes));HYLE_RETURN_IF_ERROR(q.view()->fill(std::byte{0}));
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  std::array<float,128> cpu_query;cpu_query.fill(-.125F);
  HYLE_RETURN_IF_ERROR(q.view()->write(std::as_bytes(std::span(cpu_query))));
#endif
  HYLE_ASSIGN_OR_RETURN(auto query,q.publish());
  HYLE_ASSIGN_OR_RETURN(auto output,external_pool.allocate(external_bytes));
  HYLE_RETURN_IF_ERROR(receiver.bind(inputs[0],host_desc,query));
  HYLE_RETURN_IF_ERROR(receiver.bind(output_id,host_desc,std::move(output)));
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(device.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto source,source_pool.allocate(512));
  HYLE_ASSIGN_OR_RETURN(auto hp,c::LeasePool::create(host.ref(),1024));HYLE_ASSIGN_OR_RETURN(auto h,hp.allocate(512));
  std::array<std::uint16_t,256> words{};for(std::size_t i=128;i<256;++i)words[i]=0x3400;
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  std::fill_n(words.begin(),64,std::uint16_t{0x2800});
  std::fill_n(words.begin()+64,64,std::uint16_t{0x2c00});
  std::fill_n(words.begin()+192,64,std::uint16_t{0x3800});
#endif
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
  std::array<c::Completion*,3> original_sources{};
  HYLE_ASSIGN_OR_RETURN(auto source_count,c::detail::CopyStreamAccess::retirement_sources(update,original_sources));
  HYLE_RETURN_IF_ERROR(require(source_count==3,"pending Graph lost accepted source producer/phase controls"));
  HYLE_RETURN_IF_ERROR(receiver.bind(inputs[1],k));HYLE_RETURN_IF_ERROR(receiver.bind(inputs[2],v));
  source_events.clear();recording=true;fail_source=false;
  const auto previous=n::detail::set_completion_observation_hooks({&query_event,&synchronize});
  struct Reset final {n::detail::CompletionObservationHooks previous;~Reset(){
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
    joint_checkpoint=nullptr;
#endif
    n::detail::set_completion_observation_hooks(previous);}}reset{previous};
  const auto observed=update.observe();
  auto prefix=c::detail::CopyStreamAccess::dependency(update);recording=false;
  HYLE_RETURN_IF_ERROR(require(observed.IsUnavailable() && prefix.has_value() && prefix->pending() && !source_events.empty(),"risk did not capture actual original source event"));
  if(mode==Mode::cpu_foreign_queue){
    HYLE_ASSIGN_OR_RETURN(auto cpu,c::Context::create(c::cpu::device(),c::native::provider()));
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
    auto rejected=receiver.submit(std::move(update));
    HYLE_RETURN_IF_ERROR(require(!rejected && update.valid() && !receiver.output(output_id),"CUDA pending Graph accepted known failed source or lost original Update"));
    HYLE_RETURN_IF_ERROR(require(!update.take_mutable({1}),"failed source fabricated successful writer"));
    std::fprintf(stderr,"CUDA pending Graph known-source failure preserves controls and actual retirement PASS\n");return Status::Ok();
  }
  if(mode==Mode::cancel_before){
    const auto refused=update.request_cancel();
    HYLE_RETURN_IF_ERROR(require(refused.IsUnimplemented() && update.valid() && guards.active_batches()==1 && !gate.released,
        "unsupported native cancel fabricated source retirement or lost original owner"));
  }
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  JointWaitCheckpoint checkpoint{&gate,&guards,original_sources.back(),mode,false,{}};
  joint_checkpoint=&checkpoint;
#endif
  HYLE_ASSIGN_OR_RETURN(auto submitted,receiver.submit(std::move(update)));
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  joint_checkpoint=nullptr;
  HYLE_RETURN_IF_ERROR(checkpoint.status);
  HYLE_RETURN_IF_ERROR(require(checkpoint.seen && !update.valid() && !receiver.take_update(),
      "joint receiver skipped actual checked boundary or returned retained source controls"));
#else
  HYLE_RETURN_IF_ERROR(require(!update.valid() && !receiver.output(output_id) && !receiver.take_update() && guards.active_batches()==1,
      "CUDA receiver released pending producer/output/source early"));
#endif
  HYLE_RETURN_IF_ERROR(require(receiver.prepare_batch(versions).IsUnavailable(),"CUDA pending native slot reused"));
  if(mode==Mode::abandon){
    submitted={};receiver={};
#if !defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
    HYLE_RETURN_IF_ERROR(require(guards.active_batches()==1 && !b::Invocation::create(compiled.executable,lanes,options),"abandoned common receiver released native work/slot before retirement"));
#endif
    gate.release();HYLE_RETURN_IF_ERROR(context.drain());
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(metadata.used()!=program_metadata && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    HYLE_RETURN_IF_ERROR(require(metadata.used()==program_metadata && guards.active_batches()==0,"abandoned common receiver did not retire original owner metadata/guards"));
    HYLE_ASSIGN_OR_RETURN(auto next,b::Invocation::create(compiled.executable,lanes,options));HYLE_RETURN_IF_ERROR(next.drain());
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
    std::fprintf(stderr,"CPU+CUDA actual checked-boundary result/owner abandon retires original slot/metadata/guards; gated acceptance checkpoint retained source PASS\n");
#else
    std::fprintf(stderr,"CUDA pending Graph pending aggregate/owner abandon retains then retires original slot/metadata/guards PASS\n");
#endif
    return Status::Ok();
  }
  if(mode==Mode::cancel_after){
#if !defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
    const auto refused=original_sources.back()->request_cancel();
    HYLE_RETURN_IF_ERROR(require(refused.IsUnimplemented() && guards.active_batches()==1 && !receiver.output(output_id) && !receiver.take_update(),
        "unsupported native cancel fabricated Graph/source retirement or successful output"));
#endif
  }
#if !defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  fail_source=mode==Mode::late_source_error;
#endif
  gate.release();const auto terminal=submitted.wait();
  if(mode==Mode::success || mode==Mode::cancel_before || mode==Mode::cancel_after){
    HYLE_RETURN_IF_ERROR(terminal);
    {HYLE_ASSIGN_OR_RETURN(auto output,receiver.output(output_id));HYLE_ASSIGN_OR_RETURN(auto read,output.storage_lease());
      HYLE_ASSIGN_OR_RETURN(auto view,read.view());
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
      std::array<float,128> actual{};
#else
      std::array<std::uint16_t,128> actual{};
#endif
      HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(actual))));
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
      const double weight=std::exp(.03125),expected=-(.25+.5*weight)/(1.+weight);
      for(auto word:actual)HYLE_RETURN_IF_ERROR(require(std::abs(double(word)-expected)<.001,
          "required CPU Neg/Cast and CUDA paged suffix differ from independent two-row double oracle"));
#else
      for(auto word:actual)HYLE_RETURN_IF_ERROR(require(word==0x3400,"CUDA pending Graph actual mean differs"));
#endif
      HYLE_RETURN_IF_ERROR(require(!receiver.take_update(),"old CUDA Graph output permitted source reuse"));}
    submitted={};HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
    if(mode==Mode::cancel_before){
      HYLE_RETURN_IF_ERROR(require(!update.take_mutable({1}) && guards.active_batches()==0,
          "closed original source escaped its slot or retained native guards after retirement"));
      std::fprintf(stderr,"CUDA pending Graph source cancel refusal preserved actual work/result; original closed source refuses mutable withdrawal PASS\n");
      return Status::Ok();
    }
    HYLE_ASSIGN_OR_RETURN(kw.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(vw.lease(),update.take_mutable({2}));
    HYLE_ASSIGN_OR_RETURN(auto key,kw.publish(2));HYLE_ASSIGN_OR_RETURN(auto value,vw.publish(2));
    HYLE_RETURN_IF_ERROR(require(keys.read(layout,std::span(&key,1)).has_value() && values.read(layout,std::span(&value,1)).has_value(),
        "CUDA Graph successful original page controls did not recover"));
    std::fprintf(stderr,"CUDA pending Graph actual gated numerical suffix, old output and original page writer recovery; cancel refusal=%d PASS\n",mode==Mode::cancel_after);return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(require(!fail_source && terminal.IsResourceExhausted() && !receiver.output(output_id),"source failure was hidden by successful original native suffix"));
  submitted={};
  HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
  HYLE_RETURN_IF_ERROR(require(!update.take_mutable({1}) && receiver.prepare_batch(versions).IsFailedPrecondition(),"late failed source/epoch fabricated ready output or reuse"));
  std::fprintf(stderr,"CUDA pending Graph source risk=%u unknown/late source failure merges original terminal and refuses output/reuse PASS\n",unsigned(mode));return Status::Ok();


  }();
  if(!status.ok())std::fprintf(stderr,"risk observed before cleanup: %.*s\n",int(status.message().size()),status.message().data());
  gate.release();context.drain().IgnoreError();
#if defined(HYLE_PENDING_CPU_CUDA_GRAPH_RISKS)
  cpu_context.drain().IgnoreError();
#endif
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while((metadata.used() || source_metadata.used()) && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  HYLE_RETURN_IF_ERROR(require(metadata.used()==0 && source_metadata.used()==0,"risk teardown did not retire original metadata"));
  return status;

}
}
int main(){bool failed=false;for(auto mode:{Mode::success,Mode::known_source_error,Mode::unknown_before_acceptance,Mode::late_source_error,Mode::abandon,Mode::cpu_foreign_queue,Mode::cancel_before,Mode::cancel_after}){const auto status=run(mode);if(!status.ok()){
  std::fprintf(stderr,"CUDA pending Graph risk=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());failed=true;
}}return failed?1:0;}
