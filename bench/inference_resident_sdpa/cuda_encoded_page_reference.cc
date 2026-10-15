#include "cuda_encoded_page_reference.h"
#include "cuda_encoded_page_native.h"
#include "../../src/compute/provider_internal.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/providers/flashinfer.h"
#include "native_context.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <new>
#include <vector>

namespace hyle::bench::resident {
namespace c=compute;namespace d=c::detail;namespace n=cuda_original;
namespace {
bool same_layout(const c::BatchLayoutVersion &a,const c::BatchLayoutVersion &b) noexcept {
  if(!a.valid()||!a.schema().owns(b))return false;
  const auto x=a.host(),y=b.host();
  const auto eq=[](auto u,auto v){return u.size()==v.size()&&std::equal(u.begin(),u.end(),v.begin());};
  return eq(x.query_indptr,y.query_indptr)&&eq(x.kv_indptr,y.kv_indptr)&&
      eq(x.page_indices,y.page_indices)&&eq(x.last_page_lengths,y.last_page_lengths);
}
struct Context {const CudaEncodedReferenceData *data=nullptr;CUcontext primary=nullptr;};
struct Slot {
  Context context;c::BatchCapacity capacity;c::BatchProgramOptions options;
  c::BatchLayoutVersion layout;n::Spec spec;n::Facts facts;void *plan=nullptr;
  ~Slot(){n::destroy_plan(plan);}
};
Status qualify(void *opaque,c::BatchCapacity cap,const c::BatchProgramOptions &options) noexcept {
  const auto *ctx=static_cast<Context *>(opaque);
  if(!ctx||!ctx->data||!ctx->data->ready||!ctx->primary||cap!=ctx->data->layout.schema().capacity())
    return Status::FailedPrecondition("original CUDA immutable snapshot is not ready");
  if(cap.storage!=c::BatchStorage::kPaged||cap.query_type!=c::DataType::kBFloat16||
      cap.output_type!=cap.query_type||cap.key_storage!=c::CacheStorage::float8(.5F).value()||
      cap.value_storage!=c::CacheStorage::float8(2.F).value()||cap.query_heads!=8||cap.kv_heads!=2||
      cap.page_size!=16||!((cap.qk_head_dim==256&&cap.value_head_dim==256&&options.causal&&
      (options.window_left==-1||options.window_left==16))||
      (cap.qk_head_dim==64&&cap.value_head_dim==128&&!options.causal&&options.window_left==-1)))
    return Status::Unimplemented("finite original CUDA expanded BF16 comparator");
  return Status::Ok();
}
Status create(void *ctx,c::BatchCapacity cap,const c::BatchProgramOptions &options,
    mem::AllocatorRef allocator,void **out) noexcept {
  *out=nullptr;HYLE_RETURN_IF_ERROR(qualify(ctx,cap,options));
  auto *s=allocator.construct<Slot>();if(!s)return Status::ResourceExhausted("original CUDA slot");
  s->context=*static_cast<Context *>(ctx);s->capacity=cap;s->options=options;
  const auto offsets=s->context.data->layout.schema().metadata_offsets();
  s->spec={.query_heads=cap.query_heads,.kv_heads=cap.kv_heads,.key_depth=cap.qk_head_dim,
      .value_depth=cap.value_head_dim,.page_size=cap.page_size,
      .query_offset=offsets.query_indptr,.kv_offset=offsets.kv_indptr,
      .indices_offset=offsets.page_indices,.tails_offset=offsets.last_page_lengths,
      .scale=options.scale,.window=options.window_left,.causal=options.causal};
  try{s->plan=n::create_plan();*out=s;return Status::Ok();}
  catch(const std::bad_alloc &){allocator.destroy(s);return Status::ResourceExhausted("original CUDA plan");}
}
void destroy(void *p,mem::AllocatorRef a) noexcept {a.destroy(static_cast<Slot *>(p));}
c::BatchPlanReport report(const Slot &s,bool reused=false) noexcept {
  return {.layout_generation=s.layout.generation(),.float_workspace_bytes=s.facts.floating,
      .int_workspace_bytes=s.facts.integer,.cta_tile_q=s.facts.tile,.padded_batch_size=s.facts.padded,
      .split_kv=s.facts.split,.reused_native_plan=reused};
}
Result<bool> update(void *p,const c::BatchLayoutVersion &layout,c::BatchPlanReport *r) noexcept {
  auto &s=*static_cast<Slot *>(p);
  if(!same_layout(s.layout,layout)||!same_layout(s.context.data->layout,layout))return false;
  s.layout=layout;*r=report(s,true);return true;
}
d::EnqueueResult plan(void *p,c::Queue &queue,const c::BatchLayoutVersion &layout,
    d::BatchWorkspaceViews views,c::BatchPlanReport *r) noexcept {
  auto &s=*static_cast<Slot *>(p);s.layout={};
  if(!same_layout(s.context.data->layout,layout))
    return d::EnqueueResult::rejected(Status::InvalidArgument("original CUDA snapshot topology changed"));
  auto stream=c::cuda::stream(queue);if(!stream)return d::EnqueueResult::rejected(stream.error());
  auto f=c::cuda::detail::mutable_pointer(views.float_workspace,16,1);
  if(!f)return d::EnqueueResult::rejected(f.error());
  auto i=c::cuda::detail::mutable_pointer(views.int_workspace,16,1);
  if(!i)return d::EnqueueResult::rejected(i.error());
  auto host=views.host_workspace.map_host();if(!host)return d::EnqueueResult::rejected(host.error());
  auto bytes=host->writable_bytes();if(!bytes)return d::EnqueueResult::rejected(bytes.error());
  c::cuda::detail::ScopedPrimaryContext scope(s.context.primary);
  if(!scope)return d::EnqueueResult::rejected(scope.status());
  const auto h=layout.host();auto spec=s.spec;spec.requests=layout.facts().requests;spec.query_rows=layout.facts().query_rows;
  bool accepted=false;
  try{
    n::Facts facts;
    auto status=c::cuda::status(n::workspace(spec,h.query_indptr.data(),h.kv_indptr.data(),*stream,facts));
    if(!status.ok())return d::EnqueueResult::rejected(status);
    if(facts.floating>views.float_workspace.size()||facts.integer>views.int_workspace.size()||facts.integer>bytes->size())
      return d::EnqueueResult::rejected(Status::ResourceExhausted("original CUDA plan workspace"));
    accepted=true; // The original planner may enqueue an H2D prefix before failing.
    status=c::cuda::status(n::plan(s.plan,spec,h.query_indptr.data(),h.kv_indptr.data(),
        *f,views.float_workspace.size(),*i,bytes->data(),views.int_workspace.size(),*stream,facts));
    if(status.ok()){s.spec=spec;s.facts=facts;s.layout=layout;*r=report(s);}
    return {accepted,status};
  }catch(const std::bad_alloc &){return {accepted,Status::ResourceExhausted("original CUDA planner allocation")};}
   catch(const std::exception &e){return {accepted,Status::Internal(e.what())};}
}
d::EnqueueResult run(void *p,c::Queue &queue,std::span<const c::ConstTensorView> inputs,
    d::BatchWorkspaceViews views) noexcept {
  auto &s=*static_cast<Slot *>(p);
  if(!s.layout.valid()||inputs.size()!=3)return d::EnqueueResult::rejected(Status::FailedPrecondition("original CUDA layout missing"));
  for(unsigned port=1;port<3;++port){
    auto pinned=d::PagedArenaAccess::tensor(port==1?s.context.data->keys:s.context.data->values);
    if(!pinned)return d::EnqueueResult::rejected(pinned.error());
    const auto a=inputs[port].storage(),b=pinned->storage();
    if(a.resource()!=b.resource()||a.native_handle()!=b.native_handle()||a.offset()!=b.offset()||a.size()!=b.size())
      return d::EnqueueResult::rejected(Status::InvalidArgument("original CUDA immutable FP8 frame identity changed"));
  }
  auto stream=c::cuda::stream(queue);if(!stream)return d::EnqueueResult::rejected(stream.error());
  auto q=c::cuda::detail::const_pointer(inputs[0].storage(),16,2);if(!q)return d::EnqueueResult::rejected(q.error());
  auto k=c::cuda::detail::const_pointer(s.context.data->expanded[0].view(),16,2);if(!k)return d::EnqueueResult::rejected(k.error());
  auto v=c::cuda::detail::const_pointer(s.context.data->expanded[1].view(),16,2);if(!v)return d::EnqueueResult::rejected(v.error());
  auto m=c::cuda::detail::const_pointer(views.metadata,8,1);if(!m)return d::EnqueueResult::rejected(m.error());
  auto f=c::cuda::detail::mutable_pointer(views.float_workspace,16,1);if(!f)return d::EnqueueResult::rejected(f.error());
  auto i=c::cuda::detail::mutable_pointer(views.int_workspace,16,1);if(!i)return d::EnqueueResult::rejected(i.error());
  auto o=c::cuda::detail::mutable_pointer(views.output,16,2);if(!o)return d::EnqueueResult::rejected(o.error());
  c::cuda::detail::ScopedPrimaryContext scope(s.context.primary);
  if(!scope)return d::EnqueueResult::rejected(scope.status());
  try{return d::EnqueueResult::accepted_command(c::cuda::status(n::run(s.plan,s.spec,*q,*k,*v,*m,*o,*f,*i,*stream)));}
  catch(const std::bad_alloc &){return d::EnqueueResult::accepted_command(Status::ResourceExhausted("original CUDA run allocation"));}
  catch(const std::exception &e){return d::EnqueueResult::accepted_command(Status::Internal(e.what()));}
}
const d::BatchProviderVTable batch{&qualify,&create,&destroy,&update,&plan,&run};
struct Config {const CudaEncodedReferenceData *data;};
const d::ProviderVTable &table(){static const auto t=[]{
  auto base=c::flashinfer::provider();auto t=*d::ProviderAccess::vtable(base);
  t.identity={.id={0x48594c4542454e43ULL,0x43554441454e4301ULL},.name="original-cuda-cold-expanded-cache",.version="69ff11fc-pristine-bf16"};
  t.create_context=[](const c::Provider &provider,c::Device device,void *backend,mem::AllocatorRef a,void **out) noexcept {
    *out=nullptr;if(device.backend_id()!=c::cuda::kBackendId)return Status::Unimplemented("original CUDA device");
    const auto *env=c::cuda::detail::borrow_environment(backend);
    const auto primary=c::cuda::detail::borrow_primary_context(backend);
    if(!primary||!env||env->sm_major!=8||env->sm_minor!=9)return Status::Unimplemented("original CUDA finite SM89 profile");
    const auto *data=d::ProviderAccess::payload<Config>(provider).data;
    if(!data)return Status::InvalidArgument("original CUDA data missing");
    auto *ctx=a.construct<Context>(Context{data,primary});
    if(!ctx)return Status::ResourceExhausted("original CUDA provider context");
    *out=ctx;return Status::Ok();
  };
  t.destroy_context=[](void *p,mem::AllocatorRef a) noexcept {a.destroy(static_cast<Context *>(p));};
  t.drain_context=[](void *) noexcept {return Status::Ok();};
  t.propose=[](void *,const d::ProviderOperationRequest &,std::span<d::ResourceBindingRequirement>,std::span<d::ResourceBindingRequirement>,d::ProviderOperationProposal *) noexcept {return Status::Unimplemented("original CUDA batch-only comparator");};
  t.prepare=[](void *,const d::ProviderPrepareRequest &,mem::AllocatorRef,d::ProviderPrepared *) noexcept {return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("original CUDA batch only"));};
  t.query_extension=[](void *,c::ProviderExtensionId id) noexcept ->const void * {return id==d::kBatchPlanningExtensionId?&batch:nullptr;};
  return t;
}();return t;}
}
c::Provider cuda_encoded_reference_provider(const CudaEncodedReferenceData *data) noexcept {
  return d::ProviderAccess::make(&table(),Config{data});
}
Status prepare_cuda_encoded_reference(CudaEncodedReferenceData &data,c::Queue &queue,
    c::BatchLayoutVersion layout,c::PagedArenaRead keys,c::PagedArenaRead values,
    mem::MemoryResourceRef resource) noexcept {
  if(data.ready||!layout.valid()||keys.layout()!=layout||values.layout()!=layout)
    return Status::InvalidArgument("original CUDA cold exact frames");
  const auto start=std::chrono::steady_clock::now();
  try{
    const auto cap=layout.schema().capacity();
    HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(keys,cap.key_storage));
    HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(values,cap.value_storage));
    data.layout=std::move(layout);data.keys=std::move(keys);data.values=std::move(values);
    const auto h=data.layout.host();std::vector<std::uint32_t> valid(cap.max_pages);
    for(unsigned r=0;r<data.layout.facts().requests;++r)
      for(auto ref=h.kv_indptr[r];ref<h.kv_indptr[r+1];++ref)
        valid[unsigned(h.page_indices[ref])]=std::max(valid[unsigned(h.page_indices[ref])],
            ref+1==h.kv_indptr[r+1]?unsigned(h.last_page_lengths[r]):cap.page_size);
    HYLE_ASSIGN_OR_RETURN(auto valid_buffer,mem::Buffer::create(resource,valid.size()*sizeof(valid[0]),256));
    HYLE_RETURN_IF_ERROR(valid_buffer.view().write(std::as_bytes(std::span(valid))));
    HYLE_ASSIGN_OR_RETURN(auto valid_ptr,c::cuda::pointer<std::uint32_t>(mem::ConstBufferView(valid_buffer.view())));
    HYLE_ASSIGN_OR_RETURN(auto uses_pool,c::LeaseUsePool::create({.max_batches=1,
        .max_resources_per_batch=unsigned(d::PagedArenaAccess::pages(data.keys).size()+d::PagedArenaAccess::pages(data.values).size())}));
    HYLE_ASSIGN_OR_RETURN(auto uses,uses_pool.begin());
    for(const auto &page:d::PagedArenaAccess::pages(data.keys))HYLE_RETURN_IF_ERROR(uses.read(page));
    for(const auto &page:d::PagedArenaAccess::pages(data.values))HYLE_RETURN_IF_ERROR(uses.read(page));
    std::array<const void *,2> source{};std::array<void *,2> dest{};
    std::size_t capacity_bytes=0,read_bytes=0;
    for(unsigned port=0;port<2;++port){
      const auto depth=port?cap.value_head_dim:cap.qk_head_dim;
      const auto count=std::size_t(cap.max_pages)*cap.page_size*cap.kv_heads*depth;
      HYLE_ASSIGN_OR_RETURN(data.expanded[port],mem::Buffer::create(resource,count*2,256));
      HYLE_ASSIGN_OR_RETURN(auto tensor,d::PagedArenaAccess::tensor(port?data.values:data.keys));
      HYLE_ASSIGN_OR_RETURN(source[port],c::cuda::detail::const_pointer(tensor.storage(),16,1));
      HYLE_ASSIGN_OR_RETURN(dest[port],c::cuda::detail::mutable_pointer(data.expanded[port].view(),16,2));
      capacity_bytes+=count*2;
      for(auto rows:valid)read_bytes+=std::size_t(rows)*cap.kv_heads*depth;
    }
    auto completion=c::cuda::submit(queue,[&](cudaStream_t stream) noexcept ->cudaError_t {
      for(unsigned port=0;port<2;++port){
        // Define padding in our own alternate representation without reading
        // any unborrowed FP8 row. Original masked tiles may load that padding.
        auto status=cudaMemsetAsync(dest[port],0,data.expanded[port].size(),stream);
        if(status!=cudaSuccess)return status;
        status=n::decode(source[port],dest[port],valid_ptr,
          cap.max_pages,cap.page_size,cap.kv_heads,port?cap.value_head_dim:cap.qk_head_dim,
          port?cap.value_storage.scale():cap.key_storage.scale(),stream);
        if(status!=cudaSuccess)return status;
      }return cudaSuccess;
    });
    if(!completion){Status status=completion.error();status.Update(queue.drain());return status;}
    auto status=uses.attach(*completion);
    status.Update(completion->wait()); // Also drains an accepted decoder prefix on error.
    HYLE_RETURN_IF_ERROR(status);
    data.ready=true;
    std::fprintf(stderr,"# CUDA_ENCODED_ORIGINAL_COLD extra_cache_bytes=%zu decoder_read_bytes=%zu decoder_write_bytes=%zu zeroed_owned_bytes=%zu valid_table_bytes=%zu seconds=%.9f shared_snapshot_slots=2 original_fp8_owners=retained private_pool_upper=UNKNOWN\n",
        capacity_bytes,read_bytes,read_bytes*2,capacity_bytes,valid_buffer.size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    return Status::Ok();
  }catch(const std::bad_alloc &){return Status::ResourceExhausted("original CUDA cold snapshot allocation");}
}
}
