#include "cpu_paged_reference.h"
#include "../../src/compute/provider_internal.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/backends/cpu.h"
#include "cpu_paged_original_execution.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include <cstdio>
#include <cstring>
#include "metadata_arena.h"
#include "ggml.h"
#include "hyle/mem/buffer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace hyle::bench::resident {
namespace c=compute;namespace d=c::detail;namespace n=c::ggml_runtime;
namespace {
struct Request {
  n::MetadataArena arena;
  ggml_cgraph *graph=nullptr;
  std::array<ggml_tensor *,3> inputs{};
  ggml_tensor *output=nullptr;
  std::vector<ggml_tensor *> views;
  std::vector<mem::Buffer> payload;
  std::vector<mem::HostMapping> mappings;
  std::vector<ggml_fp16_t> mask;
  CpuPagedOriginalExecution execution;
  std::size_t qoffset=0,koffset=0,voffset=0,ooffset=0;
  std::size_t qbytes=0,kbytes=0,vbytes=0,obytes=0,known=0;
};
struct Slot {
  c::BatchCapacity capacity; c::BatchProgramOptions options;mem::AllocatorRef allocator;
  const CpuPagedReferenceData *data=nullptr; c::BatchLayoutVersion layout;std::vector<std::unique_ptr<Request>> requests;std::size_t known=0;
};
Status qualify(void *context,c::BatchCapacity cap,const c::BatchProgramOptions &options)noexcept{
  const auto *data=static_cast<const CpuPagedReferenceData *>(context);
  if(!data || cap!=data->layout.schema().capacity())
    return Status::InvalidArgument("original immutable cache capacity");
  const bool plain=cap.key_storage==c::CacheStorage::plain(c::DataType::kBFloat16).value()&&
      cap.value_storage==cap.key_storage&&cap.qk_head_dim==cap.value_head_dim;
  const bool encoded=cap.key_storage==c::CacheStorage::q8_0().value()&&
      cap.value_storage==cap.key_storage;
  if(cap.storage!=c::BatchStorage::kPaged||cap.query_type!=c::DataType::kBFloat16||
     cap.output_type!=c::DataType::kBFloat16||(!plain&&!encoded)||
     cap.query_heads!=8||cap.kv_heads!=2||
     !((cap.qk_head_dim==64&&(cap.value_head_dim==64||cap.value_head_dim==128))||
       (cap.qk_head_dim==128&&cap.value_head_dim==128)||
       (cap.qk_head_dim==256&&cap.value_head_dim==256))||
     !options.resource.properties().has(mem::MemoryProperty::kHostAccessible))
    return Status::Unimplemented("finite original immutable BF16 contiguous-cache comparator");
  return Status::Ok();
}
Status create(void *context,c::BatchCapacity cap,const c::BatchProgramOptions &options,mem::AllocatorRef a,void **out)noexcept{
  *out=nullptr;HYLE_RETURN_IF_ERROR(qualify(context,cap,options));auto *s=a.construct<Slot>();
  if(!s)return Status::ResourceExhausted("original ragged slot");s->data=static_cast<const CpuPagedReferenceData *>(context);s->capacity=cap;s->options=options;s->allocator=a;*out=s;return Status::Ok();
}
void destroy(void *p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Slot *>(p));}
c::BatchPlanReport report(const Slot &s,bool reused=false)noexcept{return {.layout_generation=s.layout.generation(),.float_workspace_bytes=s.known,.native_metadata_unknown=true,.reused_native_plan=reused};}
Result<bool> update(void *p,const c::BatchLayoutVersion &next,c::BatchPlanReport *r)noexcept{
  auto &s=*static_cast<Slot *>(p);if(!s.layout.valid()||!s.layout.schema().owns(next))return false;
  const auto equal=[](auto a,auto b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());};
  if(!equal(s.layout.host().query_indptr,next.host().query_indptr)||!equal(s.layout.host().kv_indptr,next.host().kv_indptr)||!equal(s.layout.host().page_indices,next.host().page_indices)||!equal(s.layout.host().last_page_lengths,next.host().last_page_lengths))return false;
  s.layout=next;*r=report(s,true);return true;
}
Status make_request(Slot &slot,std::int32_t nq,std::int32_t nk,std::int32_t qstart,std::int32_t kstart) {
  auto request=std::make_unique<Request>();auto &s=*request;const auto cap=slot.capacity;
  HYLE_ASSIGN_OR_RETURN(s.arena,n::MetadataArena::for_graph(32,32,slot.allocator));auto *ctx=s.arena.get();
  const bool encoded=!cap.key_storage.format().is_plain();
  for(unsigned port=0;port<3;++port){
    const auto type=port&&encoded?GGML_TYPE_Q8_0:GGML_TYPE_BF16;
    s.inputs[port]=ggml_new_tensor_4d(ctx,type,port==2?cap.value_head_dim:cap.qk_head_dim,
        port?cap.kv_heads:cap.query_heads,port?nk:nq,1);ggml_set_input(s.inputs[port]);}
  auto *q=ggml_cast(ctx,ggml_permute(ctx,s.inputs[0],0,2,1,3),GGML_TYPE_F32);
  auto *k=ggml_permute(ctx,s.inputs[1],0,2,1,3),*v=ggml_permute(ctx,s.inputs[2],0,2,1,3);
  ggml_tensor *mask=nullptr;
  if(slot.options.causal){s.mask.resize(std::size_t(nq)*nk);for(int row=0;row<nq;++row)for(int j=0;j<nk;++j){
      const auto position=nk-nq+row;
      const bool outside=j>position||(slot.options.window_left>=0&&j<std::max(0,position-slot.options.window_left));
      s.mask[std::size_t(row)*nk+j]=ggml_fp32_to_fp16(outside?-INFINITY:0.F);}
    mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,nk,nq);mask->data=s.mask.data();}
  auto *attention=ggml_flash_attn_ext(ctx,q,k,v,mask,slot.options.scale,0.F,0.F);
  if(!ggml_prec_set_acc(attention,GGML_PREC_F32))return Status::Unimplemented("original flash F32 accumulation");
  // Original flash output is already [D,H,Q,1], the packed NHD caller shape.
  s.output=ggml_cast(ctx,attention,GGML_TYPE_BF16);ggml_set_output(s.output);
  s.graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(s.graph,s.output);
  for(auto *t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)){
    if(t->view_src){s.views.push_back(t);continue;}
    if(t->data||t==s.output||t==s.inputs[0]||t==s.inputs[1]||t==s.inputs[2])continue;
    HYLE_ASSIGN_OR_RETURN(auto buffer,mem::Buffer::create(mem::system_resource(),ggml_nbytes(t),GGML_MEM_ALIGN));
    HYLE_ASSIGN_OR_RETURN(auto mapping,buffer.view().map_host(mem::HostAccess::kReadWrite));HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());t->data=bytes.data();
    s.known+=buffer.size();s.payload.push_back(std::move(buffer));s.mappings.push_back(std::move(mapping));
  }
  HYLE_RETURN_IF_ERROR(s.execution.prepare(s.graph,slot.allocator));
  s.known+=s.arena.capacity()+s.execution.plan.work_size+s.mask.size()*2;
  const auto key_row=std::size_t(cap.kv_heads)*ggml_row_size(encoded?GGML_TYPE_Q8_0:GGML_TYPE_BF16,cap.qk_head_dim);
  const auto value_row=std::size_t(cap.kv_heads)*ggml_row_size(encoded?GGML_TYPE_Q8_0:GGML_TYPE_BF16,cap.value_head_dim);
  s.qoffset=std::size_t(qstart)*cap.query_heads*cap.qk_head_dim*2;s.koffset=std::size_t(kstart)*key_row;
  s.voffset=std::size_t(kstart)*value_row;s.ooffset=std::size_t(qstart)*cap.query_heads*cap.value_head_dim*2;
  s.qbytes=std::size_t(nq)*cap.query_heads*cap.qk_head_dim*2;s.kbytes=std::size_t(nk)*key_row;
  s.vbytes=std::size_t(nk)*value_row;s.obytes=std::size_t(nq)*cap.query_heads*cap.value_head_dim*2;
  slot.known+=s.known;slot.requests.push_back(std::move(request));return Status::Ok();
}
d::EnqueueResult plan(void *p,d::QueueSequence &,const c::BatchLayoutVersion &layout,d::BatchWorkspaceViews,c::BatchPlanReport *r)noexcept{
  auto &s=*static_cast<Slot *>(p);s.requests.clear();s.layout={};s.known=0;
  try{const auto h=layout.host();const auto fixed=s.data->layout.host();const auto eq=[](auto a,auto b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());};if(!s.data->layout.schema().owns(layout)||!eq(h.query_indptr,fixed.query_indptr)||!eq(h.kv_indptr,fixed.kv_indptr)||!eq(h.page_indices,fixed.page_indices)||!eq(h.last_page_lengths,fixed.last_page_lengths))return d::EnqueueResult::rejected(Status::Unimplemented("original peer immutable topology"));for(unsigned i=0;i<layout.facts().requests;++i){const auto nq=h.query_indptr[i+1]-h.query_indptr[i],refs=h.kv_indptr[i+1]-h.kv_indptr[i];const auto nk=refs?(refs-1)*static_cast<std::int32_t>(s.capacity.page_size)+h.last_page_lengths[i]:0;if(!nq)continue;
    if(!nk||(s.options.causal&&nk<nq))return d::EnqueueResult::rejected(Status::Unimplemented("original ragged empty/short KV"));
    std::int32_t kstart=0;for(unsigned j=0;j<i;++j){const auto refs_before=h.kv_indptr[j+1]-h.kv_indptr[j];kstart+=refs_before?(refs_before-1)*static_cast<std::int32_t>(s.capacity.page_size)+h.last_page_lengths[j]:0;}auto status=make_request(s,nq,nk,h.query_indptr[i],kstart);if(!status.ok()){s.requests.clear();return d::EnqueueResult::rejected(status);}}
    if(s.known>s.options.max_float_workspace_bytes)return d::EnqueueResult::rejected(Status::ResourceExhausted("original ragged known bytes"));
    s.layout=layout;*r=report(s);return d::EnqueueResult::accepted_command();
  }catch(const std::bad_alloc &){s.requests.clear();return d::EnqueueResult::rejected(Status::ResourceExhausted("original ragged cold allocations"));}
}
d::EnqueueResult run(void *p,d::QueueSequence &,std::span<const c::ConstTensorView> inputs,d::BatchWorkspaceViews workspace)noexcept{
  auto &slot=*static_cast<Slot *>(p);bool accepted=false;
  for(unsigned port=1;port<3;++port){
    auto pinned=d::PagedArenaAccess::tensor(port==1?slot.data->keys:slot.data->values);
    if(!pinned)return d::EnqueueResult::rejected(pinned.error());
    const auto actual=inputs[port].storage(),original=pinned->storage();
    if(actual.resource()!=original.resource()||actual.native_handle()!=original.native_handle()||actual.offset()!=original.offset()||actual.size()!=original.size())
      return d::EnqueueResult::rejected(Status::InvalidArgument("original peer requires its immutable pinned history"));
  }
  for(auto &owned:slot.requests){auto &s=*owned;std::array<mem::HostMapping,3> maps;
    for(unsigned port=0;port<3;++port){auto sub=(port?(port==1?slot.data->key_cache.view():slot.data->value_cache.view()):inputs[0].storage()).subview(port==0?s.qoffset:port==1?s.koffset:s.voffset,port==0?s.qbytes:port==1?s.kbytes:s.vbytes);if(!sub)return {accepted,sub.error()};auto mapped=sub->map_host();if(!mapped)return {accepted,mapped.error()};maps[port]=std::move(*mapped);s.inputs[port]->data=const_cast<std::byte *>(maps[port].bytes().data());}
    auto sub=workspace.output.subview(s.ooffset,s.obytes);if(!sub)return {accepted,sub.error()};auto mapped=sub->map_host(mem::HostAccess::kWrite);if(!mapped)return {accepted,mapped.error()};auto bytes=mapped->writable_bytes();if(!bytes)return {accepted,bytes.error()};s.output->data=bytes->data();
    for(auto *view:s.views)view->data=static_cast<std::byte *>(view->view_src->data)+view->view_offs;
    auto status=s.execution.run();accepted=true;status.Update(mapped->close());for(auto &map:maps)status.Update(map.close());if(!status.ok())return {accepted,status};
  }return d::EnqueueResult::accepted_command();
}
const d::BatchProviderVTable batch{&qualify,&create,&destroy,&update,&plan,&run};
struct Configuration final {const CpuPagedReferenceData *data;};
const d::ProviderVTable &decorated(){static const auto result=[](){auto base=c::ggml_cpu::provider({.native_threads=1});auto table=*d::ProviderAccess::vtable(base);
  table.identity={.id=c::ProviderId{0x48594c4542454e43ULL,0x5041474544430001ULL},.name="original-cpu-contiguous-immutable-cache",.version="ffa4e8b8-pristine"};
  table.create_context=[](const c::Provider &provider,c::Device device,void *,mem::AllocatorRef,void **out)noexcept{
    if(device.backend_id()!=c::cpu::kBackendId)return Status::Unimplemented("original CPU-only comparator");
    *out=const_cast<CpuPagedReferenceData *>(d::ProviderAccess::payload<Configuration>(provider).data);return *out?Status::Ok():Status::InvalidArgument("original immutable cache missing");
  };
  table.destroy_context=[](void *,mem::AllocatorRef)noexcept{};
  table.drain_context=[](void *)noexcept{return Status::Ok();};
  table.propose=[](void *,const d::ProviderOperationRequest &,std::span<d::ResourceBindingRequirement>,std::span<d::ResourceBindingRequirement>,d::ProviderOperationProposal *)noexcept{return Status::Unimplemented("private comparator only supports its batch contract");};
  table.prepare=[](void *,const d::ProviderPrepareRequest &,mem::AllocatorRef,d::ProviderPrepared *)noexcept{return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("private comparator batch only"));};
  table.query_extension=[](void *,c::ProviderExtensionId id)noexcept->const void *{return id==d::kBatchPlanningExtensionId?&batch:nullptr;};return table;}();return result;}
}
c::Provider cpu_paged_reference_provider(const CpuPagedReferenceData *data)noexcept{return d::ProviderAccess::make(&decorated(),Configuration{data});}

Result<std::shared_ptr<const CpuPagedReferenceData>> cpu_paged_reference_data(
    c::BatchLayoutVersion layout,c::PagedArenaRead keys,c::PagedArenaRead values)noexcept{
  if(!layout.valid() || keys.layout()!=layout || values.layout()!=layout)
    return std::unexpected(Status::InvalidArgument("original cache exact immutable frames"));
  try{
    auto data=std::make_shared<CpuPagedReferenceData>();data->layout=std::move(layout);data->keys=std::move(keys);data->values=std::move(values);
    const auto cap=data->layout.schema().capacity();const auto h=data->layout.host();
    const auto row_bytes=[&](unsigned port){return std::size_t(cap.kv_heads)*ggml_row_size(
        (port==1?cap.key_storage:cap.value_storage).format().is_plain()?GGML_TYPE_BF16:GGML_TYPE_Q8_0,
        port==1?cap.qk_head_dim:cap.value_head_dim);};
    HYLE_ASSIGN_OR_RETURN(data->key_cache,mem::Buffer::create(mem::system_resource(),std::size_t(data->layout.facts().kv_rows)*row_bytes(1),GGML_MEM_ALIGN));
    HYLE_ASSIGN_OR_RETURN(data->value_cache,mem::Buffer::create(mem::system_resource(),std::size_t(data->layout.facts().kv_rows)*row_bytes(2),GGML_MEM_ALIGN));
    for(unsigned port=1;port<3;++port){
      HYLE_ASSIGN_OR_RETURN(auto tensor,d::PagedArenaAccess::tensor(port==1?data->keys:data->values));
      HYLE_ASSIGN_OR_RETURN(auto source,tensor.storage().map_host());
      HYLE_ASSIGN_OR_RETURN(auto dest,(port==1?data->key_cache.view():data->value_cache.view()).map_host(mem::HostAccess::kWrite));
      HYLE_ASSIGN_OR_RETURN(auto bytes,dest.writable_bytes());std::size_t offset=0;
      for(unsigned request=0;request<data->layout.facts().requests;++request){const auto first=h.kv_indptr[request],last=h.kv_indptr[request+1];
        for(auto ref=first;ref<last;++ref){const auto count=ref+1==last?static_cast<unsigned>(h.last_page_lengths[request]):cap.page_size;
          std::memcpy(bytes.data()+offset,source.bytes().data()+std::size_t(h.page_indices[ref])*cap.page_size*row_bytes(port),count*row_bytes(port));offset+=count*row_bytes(port);
        }
      }
      if(offset!=bytes.size())return std::unexpected(Status::Internal("original cold immutable cache extent"));
      HYLE_RETURN_IF_ERROR(dest.close());HYLE_RETURN_IF_ERROR(source.close());
    }
    std::fprintf(stderr,"# CPU_PAGED_ORIGINAL_COLD copied_bytes=%zu contiguous_cache_bytes=%zu private_pool_upper=UNKNOWN immutable_page_owners=retained warm_kv_copy=0\n",data->key_cache.size()+data->value_cache.size(),data->key_cache.size()+data->value_cache.size());
    return std::shared_ptr<const CpuPagedReferenceData>(std::move(data));
  }catch(const std::bad_alloc &){return std::unexpected(Status::ResourceExhausted("original immutable cache preparation"));}
}
}
