#include "cpu_ragged_reference.h"
#include "../../src/compute/provider_internal.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "cpu_execution.h"
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
  n::CpuExecution execution;
  std::size_t qoffset=0,koffset=0,qbytes=0,kbytes=0,known=0;
};
struct Slot {
  c::BatchCapacity capacity; c::BatchProgramOptions options;mem::AllocatorRef allocator;
  c::BatchLayoutVersion layout;std::vector<std::unique_ptr<Request>> requests;std::size_t known=0;
};
Status qualify(void *,c::BatchCapacity cap,const c::BatchProgramOptions &options)noexcept{
  if(cap.storage!=c::BatchStorage::kRagged||cap.query_type!=c::DataType::kBFloat16||
     cap.query_heads!=8||cap.kv_heads!=2||(cap.qk_head_dim!=64&&cap.qk_head_dim!=256)||
     !options.resource.properties().has(mem::MemoryProperty::kHostAccessible))
    return Status::Unimplemented("finite original BF16 ragged flash comparator");
  return Status::Ok();
}
Status create(void *,c::BatchCapacity cap,const c::BatchProgramOptions &options,mem::AllocatorRef a,void **out)noexcept{
  *out=nullptr;HYLE_RETURN_IF_ERROR(qualify(nullptr,cap,options));auto *s=a.construct<Slot>();
  if(!s)return Status::ResourceExhausted("original ragged slot");s->capacity=cap;s->options=options;s->allocator=a;*out=s;return Status::Ok();
}
void destroy(void *p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Slot *>(p));}
c::BatchPlanReport report(const Slot &s,bool reused=false)noexcept{return {.layout_generation=s.layout.generation(),.float_workspace_bytes=s.known,.native_metadata_unknown=true,.reused_native_plan=reused};}
Result<bool> update(void *p,const c::BatchLayoutVersion &next,c::BatchPlanReport *r)noexcept{
  auto &s=*static_cast<Slot *>(p);if(!s.layout.valid()||!s.layout.schema().owns(next))return false;
  const auto equal=[](auto a,auto b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());};
  if(!equal(s.layout.host().query_indptr,next.host().query_indptr)||!equal(s.layout.host().kv_indptr,next.host().kv_indptr))return false;
  s.layout=next;*r=report(s,true);return true;
}
Status make_request(Slot &slot,std::int32_t nq,std::int32_t nk,std::int32_t qstart,std::int32_t kstart) {
  auto request=std::make_unique<Request>();auto &s=*request;const auto cap=slot.capacity;
  HYLE_ASSIGN_OR_RETURN(s.arena,n::MetadataArena::for_graph(32,32,slot.allocator));auto *ctx=s.arena.get();
  for(unsigned port=0;port<3;++port){s.inputs[port]=ggml_new_tensor_4d(ctx,GGML_TYPE_BF16,cap.qk_head_dim,port?cap.kv_heads:cap.query_heads,port?nk:nq,1);ggml_set_input(s.inputs[port]);}
  auto *q=ggml_cast(ctx,ggml_permute(ctx,s.inputs[0],0,2,1,3),GGML_TYPE_F32);
  auto *k=ggml_permute(ctx,s.inputs[1],0,2,1,3),*v=ggml_permute(ctx,s.inputs[2],0,2,1,3);
  ggml_tensor *mask=nullptr;
  if(slot.options.causal){s.mask.resize(std::size_t(nq)*nk);for(int row=0;row<nq;++row)for(int j=0;j<nk;++j)s.mask[std::size_t(row)*nk+j]=ggml_fp32_to_fp16(j>nk-nq+row?-INFINITY:0.F);
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
  HYLE_ASSIGN_OR_RETURN(s.execution,n::CpuExecution::prepare(s.graph,1,slot.allocator));
  s.known+=s.arena.capacity()+s.execution.workspace_bytes()+s.mask.size()*2;
  s.qoffset=std::size_t(qstart)*cap.query_heads*cap.qk_head_dim*2;s.koffset=std::size_t(kstart)*cap.kv_heads*cap.qk_head_dim*2;
  s.qbytes=std::size_t(nq)*cap.query_heads*cap.qk_head_dim*2;s.kbytes=std::size_t(nk)*cap.kv_heads*cap.qk_head_dim*2;
  slot.known+=s.known;slot.requests.push_back(std::move(request));return Status::Ok();
}
d::EnqueueResult plan(void *p,d::QueueSequence &,const c::BatchLayoutVersion &layout,d::BatchWorkspaceViews,c::BatchPlanReport *r)noexcept{
  auto &s=*static_cast<Slot *>(p);s.requests.clear();s.layout={};s.known=0;
  try{const auto h=layout.host();for(unsigned i=0;i<layout.facts().requests;++i){const auto nq=h.query_indptr[i+1]-h.query_indptr[i],nk=h.kv_indptr[i+1]-h.kv_indptr[i];if(!nq)continue;
    if(!nk||(s.options.causal&&nk<nq))return d::EnqueueResult::rejected(Status::Unimplemented("original ragged empty/short KV"));
    auto status=make_request(s,nq,nk,h.query_indptr[i],h.kv_indptr[i]);if(!status.ok()){s.requests.clear();return d::EnqueueResult::rejected(status);}}
    if(s.known>s.options.max_float_workspace_bytes)return d::EnqueueResult::rejected(Status::ResourceExhausted("original ragged known bytes"));
    s.layout=layout;*r=report(s);return d::EnqueueResult::accepted_command();
  }catch(const std::bad_alloc &){s.requests.clear();return d::EnqueueResult::rejected(Status::ResourceExhausted("original ragged cold allocations"));}
}
d::EnqueueResult run(void *p,d::QueueSequence &,std::span<const c::ConstTensorView> inputs,d::BatchWorkspaceViews workspace)noexcept{
  auto &slot=*static_cast<Slot *>(p);bool accepted=false;
  for(auto &owned:slot.requests){auto &s=*owned;std::array<mem::HostMapping,3> maps;
    for(unsigned port=0;port<3;++port){auto sub=inputs[port].storage().subview(port?s.koffset:s.qoffset,port?s.kbytes:s.qbytes);if(!sub)return {accepted,sub.error()};auto mapped=sub->map_host();if(!mapped)return {accepted,mapped.error()};maps[port]=std::move(*mapped);s.inputs[port]->data=const_cast<std::byte *>(maps[port].bytes().data());}
    auto sub=workspace.output.subview(s.qoffset,s.qbytes);if(!sub)return {accepted,sub.error()};auto mapped=sub->map_host(mem::HostAccess::kWrite);if(!mapped)return {accepted,mapped.error()};auto bytes=mapped->writable_bytes();if(!bytes)return {accepted,bytes.error()};s.output->data=bytes->data();
    for(auto *view:s.views)view->data=static_cast<std::byte *>(view->view_src->data)+view->view_offs;
    auto status=s.execution.run();accepted=true;status.Update(mapped->close());for(auto &map:maps)status.Update(map.close());if(!status.ok())return {accepted,status};
  }return d::EnqueueResult::accepted_command();
}
const d::BatchProviderVTable batch{&qualify,&create,&destroy,&update,&plan,&run};
const d::ProviderVTable &decorated(){static const auto result=[](){auto base=c::ggml_cpu::provider({.native_threads=1});auto table=*d::ProviderAccess::vtable(base);
  table.identity={.id=c::ProviderId{0x48594c4542454e43ULL,0x5241474745440001ULL},.name="original-cpu-ragged-flash",.version="ffa4e8b8"};
  table.query_extension=[](void *,c::ProviderExtensionId id)noexcept->const void *{return id==d::kBatchPlanningExtensionId?&batch:nullptr;};return table;}();return result;}
}
c::Provider cpu_ragged_reference_provider()noexcept{return d::ProviderAccess::with_vtable(c::ggml_cpu::provider({.native_threads=1}),&decorated());}
}
