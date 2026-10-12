#include <hyle/compute/batch_materialize.h>
#include <hyle/graph/batch.h>
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/compute/apple_batch.h>
#include <hyle/graph/apple.h>
#elif defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_batch.h>
#include <hyle/graph/vulkan.h>
#else
#error public native batch consumer requires the installed Apple or Vulkan route
#endif
#include <array>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;
#if defined(HYLE_COMPUTE_APPLE)
namespace n=c::apple;namespace ng=hyle::graph::apple;
constexpr const char*route="Apple";
#else
namespace n=c::vulkan;namespace ng=hyle::graph::vulkan;
constexpr const char*route="Vulkan";
#endif
using hyle::Status;
Status check(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
// Fixed oracle inputs only; native-only Compute/Graph does not supply SIMD.
std::uint16_t encoded(float value,bool bf16){
  if(value==0)return 0;
  if(value==.125f)return bf16?0x3e00u:0x3000u;
  if(value==.25f)return bf16?0x3e80u:0x3400u;
  if(value==.375f)return bf16?0x3ec0u:0x3600u;
  return bf16?0x3f80u:0x3c00u; // Only remaining declared input is 1.
}
hyle::Result<c::ReadLease> upload(n::Context&context,c::LeasePool&host,hyle::mem::MemoryResourceRef gpu,hyle::mem::MemoryResourceRef staging,c::WriteLease&&target,std::span<const std::uint16_t> words){
  HYLE_ASSIGN_OR_RETURN(auto writer,host.allocate(words.size_bytes()));HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(words)));HYLE_ASSIGN_OR_RETURN(auto source,writer.publish());
  const std::array specs{c::CopyResourceSpec{staging,words.size_bytes()},c::CopyResourceSpec{gpu,words.size_bytes()}};HYLE_ASSIGN_OR_RETURN(auto program,n::CopyBatchProgram::prepare(context,specs));HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(program));
  HYLE_RETURN_IF_ERROR(copy.bind_source({0},source));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(target)));const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,words.size_bytes()}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());HYLE_RETURN_IF_ERROR(submitted.wait());return submitted.output({0},{1},0,words.size_bytes());
}
hyle::Result<c::ReadLease> load(n::Context&context,c::LeasePool&pool,c::LeasePool&host,hyle::mem::MemoryResourceRef gpu,hyle::mem::MemoryResourceRef staging,std::span<const std::uint16_t> words){
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(words.size_bytes()));return upload(context,host,gpu,staging,std::move(writer),words);
}
Status verify(n::Context&context,n::MemoryResource&memory,const n::Results&result,c::TensorDesc description,bool bf16){
  std::array<std::uint16_t,512> words{};
#if defined(HYLE_COMPUTE_APPLE)
  (void)context;HYLE_ASSIGN_OR_RETURN(auto lease,memory.import_readonly(result.arrays()[0],description));HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(words))));
#else
  (void)memory;(void)description;HYLE_RETURN_IF_ERROR(context.download(result.tensors()[0],std::as_writable_bytes(std::span(words))));
#endif
  for(unsigned i=0;i<256;++i)HYLE_RETURN_IF_ERROR(check(words[i]==encoded(.25f,bf16),"public native batch mean oracle differs"));return Status::Ok();
}
Status exercise(bool paged,bool bf16){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_COMPUTE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto memory,n::MemoryResource::create(context));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_RETURN_IF_ERROR(check(!context.environment().software_device,"software Vulkan does not qualify actual native batch math"));
  HYLE_ASSIGN_OR_RETURN(auto memory,n::MemoryResource::create(context,n::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  struct Close{n::Context&context;~Close(){context.close().IgnoreError();}}close{context};
  c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.data_type=bf16?c::DataType::kBFloat16:c::DataType::kFloat16,.max_requests=1,.max_query_rows=2,.max_kv_rows=4,.max_pages=paged?3u:0u,.max_page_refs=paged?2u:0u,.query_heads=4,.kv_heads=2,.head_dim=64,.page_size=paged?2u:1u};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));const std::array<std::int32_t,2> qptr{0,1},kvptr{0,paged?2:3},pages{2,0};const std::array<std::int32_t,1> tail{1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kvptr,paged?std::span(pages):std::span<const std::int32_t>{},paged?std::span(tail):std::span<const std::int32_t>{}}));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=4,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=1u<<20,.scale=.125f}));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),8192));HYLE_ASSIGN_OR_RETURN(auto host,c::LeasePool::create(staging.ref(),8192));std::vector<std::uint16_t> zero(512,encoded(0,bf16));HYLE_ASSIGN_OR_RETURN(auto query,load(context,pool,host,memory.ref(),staging.ref(),zero));
  c::ReadLease ragged_k,ragged_v;c::PagedArena key_arena,value_arena;c::PagedArenaRead keys,values;
  if(paged){
    HYLE_ASSIGN_OR_RETURN(key_arena,c::PagedArena::create(schema,memory.ref(),{.max_backing_bytes=4096}));HYLE_ASSIGN_OR_RETURN(value_arena,c::PagedArena::create(schema,memory.ref(),{.max_backing_bytes=4096}));std::array<c::ReadLease,2> k,v;
    for(unsigned i=0;i<2;++i){const unsigned id=i?0:2;HYLE_ASSIGN_OR_RETURN(auto kw,key_arena.acquire(id));HYLE_ASSIGN_OR_RETURN(auto vw,value_arena.acquire(id));std::vector<std::uint16_t> a(256,encoded(0,bf16)),b(256);for(unsigned j=0;j<256;++j)b[j]=encoded(i?(j<128?.375f:1.f):(j<128?.125f:.25f),bf16);
      HYLE_ASSIGN_OR_RETURN(k[i],upload(context,host,memory.ref(),staging.ref(),std::move(kw.lease()),a));HYLE_ASSIGN_OR_RETURN(v[i],upload(context,host,memory.ref(),staging.ref(),std::move(vw.lease()),b));HYLE_RETURN_IF_ERROR(kw.complete(k[i]));HYLE_RETURN_IF_ERROR(vw.complete(v[i]));
    }HYLE_ASSIGN_OR_RETURN(keys,key_arena.read(layout,k));HYLE_ASSIGN_OR_RETURN(values,value_arena.read(layout,v));
  }else{
    HYLE_ASSIGN_OR_RETURN(ragged_k,load(context,pool,host,memory.ref(),staging.ref(),zero));std::vector<std::uint16_t> data(512);for(unsigned i=0;i<data.size();++i)data[i]=encoded(i<128?.125f:i<256?.25f:i<384?.375f:1.f,bf16);HYLE_ASSIGN_OR_RETURN(ragged_v,load(context,pool,host,memory.ref(),staging.ref(),data));
  }
  const c::MaterializedNativeBatchOptions options{.max_metadata_backing_bytes=1u<<20};
  HYLE_ASSIGN_OR_RETURN(auto direct,n::MaterializedBatchInvocation::create(context,recipe,memory,staging,options));HYLE_ASSIGN_OR_RETURN(auto graph,ng::prepare_materialized_batch(context,recipe,memory,staging,options));
  HYLE_RETURN_IF_ERROR(direct.prepare_batch(layout));HYLE_RETURN_IF_ERROR(graph.prepare_batch(layout));
  if(paged){HYLE_RETURN_IF_ERROR(direct.bind(query,keys,values));HYLE_RETURN_IF_ERROR(graph.bind(query,keys,values));}else{HYLE_RETURN_IF_ERROR(direct.bind(query,ragged_k,ragged_v));HYLE_RETURN_IF_ERROR(graph.bind(query,ragged_k,ragged_v));}
  HYLE_ASSIGN_OR_RETURN(auto d,direct.submit());HYLE_ASSIGN_OR_RETURN(auto result,d.wait());HYLE_RETURN_IF_ERROR(verify(context,memory,*result,recipe.descriptions()[0],bf16));
  HYLE_ASSIGN_OR_RETURN(auto done,graph.submit());HYLE_ASSIGN_OR_RETURN(auto g,done.wait());HYLE_RETURN_IF_ERROR(verify(context,memory,*g,recipe.descriptions()[0],bf16));
  HYLE_RETURN_IF_ERROR(check(!direct.prepare_batch(layout).ok() && !graph.prepare_batch(layout).ok(),"public native old result allowed overwrite"));
  const auto report=direct.report();HYLE_RETURN_IF_ERROR(check(report.layout_generation==layout.generation() && report.uploaded_bytes>0,"public native mirror generation/upload differs"));
  std::printf("public %s %s %s Direct/Graph actual-native/GQA/tail/old-result mean=0.25 PASS\n",route,paged?"paged":"ragged",bf16?"BF16":"F16");return Status::Ok();
}
Status run(){for(bool paged:{false,true})for(bool bf16:{false,true})HYLE_RETURN_IF_ERROR(exercise(paged,bf16));return Status::Ok();}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run();if(status.ok())return 0;char message[768]{};status.ToString(message,sizeof(message));std::fprintf(stderr,"public native batch: %s\n",message);return 1;}
