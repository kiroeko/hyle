#include <hyle/compute/batch_materialize.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/simd/dtype.h>
#include <array>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
using hyle::Status;
Status check(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
std::uint16_t encoded(float value,bool bf16){return bf16?hyle::simd::f32_to_bf16(value).bits:hyle::simd::f32_to_f16(value).bits;}
hyle::Result<c::ReadLease> load(c::LeasePool&pool,std::span<const std::uint16_t> words){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(words.size_bytes()));HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(words)));return writer.publish();}
Status exercise(bool paged,bool bf16){
  HYLE_ASSIGN_OR_RETURN(auto context,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
  c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.data_type=bf16?c::DataType::kBFloat16:c::DataType::kFloat16,.max_requests=1,.max_query_rows=2,.max_kv_rows=4,.max_pages=paged?3u:0u,.max_page_refs=paged?2u:0u,.query_heads=4,.kv_heads=2,.head_dim=64,.page_size=paged?2u:1u};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));const std::array<std::int32_t,2> qptr{0,1},kvptr{0,paged?2:3},pages{2,0};const std::array<std::int32_t,1> tail{1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kvptr,paged?std::span(pages):std::span<const std::int32_t>{},paged?std::span(tail):std::span<const std::int32_t>{}}));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=4,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=1u<<20,.scale=.125f}));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),8192));std::vector<std::uint16_t> zero(512,encoded(0,bf16));HYLE_ASSIGN_OR_RETURN(auto query,load(pool,zero));
  c::ReadLease ragged_k,ragged_v;c::PagedArena key_arena,value_arena;c::PagedArenaRead keys,values;
  if(paged){
    HYLE_ASSIGN_OR_RETURN(key_arena,c::PagedArena::create(schema,hyle::mem::system_resource(),{.max_backing_bytes=4096}));HYLE_ASSIGN_OR_RETURN(value_arena,c::PagedArena::create(schema,hyle::mem::system_resource(),{.max_backing_bytes=4096}));std::array<c::ReadLease,2> k,v;
    for(unsigned i=0;i<2;++i){const unsigned id=i?0:2;HYLE_ASSIGN_OR_RETURN(auto kw,key_arena.acquire(id));HYLE_ASSIGN_OR_RETURN(auto vw,value_arena.acquire(id));std::vector<std::uint16_t> a(256,encoded(0,bf16)),b(256);
      for(unsigned j=0;j<256;++j)b[j]=encoded(i?(j<128?.375f:1.f):(j<128?.125f:.25f),bf16);
      HYLE_RETURN_IF_ERROR(kw.lease().view()->write(std::as_bytes(std::span(a))));HYLE_RETURN_IF_ERROR(vw.lease().view()->write(std::as_bytes(std::span(b))));HYLE_ASSIGN_OR_RETURN(k[i],kw.publish(cap.page_size));HYLE_ASSIGN_OR_RETURN(v[i],vw.publish(cap.page_size));
    }HYLE_ASSIGN_OR_RETURN(keys,key_arena.read(layout,k));HYLE_ASSIGN_OR_RETURN(values,value_arena.read(layout,v));
  }else{
    HYLE_ASSIGN_OR_RETURN(ragged_k,load(pool,zero));std::vector<std::uint16_t> data(512);for(unsigned i=0;i<data.size();++i)data[i]=encoded(i<128?.125f:i<256?.25f:i<384?.375f:1.f,bf16);HYLE_ASSIGN_OR_RETURN(ragged_v,load(pool,data));
  }
  HYLE_ASSIGN_OR_RETURN(auto direct,c::MaterializedBatchInvocation::create(context,recipe,queue,{.max_payload_bytes=1u<<20,.max_workspace_bytes=1u<<20}));
  HYLE_ASSIGN_OR_RETURN(auto graph,g::batch::MaterializedInvocation::create(context,recipe,queue,{.max_output_bytes=4096}));
  HYLE_RETURN_IF_ERROR(direct.prepare_batch(layout));HYLE_RETURN_IF_ERROR(graph.prepare_batch(layout));
  if(paged){HYLE_RETURN_IF_ERROR(direct.bind(query,keys,values));HYLE_RETURN_IF_ERROR(graph.bind(query,keys,values));}else{HYLE_RETURN_IF_ERROR(direct.bind(query,ragged_k,ragged_v));HYLE_RETURN_IF_ERROR(graph.bind(query,ragged_k,ragged_v));}
  HYLE_ASSIGN_OR_RETURN(auto d,direct.submit());HYLE_RETURN_IF_ERROR(d.wait());HYLE_ASSIGN_OR_RETURN(auto output,direct.output());
  HYLE_ASSIGN_OR_RETURN(auto done,graph.submit());HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(auto result,graph.output());HYLE_ASSIGN_OR_RETURN(auto storage,result.storage_lease());
  for(const auto&lease:std::array{output,storage}){std::array<std::uint16_t,512> words{};HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(words))));for(unsigned i=0;i<256;++i)HYLE_RETURN_IF_ERROR(check(words[i]==encoded(.25f,bf16),"installed public batch independent mean oracle differs"));}
  HYLE_RETURN_IF_ERROR(check(!direct.prepare_batch(layout).ok() && !graph.prepare_batch(layout).ok(),"installed public old output did not block overwrite"));
  std::printf("public CPU %s %s Direct/Graph GQA/causal/active-capacity/page-tail/old-output mean=0.25 PASS\n",paged?"paged":"ragged",bf16?"BF16":"F16");return Status::Ok();
}
Status run(){for(bool paged:{false,true})for(bool bf16:{false,true})HYLE_RETURN_IF_ERROR(exercise(paged,bf16));return Status::Ok();}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run();if(status.ok())return 0;char message[768]{};status.ToString(message,sizeof(message));std::fprintf(stderr,"public batch: %s\n",message);return 1;}
