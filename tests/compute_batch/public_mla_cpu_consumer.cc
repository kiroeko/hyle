#include <hyle/graph/cpu.h>
#include <hyle/compute/cpu.h>
#include <hyle/compute/providers/ggml_cpu.h>
#include <hyle/graph/batch.h>

#include <array>
#include <cstdio>
#include <string_view>
#include <vector>

#include "mla_test_support.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace gb=g::batch;
using hyle::Status;
using namespace mla_test;

Status exercise(bool bf, c::MlaCacheLayout format, bool paged, std::int32_t window_left=-1) {
  c::MlaBatchCapacity mc{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
      .query_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .output_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .cache_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),.layout=format,
      .max_requests=2,.max_query_rows=4,.max_kv_rows=64,
      .max_pages=paged?4u:0u,.max_page_refs=paged?5u:0u,
      .query_heads=bf?64u:96u,.page_size=paged?16u:1u};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create_mla(mc));
  const auto cap=schema.capacity();
  const std::array<std::int32_t,3> qptr{0,1,4},kptr=paged?
      std::array<std::int32_t,3>{0,2,5}:std::array<std::int32_t,3>{0,31,64};
  std::array<std::int32_t,5> pages{3,1,0,2,1};
  const std::array<std::int32_t,2> tails{15,1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kptr,
      paged?std::span<const std::int32_t>(pages):std::span<const std::int32_t>{},
      paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));
  c::BatchProgramOptions options{.resource=hyle::mem::system_resource(),
      .max_float_workspace_bytes=16u<<20,.max_int_workspace_bytes=4096,.scale=1.F/16};
  options.window_left=window_left;
  HYLE_ASSIGN_OR_RETURN(auto default_context,c::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto default_program,c::BatchProgram::prepare(default_context,schema,options));
  const auto native_provider = c::ggml_cpu::provider();
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),native_provider));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,options));
  HYLE_ASSIGN_OR_RETURN(auto direct,c::BatchInvocation::create(program,queue));
  std::vector<std::uint16_t> q(std::size_t(4)*cap.query_heads*cap.qk_head_dim),cache(std::size_t(64)*cap.qk_head_dim);
  for(std::size_t i=0;i<q.size();++i)q[i]=encode(float(int(i%29)-14)/64,bf);
  for(std::size_t i=0;i<cache.size();++i)cache[i]=encode(float(int((i*13+i/cap.qk_head_dim)%37)-18)/32,bf);
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),
      q.size()*2+cache.size()*2+4*cap.query_heads*512*2+4096));
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(q.size()*2));
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query,writer.publish());
  c::PagedArena arena;
  std::array<c::ReadLease,4> ready;
  c::ReadLease ragged_cache;
  if(paged){
    HYLE_ASSIGN_OR_RETURN(arena,c::PagedArena::create(schema,c::BatchCacheRole::kMla,
        hyle::mem::system_resource(),{.max_backing_bytes=cache.size()*2+1}));
    HYLE_RETURN_IF_ERROR(check(!c::PagedArena::create(schema,c::BatchCacheRole::kValue,
        hyle::mem::system_resource(),{}),"MLA accepted independent V arena"));
    for(unsigned page=0;page<4;++page){
      HYLE_ASSIGN_OR_RETURN(auto w,arena.acquire(page));
      HYLE_RETURN_IF_ERROR(w.lease().view()->write(std::as_bytes(std::span(cache).subspan(
          page*16*cap.qk_head_dim,16*cap.qk_head_dim))));
      HYLE_ASSIGN_OR_RETURN(ready[page],w.publish(16));
    }
  }else{
    HYLE_ASSIGN_OR_RETURN(auto w,pool.allocate(cache.size()*2));
    HYLE_RETURN_IF_ERROR(w.view()->write(std::as_bytes(std::span(cache))));
    HYLE_ASSIGN_OR_RETURN(ragged_cache,w.publish());
  }
  for(unsigned episode=0;episode<3;++episode){
    if(episode){
      const std::array<std::int32_t,3> decode_q{0,1,2};
      if(episode==2)std::swap(pages[0],pages[1]);
      HYLE_ASSIGN_OR_RETURN(layout,schema.version({decode_q,kptr,
          paged?std::span<const std::int32_t>(pages):std::span<const std::int32_t>{},
          paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));
    }
    HYLE_ASSIGN_OR_RETURN(auto token,direct.update_batch(layout));
    HYLE_RETURN_IF_ERROR(check(!token.run(std::array{query,query,query}),"MLA accepted legacy three-payload run"));
    c::PagedArenaRead frame;
    if(paged){HYLE_ASSIGN_OR_RETURN(frame,arena.read(layout,ready));}
    auto submitted = [&]() -> hyle::Result<c::Completion> {
      if(paged)return token.run(query,frame);
      return token.run(c::MlaBatchInputs{query,ragged_cache});
    }();
    HYLE_ASSIGN_OR_RETURN(auto done,std::move(submitted));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto output,direct.output());
    HYLE_RETURN_IF_ERROR(oracle(cap,layout,q,cache,output,bf,window_left));
    if(episode==2)HYLE_RETURN_IF_ERROR(check(direct.report().reused_native_plan,"CPU MLA Q1 update rebuilt native plan"));
  }
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::batch_query_description(cap));
  HYLE_ASSIGN_OR_RETURN(auto cdesc,c::batch_cache_description(schema,c::BatchCacheRole::kMla));
  HYLE_ASSIGN_OR_RETURN(auto mdesc,schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto odesc,c::batch_output_description(cap,cap.max_query_rows));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  std::array<g::ValueId,3> inputs;
  const std::array descriptions{qdesc,cdesc,mdesc};
  for(unsigned i=0;i<3;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(paged?
      c::Operation::custom(c::PagedMlaAttention{mc,options.scale,options.causal,window_left}):
      c::Operation::custom(c::RaggedMlaAttention{mc,options.scale,options.causal,window_left}),inputs));
  HYLE_ASSIGN_OR_RETURN(auto result,builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(result));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::cpu::compile(graph,context,{.operation_fallback=g::OperationFallbackPolicy::kReject}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const gb::PrepareOptions preparation{.host_workspace = hyle::mem::system_resource(), .max_float_workspace_bytes = options.max_float_workspace_bytes, .max_int_workspace_bytes = options.max_int_workspace_bytes};
HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(gb::prepare(invocation, queues, std::array{gb::NodeLayout{node,layout}}, preparation));
  HYLE_RETURN_IF_ERROR(invocation.bind(inputs[0], qdesc, query));
  c::PagedArenaRead frame;
  if(paged){
    HYLE_ASSIGN_OR_RETURN(frame,arena.read(layout,ready));
    HYLE_RETURN_IF_ERROR(gb::bind(invocation, inputs[1], frame));
  }else HYLE_RETURN_IF_ERROR(invocation.bind(inputs[1], cdesc, ragged_cache));
  HYLE_ASSIGN_OR_RETURN(auto out,pool.allocate(4*cap.query_heads*512*2));
  HYLE_RETURN_IF_ERROR(invocation.bind(result, odesc, std::move(out)));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queues));
  HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto output,invocation.result(result));
  HYLE_ASSIGN_OR_RETURN(auto read,output.storage_lease());
  HYLE_RETURN_IF_ERROR(oracle(cap,layout,q,cache,read,bf,window_left));
  std::printf("CPU MLA %s single cache %zu bytes, Direct+Graph/Q1 plan reuse PASS\n",paged?"paged":"ragged",cache.size()*2);
  return Status::Ok();
}
Status run(bool windows_only){
  if(!windows_only)for(bool bf:{false,true})for(bool paged:{false,true})
    for(auto layout:{c::MlaCacheLayout::kCkv512,c::MlaCacheLayout::kCkv512Rope64})
      HYLE_RETURN_IF_ERROR(exercise(bf,layout,paged));
  HYLE_RETURN_IF_ERROR(exercise(true,c::MlaCacheLayout::kCkv512Rope64,true,0));
  HYLE_RETURN_IF_ERROR(exercise(false,c::MlaCacheLayout::kCkv512,false,7));
  return Status::Ok();
}
}
int main(int argc,char**argv){
  const auto status=run(argc==2 && std::string_view(argv[1])=="--window");if(status.ok())return 0;
  char message[768]{};status.ToString(message,sizeof(message));
  std::fprintf(stderr,"public CPU MLA: %s\n",message);return 1;
}
