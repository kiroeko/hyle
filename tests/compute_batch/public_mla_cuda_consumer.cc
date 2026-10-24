#include <hyle/graph/cuda.h>
#include <hyle/compute/cuda.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/graph/batch.h>
#include "mla_test_support.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace gb=g::batch;
using hyle::Status;
using mla_test::check;
using mla_test::encode;
using mla_test::decode;
using mla_test::oracle;
Status exercise(bool bf,unsigned heads,unsigned depth=576){
  const c::MlaBatchCapacity mc{.query_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .output_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .cache_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),
      .layout=depth==576?c::MlaCacheLayout::kCkv512Rope64:c::MlaCacheLayout::kCkv512,
      .max_requests=2,.max_query_rows=4,.max_kv_rows=64,.max_pages=4,.max_page_refs=5,
      .query_heads=heads,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create_mla(mc));
  const auto cap=schema.capacity();
  const std::array<std::int32_t,3> qptr{0,1,4},kptr{0,2,5};
  const std::array<std::int32_t,5> pages{3,1,0,2,1};
  const std::array<std::int32_t,2> tails{15,1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kptr,pages,tails}));
  HYLE_ASSIGN_OR_RETURN(auto default_context,c::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto default_device,c::cuda::MemoryResource::create(default_context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto default_host,c::cuda::MemoryResource::create(default_context,c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto default_program,c::BatchProgram::prepare(default_context,schema,{
      .resource=default_device.ref(),.host_workspace=default_host.ref(),.scale=1.F/16}));
  auto too_large=mc;too_large.max_query_rows=std::uint32_t(UINT32_MAX/depth/heads+1);
  HYLE_ASSIGN_OR_RETURN(auto large_schema,c::BatchSchema::create_mla(too_large));
  auto bad=c::BatchProgram::prepare(default_context,large_schema,{
      .resource=default_device.ref(),.host_workspace=default_host.ref(),.scale=1.F/16});
  HYLE_RETURN_IF_ERROR(check(!bad && bad.error().IsOutOfRange(),"MLA native query offset overflow was admitted"));
  std::puts("Default CUDA context discovers distinct MLA BatchPlanning PASS");
  c::BatchProgramOptions options{.max_float_workspace_bytes=16u<<20,.max_int_workspace_bytes=1u<<20,.scale=1.F/16};
  const auto native_provider = c::flashinfer::provider();
  HYLE_ASSIGN_OR_RETURN(auto id,c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(id,native_provider));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
  struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
  options.resource=device.ref();options.host_workspace=host.ref();
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,options));
  {
    auto tiny=options;tiny.max_float_workspace_bytes=16;tiny.max_int_workspace_bytes=16;
    HYLE_ASSIGN_OR_RETURN(auto limited,c::BatchProgram::prepare(context,schema,tiny));
    HYLE_ASSIGN_OR_RETURN(auto limited_invocation,c::BatchInvocation::create(limited,queue));
    auto exhausted=limited_invocation.prepare_batch(layout);
    HYLE_RETURN_IF_ERROR(check(!exhausted && exhausted.error().IsResourceExhausted(),"MLA workspace preflight did not reject capacity exhaustion"));
  }
  HYLE_ASSIGN_OR_RETURN(auto direct,c::BatchInvocation::create(program,queue));
  std::vector<std::uint16_t> q(std::size_t(cap.max_query_rows)*heads*depth),cache(4*16*depth);
  for(std::size_t i=0;i<q.size();++i)q[i]=encode(float(int(i%29)-14)/64,bf);
  for(std::size_t i=0;i<cache.size();++i)cache[i]=encode(float(int((i*13+i/depth)%37)-18)/32,bf);
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),q.size()*2+4*heads*512*2+4096));
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(q.size()*2));
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query,writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,c::BatchCacheRole::kMla,device.ref(),{.max_backing_bytes=cache.size()*2+1}));
  HYLE_RETURN_IF_ERROR(check(!c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=cache.size()*2+1}),"MLA accepted an independent V arena"));
  std::array<c::ReadLease,4> ready;
  for(unsigned page=0;page<4;++page){
    HYLE_ASSIGN_OR_RETURN(auto page_writer,arena.acquire(page));
    HYLE_RETURN_IF_ERROR(page_writer.lease().view()->write(std::as_bytes(std::span(cache).subspan(page*16*depth,16*depth))));
    HYLE_ASSIGN_OR_RETURN(ready[page],page_writer.publish(16));
  }
  HYLE_ASSIGN_OR_RETURN(auto frame,arena.read(layout,ready));
  HYLE_ASSIGN_OR_RETURN(auto token,direct.prepare_batch(layout));
  HYLE_RETURN_IF_ERROR(check(!token.run(std::array{query,query,query}),"MLA accepted legacy three-payload run"));
  HYLE_RETURN_IF_ERROR(token.prepare_replay(query,frame));
  HYLE_ASSIGN_OR_RETURN(auto done,token.run(query,frame));HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto output,direct.output());HYLE_RETURN_IF_ERROR(oracle(cap,layout,q,cache,output,bf));
  std::printf("MLA Direct single arena %zu bytes, native float=%zu int=%zu capture=%d PASS\n",cache.size()*2,direct.report().float_workspace_bytes,direct.report().int_workspace_bytes,int(direct.report().capture_compatible));
  HYLE_RETURN_IF_ERROR(check(!direct.prepare_batch(layout),"MLA retained output allowed overwrite"));
  output.reset();done={};token={};
  auto changed_pages=pages;std::swap(changed_pages[2],changed_pages[3]);
  HYLE_ASSIGN_OR_RETURN(auto updated,schema.version({qptr,kptr,changed_pages,tails}));
  HYLE_ASSIGN_OR_RETURN(auto updated_frame,arena.read(updated,ready));
  HYLE_ASSIGN_OR_RETURN(auto updated_token,direct.update_batch(updated));
  HYLE_RETURN_IF_ERROR(check(direct.report().reused_native_plan,"MLA page-ID-only update rebuilt its schedule"));
  HYLE_RETURN_IF_ERROR(updated_token.prepare_replay(query,updated_frame));
  HYLE_ASSIGN_OR_RETURN(auto updated_done,updated_token.run(query,updated_frame));HYLE_RETURN_IF_ERROR(updated_done.wait());
  HYLE_ASSIGN_OR_RETURN(auto updated_output,direct.output());HYLE_RETURN_IF_ERROR(oracle(cap,updated,q,cache,updated_output,bf));
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::batch_query_description(cap));
  HYLE_ASSIGN_OR_RETURN(auto cdesc,c::batch_cache_description(schema,c::BatchCacheRole::kMla));
  HYLE_ASSIGN_OR_RETURN(auto mdesc,schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto odesc,c::batch_output_description(cap,cap.max_query_rows));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  std::array<g::ValueId,3> inputs;
  const std::array descriptions{qdesc,cdesc,mdesc};
  for(unsigned i=0;i<3;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::PagedMlaAttention{mc,options.scale}),inputs));
  HYLE_ASSIGN_OR_RETURN(auto result,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(result));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::cuda::compile(graph,context,device.ref(),{.operation_fallback=g::OperationFallbackPolicy::kReject}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const gb::PrepareOptions preparation{.host_workspace = host.ref(), .max_float_workspace_bytes = options.max_float_workspace_bytes, .max_int_workspace_bytes = options.max_int_workspace_bytes};
HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(gb::prepare(invocation, queues, std::array{gb::NodeLayout{node,layout}}, preparation));
  HYLE_RETURN_IF_ERROR(invocation.bind(inputs[0], qdesc, query));HYLE_RETURN_IF_ERROR(gb::bind(invocation, inputs[1], frame));
  HYLE_ASSIGN_OR_RETURN(auto graph_writer,pool.allocate(4*heads*512*2));
  HYLE_RETURN_IF_ERROR(invocation.bind(result, odesc, std::move(graph_writer)));
  HYLE_RETURN_IF_ERROR(gb::prepare_replay(invocation, queues));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto graph_done,invocation.submit(queues));HYLE_RETURN_IF_ERROR(graph_done.wait());
  HYLE_ASSIGN_OR_RETURN(auto graph_output,invocation.result(result));
  HYLE_ASSIGN_OR_RETURN(auto graph_read,graph_output.storage_lease());HYLE_RETURN_IF_ERROR(oracle(cap,layout,q,cache,graph_read,bf));
  std::puts("MLA Graph 3 ports, same single arena, captured original Queue PASS");return Status::Ok();
}
Status split_partials(){
  constexpr unsigned rows=8192,physical=8192,heads=64,depth=576;
  const c::MlaBatchCapacity mc{.query_type=c::DataType::kBFloat16, .output_type=c::DataType::kBFloat16, .cache_storage=c::CacheStorage::plain(c::DataType::kBFloat16).value(),
      .max_requests=1,.max_query_rows=1,.max_kv_rows=rows,.max_pages=512,.max_page_refs=512,.query_heads=heads};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create_mla(mc));
  const std::array<std::int32_t,2> qptr{0,1},kptr{0,512};const std::array<std::int32_t,1>tails{16};
  std::vector<std::int32_t>ids(512);for(unsigned i=0;i<ids.size();++i)ids[i]=std::int32_t(i);
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kptr,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cuda::device(0).value(),c::flashinfer::provider()));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
  struct Drain{c::Context&c;~Drain(){c.drain().IgnoreError();}}drain{context};
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,{
      .resource=device.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=16u<<20,.max_int_workspace_bytes=1u<<20,.scale=1.F/16}));
  HYLE_ASSIGN_OR_RETURN(auto invocation,c::BatchInvocation::create(program,queue));
  std::vector<std::uint16_t> q(heads*depth,encode(0,true)),cache(physical*depth);
  for(unsigned row=0;row<physical;++row)for(unsigned d=0;d<depth;++d)
    cache[row*depth+d]=encode(d<512?((row<4096?16384.F:-16384.F)+(row==0?256.F:0.F)):.0625F,true);
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),(q.size()+cache.size())*2+1024));
  const auto load=[&](const std::vector<std::uint16_t>&words)->hyle::Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto w,pool.allocate(words.size()*2));HYLE_RETURN_IF_ERROR(w.view()->write(std::as_bytes(std::span(words))));return w.publish();
  };
  HYLE_ASSIGN_OR_RETURN(auto qr,load(q));HYLE_ASSIGN_OR_RETURN(auto cr,load(cache));
  HYLE_ASSIGN_OR_RETURN(auto prepared,invocation.prepare_batch(layout));
  HYLE_RETURN_IF_ERROR(check(invocation.report().split_kv,"long MLA did not actually schedule split partials"));
  HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(c::MlaBatchInputs{qr,cr}));HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto output,invocation.output());std::vector<std::uint16_t>actual(heads*512);
  HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
  double maximum=0;
  for(unsigned h=0;h<heads;++h)for(unsigned d=0;d<512;++d){
    const double expected=256.0/rows;
    const double error=std::abs(decode(actual[h*512+d],true)-expected);maximum=std::max(maximum,error);
    HYLE_RETURN_IF_ERROR(check(std::isfinite(error) && error<1e-7,"split MLA lost the exactly representable F32 partial residual"));
  }
  std::printf("MLA true split BF16 Q1/KV8192 cancellation full output maxAbs=%.9g floatWorkspace=%zu PASS\n",maximum,invocation.report().float_workspace_bytes);
  return Status::Ok();
}
Status run(){HYLE_RETURN_IF_ERROR(exercise(false,96));HYLE_RETURN_IF_ERROR(exercise(true,64));HYLE_RETURN_IF_ERROR(exercise(false,64,512));HYLE_RETURN_IF_ERROR(exercise(true,64,512));return split_partials();}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run();if(status.ok())return 0;char message[768]{};status.ToString(message,sizeof(message));std::fprintf(stderr,"public CUDA MLA: %s\n",message);return 1;}
