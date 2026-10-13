// Public C++ consumer: CPU normalization -> native paged GPU attention ->
// CPU cast/negation, with exact page permissions and two original Queues.
#define main hyle_g02_public_cuda_fixture_main
#include "public_batch_cuda_consumer.cc"
#undef main
#include <hyle/graph/cpu.h>
#include <hyle/graph/cpu_cuda.h>

namespace {
namespace hybrid = g::cpu_cuda;

Status hybrid_paged() {
  constexpr unsigned depth=256, heads=8, kv_heads=2, page=16;
  const c::BatchCapacity cap{
      .storage=c::BatchStorage::kPaged,.data_type=c::DataType::kFloat16,
      .max_requests=2,.max_query_rows=3,.max_kv_rows=48,
      .max_pages=4,.max_page_refs=3,.query_heads=heads,.kv_heads=kv_heads,
      .head_dim=depth,.page_size=page};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());
  const c::BatchProgramOptions options{
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=64u<<10,
      .max_invocations=2,.scale=1.f/std::sqrt(float(depth))};
  HYLE_ASSIGN_OR_RETURN(auto wrapped,gb::provider(c::flashinfer::provider(),schema,options));
  HYLE_ASSIGN_OR_RETURN(auto device_id,c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto gpu,c::Context::create(device_id,wrapped));
  HYLE_ASSIGN_OR_RETURN(auto gpu_queue,gpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(gpu,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,c::cuda::MemoryResource::create(gpu,c::cuda::MemoryKind::kPinnedHost));
  struct Drain {c::Context&cpu;c::Context&gpu;~Drain(){cpu.drain().IgnoreError();gpu.drain().IgnoreError();}} drain{cpu,gpu};
  const auto host=hyle::mem::system_resource();
  const std::array<std::int32_t,3> qptr{0,1,3},kvptr{0,2,3};
  const std::array<std::int32_t,2> tails{4,8};
  const std::array<std::int32_t,3> ids{0,1,2},reordered{2,1,0};
  HYLE_ASSIGN_OR_RETURN(auto initial,schema.version({qptr,kvptr,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto next,schema.version({qptr,kvptr,reordered,tails}));
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{3,heads,depth}));
  HYLE_ASSIGN_OR_RETURN(auto xdesc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,3>{3,heads,depth}));
  HYLE_ASSIGN_OR_RETURN(auto scale_desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{depth}));
  HYLE_ASSIGN_OR_RETURN(auto kvdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,4>{4,page,kv_heads,depth}));
  HYLE_ASSIGN_OR_RETURN(auto metadata_desc,schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x,builder.input(xdesc));
  HYLE_ASSIGN_OR_RETURN(auto scale,builder.input(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto key,builder.input(kvdesc));
  HYLE_ASSIGN_OR_RETURN(auto value,builder.input(kvdesc));
  HYLE_ASSIGN_OR_RETURN(auto metadata,builder.input(metadata_desc));
  HYLE_ASSIGN_OR_RETURN(auto normalized,builder.add(c::RmsNorm{.epsilon=1.e-6f},x,scale));
  HYLE_ASSIGN_OR_RETURN(auto query,builder.add(c::Cast{c::DataType::kFloat16},normalized));
  const std::array ports{query,key,value,metadata};
  HYLE_ASSIGN_OR_RETURN(auto attention,builder.add_node(c::Operation::custom(c::PagedAttention{cap,options.scale}),ports));
  HYLE_ASSIGN_OR_RETURN(auto attended,builder.result(attention));
  HYLE_ASSIGN_OR_RETURN(auto as_float,builder.add(c::Cast{c::DataType::kFloat32},attended));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.add(c::Unary{c::UnaryKind::kNeg},as_float));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  std::array<g::NodePlacement,5> placements;
  for(std::size_t index=0;index<placements.size();++index){
    HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(index));
    placements[index]={node,index==2?hybrid::kCudaTarget:hybrid::kCpuTarget,g::PlacementStrength::kRequired};
  }
  const std::array external{
      g::ExternalResourceContract{x,hybrid::kHostStorage},
      g::ExternalResourceContract{scale,hybrid::kHostStorage},
      g::ExternalResourceContract{key,hybrid::kDeviceStorage},
      g::ExternalResourceContract{value,hybrid::kDeviceStorage},
      g::ExternalResourceContract{metadata,hybrid::kDeviceStorage},
      g::ExternalResourceContract{output,hybrid::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,hybrid::compile(graph,{&cpu,&gpu,host,&staging,&device},
      {.node_placements=placements,.external_resources=external}));
  HYLE_RETURN_IF_ERROR(check(compiled.executable.required_queue_count()==2,"hybrid batch lost CPU/GPU Queues"));
  const std::array lanes{
      gb::LaneBinding{g::ExecutionLaneId{1},&cpu,&cpu_queue},
      gb::LaneBinding{g::ExecutionLaneId{0},&gpu,&gpu_queue,staging.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto invocation,gb::Invocation::create(compiled.executable,lanes));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1u<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1u<<20}));
  std::vector<std::uint16_t> k(4*page*kv_heads*depth),v(k.size());
  for(unsigned p=0;p<3;++p)for(unsigned row=0;row<page;++row)for(unsigned head=0;head<kv_heads;++head){
    const auto at=((p*page+row)*kv_heads+head)*depth;
    k[at]=encode(float(int(row)-8)*.5f,false);
    for(unsigned col=0;col<depth;++col)v[at+col]=encode(float(p)*.5f+float(row+1)*.0625f,false);
  }
  std::array<c::ReadLease,3> ready_k,ready_v;
  const auto elements=page*kv_heads*depth;
  for(unsigned p=0;p<3;++p){
    HYLE_ASSIGN_OR_RETURN(auto kw,keys.acquire(p));
    HYLE_ASSIGN_OR_RETURN(auto vw,values.acquire(p));
    HYLE_ASSIGN_OR_RETURN(auto kb,kw.lease().view());
    HYLE_ASSIGN_OR_RETURN(auto vb,vw.lease().view());
    HYLE_RETURN_IF_ERROR(kb.write(std::as_bytes(std::span(k).subspan(p*elements,elements))));
    HYLE_RETURN_IF_ERROR(vb.write(std::as_bytes(std::span(v).subspan(p*elements,elements))));
    HYLE_ASSIGN_OR_RETURN(ready_k[p],kw.publish());
    HYLE_ASSIGN_OR_RETURN(ready_v[p],vw.publish());
  }
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(host,256u<<10));
  const auto load_float=[&](std::span<const float> data)->hyle::Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto writer,host_pool.allocate(data.size_bytes()));
    HYLE_ASSIGN_OR_RETURN(auto bytes,writer.view());HYLE_RETURN_IF_ERROR(bytes.write(std::as_bytes(data)));
    return writer.publish();
  };
  const std::vector<float> xs(3*heads*depth,2.f),scales(depth,1.f);
  HYLE_ASSIGN_OR_RETURN(auto rx,load_float(xs));
  HYLE_ASSIGN_OR_RETURN(auto rs,load_float(scales));
  const std::vector<std::uint16_t> expected_q(3*heads*depth,encode(1.f,false));
  HYLE_ASSIGN_OR_RETURN(auto old_keys,keys.read(initial,ready_k));
  for(unsigned episode=0;episode<2;++episode){
    const auto layout=episode?next:initial;
    HYLE_ASSIGN_OR_RETURN(auto read_k,keys.read(layout,ready_k));
    HYLE_ASSIGN_OR_RETURN(auto read_v,values.read(layout,ready_v));
    HYLE_RETURN_IF_ERROR(invocation.prepare_batch(std::array{gb::NodeLayout{attention,layout}}));
    if(episode)HYLE_RETURN_IF_ERROR(check(!invocation.bind(key,old_keys).ok(),"hybrid mixed page/layout versions"));
    HYLE_RETURN_IF_ERROR(invocation.bind(x,xdesc,rx));HYLE_RETURN_IF_ERROR(invocation.bind(scale,scale_desc,rs));
    HYLE_RETURN_IF_ERROR(invocation.bind(key,read_k));HYLE_RETURN_IF_ERROR(invocation.bind(value,read_v));
    HYLE_ASSIGN_OR_RETURN(auto writer,host_pool.allocate(xs.size()*sizeof(float)));
    HYLE_RETURN_IF_ERROR(invocation.bind(output,xdesc,std::move(writer)));
    {HYLE_ASSIGN_OR_RETURN(auto unused,keys.acquire(3));HYLE_RETURN_IF_ERROR(check(unused.valid(),"hybrid page frame retained unreferenced capacity"));}
    HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit());HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto result,invocation.output(output));
    HYLE_ASSIGN_OR_RETURN(auto lease,result.storage_lease());
    std::vector<float> actual(xs.size());HYLE_ASSIGN_OR_RETURN(auto bytes,lease.view());
    HYLE_RETURN_IF_ERROR(bytes.read(std::as_writable_bytes(std::span(actual))));
    std::vector<std::uint16_t> attended_values(actual.size());
    for(std::size_t i=0;i<actual.size();++i){HYLE_RETURN_IF_ERROR(check(std::isfinite(actual[i]) && actual[i]<0.f,"CPU post-math did not negate native attention"));attended_values[i]=encode(-actual[i],false);}
    HYLE_ASSIGN_OR_RETURN(auto converted,load(host_pool,attended_values));
    HYLE_RETURN_IF_ERROR(oracle(cap,layout,expected_q,k,v,converted,false));
    HYLE_RETURN_IF_ERROR(check(!invocation.prepare_batch(std::array{gb::NodeLayout{attention,layout}}).ok(),"hybrid overwritten retained old output"));
    HYLE_ASSIGN_OR_RETURN(auto report,invocation.report(attention));
    HYLE_RETURN_IF_ERROR(check(report.layout_generation==layout.generation() && (!episode || report.reused_native_plan),"hybrid metadata generation/topology differs"));
    std::printf("G02 CPU RMS/Cast -> CUDA paged GQA8/2 D256 -> CPU Cast/Neg, episode=%u precise-pages/old-output/oracle PASS\n",episode);
  }
  HYLE_RETURN_IF_ERROR(invocation.close());HYLE_RETURN_IF_ERROR(invocation.drain());
  return Status::Ok();
}
} // namespace

int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=hybrid_paged();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"G02 hybrid: %s\n",text);return 1;}
