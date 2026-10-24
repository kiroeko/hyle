#if defined(HYLE_PENDING_GRAPH_CUDA)
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#else
#include <hyle/compute/cpu.h>
#endif
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
#include <hyle/graph/batch.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu.h>
#if !defined(HYLE_PENDING_GRAPH_CUDA)
#include <hyle/compute/providers/ggml_cpu.h>
#endif
#endif
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
#include <hyle/graph/cpu_cuda.h>
#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/providers/ggml_cpu.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>
namespace {
namespace c=hyle::compute;
namespace n=c;

using hyle::Status;
using hyle::Result;
Status require(bool v,const char* message){return v?Status::Ok():Status::Internal(message);}
double decode(std::uint16_t x,bool bf=false){
  if(bf)return std::bit_cast<float>(std::uint32_t(x)<<16);
  const unsigned e=(x>>10)&31U,f=x&1023U;
  return (x&0x8000U?-1.:1.)*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));
}
Status exercise([[maybe_unused]] c::Context&context,[[maybe_unused]] c::Queue&queue,bool bf=false,
    hyle::mem::MemoryResourceRef resource=hyle::mem::system_resource(),
    hyle::mem::MemoryResourceRef host_resource=hyle::mem::system_resource()){
  constexpr unsigned depth=256,heads=8,kv_heads=2;
  constexpr std::size_t row_elements=kv_heads*depth,row_bytes=row_elements*2;
  const c::BatchCapacity cap{.query_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,.output_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),
      .max_requests=2,.max_query_rows=18,.max_kv_rows=160,
      .max_pages=10,.max_page_refs=8,.query_heads=heads,.kv_heads=kv_heads,.qk_head_dim=depth,.value_head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
  namespace g=hyle::graph;namespace gb=g::batch;
  const c::BatchProgramOptions options{.resource=
#if defined(HYLE_PENDING_GRAPH_CUDA)
      {},
#else
      resource,
#endif
      .host_workspace=hyle::mem::system_resource(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=64u<<10,.max_invocations=2,.scale=1.F/16.F,.causal=true};
#if defined(HYLE_PENDING_GRAPH_CUDA)
  const auto native_provider = c::flashinfer::provider();
  HYLE_ASSIGN_OR_RETURN(auto gpu,c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto graph_context,c::Context::create(gpu,native_provider));
#else
  const auto native_provider = c::ggml_cpu::provider({.native_threads=2});
  HYLE_ASSIGN_OR_RETURN(auto graph_context,c::Context::create(c::cpu::device(),native_provider));
#endif
  HYLE_ASSIGN_OR_RETURN(auto graph_queue,graph_context.create_queue());
#if defined(HYLE_PENDING_GRAPH_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto graph_device,c::cuda::MemoryResource::create(graph_context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto graph_host,c::cuda::MemoryResource::create(graph_context,c::cuda::MemoryKind::kPinnedHost));
  resource=graph_device.ref();host_resource=graph_host.ref();
  auto&execution_context=graph_context;auto&execution_queue=graph_queue;
#else
  auto&execution_context=context;auto&execution_queue=queue;
#endif
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto cpu_context,c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=2})));
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu_context.create_queue());
#endif
#else
  auto&execution_context=context;auto&execution_queue=queue;
#endif
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,resource,{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,resource,{.max_backing_bytes=1<<20}));
  const auto page_bytes=keys.page_bytes();
  constexpr std::size_t query_bytes=18*heads*depth*2;
  const auto init_bytes=std::max(page_bytes,query_bytes);
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(host_resource,2*init_bytes+512,{.max_blocks=2}));
  const std::array specs{c::CopyResourceSpec{host_resource,init_bytes},c::CopyResourceSpec{resource,init_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,n::CopyBatchProgram::prepare(execution_context,specs));
  HYLE_ASSIGN_OR_RETURN(auto initialize,n::CopyBatchInvocation::create(init_program,execution_queue));
  const std::array<c::CopyPhase,1> phases{{{}}};
  const auto fill=[&](c::WriteLease&& writer,std::span<const std::uint16_t> data)->Result<c::ReadLease>{
    HYLE_RETURN_IF_ERROR(initialize.reset());
    HYLE_ASSIGN_OR_RETURN(auto hw,host_pool.allocate(data.size_bytes(),64));
    HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(data)));
    HYLE_ASSIGN_OR_RETURN(auto host,hw.publish());
    HYLE_RETURN_IF_ERROR(initialize.bind_source({0},host));
    HYLE_RETURN_IF_ERROR(initialize.bind_mutable({1},std::move(writer)));
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,data.size_bytes()}}};
    HYLE_RETURN_IF_ERROR(initialize.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto submitted,initialize.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
    return submitted.output({0},{1},0,data.size_bytes());
  };
  std::vector<std::uint16_t> zeros(page_bytes/2,0),q(18*heads*depth,bf?0x3e00:0x3000);
  std::array<c::ReadLease,10> kr,vr;
  for(unsigned p=0;p<10;++p){
    HYLE_ASSIGN_OR_RETURN(auto k,keys.acquire(p));HYLE_ASSIGN_OR_RETURN(kr[p],fill(std::move(k.lease()),zeros));HYLE_RETURN_IF_ERROR(k.complete(kr[p],p==0 || p==5?cap.page_size:0));
    HYLE_ASSIGN_OR_RETURN(auto v,values.acquire(p));HYLE_ASSIGN_OR_RETURN(vr[p],fill(std::move(v.lease()),zeros));HYLE_RETURN_IF_ERROR(v.complete(vr[p],p==0 || p==5?cap.page_size:0));
  }
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
  const std::vector<float> cpu_query_values(q.size(),-.125F);
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(hyle::mem::system_resource(),q.size()*4+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,query_pool.allocate(q.size()*4,64));
  HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(cpu_query_values))));
  HYLE_ASSIGN_OR_RETURN(auto query,qw.publish());
#else
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(resource,q.size()*2+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,query_pool.allocate(q.size()*2,64));
  HYLE_ASSIGN_OR_RETURN(auto query,fill(std::move(qw),q));
#endif
  HYLE_RETURN_IF_ERROR(initialize.reset());
  const auto source_resource=host_resource;
  const std::array update_specs{c::CopyResourceSpec{source_resource,4*row_bytes},
      c::CopyResourceSpec{resource,page_bytes},c::CopyResourceSpec{resource,page_bytes},
      c::CopyResourceSpec{resource,page_bytes},c::CopyResourceSpec{resource,page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto update_program,n::CopyBatchProgram::prepare(execution_context,update_specs));
  std::array<n::CopyBatchInvocation,2> copy_slots;
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{cap.max_query_rows,heads,depth}));
  HYLE_ASSIGN_OR_RETURN(auto kvdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,4>{cap.max_pages,cap.page_size,kv_heads,depth}));
  HYLE_ASSIGN_OR_RETURN(auto metadata,schema.metadata_description());
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto query_input_desc,c::TensorDesc::dense(c::DataType::kFloat32,qdesc.dimensions()));
  const auto output_desc=query_input_desc;constexpr auto output_bytes=2*query_bytes;
#else
  const auto query_input_desc=qdesc,output_desc=qdesc;constexpr auto output_bytes=query_bytes;
#endif
  const std::array descriptions{query_input_desc,kvdesc,kvdesc,metadata};
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());std::array<g::ValueId,4> inputs;
  for(unsigned i=0;i<inputs.size();++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
  auto ports=inputs;
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
  HYLE_ASSIGN_OR_RETURN(ports[0],builder.add(c::Unary{c::UnaryKind::kNeg},inputs[0]));
  HYLE_ASSIGN_OR_RETURN(ports[0],builder.add(c::Cast{cap.query_type},ports[0]));
#endif
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::PagedAttention{cap,options.scale,options.causal}),ports));
  HYLE_ASSIGN_OR_RETURN(auto output_id,builder.result(node));
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
  HYLE_ASSIGN_OR_RETURN(output_id,builder.add(c::Cast{c::DataType::kFloat32},output_id));
  HYLE_ASSIGN_OR_RETURN(output_id,builder.add(c::Unary{c::UnaryKind::kNeg},output_id));
#endif
  HYLE_RETURN_IF_ERROR(builder.publish(output_id));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
  namespace bridge=g::cpu_cuda;
  // Cold diagnostics distinguish missing CPU math from a cross-domain route
  // defect without replacing either required CPU fragment with GPU work.
  HYLE_ASSIGN_OR_RETURN(auto cpu_query,c::prepare(cpu_context,c::Unary{c::UnaryKind::kNeg},query_input_desc));
  HYLE_ASSIGN_OR_RETURN(auto cpu_narrow,c::prepare(cpu_context,c::Cast{cap.query_type},query_input_desc));
  HYLE_ASSIGN_OR_RETURN(auto cpu_widen,c::prepare(cpu_context,c::Cast{c::DataType::kFloat32},qdesc));
  std::fprintf(stderr,"CPU+CUDA exact CPU F32 Neg and F16/BF16 Cast preparations PASS\n");
  std::vector<g::NodePlacement> placements;
  for(std::size_t i=0;i<graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto id,graph.node_id(i));
    placements.push_back({id,id==node?bridge::kCudaTarget:bridge::kCpuTarget,g::PlacementStrength::kRequired});}
  const std::array external{g::ExternalResourceContract{inputs[0],bridge::kHostStorage},
      g::ExternalResourceContract{output_id,bridge::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,bridge::compile(graph,
      {&cpu_context,&graph_context,hyle::mem::system_resource(),&graph_host,&graph_device},
      {.node_placements=placements,.external_resources=external}));
  const auto output_resource=hyle::mem::system_resource();
#else
  const std::array storage{g::StorageDomain{resource}};
  HYLE_ASSIGN_OR_RETURN(auto target_source_0, hyle::graph::ExecutionSource::acquire(graph_context));
  const std::array targets{g::ExecutionTarget{target_source_0,g::StorageDomainId{0}}};
#if defined(HYLE_PENDING_GRAPH_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::cuda::compile(graph,{.storage_domains=storage,.targets=targets},
      {.operation_fallback=g::OperationFallbackPolicy::kReject}));
#else
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::cpu::compile(graph,{.storage_domains=storage,.targets=targets},
      {.operation_fallback=g::OperationFallbackPolicy::kReject}));
#endif
  const auto output_resource=resource;
#endif
  std::array<g::Invocation,2> slots;
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
    const std::array lanes{g::QueueBinding{g::ExecutionLaneId{0}, &graph_queue},
        g::QueueBinding{g::ExecutionLaneId{1}, &cpu_queue}};
#else
    const std::array lanes{g::QueueBinding{g::ExecutionLaneId{0}, &graph_queue}};
#endif
  const gb::PrepareOptions preparation{.host_workspace=host_resource,
      .max_float_workspace_bytes=options.max_float_workspace_bytes,
      .max_int_workspace_bytes=options.max_int_workspace_bytes,.max_source_completions=2};

  HYLE_ASSIGN_OR_RETURN(auto output_pool,c::LeasePool::create(output_resource,2*output_bytes+512,{.max_blocks=2}));
  std::array<c::WriteLease,2> output_writers;std::array<c::ReadLease,2> output_reads;
  for(auto&writer:output_writers){HYLE_ASSIGN_OR_RETURN(writer,output_pool.allocate(output_bytes,64));}
#else
  std::array<n::PagedBatchReceiver,2> slots;
#endif
  for(unsigned i=0;i<2;++i){
    HYLE_ASSIGN_OR_RETURN(copy_slots[i],n::CopyBatchInvocation::create(update_program,execution_queue));
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    HYLE_ASSIGN_OR_RETURN(slots[i],compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
#else
    HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,
        {.resource=resource,.max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,
         .max_invocations=1,.scale=1.F/16.F,.causal=true}));
    HYLE_ASSIGN_OR_RETURN(auto native,c::BatchInvocation::create(program,queue));
    HYLE_ASSIGN_OR_RETURN(slots[i],n::PagedBatchReceiver::create(std::move(native),2));
#endif
  }
  const std::array<c::CopyPhase,2> pending_phases{{{}, {.dependency_mask=1}}};
  const std::array<std::int32_t,3> decode_qp{0,1,2},first_prefill{0,17,18},second_prefill{0,1,18};
  const std::array<std::uint16_t,4> kbits=bf?std::array<std::uint16_t,4>{0x3c80,0x3d00,0x3d40,0x3d80}:std::array<std::uint16_t,4>{0x2400,0x2800,0x2a00,0x2c00},
      vbits=bf?std::array<std::uint16_t,4>{0x3e00,0x3e80,0x3ec0,0x3f00}:std::array<std::uint16_t,4>{0x3000,0x3400,0x3600,0x3800};
  std::array<std::array<std::uint16_t,128>,2> expected_k{},expected_v{};
  std::uint64_t dirty_total=0;
  for(unsigned step=0;step<64;++step){
    const std::array<unsigned,2> lengths{step+17,step/2+17};
    std::vector<std::int32_t> pages;
    std::array<std::int32_t,3> kp{0,0,0};std::array<std::int32_t,2> tails{};
    for(unsigned request=0;request<2;++request){
      const unsigned refs=(lengths[request]+15)/16;
      for(unsigned p=0;p<refs;++p)pages.push_back(std::int32_t(request*5+p));
      kp[request+1]=std::int32_t(pages.size());tails[request]=std::int32_t((lengths[request]-1)%16+1);
    }
    HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({step%3==0?second_prefill:step%3==1?first_prefill:decode_qp,kp,pages,tails}));
    // Explicit cold preparation must retire/drop all previous native imports
    // before we reacquire a ready page for this episode's exact update.
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    const std::array node_layouts{gb::NodeLayout{node,layout}};
    for(unsigned i=0;i<slots.size();++i){
      HYLE_RETURN_IF_ERROR(gb::prepare(slots[i],lanes,node_layouts,preparation));
      if(output_reads[i].valid()){HYLE_ASSIGN_OR_RETURN(output_writers[i],output_reads[i].try_write());}
    }
#else
    for(auto&receiver:slots)HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
#endif
    auto&slot=slots[step%2];auto&update=copy_slots[step%2];
    const unsigned updates=step%2==0?2:1;
    HYLE_RETURN_IF_ERROR(update.reset());
    std::vector<std::uint16_t> incoming(2*updates*row_elements);
    std::array<c::PagedPageWriter,4> writers;
    std::array<std::uint64_t,4> generations{};
    std::array<unsigned,2> destinations{};
    std::vector<c::CopyRange> ranges;
    for(unsigned request=0;request<updates;++request){
      const unsigned row=lengths[request]-1,page=request*5+row/16;
      destinations[request]=page;
      expected_k[request][row]=kbits[(step+request)%4];expected_v[request][row]=vbits[(step+2*request)%4];
      std::fill_n(incoming.begin()+2*request*row_elements,row_elements,expected_k[request][row]);
      std::fill_n(incoming.begin()+(2*request+1)*row_elements,row_elements,expected_v[request][row]);
      generations[2*request]=kr[page].generation();generations[2*request+1]=vr[page].generation();
      HYLE_ASSIGN_OR_RETURN(writers[2*request],keys.acquire(kr[page]));
      HYLE_ASSIGN_OR_RETURN(writers[2*request+1],values.acquire(vr[page]));
      for(unsigned type=0;type<2;++type){const unsigned port=2*request+type+1;
        HYLE_RETURN_IF_ERROR(update.bind_mutable({port},std::move(writers[port-1].lease())));
        ranges.push_back({{0},{port},{1},(port-1)*row_bytes,(row%16)*row_bytes,row_bytes});
      }
    }
    HYLE_ASSIGN_OR_RETURN(auto hw,host_pool.allocate(incoming.size()*2,64));
    HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(std::span(incoming))));
    HYLE_ASSIGN_OR_RETURN(auto host,hw.publish());HYLE_RETURN_IF_ERROR(update.bind_source({0},host));
    HYLE_RETURN_IF_ERROR(update.prepare_replay(pending_phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto copied,update.submit());
    HYLE_ASSIGN_OR_RETURN(auto owner,n::CopyBatchUpdate::take(std::move(update),std::move(copied)));
    std::vector<c::ReadLease> key_reads,value_reads;
    std::vector<c::PagedPageUpdate> key_updates,value_updates;
    for(auto page:pages){
      bool pending=false;
      for(unsigned request=0;request<updates;++request)if(unsigned(page)==destinations[request]){
        key_updates.push_back({&writers[2*request],{2*request+1}});
        value_updates.push_back({&writers[2*request+1],{2*request+2}});pending=true;break;
      }
      if(!pending){key_reads.push_back(kr[page]);value_reads.push_back(vr[page]);}
    }
    HYLE_ASSIGN_OR_RETURN(auto kframe,keys.pending_read(layout,key_reads,key_updates));
    HYLE_ASSIGN_OR_RETURN(auto vframe,values.pending_read(layout,value_reads,value_updates));
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    HYLE_RETURN_IF_ERROR(slot.bind(inputs[0],query_input_desc,query));
    HYLE_RETURN_IF_ERROR(gb::bind(slot,inputs[1],kframe));HYLE_RETURN_IF_ERROR(gb::bind(slot,inputs[2],vframe));
    HYLE_RETURN_IF_ERROR(slot.bind(output_id,output_desc,std::move(output_writers[step%2])));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay(lanes));HYLE_ASSIGN_OR_RETURN(auto submitted,gb::submit(slot,lanes,std::move(owner)));
#else
    HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(query,kframe,vframe,std::move(owner)));
#endif
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    HYLE_RETURN_IF_ERROR(require(!owner.valid() && !gb::take_update(slot),"pending source was not retained by original suffix"));
#else
    HYLE_RETURN_IF_ERROR(require(!owner.valid() && !slot.take_update(),"pending source was not retained by original suffix"));
#endif
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    HYLE_RETURN_IF_ERROR(require(!gb::prepare(slot,lanes,node_layouts,preparation).ok(),"retained aggregate allowed Graph prepare"));
#else
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"retained aggregate allowed receiver prepare"));
#endif
    {HYLE_ASSIGN_OR_RETURN(auto free,keys.acquire(kr[9]));HYLE_ASSIGN_OR_RETURN(kr[9],free.publish(0));}
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    HYLE_RETURN_IF_ERROR(require(!slot.result(output_id),"unobserved Graph aggregate published output"));
#else
    HYLE_RETURN_IF_ERROR(require(!slot.output(),"unobserved common aggregate published output"));
#endif
    HYLE_RETURN_IF_ERROR(submitted.wait());
    {
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
      HYLE_ASSIGN_OR_RETURN(auto graph_output,slot.result(output_id));
      HYLE_ASSIGN_OR_RETURN(auto output,graph_output.storage_lease());output_reads[step%2]=output;
#else
      HYLE_ASSIGN_OR_RETURN(auto output,slot.output());
#endif
      const auto active=layout.facts().query_rows;
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
      std::vector<float> actual(active*heads*depth);
#else
      std::vector<std::uint16_t> actual(active*heads*depth);
#endif
      HYLE_ASSIGN_OR_RETURN(auto active_output,output.view()->subview(0,std::span(actual).size_bytes()));
      HYLE_RETURN_IF_ERROR(active_output.read(std::as_writable_bytes(std::span(actual))));
      const auto ptr=layout.host().query_indptr;
      for(unsigned request=0;request<2;++request){
        const auto nq=unsigned(ptr[request+1]-ptr[request]);
        for(unsigned query_row=0;query_row<nq;++query_row){double sum=0,weighted=0;
          const auto visible=lengths[request]-nq+query_row+1;
          for(unsigned row=0;row<visible;++row){
            const double weight=std::exp(.125*depth*decode(expected_k[request][row],bf)/16.);
            sum+=weight;weighted+=weight*decode(expected_v[request][row],bf);
          }
          const double expected=
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
            -weighted/sum;
#else
            weighted/sum;
#endif
          for(unsigned i=0;i<heads*depth;++i){const auto actual_value=
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
              actual[(ptr[request]+query_row)*heads*depth+i];
#else
              decode(actual[(ptr[request]+query_row)*heads*depth+i],bf);
#endif
            HYLE_RETURN_IF_ERROR(require(std::isfinite(actual_value) && std::abs(actual_value-expected)<(bf?.008:.002),"native dirty attention independent oracle differs"));
          }
        }
      }
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
      HYLE_RETURN_IF_ERROR(require(!gb::prepare(slot,lanes,node_layouts,preparation).ok() && !gb::take_update(slot),"retained Graph result permitted prepare/control recovery"));
#else
      HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"retained native result permitted prepare/update"));
#endif
    }
    // The mapped completion is the public/native result path; source need not
    // be separately waited. Recovery uses the two original update controls.
    HYLE_RETURN_IF_ERROR(submitted.wait());
    submitted={};
#if defined(HYLE_PENDING_GRAPH_RECEIVER)
    HYLE_ASSIGN_OR_RETURN(owner,gb::take_update(slot));
#else
    HYLE_ASSIGN_OR_RETURN(owner,slot.take_update());
#endif
    HYLE_RETURN_IF_ERROR(require(owner.report().logical_bytes==2*updates*row_bytes,"pending update copied history"));
    dirty_total+=owner.report().logical_bytes;
    for(unsigned request=0;request<updates;++request){
      const auto page=destinations[request];
      HYLE_ASSIGN_OR_RETURN(writers[2*request].lease(),owner.take_mutable({2*request+1}));
      HYLE_ASSIGN_OR_RETURN(writers[2*request+1].lease(),owner.take_mutable({2*request+2}));
      HYLE_ASSIGN_OR_RETURN(kr[page],writers[2*request].publish(std::uint32_t(tails[request])));
      HYLE_ASSIGN_OR_RETURN(vr[page],writers[2*request+1].publish(std::uint32_t(tails[request])));
      HYLE_RETURN_IF_ERROR(require(kr[page].generation()==generations[2*request] && vr[page].generation()==generations[2*request+1],"pending update changed original generation"));
    }
    HYLE_ASSIGN_OR_RETURN(update,owner.release_invocation());
  }
  for(auto&update:copy_slots)HYLE_RETURN_IF_ERROR(update.close());
  for(auto&slot:slots)HYLE_RETURN_IF_ERROR(slot.drain());
  constexpr const char*route=
#if defined(HYLE_PENDING_GRAPH_CPU_CUDA)
      "CPU+CUDA Graph";
#elif defined(HYLE_PENDING_GRAPH_CUDA)
      "CUDA Graph";
#elif defined(HYLE_PENDING_GRAPH_RECEIVER)
      "CPU Graph";
#else
      "CPU receiver";
#endif
  std::fprintf(stderr,"common %s pending PASS:64 steps GQA8/2 D256, two requests growth17..80/17..48, mixed Q1/17 prefill/decode, crosspage, exact frame, untouched history, same generation, BF16=%d dirty total=%llu bytes, original prepare releases imports; cold metadata waits retained, no source wait before attention\n",route,bf,static_cast<unsigned long long>(dirty_total));
  return Status::Ok();
}
Status run(){
#if defined(HYLE_PENDING_GRAPH_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto gpu,c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(gpu,c::flashinfer::provider()));
#else
  HYLE_ASSIGN_OR_RETURN(auto context,c::cpu::create_context({.native_threads=2}));
#endif
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Close{c::Context&context;~Close(){context.drain().IgnoreError();}}close{context};
#if defined(HYLE_PENDING_GRAPH_CUDA)
  HYLE_RETURN_IF_ERROR(exercise(context,queue,false));
  return exercise(context,queue,true);
#else
  HYLE_RETURN_IF_ERROR(exercise(context,queue));return exercise(context,queue,true);
#endif
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"common receiver: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
