#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/graph/vulkan.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace v=c::vulkan;
using hyle::Result;using hyle::Status;
constexpr g::StorageDomainId host_storage{0},device_storage{1};
constexpr unsigned heads=8,kv_heads=2,depth=64,page_size=16,page_count=5;
constexpr double tolerance=.002;
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Drain{hyle::sched::Scheduler&s;~Drain(){s.drain();}};

double half(std::uint16_t bits){
  const auto exponent=(bits>>10)&31U,mantissa=bits&1023U;
  return (bits&0x8000U?-1.:1.)*(exponent?
      std::ldexp(1.+double(mantissa)/1024.,int(exponent)-15):std::ldexp(double(mantissa),-24));
}
// Construct exact Q8_0 bytes directly. This does not use either provider's
// encoder; the independent oracle below decodes the bytes actually uploaded.
std::vector<std::byte> bank(bool values){
  constexpr unsigned blocks=depth/32,row_bytes=kv_heads*blocks*34;
  std::vector<std::byte> bytes(page_count*page_size*row_bytes);
  for(unsigned row=0;row<page_count*page_size;++row)for(unsigned head=0;head<kv_heads;++head)
    for(unsigned block=0;block<blocks;++block){
      const auto offset=(row*kv_heads*blocks+head*blocks+block)*34;
      const std::uint16_t scale=values?0x2800U:0x2400U; // 1/32 and 1/64.
      bytes[offset]=std::byte(scale&255U);bytes[offset+1]=std::byte(scale>>8);
      for(unsigned lane=0;lane<32;++lane){
        const auto channel=block*32+lane;
        const int code=values?int((row*7+head*3+channel*5)%17)-8:
            int((row*3+head*5+channel)%15)-7;
        bytes[offset+2+lane]=std::byte(static_cast<std::uint8_t>(code));
      }
    }
  return bytes;
}
double value(const std::vector<std::byte>&bytes,unsigned row,unsigned head,unsigned channel){
  constexpr unsigned blocks=depth/32;
  const auto offset=(row*kv_heads*blocks+head*blocks+channel/32)*34;
  const auto bits=std::uint16_t(std::to_integer<unsigned>(bytes[offset])|
      std::to_integer<unsigned>(bytes[offset+1])<<8);
  const auto code=std::bit_cast<std::int8_t>(std::to_integer<std::uint8_t>(bytes[offset+2+channel%32]));
  return half(bits)*code;
}
std::vector<double> oracle(const c::BatchLayoutVersion&layout,std::span<const float>query,
    const std::vector<std::byte>&keys,const std::vector<std::byte>&values){
  const auto host=layout.host();std::vector<double> result(layout.facts().query_rows*heads*depth);
  for(std::size_t request=0;request<layout.facts().requests;++request){
    const auto first=host.query_indptr[request],queries=host.query_indptr[request+1]-first;
    const auto refs=host.kv_indptr[request+1]-host.kv_indptr[request];
    const auto rows=refs?(refs-1)*int(page_size)+host.last_page_lengths[request]:0;
    for(int q=0;q<queries;++q)for(unsigned head=0;head<heads;++head){
      const int end=rows-queries+q+1,begin=std::max(0,end-1-16);
      std::vector<double> scores(std::size_t(end-begin));double maximum=-INFINITY;
      const auto physical=[&](int key){return unsigned(host.page_indices[
          std::size_t(host.kv_indptr[request])+unsigned(key)/page_size])*page_size+unsigned(key)%page_size;};
      for(int key=begin;key<end;++key){
        double dot=0;for(unsigned d=0;d<depth;++d)
          dot+=std::abs(double(query[(std::size_t(first+q)*heads+head)*depth+d]))*
              value(keys,physical(key),head/(heads/kv_heads),d);
        scores[std::size_t(key-begin)]=dot*.125;maximum=std::max(maximum,dot*.125);
      }
      double denominator=0;for(auto&score:scores){score=std::exp(score-maximum);denominator+=score;}
      for(unsigned d=0;d<depth;++d){
        double sum=0;for(int key=begin;key<end;++key)
          sum+=scores[std::size_t(key-begin)]*value(values,physical(key),head/(heads/kv_heads),d);
        result[(std::size_t(first+q)*heads+head)*depth+d]=-sum/denominator;
      }
    }
  }
  return result;
}
Status check_output(g::GraphOutput&output,std::span<const double>expected){
  HYLE_ASSIGN_OR_RETURN(auto tensor,output.tensor());
  HYLE_ASSIGN_OR_RETURN(auto elements,tensor.description().element_count());
  HYLE_RETURN_IF_ERROR(require(elements==expected.size(),"semantic output kept capacity rows"));
  std::vector<float> actual(elements);HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
  double maximum=0;
  for(std::size_t i=0;i<actual.size();++i){
    if(!std::isfinite(actual[i]) || std::abs(double(actual[i])-expected[i])>tolerance){
      std::fprintf(stderr,"semantic mismatch index=%zu actual=%.9g expected=%.17g\n",i,double(actual[i]),expected[i]);
      return Status::Internal("semantic CPU/Vulkan output differs from independent FP64 attention");
    }
    maximum=std::max(maximum,std::abs(double(actual[i])-expected[i]));
  }
  std::fprintf(stderr,"semantic oracle elements=%zu max_abs=%.9g\n",actual.size(),maximum);return Status::Ok();
}

struct GraphData{g::Graph graph;g::ValueId query,key,value,converted,output;g::NodeId page;
  c::TensorDesc query_description,key_description,value_description,output_description;};
Result<GraphData> build_graph(const c::BatchLayoutVersion&initial){
  GraphData result;const auto schema=initial.schema();const auto cap=schema.capacity();
  HYLE_ASSIGN_OR_RETURN(result.query_description,c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,3>{initial.facts().query_rows,heads,depth}));
  HYLE_ASSIGN_OR_RETURN(result.key_description,c::batch_cache_description(schema,c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(result.value_description,c::batch_cache_description(schema,c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(result.query,builder.input(result.query_description));
  HYLE_ASSIGN_OR_RETURN(result.key,builder.input(result.key_description));
  HYLE_ASSIGN_OR_RETURN(result.value,builder.input(result.value_description));
  HYLE_ASSIGN_OR_RETURN(auto positive,builder.add(c::Unary{c::UnaryKind::kAbs},result.query));
  HYLE_ASSIGN_OR_RETURN(result.converted,builder.add(c::Cast{cap.query_type},positive));
  HYLE_ASSIGN_OR_RETURN(auto page,g::batch::paged_attention(initial,{.scale=.125F,.causal=true,.window_left=16}));
  const std::array inputs{result.converted,result.key,result.value};
  HYLE_ASSIGN_OR_RETURN(result.page,builder.add_node(c::Operation::custom(page),inputs));
  HYLE_ASSIGN_OR_RETURN(auto attended,builder.result(result.page));
  HYLE_ASSIGN_OR_RETURN(auto widened,builder.add(c::Cast{c::DataType::kFloat32},attended));
  HYLE_ASSIGN_OR_RETURN(result.output,builder.add(c::Unary{c::UnaryKind::kNeg},widened));
  HYLE_RETURN_IF_ERROR(builder.publish(result.output));HYLE_ASSIGN_OR_RETURN(result.graph,std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(result.output_description,result.graph.value_description(result.output));
  return result;
}

Status fill_host(c::PagedArena&arena,std::span<const std::byte>bytes,std::array<c::ReadLease,page_count>&reads){
  for(unsigned page=0;page+1<page_count;++page){
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));HYLE_ASSIGN_OR_RETURN(auto view,writer.lease().view());
    HYLE_RETURN_IF_ERROR(view.write(bytes.subspan(page*arena.page_bytes(),arena.page_bytes())));
    HYLE_ASSIGN_OR_RETURN(reads[page],writer.publish(page_size));
  }return Status::Ok();
}
Status fill_vulkan(v::Context&context,v::MemoryResource&staging,hyle::mem::MemoryResourceRef resource,c::PagedArena&arena,
    std::span<const std::byte>bytes,std::array<c::ReadLease,page_count>&reads){
  const auto page_bytes=arena.page_bytes();
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),page_bytes+64));
  const std::array resources{c::CopyResourceSpec{staging.ref(),page_bytes},c::CopyResourceSpec{resource,page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto program,v::CopyBatchProgram::prepare(context,resources));
  HYLE_ASSIGN_OR_RETURN(auto upload,v::CopyBatchInvocation::create(program));
  for(unsigned page=0;page+1<page_count;++page){
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));HYLE_RETURN_IF_ERROR(upload.reset());
    HYLE_ASSIGN_OR_RETURN(auto host,pool.allocate(page_bytes,64));
    HYLE_RETURN_IF_ERROR(host.view()->write(bytes.subspan(page*page_bytes,page_bytes)));
    HYLE_ASSIGN_OR_RETURN(auto source,host.publish());HYLE_RETURN_IF_ERROR(upload.bind_source({0},source));
    HYLE_RETURN_IF_ERROR(upload.bind_mutable({1},std::move(writer.lease())));
    const std::array<c::CopyPhase,1> phases{{{}}};
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,page_bytes}}};
    HYLE_RETURN_IF_ERROR(upload.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto done,upload.submit());
    HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(reads[page],done.output({0},{1},0,page_bytes));
    HYLE_RETURN_IF_ERROR(writer.complete(reads[page],page_size));
  }return Status::Ok();
}
Result<std::vector<g::QueueBinding>> queues_for(const g::CompileReport&report,c::Queue&queue){
  std::vector<g::QueueBinding> result;
  for(std::size_t i=0;i<report.lane_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto lane,report.lane(i));
    if(lane.queue_required){
      HYLE_RETURN_IF_ERROR(require(lane.runtime_mode==c::ProviderRuntimeMode::kSharedQueue,
          "delegated native page requested a public Compute Queue"));result.push_back({lane.lane,&queue});
    }
  }return result;
}

Status execute(const GraphData&data,const c::BatchLayoutVersion&initial,const c::BatchLayoutVersion&next,
    c::Context&cpu,v::Context&native,v::MemoryResource&device,v::MemoryResource&staging,bool use_vulkan){
  // This function changes actual sources/resources/placement only. The caller
  // passes the same immutable Graph object to both compilations.
  const auto schema=initial.schema();const bool empty=initial.facts().query_rows==0;
  const auto page_resource=use_vulkan?device.ref():hyle::mem::system_resource();
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,page_resource,{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,page_resource,{.max_backing_bytes=1<<20}));
  const auto key_bytes=bank(false),value_bytes=bank(true);
  std::array<c::ReadLease,page_count> kr,vr;
  if(!empty){
    if(use_vulkan){HYLE_RETURN_IF_ERROR(fill_vulkan(native,staging,page_resource,keys,key_bytes,kr));HYLE_RETURN_IF_ERROR(fill_vulkan(native,staging,page_resource,values,value_bytes,vr));}
    else{HYLE_RETURN_IF_ERROR(fill_host(keys,key_bytes,kr));HYLE_RETURN_IF_ERROR(fill_host(values,value_bytes,vr));}
  }
  std::array<c::PagedPageWriter,page_count> key_writers,value_writers;
  for(unsigned page=empty?0u:page_count-1;page<page_count;++page){
    HYLE_ASSIGN_OR_RETURN(key_writers[page],keys.acquire(page));
    HYLE_ASSIGN_OR_RETURN(value_writers[page],values.acquire(page));
  }
  const auto make_frame=[&](c::PagedArena&arena,const c::BatchLayoutVersion&layout,const auto&reads)->Result<c::PagedArenaRead>{
    if(empty)return arena.read(layout,{});
    const std::array selected{reads[0],reads[2],reads[layout==initial?1:3]};return arena.read(layout,selected);
  };
  HYLE_ASSIGN_OR_RETURN(auto old_key,make_frame(keys,initial,kr));HYLE_ASSIGN_OR_RETURN(auto old_value,make_frame(values,initial,vr));
  HYLE_ASSIGN_OR_RETURN(auto new_key,make_frame(keys,next,kr));HYLE_ASSIGN_OR_RETURN(auto new_value,make_frame(values,next,vr));
  HYLE_RETURN_IF_ERROR(require(old_key.page_count()==(empty?0u:3u) && new_key.page_count()==(empty?0u:3u),
      "typed semantic frame does not have the exact referenced page footprint"));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(cpu));
  HYLE_ASSIGN_OR_RETURN(auto native_source,g::vulkan::execution_source(native));
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()},g::StorageDomain{device.ref()},g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{use_vulkan?native_source:cpu_source,use_vulkan?device_storage:host_storage},
      g::ExecutionTarget{cpu_source,host_storage}};
  std::vector<g::NodePlacement> placements;
  for(std::size_t i=0;i<data.graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto node,data.graph.node_id(i));
    placements.push_back({node,g::ExecutionTargetId{use_vulkan && node!=data.page?1u:0u},g::PlacementStrength::kRequired});}
  const std::array external{g::ExternalResourceContract{data.query,host_storage},
      g::ExternalResourceContract{data.key,use_vulkan?device_storage:host_storage},
      g::ExternalResourceContract{data.value,use_vulkan?device_storage:host_storage},
      g::ExternalResourceContract{data.output,host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::batch::compile(data.graph,
      {.storage_domains=domains,.targets=std::span(targets).first(use_vulkan?2u:1u)},
      {.node_placements=placements,.external_resources=external}));
  std::size_t page_steps=0,device_uploads=0,total_query_copy_bytes=0;
  for(std::size_t i=0;i<compiled.report.step_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto step,compiled.report.step(i));
    if(step.kind==g::CompileStepKind::kCopy){
      HYLE_ASSIGN_OR_RETURN(auto input,compiled.report.step_input(i,0));HYLE_ASSIGN_OR_RETURN(auto physical,compiled.report.physical_value(input));
      if(physical.value==data.converted){
        const auto expected=std::size_t(initial.facts().query_rows)*heads*depth*2;
        HYLE_RETURN_IF_ERROR(require(step.copy_bytes==expected && physical.description.dimensions()[0]==initial.facts().query_rows,
            "semantic query copy processed capacity rows"));
        device_uploads+=step.destination_storage==device_storage;total_query_copy_bytes+=step.copy_bytes;
      }continue;
    }
    for(std::size_t n=0;n<step.node_count;++n){HYLE_ASSIGN_OR_RETURN(auto node,compiled.report.step_node(i,n));
      if(node==data.page){++page_steps;HYLE_RETURN_IF_ERROR(require(step.kind==g::CompileStepKind::kProviderRegion &&
          step.runtime.mode==(use_vulkan?c::ProviderRuntimeMode::kDelegated:c::ProviderRuntimeMode::kSharedQueue) &&
          step.runtime.blocks_host==!use_vulkan,
          "semantic page did not select its real batch implementation"));}}
  }
  HYLE_RETURN_IF_ERROR(require(page_steps==1 && (!use_vulkan || empty || device_uploads==1),
      "semantic page route/upload count differs"));
  HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());HYLE_ASSIGN_OR_RETURN(auto queues,queues_for(compiled.report,queue));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool workers(hyle::mem::system_allocator(),2);
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto elements,data.query_description.element_count());std::vector<float> query(elements);
  for(std::size_t i=0;i<query.size();++i)query[i]=float(int(i%17)-8)*.03125F;
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(hyle::mem::system_resource(),query.size()*4+64));
  HYLE_ASSIGN_OR_RETURN(auto writer,query_pool.allocate(std::max<std::size_t>(1,query.size()*4),64));
  HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{0}));
  if(!query.empty())HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(query))));
  HYLE_ASSIGN_OR_RETURN(auto input,writer.publish());
  const auto old_expected=oracle(initial,query,key_bytes,value_bytes),new_expected=oracle(next,query,key_bytes,value_bytes);
  std::array<g::Invocation,2> slots;std::array<c::LeasePool,2> output_pools;
  std::array<g::GraphOutput,2> results;std::array<c::Completion,2> completions;
  HYLE_ASSIGN_OR_RETURN(auto output_bytes,data.output_description.required_bytes());
  for(unsigned i=0;i<slots.size();++i){
    HYLE_ASSIGN_OR_RETURN(slots[i],compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(slots[i],scheduler,workers));
    HYLE_ASSIGN_OR_RETURN(output_pools[i],c::LeasePool::create(hyle::mem::system_resource(),output_bytes+64,{.max_blocks=1}));
  }
  const auto prepare=[&](unsigned index,bool next_layout)->Status{
    auto&slot=slots[index];const auto&layout=next_layout?next:initial;
    const std::array layouts{g::batch::NodeLayout{data.page,layout}};
    HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,queues,layouts));
    HYLE_RETURN_IF_ERROR(slot.bind(data.query,data.query_description,input));
    // Same schema/capacity is insufficient: the original selected layout must
    // match the frame exactly, including the immutable version identity.
    HYLE_RETURN_IF_ERROR(require(!g::batch::bind(slot,data.key,next_layout?old_key:new_key).ok(),
        "semantic reader accepted a frame from another prepared layout"));
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,data.key,next_layout?new_key:old_key));
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,data.value,next_layout?new_value:old_value));
    HYLE_ASSIGN_OR_RETURN(auto output,output_pools[index].allocate(std::max<std::size_t>(output_bytes,1),64));
    HYLE_RETURN_IF_ERROR(slot.bind(data.output,data.output_description,std::move(output)));
    return slot.prepare_replay(queues);
  };
  const auto submit=[&](unsigned index,std::span<const double>expected)->Status{
    HYLE_ASSIGN_OR_RETURN(completions[index],slots[index].submit(queues));HYLE_RETURN_IF_ERROR(completions[index].wait());
    HYLE_ASSIGN_OR_RETURN(results[index],slots[index].result(data.output));HYLE_RETURN_IF_ERROR(check_output(results[index],expected));
    for(unsigned page=empty?0u:page_count-1;page<page_count;++page)
      HYLE_RETURN_IF_ERROR(require(key_writers[page].valid() && value_writers[page].valid(),
          "semantic attention acquired an unreferenced or empty cache page"));
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(prepare(0,false));HYLE_RETURN_IF_ERROR(submit(0,old_expected));
  HYLE_RETURN_IF_ERROR(prepare(1,true));HYLE_RETURN_IF_ERROR(submit(1,new_expected));
  HYLE_RETURN_IF_ERROR(check_output(results[0],old_expected));
  const std::array next_layouts{g::batch::NodeLayout{data.page,next}};
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slots[0],queues,next_layouts).ok() &&
      !slots[0].prepare_replay(queues).ok() && !slots[1].submit(queues),
      "semantic old result did not guard prepare/replay"));
  results[1]={};completions[1]={};
  // Strict original replay: no bind, batch::prepare or prepare_replay here.
  HYLE_RETURN_IF_ERROR(submit(1,new_expected));results[1]={};completions[1]={};
  // Re-prepare unchanged bindings only after dropping public results. The native
  // page adapter must release its internal completed result before its reuse gate.
  HYLE_RETURN_IF_ERROR(slots[1].prepare_replay(queues));
  HYLE_RETURN_IF_ERROR(submit(1,new_expected));results[1]={};completions[1]={};
  results[0]={};completions[0]={};HYLE_RETURN_IF_ERROR(prepare(0,true));HYLE_RETURN_IF_ERROR(submit(0,new_expected));
  results[0]={};completions[0]={};for(auto&slot:slots)HYLE_RETURN_IF_ERROR(slot.close());
  std::fprintf(stderr,"same Graph semantic PASS endpoint=%s Q=%u capacityQ=%u slots=2 query-copy-bytes=%zu\n",
      use_vulkan?"Vulkan":"CPU",unsigned(initial.facts().query_rows),unsigned(schema.capacity().max_query_rows),total_query_copy_bytes);
  return Status::Ok();
}

Status run(){
  const auto q8=c::CacheStorage::q8_0().value();
  const c::BatchCapacity cap{.key_storage=q8,.value_storage=q8,.max_requests=2,.max_query_rows=24,
      .max_kv_rows=36,.max_pages=page_count,.max_page_refs=4,.query_heads=heads,.kv_heads=kv_heads,
      .qk_head_dim=depth,.value_head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto native,v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device,v::MemoryResource::create(native,v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,v::MemoryResource::create(native,v::MemoryKind::kStaging));
  const std::array<std::int32_t,3> kp{0,2,4},empty_ptr{0,0,0};
  const std::array<std::int32_t,4> old_pages{0,1,0,2},new_pages{0,3,0,2};
  const std::array<std::int32_t,2> old_tails{1,1},new_tails{2,2},empty_tails{0,0};
  for(unsigned count:{18u,2u,0u}){
    const std::array<std::int32_t,3> qp{0,count?1:0,std::int32_t(count)},next_qp{0,count==18?17:std::int32_t(count),std::int32_t(count)};
    HYLE_ASSIGN_OR_RETURN(auto initial,schema.version(count?c::HostBatchLayout{qp,kp,old_pages,old_tails}:c::HostBatchLayout{empty_ptr,empty_ptr,{},empty_tails}));
    HYLE_ASSIGN_OR_RETURN(auto next,schema.version(count?c::HostBatchLayout{next_qp,kp,new_pages,new_tails}:c::HostBatchLayout{empty_ptr,empty_ptr,{},empty_tails}));
    HYLE_ASSIGN_OR_RETURN(auto graph,build_graph(initial));
    HYLE_RETURN_IF_ERROR(execute(graph,initial,next,cpu,native,device,staging,false));
    HYLE_RETURN_IF_ERROR(execute(graph,initial,next,cpu,native,device,staging,true));
  }
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(!status.ok()){
  std::fprintf(stderr,"paged semantics: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
