#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/sched/scheduler.h>
#if defined(HYLE_JOINT_PAGE_APPLE)
#include <hyle/graph/apple.h>
#else
#include <hyle/graph/vulkan.h>
#endif
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/mem/buffer.h>
#include <hyle/sched/blocking.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
#if defined(HYLE_JOINT_PAGE_APPLE)
namespace n=c::apple;namespace ng=g::apple;
#else
namespace n=c::vulkan;namespace ng=g::vulkan;
#endif
using hyle::Status;using hyle::Result;
constexpr g::ExecutionTargetId kNativeTarget{0},kCpuTarget{1};
constexpr g::StorageDomainId kHostStorage{0},kDeviceStorage{1};
Result<std::vector<g::QueueBinding>> shared_queues(const g::CompileReport& report,c::Queue& queue){
  std::vector<g::QueueBinding> result;
  for(std::size_t i=0;i<report.lane_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto lane,report.lane(i));
    if(!lane.queue_required)continue;
    if(lane.target!=kCpuTarget || lane.runtime_mode!=c::ProviderRuntimeMode::kSharedQueue)
      return std::unexpected(Status::Internal("native page requested a caller Compute Queue"));
    result.push_back({lane.lane,&queue});
  }
  return result;
}
Status check(bool value,const char* message){return value?Status::Ok():Status::Internal(message);}
std::uint16_t encode(float x,bool bf){
  const auto bits=std::bit_cast<std::uint32_t>(x);
  if(bf)return std::uint16_t(bits>>16);
  if(x==0)return 0;
  return std::uint16_t(((bits>>16)&0x8000U)|(((bits>>23&255U)-112U)<<10)|((bits&0x7fffffU)>>13));
}
double decode(std::uint16_t bits,bool bf){
  if(bf)return std::bit_cast<float>(std::uint32_t(bits)<<16);
  const unsigned e=(bits>>10)&31U,f=bits&1023U;const double sign=(bits&0x8000U)?-1.:1.;
  return sign*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));
}
struct Drain final{hyle::sched::Scheduler& scheduler;~Drain(){scheduler.drain();}};
Status exercise(c::Context& cpu,n::Context& native,n::MemoryResource& device,
    n::MemoryResource& staging,unsigned depth,bool bf,bool causal){
  const c::BatchCapacity cap{.query_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,.output_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),
      .max_requests=3,.max_query_rows=24,.max_kv_rows=128,.max_pages=4,.max_page_refs=8,
      .query_heads=8,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,4> qp{0,1,18,18},next_qp{0,17,18,18},single_qp{0,18,18,18},kp{0,5,7,7};
  const std::array<std::int32_t,7> ids{3,1,3,1,3,1,0};const std::array<std::int32_t,3> tails{15,7,0};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto next,schema.version({next_qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto single,schema.version({single_qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=32,.scale=1.F/std::sqrt(float(depth)),.causal=causal}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=1<<20}));
  std::vector<float> query(std::size_t(layout.facts().query_rows)*8*depth);
  for(std::size_t i=0;i<query.size();++i)query[i]=float(int(i%13)-6)*.03125F;
  std::vector<std::uint16_t> k(64*2*depth),v(k.size());
  for(std::size_t i=0;i<k.size();++i){k[i]=encode(float(int(i%17)-8)*.015625F,bf);v[i]=encode(float(int(i%29)-14)*.03125F,bf);}
  for(unsigned row=32;row<48;++row)for(unsigned d=0;d<2*depth;++d){k[row*2*depth+d]=encode(64.F,bf);v[row*2*depth+d]=encode(128.F,bf);}
  const auto bytes=keys.page_bytes();HYLE_ASSIGN_OR_RETURN(auto hp,c::LeasePool::create(staging.ref(),2*bytes+512,{.max_blocks=2}));
  const std::array resources{c::CopyResourceSpec{staging.ref(),bytes},c::CopyResourceSpec{device.ref(),bytes}};
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchProgram::prepare(native,resources));HYLE_ASSIGN_OR_RETURN(auto upload,n::CopyBatchInvocation::create(copy));
  std::array<c::ReadLease,4> kr,vr;
  const auto fill=[&](c::PagedArena& arena,const std::vector<std::uint16_t>& data,unsigned page)->Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));HYLE_RETURN_IF_ERROR(upload.reset());
    HYLE_ASSIGN_OR_RETURN(auto host,hp.allocate(bytes,64));HYLE_RETURN_IF_ERROR(host.view()->write(std::as_bytes(std::span(data).subspan(page*bytes/2,bytes/2))));
    HYLE_ASSIGN_OR_RETURN(auto ready,host.publish());HYLE_RETURN_IF_ERROR(upload.bind_source({0},ready));HYLE_RETURN_IF_ERROR(upload.bind_mutable({1},std::move(writer.lease())));
    const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes}}};
    HYLE_RETURN_IF_ERROR(upload.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,upload.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
    HYLE_ASSIGN_OR_RETURN(auto output,submitted.output({0},{1},0,bytes));HYLE_RETURN_IF_ERROR(writer.complete(output,cap.page_size));return output;
  };
  for(unsigned p=0;p<4;++p){HYLE_ASSIGN_OR_RETURN(kr[p],fill(keys,k,p));HYLE_ASSIGN_OR_RETURN(vr[p],fill(values,v,p));}
  const std::array key_reads{kr[3],kr[0],kr[1]},value_reads{vr[1],vr[3],vr[0]};
  kr[2]={};vr[2]={};HYLE_ASSIGN_OR_RETURN(auto unused,keys.acquire(2));
  HYLE_ASSIGN_OR_RETURN(auto native_query_description,recipe.input_description(layout,0));
  HYLE_ASSIGN_OR_RETURN(auto key_description,recipe.input_description(layout,1));
  HYLE_ASSIGN_OR_RETURN(auto value_description,recipe.input_description(layout,2));
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(c::DataType::kFloat32,native_query_description.dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),query.size()*4+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,pool.allocate(query.size()*4,64));HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(query))));HYLE_ASSIGN_OR_RETURN(auto source,qw.publish());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());HYLE_ASSIGN_OR_RETURN(auto q,builder.input(qdesc));
  HYLE_ASSIGN_OR_RETURN(auto key,builder.input(key_description));HYLE_ASSIGN_OR_RETURN(auto value,builder.input(value_description));
  HYLE_ASSIGN_OR_RETURN(auto abs,builder.add(c::Unary{c::UnaryKind::kAbs},q));HYLE_ASSIGN_OR_RETURN(auto converted,builder.add(c::Cast{cap.query_type},abs));
  HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::paged_attention(layout,
      {.scale=recipe.scale(),.causal=recipe.causal(),.window_left=recipe.window_left()}));
  const std::array ports{converted,key,value};HYLE_ASSIGN_OR_RETURN(auto page_node,builder.add_node(c::Operation::custom(descriptor),ports));
  HYLE_ASSIGN_OR_RETURN(auto attended,builder.result(page_node));HYLE_ASSIGN_OR_RETURN(auto widened,builder.add(c::Cast{c::DataType::kFloat32},attended));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.add(c::Unary{c::UnaryKind::kNeg},widened));HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array placements{g::NodePlacement{*graph.node_id(0),kCpuTarget,g::PlacementStrength::kRequired},g::NodePlacement{*graph.node_id(1),kCpuTarget,g::PlacementStrength::kRequired},g::NodePlacement{*graph.node_id(3),kCpuTarget,g::PlacementStrength::kRequired},g::NodePlacement{*graph.node_id(4),kCpuTarget,g::PlacementStrength::kRequired}
      ,g::NodePlacement{page_node,kNativeTarget,g::PlacementStrength::kRequired}
  };
  HYLE_ASSIGN_OR_RETURN(auto native_source,ng::execution_source(native));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(cpu));
  const std::vector<g::StorageDomain> domains{{hyle::mem::system_resource()},{device.ref()}
#if !defined(HYLE_JOINT_PAGE_APPLE)
      ,{staging.ref()}
#endif
  };
  const std::array targets{g::ExecutionTarget{native_source,kDeviceStorage},
      g::ExecutionTarget{cpu_source,kHostStorage}};
  const std::array external{g::ExternalResourceContract{q,kHostStorage},
      g::ExternalResourceContract{key,kDeviceStorage},g::ExternalResourceContract{value,kDeviceStorage},
      g::ExternalResourceContract{output,kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto program,g::compile(graph,{.storage_domains=domains,.targets=targets},
      {.node_placements=placements,.external_resources=external}));
  HYLE_ASSIGN_OR_RETURN(auto active_query_bytes,native_query_description.required_bytes());
  std::size_t cpu_nodes=0,page_nodes=0,page_regions=0,query_copies=0,query_device_copies=0,query_copy_bytes=0;
  for(std::size_t i=0;i<program.report.step_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto step,program.report.step(i));
    if(step.kind==g::CompileStepKind::kCopy){
      HYLE_ASSIGN_OR_RETURN(auto physical,program.report.step_input(i,0));
      HYLE_ASSIGN_OR_RETURN(auto source_value,program.report.physical_value(physical));
      if(source_value.value==converted){
        HYLE_ASSIGN_OR_RETURN(auto destination,program.report.step_output(i,0));
        HYLE_ASSIGN_OR_RETURN(auto destination_value,program.report.physical_value(destination));
        std::fprintf(stderr,"ready Q copy step=%zu storage=%u->%u rows=%lld->%lld bytes=%zu\n",i,
            unsigned(source_value.storage.ordinal()),unsigned(destination_value.storage.ordinal()),
            static_cast<long long>(source_value.description.dimensions()[0]),
            static_cast<long long>(destination_value.description.dimensions()[0]),step.copy_bytes);
        HYLE_RETURN_IF_ERROR(check(source_value.description==native_query_description &&
            destination_value.description==native_query_description && destination_value.value==converted &&
            source_value.storage==step.source_storage && destination_value.storage==step.destination_storage &&
            step.copy_bytes==active_query_bytes,"ready Q copy does not preserve its active descriptor and bytes"));
        ++query_copies;query_copy_bytes+=step.copy_bytes;
        query_device_copies+=destination_value.storage==kDeviceStorage;
      }
      continue;
    }
    if(step.target==kCpuTarget)cpu_nodes+=step.node_count;
    else if(step.target==kNativeTarget){
      page_nodes+=step.node_count;
      page_regions+=step.kind==g::CompileStepKind::kProviderRegion && step.node_count==1;
    }
  }
  HYLE_RETURN_IF_ERROR(check(cpu_nodes==4 && page_nodes==1 && page_regions==1,
      "page math or either real CPU boundary lost its selected route"));
  HYLE_RETURN_IF_ERROR(check(query_copies>0 && query_device_copies==1 &&
      qdesc.dimensions()[0]==std::int64_t(layout.facts().query_rows),"ready CPU/native query boundary still copies capacity rows"));
  std::fprintf(stderr,"ready active Q rows=%u device-upload-bytes=%zu copy-hops=%zu total-copy-bytes=%zu\n",
      unsigned(layout.facts().query_rows),active_query_bytes,query_copies,query_copy_bytes);
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool completion_waiters(hyle::mem::system_allocator(),2);
  Drain drain{scheduler};HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queues,shared_queues(program.report,queue));
  HYLE_ASSIGN_OR_RETURN(auto output_description,graph.value_description(output));
  HYLE_ASSIGN_OR_RETURN(auto output_bytes,output_description.required_bytes());
  std::array<c::LeasePool,2> output_pools;
  for(auto& output_pool:output_pools){
    HYLE_ASSIGN_OR_RETURN(output_pool,c::LeasePool::create(hyle::mem::system_resource(),output_bytes+64,{.max_blocks=1}));
  }
  std::array<g::Invocation,2> slots;
  for(auto& slot:slots){
    HYLE_ASSIGN_OR_RETURN(slot,program.executable.create_invocation({.resources=g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,completion_waiters));
  }
  std::array<c::BatchLayoutVersion,2> uploaded_layouts;
  for(unsigned episode=0;episode<64;++episode){
    const auto& selected=episode%3==2?single:episode%2?next:layout;auto& slot=slots[episode%2];
    HYLE_ASSIGN_OR_RETURN(auto key_frame,keys.read(selected,key_reads));HYLE_ASSIGN_OR_RETURN(auto value_frame,values.read(selected,value_reads));
    const std::array versions{g::batch::NodeLayout{page_node,selected}};
    HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,queues,versions));
    HYLE_RETURN_IF_ERROR(slot.bind(q,qdesc,source));
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,key,key_frame));
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,value,value_frame));
    HYLE_ASSIGN_OR_RETURN(auto output_writer,output_pools[episode%2].allocate(output_bytes,64));
    HYLE_RETURN_IF_ERROR(slot.bind(output,output_description,std::move(output_writer)));
    HYLE_RETURN_IF_ERROR(check(!slot.bind(key,key_description,key_reads[0]).ok(),"ordinary lease bypassed exact page binding"));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto report,g::batch::native_report(slot,page_node));
    const auto previous=uploaded_layouts[episode%2];const auto before=previous.host().page_indices,after=selected.host().page_indices;
    const bool same_table=previous.valid() && before.size()==after.size() && std::equal(before.begin(),before.end(),after.begin());
    HYLE_RETURN_IF_ERROR(check(report.layout_generation==selected.generation() && report.uploaded_bytes<=32 &&
        (!same_table || report.uploaded_bytes==0),"page generation/table delta cost report differs"));
    uploaded_layouts[episode%2]=selected;
    HYLE_ASSIGN_OR_RETURN(auto done,slot.submit(queues));HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto result,slot.result(output));HYLE_ASSIGN_OR_RETURN(auto tensor,result.tensor());
    std::vector<float> actual(18*8*depth);HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
    std::vector<double> expected_values(actual.size());
    const auto host=selected.host();double worst=0;
    for(unsigned r=0;r<3;++r){const int nq=host.query_indptr[r+1]-host.query_indptr[r];if(!nq)continue;
      const int nk=(host.kv_indptr[r+1]-host.kv_indptr[r]-1)*16+host.last_page_lengths[r];std::vector<double> weights(nk);
      for(int row=0;row<nq;++row)for(int h=0;h<8;++h){double sum=0;
        for(int j=0;j<nk;++j){const int physical=host.page_indices[host.kv_indptr[r]+j/16]*16+j%16;double dot=0;
          for(unsigned d=0;d<depth;++d)dot+=std::abs(query[((host.query_indptr[r]+row)*8+h)*depth+d])*decode(k[(physical*2+h/4)*depth+d],bf);
          weights[j]=causal && j>nk-nq+row?0:std::exp(dot*recipe.scale());sum+=weights[j];}
        for(unsigned d=0;d<depth;++d){double expected=0;for(int j=0;j<nk;++j){const int physical=host.page_indices[host.kv_indptr[r]+j/16]*16+j%16;expected-=weights[j]/sum*decode(v[(physical*2+h/4)*depth+d],bf);}
          const auto position=((host.query_indptr[r]+row)*8+h)*depth+d;
          expected_values[position]=expected;
          const auto scalar=actual[position];HYLE_RETURN_IF_ERROR(check(std::isfinite(scalar),"joint output nonfinite"));worst=std::max(worst,std::abs(scalar-expected));}
      }
    }
    HYLE_RETURN_IF_ERROR(check(worst<(bf?.008:.002),"joint independent double differs"));
    HYLE_RETURN_IF_ERROR(check(!g::batch::prepare(slot,queues,versions).ok(),"old result did not guard page preparation"));
    HYLE_RETURN_IF_ERROR(check(!slot.reset().ok(),"old joint result did not guard reset"));
    if(episode==0){
      HYLE_RETURN_IF_ERROR(check(!slot.submit(queues),"old page result allowed a repeated submission"));
      tensor={};result={};done={};
      // Reuse the original selected page recording with the same Q, frames
      // and layout. No bind, batch preparation or replay preparation occurs.
      HYLE_ASSIGN_OR_RETURN(auto replay,slot.submit(queues));HYLE_RETURN_IF_ERROR(replay.wait());
      HYLE_ASSIGN_OR_RETURN(auto replay_result,slot.result(output));
      HYLE_ASSIGN_OR_RETURN(auto replay_tensor,replay_result.tensor());
      std::vector<float> repeated(actual.size());
      HYLE_RETURN_IF_ERROR(replay_tensor.storage().read(std::as_writable_bytes(std::span(repeated))));
      double replay_worst=0;
      for(std::size_t i=0;i<repeated.size();++i){
        HYLE_RETURN_IF_ERROR(check(std::isfinite(repeated[i]),"page strict replay output nonfinite"));
        replay_worst=std::max(replay_worst,std::abs(repeated[i]-expected_values[i]));
      }
      HYLE_RETURN_IF_ERROR(check(replay_worst<(bf?.008:.002),"page strict replay independent double differs"));
    }
  }
  for(auto& slot:slots)HYLE_RETURN_IF_ERROR(slot.close());
  std::fprintf(stderr,"joint page actual CPU/native/CPU PASS D=%u bf16=%d causal=%d episodes=64\n",depth,bf,causal);return Status::Ok();
}
#if !defined(HYLE_JOINT_PAGE_APPLE)
Status empty_episode(c::Context&cpu,n::Context&native,n::MemoryResource&device,n::MemoryResource&staging){
  const c::BatchCapacity cap{.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
      .key_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),
      .value_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),
      .max_requests=1,.max_query_rows=1,.max_kv_rows=16,.max_pages=1,.max_page_refs=1,
      .query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,2> ptr{0,0};const std::array<std::int32_t,1> tails{0};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,{},tails}));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=32,.scale=.125F,.causal=true}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=8192}));
  // No ready page exists: both actual arena pages remain exclusively writable.
  HYLE_ASSIGN_OR_RETURN(auto key_writer,keys.acquire(0));HYLE_ASSIGN_OR_RETURN(auto value_writer,values.acquire(0));
  HYLE_ASSIGN_OR_RETURN(auto key_frame,keys.read(layout,{}));HYLE_ASSIGN_OR_RETURN(auto value_frame,values.read(layout,{}));
  HYLE_RETURN_IF_ERROR(check(key_frame.valid() && value_frame.valid() &&
      key_frame.page_count()==0 && value_frame.page_count()==0,"empty frame acquired a physical page permission"));
  HYLE_ASSIGN_OR_RETURN(auto native_query_description,recipe.input_description(layout,0));
  HYLE_ASSIGN_OR_RETURN(auto key_description,recipe.input_description(layout,1));
  HYLE_ASSIGN_OR_RETURN(auto value_description,recipe.input_description(layout,2));
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(c::DataType::kFloat32,native_query_description.dimensions()));
  HYLE_RETURN_IF_ERROR(check(qdesc.dimensions()[0]==0,"empty graph still exposes a capacity-sized Q producer"));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());HYLE_ASSIGN_OR_RETURN(auto q,builder.input(qdesc));
  HYLE_ASSIGN_OR_RETURN(auto key,builder.input(key_description));HYLE_ASSIGN_OR_RETURN(auto value,builder.input(value_description));
  HYLE_ASSIGN_OR_RETURN(auto positive,builder.add(c::Unary{c::UnaryKind::kAbs},q));
  HYLE_ASSIGN_OR_RETURN(auto converted,builder.add(c::Cast{c::DataType::kFloat16},positive));
  HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::paged_attention(layout,
      {.scale=recipe.scale(),.causal=recipe.causal(),.window_left=recipe.window_left()}));
  const std::array ports{converted,key,value};HYLE_ASSIGN_OR_RETURN(auto page,builder.add_node(c::Operation::custom(descriptor),ports));
  HYLE_ASSIGN_OR_RETURN(auto attended,builder.result(page));HYLE_ASSIGN_OR_RETURN(auto widened,builder.add(c::Cast{c::DataType::kFloat32},attended));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.add(c::Unary{c::UnaryKind::kNeg},widened));
  HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  std::vector<g::NodePlacement> placements;
  for(std::size_t i=0;i<graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));
    placements.push_back({node,node==page?kNativeTarget:kCpuTarget,g::PlacementStrength::kRequired});}
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(cpu));HYLE_ASSIGN_OR_RETURN(auto native_source,g::vulkan::execution_source(native));
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()},g::StorageDomain{device.ref()},g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{native_source,kDeviceStorage},g::ExecutionTarget{cpu_source,kHostStorage}};
  const std::array external{g::ExternalResourceContract{q,kHostStorage},g::ExternalResourceContract{key,kDeviceStorage},
      g::ExternalResourceContract{value,kDeviceStorage},g::ExternalResourceContract{output,kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::compile(graph,{.storage_domains=domains,.targets=targets},
      {.node_placements=placements,.external_resources=external}));
  HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());HYLE_ASSIGN_OR_RETURN(auto queues,shared_queues(compiled.report,queue));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool workers(hyle::mem::system_allocator(),2);Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto slot,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,workers));
  const std::array layouts{g::batch::NodeLayout{page,layout}};
  HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,queues,layouts));
  HYLE_ASSIGN_OR_RETURN(auto bytes,qdesc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto input_pool,c::LeasePool::create(hyle::mem::system_resource(),bytes+64,{.max_blocks=1}));
  HYLE_ASSIGN_OR_RETURN(auto query,input_pool.allocate(std::max<std::size_t>(bytes,1),64));HYLE_RETURN_IF_ERROR(query.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto ready,query.publish());HYLE_RETURN_IF_ERROR(slot.bind(q,qdesc,ready));
  HYLE_RETURN_IF_ERROR(g::batch::bind(slot,key,key_frame));HYLE_RETURN_IF_ERROR(g::batch::bind(slot,value,value_frame));
  HYLE_ASSIGN_OR_RETURN(auto outdesc,graph.value_description(output));HYLE_ASSIGN_OR_RETURN(auto outbytes,outdesc.required_bytes());
  HYLE_RETURN_IF_ERROR(check(outbytes==0,"empty native page produced a nonempty public description"));
  // This real zero-byte caller buffer is borrowed until the original slot closes.
  HYLE_ASSIGN_OR_RETURN(auto buffer,hyle::mem::Buffer::create(hyle::mem::system_resource(),0,64));
  HYLE_ASSIGN_OR_RETURN(auto destination,c::TensorView::create(outdesc,buffer.view()));
  HYLE_RETURN_IF_ERROR(slot.bind(output,destination));HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto report,g::batch::native_report(slot,page));
  HYLE_RETURN_IF_ERROR(check(report.layout_generation==layout.generation() && report.uploaded_bytes==0,
      "empty native page uploaded a table or changed its layout"));
  HYLE_ASSIGN_OR_RETURN(auto done,slot.submit(queues));HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto result,slot.result(output));HYLE_ASSIGN_OR_RETURN(auto tensor,result.tensor());
  HYLE_ASSIGN_OR_RETURN(auto result_bytes,tensor.description().required_bytes());
  HYLE_RETURN_IF_ERROR(check(result_bytes==0 && input_pool.stats().active_uses==0 &&
      key_writer.valid() && value_writer.valid(),"empty native chain read pages or lost real terminal lifetime"));
  HYLE_RETURN_IF_ERROR(check(!g::batch::prepare(slot,queues,layouts).ok() && !slot.reset().ok(),
      "empty GraphOutput did not guard original preparation/reset"));
  tensor={};result={};done={};HYLE_RETURN_IF_ERROR(slot.reset());HYLE_RETURN_IF_ERROR(slot.close());
  // Exercise the original Direct zero-lease importer too. Its real4-byte
  // backing owns a zero-length subview; no query row is fabricated.
  HYLE_ASSIGN_OR_RETURN(auto carrier_description,c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,1>{2}));
  const std::array<std::uint16_t,2> zeros{};
  HYLE_ASSIGN_OR_RETURN(auto carrier,native.upload(carrier_description,std::as_bytes(std::span(zeros))));
  HYLE_ASSIGN_OR_RETURN(auto carrier_read,device.import_readonly(carrier));
  HYLE_ASSIGN_OR_RETURN(auto zero_read,carrier_read.subview(0,0));
  HYLE_ASSIGN_OR_RETURN(auto direct,n::PagedBatchInvocation::create(native,recipe,device,staging,{.max_metadata_backing_bytes=4096}));
  HYLE_RETURN_IF_ERROR(direct.prepare_batch(layout));HYLE_RETURN_IF_ERROR(direct.bind(zero_read,{},{}));
  {
    HYLE_ASSIGN_OR_RETURN(auto direct_done,direct.submit());HYLE_ASSIGN_OR_RETURN(auto direct_result,direct_done.wait());
    HYLE_ASSIGN_OR_RETURN(auto direct_description,c::batch_output_description(cap,0));
    HYLE_RETURN_IF_ERROR(check(direct_result->tensors().size()==1 && direct_result->tensors()[0].description()==direct_description &&
        key_writer.valid() && value_writer.valid(),"Direct zero lease changed the empty output or acquired page permission"));
    HYLE_RETURN_IF_ERROR(check(!direct.prepare_batch(layout).ok(),"Direct empty output did not guard original reuse"));
  }
  HYLE_RETURN_IF_ERROR(direct.close());
  std::fprintf(stderr,"joint page full Q0/refs0 actual CPU/native/CPU PASS: exact empty frames, both page writers held, output0, original result gate\n");
  return Status::Ok();
}
#endif
Status run(int argc,char**argv){
  unsigned selected_depth=0,selected_causal=0;
#if defined(HYLE_JOINT_PAGE_APPLE)
  const bool only_empty=false;
#else
  const bool only_empty=argc==2 && std::string_view(argv[1])=="empty";
#endif
  if(argc!=1 && !only_empty){
    if(argc!=3)return Status::InvalidArgument("expected optional depth causal (0 or 1)");
    const auto depth_text=std::string_view(argv[1]),causal_text=std::string_view(argv[2]);
    const auto depth_parse=std::from_chars(depth_text.data(),depth_text.data()+depth_text.size(),selected_depth);
    const auto causal_parse=std::from_chars(causal_text.data(),causal_text.data()+causal_text.size(),selected_causal);
    if(depth_parse.ec!=std::errc{} || depth_parse.ptr!=depth_text.data()+depth_text.size() ||
       causal_parse.ec!=std::errc{} || causal_parse.ptr!=causal_text.data()+causal_text.size() ||
       (selected_depth!=64 && selected_depth!=128 && selected_depth!=256) || selected_causal>1)
      return Status::InvalidArgument("expected depth 64/128/256 and causal 0/1");
  }
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto native,n::Context::create());
#if defined(HYLE_JOINT_PAGE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native));
  if(selected_depth){
    for(bool bf:{false,true})HYLE_RETURN_IF_ERROR(exercise(cpu,native,device,staging,selected_depth,bf,selected_causal!=0));
    return Status::Ok();
  }
  for(bool bf:{false,true})for(unsigned d:{64U,128U,256U})for(bool causal:{false,true})HYLE_RETURN_IF_ERROR(exercise(cpu,native,device,staging,d,bf,causal));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native,n::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native,n::MemoryKind::kStaging));
  if(only_empty)return empty_episode(cpu,native,device,staging);
  if(selected_depth)return exercise(cpu,native,device,staging,selected_depth,false,selected_causal!=0);
  for(unsigned d:{64U,128U,256U})for(bool causal:{false,true})HYLE_RETURN_IF_ERROR(exercise(cpu,native,device,staging,d,false,causal));
  HYLE_RETURN_IF_ERROR(empty_episode(cpu,native,device,staging));
#endif
  return Status::Ok();
}
} // namespace
int main(int argc,char**argv){const auto status=run(argc,argv);if(!status.ok()){std::fprintf(stderr,"joint page: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
