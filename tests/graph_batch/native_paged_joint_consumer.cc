#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/sched/scheduler.h>
#if defined(HYLE_JOINT_PAGE_APPLE)
#include <hyle/graph/cpu_apple.h>
#else
#include <hyle/graph/cpu_vulkan.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
#if defined(HYLE_JOINT_PAGE_APPLE)
namespace n=c::apple;namespace joint=g::cpu_apple;
#else
namespace n=c::vulkan;namespace joint=g::cpu_vulkan;
#endif
using hyle::Status;using hyle::Result;
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
  const c::BatchCapacity cap{.data_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,
      .max_requests=3,.max_query_rows=24,.max_kv_rows=128,.max_pages=4,.max_page_refs=8,
      .query_heads=8,.kv_heads=2,.head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,4> qp{0,1,18,18},next_qp{0,17,18,18},single_qp{0,18,18,18},kp{0,5,7,7};
  const std::array<std::int32_t,7> ids{3,1,3,1,3,1,0};const std::array<std::int32_t,3> tails{15,7,0};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto next,schema.version({next_qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto single,schema.version({single_qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=32,.scale=1.F/std::sqrt(float(depth)),.causal=causal}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<20}));
  std::vector<float> query(24*8*depth);
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
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(c::DataType::kFloat32,recipe.descriptions()[0].dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),query.size()*4+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,pool.allocate(query.size()*4,64));HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(query))));HYLE_ASSIGN_OR_RETURN(auto source,qw.publish());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());HYLE_ASSIGN_OR_RETURN(auto q,builder.input(qdesc));
  HYLE_ASSIGN_OR_RETURN(auto key,builder.input(recipe.descriptions()[1]));HYLE_ASSIGN_OR_RETURN(auto value,builder.input(recipe.descriptions()[2]));
  HYLE_ASSIGN_OR_RETURN(auto abs,builder.add(c::Unary{c::UnaryKind::kAbs},q));HYLE_ASSIGN_OR_RETURN(auto converted,builder.add(c::Cast{cap.data_type},abs));
  HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::native_paged_attention(recipe,layout));
  const std::array ports{converted,key,value};HYLE_ASSIGN_OR_RETURN(auto page_node,builder.add_node(c::Operation::custom(descriptor),ports));
  HYLE_ASSIGN_OR_RETURN(auto attended,builder.result(page_node));HYLE_ASSIGN_OR_RETURN(auto widened,builder.add(c::Cast{c::DataType::kFloat32},attended));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.add(c::Unary{c::UnaryKind::kNeg},widened));HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array placements{g::NodePlacement{*graph.node_id(0),joint::kCpuTarget,g::PlacementStrength::kRequired},g::NodePlacement{*graph.node_id(1),joint::kCpuTarget,g::PlacementStrength::kRequired},g::NodePlacement{*graph.node_id(3),joint::kCpuTarget,g::PlacementStrength::kRequired},g::NodePlacement{*graph.node_id(4),joint::kCpuTarget,g::PlacementStrength::kRequired}};
  const std::array bindings{g::batch::NativePagedNodeBinding{page_node,recipe,layout,{.max_metadata_backing_bytes=4096}}};
#if defined(HYLE_JOINT_PAGE_APPLE)
  const joint::Environment environment{.cpu=&cpu,.apple=&native,.native=&device};
#else
  const joint::Environment environment{.cpu=&cpu,.vulkan=&native,.device=&device,.staging=&staging};
#endif
  joint::CompileReport compiled;HYLE_ASSIGN_OR_RETURN(auto program,joint::compile(graph,environment,{.node_placements=placements,.paged_nodes=bindings},&compiled));
  HYLE_RETURN_IF_ERROR(check(compiled.fragments.size()==3,"joint page was not isolated between real CPU fragments"));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);Drain drain{scheduler};HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());
  std::array<joint::Invocation,2> slots;
  for(auto& slot:slots){HYLE_ASSIGN_OR_RETURN(slot,program.create_invocation(scheduler,queue));}
  for(unsigned episode=0;episode<64;++episode){
    const auto& selected=episode%3==2?single:episode%2?next:layout;auto& slot=slots[episode%2];
    HYLE_ASSIGN_OR_RETURN(auto key_frame,keys.read(selected,key_reads));HYLE_ASSIGN_OR_RETURN(auto value_frame,values.read(selected,value_reads));
    const std::array versions{g::batch::NodeLayout{page_node,selected}};HYLE_RETURN_IF_ERROR(slot.prepare_batch(versions));
    HYLE_RETURN_IF_ERROR(slot.bind(q,qdesc,source));HYLE_RETURN_IF_ERROR(slot.bind(key,key_frame));HYLE_RETURN_IF_ERROR(slot.bind(value,value_frame));
    HYLE_RETURN_IF_ERROR(check(!slot.bind(key,recipe.descriptions()[1],key_reads[0]).ok(),"ordinary lease bypassed exact page binding"));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay());HYLE_ASSIGN_OR_RETURN(auto report,slot.replay_report());
    HYLE_RETURN_IF_ERROR(check(report.cpu_fragments==2 && report.paged_fragments==1 && report.paged_metadata_uploaded_bytes==32,"joint page math/table cost report differs"));
    HYLE_ASSIGN_OR_RETURN(auto result,slot.run());HYLE_ASSIGN_OR_RETURN(auto tensor,result->output(output));
    std::vector<float> actual(18*8*depth);HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
    const auto host=selected.host();double worst=0;
    for(unsigned r=0;r<3;++r){const int nq=host.query_indptr[r+1]-host.query_indptr[r];if(!nq)continue;
      const int nk=(host.kv_indptr[r+1]-host.kv_indptr[r]-1)*16+host.last_page_lengths[r];std::vector<double> weights(nk);
      for(int row=0;row<nq;++row)for(int h=0;h<8;++h){double sum=0;
        for(int j=0;j<nk;++j){const int physical=host.page_indices[host.kv_indptr[r]+j/16]*16+j%16;double dot=0;
          for(unsigned d=0;d<depth;++d)dot+=std::abs(query[((host.query_indptr[r]+row)*8+h)*depth+d])*decode(k[(physical*2+h/4)*depth+d],bf);
          weights[j]=causal && j>nk-nq+row?0:std::exp(dot*recipe.scale());sum+=weights[j];}
        for(unsigned d=0;d<depth;++d){double expected=0;for(int j=0;j<nk;++j){const int physical=host.page_indices[host.kv_indptr[r]+j/16]*16+j%16;expected-=weights[j]/sum*decode(v[(physical*2+h/4)*depth+d],bf);}
          const auto scalar=actual[((host.query_indptr[r]+row)*8+h)*depth+d];HYLE_RETURN_IF_ERROR(check(std::isfinite(scalar),"joint output nonfinite"));worst=std::max(worst,std::abs(scalar-expected));}
      }
    }
    HYLE_RETURN_IF_ERROR(check(worst<(bf?.008:.002),"joint independent double differs"));
    HYLE_RETURN_IF_ERROR(check(!slot.prepare_batch(versions).ok(),"old joint result did not guard page preparation"));
    HYLE_RETURN_IF_ERROR(check(!slot.reset().ok(),"old joint result did not guard reset"));
  }
  for(auto& slot:slots)HYLE_RETURN_IF_ERROR(slot.close());
  std::fprintf(stderr,"joint page actual CPU/native/CPU PASS D=%u bf16=%d causal=%d episodes=64\n",depth,bf,causal);return Status::Ok();
}
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto native,n::Context::create());
#if defined(HYLE_JOINT_PAGE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native));
  for(bool bf:{false,true})for(unsigned d:{64U,128U,256U})for(bool causal:{false,true})HYLE_RETURN_IF_ERROR(exercise(cpu,native,device,staging,d,bf,causal));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native,n::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native,n::MemoryKind::kStaging));
  for(unsigned d:{64U,128U,256U})for(bool causal:{false,true})HYLE_RETURN_IF_ERROR(exercise(cpu,native,device,staging,d,false,causal));
#endif
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(!status.ok()){std::fprintf(stderr,"joint page: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
