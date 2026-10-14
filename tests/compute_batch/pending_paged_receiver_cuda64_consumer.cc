#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
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
Status exercise(c::Context&context,c::Queue&queue,hyle::mem::MemoryResourceRef resource,
    hyle::mem::MemoryResourceRef host_resource,bool bf=false){
  constexpr unsigned depth=256,heads=8,kv_heads=2;
  constexpr std::size_t row_elements=kv_heads*depth,row_bytes=row_elements*2;
  const c::BatchCapacity cap{.data_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,
      .max_requests=2,.max_query_rows=18,.max_kv_rows=160,
      .max_pages=10,.max_page_refs=8,.query_heads=heads,.kv_heads=kv_heads,.head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,resource,{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,resource,{.max_backing_bytes=1<<20}));
  const auto page_bytes=keys.page_bytes();
  constexpr std::size_t query_bytes=18*heads*depth*2;
  const auto init_bytes=std::max(page_bytes,query_bytes);
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(host_resource,2*init_bytes+512,{.max_blocks=2}));
  const std::array specs{c::CopyResourceSpec{host_resource,init_bytes},c::CopyResourceSpec{resource,init_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto initialize,n::CopyBatchInvocation::create(init_program,queue));
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
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(resource,q.size()*2+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,query_pool.allocate(q.size()*2,64));
  HYLE_ASSIGN_OR_RETURN(auto query,fill(std::move(qw),q));HYLE_RETURN_IF_ERROR(initialize.reset());
  const auto source_resource=host_resource;
  const std::array update_specs{c::CopyResourceSpec{source_resource,4*row_bytes},
      c::CopyResourceSpec{resource,page_bytes},c::CopyResourceSpec{resource,page_bytes},
      c::CopyResourceSpec{resource,page_bytes},c::CopyResourceSpec{resource,page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto update_program,n::CopyBatchProgram::prepare(context,update_specs));
  std::array<n::CopyBatchInvocation,2> copy_slots;
  std::array<n::PagedBatchReceiver,2> slots;
  for(unsigned i=0;i<2;++i){
    HYLE_ASSIGN_OR_RETURN(copy_slots[i],n::CopyBatchInvocation::create(update_program,queue));
    HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,
        {.resource=resource,.host_workspace=host_resource,.max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=64u<<10,
         .max_invocations=1,.scale=1.F/16.F,.causal=true}));
    HYLE_ASSIGN_OR_RETURN(auto native,c::BatchInvocation::create(program,queue));
    HYLE_ASSIGN_OR_RETURN(slots[i],n::PagedBatchReceiver::create(std::move(native),2));
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
    for(auto&receiver:slots)HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
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
    HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(query,kframe,vframe,std::move(owner)));
    HYLE_RETURN_IF_ERROR(require(!owner.valid() && !slot.take_update(),"pending source was not retained by original suffix"));
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"retained aggregate allowed receiver prepare"));
    {HYLE_ASSIGN_OR_RETURN(auto free,keys.acquire(kr[9]));HYLE_ASSIGN_OR_RETURN(kr[9],free.publish(0));}
    HYLE_RETURN_IF_ERROR(require(!slot.output(),"unobserved common aggregate published output"));
    HYLE_RETURN_IF_ERROR(submitted.wait());
    {HYLE_ASSIGN_OR_RETURN(auto output,slot.output());
      const auto active=layout.facts().query_rows;
      std::vector<std::uint16_t> actual(active*heads*depth);
      HYLE_ASSIGN_OR_RETURN(auto active_output,output.view()->subview(0,actual.size()*sizeof(std::uint16_t)));
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
          const double expected=weighted/sum;
          for(unsigned i=0;i<heads*depth;++i){const auto actual_value=decode(actual[(ptr[request]+query_row)*heads*depth+i],bf);
            HYLE_RETURN_IF_ERROR(require(std::isfinite(actual_value) && std::abs(actual_value-expected)<(bf?.008:.002),"native dirty attention independent oracle differs"));
          }
        }
      }
      HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"retained native result permitted prepare/update"));
    }
    // The mapped completion is the public/native result path; source need not
    // be separately waited. Recovery uses the two original update controls.
    HYLE_RETURN_IF_ERROR(submitted.wait());
    submitted={};
    HYLE_ASSIGN_OR_RETURN(owner,slot.take_update());
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
  std::fprintf(stderr,"common CUDA pending receiver PASS:64 steps GQA8/2 D256, two requests growth17..80/17..48, mixed Q1/17 prefill/decode, crosspage, exact frame, untouched history, same generation, BF16=%d dirty total=%llu bytes, original prepare releases imports; cold metadata waits retained, no source wait before attention\n",bf,static_cast<unsigned long long>(dirty_total));
  return Status::Ok();
}
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto id,c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(id,c::flashinfer::provider()));
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Close{c::Context&context;~Close(){context.drain().IgnoreError();}}close{context};
  HYLE_RETURN_IF_ERROR(exercise(context,queue,device.ref(),host.ref()));return exercise(context,queue,device.ref(),host.ref(),true);
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"common receiver: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
