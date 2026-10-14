#if defined(HYLE_DIRTY_PAGE_APPLE)
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#else
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>
#if defined(HYLE_DIRTY_PAGE_NATIVE_PRODUCER)
#include <mlx/ops.h>
#include <mlx/transforms.h>
#endif

namespace {
namespace c=hyle::compute;
#if defined(HYLE_DIRTY_PAGE_APPLE)
namespace n=c::apple;
#else
namespace n=c::vulkan;
#endif
using hyle::Status;
using hyle::Result;
Status require(bool v,const char* message){return v?Status::Ok():Status::Internal(message);}
double decode(std::uint16_t x,bool bf=false){
  if(bf)return std::bit_cast<float>(std::uint32_t(x)<<16);
  const unsigned e=(x>>10)&31U,f=x&1023U;
  return (x&0x8000U?-1.:1.)*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));
}
Status exercise(n::Context& context,n::MemoryResource& device,n::MemoryResource& staging,bool bf=false){
  constexpr unsigned depth=256,heads=8,kv_heads=2;
  constexpr std::size_t row_elements=kv_heads*depth,row_bytes=row_elements*2;
  const c::BatchCapacity cap{.data_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,
      .max_requests=2,.max_query_rows=18,.max_kv_rows=160,
      .max_pages=10,.max_page_refs=8,.query_heads=heads,.kv_heads=kv_heads,.head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=32,.scale=1.F/16.F,.causal=true}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<20}));
  const auto page_bytes=keys.page_bytes();
  constexpr std::size_t query_bytes=18*heads*depth*2;
  const auto init_bytes=std::max(page_bytes,query_bytes);
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),2*init_bytes+512,{.max_blocks=2}));
  const std::array specs{c::CopyResourceSpec{staging.ref(),init_bytes},c::CopyResourceSpec{device.ref(),init_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto initialize,n::CopyBatchInvocation::create(init_program));
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
    HYLE_ASSIGN_OR_RETURN(auto k,keys.acquire(p));HYLE_ASSIGN_OR_RETURN(kr[p],fill(std::move(k.lease()),zeros));HYLE_RETURN_IF_ERROR(k.complete(kr[p],cap.page_size));
    HYLE_ASSIGN_OR_RETURN(auto v,values.acquire(p));HYLE_ASSIGN_OR_RETURN(vr[p],fill(std::move(v.lease()),zeros));HYLE_RETURN_IF_ERROR(v.complete(vr[p],cap.page_size));
  }
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(device.ref(),q.size()*2+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,query_pool.allocate(q.size()*2,64));
  HYLE_ASSIGN_OR_RETURN(auto query,fill(std::move(qw),q));HYLE_RETURN_IF_ERROR(initialize.reset());
#if defined(HYLE_DIRTY_PAGE_NATIVE_PRODUCER)
  namespace mx=mlx::core;
  const auto native_type=bf?mx::bfloat16:mx::float16;
  const std::array producer_specs{n::InputSpec{{4,2,256},native_type}};
  HYLE_ASSIGN_OR_RETURN(auto producer,context.prepare(c::Unary{c::UnaryKind::kAbs},producer_specs));
  HYLE_ASSIGN_OR_RETURN(auto producer_slot,producer.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto producer_desc,c::TensorDesc::dense(cap.data_type,
      std::array<std::int64_t,3>{4,2,256}));
  const auto source_resource=device.ref();
#else
  const auto source_resource=staging.ref();
#endif
  const std::array update_specs{c::CopyResourceSpec{source_resource,4*row_bytes},
      c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes},
      c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto update_program,n::CopyBatchProgram::prepare(context,update_specs));
  HYLE_ASSIGN_OR_RETURN(auto update,n::CopyBatchInvocation::create(update_program));
  HYLE_ASSIGN_OR_RETURN(auto slot,n::PagedBatchInvocation::create(context,recipe,device,staging,{.max_metadata_backing_bytes=4096}));
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
    HYLE_RETURN_IF_ERROR(slot.prepare_batch(layout));
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
        ranges.push_back({{0},{port},{0},(port-1)*row_bytes,(row%16)*row_bytes,row_bytes});
      }
    }
#if defined(HYLE_DIRTY_PAGE_NATIVE_PRODUCER)
    // The finite producer has fixed ports; unused rows are not copied to pages.
    std::vector<std::uint16_t> negative(4*row_elements,0x8000);
    for(std::size_t i=0;i<incoming.size();++i)negative[i]=incoming[i]|0x8000U;
    auto input=mx::view(mx::array(negative.data(),mx::Shape{4,2,256},mx::uint16),native_type);
    mx::async_eval({input});mx::synchronize(mx::default_stream(mx::Device::gpu));input.wait();
    const std::array producer_inputs{input};HYLE_RETURN_IF_ERROR(producer_slot.bind(producer_inputs));
    c::ReadLease produced;
    {HYLE_ASSIGN_OR_RETURN(auto pending,producer_slot.submit());
      HYLE_ASSIGN_OR_RETURN(auto result,pending.wait());
      HYLE_ASSIGN_OR_RETURN(produced,device.import_readonly(result->arrays()[0],producer_desc));}
    HYLE_RETURN_IF_ERROR(update.bind_source({0},produced));
#else
    HYLE_ASSIGN_OR_RETURN(auto hw,host_pool.allocate(incoming.size()*2,64));
    HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(std::span(incoming))));
    HYLE_ASSIGN_OR_RETURN(auto host,hw.publish());HYLE_RETURN_IF_ERROR(update.bind_source({0},host));
#endif
    HYLE_RETURN_IF_ERROR(update.prepare_replay(phases,ranges));
    {HYLE_ASSIGN_OR_RETURN(auto submitted,update.submit());
      HYLE_RETURN_IF_ERROR(require(!update.unbind_mutable({1}),"native retained submission permitted writer withdrawal"));
      HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_RETURN_IF_ERROR(require(submitted.report().logical_bytes==2*updates*row_bytes,"native update copied cache history"));
      dirty_total+=submitted.report().logical_bytes;
    }
    for(unsigned request=0;request<updates;++request){
      const auto page=destinations[request];
      HYLE_ASSIGN_OR_RETURN(writers[2*request].lease(),update.unbind_mutable({2*request+1}));
      HYLE_ASSIGN_OR_RETURN(writers[2*request+1].lease(),update.unbind_mutable({2*request+2}));
      HYLE_ASSIGN_OR_RETURN(kr[page],writers[2*request].publish(cap.page_size));
      HYLE_ASSIGN_OR_RETURN(vr[page],writers[2*request+1].publish(cap.page_size));
      HYLE_RETURN_IF_ERROR(require(kr[page].generation()==generations[2*request] && vr[page].generation()==generations[2*request+1],"native dirty update changed original generation"));
    }
    std::vector<c::ReadLease> key_reads,value_reads;
    for(auto p:pages){key_reads.push_back(kr[p]);value_reads.push_back(vr[p]);}
    HYLE_ASSIGN_OR_RETURN(auto kframe,keys.read(layout,key_reads));
    HYLE_ASSIGN_OR_RETURN(auto vframe,values.read(layout,value_reads));
    HYLE_RETURN_IF_ERROR(slot.bind(query,kframe,vframe));
    HYLE_RETURN_IF_ERROR(require(!keys.acquire(kr[destinations[0]]),"bound native frame permitted dirty page mutation"));
    {HYLE_ASSIGN_OR_RETURN(auto free,keys.acquire(kr[9]));HYLE_ASSIGN_OR_RETURN(kr[9],free.publish(cap.page_size));}
    {HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());
      const auto active=layout.facts().query_rows;
      std::vector<std::uint16_t> actual(active*heads*depth);
#if defined(HYLE_DIRTY_PAGE_APPLE)
      HYLE_ASSIGN_OR_RETURN(auto output_desc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{active,heads,depth}));
      HYLE_ASSIGN_OR_RETURN(auto output,device.import_readonly(result->arrays()[0],output_desc));
      HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
#else
      HYLE_RETURN_IF_ERROR(context.download(result->tensors()[0],std::as_writable_bytes(std::span(actual))));
#endif
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
  }
  HYLE_RETURN_IF_ERROR(update.close());HYLE_RETURN_IF_ERROR(slot.drain());
#if defined(HYLE_DIRTY_PAGE_NATIVE_PRODUCER)
  std::fprintf(stderr,"actual native Abs producer -> readonly -> fixed F2 -> original dirty page -> attention PASS; producer program/slot reused64, no host K/V result download, explicit producer-ready waits remain\n");
#endif
  std::fprintf(stderr,"native dirty page PASS:64 steps GQA8/2 D256, two requests growth17..80/17..48, mixed Q1/17 prefill/decode, crosspage, exact frame, untouched history, same generation, BF16=%d dirty total=%llu bytes, original prepare releases imports; explicit host waits retained\n",bf,static_cast<unsigned long long>(dirty_total));
  return Status::Ok();
}
Status run(){HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_DIRTY_PAGE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device,"dirty page requires actual Vulkan GPU"));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  struct Close{n::Context& context;~Close(){context.close().IgnoreError();}}close{context};
  HYLE_RETURN_IF_ERROR(exercise(context,device,staging));
#if defined(HYLE_DIRTY_PAGE_APPLE)
  HYLE_RETURN_IF_ERROR(exercise(context,device,staging,true));
#endif
  return Status::Ok();
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"native dirty page: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
