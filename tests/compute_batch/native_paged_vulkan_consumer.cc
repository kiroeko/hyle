#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/simd/dtype.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace n = c::vulkan;
using hyle::Status;
using hyle::Result;
using Half = hyle::simd::float16_t;
Half encode(float value){return hyle::simd::f32_to_f16(value);}
double decode(Half value){
  const unsigned exponent=(value.bits>>10)&31U, fraction=value.bits&1023U;
  const double sign=(value.bits&0x8000U)?-1.0:1.0;
  if(exponent==31)return fraction?NAN:sign*INFINITY;
  return sign*(exponent?std::ldexp(1.0+double(fraction)/1024.0,int(exponent)-15):
                         std::ldexp(double(fraction),-24));
}
Status require(bool value, const char* text) {
  return value ? Status::Ok() : Status::Internal(text);
}
Result<c::ReadLease> host_read(c::LeasePool& pool,std::span<const Half> values) {
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(values.size_bytes(),64));
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(values)));
  return writer.publish();
}
Status exercise(n::Context& context, n::MemoryResource& device,
                n::MemoryResource& staging, unsigned depth, bool causal) {
  const c::BatchCapacity cap{.storage=c::BatchStorage::kPaged,
      .query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.max_requests=3,.max_query_rows=24,
      .max_kv_rows=128,.max_pages=4,.max_page_refs=8,.query_heads=8,
      .kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,4> qp{0,1,18,18}, next_qp{0,17,18,18}, kp{0,5,7,7};
  const std::array<std::int32_t,7> pages{3,1,3,1,3,1,0};
  const std::array<std::int32_t,3> tails{15,7,0};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qp,kp,pages,tails}));
  HYLE_ASSIGN_OR_RETURN(auto next,schema.version({next_qp,kp,pages,tails}));
  const float scale=1.0F/std::sqrt(float(depth));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=4096,.scale=scale,.causal=causal}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),
      {.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),
      {.max_backing_bytes=1<<20}));
  std::vector<Half> q(24*8*depth), k(64*2*depth), v(k.size());
  for(std::size_t i=0;i<q.size();++i)q[i]=encode(float(int(i%13)-6)*.03125F);
  for(std::size_t i=0;i<k.size();++i){
    k[i]=encode(float(int(i%17)-8)*.015625F);
    v[i]=encode(float(int(i%29)-14)*.03125F);
  }
  for(unsigned row=32;row<48;++row)for(unsigned z=0;z<2*depth;++z){
    k[row*2*depth+z]=encode(64.0F);v[row*2*depth+z]=encode(128.0F);
  }
  const auto descriptions=recipe.descriptions();
  HYLE_ASSIGN_OR_RETURN(auto qt,context.upload(descriptions[0],std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query,device.import_readonly(qt));
  const auto page_bytes=keys.page_bytes();
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),
      2*page_bytes+512,{.max_blocks=2}));
  const std::array specs{c::CopyResourceSpec{staging.ref(),page_bytes},
                        c::CopyResourceSpec{device.ref(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy_slot,n::CopyBatchInvocation::create(copy));
  const std::array<c::CopyPhase,1> phases{{{}}};
  const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,page_bytes}}};
  std::array<c::ReadLease,4> kr,vr;
  const auto fill=[&](c::PagedArena& arena,std::span<const Half> data,
                      unsigned page)->Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));
    HYLE_RETURN_IF_ERROR(copy_slot.reset());
    HYLE_ASSIGN_OR_RETURN(auto source,host_read(host_pool,data.subspan(page*16*2*depth,16*2*depth)));
    HYLE_RETURN_IF_ERROR(copy_slot.bind_source({0},source));
    HYLE_RETURN_IF_ERROR(copy_slot.bind_mutable({1},std::move(writer.lease())));
    HYLE_RETURN_IF_ERROR(copy_slot.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto submitted,copy_slot.submit());
    HYLE_RETURN_IF_ERROR(submitted.wait());
    HYLE_ASSIGN_OR_RETURN(auto ready,submitted.output({0},{1},0,page_bytes));
    HYLE_RETURN_IF_ERROR(writer.complete(ready,cap.page_size));return ready;
  };
  for(unsigned p=0;p<4;++p){HYLE_ASSIGN_OR_RETURN(kr[p],fill(keys,k,p));HYLE_ASSIGN_OR_RETURN(vr[p],fill(values,v,p));}
  const std::array exact_k{kr[3],kr[0],kr[1]},exact_v{vr[1],vr[3],vr[0]};
  // Unreferenced sentinel page is immediately mutable; no whole-arena guard.
  kr[2]={};vr[2]={};HYLE_ASSIGN_OR_RETURN(auto sentinel,keys.acquire(2));
  HYLE_ASSIGN_OR_RETURN(auto key_frame,keys.read(layout,exact_k));
  HYLE_ASSIGN_OR_RETURN(auto value_frame,values.read(layout,exact_v));
  HYLE_ASSIGN_OR_RETURN(auto next_keys,keys.read(next,exact_k));
  HYLE_ASSIGN_OR_RETURN(auto next_values,values.read(next,exact_v));
  HYLE_ASSIGN_OR_RETURN(auto foreign_schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto foreign,foreign_schema.version({qp,kp,pages,tails}));
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  const c::NativePagedInvocationOptions options{
      .max_metadata_backing_bytes=4096,.allocator=metadata.get()};
  HYLE_ASSIGN_OR_RETURN(auto slot,n::PagedBatchInvocation::create(context,recipe,device,staging,options));
  HYLE_ASSIGN_OR_RETURN(auto other,n::PagedBatchInvocation::create(context,recipe,device,staging,options));
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(foreign).ok(),"foreign schema accepted"));
  const auto oracle=[&](auto queries){
    std::vector<double> output(18*8*depth), weights;
    for(unsigned request=0;request<2;++request){
      const unsigned nq=queries[request+1]-queries[request];
      const unsigned nk=(kp[request+1]-kp[request]-1)*16+tails[request];
      weights.resize(nk);
      const auto physical=[&](unsigned j){return unsigned(pages[kp[request]+j/16])*16+j%16;};
      for(unsigned row=0;row<nq;++row)for(unsigned head=0;head<8;++head){
        double sum=0;
        for(unsigned j=0;j<nk;++j){
          double dot=0;for(unsigned d=0;d<depth;++d)
            dot+=decode(q[((queries[request]+row)*8+head)*depth+d])*
                 decode(k[(physical(j)*2+head/4)*depth+d]);
          weights[j]=causal && j>nk-nq+row?0.0:std::exp(dot*scale);sum+=weights[j];
        }
        for(unsigned d=0;d<depth;++d)for(unsigned j=0;j<nk;++j)
          output[((queries[request]+row)*8+head)*depth+d]+=weights[j]/sum*
              decode(v[(physical(j)*2+head/4)*depth+d]);
      }
    }return output;
  };
  const auto expected=oracle(qp), next_expected=oracle(next_qp);
  const auto verify=[&](const n::Results& result,const auto& expected)->Status{
    std::vector<Half> actual(expected.size());
    HYLE_RETURN_IF_ERROR(context.download(result.tensors()[0],std::as_writable_bytes(std::span(actual))));
    double worst=0;
    for(std::size_t i=0;i<actual.size();++i){
      HYLE_RETURN_IF_ERROR(require(std::isfinite(decode(actual[i])),"native page output nonfinite"));
      worst=std::max(worst,std::abs(decode(actual[i])-expected[i]));
    }
    return require(worst<=.002,"native page independent double oracle differs");
  };
  HYLE_RETURN_IF_ERROR(other.prepare_batch(layout));
  HYLE_RETURN_IF_ERROR(other.bind(query,key_frame,value_frame));
  HYLE_ASSIGN_OR_RETURN(auto submitted,other.submit());
  HYLE_ASSIGN_OR_RETURN(auto old,submitted.wait());
  HYLE_RETURN_IF_ERROR(verify(*old,expected));
  for(unsigned episode=0;episode<64;++episode){
    const bool changed=episode%2;
    HYLE_RETURN_IF_ERROR(slot.prepare_batch(changed?next:layout));
    if(changed)HYLE_RETURN_IF_ERROR(require(!slot.bind(query,key_frame,next_values).ok(),"mixed generations accepted"));
    HYLE_RETURN_IF_ERROR(slot.bind(query,changed?next_keys:key_frame,changed?next_values:value_frame));
    HYLE_ASSIGN_OR_RETURN(auto run,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,run.wait());
    HYLE_RETURN_IF_ERROR(verify(*result,changed?next_expected:expected));
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"old output permitted native reuse"));
    HYLE_RETURN_IF_ERROR(verify(*old,expected));
    const auto report=slot.report();
    HYLE_RETURN_IF_ERROR(require(report.uploaded_bytes==32 && report.device_banks==2 &&
        report.staging_bytes==32 && report.device_mirror_bytes==64,"native table costs expanded"));
  }
  HYLE_RETURN_IF_ERROR(slot.close());HYLE_RETURN_IF_ERROR(slot.drain());
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"closed slot accepted work"));
  other={};HYLE_RETURN_IF_ERROR(verify(*old,expected));old={};submitted={};slot={};
  HYLE_RETURN_IF_ERROR(require(metadata.used()==0,"native page controller metadata leaked"));
  std::printf("Vulkan page16 F16 D%u causal=%u: double oracle, 64 topology episodes, two slots, logical79>physical64, exact guards/costs PASS\n",depth,causal);
  return Status::Ok();
}
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device,"GPU consumer received software Vulkan"));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
  for(unsigned depth:{64U,128U,256U})for(bool causal:{false,true})
    HYLE_RETURN_IF_ERROR(exercise(context,device,staging,depth,causal));
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(!status.ok()){std::fprintf(stderr,"FAIL: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
