#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/simd/dtype.h>

#include <algorithm>
#include <bit>
#include <type_traits>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace n = c::apple;
using hyle::Status;
using hyle::Result;
using Half = hyle::simd::float16_t;
using BFloat = hyle::simd::bfloat16_t;
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
template<class T> Result<c::ReadLease> host_read(c::LeasePool& pool,std::span<const T> values) {
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(values.size_bytes(),64));
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(values)));
  return writer.publish();
}
template<class T> T encode_value(float value){if constexpr(std::is_same_v<T,Half>)return encode(value);else return hyle::simd::f32_to_bf16(value);}
template<class T> double decode_value(T value){if constexpr(std::is_same_v<T,Half>)return decode(value);else return std::bit_cast<float>(std::uint32_t(value.bits)<<16);}
template<class T> Status exercise(n::Context& context, n::MemoryResource& device,
                n::MemoryResource& staging, unsigned depth, bool causal,
                unsigned first_refs=5,unsigned query_heads=8) {
  const unsigned table_capacity=(first_refs+5)&~3U;
  const c::BatchCapacity cap{.storage=c::BatchStorage::kPaged,
      .query_type=std::is_same_v<T,Half>?c::DataType::kFloat16:c::DataType::kBFloat16,.output_type=std::is_same_v<T,Half>?c::DataType::kFloat16:c::DataType::kBFloat16,.key_storage=c::CacheStorage::plain(std::is_same_v<T,Half>?c::DataType::kFloat16:c::DataType::kBFloat16).value(),.value_storage=c::CacheStorage::plain(std::is_same_v<T,Half>?c::DataType::kFloat16:c::DataType::kBFloat16).value(),.max_requests=3,.max_query_rows=24,
      .max_kv_rows=16*first_refs+32,.max_pages=4,.max_page_refs=table_capacity,.query_heads=query_heads,
      .kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,4> qp{0,1,18,18}, next_qp{0,17,18,18},
      kp{0,int(first_refs),int(first_refs+2),int(first_refs+2)};
  std::vector<std::int32_t> pages(first_refs+2);
  for(unsigned i=0;i<first_refs;++i)pages[i]=i%2?1:3;
  pages[first_refs]=1;pages[first_refs+1]=0;
  const std::array<std::int32_t,3> tails{15,7,0};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qp,kp,pages,tails}));
  HYLE_ASSIGN_OR_RETURN(auto next,schema.version({next_qp,kp,pages,tails}));
  const float scale=1.0F/std::sqrt(float(depth));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=8192,.scale=scale,.causal=causal}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),
      {.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),
      {.max_backing_bytes=1<<20}));
  std::vector<T> q(24*query_heads*depth), k(64*2*depth), v(k.size());
  for(std::size_t i=0;i<q.size();++i)q[i]=encode_value<T>(float(int(i%13)-6)*.03125F);
  for(std::size_t i=0;i<k.size();++i){
    k[i]=encode_value<T>(float(int(i%17)-8)*.015625F);
    v[i]=encode_value<T>(float(int(i%29)-14)*.03125F);
  }
  for(unsigned row=32;row<48;++row)for(unsigned z=0;z<2*depth;++z){
    k[row*2*depth+z]=encode_value<T>(64.0F);v[row*2*depth+z]=encode_value<T>(128.0F);
  }
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(device.ref(),q.size()*sizeof(T)+512));
  HYLE_ASSIGN_OR_RETURN(auto query_writer,query_pool.allocate(q.size()*sizeof(T),64));
  HYLE_RETURN_IF_ERROR(query_writer.view()->write(std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query,query_writer.publish());
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
  const auto fill=[&](c::PagedArena& arena,std::span<const T> data,
                      unsigned page)->Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));
    HYLE_RETURN_IF_ERROR(copy_slot.reset());
    HYLE_ASSIGN_OR_RETURN(auto source,host_read<T>(host_pool,data.subspan(page*16*2*depth,16*2*depth)));
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
      .max_metadata_backing_bytes=16384,.allocator=metadata.get()};
  HYLE_ASSIGN_OR_RETURN(auto slot,n::PagedBatchInvocation::create(context,recipe,device,staging,options));
  HYLE_ASSIGN_OR_RETURN(auto other,n::PagedBatchInvocation::create(context,recipe,device,staging,options));
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(foreign).ok(),"foreign schema accepted"));
  const auto oracle=[&](auto queries){
    std::vector<double> output(18*query_heads*depth), weights;
    for(unsigned request=0;request<2;++request){
      const unsigned nq=queries[request+1]-queries[request];
      const unsigned nk=(kp[request+1]-kp[request]-1)*16+tails[request];
      weights.resize(nk);
      const auto physical=[&](unsigned j){return unsigned(pages[kp[request]+j/16])*16+j%16;};
      for(unsigned row=0;row<nq;++row)for(unsigned head=0;head<query_heads;++head){
        double sum=0;
        for(unsigned j=0;j<nk;++j){
          double dot=0;for(unsigned d=0;d<depth;++d)
            dot+=decode_value(q[((queries[request]+row)*query_heads+head)*depth+d])*
                 decode_value(k[(physical(j)*2+head/(query_heads/2))*depth+d]);
          weights[j]=causal && j>nk-nq+row?0.0:std::exp(dot*scale);sum+=weights[j];
        }
        for(unsigned d=0;d<depth;++d)for(unsigned j=0;j<nk;++j)
          output[((queries[request]+row)*query_heads+head)*depth+d]+=weights[j]/sum*
              decode_value(v[(physical(j)*2+head/(query_heads/2))*depth+d]);
      }
    }return output;
  };
  const auto expected=oracle(qp), next_expected=oracle(next_qp);
  const auto verify=[&](const n::Results& result,const auto& expected)->Status{
    std::vector<T> actual(expected.size());
    HYLE_ASSIGN_OR_RETURN(auto output_description,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{18,query_heads,depth}));
    HYLE_ASSIGN_OR_RETURN(auto output,device.import_readonly(result.arrays()[0],output_description));
    HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
    double worst=0;
    for(std::size_t i=0;i<actual.size();++i){
      HYLE_RETURN_IF_ERROR(require(std::isfinite(decode_value(actual[i])),"native page output nonfinite"));
      worst=std::max(worst,std::abs(decode_value(actual[i])-expected[i]));
    }
    return require(worst<=(std::is_same_v<T,Half>?.002:.008),"native page independent double oracle differs");
  };
  HYLE_RETURN_IF_ERROR(other.prepare_batch(layout));
  HYLE_RETURN_IF_ERROR(other.bind(query,key_frame,value_frame));
  HYLE_ASSIGN_OR_RETURN(auto submitted,other.submit());
  HYLE_ASSIGN_OR_RETURN(auto old,submitted.wait());
  HYLE_RETURN_IF_ERROR(verify(*old,expected));
  for(unsigned episode=0;episode<64;++episode)for(unsigned replay=0;replay<2;++replay){
    const bool changed=episode%2;
    HYLE_RETURN_IF_ERROR(slot.prepare_batch(changed?next:layout));
    if(changed)HYLE_RETURN_IF_ERROR(require(!slot.bind(query,key_frame,next_values).ok(),"mixed generations accepted"));
    HYLE_RETURN_IF_ERROR(slot.bind(query,changed?next_keys:key_frame,changed?next_values:value_frame));
    HYLE_ASSIGN_OR_RETURN(auto run,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,run.wait());
    HYLE_RETURN_IF_ERROR(verify(*result,changed?next_expected:expected));
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"old output permitted native reuse"));
    HYLE_RETURN_IF_ERROR(verify(*old,expected));
    const auto report=slot.report();
    HYLE_RETURN_IF_ERROR(require(report.reused_native_plan==(replay!=0),"native complete topology reuse report differs"));
    if(episode || replay)HYLE_RETURN_IF_ERROR(require(report.upload_host_waits==0,"unchanged page IDs incurred upload wait"));
    HYLE_RETURN_IF_ERROR(require((episode==0 && replay==0 ? report.uploaded_bytes>0 && report.uploaded_bytes<=4*pages.size() : report.uploaded_bytes==0) && report.device_banks==2 &&
        report.staging_bytes==4*table_capacity && report.device_mirror_bytes==8*table_capacity,"native table costs expanded"));
  }
  std::rotate(pages.begin(),pages.begin()+1,pages.end());
  HYLE_ASSIGN_OR_RETURN(auto reordered,schema.version({next_qp,kp,pages,tails}));
  HYLE_ASSIGN_OR_RETURN(auto reordered_keys,keys.read(reordered,exact_k));
  HYLE_ASSIGN_OR_RETURN(auto reordered_values,values.read(reordered,exact_v));
  {
    HYLE_RETURN_IF_ERROR(slot.prepare_batch(reordered));
    const auto report=slot.report();
    HYLE_RETURN_IF_ERROR(require(report.reused_native_plan && report.uploaded_bytes>0 &&
        report.uploaded_bytes<=4*pages.size(),"page ID change did not reuse plan with actual delta upload"));
    HYLE_RETURN_IF_ERROR(slot.bind(query,reordered_keys,reordered_values));
    HYLE_ASSIGN_OR_RETURN(auto run,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,run.wait());
    HYLE_RETURN_IF_ERROR(verify(*result,oracle(next_qp)));
    HYLE_RETURN_IF_ERROR(verify(*old,expected));
  }
  HYLE_RETURN_IF_ERROR(slot.close());HYLE_RETURN_IF_ERROR(slot.drain());
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"closed slot accepted work"));
  other={};HYLE_RETURN_IF_ERROR(verify(*old,expected));old={};submitted={};slot={};
  HYLE_RETURN_IF_ERROR(require(metadata.used()==0,"native page controller metadata leaked"));
  std::printf("Apple page16 %s D%u causal=%u H%u/2: double oracle, 64 topology episodes, two slots, logical%u>physical64, exact guards/costs PASS\n",std::is_same_v<T,Half>?"F16":"BF16",depth,causal,query_heads,16*(first_refs-1)+15);
  return Status::Ok();
}
Status run(bool long_reads){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());

  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
  if(!long_reads){
    for(unsigned depth:{64U,128U,256U})for(bool causal:{false,true})
      {HYLE_RETURN_IF_ERROR(exercise<Half>(context,device,staging,depth,causal));HYLE_RETURN_IF_ERROR(exercise<BFloat>(context,device,staging,depth,causal));}
  }else{
    for(unsigned depth:{64U,128U,256U})for(bool causal:{false,true}){
      HYLE_RETURN_IF_ERROR(exercise<Half>(context,device,staging,depth,causal,257,8));
      HYLE_RETURN_IF_ERROR(exercise<BFloat>(context,device,staging,depth,causal,257,8));
    }
    for(unsigned depth:{64U,128U}){
      HYLE_RETURN_IF_ERROR(exercise<Half>(context,device,staging,depth,false,513,16));
      HYLE_RETURN_IF_ERROR(exercise<BFloat>(context,device,staging,depth,false,513,16));
    }
  }
  return Status::Ok();
}
} // namespace
int main(int argc,char**){const auto status=run(argc>1);if(!status.ok()){std::fprintf(stderr,"FAIL: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
