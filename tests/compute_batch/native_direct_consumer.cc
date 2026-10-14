#include <hyle/compute/batch_execution.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/mem/quota_allocator.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <thread>
#include <type_traits>
#include <vector>
#include "native_arena_direct_consumer.h"
#include "ragged_append_journey.h"
#include "native_ragged_append_consumer.h"

namespace {
namespace c=hyle::compute;namespace cu=c::cuda;
using hyle::Status;template<class T>using Result=hyle::Result<T>;
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
template<class T>T encode(float x){if constexpr(std::is_same_v<T,half>)return __float2half(x);else return __float2bfloat16(x);}
template<class T>float decode(T x){if constexpr(std::is_same_v<T,half>)return __half2float(x);else return __bfloat162float(x);}
template<class T>Result<c::ReadLease> load(c::LeasePool&pool,const std::vector<T>&values){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(values.size()*sizeof(T)));HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(values))));return writer.publish();}
template<class T>Status oracle(const std::vector<T>&q,const std::vector<T>&k,const std::vector<T>&v,const c::BatchLayoutVersion&layout,c::BatchCapacity cap,const c::ReadLease&output){
  std::vector<T> result(cap.max_query_rows*cap.query_heads*cap.qk_head_dim);HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(result))));
  const auto host=layout.host();const unsigned D=cap.qk_head_dim,H=cap.query_heads,HK=cap.kv_heads;double maximum=0;
  for(unsigned r=0;r<layout.facts().requests;++r){
    const int qlen=host.query_indptr[r+1]-host.query_indptr[r];const int pages=host.kv_indptr[r+1]-host.kv_indptr[r];
    const int kvlen=cap.storage==c::BatchStorage::kPaged?(pages-1)*int(cap.page_size)+host.last_page_lengths[r]:pages;
    for(int qi=0;qi<qlen;++qi)for(unsigned h=0;h<H;++h){const int valid=kvlen-qlen+qi+1;std::vector<double>scores(valid);double high=-1e100,sum=0;
      const auto row=[&](int i){return cap.storage==c::BatchStorage::kPaged?host.page_indices[host.kv_indptr[r]+i/cap.page_size]*int(cap.page_size)+i%cap.page_size:host.kv_indptr[r]+i;};
      for(int i=0;i<valid;++i){double dot=0;for(unsigned d=0;d<D;++d)dot+=double(decode(q[((host.query_indptr[r]+qi)*H+h)*D+d]))*decode(k[(row(i)*HK+h/(H/HK))*D+d]);scores[i]=dot/std::sqrt(double(D));high=std::max(high,scores[i]);}
      for(auto&x:scores){x=std::exp(x-high);sum+=x;}for(unsigned d=0;d<D;++d){double expected=0;for(int i=0;i<valid;++i)expected+=scores[i]/sum*decode(v[(row(i)*HK+h/(H/HK))*D+d]);const double error=std::abs(decode(result[((host.query_indptr[r]+qi)*H+h)*D+d])-expected);maximum=std::max(maximum,error);if(!std::isfinite(error)||error>0.007)return Status::Internal("HYLE native batch independent oracle differs");}
    }
  }
  std::printf("HYLE batch oracle maxError=%.9g PASS\n",maximum);return Status::Ok();
}
template<class T>Status exercise(c::Context&context,c::Queue&queue,cu::MemoryResource&device,cu::MemoryResource&host,hyle::mem::AllocatorRef allocator,bool paged,unsigned depth,bool large){
  const unsigned length1=large?8193:19,length2=large?4097:10,page=16;
  const std::array<std::int32_t,3> query{0,1,3};const unsigned refs1=(length1+15)/16,refs2=(length2+15)/16;
  std::array<std::int32_t,3> kv{0,std::int32_t(paged?refs1:length1),std::int32_t(paged?refs1+refs2:length1+length2)};
  std::vector<std::int32_t> ids(paged?refs1+refs2:0);for(unsigned i=0;i<ids.size();++i)ids[i]=std::int32_t((i+7)%ids.size());
  const std::array<std::int32_t,2> tails{std::int32_t((length1-1)%page+1),std::int32_t((length2-1)%page+1)};
  c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.query_type=std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16,.output_type=std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16,.key_storage=c::CacheStorage::plain(std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16).value(),.value_storage=c::CacheStorage::plain(std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16).value(),
    .max_requests=2,.max_query_rows=3,.max_kv_rows=length1+length2,.max_pages=paged?refs1+refs2+1:0,.max_page_refs=paged?refs1+refs2:0,.query_heads=4,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth,.page_size=paged?page:1};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap,{.allocator=allocator}));
  const c::HostBatchLayout metadata{query,kv,ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version(metadata));
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,{.resource=device.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=8u*1024u*1024u,.max_int_workspace_bytes=64u*1024u,.max_invocations=1,.scale=1.f/std::sqrt(float(depth)),.allocator=allocator}));
  HYLE_ASSIGN_OR_RETURN(auto invocation,c::BatchInvocation::create(program,queue));
  std::vector<T> q(3*4*depth),k((paged?cap.max_pages*page:cap.max_kv_rows)*2*depth),v(k.size());
  for(unsigned i=0;i<q.size();++i)q[i]=encode<T>(0.2f*std::sin(float(i)*0.017f));
  for(unsigned i=0;i<k.size();++i){k[i]=encode<T>(0.3f*std::cos(float(i)*0.019f));v[i]=encode<T>(std::sin(float(i)*0.013f)+float(i%11)*0.01f);}
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),(q.size()+k.size()+v.size())*sizeof(T)+3u*256u));
  HYLE_ASSIGN_OR_RETURN(auto rq,load(pool,q));HYLE_ASSIGN_OR_RETURN(auto rk,load(pool,k));HYLE_ASSIGN_OR_RETURN(auto rv,load(pool,v));const std::array inputs{rq,rk,rv};
  HYLE_ASSIGN_OR_RETURN(auto prepared,invocation.prepare_batch(layout));HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(inputs));HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(auto output,invocation.output());
  HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,cap,output));
  const auto report=invocation.report();HYLE_RETURN_IF_ERROR(require(report.layout_generation==layout.generation(),"batch plan generation differs"));
  HYLE_RETURN_IF_ERROR(require(report.metadata_upload.logical_bytes==schema.metadata_offsets().bytes,"metadata upload byte count"));
  if(large)HYLE_RETURN_IF_ERROR(require(report.split_kv&&report.float_workspace_bytes>0,"long KV did not use original split plan"));
  prepared={};done={};HYLE_RETURN_IF_ERROR(require(invocation.prepare_batch(layout).error().IsUnavailable(),"old output did not block overwrite"));output.reset();
  auto next_query=query;next_query[1]=2;auto next_ids=ids;if(paged)std::swap(next_ids.front(),next_ids.back());
  HYLE_ASSIGN_OR_RETURN(auto next_layout,schema.version({next_query,kv,next_ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));HYLE_ASSIGN_OR_RETURN(auto next,invocation.update_batch(next_layout));
  HYLE_RETURN_IF_ERROR(require(!invocation.report().reused_native_plan,"different query topology reused an old plan"));
  HYLE_ASSIGN_OR_RETURN(auto second,next.run(inputs));HYLE_RETURN_IF_ERROR(second.wait());HYLE_ASSIGN_OR_RETURN(auto old_result,invocation.output());
  HYLE_RETURN_IF_ERROR(oracle(q,k,v,next_layout,cap,old_result));next={};second={};old_result.reset();
  if(paged)std::swap(next_ids.front(),next_ids.back());
  HYLE_ASSIGN_OR_RETURN(auto third_layout,schema.version({next_query,kv,next_ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));HYLE_ASSIGN_OR_RETURN(auto third,invocation.update_batch(third_layout));
  HYLE_RETURN_IF_ERROR(require(invocation.report().reused_native_plan,"identical native topology failed explicit metadata update"));
  HYLE_ASSIGN_OR_RETURN(auto third_done,third.run(inputs));HYLE_RETURN_IF_ERROR(third_done.wait());HYLE_ASSIGN_OR_RETURN(old_result,invocation.output());
  third={};third_done={};invocation={};HYLE_RETURN_IF_ERROR(require(c::BatchInvocation::create(program,queue).error().IsUnavailable(),"escaped result did not retain cold slot capacity"));program={};HYLE_RETURN_IF_ERROR(oracle(q,k,v,third_layout,cap,old_result));
  HYLE_ASSIGN_OR_RETURN(auto read_uses,c::LeaseUsePool::create());HYLE_ASSIGN_OR_RETURN(auto use,read_uses.begin());HYLE_RETURN_IF_ERROR(use.read(old_result));use={};
  std::printf("HYLE Direct %s %s D%u native plan/update/run/old-result PASS\n",paged?"paged":"ragged",std::is_same_v<T,half>?"F16":"BF16",depth);return Status::Ok();
}
Status run(bool ragged_only=false){hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());HYLE_ASSIGN_OR_RETURN(auto device_id,cu::device(0));HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id,c::flashinfer::provider()));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());HYLE_ASSIGN_OR_RETURN(auto device,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto host,cu::MemoryResource::create(context,cu::MemoryKind::kPinnedHost));
  struct DrainBeforeResources{c::Context&context;~DrainBeforeResources(){context.drain().IgnoreError();}}drain_before_resources{context};
  if(!ragged_only){for(auto paged:{true,false}){HYLE_RETURN_IF_ERROR(exercise<half>(context,queue,device,host,metadata.get(),paged,64,false));HYLE_RETURN_IF_ERROR(exercise<__nv_bfloat16>(context,queue,device,host,metadata.get(),paged,128,true));}HYLE_RETURN_IF_ERROR(hyle_f5_arena_fixture::run(context,queue,device,host,metadata.get()));}
  HYLE_RETURN_IF_ERROR(hyle_f5_cuda_ragged_fixture::run(context,queue,device,host));
  HYLE_RETURN_IF_ERROR(context.drain());const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(metadata.used()&&std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));HYLE_RETURN_IF_ERROR(require(metadata.used()==0,"batch metadata/provider/root ownership leaked"));return Status::Ok();}
} // namespace
int main(int argc,char**){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run(argc>1);if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F5 native Direct: %s\n",text);return 1;}
