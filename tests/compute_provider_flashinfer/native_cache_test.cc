#include <gtest/gtest.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include <hyle/compute/model_ops.h>
#include <hyle/compute/prepared_operation.h>
#include "../compute_provider_native/half_test_support.h"
#include "../compute_batch/allocation_fault_fixture.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
namespace c=hyle::compute;
hyle::Status encode_words(c::DataType type,c::CacheStorage storage,std::span<const float> values,
                          std::vector<std::uint8_t>& bytes,bool* accepted=nullptr){
  if(accepted)*accepted=false;
  auto context=c::Context::create(c::cuda::device(0).value(),c::flashinfer::provider()).value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto queue=context.create_queue().value();
  const auto input=c::TensorDesc::dense(type,std::array<std::int64_t,2>{std::int64_t(values.size()/64),64}).value();
  const auto operation=c::Operation::custom(c::model_ops::CacheEncode{storage});
  const auto output=c::infer_single_output(operation,std::array{input}).value();
  auto source=hyle::mem::Buffer::create(memory.ref(),input.required_bytes().value(),64).value();
  auto destination=hyle::mem::Buffer::create(memory.ref(),output.required_bytes().value(),64).value();
  if(type==c::DataType::kFloat32)HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(values)));
  else{
    std::vector<std::uint16_t> words(values.size());
    for(std::size_t i=0;i<values.size();++i)words[i]=type==c::DataType::kFloat16?
        half_test::encode<hyle::simd::float16_t>(values[i]).bits:half_test::encode<hyle::simd::bfloat16_t>(values[i]).bits;
    HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(words))));
  }
  const std::array inputs{c::ConstTensorView::create(input,source.view()).value()};
  const std::array outputs{c::TensorView::create(output,destination.view()).value()};
  HYLE_ASSIGN_OR_RETURN(auto prepared,c::prepare(context,operation,inputs,outputs,memory.ref()));
  EXPECT_EQ(prepared.provider().id,c::flashinfer::kProviderId);
  HYLE_ASSIGN_OR_RETURN(auto invocation,prepared.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queue,inputs,outputs));
  if(accepted)*accepted=true;
  HYLE_RETURN_IF_ERROR(done.wait());
  bytes.resize(output.required_bytes().value());
  return destination.view().read(std::as_writable_bytes(std::span(bytes)));
}
float input_value(float value,c::DataType type){
  if(type==c::DataType::kFloat16)return float(half_test::decode(half_test::encode<hyle::simd::float16_t>(value)));
  if(type==c::DataType::kBFloat16)return float(half_test::decode(half_test::encode<hyle::simd::bfloat16_t>(value)));
  return value;
}
std::uint8_t float8_oracle(double value){
  const auto sign=std::signbit(value)?std::uint8_t{128}:std::uint8_t{0};value=std::abs(value);
  unsigned selected=0;double error=std::numeric_limits<double>::infinity();
  for(unsigned code=0;code<=126;++code){
    const auto exponent=code>>3,mantissa=code&7;
    const double decoded=exponent?std::ldexp(1.+double(mantissa)/8.,int(exponent)-7):std::ldexp(double(mantissa)/8.,-6);
    const auto delta=std::abs(value-decoded);
    if(delta<error || (delta==error && !(code&1))){error=delta;selected=code;}
  }
  return static_cast<std::uint8_t>(sign|selected);
}
double fp8_value(std::uint8_t code){
  const unsigned magnitude=code&127,exponent=magnitude>>3,mantissa=magnitude&7;
  const double value=exponent?std::ldexp(1.+double(mantissa)/8.,int(exponent)-7):std::ldexp(double(mantissa)/8.,-6);
  return code&128?-value:value;
}
template<class T>void boxed_case(bool paged,bool quantized,unsigned depth,unsigned value_depth,
                                bool causal,int window,bool long_precision=false,bool long_prefill=false,bool strong_weights=false){
  SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" quantized="<<quantized<<" depth="<<depth
      <<" value_depth="<<value_depth<<" window="<<window<<" long_prefill="<<long_prefill);
  const unsigned query_rows=long_precision?1u:(long_prefill?66u:strong_weights?2u:(causal?18u:68u));
  const unsigned physical_rows=long_precision?4096u:(long_prefill?(paged?256u:274u):64u);
  const unsigned heads=long_precision?1u:8u,kv_heads=long_precision?1u:2u;
  const auto storage=quantized?c::CacheStorage::float8(.5F).value():
      c::CacheStorage::plain(half_test::data_type<T>()).value();
  const auto value_storage=quantized?c::CacheStorage::float8(2.F).value():storage;
  const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
    .query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),
    .key_storage=storage,.value_storage=value_storage,.max_requests=long_precision?1u:2u,
    .max_query_rows=query_rows,.max_kv_rows=long_prefill?274u:physical_rows,.max_pages=paged?(long_prefill?16u:4u):0u,
    .max_page_refs=paged?(long_prefill?19u:5u):0u,.query_heads=heads,.kv_heads=kv_heads,
    .qk_head_dim=depth,.value_head_dim=value_depth,.page_size=paged?16u:1u};
  auto context=c::Context::create(c::cuda::device(0).value(),c::flashinfer::provider()).value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto host=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost).value();
  auto queue=context.create_queue().value();auto schema=c::BatchSchema::create(cap).value();
  const std::vector<std::int32_t> qp=long_precision?std::vector<std::int32_t>{0,1}:
      (long_prefill?std::vector<std::int32_t>{0,1,66}:strong_weights?std::vector<std::int32_t>{0,1,2}:(causal?std::vector<std::int32_t>{0,1,18}:std::vector<std::int32_t>{0,33,68}));
  const std::vector<std::int32_t> kp=long_precision?std::vector<std::int32_t>{0,4096}:
      (long_prefill?(paged?std::vector<std::int32_t>{0,9,19}:std::vector<std::int32_t>{0,129,274}):
       (paged?std::vector<std::int32_t>{0,2,5}:std::vector<std::int32_t>{0,31,64}));
  const std::vector<std::int32_t> ids=paged?(long_prefill?
      std::vector<std::int32_t>{13,0,15,1,2,3,4,5,6,9,10,11,12,7,8,0,3,4,14}:
      std::vector<std::int32_t>{3,1,0,2,1}):std::vector<std::int32_t>{};
  const std::vector<std::int32_t> tails=paged?(long_prefill?std::vector<std::int32_t>{1,1}:
      std::vector<std::int32_t>{15,1}):std::vector<std::int32_t>{};
  auto layout=schema.version({qp,kp,ids,tails}).value();
  std::vector<T> q(std::size_t(query_rows)*heads*depth);
  std::vector<float> k(std::size_t(physical_rows)*kv_heads*depth),v(std::size_t(physical_rows)*kv_heads*value_depth);
  for(std::size_t i=0;i<q.size();++i)q[i]=half_test::encode<T>(long_precision?0.:strong_weights?.125:.17*std::sin(double(i)*.23));
  for(std::size_t i=0;i<k.size();++i)k[i]=long_precision?0.F:strong_weights?float((i/(kv_heads*depth))%5)*.25F:float(.19*std::cos(double(i)*.13));
  for(std::size_t i=0;i<v.size();++i)v[i]=long_precision?
      (i/value_depth<3072?1.F:-1.F):strong_weights?float(1+(i/(kv_heads*value_depth))%5)*.0625F:float(.23*std::sin(double(i)*.07)+.05*std::cos(double(i)*.17));
  std::vector<std::uint8_t> kb,vb;
  const auto cache_bytes=[&](std::span<const float> input,c::CacheStorage spec,std::vector<std::uint8_t>& output){
    if(quantized){const auto status=encode_words(half_test::data_type<T>(),spec,input,output);
      ASSERT_TRUE(status.ok())<<status.message();}
    else{std::vector<T> plain(input.size());for(std::size_t i=0;i<input.size();++i)plain[i]=half_test::encode<T>(input[i]);
      output.resize(plain.size()*sizeof(T));std::memcpy(output.data(),plain.data(),output.size());}
  };
  cache_bytes(k,storage,kb);cache_bytes(v,value_storage,vb);
  ASSERT_EQ(kb.size(),k.size()*(quantized?1u:sizeof(T)));
  ASSERT_EQ(vb.size(),v.size()*(quantized?1u:sizeof(T)));
  const auto value_at=[&](bool key,std::size_t index){
    if(quantized)return fp8_value((key?kb:vb)[index])*(key?storage.scale():value_storage.scale());
    return half_test::decode(half_test::encode<T>(key?k[index]:v[index]));
  };
  auto pool=c::LeasePool::create(memory.ref(),q.size()*sizeof(T)+kb.size()+vb.size()+512).value();
  const auto load=[&](std::span<const std::byte> bytes){auto writer=pool.allocate(bytes.size()).value();
    EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();};
  auto query=load(std::as_bytes(std::span(q)));
  c::ReadLease key_lease,value_lease;std::vector<c::ReadLease> key_pages(cap.max_pages),value_pages(cap.max_pages);
  c::PagedArena key_arena,value_arena;c::PagedArenaRead key_frame,value_frame;
  if(paged){
    key_arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=kb.size()+64}).value();
    value_arena=c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=vb.size()+64}).value();
    for(unsigned page=0;page<cap.max_pages;++page){auto kw=key_arena.acquire(page).value(),vw=value_arena.acquire(page).value();
      ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(kb).subspan(page*kb.size()/cap.max_pages,kb.size()/cap.max_pages))).ok());
      ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(vb).subspan(page*vb.size()/cap.max_pages,vb.size()/cap.max_pages))).ok());
      key_pages[page]=kw.publish(16).value();value_pages[page]=vw.publish(16).value();}
    auto kr=key_arena.read(layout,key_pages),vr=value_arena.read(layout,value_pages);
    ASSERT_TRUE(kr)<<kr.error().message();ASSERT_TRUE(vr)<<vr.error().message();
    key_frame=std::move(*kr);value_frame=std::move(*vr);
  }else{key_lease=load(std::as_bytes(std::span(kb)));value_lease=load(std::as_bytes(std::span(vb)));}
  const float scale=1.F/std::sqrt(float(depth));
  auto program=c::BatchProgram::prepare(context,schema,{.resource=memory.ref(),.host_workspace=host.ref(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,.scale=scale,.causal=causal,.window_left=window});
  ASSERT_TRUE(program)<<program.error().message();auto slot=c::BatchInvocation::create(*program,queue).value();
  auto prepared=slot.prepare_batch(layout);ASSERT_TRUE(prepared)<<prepared.error().message();
  if(strong_weights){
    if(long_prefill){ASSERT_GT(slot.report().cta_tile_q,16u);}
    else {ASSERT_EQ(slot.report().cta_tile_q,16u);}
  }
  const std::array inputs{query,key_lease,value_lease};
  auto done=paged?prepared->run(query,key_frame,value_frame):prepared->run(inputs);
  ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());auto result=slot.output().value();
  std::vector<T> actual(std::size_t(query_rows)*heads*value_depth);
  ASSERT_TRUE(result.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  for(unsigned request=0;request<cap.max_requests;++request){
    const int queries=qp[request+1]-qp[request],keys=long_precision?4096:(long_prefill?(request?145:129):(request?33:31));
    for(int row=0;row<queries;++row)for(unsigned head=0;head<heads;++head){
      const int position=keys-queries+row,first=window<0?0:std::max(0,position-window),last=causal?position+1:keys;
      std::vector<double> weights(static_cast<std::size_t>(keys));double sum=0;
      for(int j=first;j<last;++j){const auto physical=paged?unsigned(ids[kp[request]+j/16])*16u+unsigned(j%16):unsigned(kp[request]+j);
        double dot=0;for(unsigned d=0;d<depth;++d)dot+=half_test::decode(q[(std::size_t(qp[request]+row)*heads+head)*depth+d])*
            value_at(true,(std::size_t(physical)*kv_heads+head/(heads/kv_heads))*depth+d);
        weights[static_cast<std::size_t>(j)]=std::exp(dot*scale);sum+=weights[static_cast<std::size_t>(j)];}
      for(unsigned d=0;d<value_depth;++d){double expected=0;
        for(int j=first;j<last;++j){const auto physical=paged?unsigned(ids[kp[request]+j/16])*16u+unsigned(j%16):unsigned(kp[request]+j);
          expected+=weights[static_cast<std::size_t>(j)]*value_at(false,(std::size_t(physical)*kv_heads+head/(heads/kv_heads))*value_depth+d);}
        // Reference decodes actual FP8 bytes and independent K/V scales.
        // Tolerance includes the native tile arithmetic and output rounding.
        const auto observed=half_test::decode(actual[(std::size_t(qp[request]+row)*heads+head)*value_depth+d]);
        if(strong_weights){ASSERT_NEAR(observed,expected/sum,.0025);}
        else EXPECT_NEAR(observed,expected/sum,long_precision?.0006:.0025);}
    }
  }
  result.reset();*done={};*prepared={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
}
// The locked scheduler's minimum chunk is 128 rows (eight 16-row pages).
// These episodes exercise that actual table on both sides of its boundary;
// the expected reuse decisions do not call the implementation's predicate.
template<class T>void runtime_schedule(bool paged,bool fp8,unsigned depth,unsigned value_depth,
                                      int window,unsigned mode,bool captured=false){
  hyle_f5_fault_fixture::Allocator fault;
  {
  const auto type=half_test::data_type<T>();const bool causal=depth!=128;
  const auto key_storage=fp8?c::CacheStorage::float8(.5F).value():c::CacheStorage::plain(type).value();
  const auto value_storage=fp8?c::CacheStorage::float8(2.F).value():key_storage;
  const unsigned physical_rows=paged?2048u:2049u;
  const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
      .query_type=type,.output_type=type,.key_storage=key_storage,.value_storage=value_storage,
      .max_requests=3,.max_query_rows=18,.max_kv_rows=2049,.max_pages=paged?128u:0u,
      .max_page_refs=paged?128u:0u,.query_heads=8,.kv_heads=2,.qk_head_dim=depth,
      .value_head_dim=value_depth,.page_size=paged?16u:1u};
  auto context=c::Context::create(c::cuda::device(0).value(),c::flashinfer::provider()).value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto host=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost).value();
  auto queue=context.create_queue().value();auto schema=c::BatchSchema::create(cap).value();
  const auto sample=[](unsigned row,unsigned head,unsigned d){return float(int((row*3+head*5+d)%15)-7)*.125F;};
  const auto cache=[&](bool value){
    const auto width=value?value_depth:depth;const auto storage=value?value_storage:key_storage;
    std::vector<std::uint8_t> bytes(std::size_t(physical_rows)*2*width*(fp8?1u:2u));
    for(unsigned row=0;row<physical_rows;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<width;++d){
      const auto offset=(std::size_t(row)*2+head)*width+d;
      // Last physical row is deliberately never part of an active layout.
      const auto number=row+1==physical_rows?std::numeric_limits<float>::quiet_NaN():value?sample(row,head,d):0.F;
      if(fp8)bytes[offset]=std::isnan(number)?std::uint8_t{127}:float8_oracle(double(number)/storage.scale());
      else{const auto word=half_test::encode<T>(number);std::memcpy(bytes.data()+offset*2,&word,2);}
    }
    return bytes;
  };
  const auto kb=cache(false),vb=cache(true);
  auto pool=c::LeasePool::create(memory.ref(),kb.size()+vb.size()+18u*8u*depth*2u+512u).value();
  const auto load=[&](std::span<const std::byte> bytes){auto writer=pool.allocate(bytes.size(),64).value();
    EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();};
  std::vector<T> qwords(18u*8u*depth);const auto query=load(std::as_bytes(std::span(qwords)));
  c::ReadLease key,value;std::vector<c::ReadLease> key_pages(128),value_pages(128);c::PagedArena ka,va;
  if(paged){
    ka=c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=kb.size()+64}).value();
    va=c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=vb.size()+64}).value();
    for(unsigned page=0;page<128;++page){auto kw=ka.acquire(page).value(),vw=va.acquire(page).value();
      ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(kb).subspan(page*kb.size()/128,kb.size()/128))).ok());
      ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(vb).subspan(page*vb.size()/128,vb.size()/128))).ok());
      key_pages[page]=kw.publish(16).value();value_pages[page]=vw.publish(16).value();}
  }else{key=load(std::as_bytes(std::span(kb)));value=load(std::as_bytes(std::span(vb)));}
  auto program=c::BatchProgram::prepare(context,schema,{.resource=memory.ref(),.host_workspace=host.ref(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=16384,.scale=.125F,.causal=causal,
      .window_left=window,.allocator=fault.ref()}).value();
  auto slot=c::BatchInvocation::create(program,queue).value(),other=c::BatchInvocation::create(program,queue).value();
  c::ReadLease retained;c::BatchLayoutVersion initial;c::BatchPlanReport previous;
  const auto verify=[&](const c::ReadLease&output,const c::BatchLayoutVersion&layout){
    std::vector<T> actual(18u*8u*value_depth);ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    const auto h=layout.host();
    for(unsigned request=0;request<layout.facts().requests;++request){
      const auto queries=h.query_indptr[request+1]-h.query_indptr[request];
      const auto units=h.kv_indptr[request+1]-h.kv_indptr[request];
      const int rows=paged?(units?((units-1)*16+h.last_page_lengths[request]):0):units;
      for(int row=0;row<queries;++row){const int position=rows-queries+row;
        const int first=window<0?0:std::max(0,position-window),last=causal?position+1:rows;
        for(unsigned head=0;head<8;++head)for(unsigned d:std::array{0u,value_depth/2,value_depth-1}){
          double expected=0;
          for(int at=first;at<last;++at){const auto physical=paged?unsigned(h.page_indices[h.kv_indptr[request]+at/16])*16u+unsigned(at%16):unsigned(h.kv_indptr[request]+at);
            const auto index=(std::size_t(physical)*2+head/4)*value_depth+d;
            if(fp8)expected+=fp8_value(vb[index])*value_storage.scale();
            else{T word;std::memcpy(&word,vb.data()+index*2,2);expected+=half_test::decode(word);}}
          const auto observed=half_test::decode(actual[(std::size_t(h.query_indptr[request]+row)*8+head)*value_depth+d]);
          ASSERT_TRUE(std::isfinite(observed));ASSERT_NEAR(observed,expected/double(last-first),type==c::DataType::kFloat16?.0025:.008);
        }
      }
    }
  };
  for(unsigned step=0;step<64;++step){
    SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" fp8="<<fp8<<" d="<<depth<<" dv="<<value_depth<<" mode="<<mode<<" step="<<step);
    const bool moved_empty=step>=48 && step<56;const unsigned active=mode==1?1u:2u;
    std::vector<std::int32_t> qp{0},kp{0},ids,tails;
    for(unsigned request=0;request<=active;++request){
      const unsigned empty=moved_empty?(mode==1?0u:1u):active;
      const unsigned queries=request==empty?0u:mode==18 && qp.back()?17u:1u;
      const unsigned rows=queries?((step>=16 && step<32) || step>=56?129u+step%31u:32u+step%63u):step%2?512u:0u;
      qp.push_back(qp.back()+int(queries));
      if(paged){const auto pages=(rows+15)/16;kp.push_back(kp.back()+int(pages));tails.push_back(rows?(int(rows)-1)%16+1:0);
        for(unsigned page=0;page<pages;++page)ids.push_back(int((ids.size()*17+step*7)%127));}
      else kp.push_back(kp.back()+int(rows));
    }
    auto layout=schema.version({qp,kp,ids,tails}).value();
    const auto run=[&](c::PreparedBatch&prepared){
      if(paged){
        auto selected=ids;std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
        std::vector<c::ReadLease> keys,values;for(auto page:selected){keys.push_back(key_pages[page]);values.push_back(value_pages[page]);}
        auto kr=ka.read(layout,keys).value(),vr=va.read(layout,values).value();return prepared.run(query,kr,vr);
      }
      return prepared.run(std::array{query,key,value});
    };
    const bool reuse=step!=0 && step!=48 && step!=56 && (window>=0 || (step!=16 && step!=32));
    const auto before=fault.attempts;if(reuse)fault.fail=before;
    auto prepared=slot.update_batch(layout);fault.fail=SIZE_MAX;ASSERT_TRUE(prepared)<<prepared.error().message();
    const auto report=slot.report();EXPECT_EQ(report.reused_native_plan,reuse);
    if(reuse){EXPECT_EQ(fault.attempts,before);EXPECT_EQ(report.float_workspace_bytes,previous.float_workspace_bytes);
      EXPECT_EQ(report.int_workspace_bytes,previous.int_workspace_bytes);EXPECT_EQ(report.padded_batch_size,previous.padded_batch_size);
      EXPECT_EQ(report.host_waits,report.metadata_upload.physical_bytes?1u:0u);}
    if(captured){
      const auto prepare_capture=[&](){
        if(paged){auto selected=ids;std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
          std::vector<c::ReadLease> keys,values;for(auto page:selected){keys.push_back(key_pages[page]);values.push_back(value_pages[page]);}
          auto kr=ka.read(layout,keys).value(),vr=va.read(layout,values).value();return prepared->prepare_replay(query,kr,vr);}
        return prepared->prepare_replay(std::array{query,key,value});
      };
      if(step==0){
        fault.fail=fault.attempts;const auto rejected=prepare_capture();fault.fail=SIZE_MAX;
        ASSERT_TRUE(rejected.IsResourceExhausted())<<rejected.message();
        EXPECT_FALSE(slot.report().capture_compatible);
      }
      const auto before_capture=fault.attempts;const bool kept=reuse && previous.capture_compatible;
      if(kept)fault.fail=before_capture;
      const auto status=prepare_capture();fault.fail=SIZE_MAX;
      ASSERT_TRUE(status.ok())<<status.message();EXPECT_TRUE(slot.report().capture_compatible);
      if(kept)EXPECT_EQ(fault.attempts,before_capture);
      EXPECT_EQ(slot.report().host_waits,report.host_waits+(kept?0u:1u));
      if(step==0 && !paged && !fp8 && depth==64 && mode==1 && type==c::DataType::kFloat16){
        auto replacement_pool=c::LeasePool::create(memory.ref(),qwords.size()*sizeof(T)+256).value();
        auto writer=replacement_pool.allocate(qwords.size()*sizeof(T)).value();
        ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(qwords))).ok());
        auto replacement=writer.publish().value();const std::array changed{replacement,key,value};
        const auto rejected=prepared->run(changed);
        ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsFailedPrecondition());
        const auto waits=slot.report().host_waits;
        ASSERT_TRUE(prepared->prepare_replay(changed).ok());
        EXPECT_EQ(slot.report().host_waits,waits+1);
        ASSERT_TRUE(prepare_capture().ok());EXPECT_EQ(slot.report().host_waits,waits+2);
      }
    }
    // Capture first, before another slot executes a tensor kernel: a separate
    // fresh-process run of this test proves genuinely cold native recording.
    if(step==0){initial=layout;auto reference=other.prepare_batch(layout).value();auto done=run(reference).value();ASSERT_TRUE(done.wait().ok());retained=other.output().value();}
    previous=slot.report();auto done=run(*prepared);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    auto output=slot.output().value();*done={};*prepared={};EXPECT_FALSE(slot.update_batch(layout));
    verify(output,layout);verify(retained,initial);
  }
  retained={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(other.close().ok());ASSERT_TRUE(context.drain().ok());
  }
  EXPECT_EQ(fault.live,0u);
}
TEST(FlashinferCache, RuntimeScheduleReusesActualTilesAcrossKvExtents){
  using F16=hyle::simd::float16_t;using BF16=hyle::simd::bfloat16_t;
  for(bool paged:{false,true})for(bool fp8:{false,true}){
    for(unsigned mode:{1u,2u}){runtime_schedule<F16>(paged,fp8,64,64,-1,mode);runtime_schedule<BF16>(paged,fp8,64,64,-1,mode);}
    runtime_schedule<F16>(paged,fp8,128,128,-1,2);runtime_schedule<BF16>(paged,fp8,256,256,-1,2);
    runtime_schedule<F16>(paged,fp8,64,128,16,2);runtime_schedule<BF16>(paged,fp8,64,128,-1,18);
  }
}
TEST(FlashinferBatchReplay, ColdRecordAndRuntimeSchedule){
  using F16=hyle::simd::float16_t;using BF16=hyle::simd::bfloat16_t;
  for(bool paged:{false,true})for(bool fp8:{false,true}){
    for(unsigned mode:{1u,2u}){runtime_schedule<F16>(paged,fp8,64,64,-1,mode,true);runtime_schedule<BF16>(paged,fp8,64,64,-1,mode,true);}
    runtime_schedule<F16>(paged,fp8,128,128,-1,2,true);runtime_schedule<BF16>(paged,fp8,256,256,-1,2,true);
    runtime_schedule<F16>(paged,fp8,64,128,16,2,true);runtime_schedule<BF16>(paged,fp8,64,128,-1,18,true);
  }
}
TEST(FlashinferCache, EmptyRequestsAndAllZeroQueriesUseCheckedNoOp){
  for(bool paged:{false,true}){
    SCOPED_TRACE(::testing::Message()<<"empty paged="<<paged);
    const auto exercise=[&]() ->hyle::Status{
      HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cuda::device(0).value(),c::flashinfer::provider()));
      HYLE_ASSIGN_OR_RETURN(auto memory,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
      HYLE_ASSIGN_OR_RETURN(auto host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
      HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
      HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(c::DataType::kFloat16));
      const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
          .query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,.key_storage=plain,.value_storage=plain,
          .max_requests=2,.max_query_rows=1,.max_kv_rows=16,.max_pages=paged?1u:0u,.max_page_refs=paged?1u:0u,
          .query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=paged?16u:1u};
      HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
      HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,
          {.resource=memory.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=4096}));
      HYLE_ASSIGN_OR_RETURN(auto slot,c::BatchInvocation::create(program,queue));
      HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),16384));
      const auto load=[&](std::size_t bytes)->hyle::Result<c::ReadLease>{
        HYLE_ASSIGN_OR_RETURN(auto w,pool.allocate(bytes));HYLE_RETURN_IF_ERROR(w.view()->fill(std::byte{0}));return w.publish();};
      HYLE_ASSIGN_OR_RETURN(auto query,load(1024));HYLE_ASSIGN_OR_RETURN(auto key,load(4096));HYLE_ASSIGN_OR_RETURN(auto value,load(4096));
      const std::array inputs{query,key,value};
      for(unsigned count:{0u,2u}){
        SCOPED_TRACE(::testing::Message()<<"requests="<<count);
        const std::vector<std::int32_t> ptr(count+1,0),tails(paged?count:0,0);
        HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,{},tails}));
        HYLE_ASSIGN_OR_RETURN(auto prepared,slot.update_batch(layout));
        EXPECT_EQ(slot.report().float_workspace_bytes,0u);EXPECT_EQ(slot.report().int_workspace_bytes,0u);
        EXPECT_TRUE(prepared.prepare_replay(inputs).IsUnimplemented());
        if(paged){
          HYLE_ASSIGN_OR_RETURN(auto ka,c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=4160}));
          HYLE_ASSIGN_OR_RETURN(auto va,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=4160}));
          // Arena projection deliberately requires nonempty exact page guards.
          // Empty native execution uses the existing complete owning Q/K/V API.
          const auto kr=ka.read(layout,{}),vr=va.read(layout,{});
          EXPECT_FALSE(kr);EXPECT_FALSE(vr);
          if(kr || vr)return hyle::Status::Internal("empty arena projection bypassed its nonempty contract");
          EXPECT_TRUE(kr.error().IsInvalidArgument());EXPECT_TRUE(vr.error().IsInvalidArgument());
        }
        HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(inputs));HYLE_RETURN_IF_ERROR(done.wait());
      }
      HYLE_RETURN_IF_ERROR(slot.close());return context.drain();
    };
    const auto status=exercise();ASSERT_TRUE(status.ok())<<status.message();
  }
}

// Correlated, exactly representable FP8 rows make missing QK loads materially
// different from uniform averaging. Weak oscillating keys hid the K64 load bug.
TEST(FlashinferCache, MixedFp8ShortKeyStrongWeightsInLoopAndRepackedTiles){
  for(bool paged:{false,true})for(bool prefill:{false,true}){
    boxed_case<hyle::simd::float16_t>(paged,true,64,128,true,16,false,prefill,true);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,64,128,true,16,false,prefill,true);
  }
}
TEST(FlashinferCache, QuantizedPagedAndRaggedIndependentValueDepthAndWindow){
  for(bool paged:{false,true})for(int window:{-1,0,15,16}){
    boxed_case<hyle::simd::float16_t>(paged,true,64,128,true,window);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,64,128,true,window);
  }
}
TEST(FlashinferCache, QuantizedFullNoncausalAllowsMoreQueriesThanKeys){
  for(bool paged:{false,true}){
    boxed_case<hyle::simd::float16_t>(paged,true,64,128,false,-1);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,64,128,false,-1);
  }
}
TEST(FlashinferCache, PlainWindowAndIndependentValueDepth){
  for(bool paged:{false,true})for(int window:{0,16}){
    boxed_case<hyle::simd::float16_t>(paged,false,64,128,true,window);
    boxed_case<hyle::simd::bfloat16_t>(paged,false,64,128,true,window);
  }
}
TEST(FlashinferCache, QuantizedCoupledDepthsIncludingMainD256AndLongPrefill){
  for(bool paged:{false,true})for(unsigned depth:{64u,128u,256u}){
    boxed_case<hyle::simd::float16_t>(paged,true,depth,depth,true,16);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,depth,depth,true,16);
  }
  for(bool paged:{false,true}){
    boxed_case<hyle::simd::float16_t>(paged,true,256,256,true,-1,false,true);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,256,256,true,-1,false,true);
  }
}
TEST(FlashinferCache, PlainLongWindowAgainstCancellation){
  boxed_case<hyle::simd::float16_t>(false,false,64,128,true,4095,true);
}

TEST(FlashinferCacheEncode, IndependentBytesForThreeInputsAndScales){
  std::vector<float> values(512);
  for(std::size_t i=0;i<values.size();++i)values[i]=float(int(i%67)-33)*.0625F;
  values[0]=-0.F;values[1]=.001953125F;values[2]=.00390625F;values[3]=.125F;values[4]=.375F;values[5]=1000.F;values[6]=-1000.F;
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32})for(float scale:{.25F,2.F}){
    std::vector<std::uint8_t> bytes;const auto status=encode_words(type,c::CacheStorage::float8(scale).value(),values,bytes);
    ASSERT_TRUE(status.ok())<<status.message();
    ASSERT_EQ(bytes.size(),values.size());
    for(std::size_t i=0;i<values.size();++i)EXPECT_EQ(bytes[i],float8_oracle(double(input_value(values[i],type))/scale));
  }
}
TEST(FlashinferCacheEncode, FiniteFloat8ExtremePositiveScalesSaturateAndPreserveZero){
  const std::array samples{0.F,-0.F,1.F,-1.F,std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()};
  std::vector<float> values(64);for(std::size_t i=0;i<values.size();++i)values[i]=samples[i%samples.size()];
  for(float scale:{std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::max()}){
    SCOPED_TRACE(::testing::Message()<<"scale="<<scale);std::vector<std::uint8_t> bytes;
    const auto status=encode_words(c::DataType::kFloat32,c::CacheStorage::float8(scale).value(),values,bytes);
    ASSERT_TRUE(status.ok())<<status.message();ASSERT_EQ(bytes.size(),values.size());
    for(std::size_t i=0;i<values.size();++i)EXPECT_EQ(bytes[i],float8_oracle(double(values[i])/double(scale)))<<"index="<<i;
  }
}
TEST(FlashinferCacheEncode, NumericDomainUsesOriginalCompletionStatus){
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32})
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
      std::vector<float> values(64,1.F);values[19]=bad;std::vector<std::uint8_t> bytes;
      bool accepted=false;
      const auto status=encode_words(type,c::CacheStorage::float8(.5F).value(),values,bytes,&accepted);
      EXPECT_TRUE(accepted);
      EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
    }
}
template<class T>void split_precision_case(unsigned residual_period,bool single_partial=false){
  auto context=c::Context::create(c::cuda::device(0).value(),c::flashinfer::provider()).value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto host=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost).value();
  auto queue=context.create_queue().value();
  constexpr unsigned depth=64,heads=8,kv_heads=2;
  const unsigned rows=8192+(single_partial?64:0),queries=single_partial?2:1;
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,
      .query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),
      .key_storage=c::CacheStorage::float8(.5F).value(),.value_storage=c::CacheStorage::float8(single_partial?.75F:512.F).value(),
      .max_requests=queries,.max_query_rows=queries,.max_kv_rows=rows,.max_pages=0,.max_page_refs=0,
      .query_heads=heads,.kv_heads=kv_heads,.qk_head_dim=depth,.value_head_dim=depth,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();
  const std::vector<std::int32_t> qp=single_partial?std::vector<std::int32_t>{0,1,2}:std::vector<std::int32_t>{0,1};
  const std::vector<std::int32_t> kp=single_partial?std::vector<std::int32_t>{0,64,std::int32_t(rows)}:
      std::vector<std::int32_t>{0,std::int32_t(rows)};
  auto layout=schema.version({qp,kp,{},{}}).value();
  std::vector<float> key(rows*kv_heads*depth,0.F),value(key.size());
  for(std::size_t i=0;i<value.size();++i){
    const auto row=i/(kv_heads*depth);
    if(single_partial){
      float raw=row<64?128.F:0.F;
      if constexpr(half_test::data_type<T>()==c::DataType::kBFloat16){if(row<2)raw=144.F;}
      else if(row<4){constexpr float special[]{144.F,144.F,224.F,4.F};raw=special[row];}
      value[i]=raw*.75F;
    }else{
      value[i]=row<rows/2?65536.F:-65536.F;
      if(residual_period && row<rows/2 && row%residual_period==0)value[i]=73728.F;
    }
  }
  std::vector<std::uint8_t> kb,vb;
  ASSERT_TRUE(encode_words(c::DataType::kBFloat16,cap.key_storage,key,kb).ok());
  ASSERT_TRUE(encode_words(c::DataType::kBFloat16,cap.value_storage,value,vb).ok());
  auto pool=c::LeasePool::create(memory.ref(),kb.size()+vb.size()+4096).value();
  const auto load=[&](std::span<const std::byte> bytes){auto writer=pool.allocate(bytes.size(),64).value();
    EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();};
  std::vector<std::uint16_t> query(queries*heads*depth);
  const std::array inputs{load(std::as_bytes(std::span(query))),load(std::as_bytes(std::span(kb))),load(std::as_bytes(std::span(vb)))};
  auto program=c::BatchProgram::prepare(context,schema,{.resource=memory.ref(),.host_workspace=host.ref(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=64u<<10,.scale=.125F,.causal=true}).value();
  auto slot=c::BatchInvocation::create(program,queue).value();auto prepared=slot.prepare_batch(layout);
  ASSERT_TRUE(prepared)<<prepared.error().message();ASSERT_TRUE(slot.report().split_kv);
  auto completion=prepared->run(inputs);ASSERT_TRUE(completion)<<completion.error().message();
  ASSERT_TRUE(completion->wait().ok());auto output=slot.output().value();
  std::vector<T> actual(queries*heads*depth);
  ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  // The short request has one partial even while the long request forces split.
  // Narrowing before scaling would return96 instead of96.5/96.0625.
  const double single=half_test::data_type<T>()==c::DataType::kBFloat16?96.375:96.046875;
  const double expected=single_partial?half_test::decode(half_test::encode<T>(single)):
      (residual_period?4096./residual_period:0.);
  for(std::size_t i=0;i<actual.size();++i)
    EXPECT_NEAR(half_test::decode(actual[i]),single_partial && i>=heads*depth?0.:expected,.01);
  output.reset();*completion={};*prepared={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
}
TEST(FlashinferCache, SplitKvScalesOnlyAfterF32MergeAgainstPartialOverflow){
  split_precision_case<hyle::simd::float16_t>(0);
}
TEST(FlashinferCache, SplitKvPreservesBfloat16PartialCancellationResidual){
  split_precision_case<hyle::simd::bfloat16_t>(64);
}
TEST(FlashinferCache, SplitKvPreservesFloat16PartialCancellationResidual){
  split_precision_case<hyle::simd::float16_t>(256);
}
TEST(FlashinferCache, SplitKvSinglePartialScalesBeforeFinalNarrowing){
  split_precision_case<hyle::simd::float16_t>(0,true);
  split_precision_case<hyle::simd::bfloat16_t>(0,true);
}
}
