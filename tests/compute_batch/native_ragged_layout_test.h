#pragma once
#include <gtest/gtest.h>
#include "../compute_provider_native/half_test_support.h"
#if defined(HYLE_G01_APPLE_RAGGED)
#include "hyle/compute/apple_batch.h"
#else
#include "hyle/compute/vulkan_batch.h"
#endif
#include <array>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <vector>
#include "hyle/mem/quota_allocator.h"
namespace {
namespace c=hyle::compute;
#if defined(HYLE_G01_APPLE_RAGGED)
namespace n=c::apple;
#else
namespace n=c::vulkan;
#endif
using F16=hyle::simd::float16_t;using BF16=hyle::simd::bfloat16_t;
template<class T>void ragged_native_case(unsigned depth,bool causal){
  auto made=n::Context::create();ASSERT_TRUE(made)<<made.error().message();auto context=std::move(*made);
#if defined(HYLE_G01_APPLE_RAGGED)
  auto memory=n::MemoryResource::create(context).value();
#else
  ASSERT_FALSE(context.environment().software_device);auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
#endif
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),.key_storage=c::CacheStorage::plain(half_test::data_type<T>()).value(),.value_storage=c::CacheStorage::plain(half_test::data_type<T>()).value(),
    .max_requests=3,.max_query_rows=20,.max_kv_rows=84,.max_pages=0,.max_page_refs=0,
    .query_heads=8,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,4> qp{0,1,18,18},kp{0,31,78,78};
  auto layout=schema.version({qp,kp,{},{}}).value();const c::RaggedAttention operation{cap,1.F/std::sqrt(float(depth)),causal};
  auto prepared=n::prepare_ragged_batch(context,layout,operation);ASSERT_TRUE(prepared)<<prepared.error().message();
  auto program=std::move(*prepared);
  hyle::mem::QuotaAllocator denied_metadata(hyle::mem::system_allocator(),0);auto denied=program.create_invocation(denied_metadata.get());ASSERT_FALSE(denied);EXPECT_TRUE(denied.error().IsResourceExhausted());
  auto first=program.create_invocation().value(),second=program.create_invocation().value();
  auto qd=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{20,8,depth}).value();
  auto kd=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{84,2,depth}).value();
  std::vector<T> q(*qd.element_count()),k(*kd.element_count()),v(k.size());
  for(std::size_t i=0;i<q.size();++i)q[i]=half_test::encode<T>(float(int(i%13)-6)*.03125F);
  for(std::size_t i=0;i<k.size();++i){k[i]=half_test::encode<T>(float(int(i%17)-8)*.015625F);v[i]=half_test::encode<T>(float(int(i%29)-14)*.03125F);}
  for(unsigned row=78;row<84;++row)for(unsigned z=0;z<2*depth;++z)v[row*2*depth+z]=half_test::encode<T>(128.F);
  auto pool=c::LeasePool::create(memory.ref(),(q.size()+k.size()+v.size())*2+2048).value();
  const auto upload=[&](const c::TensorDesc &description,const auto &data){
#if defined(HYLE_G01_APPLE_RAGGED)
    auto writer=pool.allocate(data.size()*2).value();EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(data))).ok());
    return memory.binding(writer.publish().value(),description).value();
#else
    return context.upload(description,std::as_bytes(std::span(data))).value();
#endif
  };
  const std::array inputs{upload(qd,q),upload(kd,k),upload(kd,v)};
  const auto oracle=[&](auto queries){std::vector<double> result(18*8*depth),weights;
    for(unsigned r=0;r<2;++r){const int nq=queries[r+1]-queries[r],nk=kp[r+1]-kp[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(int h=0;h<8;++h){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(unsigned d=0;d<depth;++d)dot+=half_test::decode(q[((queries[r]+row)*8+h)*depth+d])*half_test::decode(k[((kp[r]+j)*2+h/4)*depth+d]);
          weights[j]=causal&&j>nk-nq+row?0.:std::exp(dot*operation.scale);sum+=weights[j];}
        for(unsigned d=0;d<depth;++d)for(int j=0;j<nk;++j)result[((queries[r]+row)*8+h)*depth+d]+=weights[j]/sum*half_test::decode(v[((kp[r]+j)*2+h/4)*depth+d]);
      }}return result;};
  const auto read=[&](const n::Results &result){std::vector<T> actual(18*8*depth);
#if defined(HYLE_G01_APPLE_RAGGED)
    auto od=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{18,8,depth}).value();
    auto lease=memory.import_readonly(result.arrays()[0],od).value();EXPECT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
#else
    EXPECT_TRUE(context.download(result.tensors()[0],std::as_writable_bytes(std::span(actual))).ok());
#endif
    return actual;};
  const auto verify=[&](const auto &actual,const auto &expected){double worst=0;
    for(std::size_t i=0;i<expected.size();++i){ASSERT_TRUE(std::isfinite(half_test::decode(actual[i])));worst=std::max(worst,std::abs(half_test::decode(actual[i])-expected[i]));}
    EXPECT_LE(worst,(std::is_same_v<T,BF16>?8e-3:2e-3));};
  ASSERT_TRUE(first.bind(inputs).ok());auto a=first.submit().value();auto old=a.wait().value();const auto expected=oracle(qp);verify(read(*old),expected);
  ASSERT_TRUE(second.bind(inputs).ok());auto b=second.submit().value();auto independent=b.wait().value();verify(read(*independent),expected);verify(read(*old),expected);
  const std::array<std::int32_t,4> next_q{0,17,18,18};auto next_layout=schema.version({next_q,kp,{},{}}).value();
  auto next_program=n::prepare_ragged_batch(context,next_layout,operation);ASSERT_TRUE(next_program)<<next_program.error().message();auto next=next_program->create_invocation().value();
  ASSERT_TRUE(next.bind(inputs).ok());auto submitted=next.submit().value();auto current=submitted.wait().value();verify(read(*current),oracle(next_q));verify(read(*old),expected);
  auto wrong=operation;wrong.capacity.max_query_rows=21;auto mismatch=n::prepare_ragged_batch(context,layout,wrong);ASSERT_FALSE(mismatch);EXPECT_TRUE(mismatch.error().IsInvalidArgument());
  for(unsigned port=1;port<3;++port){std::vector<T> actual(k.size());
#if defined(HYLE_G01_APPLE_RAGGED)
    auto lease=memory.import_readonly(inputs[port],kd).value();ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
#else
    ASSERT_TRUE(context.download(inputs[port],std::as_writable_bytes(std::span(actual))).ok());
#endif
    const auto &expected_values=port==1?k:v;EXPECT_EQ(std::memcmp(actual.data(),expected_values.data(),actual.size()*sizeof(T)),0);
  }
  const std::array<std::int32_t,4> empty_q{0,0,0,0};auto empty=schema.version({empty_q,kp,{},{}}).value();
  auto empty_program=n::prepare_ragged_batch(context,empty,operation);ASSERT_TRUE(empty_program)<<empty_program.error().message();
  auto empty_slot=empty_program->create_invocation().value();ASSERT_TRUE(empty_slot.bind(inputs).ok());auto empty_run=empty_slot.submit().value();auto empty_result=empty_run.wait();ASSERT_TRUE(empty_result)<<empty_result.error().message();
  auto empty_ready=std::move(*empty_result);empty_ready={};empty_run={};
  current={};submitted={};independent={};b={};old={};a={};ASSERT_TRUE(context.close().ok());
}
TEST(NativeRaggedLayout, PackedTwoRequestsUseOriginalMathAndPreserveOldOutputs){
  for(unsigned depth:{64u,128u,256u})ragged_native_case<F16>(depth,depth!=64);
#if defined(HYLE_G01_APPLE_RAGGED)
  for(unsigned depth:{64u,128u,256u})ragged_native_case<BF16>(depth,depth!=64);
#endif
}
TEST(NativeRaggedLayout, EightRequestsJoinOnlyTheirActiveRows){
  auto context=n::Context::create().value();
#if defined(HYLE_G01_APPLE_RAGGED)
  auto memory=n::MemoryResource::create(context).value();
#else
  ASSERT_FALSE(context.environment().software_device);auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
#endif
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=8,.max_query_rows=10,.max_kv_rows=11,
    .max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,9> ptr{0,1,2,3,4,5,6,7,8};auto layout=schema.version({ptr,ptr,{},{}}).value();
  auto program=n::prepare_ragged_batch(context,layout,{cap,.125F,true});ASSERT_TRUE(program)<<program.error().message();auto made_slot=program->create_invocation();ASSERT_TRUE(made_slot)<<made_slot.error().message();auto slot=std::move(*made_slot);
  const auto qd=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{10,8,64}).value();
  const auto kd=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{11,2,64}).value();
  std::vector<F16> q(*qd.element_count()),k(*kd.element_count()),v(k.size(),half_test::encode<F16>(128.F));
  for(unsigned r=0;r<8;++r)for(unsigned i=0;i<128;++i)v[r*128+i]=half_test::encode<F16>(float(r)*.125F);
  auto pool=c::LeasePool::create(memory.ref(),(q.size()+k.size()+v.size())*2+2048).value();
  const auto upload=[&](const c::TensorDesc &description,const auto &values){
#if defined(HYLE_G01_APPLE_RAGGED)
    auto writer=pool.allocate(values.size()*2).value();EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());return memory.binding(writer.publish().value(),description).value();
#else
    return context.upload(description,std::as_bytes(std::span(values))).value();
#endif
  };
  const std::array inputs{upload(qd,q),upload(kd,k),upload(kd,v)};ASSERT_TRUE(slot.bind(inputs).ok());auto submitted=slot.submit();ASSERT_TRUE(submitted)<<submitted.error().message();auto done=std::move(*submitted);auto waited=done.wait();ASSERT_TRUE(waited)<<waited.error().message();auto result=std::move(*waited);std::vector<F16> actual(8*8*64);
#if defined(HYLE_G01_APPLE_RAGGED)
  auto od=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{8,8,64}).value();auto output=memory.import_readonly(result->arrays()[0],od).value();ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());output.reset();
#else
  ASSERT_TRUE(context.download(result->tensors()[0],std::as_writable_bytes(std::span(actual))).ok());
#endif
  for(unsigned r=0;r<8;++r)for(unsigned i=0;i<512;++i)EXPECT_EQ(actual[r*512+i].bits,half_test::encode<F16>(float(r)*.125F).bits);
  result={};done={};ASSERT_TRUE(context.close().ok());
}

// The physical ports stay at capacity; only this immutable host layout changes.
// Zero Q/K makes the expected result an independent average of physical V.
template<class T>void runtime_native_ragged(bool encoded,unsigned depth,unsigned value_depth,int window,unsigned mode){
  SCOPED_TRACE(::testing::Message()<<"encoded="<<encoded<<" D="<<depth<<" Dv="<<value_depth<<" window="<<window<<" mode="<<mode);
  auto context=n::Context::create().value();
#if defined(HYLE_G01_APPLE_RAGGED)
  auto memory=n::MemoryResource::create(context).value();
#else
  ASSERT_FALSE(context.environment().software_device);auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
#endif
  const auto type=half_test::data_type<T>();const bool causal=window>=0 || depth!=128;
  const auto storage=encoded?c::CacheStorage::q8_0().value():c::CacheStorage::plain(type).value();
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.query_type=type,.output_type=type,
    .key_storage=storage,.value_storage=storage,.max_requests=3,.max_query_rows=24,.max_kv_rows=192,
    .max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=value_depth,.page_size=1};
  const std::vector<std::int32_t> qp=mode==0?std::vector<std::int32_t>{0,1}:
      mode==1?std::vector<std::int32_t>{0,1,2,2}:std::vector<std::int32_t>{0,17,18,18};
  std::vector<std::int32_t> kp(qp.size(),0);
  for(std::size_t r=0;r+1<qp.size();++r)kp[r+1]=kp[r]+qp[r+1]-qp[r];
  auto schema=c::BatchSchema::create(cap).value();auto initial=schema.version({qp,kp,{},{}}).value();
  auto program=n::prepare_ragged_batch(context,initial,{cap,.125F,causal,window}).value();
  auto slot=program.create_invocation().value(),other=program.create_invocation().value();
  n::Invocation invalid;EXPECT_FALSE(n::update_ragged_batch(invalid,initial));
  const auto qd=c::batch_query_description(cap).value(),kd=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value(),
      vd=c::batch_cache_description(cap,c::BatchCacheRole::kValue).value();
#if defined(HYLE_G01_APPLE_RAGGED)
  auto pool=c::LeasePool::create(memory.ref(),qd.required_bytes().value()+kd.required_bytes().value()+vd.required_bytes().value()+256).value();
#endif
  const auto upload=[&](const c::TensorDesc& desc,const std::vector<std::byte>& bytes){
#if defined(HYLE_G01_APPLE_RAGGED)
    auto writer=pool.allocate(bytes.size(),64).value();EXPECT_TRUE(writer.view()->write(bytes).ok());return memory.binding(writer.publish().value(),desc).value();
#else
    return context.upload(desc,bytes).value();
#endif
  };
  const auto code=[](unsigned row,unsigned head){return int((row*3+head*5)%63)-31;};
  const auto cache_bytes=[&](bool value){const auto desc=value?vd:kd;const auto width=value?value_depth:depth;
    std::vector<std::byte> bytes(desc.required_bytes().value(),std::byte{0});
    if(encoded){
      for(unsigned row=0;row<192;++row)for(unsigned head=0;head<2;++head)for(unsigned block=0;block<width/32;++block){
        const auto offset=((row*2+head)*(width/32)+block)*34;
        const std::uint16_t scale=row==191?0x7e00:value?0x2400:0;std::memcpy(bytes.data()+offset,&scale,2);
        std::fill_n(bytes.begin()+offset+2,32,std::byte{std::uint8_t(value?code(row,head):0)});
      }
    }else{
      std::vector<T> words(192*2*width);
      for(unsigned row=0;row<192;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<width;++d)
        words[(row*2+head)*width+d]=half_test::encode<T>(row==191?std::numeric_limits<float>::quiet_NaN():value?float(code(row,head))/64.F:0.F);
      std::memcpy(bytes.data(),words.data(),bytes.size());
    }return bytes;
  };
  const std::array inputs{upload(qd,std::vector<std::byte>(qd.required_bytes().value(),std::byte{0})),upload(kd,cache_bytes(false)),upload(vd,cache_bytes(true))};
  const auto verify=[&](const n::Results& result,const c::BatchLayoutVersion& selected){
    const auto host=selected.host();std::vector<T> actual(std::size_t(qp.back())*8*value_depth);
#if defined(HYLE_G01_APPLE_RAGGED)
    auto od=c::TensorDesc::dense(type,std::array<std::int64_t,3>{qp.back(),8,value_depth}).value();
    auto lease=memory.import_readonly(result.arrays()[0],od).value();ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
#else
    ASSERT_TRUE(context.download(result.tensors()[0],std::as_writable_bytes(std::span(actual))).ok());
#endif
    for(std::size_t r=0;r+1<qp.size();++r){const auto nq=unsigned(qp[r+1]-qp[r]),nk=unsigned(host.kv_indptr[r+1]-host.kv_indptr[r]);
      for(unsigned row=0;row<nq;++row){const auto end=causal?nk-nq+row+1:nk;
        const auto first=window<0?0:end>unsigned(window+1)?end-unsigned(window+1):0;
        for(unsigned head=0;head<8;++head){double expected=0;
          for(unsigned j=first;j<end;++j)expected+=double(code(unsigned(host.kv_indptr[r])+j,head/4))/64.;
          expected/=double(end-first);
          for(unsigned d=0;d<value_depth;++d){const auto value=half_test::decode(actual[(std::size_t(qp[r])+row)*8*value_depth+head*value_depth+d]);
            ASSERT_TRUE(std::isfinite(value));EXPECT_NEAR(value,expected,std::is_same_v<T,BF16>?.008:.002);
          }
        }
      }
    }
  };
  ASSERT_TRUE(other.bind(inputs).ok());auto old_run=other.submit().value();auto old=old_run.wait().value();verify(*old,initial);
  ASSERT_TRUE(slot.bind(inputs).ok());auto foreign_schema=c::BatchSchema::create(cap).value();
  auto foreign=foreign_schema.version({qp,kp,{},{}}).value();EXPECT_TRUE(n::update_ragged_batch(slot,foreign).IsInvalidArgument());
  auto no_queries=qp;std::fill(no_queries.begin(),no_queries.end(),0);auto changed=schema.version({no_queries,kp,{},{}}).value();
  EXPECT_TRUE(n::update_ragged_batch(slot,changed).IsInvalidArgument());
  if(causal && mode==2){const std::array<std::int32_t,4> short_kv{0,1,2,2};auto short_layout=schema.version({qp,short_kv,{},{}}).value();
    EXPECT_TRUE(n::update_ragged_batch(slot,short_layout).IsInvalidArgument());}
  {auto preserved=slot.submit().value();auto result=preserved.wait().value();verify(*result,initial);}
  for(unsigned step=0;step<64;++step){
    SCOPED_TRACE(step);std::vector<std::int32_t> current_k{0};
    for(std::size_t r=0;r+1<qp.size();++r){const auto nq=unsigned(qp[r+1]-qp[r]);
      current_k.push_back(current_k.back()+int(nq?nq+(step*(17+unsigned(r)*6))%79:0));}
    auto layout=schema.version({qp,current_k,{},{}}).value();
    ASSERT_TRUE(n::update_ragged_batch(slot,layout).ok());EXPECT_FALSE(slot.submit()); // Success unbinds old imports.
    ASSERT_TRUE(slot.bind(inputs).ok());auto run=slot.submit().value();
    EXPECT_FALSE(n::update_ragged_batch(slot,layout)); // Owning completion/pending work excludes reuse.
    auto result=run.wait().value();verify(*result,layout);verify(*old,initial);
    EXPECT_FALSE(n::update_ragged_batch(slot,layout)); // The old output still excludes reuse.
  }
  old={};old_run={};ASSERT_TRUE(context.close().ok());EXPECT_FALSE(n::update_ragged_batch(slot,initial));
}
TEST(NativeRaggedLayout, PublicRuntimeKvUpdatesKeepIndependentSlotsAndCheckedBoundaries){
  for(unsigned mode:{0U,1U,2U})for(bool encoded:{false,true}){
    for(unsigned depth:{64U,128U,256U}){
      runtime_native_ragged<F16>(encoded,depth,depth,-1,mode);
#if defined(HYLE_G01_APPLE_RAGGED)
      runtime_native_ragged<BF16>(encoded,depth,depth,-1,mode);
#endif
    }
    runtime_native_ragged<F16>(encoded,64,128,16,mode);
#if defined(HYLE_G01_APPLE_RAGGED)
    runtime_native_ragged<BF16>(encoded,64,128,16,mode);
#endif
  }
}
} // namespace
