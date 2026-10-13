#pragma once
#include <gtest/gtest.h>
#include "../compute_provider_native/half_test_support.h"
#if defined(HYLE_G01_APPLE_RAGGED)
#include "hyle/compute/apple_batch.h"
#else
#include "hyle/compute/vulkan_batch.h"
#endif
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
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
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.data_type=half_test::data_type<T>(),
    .max_requests=3,.max_query_rows=20,.max_kv_rows=84,.max_pages=0,.max_page_refs=0,
    .query_heads=8,.kv_heads=2,.head_dim=depth,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,4> qp{0,1,18,18},kp{0,31,78,78};
  auto layout=schema.version({qp,kp,{},{}}).value();const c::RaggedAttention operation{cap,1.F/std::sqrt(float(depth)),causal};
  auto prepared=n::prepare_ragged_batch(context,layout,operation);ASSERT_TRUE(prepared)<<prepared.error().message();
  auto program=std::move(*prepared);auto first=program.create_invocation().value(),second=program.create_invocation().value();
  auto qd=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{20,8,depth}).value();
  auto kd=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{84,2,depth}).value();
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
    auto od=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{18,8,depth}).value();
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
    .max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.head_dim=64,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,9> ptr{0,1,2,3,4,5,6,7,8};auto layout=schema.version({ptr,ptr,{},{}}).value();
  auto program=n::prepare_ragged_batch(context,layout,{cap,.125F,true});ASSERT_TRUE(program)<<program.error().message();auto slot=program->create_invocation().value();
  const auto qd=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{10,8,64}).value();
  const auto kd=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{11,2,64}).value();
  std::vector<F16> q(*qd.element_count()),k(*kd.element_count()),v(k.size(),half_test::encode<F16>(128.F));
  for(unsigned r=0;r<8;++r)for(unsigned i=0;i<128;++i)v[r*128+i]=half_test::encode<F16>(float(r)*.125F);
  auto pool=c::LeasePool::create(memory.ref(),8192).value();
  const auto upload=[&](const c::TensorDesc &description,const auto &values){
#if defined(HYLE_G01_APPLE_RAGGED)
    auto writer=pool.allocate(values.size()*2).value();EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());return memory.binding(writer.publish().value(),description).value();
#else
    return context.upload(description,std::as_bytes(std::span(values))).value();
#endif
  };
  const std::array inputs{upload(qd,q),upload(kd,k),upload(kd,v)};ASSERT_TRUE(slot.bind(inputs).ok());auto done=slot.submit().value();auto result=done.wait().value();std::vector<F16> actual(8*8*64);
#if defined(HYLE_G01_APPLE_RAGGED)
  auto od=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{8,8,64}).value();auto output=memory.import_readonly(result->arrays()[0],od).value();ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());output.reset();
#else
  ASSERT_TRUE(context.download(result->tensors()[0],std::as_writable_bytes(std::span(actual))).ok());
#endif
  for(unsigned r=0;r<8;++r)for(unsigned i=0;i<512;++i)EXPECT_EQ(actual[r*512+i].bits,half_test::encode<F16>(float(r)*.125F).bits);
  result={};done={};ASSERT_TRUE(context.close().ok());
}
} // namespace
