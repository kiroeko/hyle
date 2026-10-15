#include <gtest/gtest.h>
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/compute/model_ops.h>
#include "../compute_provider_native/half_test_support.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
namespace c=hyle::compute;
namespace n=c::apple;
using F16=hyle::simd::float16_t;
using BF16=hyle::simd::bfloat16_t;

hyle::Status encode_words(n::Context& context,n::MemoryResource& memory,
    c::DataType type,std::span<const float> values,std::vector<std::uint8_t>& bytes,
    bool* accepted=nullptr) {
  if(accepted)*accepted=false;
  HYLE_ASSIGN_OR_RETURN(auto input,c::TensorDesc::dense(type,
      std::array<std::int64_t,2>{std::int64_t(values.size()/64),64}));
  const auto operation=c::Operation::custom(c::model_ops::CacheEncode{c::CacheStorage::q8_0().value()});
  HYLE_ASSIGN_OR_RETURN(auto output,c::infer_single_output(operation,std::array{input}));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),*input.required_bytes()+64));
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(*input.required_bytes()));
  if(type==c::DataType::kFloat32)HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(values)));
  else {
    std::vector<std::uint16_t> words(values.size());
    for(std::size_t i=0;i<values.size();++i)words[i]=type==c::DataType::kFloat16?
        half_test::encode<F16>(values[i]).bits:half_test::encode<BF16>(values[i]).bits;
    HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(words))));
  }
  HYLE_ASSIGN_OR_RETURN(auto read,writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto binding,memory.binding(std::move(read),input));
  HYLE_ASSIGN_OR_RETURN(auto spec,n::input_spec(input));
  HYLE_ASSIGN_OR_RETURN(auto program,context.prepare(operation,std::span(&spec,1)));
  HYLE_ASSIGN_OR_RETURN(auto slot,program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(std::span(&binding,1)));
  HYLE_ASSIGN_OR_RETURN(auto done,slot.submit());
  if(accepted)*accepted=true;
  HYLE_ASSIGN_OR_RETURN(auto result,done.wait());
  HYLE_ASSIGN_OR_RETURN(auto encoded,memory.import_readonly(result->arrays()[0],output));
  bytes.resize(*output.required_bytes());
  return encoded.view()->read(std::as_writable_bytes(std::span(bytes)));
}

double q8_value(std::span<const std::uint8_t> bytes,std::size_t index) {
  const auto block=index/32*34;
  const F16 scale{std::uint16_t(bytes[block]|std::uint16_t(bytes[block+1])<<8)};
  const int code=bytes[block+2+index%32]<128?bytes[block+2+index%32]:int(bytes[block+2+index%32])-256;
  return half_test::decode(scale)*code;
}

template<class T>void boxed_case(bool paged,bool quantized,unsigned dk,unsigned dv,
    bool causal,int window,bool long_prefill=false,bool large_values=false,bool bounded_long=false,
    bool small_probability=false) {
  SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" q8="<<quantized<<" dk="<<dk
      <<" dv="<<dv<<" window="<<window<<" long="<<long_prefill<<" large="<<large_values);
  const unsigned qr=large_values?(long_prefill?65:1):(long_prefill?66:(causal?18:68));
  const unsigned physical=large_values?8192:(long_prefill?(paged?256:274):64);
  const auto storage=quantized?c::CacheStorage::q8_0().value():c::CacheStorage::plain(half_test::data_type<T>()).value();
  const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
      .query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),
      .key_storage=storage,.value_storage=storage,.max_requests=large_values?1u:2u,
      .max_query_rows=qr,.max_kv_rows=large_values?physical:(long_prefill?274u:physical),
      .max_pages=paged?(long_prefill?16u:4u):0u,.max_page_refs=paged?(long_prefill?19u:5u):0u,
      .query_heads=8,.kv_heads=2,.qk_head_dim=dk,.value_head_dim=dv,.page_size=paged?16u:1u};
  auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context).value();
  auto staging=n::MemoryResource::create(context).value();
  auto schema=c::BatchSchema::create(cap).value();
  const std::vector<std::int32_t> qp=large_values?std::vector<std::int32_t>{0,int(qr)}:
      (long_prefill?std::vector<std::int32_t>{0,1,66}:(causal?std::vector<std::int32_t>{0,1,18}:std::vector<std::int32_t>{0,33,68}));
  const std::vector<std::int32_t> kp=large_values?std::vector<std::int32_t>{0,8192}:
      (long_prefill?(paged?std::vector<std::int32_t>{0,9,19}:std::vector<std::int32_t>{0,129,274}):
       (paged?std::vector<std::int32_t>{0,2,5}:std::vector<std::int32_t>{0,31,64}));
  const std::vector<std::int32_t> ids=paged?(long_prefill?
      std::vector<std::int32_t>{13,0,15,1,2,3,4,5,6,9,10,11,12,7,8,0,3,4,14}:
      std::vector<std::int32_t>{3,1,0,2,1}):std::vector<std::int32_t>{};
  const std::vector<std::int32_t> tails=paged?(long_prefill?std::vector<std::int32_t>{1,1}:
      std::vector<std::int32_t>{15,1}):std::vector<std::int32_t>{};
  auto layout=schema.version({qp,kp,ids,tails}).value();
  std::vector<T> query(std::size_t(qr)*8*dk);
  std::vector<float> key(std::size_t(physical)*2*dk),value(std::size_t(physical)*2*dv);
  for(std::size_t i=0;i<query.size();++i)query[i]=half_test::encode<T>(large_values?0.F:float(.17*std::sin(double(i)*.23)));
  for(std::size_t i=0;i<key.size();++i)key[i]=large_values?0.F:float(.19*std::cos(double(i)*.13));
  for(std::size_t i=0;i<value.size();++i)value[i]=large_values&&!bounded_long?(i<value.size()/2?65504.F:-65504.F):
      float(.23*std::sin(double(i)*.07)+.05*std::cos(double(i)*.17));
  if(small_probability) {
    ASSERT_TRUE(large_values&&long_prefill&&!paged&&quantized&&dk==64);
    std::fill(query.begin(),query.end(),half_test::encode<T>(1.F));
    std::fill(key.begin(),key.end(),0.F);
    std::fill_n(key.begin(),2*dk,2.F);
    std::fill(value.begin(),value.end(),60000.F);
    std::fill_n(value.begin(),2*dv,0.F);
  }
  std::vector<std::uint8_t> kb,vb;
  const auto cache_bytes=[&](std::span<const float> source,std::vector<std::uint8_t>& output) {
    if(quantized){const auto status=encode_words(context,memory,half_test::data_type<T>(),source,output);
      ASSERT_TRUE(status.ok())<<status.message();}
    else {std::vector<T> plain(source.size());for(std::size_t i=0;i<source.size();++i)plain[i]=half_test::encode<T>(source[i]);
      output.resize(plain.size()*2);std::memcpy(output.data(),plain.data(),output.size());}
  };
  cache_bytes(key,kb);cache_bytes(value,vb);
  ASSERT_EQ(kb.size(),key.size()*(quantized?34u:64u)/32);
  ASSERT_EQ(vb.size(),value.size()*(quantized?34u:64u)/32);
  auto pool=c::LeasePool::create(memory.ref(),query.size()*2+kb.size()+vb.size()+1024).value();
  const auto load=[&](std::span<const std::byte> bytes) {
    auto writer=pool.allocate(bytes.size()).value();EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();
  };
  auto q=load(std::as_bytes(std::span(query)));
  const float scale=1.F/std::sqrt(float(dk));
  c::ResultLease<n::Results> result;
  if(paged) {
    auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=kb.size()+64}).value();
    auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=vb.size()+64}).value();
    std::vector<c::ReadLease> kr(cap.max_pages),vr(cap.max_pages);
    for(unsigned page=0;page<cap.max_pages;++page) {
      auto kw=keys.acquire(page).value(),vw=values.acquire(page).value();
      ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(kb).subspan(page*kb.size()/cap.max_pages,kb.size()/cap.max_pages))).ok());
      ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(vb).subspan(page*vb.size()/cap.max_pages,vb.size()/cap.max_pages))).ok());
      kr[page]=kw.publish(16).value();vr[page]=vw.publish(16).value();
    }
    auto kframe=keys.read(layout,kr).value(),vframe=values.read(layout,vr).value();
    auto made_recipe=c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=8192,.scale=scale,.causal=causal,.window_left=window});
    ASSERT_TRUE(made_recipe)<<made_recipe.error().message();auto recipe=std::move(*made_recipe);
    auto slot=n::PagedBatchInvocation::create(context,recipe,memory,staging,{.max_metadata_backing_bytes=4u<<20}).value();
    const auto prepared=slot.prepare_batch(layout);ASSERT_TRUE(prepared.ok())<<prepared.message();
    ASSERT_TRUE(slot.bind(q,kframe,vframe).ok());auto run=slot.submit();ASSERT_TRUE(run)<<run.error().message();
    auto waited=run->wait();ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
  } else {
    auto kd=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value(),vd=c::batch_cache_description(cap,c::BatchCacheRole::kValue).value();
    const std::array inputs{memory.binding(q,c::batch_query_description(cap).value()).value(),
        memory.binding(load(std::as_bytes(std::span(kb))),kd).value(),
        memory.binding(load(std::as_bytes(std::span(vb))),vd).value()};
    auto program=n::prepare_ragged_batch(context,layout,{cap,scale,causal,window});ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value();ASSERT_TRUE(slot.bind(inputs).ok());
    auto run=slot.submit();ASSERT_TRUE(run)<<run.error().message();auto waited=run->wait();
    ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
  }
  auto description=c::batch_output_description(cap,qr).value();
  auto output=memory.import_readonly(result->arrays()[0],description).value();std::vector<T> actual(std::size_t(qr)*8*dv);
  ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  const auto at=[&](bool k,std::size_t index) {
    if(quantized)return q8_value(k?kb:vb,index);
    return half_test::decode(half_test::encode<T>(k?key[index]:value[index]));
  };
  for(unsigned request=0;request<cap.max_requests;++request) {
    const int queries=qp[request+1]-qp[request],rows=large_values?8192:(long_prefill?(request?145:129):(request?33:31));
    for(int row=0;row<queries;++row)for(unsigned head=0;head<8;++head) {
      const int position=rows-queries+row,first=window<0?0:std::max(0,position-window),last=causal?position+1:rows;
      std::vector<double> weights(static_cast<std::size_t>(rows));double sum=0;
      for(int j=first;j<last;++j) {
        const auto physical_row=paged?unsigned(ids[kp[request]+j/16])*16u+unsigned(j%16):unsigned(kp[request]+j);
        double dot=0;for(unsigned d=0;d<dk;++d)dot+=half_test::decode(query[(std::size_t(qp[request]+row)*8+head)*dk+d])*
            at(true,(std::size_t(physical_row)*2+head/4)*dk+d);
        weights[static_cast<std::size_t>(j)]=std::exp(dot*scale);sum+=weights[static_cast<std::size_t>(j)];
      }
      for(unsigned d=0;d<dv;++d) {
        double expected=0;for(int j=first;j<last;++j) {
          const auto physical_row=paged?unsigned(ids[kp[request]+j/16])*16u+unsigned(j%16):unsigned(kp[request]+j);
          expected+=weights[static_cast<std::size_t>(j)]/sum*at(false,(std::size_t(physical_row)*2+head/4)*dv+d);
        }
        const double decoded=half_test::decode(actual[(std::size_t(qp[request]+row)*8+head)*dv+d]);
        if(small_probability)ASSERT_GT(expected,1.0);
        ASSERT_TRUE(std::isfinite(decoded));EXPECT_NEAR(decoded,expected,(large_values&&!bounded_long?.75:(std::is_same_v<T,BF16>?.008:.0025)));
      }
    }
  }
  output.reset();result={};ASSERT_TRUE(context.close().ok());
}

TEST(AppleCache, Q8PagedRaggedWindowsAndIndependentValueDepth) {
  for(bool paged:{false,true})for(int window:{-1,0,15,16}) {
    boxed_case<F16>(paged,true,64,128,true,window);boxed_case<BF16>(paged,true,64,128,true,window);
  }
}
TEST(AppleCache, FullNoncausalAllowsMoreQueriesThanKeys) {
  for(bool paged:{false,true}) {boxed_case<F16>(paged,true,64,128,false,-1);boxed_case<BF16>(paged,true,64,128,false,-1);}
}
TEST(AppleCache, PlainWindowAndIndependentValueDepth) {
  for(bool paged:{false,true})for(int window:{0,16}) {boxed_case<F16>(paged,false,64,128,true,window);boxed_case<BF16>(paged,false,64,128,true,window);}
}
TEST(AppleCache, OriginalPlainCoupledUnboundedPaths) {
  for(bool paged:{false,true})for(unsigned depth:{64u,128u,256u}) {
    boxed_case<F16>(paged,false,depth,depth,true,-1);
    boxed_case<BF16>(paged,false,depth,depth,true,-1);
  }
}
TEST(AppleCache, MainDepthsAndLongPrefill) {
  for(bool paged:{false,true})for(unsigned d:{64u,128u,256u}) {
    boxed_case<F16>(paged,true,d,d,true,16);boxed_case<BF16>(paged,true,d,d,true,16);
  }
  for(bool paged:{false,true}) {boxed_case<F16>(paged,true,256,256,true,-1,true);boxed_case<BF16>(paged,true,256,256,true,-1,true);}
}
TEST(AppleCache, Q8DecodeAndTwoPassKeepFiniteCancellation) {
  boxed_case<F16>(false,true,64,64,true,-1,false,true);
  boxed_case<F16>(false,true,64,64,true,-1,true,true);
}
TEST(AppleCache, SmallSoftmaxWeightsKeepLargeValueContributions) {
  // exp(-16) is a half subnormal. Rounding each probability before a 60000
  // value would change the accumulated contribution by several output units.
  boxed_case<F16>(false,true,64,64,true,-1,true,true,false,true);
}
TEST(AppleCache, TwoPassWindowPipelinesPreserveTheirSpecialization) {
  for(int window:{-1,0,16,-1}) {
    boxed_case<F16>(false,true,64,64,true,window,false,true,true);
    boxed_case<BF16>(false,true,64,64,true,window,false,true,true);
    boxed_case<F16>(false,false,64,64,true,window,false,true,true);
    boxed_case<BF16>(false,false,64,64,true,window,false,true,true);
  }
}
TEST(AppleCache, EmptyIndependentValueDepthKeepsOutputShape) {
  for(bool paged:{false,true})for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16}) {
    SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" type="<<int(type));
    auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context).value();
    auto staging=n::MemoryResource::create(context).value();
    const auto q8=c::CacheStorage::q8_0().value();
    const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
        .query_type=type,.output_type=type,.key_storage=q8,.value_storage=q8,
        .max_query_rows=1,.max_kv_rows=16,.max_pages=paged?1u:0u,.max_page_refs=paged?1u:0u,
        .query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=128,.page_size=paged?16u:1u};
    auto made_schema=c::BatchSchema::create(cap);ASSERT_TRUE(made_schema)<<made_schema.error().message();auto schema=std::move(*made_schema);
    const std::array<std::int32_t,2> ptr{};
    const std::array<std::int32_t,1> tail{};
    const auto tails=paged?std::span<const std::int32_t>(tail):std::span<const std::int32_t>{};
    auto made_layout=schema.version({ptr,ptr,{},tails});ASSERT_TRUE(made_layout)<<made_layout.error().message();auto layout=std::move(*made_layout);
    auto pool=c::LeasePool::create(memory.ref(),8192).value();
    const auto load=[&](const c::TensorDesc& description) {
      std::vector<std::byte> bytes(*description.required_bytes());auto writer=pool.allocate(bytes.size(),64).value();
      EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();
    };
    const auto qd=c::batch_query_description(cap).value();auto query=load(qd);
    c::ResultLease<n::Results> result;
    if(paged) {
      auto made_recipe=c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=8192,.scale=.125F});
      ASSERT_TRUE(made_recipe)<<made_recipe.error().message();auto recipe=std::move(*made_recipe);
      auto made_slot=n::PagedBatchInvocation::create(context,recipe,memory,staging,{.max_metadata_backing_bytes=4096});
      ASSERT_TRUE(made_slot)<<made_slot.error().message();auto slot=std::move(*made_slot);
      ASSERT_TRUE(slot.prepare_batch(layout).ok());ASSERT_TRUE(slot.bind(query,{},{}).ok());
      auto submitted=slot.submit();ASSERT_TRUE(submitted)<<submitted.error().message();
      auto waited=submitted->wait();ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
    } else {
      const auto kd=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value(),vd=c::batch_cache_description(cap,c::BatchCacheRole::kValue).value();
      const std::array inputs{memory.binding(query,qd).value(),memory.binding(load(kd),kd).value(),memory.binding(load(vd),vd).value()};
      auto program=n::prepare_ragged_batch(context,layout,{cap,.125F,true});ASSERT_TRUE(program)<<program.error().message();
      auto slot=program->create_invocation().value();ASSERT_TRUE(slot.bind(inputs).ok());
      auto submitted=slot.submit();ASSERT_TRUE(submitted)<<submitted.error().message();
      auto waited=submitted->wait();ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
    }
    EXPECT_EQ(result->arrays()[0].shape(),(mlx::core::Shape{0,8,128}));
    result={};ASSERT_TRUE(context.close().ok());
  }
}
TEST(AppleCacheEncode, IndependentBlockBytesForThreeInputs) {
  auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context).value();
  std::vector<float> source(512);for(std::size_t i=0;i<source.size();++i)source[i]=float(int(i%67)-33)*.0625F;
  std::fill_n(source.begin(),32,0.F);
  source[32]=127.F;
  source[33]=std::nextafter(.5F,0.F);source[34]=-source[33];
  source[35]=.5F;source[36]=-.5F;
  source[37]=std::nextafter(.5F,1.F);source[38]=-source[37];
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32}) {
    std::vector<float> values=source;
    for(auto& x:values)if(type!=c::DataType::kFloat32)x=float(type==c::DataType::kFloat16?
        half_test::decode(half_test::encode<F16>(x)):half_test::decode(half_test::encode<BF16>(x)));
    std::vector<std::uint8_t> bytes;const auto status=encode_words(context,memory,type,source,bytes);ASSERT_TRUE(status.ok())<<status.message();
    ASSERT_EQ(bytes.size(),16u*34);
    for(std::size_t block=0;block<16;++block) {
      float maximum=0;for(unsigned j=0;j<32;++j)maximum=std::max(maximum,std::abs(values[block*32+j]));
      const float scale=maximum/127,inv=scale?1/scale:0;const auto bits=half_test::encode<F16>(scale).bits;
      EXPECT_EQ(bytes[block*34],bits&255);EXPECT_EQ(bytes[block*34+1],bits>>8);
      for(unsigned j=0;j<32;++j)EXPECT_EQ(bytes[block*34+2+j],std::uint8_t(int(std::round(values[block*32+j]*inv))));
    }
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(AppleCacheEncode, FiniteQ8ScaleBoundariesHaveCanonicalZeroBlocks){
  auto context=n::Context::create().value();
  struct Close final {n::Context &context;~Close(){context.close().IgnoreError();}}close{context};
  auto memory=n::MemoryResource::create(context).value();
  using Half=hyle::simd::float16_t;using BFloat=hyle::simd::bfloat16_t;
  const float zero_boundary=127.F*std::ldexp(1.F,-25),minimum_scale=127.F*std::ldexp(1.F,-24);
  const std::array amplitudes{0.F,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),
      std::ldexp(1.F,-122),std::ldexp(1.F,-120),std::ldexp(1.F,-24),
      std::nextafter(zero_boundary,0.F),zero_boundary,std::nextafter(zero_boundary,1.F),
      std::nextafter(minimum_scale,0.F),minimum_scale,std::nextafter(minimum_scale,1.F),
      std::ldexp(1.F,-14),65504.F};
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32}){
    SCOPED_TRACE(::testing::Message()<<"input type="<<int(type));
    const auto rows=amplitudes.size()+(type==c::DataType::kFloat32?1:0);
    std::vector<float> values(rows*64);
    for(std::size_t i=0;i<values.size();++i){const float a=i/64<amplitudes.size()?amplitudes[i/64]:65504.F*127.F;
      values[i]=i%8==0?a:i%8==1?-a:i%8==2?a*.125F:i%8==3?-a*.125F:i%8==5?-0.F:0.F;}
    const auto rounded=[&](float value){
      if(type==c::DataType::kFloat16)return float(half_test::decode(half_test::encode<Half>(value)));
      if(type==c::DataType::kBFloat16)return float(half_test::decode(half_test::encode<BFloat>(value)));
      return value;
    };
    std::vector<std::uint8_t> bytes;
    const auto status=encode_words(context,memory,type,values,bytes);
    ASSERT_TRUE(status.ok())<<status.message();ASSERT_EQ(bytes.size(),rows*68);
    std::vector<std::uint8_t> expected(bytes.size(),0);
    unsigned zeros=0,nonzero=0;
    for(std::size_t block=0;block<values.size()/32;++block){
      float maximum=0;for(unsigned j=0;j<32;++j)maximum=std::max(maximum,std::abs(rounded(values[block*32+j])));
      const auto scale=maximum/127.F;const auto stored=half_test::encode<Half>(scale).bits;
      if(!stored){++zeros;continue;}
      ++nonzero;expected[block*34]=std::uint8_t(stored);expected[block*34+1]=std::uint8_t(stored>>8);
      const auto inverse=1.F/scale;ASSERT_TRUE(std::isfinite(inverse));
      for(unsigned j=0;j<32;++j){
        const auto code=int(std::round(rounded(values[block*32+j])*inverse));
        ASSERT_GE(code,-127);ASSERT_LE(code,127);expected[block*34+2+j]=std::uint8_t(code);
      }
    }
    ASSERT_GT(zeros,0u);ASSERT_GT(nonzero,0u);
    for(std::size_t i=0;i<bytes.size();++i)EXPECT_EQ(bytes[i],expected[i])<<"block="<<i/34<<" byte="<<i%34;
  }
}

TEST(AppleCacheEncode, AcceptedNumericFailureUsesOriginalResultsStatus) {
  auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context).value();
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32})
    for(float bad:{NAN,INFINITY}) {
      std::vector<float> values(64,1.F);values[19]=bad;std::vector<std::uint8_t> bytes;bool accepted=false;
      const auto status=encode_words(context,memory,type,values,bytes,&accepted);
      EXPECT_TRUE(accepted);EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
    }
  std::vector<float> huge(64,std::numeric_limits<float>::max());std::vector<std::uint8_t> bytes;bool accepted=false;
  const auto overflow=encode_words(context,memory,c::DataType::kFloat32,huge,bytes,&accepted);
  EXPECT_TRUE(accepted);EXPECT_TRUE(overflow.IsInvalidArgument())<<overflow.message();
  ASSERT_TRUE(context.close().ok());
}
} // namespace
