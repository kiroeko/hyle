#include <gtest/gtest.h>
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
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
namespace n=c::vulkan;
using F16=hyle::simd::float16_t;
using BF16=hyle::simd::bfloat16_t;

hyle::Status encode_words(n::Context& context,[[maybe_unused]] n::MemoryResource& memory,
    c::DataType type,std::span<const float> values,std::vector<std::uint8_t>& bytes,
    bool* accepted=nullptr) {
  if(accepted)*accepted=false;
  HYLE_ASSIGN_OR_RETURN(auto input,c::TensorDesc::dense(type,
      std::array<std::int64_t,2>{std::int64_t(values.size()/64),64}));
  const auto operation=c::Operation::custom(c::model_ops::CacheEncode{c::CacheStorage::q8_0().value()});
  HYLE_ASSIGN_OR_RETURN(auto output,c::infer_single_output(operation,std::array{input}));
  n::Tensor tensor;
  if(type==c::DataType::kFloat32){HYLE_ASSIGN_OR_RETURN(tensor,context.upload(input,std::as_bytes(values)));}
  else {
    std::vector<std::uint16_t> words(values.size());
    for(std::size_t i=0;i<values.size();++i)words[i]=type==c::DataType::kFloat16?
        half_test::encode<F16>(values[i]).bits:half_test::encode<BF16>(values[i]).bits;
    HYLE_ASSIGN_OR_RETURN(tensor,context.upload(input,std::as_bytes(std::span(words))));
  }
  HYLE_ASSIGN_OR_RETURN(auto program,n::prepare(context,operation,input));
  HYLE_ASSIGN_OR_RETURN(auto slot,program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(std::span(&tensor,1)));
  HYLE_ASSIGN_OR_RETURN(auto done,slot.submit());if(accepted)*accepted=true;
  HYLE_ASSIGN_OR_RETURN(auto result,done.wait());bytes.resize(*output.required_bytes());
  return context.download(result->tensors()[0],std::as_writable_bytes(std::span(bytes)));
}

double q8_value(std::span<const std::uint8_t> bytes,std::size_t index) {
  const auto block=index/32*34;
  const F16 scale{std::uint16_t(bytes[block]|std::uint16_t(bytes[block+1])<<8)};
  const int code=bytes[block+2+index%32]<128?bytes[block+2+index%32]:int(bytes[block+2+index%32])-256;
  return half_test::decode(scale)*code;
}

template<class T>void boxed_case(bool paged,bool quantized,unsigned dk,unsigned dv,
    bool causal,int window,bool long_prefill=false,bool large_values=false,bool bounded_long=false,
    unsigned replays=1,bool resident_history=false) {
  SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" q8="<<quantized<<" dk="<<dk
      <<" dv="<<dv<<" window="<<window<<" long="<<long_prefill<<" large="<<large_values);
  const unsigned qr=large_values?(long_prefill?65:1):(long_prefill?66:(causal?18:68));
  const unsigned physical=resident_history?1856:(large_values?8192:(long_prefill?(paged?256:274):64));
  const auto storage=quantized?c::CacheStorage::q8_0().value():c::CacheStorage::plain(half_test::data_type<T>()).value();
  const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
      .query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),
      .key_storage=storage,.value_storage=storage,.max_requests=large_values?1u:2u,
      .max_query_rows=qr,.max_kv_rows=resident_history?1792u:(large_values?physical:(long_prefill?274u:physical)),
      .max_pages=paged?(resident_history?116u:(long_prefill?16u:4u)):0u,
      .max_page_refs=paged?(resident_history?112u:(long_prefill?19u:5u)):0u,
      .query_heads=8,.kv_heads=2,.qk_head_dim=dk,.value_head_dim=dv,.page_size=paged?16u:1u};
  auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
  auto staging=n::MemoryResource::create(context,n::MemoryKind::kStaging).value();
  auto schema=c::BatchSchema::create(cap).value();
  const std::vector<std::int32_t> qp=large_values?std::vector<std::int32_t>{0,int(qr)}:
      (long_prefill?std::vector<std::int32_t>{0,1,66}:(causal?std::vector<std::int32_t>{0,1,18}:std::vector<std::int32_t>{0,33,68}));
  const std::vector<std::int32_t> kp=resident_history?std::vector<std::int32_t>{0,64,112}:
      (large_values?std::vector<std::int32_t>{0,8192}:
      (long_prefill?(paged?std::vector<std::int32_t>{0,9,19}:std::vector<std::int32_t>{0,129,274}):
       (paged?std::vector<std::int32_t>{0,2,5}:std::vector<std::int32_t>{0,31,64})));
  std::vector<std::int32_t> ids=paged?(long_prefill?
      std::vector<std::int32_t>{13,0,15,1,2,3,4,5,6,9,10,11,12,7,8,0,3,4,14}:
      std::vector<std::int32_t>{3,1,0,2,1}):std::vector<std::int32_t>{};
  if(resident_history){ids.resize(112);for(unsigned i=0;i<ids.size();++i)ids[i]=111-int(i);}
  const std::vector<std::int32_t> tails=paged?(resident_history?std::vector<std::int32_t>{16,16}:
      (long_prefill?std::vector<std::int32_t>{1,1}:std::vector<std::int32_t>{15,1})):std::vector<std::int32_t>{};
  auto layout=schema.version({qp,kp,ids,tails}).value();
  std::vector<T> query(std::size_t(qr)*8*dk);
  std::vector<float> key(std::size_t(physical)*2*dk),value(std::size_t(physical)*2*dv);
  for(std::size_t i=0;i<query.size();++i)query[i]=half_test::encode<T>(large_values?0.F:float(.17*std::sin(double(i)*.23)));
  for(std::size_t i=0;i<key.size();++i)key[i]=large_values?0.F:float(.19*std::cos(double(i)*.13));
  for(std::size_t i=0;i<value.size();++i)value[i]=large_values&&!bounded_long?(i<value.size()/2?65504.F:-65504.F):
      float(.23*std::sin(double(i)*.07)+.05*std::cos(double(i)*.17));
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
  ASSERT_FALSE(context.environment().software_device);
  const auto qd=c::batch_query_description(cap).value();
  auto query_tensor=context.upload(qd,std::as_bytes(std::span(query))).value();
  auto q=memory.import_readonly(query_tensor).value();
  const float scale=1.F/std::sqrt(float(dk));
  c::ResultLease<n::Results> result;
  if(paged) {
    auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=kb.size()+64}).value();
    auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=vb.size()+64}).value();
    const unsigned ready_pages=resident_history?112u:cap.max_pages;
    std::vector<c::ReadLease> kr(ready_pages),vr(ready_pages);
    const auto maximum=std::max(kb.size(),vb.size())/cap.max_pages;
    auto host=c::LeasePool::create(staging.ref(),maximum+64).value();
    const std::array specs{c::CopyResourceSpec{staging.ref(),maximum},c::CopyResourceSpec{memory.ref(),maximum}};
    auto copy=n::CopyBatchProgram::prepare(context,specs).value();auto copier=n::CopyBatchInvocation::create(copy).value();
    const auto fill=[&](c::PagedArena& arena,std::span<const std::uint8_t> payload,unsigned page)->hyle::Result<c::ReadLease>{
      const auto size=payload.size()/cap.max_pages;
      HYLE_ASSIGN_OR_RETURN(auto target,arena.acquire(page));HYLE_RETURN_IF_ERROR(copier.reset());
      HYLE_ASSIGN_OR_RETURN(auto writer,host.allocate(size,64));
      HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(payload.subspan(page*size,size))));
      HYLE_ASSIGN_OR_RETURN(auto source,writer.publish());
      HYLE_RETURN_IF_ERROR(copier.bind_source({0},source));HYLE_RETURN_IF_ERROR(copier.bind_mutable({1},std::move(target.lease())));
      const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,size}}};
      HYLE_RETURN_IF_ERROR(copier.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,copier.submit());
      HYLE_RETURN_IF_ERROR(submitted.wait());HYLE_ASSIGN_OR_RETURN(auto ready,submitted.output({0},{1},0,size));
      HYLE_RETURN_IF_ERROR(target.complete(ready,16));return ready;
    };
    for(unsigned page=0;page<ready_pages;++page){auto k=fill(keys,kb,page),v=fill(values,vb,page);
      ASSERT_TRUE(k)<<k.error().message();ASSERT_TRUE(v)<<v.error().message();kr[page]=std::move(*k);vr[page]=std::move(*v);}
    auto kframe=keys.read(layout,kr).value(),vframe=values.read(layout,vr).value();
    auto made_recipe=c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=8192,.scale=scale,.causal=causal,.window_left=window});
    ASSERT_TRUE(made_recipe)<<made_recipe.error().message();auto recipe=std::move(*made_recipe);
    auto slot=n::PagedBatchInvocation::create(context,recipe,memory,staging,{.max_metadata_backing_bytes=4u<<20}).value();
    for(unsigned replay=0;replay<replays;++replay){
      SCOPED_TRACE(::testing::Message()<<"replay="<<replay);
      result={};
      const auto prepared=slot.prepare_batch(layout);ASSERT_TRUE(prepared.ok())<<prepared.message();
      EXPECT_EQ(slot.report().reused_native_plan,replay!=0);
      ASSERT_TRUE(slot.bind(q,kframe,vframe).ok());auto run=slot.submit();ASSERT_TRUE(run)<<run.error().message();
      auto waited=run->wait();ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
    }
  } else {
    auto kd=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value(),vd=c::batch_cache_description(cap,c::BatchCacheRole::kValue).value();
    const std::array inputs{query_tensor,context.upload(kd,std::as_bytes(std::span(kb))).value(),
        context.upload(vd,std::as_bytes(std::span(vb))).value()};
    auto program=n::prepare_ragged_batch(context,layout,{cap,scale,causal,window});ASSERT_TRUE(program)<<program.error().message();
    auto slot=program->create_invocation().value();ASSERT_TRUE(slot.bind(inputs).ok());
    auto run=slot.submit();ASSERT_TRUE(run)<<run.error().message();auto waited=run->wait();
    ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
  }
  auto description=c::batch_output_description(cap,qr).value();
  ASSERT_EQ(result->tensors()[0].description(),description);
  std::vector<T> actual(std::size_t(qr)*8*dv);
  ASSERT_TRUE(context.download(result->tensors()[0],std::as_writable_bytes(std::span(actual))).ok());
  const auto at=[&](bool k,std::size_t index) {
    if(quantized)return q8_value(k?kb:vb,index);
    return half_test::decode(half_test::encode<T>(k?key[index]:value[index]));
  };
  for(unsigned request=0;request<cap.max_requests;++request) {
    const int queries=qp[request+1]-qp[request],rows=resident_history?(request?768:1024):
        (large_values?8192:(long_prefill?(request?145:129):(request?33:31)));
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
        ASSERT_TRUE(std::isfinite(decoded));EXPECT_NEAR(decoded,expected,(large_values&&!bounded_long?.75:(std::is_same_v<T,BF16>?.008:.0025)));
      }
    }
  }
  result={};ASSERT_TRUE(context.close().ok());
}

TEST(VulkanCache, Q8PagedRaggedWindowsAndIndependentValueDepth) {
  for(bool paged:{false,true})for(int window:{-1,0,15,16,std::numeric_limits<int>::max()}) {
    boxed_case<F16>(paged,true,64,128,true,window);
  }
}
TEST(VulkanCache, FullNoncausalAllowsMoreQueriesThanKeys) {
  for(bool paged:{false,true}) {boxed_case<F16>(paged,true,64,128,false,-1);}
}
TEST(VulkanCache, PlainWindowAndIndependentValueDepth) {
  for(bool paged:{false,true})for(int window:{0,16}) {boxed_case<F16>(paged,false,64,128,true,window);}
}
TEST(VulkanCache, OriginalPlainCoupledUnboundedPaths) {
  for(bool paged:{false,true})for(unsigned depth:{64u,128u,256u}) {
    boxed_case<F16>(paged,false,depth,depth,true,-1);

  }
}
TEST(VulkanCache, PlainPagedReusesPreparedNativeSlot) {
  boxed_case<F16>(true,false,256,256,true,-1,false,false,false,3,true);
}
TEST(VulkanCache, MainDepthsAndLongPrefill) {
  for(bool paged:{false,true})for(unsigned d:{64u,128u,256u}) {
    boxed_case<F16>(paged,true,d,d,true,16);
  }
  for(bool paged:{false,true}) {boxed_case<F16>(paged,true,256,256,true,-1,true);}
}
TEST(VulkanCache, Q8DecodeAndSplitKeepFiniteCancellation) {
  boxed_case<F16>(false,true,64,64,true,-1,false,true);
  boxed_case<F16>(false,true,64,64,true,-1,true,true);
}
TEST(VulkanCache, LongWindowPipelinesPreserveTheirSpecialization) {
  for(int window:{-1,0,16,-1}) {
    boxed_case<F16>(false,true,64,64,true,window,false,true,true);

    boxed_case<F16>(false,false,64,64,true,window,false,true,true);

  }
}
TEST(VulkanCache, EmptyIndependentValueDepthKeepsOutputShape) {
  for(bool paged:{false,true})for(auto type:{c::DataType::kFloat16}) {
    SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" type="<<int(type));
    auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
    auto staging=n::MemoryResource::create(context,n::MemoryKind::kStaging).value();
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
    const auto qd=c::batch_query_description(cap).value();std::vector<std::byte> zeros(*qd.required_bytes());
    auto query_tensor=context.upload(qd,zeros).value();auto query=memory.import_readonly(query_tensor).value();
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
      const std::array inputs{query_tensor,context.upload(kd,std::vector<std::byte>(*kd.required_bytes())).value(),
          context.upload(vd,std::vector<std::byte>(*vd.required_bytes())).value()};
      auto program=n::prepare_ragged_batch(context,layout,{cap,.125F,true});ASSERT_TRUE(program)<<program.error().message();
      auto slot=program->create_invocation().value();ASSERT_TRUE(slot.bind(inputs).ok());
      auto submitted=slot.submit();ASSERT_TRUE(submitted)<<submitted.error().message();
      auto waited=submitted->wait();ASSERT_TRUE(waited)<<waited.error().message();result=std::move(*waited);
    }
    EXPECT_TRUE(std::ranges::equal(result->tensors()[0].description().dimensions(),std::array<std::int64_t,3>{0,8,128}));
    result={};ASSERT_TRUE(context.close().ok());
  }
}
TEST(VulkanCacheEncode, IndependentBlockBytesForThreeInputs) {
  auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
  std::vector<float> source(4096);for(std::size_t i=0;i<source.size();++i)source[i]=float(int(i%67)-33)*.0625F;
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
    ASSERT_EQ(bytes.size(),source.size()/32*34);
    for(std::size_t block=0;block<source.size()/32;++block) {
      SCOPED_TRACE(::testing::Message()<<"type="<<int(type)<<" block="<<block);
      float maximum=0;for(unsigned j=0;j<32;++j)maximum=std::max(maximum,std::abs(values[block*32+j]));
      const float scale=maximum/127,inv=scale?1/scale:0;const auto bits=half_test::encode<F16>(scale).bits;
      EXPECT_EQ(bytes[block*34],bits&255);EXPECT_EQ(bytes[block*34+1],bits>>8);
      for(unsigned j=0;j<32;++j)EXPECT_EQ(bytes[block*34+2+j],std::uint8_t(int(std::round(values[block*32+j]*inv))));
    }
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(VulkanCacheEncode, FiniteQ8ScaleBoundariesHaveCanonicalZeroBlocks){
  auto context=n::Context::create().value();
  struct Close final {n::Context &context;~Close(){context.close().IgnoreError();}}close{context};
  auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
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

TEST(VulkanCacheEncode, AcceptedNumericFailureUsesOriginalResultsStatus) {
  auto context=n::Context::create().value();auto memory=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
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
