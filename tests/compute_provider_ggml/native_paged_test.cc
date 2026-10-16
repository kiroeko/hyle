#include <gtest/gtest.h>
#include <hyle/compute/cpu.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/detail/batch_graph_access.h>
#include <hyle/compute/paged_arena.h>
#include <hyle/compute/model_ops.h>
#include <hyle/compute/prepared_operation.h>
#include <hyle/compute/providers/ggml_cpu.h>
#include "../compute_provider_native/half_test_support.h"
#include "../compute_batch/allocation_fault_fixture.h"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <future>
#include <limits>
#include <vector>

namespace {
namespace c=hyle::compute;
template<class T>void runtime_q1_case(bool encoded,unsigned depth,unsigned value_depth,int window,unsigned requested){
  hyle_f5_fault_fixture::Allocator fault;
  {
  const auto type=half_test::data_type<T>();const bool causal=depth!=128;
  const auto storage=encoded?c::CacheStorage::q8_0().value():c::CacheStorage::plain(type).value();
  c::BatchCapacity cap{.storage=c::BatchStorage::kPaged,.query_type=type,.output_type=type,.key_storage=storage,.value_storage=storage,
    .max_requests=3,.max_query_rows=2,.max_kv_rows=2048,.max_pages=5,.max_page_refs=128,
    .query_heads=8,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=value_depth,.page_size=16};
  auto schema=c::BatchSchema::create(cap).value();auto context=c::cpu::create_context({.native_threads=2}).value();auto queue=context.create_queue().value();
  auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,hyle::mem::system_resource(),{.max_backing_bytes=1u<<20}).value();
  auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,hyle::mem::system_resource(),{.max_backing_bytes=1u<<20}).value();
  std::array<c::ReadLease,5> key_pages,value_pages;
  const auto code=[](unsigned page,unsigned row,unsigned head){return int((page*17+row*3+head*5)%63)-31;};
  for(unsigned page=0;page<5;++page)for(unsigned role=0;role<2;++role){
    auto&arena=role?values:keys;auto writer=arena.acquire(page).value();const auto width=role?value_depth:depth;
    if(encoded){
      std::vector<std::byte> bytes(arena.page_bytes(),std::byte{0});
      for(unsigned row=0;row<16;++row)for(unsigned head=0;head<2;++head)for(unsigned block=0;block<width/32;++block){
        const auto offset=((row*2+head)*(width/32)+block)*34;
        const std::uint16_t scale=page==4?0x7e00:role?0x2400:0;std::memcpy(bytes.data()+offset,&scale,2);
        const auto value=static_cast<std::uint8_t>(page==4?1:role?code(page,row,head):0);
        std::fill_n(bytes.begin()+offset+2,32,std::byte{value});
      }
      ASSERT_TRUE(writer.lease().view()->write(bytes).ok());
    }else{
      std::vector<T> words(16*2*width);
      for(unsigned row=0;row<16;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<width;++d)
        words[(row*2+head)*width+d]=half_test::encode<T>(page==4?std::numeric_limits<float>::quiet_NaN():role?float(code(page,row,head))/64.F:0.F);
      ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(words))).ok());
    }
    (role?value_pages:key_pages)[page]=writer.publish(16).value();
  }
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),2*8*depth*2+64).value();auto qw=pool.allocate(2*8*depth*2).value();
  ASSERT_TRUE(qw.view()->fill(std::byte{0}).ok());auto query=qw.publish().value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=8u<<20,
    .max_int_workspace_bytes=4096,.scale=.125F,.causal=causal,.window_left=window,.allocator=fault.ref()}).value();
  auto slot=c::BatchInvocation::create(program,queue).value();std::vector<std::int32_t> previous;
  c::BatchLayoutVersion initial;std::size_t prior_workspace=0;
  for(unsigned step=0;step<64;++step){
    SCOPED_TRACE(::testing::Message()<<"encoded="<<encoded<<" d="<<depth<<" dv="<<value_depth<<" requests="<<requested<<" step="<<step);
    unsigned count=requested;if(requested==2 && step>=16 && step<24)count=1;
    if(requested==2 && step>=32 && step<48)count=3;
    std::vector<std::int32_t> q{0},k{0},ids,tails;unsigned active=0;
    for(unsigned request=0;request<count;++request){
      const bool live=count!=3 || request!=1;q.push_back(q.back()+(live?1:0));
      if(live){const unsigned rows=1+(step*(17+active*6)+active*509)%1024,refs=(rows+15)/16;
        for(unsigned i=0;i<refs;++i)ids.push_back(std::int32_t((i+step+active*2)%4));
        tails.push_back(std::int32_t((rows-1)%16+1));++active;
      }else tails.push_back(0);
      k.push_back(std::int32_t(ids.size()));
    }
    auto version=schema.version({q,k,ids,tails}).value();if(step==0)initial=version;
    std::vector<c::ReadLease> kr,vr;
    for(unsigned page=0;page<4;++page)if(std::find(ids.begin(),ids.end(),std::int32_t(page))!=ids.end()){
      kr.push_back(key_pages[page]);vr.push_back(value_pages[page]);
    }
    auto key_frame=keys.read(version,kr).value(),value_frame=values.read(version,vr).value();
    const bool reuse=!previous.empty() && previous==q;const auto before=fault.attempts;if(reuse)fault.fail=before;
    auto prepared=slot.update_batch(version);fault.fail=SIZE_MAX;ASSERT_TRUE(prepared)<<prepared.error().message();
    EXPECT_EQ(slot.report().reused_native_plan,reuse);
    if(reuse){EXPECT_EQ(fault.attempts,before);EXPECT_EQ(slot.report().float_workspace_bytes,prior_workspace);EXPECT_EQ(slot.report().host_waits,1u);}
    prior_workspace=slot.report().float_workspace_bytes;previous=q;
    auto done=prepared->run(query,key_frame,value_frame);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    auto output=slot.output().value();*done={};*prepared={};
    EXPECT_FALSE(slot.update_batch(version)); // The original owning output still guards reuse.
    std::vector<T> actual(2*8*value_depth);ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    for(unsigned request=0;request<count;++request){if(q[request]==q[request+1])continue;
      const unsigned rows=unsigned(k[request+1]-k[request]-1)*16+unsigned(tails[request]);
      const unsigned first=window<0?0:rows>unsigned(window+1)?rows-unsigned(window+1):0;
      for(unsigned head=0;head<8;++head){double expected=0;
        for(unsigned row=first;row<rows;++row)expected+=double(code(unsigned(ids[k[request]+row/16]),row%16,head/4))/64.;
        expected/=double(rows-first);
        for(unsigned d=0;d<value_depth;++d)EXPECT_NEAR(half_test::decode(actual[(q[request]*8+head)*value_depth+d]),expected,type==c::DataType::kFloat16?.002:.008);
      }
    }
    auto untouched=keys.acquire(key_pages[4]);ASSERT_TRUE(untouched)<<untouched.error().message();key_pages[4]=untouched->publish(16).value();
  }
  EXPECT_EQ(initial.host().last_page_lengths[0],1);ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
  }
  EXPECT_EQ(fault.live,0u);
}
TEST(GgmlCpuPaged, RuntimeQ1UsesOriginalDecodeSlotAcrossPagesAndTails){
  for(unsigned requests:{1U,2U})for(bool encoded:{false,true})for(unsigned d:{64U,128U,256U}){
    runtime_q1_case<hyle::simd::float16_t>(encoded,d,d,-1,requests);runtime_q1_case<hyle::simd::bfloat16_t>(encoded,d,d,-1,requests);
  }
  for(unsigned requests:{1U,2U})for(bool encoded:{false,true}){
    runtime_q1_case<hyle::simd::float16_t>(encoded,64,128,16,requests);runtime_q1_case<hyle::simd::bfloat16_t>(encoded,64,128,16,requests);
  }
}
hyle::Status encode_words(c::DataType type,c::CacheStorage storage,std::span<const float> values,
                          std::vector<std::uint8_t>& bytes,unsigned threads=3){
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=threads})).value();
  auto queue=context.create_queue().value();
  const auto input=c::TensorDesc::dense(type,std::array<std::int64_t,2>{std::int64_t(values.size()/64),64}).value();
  const auto operation=c::Operation::custom(c::model_ops::CacheEncode{storage});
  const auto output=c::infer_single_output(operation,std::array{input}).value();
  auto source=hyle::mem::Buffer::create(hyle::mem::system_resource(),input.required_bytes().value(),64).value();
  auto destination=hyle::mem::Buffer::create(hyle::mem::system_resource(),output.required_bytes().value(),64).value();
  if(type==c::DataType::kFloat32)HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(values)));
  else{
    std::vector<std::uint16_t> words(values.size());
    for(std::size_t i=0;i<values.size();++i)words[i]=type==c::DataType::kFloat16?
        hyle::simd::f32_to_f16(values[i]).bits:hyle::simd::f32_to_bf16(values[i]).bits;
    HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(words))));
  }
  const std::array inputs{c::ConstTensorView::create(input,source.view()).value()};
  const std::array outputs{c::TensorView::create(output,destination.view()).value()};
  HYLE_ASSIGN_OR_RETURN(auto prepared,c::prepare(context,operation,inputs,outputs));
  EXPECT_EQ(prepared.provider().id,c::ggml_cpu::kProviderId);
  HYLE_ASSIGN_OR_RETURN(auto invocation,prepared.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queue,inputs,outputs));
  HYLE_RETURN_IF_ERROR(done.wait());
  bytes.resize(output.required_bytes().value());
  return destination.view().read(std::as_writable_bytes(std::span(bytes)));
}
float input_value(float value,c::DataType type){
  if(type==c::DataType::kFloat16)return hyle::simd::f16_to_f32(hyle::simd::f32_to_f16(value));
  if(type==c::DataType::kBFloat16)return hyle::simd::bf16_to_f32(hyle::simd::f32_to_bf16(value));
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
TEST(GgmlCpuCacheEncode, Q8ReferenceBytesTiesZeroAndThreeInputTypes){
  std::vector<float> values(256);
  for(std::size_t i=0;i<values.size();++i)values[i]=float(int(i%23)-11)*.03125F;
  values[0]=127.F;values[1]=.5F;values[2]=1.5F;values[3]=-.5F;values[4]=-1.5F;
  std::fill(values.begin()+32,values.begin()+64,0.F);
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32}){
    std::vector<std::uint8_t> bytes;
    ASSERT_TRUE(encode_words(type,c::CacheStorage::q8_0().value(),values,bytes).ok());
    ASSERT_EQ(bytes.size(),8u*34);
    for(unsigned block=0;block<8;++block){
      float maximum=0;
      for(unsigned j=0;j<32;++j)maximum=std::max(maximum,std::abs(input_value(values[block*32+j],type)));
      const auto scale=maximum/127.F;const auto half=hyle::simd::f32_to_f16(scale).bits;
      std::uint16_t actual_scale;std::memcpy(&actual_scale,bytes.data()+block*34,2);EXPECT_EQ(actual_scale,half);
      for(unsigned j=0;j<32;++j){
        const auto code=scale?std::round(input_value(values[block*32+j],type)*(1.F/scale)):0.F;
        EXPECT_EQ(static_cast<std::int8_t>(bytes[block*34+2+j]),static_cast<std::int8_t>(code));
      }
    }
    EXPECT_EQ(bytes[3],1u);EXPECT_EQ(bytes[4],2u);EXPECT_EQ(static_cast<std::int8_t>(bytes[5]),-1);
  }
}
TEST(GgmlCpuCacheEncode, Float8CodesMatchIndependentEnumerationIncludingRoundingAndSaturation){
  std::vector<float> values(512);
  for(std::size_t i=0;i<values.size();++i)values[i]=float(int(i%67)-33)*.0625F;
  values[0]=-0.F;values[1]=.001953125F;values[2]=.00390625F;values[3]=.125F;values[4]=.375F;values[5]=1000.F;values[6]=-1000.F;
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32})for(float scale:{.25F,2.F}){
    std::vector<std::uint8_t> bytes;
    ASSERT_TRUE(encode_words(type,c::CacheStorage::float8(scale).value(),values,bytes).ok());
    ASSERT_EQ(bytes.size(),values.size());
    for(std::size_t i=0;i<values.size();++i)EXPECT_EQ(bytes[i],float8_oracle(double(input_value(values[i],type))/scale));
  }
}
TEST(GgmlCpuCacheEncode, FiniteQ8ScaleBoundariesHaveCanonicalZeroBlocks){
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
    const auto status=encode_words(type,c::CacheStorage::q8_0().value(),values,bytes);
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

TEST(GgmlCpuCacheEncode, FiniteFloat8ExtremePositiveScalesSaturateAndPreserveZero){
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
TEST(GgmlCpuCacheEncode, InvalidPayloadReturnsOriginalCompletionErrorAfterWorkersRetire){
  for(auto storage:{c::CacheStorage::q8_0().value(),c::CacheStorage::float8(.5F).value()})
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
      std::vector<float> values(64,1.F);values[19]=bad;std::vector<std::uint8_t> bytes;
      const auto status=encode_words(c::DataType::kFloat32,storage,values,bytes);
      EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
    }
  std::vector<float> overflow(64,9000000.F);std::vector<std::uint8_t> bytes;
  EXPECT_TRUE(encode_words(c::DataType::kFloat32,c::CacheStorage::q8_0().value(),overflow,bytes).IsInvalidArgument());
}
TEST(GgmlCpuCacheEncode, IndependentInvocationsAndFreshRecoveryAfterOriginalWorkersRetire){
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=3})).value();
  auto queue=context.create_queue().value();
  const auto input=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{2,64}).value();
  const auto operation=c::Operation::custom(c::model_ops::CacheEncode{c::CacheStorage::float8(.5F).value()});
  const auto output=c::infer_single_output(operation,std::array{input}).value();
  auto source=hyle::mem::Buffer::create(hyle::mem::system_resource(),512,64).value();
  auto destination=hyle::mem::Buffer::create(hyle::mem::system_resource(),128,64).value();
  const std::array inputs{c::ConstTensorView::create(input,source.view()).value()};
  const std::array outputs{c::TensorView::create(output,destination.view()).value()};
  auto prepared=c::prepare(context,operation,inputs,outputs).value();
  auto first=prepared.create_invocation().value(),second=prepared.create_invocation().value();
  for(unsigned episode=0;episode<8;++episode){
    std::vector<float> values(128,float(episode)*.125F);const bool bad=episode==0;
    if(bad)values[19]=std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
    auto work=(episode%2?second:first).submit(queue,inputs,outputs);ASSERT_TRUE(work)<<work.error().message();
    const auto status=work->wait();
    if(bad){EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
      auto refused=first.submit(queue,inputs,outputs);EXPECT_FALSE(refused);
      *work={};first=prepared.create_invocation().value();}
    else{ASSERT_TRUE(status.ok())<<status.message();std::array<std::uint8_t,128> actual;
      ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(actual))).ok());
      for(auto value:actual)EXPECT_EQ(value,float8_oracle(double(values[0])/.5));}
  }
  ASSERT_TRUE(context.drain().ok());
}

TEST(GgmlCpuCacheEncode, ConcurrentOriginalLanesKeepNumericFailureAndBytesIndependent) {
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=2})).value();
  const auto input=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{2,64}).value();
  for(auto storage:{c::CacheStorage::q8_0().value(),c::CacheStorage::float8(.5F).value()}) {
    const auto operation=c::Operation::custom(c::model_ops::CacheEncode{storage});
    const auto output=c::infer_single_output(operation,std::array{input}).value();
    std::array<hyle::mem::Buffer,2> sources,destinations;
    for(unsigned lane=0;lane<2;++lane) {
      sources[lane]=hyle::mem::Buffer::create(hyle::mem::system_resource(),512,64).value();
      destinations[lane]=hyle::mem::Buffer::create(hyle::mem::system_resource(),output.required_bytes().value(),64).value();
    }
    const std::array prototype_inputs{c::ConstTensorView::create(input,sources[0].view()).value()};
    const std::array prototype_outputs{c::TensorView::create(output,destinations[0].view()).value()};
    auto prepared=c::prepare(context,operation,prototype_inputs,prototype_outputs).value();
    ASSERT_TRUE(prepared.runtime_properties().supports_concurrent_invocations);
    std::array queues{context.create_queue().value(),context.create_queue().value()};
    std::array invocations{prepared.create_invocation().value(),prepared.create_invocation().value()};
    std::promise<void> start;
    const auto ready=start.get_future().share();
    const auto run=[&](unsigned lane)->hyle::Status {
      auto &queue=queues[lane];auto &invocation=invocations[lane];
      const std::array inputs{c::ConstTensorView::create(input,sources[lane].view()).value()};
      const std::array outputs{c::TensorView::create(output,destinations[lane].view()).value()};
      ready.wait();
      for(unsigned step=0;step<8;++step) {
        const float value=(lane?-1.F:1.F)*float(step+1)*.25F;
        std::array<float,128> values;values.fill(value);
        const bool bad=lane==0 && step==0;
        if(bad)values[17]=std::numeric_limits<float>::quiet_NaN();
        HYLE_RETURN_IF_ERROR(sources[lane].view().write(std::as_bytes(std::span(values))));
        HYLE_ASSIGN_OR_RETURN(auto work,invocation.submit(queue,inputs,outputs));
        const auto terminal=work.wait();
        if(bad) {
          if(!terminal.IsInvalidArgument() || invocation.submit(queue,inputs,outputs))
            return hyle::Status::Internal("concurrent failed encoder lost its original terminal/retirement");
          work={};HYLE_ASSIGN_OR_RETURN(invocation,prepared.create_invocation());
          continue;
        }
        HYLE_RETURN_IF_ERROR(terminal);
        std::vector<std::uint8_t> actual(output.required_bytes().value()),expected(actual.size());
        HYLE_RETURN_IF_ERROR(destinations[lane].view().read(std::as_writable_bytes(std::span(actual))));
        if(storage==c::CacheStorage::q8_0().value()) {
          const auto scale=half_test::encode<hyle::simd::float16_t>(std::abs(value)/127.F).bits;
          for(std::size_t block=0;block<expected.size()/34;++block) {
            expected[block*34]=std::uint8_t(scale);expected[block*34+1]=std::uint8_t(scale>>8);
            std::fill_n(expected.begin()+block*34+2,32,std::uint8_t(lane?-127:127));
          }
        } else std::fill(expected.begin(),expected.end(),float8_oracle(double(value)/.5));
        if(actual!=expected)return hyle::Status::Internal("concurrent encoder lanes shared failure flags or payload bytes");
      }
      return queue.drain();
    };
    auto first=std::async(std::launch::async,run,0);
    std::future<hyle::Status> second;
    try {second=std::async(std::launch::async,run,1);}
    catch(...) {start.set_value();first.wait();throw;}
    start.set_value();
    const auto a=first.get(),b=second.get();
    ASSERT_TRUE(a.ok())<<a.message();ASSERT_TRUE(b.ok())<<b.message();
  }
  ASSERT_TRUE(context.drain().ok());
}
double q8_value(std::span<const std::uint8_t> bytes,std::size_t element){
  const auto block=element/32,within=element%32;
  std::uint16_t scale;std::memcpy(&scale,bytes.data()+block*34,2);
  return half_test::decode_bits<hyle::simd::float16_t>(scale)*
      static_cast<std::int8_t>(bytes[block*34+2+within]);
}
template<class T>void boxed_case(bool paged,bool quantized,unsigned depth,unsigned value_depth,
                                bool causal,int window,bool long_precision=false,bool long_prefill=false){
  SCOPED_TRACE(::testing::Message()<<"paged="<<paged<<" quantized="<<quantized<<" depth="<<depth
      <<" value_depth="<<value_depth<<" window="<<window<<" long_prefill="<<long_prefill);
  const unsigned query_rows=long_precision?1u:(long_prefill?66u:(causal?18u:68u));
  const unsigned physical_rows=long_precision?4096u:(long_prefill?(paged?256u:274u):64u);
  const unsigned heads=long_precision?1u:8u,kv_heads=long_precision?1u:2u;
  const auto storage=quantized?c::CacheStorage::q8_0().value():
      c::CacheStorage::plain(half_test::data_type<T>()).value();
  const c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
    .query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),
    .key_storage=storage,.value_storage=storage,.max_requests=long_precision?1u:2u,
    .max_query_rows=query_rows,.max_kv_rows=long_prefill?274u:physical_rows,.max_pages=paged?(long_prefill?16u:4u):0u,
    .max_page_refs=paged?(long_prefill?19u:5u):0u,.query_heads=heads,.kv_heads=kv_heads,
    .qk_head_dim=depth,.value_head_dim=value_depth,.page_size=paged?16u:1u};
  auto context=c::cpu::create_context({.native_threads=long_precision?1u:3u}).value();
  auto queue=context.create_queue().value();auto schema=c::BatchSchema::create(cap).value();
  const std::vector<std::int32_t> qp=long_precision?std::vector<std::int32_t>{0,1}:
      (long_prefill?std::vector<std::int32_t>{0,1,66}:(causal?std::vector<std::int32_t>{0,1,18}:std::vector<std::int32_t>{0,33,68}));
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
  for(std::size_t i=0;i<q.size();++i)q[i]=half_test::encode<T>(long_precision?0.:.17*std::sin(double(i)*.23));
  for(std::size_t i=0;i<k.size();++i)k[i]=long_precision?0.F:float(.19*std::cos(double(i)*.13));
  for(std::size_t i=0;i<v.size();++i)v[i]=long_precision?
      (i/value_depth<3072?1.F:-1.F):float(.23*std::sin(double(i)*.07)+.05*std::cos(double(i)*.17));
  std::vector<std::uint8_t> kb,vb;
  const auto cache_bytes=[&](std::span<const float> input,std::vector<std::uint8_t>& output){
    if(quantized){ASSERT_TRUE(encode_words(half_test::data_type<T>(),storage,input,output).ok());}
    else{std::vector<T> plain(input.size());for(std::size_t i=0;i<input.size();++i)plain[i]=half_test::encode<T>(input[i]);
      output.resize(plain.size()*sizeof(T));std::memcpy(output.data(),plain.data(),output.size());}
  };
  cache_bytes(k,kb);cache_bytes(v,vb);
  const auto value_at=[&](bool key,std::size_t index){
    if(quantized)return q8_value(key?std::span<const std::uint8_t>(kb):std::span<const std::uint8_t>(vb),index);
    return half_test::decode(half_test::encode<T>(key?k[index]:v[index]));
  };
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),q.size()*sizeof(T)+kb.size()+vb.size()+512).value();
  const auto load=[&](std::span<const std::byte> bytes){auto writer=pool.allocate(bytes.size()).value();
    EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();};
  auto query=load(std::as_bytes(std::span(q)));
  c::ReadLease key_lease,value_lease;std::vector<c::ReadLease> key_pages(cap.max_pages),value_pages(cap.max_pages);
  c::PagedArena key_arena,value_arena;c::PagedArenaRead key_frame,value_frame;
  if(paged){
    key_arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,hyle::mem::system_resource(),{.max_backing_bytes=kb.size()+64}).value();
    value_arena=c::PagedArena::create(schema,c::BatchCacheRole::kValue,hyle::mem::system_resource(),{.max_backing_bytes=vb.size()+64}).value();
    for(unsigned page=0;page<cap.max_pages;++page){auto kw=key_arena.acquire(page).value(),vw=value_arena.acquire(page).value();
      ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(kb).subspan(page*kb.size()/cap.max_pages,kb.size()/cap.max_pages))).ok());
      ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(vb).subspan(page*vb.size()/cap.max_pages,vb.size()/cap.max_pages))).ok());
      key_pages[page]=kw.publish(16).value();value_pages[page]=vw.publish(16).value();}
    auto kr=key_arena.read(layout,key_pages),vr=value_arena.read(layout,value_pages);
    ASSERT_TRUE(kr)<<kr.error().message();ASSERT_TRUE(vr)<<vr.error().message();
    key_frame=std::move(*kr);value_frame=std::move(*vr);
  }else{key_lease=load(std::as_bytes(std::span(kb)));value_lease=load(std::as_bytes(std::span(vb)));}
  const float scale=1.F/std::sqrt(float(depth));
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,.scale=scale,.causal=causal,.window_left=window});
  ASSERT_TRUE(program)<<program.error().message();auto slot=c::BatchInvocation::create(*program,queue).value();
  auto prepared=slot.prepare_batch(layout);ASSERT_TRUE(prepared)<<prepared.error().message();
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
        // Q8 CPU converts Q to its vec-dot type; tolerate that bounded extra
        // quantization as well as the declared output rounding.
        EXPECT_NEAR(half_test::decode(actual[(std::size_t(qp[request]+row)*heads+head)*value_depth+d]),expected/sum,
                    long_precision?.0006:.0025);}
    }
  }
  result.reset();*done={};*prepared={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
}
TEST(GgmlCpuBoxed, QuantizedPagedAndRaggedIndependentValueDepthAndWindow){
  for(bool paged:{false,true})for(int window:{-1,0,15,16}){
    boxed_case<hyle::simd::float16_t>(paged,true,64,128,true,window);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,64,128,true,window);
  }
}
TEST(GgmlCpuBoxed, QuantizedFullNoncausalAllowsMoreQueriesThanKeys){
  for(bool paged:{false,true}){
    boxed_case<hyle::simd::float16_t>(paged,true,64,128,false,-1);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,64,128,false,-1);
  }
}
TEST(GgmlCpuBoxed, PlainWindowAndIndependentValueDepth){
  for(bool paged:{false,true})for(int window:{0,16}){
    boxed_case<hyle::simd::float16_t>(paged,false,64,128,true,window);
    boxed_case<hyle::simd::bfloat16_t>(paged,false,64,128,true,window);
  }
}
TEST(GgmlCpuBoxed, QuantizedCoupledDepthsIncludingMainD256AndLongPrefill){
  for(bool paged:{false,true})for(unsigned depth:{64u,128u,256u}){
    boxed_case<hyle::simd::float16_t>(paged,true,depth,depth,true,16);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,depth,depth,true,16);
  }
  for(bool paged:{false,true}){
    boxed_case<hyle::simd::float16_t>(paged,true,256,256,true,-1,false,true);
    boxed_case<hyle::simd::bfloat16_t>(paged,true,256,256,true,-1,false,true);
  }
}
TEST(GgmlCpuBoxed, LongF16WindowUsesF32AccumulatorAgainstCancellation){
  boxed_case<hyle::simd::float16_t>(false,false,64,128,true,4095,true);
}
TEST(GgmlCpuBoxed, FiniteBf16QueryScaleOverflowIsAnAcceptedErrorAndFreshSlotSucceeds){
  using T=hyle::simd::bfloat16_t;
  auto context=c::cpu::create_context({.native_threads=3}).value();auto queue=context.create_queue().value();
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kBFloat16,
    .output_type=c::DataType::kBFloat16,.key_storage=c::CacheStorage::q8_0().value(),
    .value_storage=c::CacheStorage::q8_0().value(),.max_query_rows=1,.max_kv_rows=1,
    .max_pages=0,.max_page_refs=0,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,2> indptr{0,1};
  auto layout=schema.version({indptr,indptr,{},{}}).value();
  std::vector<float> zero(64,0.F),one(64,1.F);std::vector<std::uint8_t> kb,vb;
  ASSERT_TRUE(encode_words(c::DataType::kBFloat16,cap.key_storage,zero,kb).ok());
  ASSERT_TRUE(encode_words(c::DataType::kBFloat16,cap.value_storage,one,vb).ok());
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),2048).value();
  const auto load=[&](std::span<const std::byte> bytes){auto writer=pool.allocate(bytes.size()).value();
    EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();};
  auto k=load(std::as_bytes(std::span(kb))),v=load(std::as_bytes(std::span(vb)));
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.scale=.125F}).value();
  for(float value:{9000000.F,std::numeric_limits<float>::infinity(),1.F}){
    auto slot=c::BatchInvocation::create(program,queue).value();
    std::vector<T> query(64,half_test::encode<T>(value));auto q=load(std::as_bytes(std::span(query)));
    auto prepared=slot.update_batch(layout);ASSERT_TRUE(prepared)<<prepared.error().message();
    const std::array inputs{q,k,v};auto completion=prepared->run(inputs);
    ASSERT_TRUE(completion)<<completion.error().message();const auto status=completion->wait();
    if(value==1.F){ASSERT_TRUE(status.ok())<<status.message();auto output=slot.output().value();std::vector<T> actual(64);
      ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
      for(auto element:actual)EXPECT_NEAR(half_test::decode(element),1.,.01);
    }else{EXPECT_TRUE(status.IsInvalidArgument())<<status.message();EXPECT_FALSE(slot.output());
      *completion={};*prepared={};
      auto refused=slot.update_batch(layout);ASSERT_FALSE(refused);EXPECT_TRUE(refused.error().IsFailedPrecondition())<<refused.error().message();}
    *completion={};*prepared={};
    ASSERT_TRUE(slot.close().ok());
  }
  ASSERT_TRUE(context.drain().ok());
}

template<class T>void query_representation_boundaries(std::span<const std::uint16_t> codes){
  auto context=c::cpu::create_context({.native_threads=1}).value();
  auto queue=context.create_queue().value();
  const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.query_type=half_test::data_type<T>(),
    .output_type=half_test::data_type<T>(),.key_storage=c::CacheStorage::q8_0().value(),
    .value_storage=c::CacheStorage::q8_0().value(),.max_query_rows=1,.max_kv_rows=1,
    .max_pages=0,.max_page_refs=0,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();
  const std::array<std::int32_t,2> indptr{0,1};auto layout=schema.version({indptr,indptr,{},{}}).value();
  std::vector<float> zero(64,0.F),one(64,1.F);std::vector<std::uint8_t> kb,vb;
  ASSERT_TRUE(encode_words(half_test::data_type<T>(),cap.key_storage,zero,kb).ok());
  ASSERT_TRUE(encode_words(half_test::data_type<T>(),cap.value_storage,one,vb).ok());
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),2048).value();
  const auto load=[&](std::span<const std::byte> bytes){auto writer=pool.allocate(bytes.size()).value();
    EXPECT_TRUE(writer.view()->write(bytes).ok());return writer.publish().value();};
  auto k=load(std::as_bytes(std::span(kb))),v=load(std::as_bytes(std::span(vb)));
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.scale=.125F}).value();
  for(auto code:codes)for(unsigned index:{0u,63u}){
    SCOPED_TRACE(::testing::Message()<<"code="<<code<<" index="<<index);
    std::array<T,64> query;query.fill(half_test::encode<T>(1.F));query[index]=T::from_bits(code);
    const auto value=half_test::decode(query[index]);
    const bool valid=std::isfinite(value)&&std::abs(value)<=65504.*127.;
    auto slot=c::BatchInvocation::create(program,queue).value();auto prepared=slot.update_batch(layout);
    ASSERT_TRUE(prepared)<<prepared.error().message();
    auto q=load(std::as_bytes(std::span(query)));const std::array inputs{q,k,v};
    auto completion=prepared->run(inputs);ASSERT_TRUE(completion)<<completion.error().message();
    const auto status=completion->wait();
    if(valid){
      ASSERT_TRUE(status.ok())<<status.message();auto output=slot.output().value();std::array<T,64> actual;
      ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
      for(auto element:actual)EXPECT_NEAR(half_test::decode(element),1.,.01);
    }else{
      EXPECT_TRUE(status.IsInvalidArgument())<<status.message();EXPECT_FALSE(slot.output());
      *completion={};*prepared={};auto refused=slot.update_batch(layout);
      ASSERT_FALSE(refused);EXPECT_TRUE(refused.error().IsFailedPrecondition());
    }
    *completion={};*prepared={};ASSERT_TRUE(slot.close().ok());
  }
  ASSERT_TRUE(context.drain().ok());
}
TEST(GgmlCpuBoxed, QuerySafetyChecksBothEndsAtRepresentationBoundaries){
  const std::array<std::uint16_t,20> half{0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
    0x3c00,0xbc00,0x7bff,0xfbff,0x7c00,0xfc00,0x7c01,0xfc01,0x7e00,0xfe00,0x7fff,0xffff};
  query_representation_boundaries<hyle::simd::float16_t>(half);
  constexpr auto edge=std::uint16_t(std::bit_cast<std::uint32_t>(65504.F*127.F)>>16);
  const std::array<std::uint16_t,26> bfloat{0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,
    0x3f80,0xbf80,std::uint16_t(edge-1),edge,std::uint16_t(edge+1),
    std::uint16_t((edge-1)|0x8000),std::uint16_t(edge|0x8000),std::uint16_t((edge+1)|0x8000),
    0x7f7f,0xff7f,0x7f80,0xff80,0x7f81,0xff81,0x7fc0,0xffc0,0x7fff,0xffff};
  query_representation_boundaries<hyle::simd::bfloat16_t>(bfloat);
}

template<class T>void paged_case(unsigned depth,bool causal){
  auto context=c::cpu::create_context({.native_threads=2}).value();auto queue=context.create_queue().value();
  const c::BatchCapacity cap{.storage=c::BatchStorage::kPaged,.query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),.key_storage=c::CacheStorage::plain(half_test::data_type<T>()).value(),.value_storage=c::CacheStorage::plain(half_test::data_type<T>()).value(),
    .max_requests=2,.max_query_rows=21,.max_kv_rows=80,.max_pages=6,.max_page_refs=5,
    .query_heads=8,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth,.page_size=16};
  auto schema=c::BatchSchema::create(cap).value();
  const std::array<std::int32_t,3> qp{0,1,18},kp{0,2,5};
  const std::array<std::int32_t,5> ids{4,1,3,0,1};const std::array<std::int32_t,2> tails{15,15};
  auto layout=schema.version({qp,kp,ids,tails}).value();
  auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,hyle::mem::system_resource(),{.max_backing_bytes=1u<<20}).value();
  auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,hyle::mem::system_resource(),{.max_backing_bytes=1u<<20}).value();
  const std::array<unsigned,4> pages{0,1,3,4};std::array<c::ReadLease,4> k,v;
  for(unsigned index=0;index<pages.size();++index){
    auto kw=keys.acquire(pages[index]).value(),vw=values.acquire(pages[index]).value();
    std::vector<T> zero(16*2*depth,half_test::encode<T>(0.F)),data(zero.size());
    for(unsigned row=0;row<16;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<depth;++d){
      zero[(row*2+head)*depth+d]=half_test::encode<T>(float(int((pages[index]*7+row*3+head*5+d)%17)-8)*.03125F);
      data[(row*2+head)*depth+d]=half_test::encode<T>(float(pages[index])*.0625F+float(row)*.00390625F+float(head)*.03125F);
    }
    ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(zero))).ok());
    ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(data))).ok());k[index]=kw.publish(cap.page_size).value();v[index]=vw.publish(cap.page_size).value();
  }
  auto kr=keys.read(layout,k).value(),vr=values.read(layout,v).value();
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),21*8*depth*2+64).value();auto writer=pool.allocate(21*8*depth*2).value();
  std::vector<T> zero(21*8*depth,half_test::encode<T>(0.F));
  for(std::size_t i=0;i<zero.size();++i)zero[i]=half_test::encode<T>(float(int((i*11)%19)-9)*.03125F);
  ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(zero))).ok());auto q=writer.publish().value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,.scale=1.F/std::sqrt(float(depth)),.causal=causal});
  ASSERT_TRUE(program)<<program.error().message();auto slot=c::BatchInvocation::create(*program,queue).value();auto token=slot.prepare_batch(layout);ASSERT_TRUE(token)<<token.error().message();
  auto done=token->run(q,kr,vr);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());auto output=slot.output().value();std::vector<T> actual(zero.size());ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  for(unsigned request=0;request<2;++request){const unsigned count=request?47:31,queries=request?17:1;
    for(unsigned row=0;row<queries;++row)for(unsigned head=0;head<8;++head){const unsigned valid=causal?count-queries+row+1:count;double expected=0,sum=0;
      for(unsigned index=0;index<valid;++index){const unsigned physical=ids[kp[request]+index/16],offset=index%16;double dot=0;
        for(unsigned d=0;d<depth;++d){const double key=half_test::decode(half_test::encode<T>(double(int((physical*7+offset*3+(head/4)*5+d)%17)-8)*.03125));dot+=half_test::decode(zero[((qp[request]+row)*8+head)*depth+d])*key;}
        const double weight=std::exp(dot/double(std::sqrt(float(depth))));sum+=weight;
        expected+=weight*(double(physical)*.0625+double(offset)*.00390625+double(head/4)*.03125);
      }
      expected/=sum;
      for(unsigned d=0;d<depth;++d)EXPECT_NEAR(half_test::decode(actual[((qp[request]+row)*8+head)*depth+d]),expected,.004);
    }
  }
  EXPECT_FALSE(slot.update_batch(layout));auto free_page=keys.acquire(5);ASSERT_TRUE(free_page)<<free_page.error().message();
  EXPECT_FALSE(keys.acquire(1));output.reset();*done={};*token={};ASSERT_TRUE(context.drain().ok());
  // New immutable metadata owners with identical content must not leave
  // borrowed page spans dangling in the reused original native graph.
  for(unsigned step=0;step<64;++step){
    auto next=schema.version({qp,kp,ids,tails}).value();
    auto next_keys=keys.read(next,k).value(),next_values=values.read(next,v).value();
    auto reused=slot.update_batch(next);ASSERT_TRUE(reused)<<reused.error().message();
    EXPECT_TRUE(slot.report().reused_native_plan);EXPECT_EQ(slot.report().layout_generation,next.generation());
    EXPECT_GT(slot.report().metadata_upload.logical_bytes,0u);
    EXPECT_LE(slot.report().metadata_upload.logical_bytes,8u); // Only generation words changed.
    EXPECT_EQ(slot.report().host_waits,1u);
    auto replay=reused->run(q,next_keys,next_values);ASSERT_TRUE(replay)<<replay.error().message();ASSERT_TRUE(replay->wait().ok());
    auto ready=slot.output().value();std::vector<T> current(zero.size());ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(current))).ok());
    for(unsigned i=0;i<18*8*depth;++i)EXPECT_EQ(current[i].bits,actual[i].bits);
    ready.reset();*replay={};
    *reused={};auto unchanged=slot.update_batch(next);ASSERT_TRUE(unchanged)<<unchanged.error().message();
    EXPECT_TRUE(slot.report().reused_native_plan);EXPECT_EQ(slot.report().metadata_upload.logical_bytes,0u);
    EXPECT_EQ(slot.report().metadata_staging_bytes,0u);EXPECT_EQ(slot.report().host_waits,0u);
  }
  ASSERT_TRUE(context.drain().ok());
  const std::array<std::int32_t,5> reordered{1,4,0,3,1};
  const std::array<std::int32_t,2> shorter{14,14};
  auto changed=schema.version({qp,kp,reordered,shorter}).value();
  auto changed_keys=keys.read(changed,k).value(),changed_values=values.read(changed,v).value();
  auto rebuilt=slot.update_batch(changed);ASSERT_TRUE(rebuilt)<<rebuilt.error().message();
  EXPECT_FALSE(slot.report().reused_native_plan);
  EXPECT_FALSE(rebuilt->run(q,kr,vr));
  auto final=rebuilt->run(q,changed_keys,changed_values);ASSERT_TRUE(final)<<final.error().message();ASSERT_TRUE(final->wait().ok());
  auto changed_output=slot.output().value();std::vector<T> current(zero.size());ASSERT_TRUE(changed_output.view()->read(std::as_writable_bytes(std::span(current))).ok());
  for(unsigned request=0;request<2;++request){const unsigned count=request?46:30,queries=request?17:1;
    for(unsigned row=0;row<queries;++row)for(unsigned head=0;head<8;++head){const unsigned valid=causal?count-queries+row+1:count;double sum=0,expected=0;
      for(unsigned index=0;index<valid;++index){const unsigned physical=reordered[kp[request]+index/16],offset=index%16;double dot=0;
        for(unsigned d=0;d<depth;++d)dot+=half_test::decode(zero[((qp[request]+row)*8+head)*depth+d])*half_test::decode(half_test::encode<T>(double(int((physical*7+offset*3+(head/4)*5+d)%17)-8)*.03125));
        const double weight=std::exp(dot/double(std::sqrt(float(depth))));sum+=weight;expected+=weight*(double(physical)*.0625+double(offset)*.00390625+double(head/4)*.03125);
      }
      for(unsigned d=0;d<depth;++d)EXPECT_NEAR(half_test::decode(current[((qp[request]+row)*8+head)*depth+d]),expected/sum,.004);
    }
  }
  // A second original slot owns a different page topology while the first
  // keeps its result and borrowed native page table alive.
  auto independent=c::BatchInvocation::create(*program,queue).value();
  auto independent_plan=independent.prepare_batch(layout);ASSERT_TRUE(independent_plan)<<independent_plan.error().message();
  auto independent_work=independent_plan->run(q,kr,vr);ASSERT_TRUE(independent_work)<<independent_work.error().message();
  ASSERT_TRUE(independent_work->wait().ok());auto independent_output=independent.output().value();
  std::vector<T> other(zero.size()),retained(zero.size());
  ASSERT_TRUE(independent_output.view()->read(std::as_writable_bytes(std::span(other))).ok());
  ASSERT_TRUE(changed_output.view()->read(std::as_writable_bytes(std::span(retained))).ok());
  for(unsigned i=0;i<18*8*depth;++i){EXPECT_EQ(other[i].bits,actual[i].bits);EXPECT_EQ(retained[i].bits,current[i].bits);}
  independent_output.reset();*independent_work={};*independent_plan={};ASSERT_TRUE(independent.close().ok());
  changed_output.reset();*final={};*rebuilt={};ASSERT_TRUE(context.drain().ok());
  for(unsigned step=0;step<64;++step){
    const unsigned first_count=15+step%17,second_count=step%8?17+step%31:0;
    const unsigned first_refs=(first_count+15)/16,second_refs=(second_count+15)/16;
    const std::array<std::int32_t,3> active_q{0,1,second_count?18:1},active_k{0,static_cast<std::int32_t>(first_refs),static_cast<std::int32_t>(first_refs+second_refs)};
    const std::array<std::int32_t,2> active_tails{static_cast<std::int32_t>((first_count-1)%16+1),second_count?static_cast<std::int32_t>((second_count-1)%16+1):0};
    std::vector<std::int32_t> order;
    for(unsigned i=0;i<first_refs;++i)order.push_back(step%2?(i?4:1):(i?1:4));
    for(unsigned i=0;i<second_refs;++i)order.push_back(step%2?(i==0?0:i==1?3:1):(i==0?3:i==1?0:1));
    auto evolving=schema.version({active_q,active_k,order,active_tails}).value();
    std::vector<c::ReadLease> active_keys,active_values;
    for(unsigned i=0;i<pages.size();++i)if(std::find(order.begin(),order.end(),pages[i])!=order.end()){active_keys.push_back(k[i]);active_values.push_back(v[i]);}
    auto frame_k=keys.read(evolving,active_keys).value(),frame_v=values.read(evolving,active_values).value();
    auto planned=slot.update_batch(evolving);ASSERT_TRUE(planned)<<planned.error().message();
    auto work=planned->run(q,frame_k,frame_v);ASSERT_TRUE(work)<<work.error().message();ASSERT_TRUE(work->wait().ok());
    auto result=slot.output().value();std::vector<T> evolving_output(zero.size());ASSERT_TRUE(result.view()->read(std::as_writable_bytes(std::span(evolving_output))).ok());
    for(unsigned request=0;request<2;++request){const unsigned count=request?second_count:first_count,queries=request?(second_count?17:0):1;
      for(unsigned row=0;row<queries;++row)for(unsigned head=0;head<8;++head){const unsigned valid=causal?count-queries+row+1:count;double sum=0,expected=0;
        for(unsigned index=0;index<valid;++index){const unsigned physical=order[active_k[request]+index/16],offset=index%16;double dot=0;
          for(unsigned d=0;d<depth;++d)dot+=half_test::decode(zero[((active_q[request]+row)*8+head)*depth+d])*half_test::decode(half_test::encode<T>(double(int((physical*7+offset*3+(head/4)*5+d)%17)-8)*.03125));
          const double weight=std::exp(dot/double(std::sqrt(float(depth))));sum+=weight;expected+=weight*(double(physical)*.0625+double(offset)*.00390625+double(head/4)*.03125);
        }
        for(unsigned d=0;d<depth;++d)EXPECT_NEAR(half_test::decode(evolving_output[((active_q[request]+row)*8+head)*depth+d]),expected/sum,.004);
      }
    }
    result.reset();*work={};*planned={};
  }
  ASSERT_TRUE(context.drain().ok());
}
TEST(GgmlCpuPaged, OriginalFlashReadsExactPhysicalPagesAndTail){
  for(unsigned d:{64U,128U,256U})for(bool causal:{false,true}){
    paged_case<hyle::simd::float16_t>(d,causal);paged_case<hyle::simd::bfloat16_t>(d,causal);
  }
}
}

TEST(GgmlCpuPaged, DeltaMetadataMirrorMatchesIndependentFullWireAcrossGrowShrinkAndEmpty){
  const auto resource=hyle::mem::system_resource();
  c::BatchCapacity cap{.max_requests=3,.max_query_rows=3,.max_kv_rows=96,.max_pages=6,.max_page_refs=6,
      .query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=16};
  auto schema=c::BatchSchema::create(cap).value();
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=2})).value();
  auto queue=context.create_queue().value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=resource,.max_float_workspace_bytes=2u<<20,.max_int_workspace_bytes=4096,.scale=.125F}).value();
  auto slot=c::BatchInvocation::create(program,queue).value();c::BatchLayoutVersion version;
  for(unsigned step=0;step<64;++step){
    std::vector<std::int32_t> q,k,ids,tails;
    switch((step/2)%6){
      case 0:q={0};k={0};break;
      case 1:q={0,1};k={0,2};ids={2,0};tails={3};break;
      case 2:q={0,1,2};k={0,1,3};ids={4,1,4};tails={16,7};break;
      case 3:q={0,1,2};k={0,1,3};ids={1,4,1};tails={7,16};break;
      case 4:q={0,0,1};k={0,0,1};ids={3};tails={0,1};break;
      default:q={0,1,1,2};k={0,2,2,4};ids={0,2,3,1};tails={16,0,8};break;
    }
    if(!(step%2))version=schema.version({q,k,ids,tails}).value();
    auto prepared=slot.update_batch(version);ASSERT_TRUE(prepared)<<prepared.error().message();
    auto mirror=c::detail::BatchGraphAccess::metadata(slot);ASSERT_TRUE(mirror)<<mirror.error().message();
    const auto offsets=schema.metadata_offsets();std::vector<std::byte> expected(offsets.bytes,std::byte{0}),actual(offsets.bytes);
    const auto put=[&](std::size_t offset,const auto&values){const auto bytes=std::as_bytes(std::span(values));if(!bytes.empty())std::memcpy(expected.data()+offset,bytes.data(),bytes.size());};
    std::uint32_t rows=0;
    for(std::size_t r=0;r<tails.size();++r)if(k[r+1]!=k[r])rows+=std::uint32_t(k[r+1]-k[r]-1)*16+std::uint32_t(tails[r]);
    const std::array<std::uint64_t,2> prefix{0x48594c45424c0001ULL,version.generation()};
    const std::array<std::uint32_t,4> facts{std::uint32_t(q.size()-1),std::uint32_t(q.back()),rows,std::uint32_t(ids.size())};
    put(0,prefix);put(16,facts);put(offsets.query_indptr,q);put(offsets.kv_indptr,k);put(offsets.page_indices,ids);put(offsets.last_page_lengths,tails);
    ASSERT_TRUE(mirror->view()->read(actual).ok());EXPECT_EQ(actual,expected)<<"step="<<step;
    if(step%2){EXPECT_TRUE(slot.report().reused_native_plan);EXPECT_EQ(slot.report().metadata_staging_bytes,0u);EXPECT_EQ(slot.report().metadata_upload.logical_bytes,0u);EXPECT_EQ(slot.report().host_waits,0u);}
  }
  ASSERT_TRUE(context.drain().ok());
}
