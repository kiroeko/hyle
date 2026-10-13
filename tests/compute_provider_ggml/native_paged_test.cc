#include <gtest/gtest.h>
#include <hyle/compute/cpu.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include "../compute_provider_native/half_test_support.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
namespace c=hyle::compute;
template<class T>void paged_case(unsigned depth,bool causal){
  auto context=c::cpu::create_context({.native_threads=2}).value();auto queue=context.create_queue().value();
  const c::BatchCapacity cap{.storage=c::BatchStorage::kPaged,.data_type=half_test::data_type<T>(),
    .max_requests=2,.max_query_rows=21,.max_kv_rows=80,.max_pages=6,.max_page_refs=5,
    .query_heads=8,.kv_heads=2,.head_dim=depth,.page_size=16};
  auto schema=c::BatchSchema::create(cap).value();
  const std::array<std::int32_t,3> qp{0,1,18},kp{0,2,5};
  const std::array<std::int32_t,5> ids{4,1,3,0,1};const std::array<std::int32_t,2> tails{15,15};
  auto layout=schema.version({qp,kp,ids,tails}).value();
  auto keys=c::PagedArena::create(schema,hyle::mem::system_resource(),{.max_backing_bytes=1u<<20}).value();
  auto values=c::PagedArena::create(schema,hyle::mem::system_resource(),{.max_backing_bytes=1u<<20}).value();
  const std::array<unsigned,4> pages{0,1,3,4};std::array<c::ReadLease,4> k,v;
  for(unsigned index=0;index<pages.size();++index){
    auto kw=keys.acquire(pages[index]).value(),vw=values.acquire(pages[index]).value();
    std::vector<T> zero(16*2*depth,half_test::encode<T>(0.F)),data(zero.size());
    for(unsigned row=0;row<16;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<depth;++d){
      zero[(row*2+head)*depth+d]=half_test::encode<T>(float(int((pages[index]*7+row*3+head*5+d)%17)-8)*.03125F);
      data[(row*2+head)*depth+d]=half_test::encode<T>(float(pages[index])*.0625F+float(row)*.00390625F+float(head)*.03125F);
    }
    ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(zero))).ok());
    ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(data))).ok());k[index]=kw.publish().value();v[index]=vw.publish().value();
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
    auto replay=reused->run(q,next_keys,next_values);ASSERT_TRUE(replay)<<replay.error().message();ASSERT_TRUE(replay->wait().ok());
    auto ready=slot.output().value();std::vector<T> current(zero.size());ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(current))).ok());
    for(unsigned i=0;i<18*8*depth;++i)EXPECT_EQ(current[i].bits,actual[i].bits);
    ready.reset();*replay={};
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
