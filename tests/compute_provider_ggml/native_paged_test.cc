#include <gtest/gtest.h>
#include <hyle/compute/cpu.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include "../compute_provider_native/half_test_support.h"
#include <array>
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
    for(unsigned row=0;row<16;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<depth;++d)
      data[(row*2+head)*depth+d]=half_test::encode<T>(float(pages[index])*.0625F+float(row)*.00390625F+float(head)*.03125F);
    ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(zero))).ok());
    ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(data))).ok());k[index]=kw.publish().value();v[index]=vw.publish().value();
  }
  auto kr=keys.read(layout,k).value(),vr=values.read(layout,v).value();
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),21*8*depth*2+64).value();auto writer=pool.allocate(21*8*depth*2).value();
  std::vector<T> zero(21*8*depth,half_test::encode<T>(0.F));ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(zero))).ok());auto q=writer.publish().value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,.scale=1.F/std::sqrt(float(depth)),.causal=causal});
  ASSERT_TRUE(program)<<program.error().message();auto slot=c::BatchInvocation::create(*program,queue).value();auto token=slot.prepare_batch(layout);ASSERT_TRUE(token)<<token.error().message();
  auto done=token->run(q,kr,vr);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());auto output=slot.output().value();std::vector<T> actual(zero.size());ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  for(unsigned request=0;request<2;++request){const unsigned count=request?47:31,queries=request?17:1;
    for(unsigned row=0;row<queries;++row)for(unsigned head=0;head<8;++head){const unsigned valid=causal?count-queries+row+1:count;double expected=0;
      for(unsigned index=0;index<valid;++index)expected+=(double(ids[kp[request]+index/16])*.0625+double(index%16)*.00390625+double(head/4)*.03125)/valid;
      for(unsigned d=0;d<depth;++d)EXPECT_NEAR(half_test::decode(actual[((qp[request]+row)*8+head)*depth+d]),expected,.004);
    }
  }
  EXPECT_FALSE(slot.update_batch(layout));auto free_page=keys.acquire(5);ASSERT_TRUE(free_page)<<free_page.error().message();
  EXPECT_FALSE(keys.acquire(1));output.reset();*done={};ASSERT_TRUE(context.drain().ok());
}
TEST(GgmlCpuPaged, OriginalFlashReadsExactPhysicalPagesAndTail){
  for(unsigned d:{64U,128U,256U})for(bool causal:{false,true}){
    paged_case<hyle::simd::float16_t>(d,causal);paged_case<hyle::simd::bfloat16_t>(d,causal);
  }
}
}
