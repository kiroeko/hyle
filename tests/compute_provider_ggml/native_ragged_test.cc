#include <gtest/gtest.h>
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "../compute_provider_native/half_test_support.h"
#include "../compute_batch/allocation_fault_fixture.h"
#include "../../src/compute/context_state.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
namespace c = hyle::compute;
using F16 = hyle::simd::float16_t;
using BF16 = hyle::simd::bfloat16_t;
struct AcceptedFailure final {
  c::detail::ContextState *state;
  const c::detail::BackendVTable *original;
  c::detail::BackendVTable table;
  static inline AcceptedFailure *active=nullptr;
  explicit AcceptedFailure(c::Context &context):state(c::detail::ContextAccess::state(context)),
      original(state->backend_vtable),table(*original) {
    table.finish_sequence=[](void *queue,void *sequence,bool accepted,hyle::Status terminal)noexcept {
      EXPECT_TRUE(accepted);terminal.Update(hyle::Status::ResourceExhausted("CPU ragged accepted math prefix"));
      return active->original->finish_sequence(queue,sequence,accepted,std::move(terminal));
    };
    active=this;state->backend_vtable=&table;
  }
  ~AcceptedFailure(){state->backend_vtable=original;active=nullptr;}
};
template<class T> T encode(float x) { return half_test::encode<T>(x); }
template<class T> double decode(T x) { return half_test::decode(x); }
template<class T> c::ReadLease load(c::LeasePool &pool, const std::vector<T> &values) {
  auto writer = pool.allocate(values.size()*sizeof(T)).value();
  EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());
  return writer.publish().value();
}

template<class T>void runtime_ragged_q1(bool encoded,unsigned depth,unsigned value_depth,int window,unsigned requested){
  hyle_f5_fault_fixture::Allocator fault;
  {
  const auto type=half_test::data_type<T>();const bool causal=depth!=128;
  const auto storage=encoded?c::CacheStorage::q8_0().value():c::CacheStorage::plain(type).value();
  c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.query_type=type,.output_type=type,
      .key_storage=storage,.value_storage=storage,.max_requests=3,.max_query_rows=2,.max_kv_rows=2049,
      .max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.qk_head_dim=depth,
      .value_head_dim=value_depth,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();
  auto context=c::cpu::create_context({.native_threads=2}).value();auto queue=context.create_queue().value();
  const auto kd=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value();
  const auto vd=c::batch_cache_description(cap,c::BatchCacheRole::kValue).value();
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),kd.required_bytes().value()+vd.required_bytes().value()+2*8*depth*2+256).value();
  const auto code=[](unsigned row,unsigned head){return int((row*3+head*5)%63)-31;};
  const auto make=[&](bool value)->c::ReadLease{
    const auto desc=value?vd:kd;const auto width=value?value_depth:depth;
    auto writer=pool.allocate(desc.required_bytes().value(),64).value();
    if(encoded){
      std::vector<std::byte> bytes(desc.required_bytes().value(),std::byte{0});
      for(unsigned row=0;row<2049;++row)for(unsigned head=0;head<2;++head)for(unsigned block=0;block<width/32;++block){
        const auto offset=((row*2+head)*(width/32)+block)*34;
        const std::uint16_t scale=row==2048?0x7e00:value?0x2400:0;std::memcpy(bytes.data()+offset,&scale,2);
        const auto sample=std::uint8_t(value?code(row,head):0);
        std::fill_n(bytes.begin()+offset+2,32,std::byte{sample});
      }
      EXPECT_TRUE(writer.view()->write(bytes).ok());
    }else{
      std::vector<T> words(2049*2*width);
      for(unsigned row=0;row<2049;++row)for(unsigned head=0;head<2;++head)for(unsigned d=0;d<width;++d)
        words[(row*2+head)*width+d]=half_test::encode<T>(row==2048?std::numeric_limits<float>::quiet_NaN():value?float(code(row,head))/64.F:0.F);
      EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(words))).ok());
    }
    return writer.publish().value();
  };
  const auto key=make(false),value=make(true);
  auto qw=pool.allocate(2*8*depth*2).value();ASSERT_TRUE(qw.view()->fill(std::byte{0}).ok());
  const std::array inputs{qw.publish().value(),key,value};
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=8u<<20,
      .max_int_workspace_bytes=4096,.scale=.125F,.causal=causal,.window_left=window,.allocator=fault.ref()}).value();
  auto slot=c::BatchInvocation::create(program,queue).value(),other=c::BatchInvocation::create(program,queue).value();
  c::ReadLease retained;std::vector<std::int32_t> previous;std::size_t workspace=0;
  for(unsigned step=0;step<64;++step){
    SCOPED_TRACE(::testing::Message()<<"ragged encoded="<<encoded<<" d="<<depth<<" dv="<<value_depth<<" requests="<<requested<<" step="<<step);
    unsigned count=requested;if(requested==2 && step>=16 && step<24)count=1;
    if(requested==2 && step>=32 && step<48)count=3;
    std::vector<std::int32_t> q{0},k{0};unsigned active=0;
    for(unsigned request=0;request<count;++request){
      const bool live=count!=3 || request!=1;q.push_back(q.back()+(live?1:0));
      const unsigned rows=live?1+(step*(17+active*6)+active*509)%1024:0;
      k.push_back(k.back()+int(rows));if(live)++active;
    }
    auto layout=schema.version({q,k,{},{}}).value();
    if(step==0){auto prepared=other.prepare_batch(layout).value();auto done=prepared.run(inputs).value();ASSERT_TRUE(done.wait().ok());retained=other.output().value();}
    const bool reuse=!previous.empty() && previous==q;const auto before=fault.attempts;if(reuse)fault.fail=before;
    auto prepared=slot.update_batch(layout);fault.fail=SIZE_MAX;ASSERT_TRUE(prepared)<<prepared.error().message();
    EXPECT_EQ(slot.report().reused_native_plan,reuse);
    if(reuse){EXPECT_EQ(fault.attempts,before);EXPECT_EQ(slot.report().float_workspace_bytes,workspace);}
    workspace=slot.report().float_workspace_bytes;previous=q;
    auto done=prepared->run(inputs);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    auto output=slot.output().value();*done={};*prepared={};EXPECT_FALSE(slot.update_batch(layout));
    std::vector<T> actual(2*8*value_depth);ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    for(unsigned request=0;request<count;++request){if(q[request]==q[request+1])continue;
      const auto rows=unsigned(k[request+1]-k[request]);
      const auto first=window<0?0:rows>unsigned(window+1)?rows-unsigned(window+1):0;
      for(unsigned head=0;head<8;++head){double expected=0;
        for(unsigned row=first;row<rows;++row)expected+=double(code(unsigned(k[request])+row,head/4))/64.;
        expected/=double(rows-first);
        for(unsigned d=0;d<value_depth;++d)EXPECT_NEAR(half_test::decode(actual[(q[request]*8+head)*value_depth+d]),expected,type==c::DataType::kFloat16?.002:.008);
      }
    }
    std::vector<T> old(2*8*value_depth);ASSERT_TRUE(retained.view()->read(std::as_writable_bytes(std::span(old))).ok());
    for(unsigned head=0;head<8;++head)for(unsigned d=0;d<value_depth;++d)
      EXPECT_NEAR(half_test::decode(old[head*value_depth+d]),double(code(0,head/4))/64.,type==c::DataType::kFloat16?.002:.008);
  }
  retained={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(other.close().ok());ASSERT_TRUE(context.drain().ok());
  }
  EXPECT_EQ(fault.live,0u);
}
TEST(GgmlCpuRagged, RuntimeQ1KeepsGraphAcrossPackedKvExtents){
  for(unsigned requests:{1U,2U})for(bool encoded:{false,true})for(unsigned d:{64U,128U,256U}){
    runtime_ragged_q1<F16>(encoded,d,d,-1,requests);runtime_ragged_q1<BF16>(encoded,d,d,-1,requests);
  }
  for(unsigned requests:{1U,2U})for(bool encoded:{false,true}){
    runtime_ragged_q1<F16>(encoded,64,128,16,requests);runtime_ragged_q1<BF16>(encoded,64,128,16,requests);
  }
}
template<class T> void exercise(int depth, bool causal) {
  constexpr int heads=8, kv_heads=2, queries=18, keys=78;
  auto context=c::cpu::create_context({.native_threads=1}).value();
  auto queue=context.create_queue().value();
  const c::BatchCapacity capacity{.storage=c::BatchStorage::kRagged,
      .query_type=half_test::data_type<T>(),.output_type=half_test::data_type<T>(),.key_storage=c::CacheStorage::plain(half_test::data_type<T>()).value(),.value_storage=c::CacheStorage::plain(half_test::data_type<T>()).value(),
      .max_requests=3,.max_query_rows=20,.max_kv_rows=84,.max_pages=0,.max_page_refs=0,
      .query_heads=heads,.kv_heads=kv_heads,.qk_head_dim=static_cast<std::uint32_t>(depth),.value_head_dim=static_cast<std::uint32_t>(depth),.page_size=1};
  auto schema_result=c::BatchSchema::create(capacity);ASSERT_TRUE(schema_result)<<schema_result.error().message();
  auto schema=std::move(*schema_result);
  const std::array<std::int32_t,4> qp{0,1,18,18},kp{0,31,78,78};
  auto layout_result=schema.version({qp,kp,{},{}});ASSERT_TRUE(layout_result)<<layout_result.error().message();
  auto layout=std::move(*layout_result);
  auto program_result=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,.max_invocations=2,
      .scale=1.F/std::sqrt(float(depth)),.causal=causal});ASSERT_TRUE(program_result)<<program_result.error().message();
  auto program=std::move(*program_result);
  auto first_result=c::BatchInvocation::create(program,queue);ASSERT_TRUE(first_result)<<first_result.error().message();
  auto first=std::move(*first_result);
  auto second_result=c::BatchInvocation::create(program,queue);ASSERT_TRUE(second_result)<<second_result.error().message();
  auto second=std::move(*second_result);
  auto foreign=c::BatchSchema::create(capacity).value();
  auto foreign_layout=foreign.version({qp,kp,{},{}}).value();
  auto rejected=first.prepare_batch(foreign_layout);
  ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsInvalidArgument());
  std::vector<T> q(20*heads*depth),k(84*kv_heads*depth),v(k.size());
  for(std::size_t i=0;i<q.size();++i)q[i]=encode<T>(float(int(i%13)-6)*.03125F);
  for(std::size_t i=0;i<k.size();++i){k[i]=encode<T>(float(int(i%17)-8)*.015625F);v[i]=encode<T>(float(int(i%29)-14)*.03125F);}
  for(int row=keys;row<84;++row)for(int z=0;z<kv_heads*depth;++z)v[row*kv_heads*depth+z]=encode<T>(128.F);
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),(q.size()+k.size()+v.size())*sizeof(T)+2048).value();
  const std::array inputs{load(pool,q),load(pool,k),load(pool,v)};
  const auto expected=[&](auto query,auto kv) {
    std::vector<double> result(queries*heads*depth),weights;
    for(int r=0;r<2;++r){const auto nq=query[r+1]-query[r],nk=kv[r+1]-kv[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(int head=0;head<heads;++head){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(int d=0;d<depth;++d)
          dot+=decode(q[((query[r]+row)*heads+head)*depth+d])*decode(k[((kv[r]+j)*kv_heads+head/4)*depth+d]);
          weights[j]=causal&&j>nk-nq+row?0.:std::exp(dot/double(std::sqrt(float(depth))));sum+=weights[j];}
        for(int d=0;d<depth;++d)for(int j=0;j<nk;++j)result[((query[r]+row)*heads+head)*depth+d]+=
          weights[j]/sum*decode(v[((kv[r]+j)*kv_heads+head/4)*depth+d]);}}
    return result;
  };
  const auto verify=[&](const c::ReadLease &lease, const auto &oracle) {
    std::vector<T> actual(20*heads*depth);ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    double maximum=0;for(std::size_t i=0;i<oracle.size();++i){ASSERT_TRUE(std::isfinite(decode(actual[i])));maximum=std::max(maximum,std::abs(decode(actual[i])-oracle[i]));}
    const auto tolerance=std::is_same_v<T,float>?6e-6:(std::is_same_v<T,F16>?2e-3:8e-3);
    EXPECT_LE(maximum,tolerance);
  };
  const auto original=expected(qp,kp);
  auto prepared_result=first.prepare_batch(layout);ASSERT_TRUE(prepared_result)<<prepared_result.error().message();
  auto prepared=std::move(*prepared_result);auto done_result=prepared.run(inputs);ASSERT_TRUE(done_result)<<done_result.error().message();
  auto done=std::move(*done_result);ASSERT_TRUE(done.wait().ok());
  auto output=first.output().value();verify(output,original);
  EXPECT_GT(first.report().float_workspace_bytes,0U);EXPECT_TRUE(first.report().native_metadata_unknown);
  auto independent_result=second.prepare_batch(layout);ASSERT_TRUE(independent_result)<<independent_result.error().message();
  auto independent=std::move(*independent_result);auto completed_result=independent.run(inputs);ASSERT_TRUE(completed_result)<<completed_result.error().message();
  auto completed=std::move(*completed_result);ASSERT_TRUE(completed.wait().ok());
  auto other=second.output().value();verify(other,original);
  prepared={};done={};EXPECT_TRUE(first.update_batch(layout).error().IsUnavailable());output.reset();
  auto same=schema.version({qp,kp,{},{}}).value();auto reused_result=first.update_batch(same);ASSERT_TRUE(reused_result)<<reused_result.error().message();
  auto reused=std::move(*reused_result);
  EXPECT_TRUE(first.report().reused_native_plan);auto repeated=reused.run(inputs).value();ASSERT_TRUE(repeated.wait().ok());
  auto replay=first.output().value();verify(replay,original);reused={};repeated={};replay.reset();
  const std::array<std::int32_t,4> next_q{0,17,18,18};auto changed=schema.version({next_q,kp,{},{}}).value();
  auto rebuilt_result=first.update_batch(changed);ASSERT_TRUE(rebuilt_result)<<rebuilt_result.error().message();
  auto rebuilt=std::move(*rebuilt_result);EXPECT_FALSE(first.report().reused_native_plan);
  auto final=rebuilt.run(inputs).value();ASSERT_TRUE(final.wait().ok());auto reordered=first.output().value();verify(reordered,expected(next_q,kp));
  for(int port=1;port<3;++port){std::vector<T> after(port==1?k.size():v.size());ASSERT_TRUE(inputs[port].view()->read(std::as_writable_bytes(std::span(after))).ok());
    const auto &before=port==1?k:v;EXPECT_EQ(std::memcmp(after.data(),before.data(),after.size()*sizeof(T)),0);}
  rebuilt={};final={};reordered.reset();independent={};completed={};other.reset();
  ASSERT_TRUE(first.close().ok());ASSERT_TRUE(second.close().ok());ASSERT_TRUE(context.close().ok());
}
TEST(GgmlCpuRagged, NativeFlashReadsTwoRequestsWithoutKvMaterialization) {
  exercise<F16>(256,true);exercise<BF16>(256,true);exercise<F16>(64,false);exercise<BF16>(64,false);
  exercise<F16>(128,true);exercise<BF16>(128,true);
}
TEST(GgmlCpuRagged, ColdBudgetAndPagedQualificationAreExplicit) {
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=1})).value();
  auto queue=context.create_queue().value();
  c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_query_rows=1,.max_kv_rows=3,
      .max_pages=0,.max_page_refs=0,.query_heads=4,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,2> q{0,1},k{0,3};
  auto layout=schema.version({q,k,{},{}}).value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=1,.max_int_workspace_bytes=1024}).value();
  auto invocation=c::BatchInvocation::create(program,queue).value();auto prepared=invocation.prepare_batch(layout);
  ASSERT_FALSE(prepared);EXPECT_TRUE(prepared.error().IsResourceExhausted());EXPECT_FALSE(invocation.in_flight());
  ASSERT_TRUE(invocation.close().ok());
  cap.storage=c::BatchStorage::kPaged;cap.max_pages=2;cap.max_page_refs=2;cap.page_size=16;cap.max_kv_rows=32;
  auto paged=c::BatchSchema::create(cap).value();auto absent=c::BatchProgram::prepare(context,paged,{.resource=hyle::mem::system_resource()});
  ASSERT_TRUE(absent)<<absent.error().message();
  const std::array<std::int32_t,2> page_ptr{0,1};
  const std::array<std::int32_t,1> page_id{1},tail{3};
  auto page_layout=paged.version({q,page_ptr,page_id,tail}).value();
  auto small=c::BatchProgram::prepare(context,paged,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=1,.max_int_workspace_bytes=1024}).value();
  auto page_slot=c::BatchInvocation::create(small,queue).value();
  auto refused=page_slot.prepare_batch(page_layout);ASSERT_FALSE(refused);EXPECT_TRUE(refused.error().IsResourceExhausted());
  EXPECT_FALSE(page_slot.in_flight());ASSERT_TRUE(page_slot.close().ok());
  ASSERT_TRUE(context.close().ok());
}
TEST(GgmlCpuRagged, EveryColdAllocationFailureReleasesThePartialNativePlan) {
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=1})).value();
  auto queue=context.create_queue().value();
  for(auto storage:{c::BatchStorage::kRagged,c::BatchStorage::kPaged}){
  SCOPED_TRACE(static_cast<unsigned>(storage));
  c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=2,.max_query_rows=2,.max_kv_rows=6,
      .max_pages=0,.max_page_refs=0,.query_heads=4,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  const bool paged=storage==c::BatchStorage::kPaged;
  if(paged){cap.storage=storage;cap.max_pages=2;cap.max_page_refs=2;cap.page_size=16;cap.max_kv_rows=32;}
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,3> q{0,1,2},k{0,paged?1:3,paged?2:6};
  const std::array<std::int32_t,2> ids{1,0},tails{3,3};
  auto layout=schema.version({q,k,paged?std::span<const std::int32_t>{ids}:std::span<const std::int32_t>{},paged?std::span<const std::int32_t>{tails}:std::span<const std::int32_t>{}}).value();
  const auto episode=[&](hyle_f5_fault_fixture::Allocator &allocator)->hyle::Status {
    auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),
        .max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=2048,.allocator=allocator.ref()});
    if(!program)return program.error();
    auto invocation=c::BatchInvocation::create(*program,queue);if(!invocation)return invocation.error();
    auto prepared=invocation->prepare_batch(layout);
    const hyle::Status status=prepared?hyle::Status::Ok():hyle::Status(prepared.error());
    if(prepared)*prepared={};
    const auto closed=invocation->close();if(!closed.ok())return closed;
    return status;
  };
  hyle_f5_fault_fixture::Allocator census;ASSERT_TRUE(episode(census).ok());EXPECT_EQ(census.live,0U);
  ASSERT_GT(census.attempts,0U);
  for(std::size_t point=0;point<census.attempts;++point){SCOPED_TRACE(point);
    hyle_f5_fault_fixture::Allocator fault;fault.fail=point;
    const auto status=episode(fault);EXPECT_TRUE(status.IsResourceExhausted())<<status.message();
    EXPECT_EQ(fault.live,0U);
  }
  }
  ASSERT_TRUE(context.close().ok());
}
void accepted_failure(bool paged) {
  auto context=c::Context::create(c::cpu::device(),c::ggml_cpu::provider({.native_threads=1})).value();
  auto queue=context.create_queue().value();
  c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=2,.max_query_rows=2,.max_kv_rows=6,
      .max_pages=0,.max_page_refs=0,.query_heads=4,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  if(paged){cap.storage=c::BatchStorage::kPaged;cap.max_pages=2;cap.max_page_refs=2;cap.page_size=16;cap.max_kv_rows=32;}
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,3> q{0,1,2},k{0,paged?1:3,paged?2:6};
  const std::array<std::int32_t,2> ids{1,0},tails{3,3};
  auto layout=schema.version({q,k,paged?std::span<const std::int32_t>{ids}:std::span<const std::int32_t>{},paged?std::span<const std::int32_t>{tails}:std::span<const std::int32_t>{}}).value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),
      .max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=2048}).value();
  auto invocation=c::BatchInvocation::create(program,queue).value();auto prepared=invocation.prepare_batch(layout).value();
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),8192).value();
  const std::array inputs{load(pool,std::vector<F16>(2*4*64)),load(pool,std::vector<F16>(6*2*64)),load(pool,std::vector<F16>(6*2*64))};
  c::PagedArena keys,values;c::PagedArenaRead key_frame,value_frame;
  std::array<c::ReadLease,2> key_guards,value_guards;
  if(paged){
    auto key_arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,hyle::mem::system_resource(),{.max_backing_bytes=8193});
    auto value_arena=c::PagedArena::create(schema,c::BatchCacheRole::kValue,hyle::mem::system_resource(),{.max_backing_bytes=8193});
    ASSERT_TRUE(key_arena)<<key_arena.error().message();ASSERT_TRUE(value_arena)<<value_arena.error().message();
    keys=std::move(*key_arena);values=std::move(*value_arena);
    const std::vector<F16> zero(16*2*64);
    for(unsigned page=0;page<2;++page){
      auto kw=keys.acquire(page).value(),vw=values.acquire(page).value();
      ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(zero))).ok());
      ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(zero))).ok());
      auto ready_key=kw.publish(cap.page_size),ready_value=vw.publish(cap.page_size);
      ASSERT_TRUE(ready_key)<<ready_key.error().message();ASSERT_TRUE(ready_value)<<ready_value.error().message();
      key_guards[page]=std::move(*ready_key);value_guards[page]=std::move(*ready_value);
    }
    key_frame=keys.read(layout,key_guards).value();value_frame=values.read(layout,value_guards).value();
  }
  hyle::Result<c::Completion> completion;
  {AcceptedFailure failure(context);completion=paged?prepared.run(inputs[0],key_frame,value_frame):prepared.run(inputs);}
  const hyle::Status terminal=completion?completion->wait():hyle::Status(completion.error());EXPECT_TRUE(terminal.IsResourceExhausted());
  EXPECT_FALSE(invocation.output());EXPECT_FALSE(invocation.in_flight());EXPECT_EQ(pool.stats().active_uses,0U);
  if(completion)*completion={};prepared={};
  EXPECT_FALSE(invocation.update_batch(layout));EXPECT_TRUE(invocation.drain().IsResourceExhausted());
  if(paged){
    key_frame={};value_frame={};for(auto &lease:key_guards)lease.reset();for(auto &lease:value_guards)lease.reset();
    for(unsigned page=0;page<2;++page){auto kw=keys.acquire(page),vw=values.acquire(page);EXPECT_TRUE(kw);EXPECT_TRUE(vw);}
    ASSERT_TRUE(keys.close().ok());ASSERT_TRUE(values.close().ok());
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(GgmlCpuRagged, AcceptedMathFailureRetiresUsesWithoutPublishingOutput) {
  accepted_failure(false);accepted_failure(true);
}
} // namespace
