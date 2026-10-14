#pragma once
#if defined(HYLE_G01_PAGE_RISK_APPLE)
#include <hyle/compute/apple_batch.h>
#else
#include <hyle/compute/vulkan_batch.h>
#endif
#include <hyle/mem/quota_allocator.h>
#include <hyle/mem/buffer.h>
#include <hyle/simd/dtype.h>
#include "allocation_fault_fixture.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hyle_g01_page_risk {
namespace c=hyle::compute;
#if defined(HYLE_G01_PAGE_RISK_APPLE)
namespace n=c::apple;
inline constexpr const char* route="Apple";
#else
namespace n=c::vulkan;
inline constexpr const char* route="Vulkan";
#endif
using hyle::Status;using hyle::Result;
inline Status require(bool good,const char* message){return good?Status::Ok():Status::Internal(message);}
struct Inputs final {
  c::BatchSchema schema;c::BatchLayoutVersion layout,next;
  c::NativePagedBatchProgram recipe;c::PagedArena keys,values;
  c::LeasePool query_pool;c::ReadLease query;
  std::array<c::ReadLease,2> key_reads,value_reads;
  c::PagedArenaRead key_frame,value_frame,next_keys,next_values;
  void drop_external()noexcept{query={};key_reads={};value_reads={};key_frame={};value_frame={};next_keys={};next_values={};}
};
inline Result<Inputs> inputs(n::Context& context,n::MemoryResource& device,n::MemoryResource& staging){
  Inputs data;
  const c::BatchCapacity cap{.storage=c::BatchStorage::kPaged,.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),
      .max_requests=2,.max_query_rows=2,.max_kv_rows=48,.max_pages=4,.max_page_refs=4,
      .query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(data.schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,3> qp{0,1,2},kp{0,2,3};
  const std::array<std::int32_t,3> pages{2,0,2},next_pages{0,2,0};
  const std::array<std::int32_t,2> tails{7,15};
  HYLE_ASSIGN_OR_RETURN(data.layout,data.schema.version({qp,kp,pages,tails}));
  HYLE_ASSIGN_OR_RETURN(data.next,data.schema.version({qp,kp,next_pages,tails}));
  HYLE_ASSIGN_OR_RETURN(data.recipe,c::NativePagedBatchProgram::prepare(data.schema,
      {.max_host_metadata_bytes=4096,.scale=.125F}));
  HYLE_ASSIGN_OR_RETURN(data.keys,c::PagedArena::create(data.schema,c::BatchCacheRole::kKey,device.ref(),{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(data.values,c::PagedArena::create(data.schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=1<<20}));
  const auto page_bytes=data.keys.page_bytes();
  HYLE_ASSIGN_OR_RETURN(data.query_pool,c::LeasePool::create(device.ref(),4096));
  HYLE_ASSIGN_OR_RETURN(auto host,c::LeasePool::create(staging.ref(),2*page_bytes+512));
  const std::array specs{c::CopyResourceSpec{staging.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(program));
  const auto upload=[&](c::WriteLease& writer,std::span<const std::uint16_t> words)->Result<c::ReadLease>{
    HYLE_RETURN_IF_ERROR(copy.reset());
    HYLE_ASSIGN_OR_RETURN(auto source,host.allocate(words.size_bytes(),64));
    HYLE_RETURN_IF_ERROR(source.view()->write(std::as_bytes(words)));
    HYLE_ASSIGN_OR_RETURN(auto ready,source.publish());
    HYLE_RETURN_IF_ERROR(copy.bind_source({0},ready));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer)));
    const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,words.size_bytes()}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
    HYLE_RETURN_IF_ERROR(submitted.wait());return submitted.output({0},{1},0,words.size_bytes());
  };
  std::vector<std::uint16_t> words(page_bytes/2);
  for(unsigned i=0;i<words.size();++i)words[i]=hyle::simd::f32_to_f16(float(int(i%13)-6)/32.F).bits;
  HYLE_ASSIGN_OR_RETURN(auto qw,data.query_pool.allocate(2048,64));
  HYLE_ASSIGN_OR_RETURN(data.query,upload(qw,std::span(words).first(1024)));
  for(unsigned i=0;i<2;++i){
    const unsigned page=i?2:0;
    HYLE_ASSIGN_OR_RETURN(auto kw,data.keys.acquire(page));HYLE_ASSIGN_OR_RETURN(data.key_reads[i],upload(kw.lease(),words));HYLE_RETURN_IF_ERROR(kw.complete(data.key_reads[i],cap.page_size));
    for(unsigned j=0;j<words.size();++j)words[j]=hyle::simd::f32_to_f16(float((j+page)%17)/32.F).bits;
    HYLE_ASSIGN_OR_RETURN(auto vw,data.values.acquire(page));HYLE_ASSIGN_OR_RETURN(data.value_reads[i],upload(vw.lease(),words));HYLE_RETURN_IF_ERROR(vw.complete(data.value_reads[i],cap.page_size));
  }
  HYLE_ASSIGN_OR_RETURN(data.key_frame,data.keys.read(data.layout,data.key_reads));
  HYLE_ASSIGN_OR_RETURN(data.value_frame,data.values.read(data.layout,data.value_reads));
  HYLE_ASSIGN_OR_RETURN(data.next_keys,data.keys.read(data.next,data.key_reads));
  HYLE_ASSIGN_OR_RETURN(data.next_values,data.values.read(data.next,data.value_reads));return data;
}
inline Result<n::PagedBatchInvocation> create(n::Context& context,n::MemoryResource& device,
    n::MemoryResource& staging,const Inputs& data,hyle::mem::AllocatorRef allocator=hyle::mem::system_allocator()){
  return n::PagedBatchInvocation::create(context,data.recipe,device,staging,
      {.max_metadata_backing_bytes=4096,.allocator=allocator});
}
inline Status reusable_pages(Inputs& data){
  HYLE_ASSIGN_OR_RETURN(auto k0,data.keys.acquire(0));HYLE_ASSIGN_OR_RETURN(auto k2,data.keys.acquire(2));
  HYLE_ASSIGN_OR_RETURN(auto v0,data.values.acquire(0));HYLE_ASSIGN_OR_RETURN(auto v2,data.values.acquire(2));
  return require(data.query_pool.stats().active_uses==0,"native query guard did not retire");
}
inline Status cold_failures(n::Context& context,n::MemoryResource& device,n::MemoryResource& staging,Inputs& data){
  hyle_f5_fault_fixture::Allocator fault;
  {HYLE_ASSIGN_OR_RETURN(auto good,create(context,device,staging,data,fault.ref()));}
  const auto count=fault.attempts;HYLE_RETURN_IF_ERROR(require(fault.live==0,"cold create baseline metadata leaked"));
  for(std::size_t nth=0;nth<count;++nth){
    fault.attempts=0;fault.fail=nth;
    auto made=create(context,device,staging,data,fault.ref());
    HYLE_RETURN_IF_ERROR(require(!made && made.error().IsResourceExhausted() && fault.live==0,"cold controller allocation failure leaked/published"));
  }
  fault.fail=SIZE_MAX;fault.attempts=0;
  {HYLE_ASSIGN_OR_RETURN(auto slot,create(context,device,staging,data,fault.ref()));
    HYLE_RETURN_IF_ERROR(slot.prepare_batch(data.layout));
    fault.fail=fault.attempts;
    const auto denied=slot.prepare_batch(data.next);
    HYLE_RETURN_IF_ERROR(require(denied.IsResourceExhausted() && slot.report().layout_generation==data.layout.generation(),"cold native allocation rejection changed generation"));
    fault.fail=SIZE_MAX;
    HYLE_RETURN_IF_ERROR(slot.bind(data.query,data.key_frame,data.value_frame));
    HYLE_ASSIGN_OR_RETURN(auto done,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,done.wait());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
    HYLE_RETURN_IF_ERROR(require(result->arrays().size()==1,"old prepared recipe lost after cold native rejection"));
#else
    HYLE_RETURN_IF_ERROR(require(result->tensors().size()==1,"old prepared recipe lost after cold native rejection"));
#endif
  }
  HYLE_RETURN_IF_ERROR(require(fault.live==0,"cold prepare rejection metadata leaked"));
  std::printf("%s native page cold create all %zu allocation faults and rejected prepare/old generation PASS\n",route,count);return Status::Ok();
}
inline Status domain_checks(n::Context& context,n::MemoryResource& device,n::MemoryResource& staging,Inputs& data){
  auto too_small=n::PagedBatchInvocation::create(context,data.recipe,device,staging,
      {.max_metadata_backing_bytes=236});
  HYLE_RETURN_IF_ERROR(require(!too_small && too_small.error().IsResourceExhausted(),"fixed page metadata backing cap ignored"));
  HYLE_ASSIGN_OR_RETURN(auto slot,create(context,device,staging,data));
  const std::array<std::int32_t,3> empty_ptr{0,0,0};const std::array<std::int32_t,2> empty_tails{0,0};
  HYLE_ASSIGN_OR_RETURN(auto empty,data.schema.version({empty_ptr,empty_ptr,{},empty_tails}));
  HYLE_RETURN_IF_ERROR(slot.prepare_batch(empty));
  HYLE_RETURN_IF_ERROR(require(slot.bind(data.query,data.key_frame,data.value_frame).IsInvalidArgument(),"empty page episode accepted mismatched live KV frames"));
  HYLE_RETURN_IF_ERROR(slot.bind(data.query,{},{}));
  {HYLE_ASSIGN_OR_RETURN(auto done,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,done.wait());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
    HYLE_RETURN_IF_ERROR(require(result->arrays().size()==1 && result->arrays()[0].size()==0,"empty page output is not empty"));
#else
    HYLE_RETURN_IF_ERROR(require(result->tensors().size()==1 && result->tensors()[0].description().element_count()==std::size_t{0},"empty page output is not empty"));
#endif
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(data.layout).ok(),"empty old result allowed slot reuse"));
  }
  HYLE_RETURN_IF_ERROR(slot.prepare_batch(data.layout));
  HYLE_ASSIGN_OR_RETURN(auto wrong_buffer,hyle::mem::Buffer::create(hyle::mem::system_resource(),2048,64));
  HYLE_RETURN_IF_ERROR(wrong_buffer.view().fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto wrong_query,c::ReadLease::adopt(std::move(wrong_buffer)));
  HYLE_RETURN_IF_ERROR(require(slot.bind(wrong_query,data.key_frame,data.value_frame).IsInvalidArgument() && !slot.submit(),"foreign source resource accepted page binding"));
  HYLE_RETURN_IF_ERROR(slot.bind(data.query,data.key_frame,data.value_frame));
  {HYLE_ASSIGN_OR_RETURN(auto done,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,done.wait());}
  std::printf("%s native page exact backing cap, all-empty output/old result and foreign resource refusal PASS\n",route);return Status::Ok();
}
} // namespace hyle_g01_page_risk
