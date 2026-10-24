#pragma once
#include "hyle/compute/row_cache_paged.h"
#include "row_cache_fixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>
#include <string>

namespace row_cache_paged_test {
namespace c=hyle::compute;namespace m=hyle::mem;
template<class T>inline T take(hyle::Result<T> result,const char*what){
  if(!result)throw std::runtime_error(std::string(what)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline c::RowCacheBatchCapacity capacity(c::RowCacheEncoding encoding){
  return {.encoding=encoding,.max_requests=3,.max_query_rows=4,.max_kv_rows=8,
          .max_pages=5,.max_page_refs=4,.page_size=2};
}
inline c::RowCacheLayoutVersion layout(const c::RowCacheBatchSchema&s,int phase=0){
  const std::array<std::int32_t,4> q{0,2,2,4},kv{0,2,3,4};
  const std::array<std::int32_t,4> pages=phase?std::array<std::int32_t,4>{1,4,3,2}:std::array<std::int32_t,4>{4,1,3,0};
  const std::array<std::int32_t,3> tails{1,2,2};return take(s.version({q,kv,pages,tails}),"layout");
}
inline std::array<std::int32_t,12> ids(int phase=0){
  return phase?std::array<std::int32_t,12>{0,2,1,2,0,2,0,1,0,1,0,1}:
               std::array<std::int32_t,12>{2,0,2,1,2,0,1,0,1,0,1,0};
}
inline row_cache_test::Reference data(c::RowCacheEncoding encoding){
  auto result=row_cache_test::reference(encoding,row_cache_test::input_values(encoding,0));
  const auto next=row_cache_test::reference(encoding,row_cache_test::input_values(encoding,1));
  result.bytes.insert(result.bytes.end(),next.bytes.begin(),next.bytes.end());
  result.decoded.insert(result.decoded.end(),next.decoded.begin(),next.decoded.end());return result;
}
inline std::vector<c::ReadLease> pages(c::PagedArena&arena,c::RowCacheEncoding encoding,
    const row_cache_test::Reference&reference){
  std::vector<c::ReadLease> result;
  for(unsigned p=0;p<5;++p){auto writer=arena.acquire(p).value();
    EXPECT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(reference.bytes).subspan(2*p*encoding.row_bytes(),2*encoding.row_bytes()))).ok());
    result.push_back(writer.publish(2).value());}
  return result;
}
inline c::PagedRowCacheRead frame(c::PagedArena&arena,c::RowCacheLayoutVersion version,
    const std::vector<c::ReadLease>&pages,int phase=0){
  // Exact footprint, intentionally unrelated to page-table order.
  const std::array source=phase?std::array{pages[3],pages[2],pages[1],pages[4]}:
                                std::array{pages[0],pages[3],pages[1],pages[4]};
  return arena.read(std::move(version),source).value();
}
inline c::ReadLease load(c::LeasePool&pool,std::span<const std::byte> bytes){
  auto writer=pool.allocate(std::max<std::size_t>(4,bytes.size())).value();
  if(!bytes.empty())EXPECT_TRUE(writer.view()->subview(0,bytes.size())->write(bytes).ok());return writer.publish().value();
}
inline std::vector<std::uint16_t> read(const c::WriteLease&lease,std::size_t count){
  std::vector<std::uint16_t> result(count);EXPECT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(result))).ok());return result;
}
inline std::vector<std::uint16_t> expected(c::RowCacheEncoding encoding,
    const row_cache_test::Reference&source,std::span<const std::int32_t>selected,int phase=0){
  std::vector<std::uint16_t> result;
  for(std::size_t i=0;i<selected.size();++i){const bool second=i/3>=2;const auto id=selected[i];
    const auto page=second?(phase?2:0):(id<2?(phase?1:4):(phase?4:1));
    const auto physical=page*2+id%2;const auto first=source.decoded.begin()+physical*encoding.depth();
    result.insert(result.end(),first,first+encoding.depth());}
  return result;
}
inline void exercise(c::Context&context,m::MemoryResourceRef resource){
  for(auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()}){
    SCOPED_TRACE(encoding.depth());auto queue=take(context.create_queue(),"queue");
    auto schema=take(c::RowCacheBatchSchema::create(capacity(encoding)),"schema");
    auto arena=take(c::PagedArena::create(schema,resource,{.max_backing_bytes=10*encoding.row_bytes()+1}),"arena");
    const auto reference=data(encoding);auto ready=pages(arena,encoding,reference);
    const auto desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{4,3}).value();
    EXPECT_TRUE(c::PagedRowCacheGatherInvocation::create(context,queue,schema,desc,
        {.resource=resource,.max_backing_bytes=1}).error().IsResourceExhausted());
    auto slot=take(c::PagedRowCacheGatherInvocation::create(context,queue,schema,desc,{.resource=resource}),"reader");
    auto pool=c::LeasePool::create(resource,32768).value();
    auto output=pool.allocate(slot.output_description().required_bytes().value()).value();
    std::vector<std::uint16_t> old;
    for(int phase:{0,1}){auto version=layout(schema,phase);auto selected=ids(phase);
      auto input=load(pool,std::as_bytes(std::span(selected)));auto owned=frame(arena,version,ready,phase);
      ASSERT_TRUE(slot.prepare_layout(version).ok());ASSERT_TRUE(slot.prepare_layout(version).ok());
      EXPECT_TRUE(slot.submit(frame(arena,layout(schema,phase),ready,phase),input,output).error().IsInvalidArgument());
      {auto done=slot.submit(owned,input,output);ASSERT_TRUE(done)<<done.error().message();
        EXPECT_TRUE(slot.prepare_layout(version).IsUnavailable());ASSERT_TRUE(done->wait().ok());}
      const auto actual=read(output,12*encoding.depth());EXPECT_EQ(actual,expected(encoding,reference,selected,phase));
      if(phase)EXPECT_NE(actual,old);else old=actual;
    }
    ASSERT_TRUE(slot.drain().ok());
    // Empty K permits a nonempty query grouping with no logical/physical KV.
    const std::array<std::int32_t,4> q{0,2,2,4},kv{};const std::array<std::int32_t,3> tails{};
    const auto empty=schema.version({q,kv,{},tails}).value();
    const auto empty_desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{4,0}).value();
    auto empty_slot=c::PagedRowCacheGatherInvocation::create(context,queue,schema,empty_desc,{.resource=resource}).value();
    ASSERT_TRUE(empty_slot.prepare_layout(empty).ok());auto no_ids=load(pool,{});auto no_output=pool.allocate(4).value();
    {auto done=empty_slot.submit(arena.read(empty,{}).value(),no_ids,no_output);ASSERT_TRUE(done);EXPECT_TRUE(done->wait().ok());}
    // A logical ID that is physically addressable must still fail before any output write.
    for(auto bad:{-1,3,4,2}){auto failed_queue=context.create_queue().value();
      auto failed=c::PagedRowCacheGatherInvocation::create(context,failed_queue,schema,desc,{.resource=resource}).value();
      auto version=layout(schema);ASSERT_TRUE(failed.prepare_layout(version).ok());auto selected=ids();selected[bad==2?11:5]=bad;
      auto input=load(pool,std::as_bytes(std::span(selected)));
      auto failed_output=pool.allocate(12*encoding.depth()*2).value();auto raw=failed_output.view().value();
      const std::vector<std::uint16_t> sentinel(12*encoding.depth(),0x5a5a);ASSERT_TRUE(raw.write(std::as_bytes(std::span(sentinel))).ok());
      {auto done=failed.submit(frame(arena,version,ready),input,failed_output);ASSERT_TRUE(done);EXPECT_TRUE(done->wait().IsOutOfRange());}
      // F1 poisons the failed writer; inspect the still-owned backing only
      // after actual completion to prove the codec never partially wrote it.
      std::vector<std::uint16_t> actual(sentinel.size());ASSERT_TRUE(raw.read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual,sentinel);EXPECT_FALSE(failed_output.publish());
    }
  }
}
} // namespace row_cache_paged_test
