#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/batch_materialize.h>
#include <hyle/compute/batch_paged_native.h>
#include <hyle/compute/detail/paged_arena_access.h>
#include <hyle/compute/detail/batch_provider.h>
#include "allocation_fault_fixture.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <type_traits>

namespace {
namespace c=hyle::compute;namespace m=hyle::mem;
static_assert(!std::is_convertible_v<c::RowCacheBatchSchema,c::BatchSchema>);
static_assert(!std::is_convertible_v<c::RowCacheLayoutVersion,c::BatchLayoutVersion>);
static_assert(!std::is_convertible_v<c::PagedRowCacheRead,c::PagedArenaRead>);
static_assert(sizeof(c::RowCacheBatchSchema)==sizeof(c::BatchSchema));
static_assert(sizeof(c::RowCacheLayoutVersion)==sizeof(c::BatchLayoutVersion));
static_assert(sizeof(c::PagedRowCacheRead)==sizeof(c::PagedArenaRead));
template<class Layout,class Frame>
concept CanReframe=requires(const c::PagedArena&arena,Layout version,std::span<const Frame> sources){
  arena.reframe(version,sources);
};
static_assert(CanReframe<c::RowCacheLayoutVersion,c::PagedRowCacheRead>);
static_assert(!CanReframe<c::RowCacheLayoutVersion,c::PagedArenaRead>);
static_assert(!CanReframe<c::BatchLayoutVersion,c::PagedRowCacheRead>);
c::RowCacheBatchCapacity capacity(c::RowCacheEncoding encoding){
  return {.encoding=encoding,.max_requests=2,.max_query_rows=3,.max_kv_rows=8,
          .max_pages=3,.max_page_refs=4,.page_size=2};
}
c::RowCacheLayoutVersion layout(const c::RowCacheBatchSchema&s,std::span<const std::int32_t> pages,int tail=2){
  const std::array<std::int32_t,2> q{0,3},kv{0,std::int32_t(pages.size())};
  const std::array<std::int32_t,1> tails{tail};return s.version({q,kv,pages,tails}).value();
}
c::ReadLease fill(c::PagedArena&a,unsigned page,std::byte value,unsigned rows=2){
  auto writer=a.acquire(page).value();EXPECT_TRUE(writer.lease().view()->fill(value).ok());return writer.publish(rows).value();
}
TEST(RowCachePaged, TypedGeometryExactPageGuardsAndInitializedPrefixes){
  for(auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()}){
    auto schema=c::RowCacheBatchSchema::create(capacity(encoding)).value();
    EXPECT_EQ(schema.capacity(),capacity(encoding));
    auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
    EXPECT_EQ(arena.page_bytes(),2u*encoding.row_bytes());
    auto zero=fill(arena,0,std::byte{11}),two=fill(arena,2,std::byte{29},1);
    auto version=layout(schema,std::array<std::int32_t,2>{0,2},1);
    auto frame=arena.read(version,std::array{two,zero}).value();
    EXPECT_EQ(frame.layout(),version);EXPECT_EQ(frame.layout().schema().capacity(),schema.capacity());EXPECT_EQ(frame.page_count(),2u);
    const auto tensor=c::detail::PagedArenaAccess::tensor(frame).value();
    EXPECT_EQ(tensor.description(),c::row_cache_paged_description(schema.capacity()).value());
    EXPECT_EQ(tensor.storage().size(),6u*encoding.row_bytes());
    EXPECT_TRUE(c::detail::PagedArenaAccess::validate_storage(frame,encoding).ok());
    const auto other=encoding==c::RowCacheEncoding::float4_block32()?c::RowCacheEncoding::float8_block64_tail64():c::RowCacheEncoding::float4_block32();
    EXPECT_TRUE(c::detail::PagedArenaAccess::validate_storage(frame,other).IsInvalidArgument());
    EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,2>{0,2}),std::array{zero,two}).error().IsFailedPrecondition());
    EXPECT_TRUE(arena.read(version,std::array{zero}).error().IsInvalidArgument());
    auto free=fill(arena,1,std::byte{37});
    EXPECT_TRUE(arena.read(version,std::array{zero,two,free}).error().IsInvalidArgument());
    EXPECT_TRUE(arena.read(version,std::array{zero,two.subview(0,encoding.row_bytes()).value()}).error().IsInvalidArgument());
    auto foreign=c::RowCacheBatchSchema::create(schema.capacity()).value();
    EXPECT_TRUE(arena.read(layout(foreign,std::array<std::int32_t,2>{0,2},1),std::array{zero,two}).error().IsInvalidArgument());
    auto other_arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
    auto foreign_page=fill(other_arena,0,std::byte{41});
    EXPECT_TRUE(arena.read(version,std::array{foreign_page,two}).error().IsInvalidArgument());
    auto repeated=arena.read(layout(schema,std::array<std::int32_t,2>{0,0}),std::array{zero,zero}).value();EXPECT_EQ(repeated.page_count(),1u);
    auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();
    for(const auto&page:c::detail::PagedArenaAccess::pages(frame))ASSERT_TRUE(use.read(page).ok());
    frame={};repeated={};zero.reset();two.reset();EXPECT_FALSE(arena.acquire(0));EXPECT_FALSE(arena.acquire(2));
    use={};EXPECT_TRUE(arena.acquire(0));EXPECT_TRUE(arena.acquire(2));
  }
}
TEST(RowCachePaged, EmptyHistoryPreservesRowGroupsAndHasNoPagePermission){
  auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float8_block64_tail64())).value();
  const std::array<std::int32_t,3> q{0,2,3},kv{0,0,0};const std::array<std::int32_t,2> tails{};
  auto version=schema.version({q,kv,{},tails}).value();
  EXPECT_EQ(version.facts().query_rows,3u);EXPECT_EQ(version.facts().kv_rows,0u);EXPECT_EQ(version.facts().page_refs,0u);
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto empty=arena.read(version,{}).value();EXPECT_TRUE(empty.valid());EXPECT_EQ(empty.page_count(),0u);EXPECT_TRUE(c::detail::PagedArenaAccess::pages(empty).empty());
  auto metadata=version.metadata().value();EXPECT_EQ(metadata.size(),schema.metadata_offsets().bytes);
  auto next=schema.version({q,kv,{},tails}).value();EXPECT_NE(version,next);EXPECT_GT(next.generation(),version.generation());
  auto page=fill(arena,0,std::byte{43});EXPECT_TRUE(arena.read(version,std::array{page}).error().IsInvalidArgument());
  EXPECT_TRUE(arena.acquire(page));
  // Escaped metadata remains its original independent readonly F1 owner.
  empty={};version={};next={};schema={};arena={};auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();EXPECT_TRUE(use.read(metadata).ok());
}
TEST(RowCachePaged, OriginalF2WholePageWriterAndOldBranchRemainIndependent){
  auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float4_block32())).value();
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto old=fill(arena,0,std::byte{17});auto old_version=layout(schema,std::array<std::int32_t,1>{0});auto old_frame=arena.read(old_version,std::array{old}).value();
  auto writer=arena.acquire(1).value();const auto generation=writer.lease().generation();
  auto context=c::Context::create(c::cpu::device()).value();auto queue=context.create_queue().value();
  auto program=c::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{m::system_resource(),arena.page_bytes()},c::CopyResourceSpec{m::system_resource(),arena.page_bytes()}}).value();
  auto copy=c::CopyBatchInvocation::create(program,queue).value();ASSERT_TRUE(copy.bind_source({0},old).ok());ASSERT_TRUE(copy.bind_mutable({1},std::move(writer.lease())).ok());
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,arena.page_bytes()}}};ASSERT_TRUE(copy.prepare_replay(phases,ranges).ok());
  {auto submitted=copy.submit().value();ASSERT_TRUE(submitted.wait().ok());EXPECT_FALSE(copy.unbind_mutable({1}));}
  writer.lease()=copy.unbind_mutable({1}).value();EXPECT_EQ(writer.lease().generation(),generation);
  auto next=writer.publish(2).value();auto next_version=layout(schema,std::array<std::int32_t,1>{1});
  EXPECT_TRUE(arena.read(next_version,std::array{next}));EXPECT_EQ(old_frame.layout(),old_version);
  std::array<std::byte,136> before{},after{};ASSERT_TRUE(old.view()->read(before).ok());ASSERT_TRUE(next.view()->read(after).ok());EXPECT_EQ(before,after);
  EXPECT_FALSE(arena.acquire(old));ASSERT_TRUE(copy.reset().ok());
}
TEST(RowCachePaged, ColdAllocationFailureAndWrongAttentionDomainsRemainRejected){
  hyle_f5_fault_fixture::Allocator allocator;
  {auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float8_block64_tail64()),{.allocator=allocator.ref()}).value();
    allocator.fail=allocator.attempts;EXPECT_TRUE(schema.version({}).error().IsInvalidArgument());
    const std::array<std::int32_t,2> q{0,3},kv{0,0};const std::array<std::int32_t,1> tail{0};
    EXPECT_TRUE(schema.version({q,kv,{},tail}).error().IsResourceExhausted());allocator.fail=SIZE_MAX;
    auto version=schema.version({q,kv,{},tail}).value();
    auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096,.allocator=allocator.ref()}).value();
    allocator.fail=allocator.attempts;EXPECT_TRUE(arena.read(version,{}).error().IsResourceExhausted());allocator.fail=SIZE_MAX;EXPECT_TRUE(arena.read(version,{}));
    // Trusted adapters cannot accidentally send the hidden common owner to an
    // old attention entry point. Public row handles have no such conversion.
    const auto &hidden=c::detail::BatchLayoutAccess::owner(schema);const auto &hidden_version=c::detail::BatchLayoutAccess::owner(version);
    EXPECT_FALSE(c::validate_batch_capacity(hidden.capacity()).ok());
    EXPECT_TRUE(c::validate_batch_attention_layout(hidden_version,false).IsInvalidArgument());
    EXPECT_TRUE(c::batch_cache_description(hidden,c::BatchCacheRole::kKey).error().IsInvalidArgument());
    EXPECT_TRUE(c::NativePagedBatchProgram::prepare(hidden,{}).error().IsInvalidArgument());
    EXPECT_TRUE(c::MaterializedBatchProgram::prepare(hidden,{}).error().IsInvalidArgument());
    auto context=c::Context::create(c::cpu::device()).value();EXPECT_TRUE(c::BatchProgram::prepare(context,hidden,{}).error().IsInvalidArgument());
    EXPECT_TRUE(c::PagedArena::create(hidden,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).error().IsInvalidArgument());
    EXPECT_TRUE(arena.read(hidden_version,{}).error().IsInvalidArgument());
    EXPECT_TRUE(arena.pending_read(hidden_version,{},{}).error().IsInvalidArgument());
  }EXPECT_EQ(allocator.live,0u);
  auto cap=capacity(c::RowCacheEncoding::float4_block32());cap.page_size=0;EXPECT_FALSE(c::RowCacheBatchSchema::create(cap));
  cap=capacity(c::RowCacheEncoding::float4_block32());cap.max_pages=UINT32_MAX;EXPECT_TRUE(c::RowCacheBatchSchema::create(cap).error().IsOutOfRange());
  auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float4_block32())).value();
  EXPECT_TRUE(c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=408}).error().IsResourceExhausted());
}
TEST(RowCachePaged, ReframeCombinesAndProjectsExactTypedGuardsThroughActualUseRetirement){
  auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float4_block32())).value();
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto zero=fill(arena,0,std::byte{17}),one=fill(arena,1,std::byte{29}),two=fill(arena,2,std::byte{43});
  const auto a_layout=layout(schema,std::array<std::int32_t,2>{0,1},1);
  const auto b_layout=layout(schema,std::array<std::int32_t,2>{0,2});
  auto a=arena.read(a_layout,std::array{one,zero}).value();
  auto b=arena.read(b_layout,std::array{two,zero}).value();
  std::array sources{a,b,a};
  const std::array<std::int32_t,3> groups{0,1,3},kv{0,2,4};
  const std::array<std::int32_t,4> pages{0,1,0,2};const std::array<std::int32_t,2> tails{1,2};
  const auto batched_layout=schema.version({groups,kv,pages,tails}).value();
  auto batched_result=arena.reframe(batched_layout,sources);ASSERT_TRUE(batched_result);
  auto batched=std::move(*batched_result);
  EXPECT_EQ(batched.layout(),batched_layout);EXPECT_EQ(batched.page_count(),3u);
  EXPECT_EQ(c::detail::PagedArenaAccess::tensor(batched)->description(),c::row_cache_paged_description(schema.capacity()).value());
  EXPECT_EQ(sources[0].layout(),a_layout);EXPECT_EQ(sources[1].layout(),b_layout);
  auto projected_result=arena.reframe(a_layout,std::span(&batched,1));ASSERT_TRUE(projected_result);
  auto projected=std::move(*projected_result);EXPECT_EQ(projected.page_count(),2u);
  EXPECT_EQ(projected.layout(),a_layout);EXPECT_EQ(c::detail::PagedArenaAccess::pages(projected).size(),2u);
  // A private history from another request cannot be recovered from this
  // projection just because it is physically present in the same row arena.
  EXPECT_FALSE(arena.reframe(b_layout,std::span(&projected,1)));
  std::array<std::byte,1> unchanged{};
  ASSERT_TRUE(one.view()->subview(0,1)->read(unchanged).ok());EXPECT_EQ(unchanged[0],std::byte{29});
  a={};b={};sources={};batched={};zero.reset();one.reset();two.reset();
  EXPECT_TRUE(arena.acquire(2));EXPECT_FALSE(arena.acquire(0));EXPECT_FALSE(arena.acquire(1));
  auto uses=c::LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=2}).value();auto use=uses.begin().value();
  for(const auto&guard:c::detail::PagedArenaAccess::pages(projected))ASSERT_TRUE(use.read(guard).ok());
  projected={};
  EXPECT_FALSE(arena.acquire(0));EXPECT_FALSE(arena.acquire(1));
  use={};for(unsigned page=0;page<3;++page)EXPECT_TRUE(arena.acquire(page));
}
TEST(RowCachePaged, ReframeDoesNotWidenLogicalPrefixesAndEmptyHistoryPreservesSelectionGroups){
  auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float8_block64_tail64())).value();
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto page=fill(arena,0,std::byte{53});
  const auto one=layout(schema,std::array<std::int32_t,1>{0},1),two=layout(schema,std::array<std::int32_t,1>{0},2);
  auto short_frame=arena.read(one,std::array{page}).value();
  ASSERT_EQ(c::detail::PagedArenaAccess::pages(short_frame)[0].size(),arena.page_bytes());
  auto expanded=arena.reframe(two,std::array{short_frame});ASSERT_FALSE(expanded);
  EXPECT_TRUE(expanded.error().IsFailedPrecondition());EXPECT_EQ(short_frame.layout(),one);
  auto long_frame=arena.read(two,std::array{page}).value();
  auto merged=arena.reframe(two,std::array{short_frame,long_frame,short_frame}).value();
  EXPECT_EQ(merged.page_count(),1u);EXPECT_EQ(short_frame.layout(),one);
  const std::array<std::int32_t,3> groups{0,2,3},kv{0,0,0};const std::array<std::int32_t,2> tails{};
  const auto empty_layout=schema.version({groups,kv,{},tails}).value();
  auto empty=arena.reframe(empty_layout,std::array{merged}).value();
  auto no_sources=arena.reframe(empty_layout,{});ASSERT_TRUE(no_sources);
  EXPECT_EQ(empty.layout(),empty_layout);EXPECT_EQ(empty.layout().facts().query_rows,3u);
  EXPECT_EQ(empty.layout().facts().requests,2u);EXPECT_EQ(empty.page_count(),0u);
  EXPECT_TRUE(c::detail::PagedArenaAccess::pages(empty).empty());
  std::array<std::byte,1> unchanged{};
  ASSERT_TRUE(page.view()->subview(0,1)->read(unchanged).ok());EXPECT_EQ(unchanged[0],std::byte{53});
  short_frame={};long_frame={};merged={};page.reset();
  for(unsigned id=0;id<3;++id)EXPECT_TRUE(arena.acquire(id));
}
TEST(RowCachePaged, ReframeRejectsForeignTypedOwnersAndAttentionDomainsWithoutChangingSource){
  auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float4_block32())).value();
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto page=fill(arena,0,std::byte{61});const auto current=layout(schema,std::array<std::int32_t,1>{0});
  auto source=arena.read(current,std::array{page}).value();
  auto other=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto other_page=fill(other,0,std::byte{67});auto foreign=other.read(current,std::array{other_page}).value();
  auto other_schema=c::RowCacheBatchSchema::create(schema.capacity()).value();
  const auto invalid=[](const auto&result){EXPECT_FALSE(result);if(!result)EXPECT_TRUE(result.error().IsInvalidArgument());};
  invalid(arena.reframe(c::RowCacheLayoutVersion{},std::array{source}));
  invalid(arena.reframe(layout(other_schema,std::array<std::int32_t,1>{0}),std::array{source}));
  invalid(arena.reframe(current,std::array{source,c::PagedRowCacheRead{}}));
  invalid(arena.reframe(current,std::array{source,foreign}));
  auto missing=arena.reframe(layout(schema,std::array<std::int32_t,1>{2}),std::array{source});
  ASSERT_FALSE(missing);EXPECT_TRUE(missing.error().IsFailedPrecondition());
  const std::array<std::int32_t,2> groups{0,3},kv{0,0};const std::array<std::int32_t,1> tail{};
  const auto empty_layout=schema.version({groups,kv,{},tail}).value();
  invalid(arena.reframe(empty_layout,std::array{c::PagedRowCacheRead{}}));
  invalid(arena.reframe(empty_layout,std::array{foreign}));
  c::BatchCapacity attention_cap{.max_kv_rows=8,.max_pages=3,.max_page_refs=4,.page_size=2};
  auto attention_schema=c::BatchSchema::create(attention_cap).value();
  auto attention_arena=c::PagedArena::create(attention_schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  invalid(attention_arena.reframe(current,std::array{source}));
  const std::array<std::int32_t,2> attention_q{0,1},attention_kv{0,1};
  const std::array<std::int32_t,1> attention_pages{0},attention_tail{1};
  const auto attention_layout=attention_schema.version({attention_q,attention_kv,attention_pages,attention_tail}).value();
  invalid(arena.reframe(attention_layout,std::span<const c::PagedArenaRead>{}));
  EXPECT_EQ(source.layout(),current);EXPECT_EQ(source.page_count(),1u);
  std::array<std::byte,1> unchanged{};
  ASSERT_TRUE(page.view()->subview(0,1)->read(unchanged).ok());EXPECT_EQ(unchanged[0],std::byte{61});
  EXPECT_TRUE(arena.reframe(current,std::array{source}));
}
} // namespace
