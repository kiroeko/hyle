#include <hyle/compute/detail/paged_arena_access.h>
#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/state_versions.h>
#include <hyle/mem/quota_allocator.h>
#include <gtest/gtest.h>
#include <array>

namespace {
namespace c=hyle::compute;namespace m=hyle::mem;
c::BatchCapacity capacity(){return {.max_requests=1,.max_query_rows=1,.max_kv_rows=8,.max_pages=4,.max_page_refs=4,.query_heads=2,.kv_heads=2,.head_dim=64,.page_size=2};}
c::BatchLayoutVersion layout(const c::BatchSchema&schema,std::span<const std::int32_t> pages,int tail=2){const std::array<std::int32_t,2> q{0,1},kv{0,std::int32_t(pages.size())};const std::array<std::int32_t,1> tails{tail};return schema.version({q,kv,pages,tails}).value();}
c::ReadLease fill(c::PagedArena&arena,unsigned id,std::byte value){auto writer=arena.acquire(id).value();EXPECT_TRUE(writer.lease().view()->fill(value).ok());return writer.publish(2).value();}
std::byte first(const c::ReadLease&lease){std::array<std::byte,1> bytes{};EXPECT_TRUE(lease.view()->subview(0,1)->read(bytes).ok());return bytes[0];}
TEST(PagedArena, UsedPageReadersDoNotStopWritingAnUnreferencedPhysicalPage){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7});auto one=fill(arena,1,std::byte{11});auto version=layout(schema,std::array<std::int32_t,2>{1,0});auto frame=arena.read(version,std::array{zero,one}).value();
  EXPECT_EQ(frame.page_count(),2u);EXPECT_TRUE(arena.acquire(0).error().IsUnavailable());auto free=arena.acquire(2).value();ASSERT_TRUE(free.lease().view()->fill(std::byte{23}).ok());auto next=free.publish(2).value();EXPECT_EQ(first(zero),std::byte{7});EXPECT_EQ(first(next),std::byte{23});
  auto tensor=c::detail::PagedArenaAccess::tensor(frame).value();EXPECT_EQ(tensor.storage().size(),arena.page_bytes()*4);auto use_pool=c::LeaseUsePool::create().value();auto use=use_pool.begin().value();for(const auto&guard:c::detail::PagedArenaAccess::pages(frame))ASSERT_TRUE(use.read(guard).ok());frame={};zero.reset();one.reset();EXPECT_FALSE(arena.acquire(0));use={};EXPECT_TRUE(arena.acquire(0));
}
TEST(PagedArena, ForeignSchemaPageRangeAndIncompleteOrExcessiveSetsReject){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7}),one=fill(arena,1,std::byte{11});auto version=layout(schema,std::array<std::int32_t,1>{0});
  EXPECT_FALSE(arena.read(version,std::array{one}));EXPECT_FALSE(arena.read(version,std::array{zero,one}));EXPECT_FALSE(arena.read(version,std::array{zero.subview(0,256).value()}));
  auto foreign_schema=c::BatchSchema::create(capacity()).value();EXPECT_FALSE(arena.read(layout(foreign_schema,std::array<std::int32_t,1>{0}),std::array{zero}));auto other=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto foreign=fill(other,0,std::byte{7});EXPECT_FALSE(arena.read(version,std::array{foreign}));
  EXPECT_TRUE(arena.acquire(4).error().IsOutOfRange());
}
TEST(PagedArena, RepeatedPhysicalReferencesKeepOneActualGuard){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7});auto frame=arena.read(layout(schema,std::array<std::int32_t,2>{0,0}),std::array{zero,zero}).value();EXPECT_EQ(frame.page_count(),1u);EXPECT_EQ(c::detail::PagedArenaAccess::pages(frame).size(),1u);
}
TEST(PagedArena, F4LedgerAndArenaDoNotPinEachOtherAfterLastReaderRetires){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto page=fill(arena,0,std::byte{7});auto description=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{2,2,64}).value();auto state_schema=c::StateSchema::create(std::array{c::StateEntrySpec{.description=description}}).value();auto id=state_schema.entry(0).value();auto store=c::StateStore::create(state_schema).value();auto version=store.import(std::array{c::StateMember{id,page}}).value();auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();ASSERT_TRUE(use.read(page).ok());page.reset();version={};store.collect().IgnoreError();EXPECT_FALSE(arena.acquire(0));use={};ASSERT_TRUE(store.collect().ok());EXPECT_TRUE(arena.acquire(0));
}
TEST(PagedArena, UnsubmittedWriterRestoresButConsumedAbandonNeverPretendsReady){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();{auto writer=arena.acquire(0).value();}EXPECT_TRUE(arena.acquire(0));
  c::WriteLease escaped;{auto writer=arena.acquire(1).value();escaped=std::move(writer.lease());}EXPECT_TRUE(arena.acquire(1).error().IsUnavailable());ASSERT_TRUE(escaped.view()->fill(std::byte{7}).ok());auto ready=escaped.publish().value();EXPECT_FALSE(arena.read(layout(schema,std::array<std::int32_t,1>{1}),std::array{ready}));
}
TEST(PagedArena, ByteCapacityAndMetadataOomAreExplicitAndOldGuardOutlivesControls){
  auto schema=c::BatchSchema::create(capacity()).value();EXPECT_TRUE(c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=2048}).error().IsResourceExhausted());m::QuotaAllocator quota(m::system_allocator());c::LeaseUseBatch use;
  auto uses=c::LeaseUsePool::create().value();{
    auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096,.allocator=quota.get()}).value();auto zero=fill(arena,0,std::byte{7});auto version=layout(schema,std::array<std::int32_t,1>{0});auto frame=arena.read(version,std::array{zero}).value();use=uses.begin().value();ASSERT_TRUE(use.read(c::detail::PagedArenaAccess::pages(frame)[0]).ok());
    {auto reserved=quota.try_reserve_bytes(quota.remaining());ASSERT_TRUE(reserved);EXPECT_TRUE(arena.read(version,std::array{zero}).error().IsResourceExhausted());}ASSERT_TRUE(arena.close().ok());EXPECT_FALSE(arena.acquire(1));EXPECT_FALSE(arena.read(version,std::array{zero}));EXPECT_TRUE(c::detail::PagedArenaAccess::tensor(frame));
  }EXPECT_GT(quota.used(),0u);use={};EXPECT_EQ(quota.used(),0u);
}
TEST(PagedArena, ActualF2PageCloneAppendAndF4PublishKeepOldStateAndShareCleanPage){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7}),one=fill(arena,1,std::byte{11});auto old_layout=layout(schema,std::array<std::int32_t,2>{0,1},1);auto old_frame=arena.read(old_layout,std::array{zero,one}).value();
  auto page_desc=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{2,2,64}).value();auto state_schema=c::StateSchema::create(std::array{c::StateEntrySpec{.description=page_desc},c::StateEntrySpec{.description=page_desc},c::StateEntrySpec{.description=schema.metadata_description().value()}}).value();const auto id0=state_schema.entry(0).value(),id1=state_schema.entry(1).value(),id2=state_schema.entry(2).value();auto store=c::StateStore::create(state_schema).value();auto base=store.import(std::array{c::StateMember{id0,zero},c::StateMember{id1,one},c::StateMember{id2,old_layout.metadata().value()}}).value();auto head=c::StateHead::create(store,base).value();auto draft=store.begin(base).value();
  auto context=c::Context::create(c::cpu::device()).value();auto queue=context.create_queue().value();auto copy=c::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{m::system_resource(),arena.page_bytes()},c::CopyResourceSpec{m::system_resource(),arena.page_bytes()}}).value();auto invocation=c::CopyBatchInvocation::create(copy,queue).value();auto private_page=arena.acquire(2).value();ASSERT_TRUE(invocation.bind_source({0},base.member(id1).value()).ok());ASSERT_TRUE(invocation.bind_mutable({1},std::move(private_page.lease())).ok());const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,arena.page_bytes()}}};ASSERT_TRUE(invocation.prepare_replay(phases,ranges).ok());auto copied=invocation.submit().value();ASSERT_TRUE(copied.wait().ok());auto cloned=copied.output({0},{1},0,arena.page_bytes()).value();EXPECT_EQ(copied.report().logical_bytes,arena.page_bytes());copied={};ASSERT_TRUE(invocation.reset().ok());
  private_page.lease()=cloned.try_write().value();std::array<std::byte,256> appended;appended.fill(std::byte{23});ASSERT_TRUE(private_page.lease().view()->subview(256,256)->write(appended).ok());auto ready=private_page.publish(2).value();auto new_layout=layout(schema,std::array<std::int32_t,2>{0,2},2);ASSERT_TRUE(draft.replace(id1,ready).ok());ASSERT_TRUE(draft.replace(id2,new_layout.metadata().value()).ok());auto candidate=draft.finalize().value();ASSERT_TRUE(head.publish(base.identity(),candidate).ok());
  auto current=head.current().value();auto new_frame=arena.read(new_layout,std::array{current.member(id0).value(),current.member(id1).value()}).value();EXPECT_EQ(first(base.member(id1).value()),std::byte{11});std::array<std::byte,1> appended_byte;ASSERT_TRUE(current.member(id1)->view()->subview(256,1)->read(appended_byte).ok());EXPECT_EQ(appended_byte[0],std::byte{23});EXPECT_EQ(base.member(id0)->view()->native_handle(),current.member(id0)->view()->native_handle());EXPECT_EQ(old_frame.layout(),old_layout);EXPECT_EQ(new_frame.layout(),new_layout);EXPECT_FALSE(arena.acquire(1));EXPECT_FALSE(arena.acquire(2));
}

TEST(PagedArena, OriginalReadyPageReacquiresWithoutCopyAllocationOrGenerationChange){
  auto schema=c::BatchSchema::create(capacity()).value();m::QuotaAllocator quota(m::system_allocator());
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096,.allocator=quota.get()}).value();
  auto page=fill(arena,0,std::byte{17});const auto generation=page.generation();const auto origin=page.view()->offset();
  auto alias=page;EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());alias.reset();
  auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();ASSERT_TRUE(use.read(page).ok());
  EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());use={};
  auto partial=page.subview(0,256).value();EXPECT_FALSE(arena.acquire(partial));EXPECT_TRUE(partial.valid());partial.reset();
  auto other=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();auto foreign=fill(other,0,std::byte{23});
  EXPECT_FALSE(arena.acquire(foreign));EXPECT_TRUE(foreign.valid());
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();auto writer=arena.acquire(page).value();
    EXPECT_FALSE(page.valid());EXPECT_EQ(writer.page(),0u);EXPECT_EQ(writer.lease().generation(),generation);
    std::array<std::byte,1> actual;ASSERT_TRUE(writer.lease().view()->subview(0,1)->read(actual).ok());EXPECT_EQ(actual[0],std::byte{17});
    page=writer.publish(2).value();EXPECT_EQ(page.generation(),generation);EXPECT_EQ(page.view()->offset(),origin);}
  ASSERT_TRUE(arena.close().ok());EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());
}
TEST(PagedArena, F2MutableWithdrawalPreservesInitializedUntouchedBytesAndRejectsRetainedObservation){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto page=fill(arena,0,std::byte{17});auto source=fill(arena,1,std::byte{23});auto writer=arena.acquire(page).value();
  auto context=c::Context::create(c::cpu::device()).value();auto queue=context.create_queue().value();
  auto program=c::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{m::system_resource(),arena.page_bytes()},c::CopyResourceSpec{m::system_resource(),arena.page_bytes()}}).value();
  auto copy=c::CopyBatchInvocation::create(program,queue).value();ASSERT_TRUE(copy.bind_source({0},source).ok());
  EXPECT_FALSE(copy.unbind_mutable({0}));EXPECT_FALSE(copy.unbind_mutable({7}));
  ASSERT_TRUE(copy.bind_mutable({1},std::move(writer.lease())).ok());
  writer.lease()=copy.unbind_mutable({1}).value();EXPECT_FALSE(copy.unbind_mutable({1}));
  ASSERT_TRUE(copy.bind_mutable({1},std::move(writer.lease())).ok());
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,256,256}}};
  ASSERT_TRUE(copy.prepare_replay(phases,ranges).ok());
  {auto submitted=copy.submit().value();EXPECT_FALSE(copy.unbind_mutable({1}));ASSERT_TRUE(submitted.wait().ok());EXPECT_FALSE(copy.unbind_mutable({1}));EXPECT_EQ(submitted.report().logical_bytes,256u);}
  writer.lease()=copy.unbind_mutable({1}).value();EXPECT_FALSE(copy.submit());page=writer.publish(2).value();
  EXPECT_EQ(first(page),std::byte{17});std::array<std::byte,1> actual;ASSERT_TRUE(page.view()->subview(256,1)->read(actual).ok());EXPECT_EQ(actual[0],std::byte{23});
  ASSERT_TRUE(copy.reset().ok());auto next=arena.acquire(page).value();ASSERT_TRUE(copy.bind_source({0},source).ok());ASSERT_TRUE(copy.bind_mutable({1},std::move(next.lease())).ok());
  const std::array<c::CopyRange,1> full{{{{0},{1},{0},0,0,arena.page_bytes()}}};ASSERT_TRUE(copy.prepare_replay(phases,full).ok());
  {auto submitted=copy.submit().value();ASSERT_TRUE(submitted.wait().ok());page=submitted.output({0},{1},0,arena.page_bytes()).value();ASSERT_TRUE(next.complete(page,2).ok());}
  EXPECT_FALSE(copy.unbind_mutable({1}));ASSERT_TRUE(copy.reset().ok());
  auto doomed=arena.acquire(page).value();ASSERT_TRUE(copy.bind_mutable({1},std::move(doomed.lease())).ok());ASSERT_TRUE(copy.close().ok());EXPECT_FALSE(copy.unbind_mutable({1}));
}
TEST(PagedArena, PartialPrefixRejectsUnwrittenTailBeforeAllocatingFrame){
  auto schema=c::BatchSchema::create(capacity()).value();m::QuotaAllocator quota(m::system_allocator());
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096,.allocator=quota.get()}).value();
  auto writer=arena.acquire(0).value();EXPECT_EQ(writer.initialized_rows(),0u);
  ASSERT_TRUE(writer.lease().view()->subview(0,256)->fill(std::byte{29}).ok());
  EXPECT_TRUE(writer.publish(3).error().IsInvalidArgument());EXPECT_TRUE(writer.lease().valid());
  auto page=writer.publish(1).value();auto one=layout(schema,std::array<std::int32_t,1>{0},1);
  auto two=layout(schema,std::array<std::int32_t,1>{0},2);
  {auto frame=arena.read(one,std::array{page}).value();EXPECT_EQ(frame.page_count(),1u);EXPECT_EQ(first(page),std::byte{29});}
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();
    EXPECT_TRUE(arena.read(two,std::array{page}).error().IsFailedPrecondition());
    EXPECT_TRUE(arena.read(one,std::array{page}).error().IsResourceExhausted());}
  {auto restored=arena.acquire(page).value();EXPECT_EQ(restored.initialized_rows(),1u);
    c::PagedPageWriter moved(std::move(restored));EXPECT_FALSE(restored.valid());EXPECT_EQ(moved.initialized_rows(),1u);}
  writer=arena.acquire(0).value();EXPECT_EQ(writer.initialized_rows(),1u);
  page=writer.publish(0).value();EXPECT_TRUE(arena.read(one,std::array{page}).error().IsFailedPrecondition());
  page.reset();writer=arena.acquire(0).value();EXPECT_EQ(writer.initialized_rows(),0u);
}
TEST(PagedArena, SharedPageMustCoverEveryRequestAndEveryNonTailReference){
  auto cap=capacity();cap.max_requests=cap.max_query_rows=2;
  auto schema=c::BatchSchema::create(cap).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto writer=arena.acquire(0).value();ASSERT_TRUE(writer.lease().view()->subview(0,256)->fill(std::byte{13}).ok());
  auto page=writer.publish(1).value();const std::array<std::int32_t,3> ptr{0,1,2};
  const std::array<std::int32_t,2> ids{0,0},short_tails{1,1},mixed_tails{1,2};
  auto short_layout=schema.version({ptr,ptr,ids,short_tails}).value();auto mixed=schema.version({ptr,ptr,ids,mixed_tails}).value();
  {auto frame=arena.read(short_layout,std::array{page}).value();EXPECT_EQ(frame.page_count(),1u);
    EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());}
  EXPECT_TRUE(arena.read(mixed,std::array{page}).error().IsFailedPrecondition());
  const std::array<std::int32_t,2> q{0,1},kv{0,2},repeated{0,0};const std::array<std::int32_t,1> tail{1};
  auto non_tail=schema.version({q,kv,repeated,tail}).value();
  EXPECT_TRUE(arena.read(non_tail,std::array{page}).error().IsFailedPrecondition());
  writer=arena.acquire(page).value();ASSERT_TRUE(writer.lease().view()->subview(256,256)->fill(std::byte{31}).ok());
  page=writer.publish(2).value();EXPECT_TRUE(arena.read(mixed,std::array{page}));EXPECT_TRUE(arena.read(non_tail,std::array{page}));
}
TEST(PagedArena, SixtyFourNumericRowAppendsPreservePrefixAndOriginalGeneration){
  auto cap=capacity();cap.max_kv_rows=cap.page_size=64;cap.max_pages=cap.max_page_refs=1;
  auto schema=c::BatchSchema::create(cap).value();m::QuotaAllocator quota(m::system_allocator());
  auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=32768,.allocator=quota.get()}).value();
  c::ReadLease page;std::uint64_t generation=0;
  for(unsigned row=0;row<64;++row){
    {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();
      auto writer=(row?arena.acquire(page):arena.acquire(0)).value();EXPECT_EQ(writer.initialized_rows(),row);
      if(!row)generation=writer.lease().generation();EXPECT_EQ(writer.lease().generation(),generation);
      ASSERT_TRUE(writer.lease().view()->subview(std::size_t(row)*256,256)->fill(std::byte(row+1)).ok());
      page=writer.publish(row+1).value();}
    EXPECT_EQ(page.generation(),generation);std::array<std::byte,16384> actual{};const auto valid_bytes=std::size_t(row+1)*256;
    ASSERT_TRUE(page.view()->subview(0,valid_bytes)->read(std::span(actual).first(valid_bytes)).ok());
    for(unsigned prior=0;prior<=row;++prior)for(unsigned byte=0;byte<256;++byte)ASSERT_EQ(actual[prior*256+byte],std::byte(prior+1));
    auto current=layout(schema,std::array<std::int32_t,1>{0},int(row+1));
    {auto frame=arena.read(current,std::array{page}).value();EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());}
    if(row<63)EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},int(row+2)),std::array{page}).error().IsFailedPrecondition());
  }
  page.reset();auto recycled=arena.acquire(0).value();EXPECT_NE(recycled.lease().generation(),generation);EXPECT_EQ(recycled.initialized_rows(),0u);
}
TEST(PagedArena, ActualPartialF2SuccessCommitsOnlyOneInitializedRow){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto source=fill(arena,1,std::byte{47});auto writer=arena.acquire(0).value();
  auto context=c::Context::create(c::cpu::device()).value();auto queue=context.create_queue().value();
  auto program=c::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{m::system_resource(),arena.page_bytes()},c::CopyResourceSpec{m::system_resource(),arena.page_bytes()}}).value();
  auto copy=c::CopyBatchInvocation::create(program,queue).value();ASSERT_TRUE(copy.bind_source({0},source).ok());ASSERT_TRUE(copy.bind_mutable({1},std::move(writer.lease())).ok());
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,256}}};ASSERT_TRUE(copy.prepare_replay(phases,ranges).ok());
  {auto submitted=copy.submit().value();ASSERT_TRUE(submitted.wait().ok());EXPECT_EQ(submitted.report().logical_bytes,256u);EXPECT_FALSE(copy.unbind_mutable({1}));}
  writer.lease()=copy.unbind_mutable({1}).value();EXPECT_EQ(writer.initialized_rows(),0u);
  auto page=writer.publish(1).value();EXPECT_EQ(first(page),std::byte{47});
  EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},1),std::array{page}));
  EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},2),std::array{page}).error().IsFailedPrecondition());ASSERT_TRUE(copy.reset().ok());
}
TEST(PagedArena, CompleteRejectsInvalidPrefixWithoutConsumingReservation){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto writer=arena.acquire(0).value();ASSERT_TRUE(writer.lease().view()->subview(0,256)->fill(std::byte{53}).ok());
  auto raw=writer.lease().publish().value();EXPECT_TRUE(writer.complete(raw,3).IsInvalidArgument());EXPECT_TRUE(writer.valid());EXPECT_FALSE(arena.acquire(0));
  auto foreign=fill(arena,1,std::byte{59});EXPECT_TRUE(writer.complete(foreign,1).IsInvalidArgument());EXPECT_TRUE(writer.valid());
  ASSERT_TRUE(writer.complete(raw,1).ok());EXPECT_EQ(first(raw),std::byte{53});
  EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},1),std::array{raw}));EXPECT_FALSE(arena.read(layout(schema,std::array<std::int32_t,1>{0},2),std::array{raw}));
}
} // namespace
