#include <hyle/compute/detail/paged_arena_access.h>
#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/state_versions.h>
#include <hyle/compute/detail/lease_access.h>
#include <hyle/mem/quota_allocator.h>
#include "allocation_fault_fixture.h"
#include <gtest/gtest.h>
#include <array>

namespace {
namespace c=hyle::compute;namespace m=hyle::mem;
c::BatchCapacity capacity(){return {.max_requests=1,.max_query_rows=1,.max_kv_rows=8,.max_pages=4,.max_page_refs=4,.query_heads=2,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=2};}
c::BatchLayoutVersion layout(const c::BatchSchema&schema,std::span<const std::int32_t> pages,int tail=2){const std::array<std::int32_t,2> q{0,1},kv{0,std::int32_t(pages.size())};const std::array<std::int32_t,1> tails{tail};return schema.version({q,kv,pages,tails}).value();}
c::ReadLease fill(c::PagedArena&arena,unsigned id,std::byte value){auto writer=arena.acquire(id).value();EXPECT_TRUE(writer.lease().view()->fill(value).ok());return writer.publish(2).value();}
std::byte first(const c::ReadLease&lease){std::array<std::byte,1> bytes{};EXPECT_TRUE(lease.view()->subview(0,1)->read(bytes).ok());return bytes[0];}

TEST(PagedArena, EncodedKeyAndValuePagesKeepSeparateBytePrefixesAndPermissions){
  auto cap=capacity();cap.value_head_dim=128;
  cap.key_storage=cap.value_storage=c::CacheStorage::q8_0().value();
  auto schema=c::BatchSchema::create(cap).value();
  auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,m::system_resource(),{.max_backing_bytes=4096}).value();
  EXPECT_EQ(keys.page_bytes(),272u);EXPECT_EQ(values.page_bytes(),544u);
  auto kw=keys.acquire(0).value(),vw=values.acquire(0).value();
  ASSERT_TRUE(kw.lease().view()->fill(std::byte{7}).ok());ASSERT_TRUE(vw.lease().view()->fill(std::byte{9}).ok());
  auto k=kw.publish(1).value(),v=vw.publish(1).value();
  auto one=layout(schema,std::array<std::int32_t,1>{0},1);
  auto kr=keys.read(one,std::span(&k,1));auto vr=values.read(one,std::span(&v,1));ASSERT_TRUE(kr);ASSERT_TRUE(vr);
  EXPECT_EQ(c::detail::PagedArenaAccess::tensor(*kr)->description(),c::batch_cache_description(cap,c::BatchCacheRole::kKey).value());
  EXPECT_EQ(c::detail::PagedArenaAccess::tensor(*vr)->description(),c::batch_cache_description(cap,c::BatchCacheRole::kValue).value());
  auto two=layout(schema,std::array<std::int32_t,1>{0},2);
  EXPECT_FALSE(keys.read(two,std::span(&k,1)));EXPECT_FALSE(values.read(two,std::span(&v,1)));
  EXPECT_FALSE(keys.acquire(0));EXPECT_TRUE(keys.acquire(1));
}

TEST(PagedArena, SameFloat8BytesDoNotProveTheOtherPortsScale){
  auto cap=capacity();cap.key_storage=c::CacheStorage::float8(.25F).value();cap.value_storage=c::CacheStorage::float8(2.F).value();
  auto schema=c::BatchSchema::create(cap).value();
  auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto k=fill(keys,0,std::byte{7}),v=fill(values,0,std::byte{9});auto current=layout(schema,std::array<std::int32_t,1>{0});
  auto kr=keys.read(current,std::span(&k,1)).value(),vr=values.read(current,std::span(&v,1)).value();
  EXPECT_EQ(c::detail::PagedArenaAccess::tensor(kr)->description(),c::detail::PagedArenaAccess::tensor(vr)->description());
  EXPECT_TRUE(c::detail::PagedArenaAccess::validate_storage(kr,cap.key_storage).ok());
  EXPECT_TRUE(c::detail::PagedArenaAccess::validate_storage(vr,cap.value_storage).ok());
  EXPECT_TRUE(c::detail::PagedArenaAccess::validate_storage(kr,cap.value_storage).IsInvalidArgument());
  EXPECT_TRUE(c::detail::PagedArenaAccess::validate_storage(vr,cap.key_storage).IsInvalidArgument());
}
TEST(PagedArena, UsedPageReadersDoNotStopWritingAnUnreferencedPhysicalPage){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7});auto one=fill(arena,1,std::byte{11});auto version=layout(schema,std::array<std::int32_t,2>{1,0});auto frame=arena.read(version,std::array{zero,one}).value();
  EXPECT_EQ(frame.page_count(),2u);EXPECT_TRUE(arena.acquire(0).error().IsUnavailable());auto free=arena.acquire(2).value();ASSERT_TRUE(free.lease().view()->fill(std::byte{23}).ok());auto next=free.publish(2).value();EXPECT_EQ(first(zero),std::byte{7});EXPECT_EQ(first(next),std::byte{23});
  auto tensor=c::detail::PagedArenaAccess::tensor(frame).value();EXPECT_EQ(tensor.storage().size(),arena.page_bytes()*4);auto use_pool=c::LeaseUsePool::create().value();auto use=use_pool.begin().value();for(const auto&guard:c::detail::PagedArenaAccess::pages(frame))ASSERT_TRUE(use.read(guard).ok());frame={};zero.reset();one.reset();EXPECT_FALSE(arena.acquire(0));use={};EXPECT_TRUE(arena.acquire(0));
}
TEST(PagedArena, ForeignSchemaPageRangeAndIncompleteOrExcessiveSetsReject){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7}),one=fill(arena,1,std::byte{11});auto version=layout(schema,std::array<std::int32_t,1>{0});
  EXPECT_FALSE(arena.read(version,std::array{one}));EXPECT_FALSE(arena.read(version,std::array{zero,one}));EXPECT_FALSE(arena.read(version,std::array{zero.subview(0,256).value()}));
  auto foreign_schema=c::BatchSchema::create(capacity()).value();EXPECT_FALSE(arena.read(layout(foreign_schema,std::array<std::int32_t,1>{0}),std::array{zero}));auto other=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto foreign=fill(other,0,std::byte{7});EXPECT_FALSE(arena.read(version,std::array{foreign}));
  EXPECT_TRUE(arena.acquire(4).error().IsOutOfRange());
}
TEST(PagedArena, RepeatedPhysicalReferencesKeepOneActualGuard){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7});auto frame=arena.read(layout(schema,std::array<std::int32_t,2>{0,0}),std::array{zero,zero}).value();EXPECT_EQ(frame.page_count(),1u);EXPECT_EQ(c::detail::PagedArenaAccess::pages(frame).size(),1u);
}
TEST(PagedArena, TrueEmptyAttentionFrameRetainsCapacityWithoutPagePermissions){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),
      {.max_backing_bytes=4096}).value();
  const std::array<std::int32_t,2> zero{0,0};
  const std::array<std::int32_t,1> tail{0};
  auto empty=schema.version({zero,zero,{},tail}).value();
  auto frame=arena.read(empty,{});ASSERT_TRUE(frame)<<frame.error().message();
  EXPECT_TRUE(frame->valid());EXPECT_EQ(frame->page_count(),0U);
  EXPECT_TRUE(c::detail::PagedArenaAccess::pages(*frame).empty());
  auto physical=c::detail::PagedArenaAccess::tensor(*frame);ASSERT_TRUE(physical);
  EXPECT_EQ(physical->description(),c::batch_cache_description(capacity(),c::BatchCacheRole::kKey).value());
  EXPECT_EQ(physical->storage().size(),arena.page_bytes()*capacity().max_pages);
  // Holding an address-only empty frame must not pin any page writer.
  for(unsigned page=0;page<capacity().max_pages;++page)EXPECT_TRUE(arena.acquire(page));
  EXPECT_FALSE(arena.read(layout(schema,std::array<std::int32_t,1>{0}),{}));
  auto other=c::BatchSchema::create(capacity()).value();
  EXPECT_FALSE(arena.read(other.version({zero,zero,{},tail}).value(),{}));
}
TEST(PagedArena, SharedLayoutFootprintDoesNotCacheOrRetainPagePermissions){
  hyle_f5_fault_fixture::Allocator layout_metadata,arena_metadata;
  auto schema=c::BatchSchema::create(capacity(),{.allocator=layout_metadata.ref()}).value();
  auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096,.allocator=arena_metadata.ref()}).value();
  auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,m::system_resource(),{.max_backing_bytes=4096,.allocator=arena_metadata.ref()}).value();
  std::array k{fill(keys,0,std::byte{7}),fill(keys,1,std::byte{11})};
  std::array v{fill(values,0,std::byte{13}),fill(values,1,std::byte{17})};
  auto old=layout(schema,std::array<std::int32_t,3>{1,0,1});
  const auto layout_before=layout_metadata.attempts,arena_before=arena_metadata.attempts;
  {auto kr=keys.read(old,k).value();auto vr=values.read(old,v).value();
    EXPECT_EQ(layout_metadata.attempts-layout_before,1u);EXPECT_EQ(arena_metadata.attempts-arena_before,4u);
    EXPECT_EQ(kr.page_count(),2u);EXPECT_EQ(vr.page_count(),2u);EXPECT_FALSE(keys.acquire(k[0]));
    for(unsigned repeat=0;repeat<8;++repeat){auto another=keys.read(old,k).value();EXPECT_EQ(another.page_count(),2u);}
    EXPECT_EQ(layout_metadata.attempts-layout_before,1u);
    // A different immutable version builds its own exact set; the old one and
    // every already-admitted page permission keep their original identity.
    auto next=layout(schema,std::array<std::int32_t,1>{0});
    const auto next_before=layout_metadata.attempts;
    EXPECT_FALSE(keys.read(next,std::array{k[1]}));
    EXPECT_EQ(layout_metadata.attempts-next_before,1u);
    EXPECT_TRUE(keys.read(next,std::array{k[0]}));
    EXPECT_EQ(layout_metadata.attempts-next_before,1u);
    EXPECT_EQ(kr.layout(),old);EXPECT_EQ(first(k[0]),std::byte{7});}
  // Only the caller's ready leases remain; retaining the version/footprint
  // must not obstruct the original permission's ordinary writer transfer.
  auto writer=keys.acquire(k[0]);ASSERT_TRUE(writer);k[0]=writer->publish(2).value();
  EXPECT_TRUE(keys.read(old,k));
}
TEST(PagedArena, F4LedgerAndArenaDoNotPinEachOtherAfterLastReaderRetires){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto page=fill(arena,0,std::byte{7});auto description=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{2,2,64}).value();auto state_schema=c::StateSchema::create(std::array{c::StateEntrySpec{.description=description}}).value();auto id=state_schema.entry(0).value();auto store=c::StateStore::create(state_schema).value();auto version=store.import(std::array{c::StateMember{id,page}}).value();auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();ASSERT_TRUE(use.read(page).ok());page.reset();version={};store.collect().IgnoreError();EXPECT_FALSE(arena.acquire(0));use={};ASSERT_TRUE(store.collect().ok());EXPECT_TRUE(arena.acquire(0));
}
TEST(PagedArena, UnsubmittedWriterRestoresButConsumedAbandonNeverPretendsReady){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();{auto writer=arena.acquire(0).value();}EXPECT_TRUE(arena.acquire(0));
  c::WriteLease escaped;{auto writer=arena.acquire(1).value();escaped=std::move(writer.lease());}EXPECT_TRUE(arena.acquire(1).error().IsUnavailable());ASSERT_TRUE(escaped.view()->fill(std::byte{7}).ok());auto ready=escaped.publish().value();EXPECT_FALSE(arena.read(layout(schema,std::array<std::int32_t,1>{1}),std::array{ready}));
}
TEST(PagedArena, DiscardWaitsForOriginalPermissionsAndStartsANewUninitializedGeneration){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto writer=arena.acquire(0).value();const auto generation=writer.lease().generation();
  const auto offset=writer.lease().view()->offset();
  EXPECT_TRUE(writer.discard().IsFailedPrecondition());EXPECT_TRUE(writer.lease().valid());
  auto original=std::move(writer.lease());
  EXPECT_TRUE(writer.discard().IsUnavailable());EXPECT_TRUE(writer.valid());
  ASSERT_TRUE(original.view()->fill(std::byte{29}).ok());
  auto retained=original.publish().value();
  EXPECT_TRUE(writer.discard().IsUnavailable());
  auto uses=c::LeaseUsePool::create().value();auto reading=uses.begin().value();
  ASSERT_TRUE(reading.read(retained).ok());retained.reset();
  EXPECT_TRUE(writer.discard().IsUnavailable());
  reading={};ASSERT_TRUE(writer.discard().ok());
  EXPECT_TRUE(writer.lease().valid());EXPECT_GT(writer.lease().generation(),generation);
  EXPECT_EQ(writer.lease().view()->offset(),offset);EXPECT_EQ(writer.initialized_rows(),0u);
  auto empty=writer.publish(0).value();
  EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},1),std::array{empty}).error().IsFailedPrecondition());
  EXPECT_TRUE(writer.discard().IsFailedPrecondition());
  auto retry=arena.acquire(empty).value();
  ASSERT_TRUE(retry.lease().view()->subview(0,256)->fill(std::byte{47}).ok());
  auto ready=retry.publish(1).value();
  EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},1),std::array{ready}));
  EXPECT_EQ(first(ready),std::byte{47});
}
TEST(PagedArena, DiscardAfterAcceptedFailureDoesNotReuseAStillRetainedUseOrModifyCleanState){
  auto context=c::Context::create(c::cpu::device()).value();auto queue=context.create_queue().value();
  for(const auto terminal:{hyle::Status::DataLoss("accepted page failure"),hyle::Status::Cancelled("accepted page cancelled")}){
    auto schema=c::BatchSchema::create(capacity()).value();
    auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
    auto clean=fill(arena,1,std::byte{17});auto writer=arena.acquire(0).value();
    const auto generation=writer.lease().generation();
    auto original=std::move(writer.lease());
    auto uses=c::LeaseUsePool::create({.max_batches=2}).value();
    auto produced=uses.begin().value(),reading=uses.begin().value();
    ASSERT_TRUE(c::detail::LeaseAccess::write_unique(produced,original,false).ok());
    ASSERT_TRUE(c::detail::LeaseAccess::ordered_read_unique(reading,original).ok());
    auto completion=c::cpu::submit_host(queue,[terminal]()noexcept{return terminal;}).value();
    ASSERT_TRUE(produced.attach(completion).ok());EXPECT_FALSE(completion.wait().ok());
    EXPECT_FALSE(original.publish());EXPECT_TRUE(writer.discard().IsUnavailable());
    original.reset();completion={};
    // A terminal error and dropping its public observer do not retire a
    // separately retained permission. Recovery uses the actual pool lifetime.
    EXPECT_TRUE(writer.discard().IsUnavailable());
    reading={};ASSERT_TRUE(writer.discard().ok());
    EXPECT_GT(writer.lease().generation(),generation);EXPECT_EQ(writer.initialized_rows(),0u);
    ASSERT_TRUE(writer.lease().view()->fill(std::byte{61}).ok());auto ready=writer.publish(2).value();
    EXPECT_EQ(first(ready),std::byte{61});EXPECT_EQ(first(clean),std::byte{17});
  }
}
TEST(PagedArena, DiscardInvalidatesTheOldPendingFrameAndRespectsClose){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto writer=arena.acquire(0).value();auto original=std::move(writer.lease());
  const std::array<c::PagedPageUpdate,1> updates{{{&writer,{1}}}};
  auto pending=arena.pending_read(layout(schema,std::array<std::int32_t,1>{0}),{},updates).value();
  original.reset();ASSERT_TRUE(writer.discard().ok());
  c::detail::CopyUpdateInput input{};input.writer=&writer.lease();
  input.root=c::detail::LeaseAccess::identity(writer.lease());input.view=writer.lease().view().value();
  EXPECT_TRUE(c::detail::PagedArenaAccess::validate_page(pending,0,input).IsFailedPrecondition());
  writer.lease().reset();ASSERT_TRUE(arena.close().ok());
  EXPECT_TRUE(writer.discard().IsFailedPrecondition());EXPECT_FALSE(writer.lease().valid());
  EXPECT_TRUE(arena.acquire(0).error().IsFailedPrecondition());
  c::PagedPageWriter empty;EXPECT_TRUE(empty.discard().IsFailedPrecondition());
}
TEST(PagedArena, ByteCapacityAndMetadataOomAreExplicitAndOldGuardOutlivesControls){
  auto schema=c::BatchSchema::create(capacity()).value();EXPECT_TRUE(c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=2048}).error().IsResourceExhausted());m::QuotaAllocator quota(m::system_allocator());c::LeaseUseBatch use;
  auto uses=c::LeaseUsePool::create().value();{
    auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096,.allocator=quota.get()}).value();auto zero=fill(arena,0,std::byte{7});auto version=layout(schema,std::array<std::int32_t,1>{0});auto frame=arena.read(version,std::array{zero}).value();use=uses.begin().value();ASSERT_TRUE(use.read(c::detail::PagedArenaAccess::pages(frame)[0]).ok());
    {auto reserved=quota.try_reserve_bytes(quota.remaining());ASSERT_TRUE(reserved);EXPECT_TRUE(arena.read(version,std::array{zero}).error().IsResourceExhausted());}ASSERT_TRUE(arena.close().ok());EXPECT_FALSE(arena.acquire(1));EXPECT_FALSE(arena.read(version,std::array{zero}));EXPECT_TRUE(c::detail::PagedArenaAccess::tensor(frame));
  }EXPECT_GT(quota.used(),0u);use={};EXPECT_EQ(quota.used(),0u);
}
TEST(PagedArena, ActualF2PageCloneAppendAndF4PublishKeepOldStateAndShareCleanPage){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto zero=fill(arena,0,std::byte{7}),one=fill(arena,1,std::byte{11});auto old_layout=layout(schema,std::array<std::int32_t,2>{0,1},1);auto old_frame=arena.read(old_layout,std::array{zero,one}).value();
  auto page_desc=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{2,2,64}).value();auto state_schema=c::StateSchema::create(std::array{c::StateEntrySpec{.description=page_desc},c::StateEntrySpec{.description=page_desc},c::StateEntrySpec{.description=schema.metadata_description().value()}}).value();const auto id0=state_schema.entry(0).value(),id1=state_schema.entry(1).value(),id2=state_schema.entry(2).value();auto store=c::StateStore::create(state_schema).value();auto base=store.import(std::array{c::StateMember{id0,zero},c::StateMember{id1,one},c::StateMember{id2,old_layout.metadata().value()}}).value();auto head=c::StateHead::create(store,base).value();auto draft=store.begin(base).value();
  auto context=c::Context::create(c::cpu::device()).value();auto queue=context.create_queue().value();auto copy=c::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{m::system_resource(),arena.page_bytes()},c::CopyResourceSpec{m::system_resource(),arena.page_bytes()}}).value();auto invocation=c::CopyBatchInvocation::create(copy,queue).value();auto private_page=arena.acquire(2).value();ASSERT_TRUE(invocation.bind_source({0},base.member(id1).value()).ok());ASSERT_TRUE(invocation.bind_mutable({1},std::move(private_page.lease())).ok());const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,arena.page_bytes()}}};ASSERT_TRUE(invocation.prepare_replay(phases,ranges).ok());auto copied=invocation.submit().value();ASSERT_TRUE(copied.wait().ok());auto cloned=copied.output({0},{1},0,arena.page_bytes()).value();EXPECT_EQ(copied.report().logical_bytes,arena.page_bytes());copied={};ASSERT_TRUE(invocation.reset().ok());
  private_page.lease()=cloned.try_write().value();std::array<std::byte,256> appended;appended.fill(std::byte{23});ASSERT_TRUE(private_page.lease().view()->subview(256,256)->write(appended).ok());auto ready=private_page.publish(2).value();auto new_layout=layout(schema,std::array<std::int32_t,2>{0,2},2);ASSERT_TRUE(draft.replace(id1,ready).ok());ASSERT_TRUE(draft.replace(id2,new_layout.metadata().value()).ok());auto candidate=draft.finalize().value();ASSERT_TRUE(head.publish(base.identity(),candidate).ok());
  auto current=head.current().value();auto new_frame=arena.read(new_layout,std::array{current.member(id0).value(),current.member(id1).value()}).value();EXPECT_EQ(first(base.member(id1).value()),std::byte{11});std::array<std::byte,1> appended_byte;ASSERT_TRUE(current.member(id1)->view()->subview(256,1)->read(appended_byte).ok());EXPECT_EQ(appended_byte[0],std::byte{23});EXPECT_EQ(base.member(id0)->view()->native_handle(),current.member(id0)->view()->native_handle());EXPECT_EQ(old_frame.layout(),old_layout);EXPECT_EQ(new_frame.layout(),new_layout);EXPECT_FALSE(arena.acquire(1));EXPECT_FALSE(arena.acquire(2));
}

TEST(PagedArena, OriginalReadyPageReacquiresWithoutCopyAllocationOrGenerationChange){
  auto schema=c::BatchSchema::create(capacity()).value();m::QuotaAllocator quota(m::system_allocator());
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096,.allocator=quota.get()}).value();
  auto page=fill(arena,0,std::byte{17});const auto generation=page.generation();const auto origin=page.view()->offset();
  auto alias=page;EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());alias.reset();
  auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();ASSERT_TRUE(use.read(page).ok());
  EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());use={};
  auto partial=page.subview(0,256).value();EXPECT_FALSE(arena.acquire(partial));EXPECT_TRUE(partial.valid());partial.reset();
  auto other=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();auto foreign=fill(other,0,std::byte{23});
  EXPECT_FALSE(arena.acquire(foreign));EXPECT_TRUE(foreign.valid());
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();auto writer=arena.acquire(page).value();
    EXPECT_FALSE(page.valid());EXPECT_EQ(writer.page(),0u);EXPECT_EQ(writer.lease().generation(),generation);
    std::array<std::byte,1> actual;ASSERT_TRUE(writer.lease().view()->subview(0,1)->read(actual).ok());EXPECT_EQ(actual[0],std::byte{17});
    page=writer.publish(2).value();EXPECT_EQ(page.generation(),generation);EXPECT_EQ(page.view()->offset(),origin);}
  ASSERT_TRUE(arena.close().ok());EXPECT_FALSE(arena.acquire(page));EXPECT_TRUE(page.valid());
}
TEST(PagedArena, F2MutableWithdrawalPreservesInitializedUntouchedBytesAndRejectsRetainedObservation){
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
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
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096,.allocator=quota.get()}).value();
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
  auto schema=c::BatchSchema::create(cap).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
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
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=32768,.allocator=quota.get()}).value();
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
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
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
  auto schema=c::BatchSchema::create(capacity()).value();auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto writer=arena.acquire(0).value();ASSERT_TRUE(writer.lease().view()->subview(0,256)->fill(std::byte{53}).ok());
  auto raw=writer.lease().publish().value();EXPECT_TRUE(writer.complete(raw,3).IsInvalidArgument());EXPECT_TRUE(writer.valid());EXPECT_FALSE(arena.acquire(0));
  auto foreign=fill(arena,1,std::byte{59});EXPECT_TRUE(writer.complete(foreign,1).IsInvalidArgument());EXPECT_TRUE(writer.valid());
  ASSERT_TRUE(writer.complete(raw,1).ok());EXPECT_EQ(first(raw),std::byte{53});
  EXPECT_TRUE(arena.read(layout(schema,std::array<std::int32_t,1>{0},1),std::array{raw}));EXPECT_FALSE(arena.read(layout(schema,std::array<std::int32_t,1>{0},2),std::array{raw}));
}
TEST(PagedArena, ReframeComposesHistoriesAndRetainsOnlyItsExactPagePermissions){
  auto cap=capacity();cap.max_requests=cap.max_query_rows=2;
  auto schema=c::BatchSchema::create(cap).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto zero=fill(arena,0,std::byte{7}),one=fill(arena,1,std::byte{11});
  auto two=fill(arena,2,std::byte{17}),three=fill(arena,3,std::byte{23});
  const auto a_layout=layout(schema,std::array<std::int32_t,3>{3,0,1});
  const auto b_layout=layout(schema,std::array<std::int32_t,2>{0,2});
  auto a=arena.read(a_layout,std::array{three,zero,one}).value();
  auto b=arena.read(b_layout,std::array{two,zero}).value();
  std::array sources{a,b,a}; // Repeated frames and the shared page grant one permission.
  const std::array<std::int32_t,3> q{0,1,2},kv{0,2,4};
  const std::array<std::int32_t,4> pages{0,1,0,2};
  const std::array<std::int32_t,2> tails{2,1};
  const auto target=schema.version({q,kv,pages,tails}).value();
  auto result=arena.reframe(target,sources);ASSERT_TRUE(result)<<result.error().message();
  auto combined=std::move(*result);
  EXPECT_EQ(combined.layout(),target);EXPECT_EQ(combined.page_count(),3u);
  EXPECT_EQ(c::detail::PagedArenaAccess::pages(combined).size(),3u);
  EXPECT_EQ(sources[0].layout(),a_layout);EXPECT_EQ(sources[1].layout(),b_layout);
  EXPECT_EQ(sources[2].layout(),a_layout);EXPECT_EQ(sources[0].page_count(),3u);
  EXPECT_EQ(first(zero),std::byte{7});EXPECT_EQ(first(one),std::byte{11});
  EXPECT_EQ(first(two),std::byte{17});EXPECT_EQ(first(three),std::byte{23});
  a={};b={};sources={};zero.reset();one.reset();two.reset();three.reset();
  // The result owns selected original leases, not the input frames: their
  // unrelated page becomes writable while every selected page remains pinned.
  EXPECT_TRUE(arena.acquire(3));
  for(unsigned page=0;page<3;++page)EXPECT_TRUE(arena.acquire(page).error().IsUnavailable());
  auto uses=c::LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=3}).value();
  auto reading=uses.begin().value();
  for(const auto&guard:c::detail::PagedArenaAccess::pages(combined))ASSERT_TRUE(reading.read(guard).ok());
  combined={};
  // Dropping the result is not actual retirement of an already admitted use.
  for(unsigned page=0;page<3;++page)EXPECT_FALSE(arena.acquire(page));
  reading={};
  for(unsigned page=0;page<4;++page)EXPECT_TRUE(arena.acquire(page));
}
TEST(PagedArena, ReframeCannotExpandASourcesLogicalPrefixThroughItsWholePageGuard){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto page=fill(arena,0,std::byte{29});
  const auto one=layout(schema,std::array<std::int32_t,1>{0},1);
  const auto two=layout(schema,std::array<std::int32_t,1>{0},2);
  auto old=arena.read(one,std::array{page}).value();
  ASSERT_EQ(c::detail::PagedArenaAccess::pages(old)[0].size(),arena.page_bytes());
  auto expanded=arena.reframe(two,std::array{old});
  ASSERT_FALSE(expanded);EXPECT_TRUE(expanded.error().IsFailedPrecondition());
  EXPECT_EQ(old.layout(),one);EXPECT_EQ(first(page),std::byte{29});
  EXPECT_TRUE(arena.reframe(one,std::array{old}));
  // A second frame explicitly grants the longer evaluated prefix. Merging
  // the same page chooses this authority, not the order of source handles.
  auto longer=arena.read(two,std::array{page}).value();
  auto merged=arena.reframe(two,std::array{old,longer,old});ASSERT_TRUE(merged);
  EXPECT_EQ(merged->page_count(),1u);EXPECT_EQ(old.layout(),one);
  EXPECT_EQ(longer.layout(),two);EXPECT_EQ(first(page),std::byte{29});
}
TEST(PagedArena, ReframeKeepsOldAppendablePrefixBoundedAfterTheFrontierAdvances){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto writer=arena.acquire(0).value();const auto generation=writer.lease().generation();
  const auto row_bytes=arena.page_bytes()/capacity().page_size;
  ASSERT_TRUE(writer.lease().view()->subview(0,row_bytes)->fill(std::byte{31}).ok());
  auto published=writer.publish_appendable(1).value();
  const auto one=layout(schema,std::array<std::int32_t,1>{0},1);
  const auto two=layout(schema,std::array<std::int32_t,1>{0},2);
  auto old=arena.read(one,std::array{published.prefix}).value();
  ASSERT_TRUE(published.appender.reserve_rows(1).ok());
  ASSERT_TRUE(published.appender.view()->fill(std::byte{47}).ok());
  auto current=published.appender.publish().value();
  EXPECT_FALSE(published.appender.valid());EXPECT_EQ(current.generation(),generation);
  EXPECT_EQ(published.prefix.size(),row_bytes);EXPECT_EQ(current.size(),2*row_bytes);
  auto expanded=arena.reframe(two,std::array{old});
  ASSERT_FALSE(expanded);EXPECT_TRUE(expanded.error().IsFailedPrecondition());
  EXPECT_EQ(old.layout(),one);EXPECT_EQ(c::detail::PagedArenaAccess::pages(old)[0].size(),row_bytes);
  EXPECT_EQ(first(published.prefix),std::byte{31});
  EXPECT_TRUE(arena.reframe(one,std::array{old}));
  auto longer=arena.read(two,std::array{current}).value();
  auto combined=arena.reframe(two,std::array{longer,old});ASSERT_TRUE(combined);
  EXPECT_EQ(combined->page_count(),1u);EXPECT_EQ(first(current),std::byte{31});
  std::array<std::byte,1> appended{};
  ASSERT_TRUE(current.view()->subview(row_bytes,1)->read(appended).ok());
  EXPECT_EQ(appended[0],std::byte{47});
}
TEST(PagedArena, ReframeRejectsInvalidForeignAndMissingSourcesWithoutChangingSnapshots){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto page=fill(arena,0,std::byte{53});const auto target=layout(schema,std::array<std::int32_t,1>{0});
  auto source=arena.read(target,std::array{page}).value();
  auto same_schema_arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto foreign_page=fill(same_schema_arena,0,std::byte{59});
  auto foreign=same_schema_arena.read(target,std::array{foreign_page}).value();
  auto other_schema=c::BatchSchema::create(capacity()).value();
  const auto foreign_target=layout(other_schema,std::array<std::int32_t,1>{0});
  const auto invalid=[](const auto&result){EXPECT_FALSE(result);if(!result)EXPECT_TRUE(result.error().IsInvalidArgument());};
  invalid(arena.reframe(c::BatchLayoutVersion{},std::array{source}));
  invalid(arena.reframe(foreign_target,std::array{source}));
  invalid(arena.reframe(target,std::array{c::PagedArenaRead{}}));
  invalid(arena.reframe(target,std::array{source,foreign}));
  auto missing=arena.reframe(layout(schema,std::array<std::int32_t,1>{1}),std::array{source});
  ASSERT_FALSE(missing);EXPECT_TRUE(missing.error().IsFailedPrecondition());
  auto absent=arena.reframe(target,{});ASSERT_FALSE(absent);EXPECT_TRUE(absent.error().IsFailedPrecondition());
  EXPECT_EQ(source.layout(),target);EXPECT_EQ(source.page_count(),1u);EXPECT_EQ(first(page),std::byte{53});
  EXPECT_EQ(foreign.layout(),target);EXPECT_EQ(first(foreign_page),std::byte{59});
  EXPECT_TRUE(arena.reframe(target,std::array{source}));
  c::PagedArena empty_arena;auto empty_result=empty_arena.reframe(target,std::array{source});
  ASSERT_FALSE(empty_result);EXPECT_TRUE(empty_result.error().IsFailedPrecondition());
  ASSERT_TRUE(arena.close().ok());
  auto closed=arena.reframe(target,std::array{source});ASSERT_FALSE(closed);EXPECT_TRUE(closed.error().IsFailedPrecondition());
  // Closing allocation/admission does not invalidate an immutable observation.
  EXPECT_EQ(source.layout(),target);EXPECT_EQ(first(page),std::byte{53});
}
TEST(PagedArena, ReframeEmptyTargetPinsNoPageAndStillValidatesEverySuppliedSource){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto page=fill(arena,0,std::byte{61});
  auto source=arena.read(layout(schema,std::array<std::int32_t,1>{0}),std::array{page}).value();
  const std::array<std::int32_t,2> zero{0,0};const std::array<std::int32_t,1> tail{0};
  const auto empty=schema.version({zero,zero,{},tail}).value();
  auto from_nothing=arena.reframe(empty,{});ASSERT_TRUE(from_nothing);
  auto from_source=arena.reframe(empty,std::array{source});ASSERT_TRUE(from_source);
  EXPECT_EQ(from_source->layout(),empty);EXPECT_EQ(from_source->page_count(),0u);
  EXPECT_TRUE(c::detail::PagedArenaAccess::pages(*from_source).empty());
  EXPECT_EQ(c::detail::PagedArenaAccess::tensor(*from_source)->storage().size(),arena.page_bytes()*capacity().max_pages);
  auto invalid=arena.reframe(empty,std::array{source,c::PagedArenaRead{}});
  ASSERT_FALSE(invalid);EXPECT_TRUE(invalid.error().IsInvalidArgument());
  auto foreign_arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto foreign=foreign_arena.reframe(empty,{}).value();
  auto rejected=arena.reframe(empty,std::array{foreign});ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsInvalidArgument());
  source={};page.reset();
  for(unsigned id=0;id<capacity().max_pages;++id)EXPECT_TRUE(arena.acquire(id));
}
TEST(PagedArena, ReframeMetadataFailureLeavesSourcesReadyForRetry){
  auto schema=c::BatchSchema::create(capacity()).value();m::QuotaAllocator quota(m::system_allocator());
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),
      {.max_backing_bytes=4096,.allocator=quota.get()}).value();
  auto page=fill(arena,0,std::byte{67});const auto target=layout(schema,std::array<std::int32_t,1>{0});
  auto source=arena.read(target,std::array{page}).value();
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();
    auto failed=arena.reframe(target,std::array{source});
    ASSERT_FALSE(failed);EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_EQ(source.layout(),target);EXPECT_EQ(first(page),std::byte{67});}
  auto retried=arena.reframe(target,std::array{source});ASSERT_TRUE(retried);
  EXPECT_EQ(retried->page_count(),1u);EXPECT_EQ(source.layout(),target);EXPECT_EQ(first(page),std::byte{67});
}
} // namespace
TEST(PagedAdmission, ExactBlocksDeduplicateSubviewsWithoutCollapsingSharedBacking){
  auto pool=c::LeasePool::create(m::system_resource(),4096,{.max_blocks=4}).value();
  auto wa=pool.allocate(512).value(),wb=pool.allocate(512).value();
  auto a=wa.publish().value(),b=wb.publish().value();auto sub=a.subview(64,128).value();
  ASSERT_EQ(c::detail::LeaseAccess::backing(a)->identity,c::detail::LeaseAccess::backing(b)->identity);
  c::detail::BatchPageReads reads;ASSERT_TRUE(reads.reserve(6).ok());
  ASSERT_TRUE(reads.add(a).ok());ASSERT_TRUE(reads.add(b).ok());ASSERT_TRUE(reads.add(sub).ok());ASSERT_TRUE(reads.add(b).ok());
  auto uses=c::LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=2}).value();auto use=uses.begin().value();
  ASSERT_TRUE(reads.admit(use).ok());EXPECT_EQ(pool.stats().active_uses,2u);use={};EXPECT_EQ(pool.stats().active_uses,0u);
  ASSERT_TRUE(reads.add(b).ok());ASSERT_TRUE(reads.add(b).ok());use=uses.begin().value();
  ASSERT_TRUE(reads.admit(use).ok());EXPECT_EQ(pool.stats().active_uses,1u);use={};EXPECT_EQ(pool.stats().active_uses,0u);
}
TEST(PagedAdmission, ReserveLimitsAndOriginalClosedPermissionGateRemainEffective){
  auto pool=c::LeasePool::create(m::system_resource(),4096).value();auto writer=pool.allocate(512).value();auto ready=writer.publish().value();
  c::detail::BatchPageReads reads;EXPECT_TRUE(reads.add(ready).IsResourceExhausted());ASSERT_TRUE(reads.reserve(1).ok());
  ASSERT_TRUE(reads.add(ready).ok());EXPECT_TRUE(reads.add(ready).IsResourceExhausted());reads.reset();
  EXPECT_TRUE(reads.add(c::ReadLease{}).IsFailedPrecondition());ASSERT_TRUE(pool.close().ok());ASSERT_TRUE(reads.add(ready).ok());
  auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();EXPECT_TRUE(reads.admit(use).IsFailedPrecondition());EXPECT_EQ(pool.stats().active_uses,0u);
}
