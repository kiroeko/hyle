#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/detail/paged_arena_access.h>
#include <hyle/mem/quota_allocator.h>
#include <gtest/gtest.h>
#include "allocation_fault_fixture.h"
#include <array>
#include <algorithm>

namespace {
namespace c=hyle::compute;
namespace m=hyle::mem;
using Access=c::detail::PagedArenaAccess;
using CopyAccess=c::detail::CopyStreamAccess;
c::BatchCapacity capacity(){return {.max_requests=2,.max_query_rows=2,.max_kv_rows=128,
  .max_pages=4,.max_page_refs=4,.query_heads=2,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=2};}
c::BatchLayoutVersion one(const c::BatchSchema&schema,std::span<const std::int32_t> ids,int tail){
  const std::array<std::int32_t,2> q{0,1},kv{0,std::int32_t(ids.size())};
  const std::array<std::int32_t,1> tails{tail};return schema.version({q,kv,ids,tails}).value();
}
c::ReadLease fill(c::PagedArena&arena,unsigned id,unsigned rows,std::byte value){
  auto writer=arena.acquire(id).value();EXPECT_TRUE(writer.lease().view()->fill(value).ok());
  return writer.publish(rows).value();
}
struct Copy final {
  explicit Copy(std::size_t bytes):context(c::Context::create(c::cpu::device()).value()),
      queue(context.create_queue().value()),program(c::CopyBatchProgram::prepare(context,
        std::array{c::CopyResourceSpec{m::system_resource(),bytes},
                   c::CopyResourceSpec{m::system_resource(),bytes}}).value()),
      invocation(c::CopyBatchInvocation::create(program,queue).value()){}
  Copy(const Copy&)=delete;Copy&operator=(const Copy&)=delete;
  c::Context context;c::Queue queue;c::CopyBatchProgram program;c::CopyBatchInvocation invocation;
};
hyle::Status validate(const c::PagedArenaPendingRead&frame,const c::CopyBatchUpdate&update){
  return Access::validate<c::CopyBatchUpdate,CopyAccess>(frame,update);
}
hyle::Status admit(c::LeaseUseBatch&uses,const c::PagedArenaPendingRead&frame,const c::CopyBatchUpdate&update){
  return Access::admit_read<c::CopyBatchUpdate,CopyAccess>(uses,frame,update);
}
c::CopyBatchUpdate submit(Copy&copy,const c::ReadLease&source,c::PagedPageWriter&writer,
    std::span<const c::CopyPhase> phases,std::span<const c::CopyRange> ranges){
  EXPECT_TRUE(copy.invocation.bind_source({0},source).ok());
  EXPECT_TRUE(copy.invocation.bind_mutable({1},std::move(writer.lease())).ok());
  EXPECT_TRUE(copy.invocation.prepare_replay(phases,ranges).ok());
  auto submitted=copy.invocation.submit().value();
  return c::CopyBatchUpdate::take(std::move(copy.invocation),std::move(submitted)).value();
}
TEST(PagedArenaPending, ConstructionKeepsExactOwnedLayoutAndRejectsUnconsumedForeignOrDuplicatePages){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto clean=fill(arena,1,2,std::byte{7});auto writer=arena.acquire(0).value();
  auto layout=one(schema,std::array<std::int32_t,2>{0,1},1);
  const std::array<c::PagedPageUpdate,1> updates{{{&writer,{1}}}};
  EXPECT_FALSE(arena.pending_read(layout,std::array{clean},updates));
  auto escaped=std::move(writer.lease());
  auto frame=arena.pending_read(layout,std::array{clean},updates).value();
  EXPECT_EQ(frame.layout(),layout);EXPECT_EQ(frame.page_count(),2u);EXPECT_EQ(frame.pending_count(),1u);
  auto retained=frame;frame={};EXPECT_TRUE(retained.valid());EXPECT_FALSE(arena.acquire(1));
  c::CopyBatchUpdate empty;EXPECT_FALSE(validate(retained,empty).ok());
  EXPECT_FALSE(arena.pending_read(layout,{},updates));
  EXPECT_FALSE(arena.pending_read(layout,std::array{clean,clean},updates));
  const std::array<c::PagedPageUpdate,2> duplicate{{{&writer,{1}},{&writer,{2}}}};
  EXPECT_FALSE(arena.pending_read(one(schema,std::array<std::int32_t,2>{0,0},1),{},duplicate));
  auto foreign=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto other=foreign.acquire(0).value();auto other_lease=std::move(other.lease());
  EXPECT_FALSE(arena.pending_read(layout,std::array{clean},std::array<c::PagedPageUpdate,1>{{{&other,{1}}}}));
  EXPECT_FALSE(arena.pending_read(layout,std::array{clean},std::array<c::PagedPageUpdate,1>{{{&writer,{}}}}));
  auto foreign_schema=c::BatchSchema::create(capacity()).value();
  EXPECT_FALSE(arena.pending_read(one(foreign_schema,std::array<std::int32_t,2>{0,1},1),std::array{clean},updates));
  writer.lease()=std::move(escaped);other.lease()=std::move(other_lease);
}
TEST(PagedArenaPending, AcceptedCoverageRejectsByteHolesAndDoesNotConsumeOriginalControls){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto source=fill(arena,1,2,std::byte{23});auto writer=arena.acquire(0).value();Copy copy(arena.page_bytes());
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,2> ranges{{{{0},{1},{0},0,0,128},{{0},{1},{1},256,256,256}}};
  auto update=submit(copy,source,writer,phases,ranges);
  auto frame=arena.pending_read(one(schema,std::array<std::int32_t,1>{0},2),{},std::array<c::PagedPageUpdate,1>{{{&writer,{1}}}}).value();
  EXPECT_TRUE(CopyAccess::covers(update,{1},0,128).ok());
  EXPECT_TRUE(CopyAccess::covers(update,{1},256,256).ok());
  EXPECT_TRUE(CopyAccess::covers(update,{1},0,512).IsOutOfRange());
  EXPECT_FALSE(CopyAccess::covers(update,{0},0,128).ok());
  EXPECT_TRUE(CopyAccess::covers(update,{1},SIZE_MAX,2).IsOutOfRange());
  EXPECT_TRUE(validate(frame,update).IsOutOfRange());EXPECT_TRUE(update.valid());
  auto pool=c::LeaseUsePool::create().value();auto use=pool.begin().value();EXPECT_FALSE(admit(use,frame,update).ok());
  use={};ASSERT_TRUE(update.wait().ok());writer.lease()=update.take_mutable({1}).value();
  EXPECT_EQ(writer.initialized_rows(),0u);auto ready=writer.publish(0).value();
  EXPECT_FALSE(arena.read(one(schema,std::array<std::int32_t,1>{0},1),std::array{ready}));
}
TEST(PagedArenaPending, SharedAndNonTailReferencesDemandTheMaximumInitializedDomain){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto source=fill(arena,1,2,std::byte{31});auto writer=arena.acquire(0).value();Copy copy(arena.page_bytes());
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,256}}};
  auto update=submit(copy,source,writer,phases,ranges);
  const std::array<std::int32_t,3> ptr{0,1,2};const std::array<std::int32_t,2> ids{0,0},tails{1,2};
  auto layout=schema.version({ptr,ptr,ids,tails}).value();
  const std::array<c::PagedPageUpdate,1> updates{{{&writer,{1}}}};
  auto frame=arena.pending_read(layout,{},updates).value();EXPECT_EQ(frame.page_count(),1u);
  EXPECT_EQ(Access::pending(frame)[0].required_rows,2u);EXPECT_TRUE(validate(frame,update).IsOutOfRange());
  auto non_tail=arena.pending_read(one(schema,std::array<std::int32_t,2>{0,0},1),{},updates).value();
  EXPECT_TRUE(validate(non_tail,update).IsOutOfRange());
  auto short_frame=arena.pending_read(one(schema,std::array<std::int32_t,1>{0},1),{},updates).value();
  EXPECT_TRUE(validate(short_frame,update).ok());ASSERT_TRUE(update.wait().ok());
  writer.lease()=update.take_mutable({1}).value();auto ready=writer.publish(1).value();
  EXPECT_FALSE(arena.read(layout,std::array{ready}));
}
TEST(PagedArenaPending, OriginalRootFullRangeAndReservationAreRequiredAtAdmission){
  auto schema=c::BatchSchema::create(capacity()).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto source=fill(arena,1,2,std::byte{37});auto old=fill(arena,0,1,std::byte{11});
  auto writer=arena.acquire(old).value();Copy copy(arena.page_bytes());
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,256,256}}};
  auto update=submit(copy,source,writer,phases,ranges);
  auto frame=arena.pending_read(one(schema,std::array<std::int32_t,1>{0},2),{},std::array<c::PagedPageUpdate,1>{{{&writer,{1}}}}).value();
  const auto record=Access::pending(frame)[0];EXPECT_EQ(record.new_offset,256u);EXPECT_EQ(record.new_bytes,256u);
  EXPECT_TRUE(validate(frame,update).ok());auto original=CopyAccess::input(update,{1}).value();
  auto partial=original;partial.view=original.view.subview(0,256).value();EXPECT_TRUE(Access::validate_page(frame,0,partial).IsInvalidArgument());
  auto wrong=original;wrong.root=nullptr;EXPECT_TRUE(Access::validate_page(frame,0,wrong).IsInvalidArgument());
  auto foreign=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096}).value();
  auto other=foreign.acquire(0).value();Copy other_copy(arena.page_bytes());auto other_update=submit(other_copy,source,other,phases,ranges);
  EXPECT_TRUE(validate(frame,other_update).IsInvalidArgument());EXPECT_TRUE(other_update.valid());
  ASSERT_TRUE(update.wait().ok());writer.lease()=update.take_mutable({1}).value();old=writer.publish(2).value();
  EXPECT_TRUE(Access::validate_page(frame,0,original).IsFailedPrecondition());
  // The snapshot's root is not permission to admit another reservation.
  auto next=arena.acquire(old).value();EXPECT_TRUE(Access::validate_page(frame,0,original).IsFailedPrecondition());
}
TEST(PagedArenaPending, EveryColdFrameAllocationFailureRollsBackWithoutConsumingSource){
  auto schema=c::BatchSchema::create(capacity()).value();hyle_f5_fault_fixture::Allocator fault;
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=4096,.allocator=fault.ref()}).value();
  auto clean=fill(arena,1,2,std::byte{41});auto writer=arena.acquire(0).value();Copy copy(arena.page_bytes());
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,512}}};
  auto update=submit(copy,clean,writer,phases,ranges);auto layout=one(schema,std::array<std::int32_t,2>{0,1},2);
  const std::array<c::PagedPageUpdate,1> updates{{{&writer,{1}}}};const auto live=fault.live;fault.attempts=0;
  {auto frame=arena.pending_read(layout,std::array{clean},updates).value();EXPECT_TRUE(validate(frame,update).ok());}
  const auto count=fault.attempts;EXPECT_EQ(count,4u);EXPECT_EQ(fault.live,live);
  for(std::size_t nth=0;nth<count;++nth){fault.attempts=0;fault.fail=nth;
    auto frame=arena.pending_read(layout,std::array{clean},updates);ASSERT_FALSE(frame);EXPECT_TRUE(frame.error().IsResourceExhausted());
    EXPECT_EQ(fault.live,live);EXPECT_TRUE(update.valid());EXPECT_TRUE(writer.valid());}
  fault.fail=SIZE_MAX;EXPECT_TRUE(arena.pending_read(layout,std::array{clean},updates));ASSERT_TRUE(update.wait().ok());
}
TEST(PagedArenaPending, TwoRequestsAndTwoOriginalCopySlotsAppendSixtyFourRowsWithoutRewritingHistory){
  auto cap=capacity();cap.page_size=32;auto schema=c::BatchSchema::create(cap).value();m::QuotaAllocator quota(m::system_allocator());
  auto made=c::PagedArena::create(schema,c::BatchCacheRole::kKey,m::system_resource(),{.max_backing_bytes=32769,.allocator=quota.get()});
  ASSERT_TRUE(made)<<made.error().message();auto arena=std::move(*made);
  auto shared=fill(arena,2,1,std::byte{5});Copy copy(arena.page_bytes());
  std::array<c::CopyBatchInvocation,2> invocations{std::move(copy.invocation),c::CopyBatchInvocation::create(copy.program,copy.queue).value()};
  auto source_pool=c::LeasePool::create(m::system_resource(),512).value();auto incoming=source_pool.allocate(256).value();c::ReadLease source;
  auto uses=c::LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=4,.allocator=quota.get()}).value();
  std::array<c::ReadLease,2> ready;std::array<std::uint64_t,2> generations{};std::array<unsigned,2> rows{};
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<std::int32_t,3> ptr{0,1,2};
  for(unsigned step=0;step<64;++step){
    const auto slot=step%2;if(source.valid())incoming=source.try_write().value();
    ASSERT_TRUE(incoming.view()->fill(std::byte(step+1)).ok());source=incoming.publish().value();
    if(ready[slot].valid()){
      auto old_frame=arena.read(one(schema,std::array<std::int32_t,1>{std::int32_t(slot)},int(rows[slot])),std::array{ready[slot]}).value();
      EXPECT_FALSE(arena.acquire(ready[slot]));EXPECT_TRUE(ready[slot].valid());
    }
    auto writer=(ready[slot].valid()?arena.acquire(ready[slot]):arena.acquire(slot)).value();
    EXPECT_EQ(writer.initialized_rows(),rows[slot]);if(!rows[slot])generations[slot]=writer.lease().generation();
    ASSERT_TRUE(invocations[slot].bind_source({0},source).ok());ASSERT_TRUE(invocations[slot].bind_mutable({1},std::move(writer.lease())).ok());
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,std::size_t(rows[slot])*256,256}}};
    ASSERT_TRUE(invocations[slot].prepare_replay(phases,ranges).ok());auto submitted=invocations[slot].submit().value();
    auto update=c::CopyBatchUpdate::take(std::move(invocations[slot]),std::move(submitted)).value();
    const std::array<std::int32_t,2> ids{std::int32_t(slot),2},tails{std::int32_t(rows[slot]+1),1};
    auto layout=schema.version({ptr,ptr,ids,tails}).value();
    auto frame=arena.pending_read(layout,std::array{shared},std::array<c::PagedPageUpdate,1>{{{&writer,{1}}}}).value();
    EXPECT_EQ(frame.page_count(),2u);EXPECT_EQ(frame.pending_count(),1u);
    auto reading=uses.begin().value();
    {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();ASSERT_TRUE(validate(frame,update).ok());ASSERT_TRUE(admit(reading,frame,update).ok());EXPECT_TRUE(arena.acquire(3));}
    ASSERT_TRUE(update.wait().ok());EXPECT_EQ(update.report().logical_bytes,256u);
    auto original=CopyAccess::input(update,{1}).value();std::array<std::byte,8192> actual{};
    ASSERT_TRUE(original.view.subview(0,std::size_t(rows[slot]+1)*256)->read(std::span(actual).first(std::size_t(rows[slot]+1)*256)).ok());
    for(unsigned row=0;row<=rows[slot];++row)for(unsigned byte=0;byte<256;++byte)ASSERT_EQ(actual[row*256+byte],std::byte(2*row+slot+1));
    EXPECT_FALSE(update.take_mutable({1}));reading={};writer.lease()=update.take_mutable({1}).value();
    ready[slot]=writer.publish(++rows[slot]).value();EXPECT_EQ(ready[slot].generation(),generations[slot]);
    invocations[slot]=update.release_invocation().value();ASSERT_TRUE(invocations[slot].reset().ok());
    auto complete=arena.read(layout,std::array{ready[slot],shared}).value();EXPECT_EQ(complete.page_count(),2u);
  }
}
} // namespace
