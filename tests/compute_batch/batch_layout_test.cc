#include <hyle/compute/batch_ops.h>
#include <hyle/compute/state_versions.h>
#include <hyle/mem/quota_allocator.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace {
namespace c=hyle::compute;
c::BatchCapacity capacity(){return {.max_requests=3,.max_query_rows=8,.max_kv_rows=128,.max_pages=8,.max_page_refs=8,.query_heads=4,.kv_heads=2,.head_dim=64,.page_size=16};}
struct Layout {
  std::array<std::int32_t,3> q{0,1,3},kv{0,2,3},pages{2,0,3};std::array<std::int32_t,2> tails{3,10};
  c::HostBatchLayout host()const{return {q,kv,pages,tails};}
};
TEST(BatchLayout, ActiveRowsAreDistinctFromPhysicalCapacityAndHostIsOwned){
  const auto schema=c::BatchSchema::create(capacity()).value();Layout input;auto version=schema.version(input.host()).value();
  EXPECT_EQ(version.facts().requests,2u);EXPECT_EQ(version.facts().query_rows,3u);EXPECT_EQ(version.facts().kv_rows,29u);EXPECT_EQ(version.facts().page_refs,3u);
  input.pages[0]=7;input.q[1]=2;input.tails[0]=16;
  EXPECT_EQ(version.host().page_indices[0],2);EXPECT_EQ(version.host().query_indptr[1],1);EXPECT_EQ(version.host().last_page_lengths[0],3);
  EXPECT_EQ(version.metadata()->size(),schema.metadata_offsets().bytes);EXPECT_GT(version.metadata()->size(),input.q.size()*sizeof(int));
}
TEST(BatchLayout, SameShapeNewPageValuesHaveNewGenerationAndOldVersionRemains){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;auto old=schema.version(input.host()).value();input.pages={6,5,4};auto next=schema.version(input.host()).value();
  EXPECT_GT(next.generation(),old.generation());EXPECT_FALSE(next==old);EXPECT_EQ(old.host().page_indices[0],2);EXPECT_EQ(next.host().page_indices[0],6);
  auto copy=old;EXPECT_TRUE(copy==old);schema={};old={};EXPECT_EQ(copy.host().last_page_lengths[1],10);
}
TEST(BatchLayout, ForeignSchemaCannotAdmitAnIdenticalCapacityVersion){
  auto a=c::BatchSchema::create(capacity()).value(),b=c::BatchSchema::create(capacity()).value();Layout input;auto version=a.version(input.host()).value();
  EXPECT_TRUE(a.owns(version));EXPECT_FALSE(b.owns(version));EXPECT_FALSE(a.owns({}));EXPECT_FALSE(c::BatchSchema{}.owns(version));
}
TEST(BatchLayout, WrongIndptrStartMonotonicEndAndRequestCountAreRejected){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;input.q[0]=1;EXPECT_FALSE(schema.version(input.host()));input.q={0,3,2};EXPECT_FALSE(schema.version(input.host()));
  input.q={0,1,3};input.kv[0]=1;EXPECT_FALSE(schema.version(input.host()));input.kv={0,3,2};EXPECT_FALSE(schema.version(input.host()));input.kv={0,1,2};EXPECT_FALSE(schema.version(input.host()));
  input.kv={0,2,3};auto h=input.host();h.kv_indptr=h.kv_indptr.first(2);EXPECT_FALSE(schema.version(h));h=input.host();h.query_indptr={};EXPECT_FALSE(schema.version(h));
}
TEST(BatchLayout, PhysicalPageBoundsAndTailBoundsAreCheckedBeforeSnapshot){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;input.pages[0]=-1;EXPECT_TRUE(schema.version(input.host()).error().IsOutOfRange());input.pages[0]=8;EXPECT_FALSE(schema.version(input.host()));input.pages[0]=2;
  input.tails[0]=0;EXPECT_FALSE(schema.version(input.host()));input.tails[0]=17;EXPECT_FALSE(schema.version(input.host()));input.tails[0]=-1;EXPECT_FALSE(schema.version(input.host()));
}
TEST(BatchLayout, DuplicatePagesAreLegalReadonlySharing){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;input.pages={2,2,2};auto version=schema.version(input.host()).value();EXPECT_EQ(version.facts().kv_rows,29u);EXPECT_EQ(version.facts().page_refs,3u);
}
TEST(BatchLayout, QueryWithoutKvAndCapacityExceededAreRejected){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;input.q={0,20,22};EXPECT_FALSE(schema.version(input.host()));input.q={0,1,3};input.kv={0,0,3};input.tails[0]=0;EXPECT_FALSE(schema.version(input.host()));
  input.kv={0,2,3};input.tails={16,16};auto smaller=capacity();smaller.max_kv_rows=29;auto limited=c::BatchSchema::create(smaller).value();EXPECT_TRUE(limited.version(input.host()).error().IsOutOfRange());
}
TEST(BatchLayout, EmptyBatchAndEmptyRequestHaveExplicitEmptyEffectiveRegions){
  auto schema=c::BatchSchema::create(capacity()).value();std::array<std::int32_t,1> zero{0};auto empty=schema.version({zero,zero,{},{}}).value();EXPECT_EQ(empty.facts().requests,0u);EXPECT_EQ(empty.facts().query_rows,0u);EXPECT_TRUE(empty.metadata()->valid());
  std::array<std::int32_t,2> ptr{0,0};std::array<std::int32_t,1> tail{0};auto request=schema.version({ptr,ptr,{},tail}).value();EXPECT_EQ(request.facts().requests,1u);EXPECT_EQ(request.facts().kv_rows,0u);tail[0]=1;EXPECT_FALSE(schema.version({ptr,ptr,{},tail}));
}
TEST(BatchLayout, RaggedUsesPackedRowsAndRejectsPagedSideTables){
  auto selected=capacity();selected.storage=c::BatchStorage::kRagged;selected.max_pages=selected.max_page_refs=0;selected.page_size=1;
  auto schema=c::BatchSchema::create(selected).value();Layout input;std::array<std::int32_t,3> kv{0,19,29};auto version=schema.version({input.q,kv,{},{}}).value();EXPECT_EQ(version.facts().kv_rows,29u);EXPECT_TRUE(version.host().page_indices.empty());
  EXPECT_FALSE(schema.version({input.q,kv,input.pages,input.tails}));
}
TEST(BatchLayout, DtypeGqaIntegerAndPhysicalStrideOverflowAreRejected){
  auto input=capacity();input.data_type=c::DataType::kFloat32;EXPECT_FALSE(c::BatchSchema::create(input));input=capacity();input.query_heads=3;EXPECT_FALSE(c::BatchSchema::create(input));input=capacity();input.max_query_rows=UINT32_MAX;EXPECT_TRUE(c::BatchSchema::create(input).error().IsOutOfRange());
  input=capacity();input.kv_heads=input.query_heads=1u<<30;input.head_dim=input.page_size=1u<<30;EXPECT_TRUE(c::BatchSchema::create(input).error().IsOutOfRange());input=capacity();input.max_pages=0;EXPECT_FALSE(c::BatchSchema::create(input));
}
TEST(BatchLayout, QuotaFailureDoesNotConsumeCallerSpansOrExistingVersion){
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),1024u*1024u);{
    auto schema=c::BatchSchema::create(capacity(),{.allocator=quota.get()}).value();Layout input;auto old=schema.version(input.host()).value();auto reserved=quota.try_reserve_bytes(quota.remaining());ASSERT_TRUE(reserved);
    EXPECT_TRUE(schema.version(input.host()).error().IsResourceExhausted());EXPECT_EQ(quota.used(),quota.limit());EXPECT_EQ(old.host().page_indices[0],2);EXPECT_EQ(input.pages[0],2);
  }EXPECT_EQ(quota.used(),0u);
}
TEST(BatchLayout, ExportedReadonlyBlobOutlivesAllSchemaAndVersionHandles){
  c::ReadLease blob;hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());{
    auto schema=c::BatchSchema::create(capacity(),{.allocator=quota.get()}).value();Layout input;auto version=schema.version(input.host()).value();blob=version.metadata().value();
  }EXPECT_GT(quota.used(),0u);std::array<std::byte,32> header;EXPECT_TRUE(blob.view()->subview(0,header.size())->read(header).ok());EXPECT_FALSE(blob.try_write());blob.reset();EXPECT_EQ(quota.used(),0u);
}
TEST(BatchLayout, ConcurrentColdVersionsHaveDistinctGenerationAndOwnedData){
  auto schema=c::BatchSchema::create(capacity()).value();std::array<std::uint64_t,8> ids{};std::array<std::thread,8> threads;
  for(unsigned i=0;i<threads.size();++i)threads[i]=std::thread([&,i]{Layout input;input.pages[0]=std::int32_t(i);auto version=schema.version(input.host()).value();ids[i]=version.generation();EXPECT_EQ(version.host().page_indices[0],std::int32_t(i));});
  for(auto&t:threads)t.join();std::sort(ids.begin(),ids.end());EXPECT_EQ(std::unique(ids.begin(),ids.end()),ids.end());EXPECT_GT(ids[0],0u);
}
TEST(BatchLayout, TypedAttentionPortShapesAndOpaqueMetadataEncodingAreChecked){
  auto cap=capacity();auto schema=c::BatchSchema::create(cap).value();auto q=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{8,4,64}).value();auto kv=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,4>{8,16,2,64}).value();
  std::array inputs{q,kv,kv,schema.metadata_description().value()};std::array<c::TensorDesc,1> outputs;
  auto op=c::Operation::custom(c::PagedAttention{cap});EXPECT_TRUE(c::PagedAttention{cap}.infer_output_descriptions(inputs,outputs).ok());EXPECT_EQ(outputs[0],q);EXPECT_TRUE(op.custom_as<c::PagedAttention>());EXPECT_FALSE(op.custom_as<c::RaggedAttention>());
  inputs[3]=q;EXPECT_FALSE(c::PagedAttention{cap}.infer_output_descriptions(inputs,outputs).ok());
}
TEST(BatchLayout, F4BranchSharesAndPublishesLayoutBlobWithoutMutatingOldVersion){
  auto batch=c::BatchSchema::create(capacity()).value();Layout input;auto layout=batch.version(input.host()).value();
  auto schema=c::StateSchema::create(std::array{c::StateEntrySpec{.description=batch.metadata_description().value()}}).value();auto entry=schema.entry(0).value();auto store=c::StateStore::create(schema).value();
  auto initial=store.import(std::array{c::StateMember{entry,layout.metadata().value()}}).value();auto head=c::StateHead::create(store,initial).value();auto branch=store.begin(initial).value();
  input.pages={6,5,4};auto next=batch.version(input.host()).value();ASSERT_TRUE(branch.replace(entry,next.metadata().value()).ok());auto candidate=branch.finalize().value();ASSERT_TRUE(head.publish(initial.identity(),candidate).ok());
  auto old=initial.member(entry).value(),current=head.current()->member(entry).value();
  auto old_map=old.view()->map_host().value(),new_map=current.view()->map_host().value();const auto offset=batch.metadata_offsets().page_indices;
  std::int32_t old_id=0,new_id=0;std::memcpy(&old_id,old_map.bytes().data()+offset,sizeof(old_id));std::memcpy(&new_id,new_map.bytes().data()+offset,sizeof(new_id));
  EXPECT_EQ(old_id,2);EXPECT_EQ(new_id,6);EXPECT_TRUE(batch.owns(layout));EXPECT_TRUE(batch.owns(next));
}
} // namespace
