#include <hyle/compute/batch_ops.h>
#include <hyle/compute/state_versions.h>
#include <hyle/compute/detail/lease_access.h>
#include <hyle/compute/detail/batch_provider.h>
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
c::BatchCapacity capacity(){return {.max_requests=3,.max_query_rows=8,.max_kv_rows=128,.max_pages=8,.max_page_refs=8,.query_heads=4,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=16};}
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
TEST(BatchLayout, ActiveSnapshotIsCompactAndExplicitExportZerosAllInactiveCapacity){
  auto cap=capacity();cap.max_requests=4096;cap.max_page_refs=65536;
  auto schema=c::BatchSchema::create(cap).value();Layout input;auto version=schema.version(input.host()).value();
  const auto host=version.host();
  EXPECT_EQ(reinterpret_cast<const std::byte*>(host.kv_indptr.data())-reinterpret_cast<const std::byte*>(host.query_indptr.data()),std::ptrdiff_t(input.q.size()*4));
  std::size_t active=0;for(const auto&segment:c::detail::BatchLayoutAccess::segments(version))active+=segment.bytes.size();
  EXPECT_EQ(active,32u+input.q.size()*4+input.kv.size()*4+input.pages.size()*4+input.tails.size()*4);
  EXPECT_LT(active,schema.metadata_offsets().bytes/1000);
  std::vector<std::byte> expected(schema.metadata_offsets().bytes,std::byte{0});
  const auto put=[&](std::size_t at,const auto&values){const auto bytes=std::as_bytes(std::span(values));std::memcpy(expected.data()+at,bytes.data(),bytes.size());};
  const std::array<std::uint64_t,2> prefix{0x48594c45424c0001ULL,version.generation()};
  const std::array<std::uint32_t,4> facts{2,3,29,3};put(0,prefix);put(16,facts);
  const auto offsets=schema.metadata_offsets();put(offsets.query_indptr,input.q);put(offsets.kv_indptr,input.kv);put(offsets.page_indices,input.pages);put(offsets.last_page_lengths,input.tails);
  auto exported=version.metadata().value();std::vector<std::byte> actual(expected.size());
  ASSERT_TRUE(exported.view()->read(actual).ok());EXPECT_EQ(actual,expected);
  EXPECT_EQ(c::detail::LeaseAccess::identity(exported),c::detail::LeaseAccess::identity(version.metadata().value()));
}
TEST(BatchLayout, ConcurrentFirstFullExportsShareOneImmutableBacking){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;auto version=schema.version(input.host()).value();
  std::array<c::ReadLease,8> results;std::array<std::thread,8> threads;
  for(unsigned i=0;i<threads.size();++i)threads[i]=std::thread([&,i]{auto result=version.metadata();EXPECT_TRUE(result);if(result)results[i]=std::move(*result);});
  for(auto&thread:threads)thread.join();
  for(const auto&lease:results){ASSERT_TRUE(lease.valid());EXPECT_EQ(c::detail::LeaseAccess::identity(lease),c::detail::LeaseAccess::identity(results[0]));}
}
TEST(BatchLayout, LazyPageFootprintIsActiveSizedImmutableAndRetryable){
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),1u<<20);
  auto cap=capacity();cap.max_page_refs=65536;
  auto schema=c::BatchSchema::create(cap,{.allocator=quota.get()}).value();
  Layout input;input.pages={3,0,3};auto old=schema.version(input.host()).value();
  const auto before=quota.used();
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();
    auto failed=c::detail::BatchLayoutAccess::page_footprint(old);ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsResourceExhausted());EXPECT_EQ(old.host().page_indices[0],3);}
  auto footprint=c::detail::BatchLayoutAccess::page_footprint(old).value();
  ASSERT_EQ(footprint.size(),2u);EXPECT_EQ(footprint[0],0);EXPECT_EQ(footprint[1],3);
  EXPECT_EQ(quota.used()-before,input.pages.size()*sizeof(std::int32_t));
  const auto*original=footprint.data();
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();
    EXPECT_EQ(c::detail::BatchLayoutAccess::page_footprint(old)->data(),original);}
  input.pages={7,2,7};auto next=schema.version(input.host()).value();
  auto next_footprint=c::detail::BatchLayoutAccess::page_footprint(next).value();
  EXPECT_EQ(next_footprint[0],2);EXPECT_EQ(next_footprint[1],7);
  EXPECT_EQ(footprint.data(),original);EXPECT_EQ(footprint[0],0);EXPECT_EQ(footprint[1],3);
  std::array<std::int32_t,1> zero{0};auto empty=schema.version({zero,zero,{},{}}).value();
  {auto reserved=quota.try_reserve_bytes(quota.remaining()).value();
    EXPECT_TRUE(c::detail::BatchLayoutAccess::page_footprint(empty)->empty());}
  EXPECT_TRUE(c::detail::BatchLayoutAccess::page_footprint({}).error().IsFailedPrecondition());
}
TEST(BatchLayout, ConcurrentFirstFootprintsAndExportsShareImmutableStorage){
  auto schema=c::BatchSchema::create(capacity()).value();Layout input;input.pages={3,0,3};
  auto version=schema.version(input.host()).value();
  std::array<std::span<const std::int32_t>,8> results;std::array<std::thread,8> threads;
  for(unsigned i=0;i<threads.size();++i)threads[i]=std::thread([&,i]{
    if(i%2)EXPECT_TRUE(version.metadata());
    auto result=c::detail::BatchLayoutAccess::page_footprint(version);EXPECT_TRUE(result);if(result)results[i]=*result;
  });
  for(auto&thread:threads)thread.join();
  for(auto footprint:results){ASSERT_EQ(footprint.size(),2u);EXPECT_EQ(footprint.data(),results[0].data());EXPECT_EQ(footprint[0],0);EXPECT_EQ(footprint[1],3);}
}
TEST(BatchLayout, PageFootprintUsesTheOriginalBudgetAndRetiresWithItsVersion){
  const auto resource=hyle::mem::system_resource();
  const std::array limits{c::ResourceBudgetLimit{resource,1u<<20}};
  auto budget=c::ResourceBudget::create({.limits=limits}).value();
  const auto base=budget.usage(resource)->live_bytes;
  {auto schema=c::BatchSchema::create(capacity(),{.resource_budget=budget}).value();Layout input;
    auto version=schema.version(input.host()).value();const auto before=budget.usage(resource).value();
    {auto pressure=budget.reserve(resource,before.limit_bytes-before.live_bytes).value();
      EXPECT_TRUE(c::detail::BatchLayoutAccess::page_footprint(version).error().IsResourceExhausted());}
    ASSERT_TRUE(c::detail::BatchLayoutAccess::page_footprint(version));
    const auto after=budget.usage(resource).value();
    EXPECT_EQ(after.live_bytes-before.live_bytes,input.pages.size()*sizeof(std::int32_t));
    EXPECT_EQ(after.live_by_category[0]-before.live_by_category[0],input.pages.size()*sizeof(std::int32_t));
    const auto retained=after.live_bytes;EXPECT_TRUE(c::detail::BatchLayoutAccess::page_footprint(version));
    EXPECT_EQ(budget.usage(resource)->live_bytes,retained);}
  EXPECT_EQ(budget.usage(resource)->live_bytes,base);
}
TEST(BatchLayout, MetadataBackingKeepsItsCategoryQuotaAndActualReaderLifetime){
  const auto resource=hyle::mem::system_resource();
  const std::array limits{c::ResourceBudgetLimit{resource,1u<<20}};
  auto budget=c::ResourceBudget::create({.limits=limits}).value();
  const auto base=budget.usage(resource)->live_bytes;
  const auto external=static_cast<std::size_t>(c::ResourceBudgetCategory::kExternal);
  const auto metadata=static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata);
  c::ReadLease escaped;
  {
    auto schema=c::BatchSchema::create(capacity(),{.resource_budget=budget}).value();Layout input;
    auto version=schema.version(input.host()).value();
    auto usage=budget.usage(resource).value();
    EXPECT_EQ(usage.live_by_category[external],0u);
    EXPECT_EQ(usage.live_by_category[metadata],usage.live_bytes);
    {
      auto pressure=budget.reserve(resource,usage.limit_bytes-usage.live_bytes).value();
      auto failed=version.metadata();ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
      EXPECT_EQ(version.host().page_indices[0],2);
    }
    escaped=version.metadata().value();
    // Promotion into the shared backing ledger consumes the same metadata
    // charge once, rather than reclassifying it as external model state.
    ASSERT_TRUE(c::detail::LeaseAccess::retain_budget(escaped,resource,budget).ok());
    const auto retained=budget.usage(resource)->live_bytes;
    ASSERT_TRUE(c::detail::LeaseAccess::retain_budget(escaped,resource,budget).ok());
    EXPECT_EQ(budget.usage(resource)->live_bytes,retained);
    usage=budget.usage(resource).value();
    EXPECT_EQ(usage.live_by_category[external],0u);
    EXPECT_EQ(usage.live_by_category[metadata],usage.live_bytes);
  }
  EXPECT_GT(budget.usage(resource)->live_bytes,base);
  auto uses=c::LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=1}).value();
  auto batch=uses.begin().value();ASSERT_TRUE(batch.read(escaped).ok());escaped.reset();
  EXPECT_GT(budget.usage(resource)->live_bytes,base);
  auto done=c::Completion::completed();ASSERT_TRUE(batch.attach(done).ok());
  EXPECT_EQ(budget.usage(resource)->live_bytes,base);
}
TEST(BatchLayout, FailedExplicitExportPreservesActiveVersionAndCanRetry){
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),1u<<20);
  auto schema=c::BatchSchema::create(capacity(),{.allocator=quota.get()}).value();Layout input;auto version=schema.version(input.host()).value();
  {auto reserved=quota.try_reserve_bytes(quota.remaining());ASSERT_TRUE(reserved);auto failed=version.metadata();ASSERT_FALSE(failed);EXPECT_TRUE(failed.error().IsResourceExhausted());EXPECT_EQ(version.host().page_indices[0],2);}
  auto exported=version.metadata();ASSERT_TRUE(exported);EXPECT_EQ(exported->size(),schema.metadata_offsets().bytes);
  {auto reserved=quota.try_reserve_bytes(quota.remaining());ASSERT_TRUE(reserved);EXPECT_TRUE(version.metadata());}
}
TEST(BatchLayout, ForeignSchemaCannotAdmitAnIdenticalCapacityVersion){
  auto a=c::BatchSchema::create(capacity()).value(),b=c::BatchSchema::create(capacity()).value();Layout input;auto version=a.version(input.host()).value();
  EXPECT_TRUE(a.owns(version));EXPECT_FALSE(b.owns(version));EXPECT_FALSE(a.owns({}));EXPECT_FALSE(c::BatchSchema{}.owns(version));
}
TEST(BatchLayout, StorageViewsShareExplicitRootOwnershipAndTheOriginalMetadata){
  auto cap=capacity();cap.output_type=c::DataType::kBFloat16;cap.value_head_dim=128;
  auto root=c::BatchSchema::create(cap).value();Layout input;
  auto layout=root.version(input.host()).value();auto metadata=layout.metadata().value();
  const auto identity=c::detail::LeaseAccess::identity(metadata);
  const auto key=c::CacheStorage::float8(.5F).value(),value=c::CacheStorage::float8(.25F).value();
  auto view=root.with_cache_storage(key,value).value();
  auto expected=cap;expected.key_storage=key;expected.value_storage=value;
  EXPECT_EQ(view.capacity(),expected);EXPECT_EQ(root.capacity(),cap);
  EXPECT_EQ(view.mla_layout(),root.mla_layout());
  EXPECT_EQ(view.metadata_description().value(),root.metadata_description().value());
  const auto a=view.metadata_offsets(),b=root.metadata_offsets();
  EXPECT_EQ(a.query_indptr,b.query_indptr);EXPECT_EQ(a.kv_indptr,b.kv_indptr);
  EXPECT_EQ(a.page_indices,b.page_indices);EXPECT_EQ(a.last_page_lengths,b.last_page_lengths);EXPECT_EQ(a.bytes,b.bytes);
  EXPECT_TRUE(view.owns(layout));EXPECT_TRUE(layout.origin_schema().owns(layout));
  EXPECT_EQ(layout.origin_schema().capacity(),cap);
  auto another=view.with_cache_storage(c::CacheStorage::float8(2.F).value(),value).value();
  EXPECT_TRUE(another.owns(layout));EXPECT_EQ(layout.host().page_indices[0],input.pages[0]);
  EXPECT_EQ(c::detail::LeaseAccess::identity(layout.metadata().value()),identity);
  auto next=view.version(input.host()).value();auto last=another.version(input.host()).value();
  EXPECT_EQ(next.generation(),layout.generation()+1);EXPECT_EQ(last.generation(),next.generation()+1);
  EXPECT_TRUE(root.owns(next));EXPECT_TRUE(view.owns(last));EXPECT_TRUE(another.owns(next));
  EXPECT_EQ(next.origin_schema().capacity(),cap);EXPECT_EQ(last.origin_schema().capacity(),cap);
  auto foreign=c::BatchSchema::create(expected).value();
  EXPECT_FALSE(foreign.owns(layout));EXPECT_FALSE(foreign.owns(next));
  auto foreign_layout=foreign.version(input.host()).value();
  EXPECT_FALSE(root.owns(foreign_layout));EXPECT_FALSE(view.owns(foreign_layout));
}
TEST(BatchLayout, StorageViewAloneKeepsRootAllocatorAndGenerationDomainAlive){
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),1u<<20);
  c::BatchSchema view;std::uint64_t generation=0;Layout input;
  {
    auto root=c::BatchSchema::create(capacity(),{.allocator=quota.get()}).value();
    auto initial=root.version(input.host()).value();generation=initial.generation();
    view=root.with_cache_storage(c::CacheStorage::float8(.5F).value(),c::CacheStorage::float8(.25F).value()).value();
  }
  EXPECT_GT(quota.used(),0u);
  auto next=view.version(input.host()).value();EXPECT_EQ(next.generation(),generation+1);
  auto origin=next.origin_schema();EXPECT_TRUE(origin.owns(next));EXPECT_TRUE(view.owns(next));
  view={};EXPECT_EQ(next.host().page_indices[0],input.pages[0]);EXPECT_TRUE(next.metadata());
  next={};origin={};EXPECT_EQ(quota.used(),0u);
}
TEST(BatchLayout, StorageViewKeepsOriginalBudgetAfterExternalHandlesDropAndReaderEscapes){
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),1u<<20);
  c::ReadLease escaped;
  {
    const auto resource=hyle::mem::system_resource();const std::array limits{c::ResourceBudgetLimit{resource,1u<<20}};
    auto budget=c::ResourceBudget::create({.limits=limits,.allocator=quota.get()}).value();
    auto root=c::BatchSchema::create(capacity(),{.resource_budget=budget}).value();Layout input;
    auto initial=root.version(input.host()).value();const auto generation=initial.generation();
    {
      const auto used=budget.usage(resource).value();auto pressure=budget.reserve(resource,used.limit_bytes-used.live_bytes).value();
      auto failed=root.with_cache_storage(c::CacheStorage::float8(.5F).value(),c::CacheStorage::float8(.25F).value());
      ASSERT_FALSE(failed);EXPECT_TRUE(failed.error().IsResourceExhausted());EXPECT_TRUE(root.owns(initial));
    }
    auto view=root.with_cache_storage(c::CacheStorage::float8(.5F).value(),c::CacheStorage::float8(.25F).value()).value();
    root={};initial={};budget={};
    auto next=view.version(input.host()).value();EXPECT_EQ(next.generation(),generation+1);
    escaped=next.metadata().value();EXPECT_TRUE(escaped.valid());
    view={};next={};
  }
  EXPECT_GT(quota.used(),0u);escaped.reset();EXPECT_EQ(quota.used(),0u);
}
TEST(BatchLayout, StorageViewsPreserveRaggedAndMlaLayoutsAndRejectInvalidInterpretations){
  auto ragged=capacity();ragged.storage=c::BatchStorage::kRagged;ragged.max_pages=ragged.max_page_refs=0;ragged.page_size=1;
  auto root=c::BatchSchema::create(ragged).value();
  auto view=root.with_cache_storage(c::CacheStorage::float8(.5F).value(),c::CacheStorage::plain(c::DataType::kFloat32).value()).value();
  auto expected=ragged;expected.key_storage=c::CacheStorage::float8(.5F).value();expected.value_storage=c::CacheStorage::plain(c::DataType::kFloat32).value();
  EXPECT_EQ(view.capacity(),expected);EXPECT_EQ(view.mla_layout(),c::MlaCacheLayout::kNone);
  Layout input;const std::array<std::int32_t,3> kv{0,19,29};auto layout=view.version({input.q,kv,{},{}}).value();
  EXPECT_TRUE(root.owns(layout));EXPECT_EQ(layout.origin_schema().capacity(),ragged);
  for(const auto kind:{c::MlaCacheLayout::kCkv512,c::MlaCacheLayout::kCkv512Rope64}){
    c::MlaBatchCapacity mla;mla.layout=kind;auto original=c::BatchSchema::create_mla(mla).value();
    const auto storage=c::CacheStorage::float8(.125F).value();
    auto scaled=original.with_cache_storage(storage,storage).value();
    auto wanted=original.capacity();wanted.key_storage=wanted.value_storage=storage;
    EXPECT_EQ(scaled.capacity(),wanted);EXPECT_EQ(scaled.mla_layout(),kind);
    auto mismatch=scaled.with_cache_storage(storage,c::CacheStorage::float8(.25F).value());
    ASSERT_FALSE(mismatch);EXPECT_TRUE(mismatch.error().IsInvalidArgument());
    auto blocked=scaled.with_cache_storage(c::CacheStorage::q8_0().value(),c::CacheStorage::q8_0().value());
    ASSERT_FALSE(blocked);EXPECT_TRUE(blocked.error().IsUnimplemented());
    EXPECT_EQ(scaled.capacity(),wanted);EXPECT_EQ(original.capacity(),c::mla_batch_capacity(mla).value());
  }
  auto incompatible=capacity();incompatible.qk_head_dim=33;
  auto scalar=c::BatchSchema::create(incompatible).value();auto old=scalar.version(input.host()).value();
  const auto identity=c::detail::LeaseAccess::identity(old.metadata().value());
  auto invalid=scalar.with_cache_storage(c::CacheStorage::q8_0().value(),c::CacheStorage{});
  ASSERT_FALSE(invalid);EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_EQ(scalar.capacity(),incompatible);EXPECT_TRUE(scalar.owns(old));
  EXPECT_EQ(c::detail::LeaseAccess::identity(old.metadata().value()),identity);
  EXPECT_EQ(scalar.version(input.host())->generation(),old.generation()+1);
  EXPECT_TRUE(c::BatchSchema{}.with_cache_storage({},{}).error().IsFailedPrecondition());
  auto row=c::RowCacheBatchSchema::create({.encoding=c::RowCacheEncoding::float8_block64_tail64()}).value();
  EXPECT_TRUE(c::detail::BatchLayoutAccess::owner(row).with_cache_storage({},{}).error().IsInvalidArgument());
}
TEST(BatchLayout, StorageViewsUseOneConcurrentRootGenerationCounter){
  auto root=c::BatchSchema::create(capacity()).value();
  auto a=root.with_cache_storage(c::CacheStorage::float8(.5F).value(),c::CacheStorage::float8(.25F).value()).value();
  auto b=a.with_cache_storage(c::CacheStorage::float8(2.F).value(),c::CacheStorage::float8(4.F).value()).value();
  std::array<std::uint64_t,8> ids{};std::array<std::thread,8> threads;
  for(unsigned i=0;i<threads.size();++i)threads[i]=std::thread([&,i]{
    Layout input;auto version=(i%3==0?root:i%3==1?a:b).version(input.host()).value();
    EXPECT_TRUE(root.owns(version));EXPECT_TRUE(a.owns(version));EXPECT_TRUE(b.owns(version));ids[i]=version.generation();
  });
  for(auto&thread:threads)thread.join();std::sort(ids.begin(),ids.end());
  EXPECT_EQ(std::unique(ids.begin(),ids.end()),ids.end());EXPECT_EQ(ids.front(),1u);EXPECT_EQ(ids.back(),ids.size());
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
  auto input=capacity();input.query_type=c::DataType::kFloat64;EXPECT_FALSE(c::BatchSchema::create(input));input=capacity();input.query_heads=3;EXPECT_FALSE(c::BatchSchema::create(input));input=capacity();input.max_query_rows=UINT32_MAX;EXPECT_TRUE(c::BatchSchema::create(input).error().IsOutOfRange());
  input=capacity();input.kv_heads=input.query_heads=1u<<30;input.qk_head_dim=input.page_size=1u<<30;EXPECT_TRUE(c::BatchSchema::create(input).error().IsOutOfRange());input=capacity();input.max_pages=0;EXPECT_FALSE(c::BatchSchema::create(input));
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
  auto cap=capacity();auto schema=c::BatchSchema::create(cap).value();auto q=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{8,4,64}).value();auto kv=c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,4>{8,16,2,64}).value();
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
TEST(BatchLayout, FixedBlockRootTransferKeepsNativeAdmissionWithoutOwnerCycle){
  std::atomic<unsigned> released{0};struct Owner{std::atomic<unsigned>*released;};
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());c::ReadLease reader;
  {
    auto pool=c::LeasePool::create(hyle::mem::system_resource(),1024,{.max_blocks=1,.allocator=quota.get()}).value();
    auto writer=pool.allocate(256).value();ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
    c::detail::LeaseAccess::retain_resource_owner(pool,new Owner{&released},[](void*p)noexcept{auto*owner=static_cast<Owner*>(p);++*owner->released;delete owner;});
    c::detail::LeaseAccess::detach_pool_root(pool);EXPECT_FALSE(pool.valid());reader=writer.publish().value();
  }
  auto use_pool=c::LeaseUsePool::create().value();auto use=use_pool.begin().value();ASSERT_TRUE(use.read(reader).ok());
  reader.reset();EXPECT_EQ(released.load(),0u);use={};EXPECT_EQ(released.load(),1u);EXPECT_EQ(quota.used(),0u);
}
TEST(BatchLayout, WholeArenaSubviewCannotBecomeAnIndependentFreePageWriter){
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),4096,{.max_blocks=2}).value();auto writer=pool.allocate(1024).value();ASSERT_TRUE(writer.view()->fill(std::byte{7}).ok());auto arena=writer.publish().value();auto old_version=arena;auto free_page=arena.subview(512,512).value();
  EXPECT_TRUE(free_page.try_write().error().IsUnavailable());auto description=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,4>{2,2,2,64}).value();EXPECT_TRUE(arena.tensor(description));EXPECT_FALSE(free_page.tensor(description));
  std::array<std::byte,1> bytes;ASSERT_TRUE(old_version.view()->subview(0,1)->read(bytes).ok());EXPECT_EQ(bytes[0],std::byte{7});
}
TEST(BatchLayout, SeparatePageBlocksKeepReadUseAndFreePageWriteIndependent){
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),4096,{.max_blocks=4}).value();auto first=pool.allocate(512).value();auto second=pool.allocate(512).value();ASSERT_TRUE(first.view()->fill(std::byte{7}).ok());auto used=first.publish().value();auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();ASSERT_TRUE(use.read(used).ok());
  ASSERT_TRUE(second.view()->fill(std::byte{9}).ok());auto next=second.publish().value();EXPECT_EQ(c::detail::LeaseAccess::backing(used)->identity,c::detail::LeaseAccess::backing(next)->identity);
  // F1 independently tracks the pages; neither page permission can describe
  // the whole contiguous arena demanded by the current mathematical port.
  auto description=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,4>{2,2,2,64}).value();EXPECT_FALSE(used.tensor(description));EXPECT_FALSE(next.tensor(description));
  std::array<std::byte,1> bytes;ASSERT_TRUE(used.view()->subview(0,1)->read(bytes).ok());EXPECT_EQ(bytes[0],std::byte{7});ASSERT_TRUE(next.view()->subview(0,1)->read(bytes).ok());EXPECT_EQ(bytes[0],std::byte{9});
}

TEST(BatchLayout, EncodedKeysValuesAndOutputHaveIndependentTruthfulGeometry){
  auto cap=capacity();cap.qk_head_dim=64;cap.value_head_dim=128;
  cap.output_type=c::DataType::kBFloat16;
  cap.key_storage=cap.value_storage=c::CacheStorage::q8_0().value();
  auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  auto query=c::batch_query_description(cap);auto keys=c::batch_cache_description(cap,c::BatchCacheRole::kKey);
  auto values=c::batch_cache_description(cap,c::BatchCacheRole::kValue);
  auto output=c::batch_output_description(cap,cap.max_query_rows);
  ASSERT_TRUE(query);ASSERT_TRUE(keys);ASSERT_TRUE(values);ASSERT_TRUE(output);
  EXPECT_EQ(keys->block_axis(),3u);EXPECT_EQ(values->block_axis(),3u);
  EXPECT_EQ(keys->required_bytes().value(),17408u);EXPECT_EQ(values->required_bytes().value(),34816u);
  EXPECT_EQ(output->data_type(),c::DataType::kBFloat16);EXPECT_EQ(output->required_bytes().value(),8192u);
  std::array inputs{*query,*keys,*values,schema->metadata_description().value()};
  std::array<c::TensorDesc,1> actual;
  EXPECT_TRUE(c::PagedAttention{cap}.infer_output_descriptions(inputs,actual).ok());EXPECT_EQ(actual[0],*output);
  inputs[2]=*keys;EXPECT_FALSE(c::PagedAttention{cap}.infer_output_descriptions(inputs,actual).ok());
  EXPECT_TRUE(c::batch_output_description(cap,cap.max_query_rows+1).error().IsOutOfRange());
}

TEST(BatchLayout, AttentionConcreteQueryExtentIsBoundedByTheUnchangedSchemaCapacity){
  for(const auto storage:{c::BatchStorage::kPaged,c::BatchStorage::kRagged}){
    auto cap=capacity();cap.storage=storage;cap.max_query_rows=128;
    cap.value_head_dim=128;cap.output_type=c::DataType::kBFloat16;
    if(storage==c::BatchStorage::kRagged){cap.max_pages=cap.max_page_refs=0;cap.page_size=1;}
    auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
    const auto metadata=schema->metadata_description().value();
    const auto keys=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value();
    const auto values=c::batch_cache_description(cap,c::BatchCacheRole::kValue).value();
    const auto operation=storage==c::BatchStorage::kPaged?
        c::Operation::custom(c::PagedAttention{cap}):c::Operation::custom(c::RaggedAttention{cap});
    EXPECT_EQ(c::batch_query_description(cap).value(),c::batch_query_description(cap,128).value());
    for(const std::uint32_t rows:{1u,128u}){
      const auto query=c::batch_query_description(cap,rows).value();
      const auto expected=c::batch_output_description(cap,rows).value();
      std::array inputs{query,keys,values,metadata};std::array<c::TensorDesc,1> output;
      ASSERT_TRUE(c::infer_output_descriptions(operation,inputs,output).ok());
      EXPECT_EQ(output[0],expected);
      EXPECT_EQ(query.required_bytes().value(),std::size_t(rows)*cap.query_heads*cap.qk_head_dim*2);
      EXPECT_EQ(output[0].required_bytes().value(),std::size_t(rows)*cap.query_heads*cap.value_head_dim*2);
      EXPECT_EQ(schema->capacity(),cap);EXPECT_EQ(inputs[3],metadata);
      auto bad=inputs;
      bad[0]=c::TensorDesc::dense(c::DataType::kBFloat16,
          std::array<std::int64_t,3>{rows,cap.query_heads,cap.qk_head_dim}).value();
      EXPECT_FALSE(c::infer_output_descriptions(operation,bad,output).ok());
      bad=inputs;bad[0]=c::TensorDesc::dense(cap.query_type,
          std::array<std::int64_t,3>{rows,cap.query_heads+1,cap.qk_head_dim}).value();
      EXPECT_FALSE(c::infer_output_descriptions(operation,bad,output).ok());
      bad=inputs;bad[2]=keys;
      EXPECT_FALSE(c::infer_output_descriptions(operation,bad,output).ok());
      bad=inputs;bad[3]=c::TensorDesc::dense(c::DataType::kUInt8,
          std::array<std::int64_t,1>{std::int64_t(metadata.required_bytes().value()+4)}).value();
      EXPECT_FALSE(c::infer_output_descriptions(operation,bad,output).ok());
    }
    EXPECT_TRUE(c::batch_query_description(cap,129).error().IsOutOfRange());
    const auto oversized=c::TensorDesc::dense(cap.query_type,
        std::array<std::int64_t,3>{129,cap.query_heads,cap.qk_head_dim}).value();
    const std::array inputs{oversized,keys,values,metadata};std::array<c::TensorDesc,1> output;
    EXPECT_TRUE(c::infer_output_descriptions(operation,inputs,output).IsOutOfRange());
    const auto strided=c::TensorDesc::strided_bytes(cap.query_type,
        std::array<std::int64_t,3>{1,cap.query_heads,cap.qk_head_dim},
        std::array<std::int64_t,3>{2048,256,2}).value();
    EXPECT_FALSE(c::infer_output_descriptions(operation,std::array{strided,keys,values,metadata},output).ok());
  }
}

TEST(BatchLayout, OrdinaryQueryExtentAdmissionDoesNotChangeTheSeparateMlaContract){
  for(const auto storage:{c::BatchStorage::kPaged,c::BatchStorage::kRagged}){
    c::MlaBatchCapacity mla;mla.storage=storage;mla.max_query_rows=128;mla.max_kv_rows=128;
    if(storage==c::BatchStorage::kRagged){mla.max_pages=mla.max_page_refs=0;mla.page_size=1;}
    const auto cap=c::mla_batch_capacity(mla).value();
    const auto cache=c::batch_cache_description(cap,c::BatchCacheRole::kKey).value();
    const auto metadata=c::batch_metadata_description(cap).value();
    const auto operation=storage==c::BatchStorage::kPaged?
        c::Operation::custom(c::PagedMlaAttention{mla}):c::Operation::custom(c::RaggedMlaAttention{mla});
    std::array<c::TensorDesc,1> output;
    EXPECT_TRUE(c::infer_output_descriptions(operation,
        std::array{c::batch_query_description(cap).value(),cache,metadata},output).ok());
    EXPECT_FALSE(c::infer_output_descriptions(operation,
        std::array{c::batch_query_description(cap,1).value(),cache,metadata},output).ok());
  }
}

TEST(BatchLayout, StructuralNoncausalQueryLengthDoesNotMintCausalQualification){
  auto cap=capacity();cap.max_query_rows=2;cap.max_kv_rows=1;
  auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  const std::array<std::int32_t,2> q{0,2},k{0,1};const std::array<std::int32_t,1> ids{3},tail{1};
  auto layout=schema->version({q,k,ids,tail});ASSERT_TRUE(layout);
  EXPECT_TRUE(c::validate_batch_attention_layout(*layout,false).ok());
  EXPECT_TRUE(c::validate_batch_attention_layout(*layout,true).IsInvalidArgument());
  EXPECT_TRUE(c::validate_batch_attention_options(false,0).IsInvalidArgument());
  EXPECT_TRUE(c::validate_batch_attention_options(true,-2).IsInvalidArgument());
  EXPECT_TRUE(c::validate_batch_attention_options(true,0).ok());
  cap.storage=c::BatchStorage::kRagged;cap.max_pages=cap.max_page_refs=0;cap.page_size=1;
  schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  layout=schema->version({q,k,{},{}});ASSERT_TRUE(layout);
  EXPECT_TRUE(c::validate_batch_attention_layout(*layout,false).ok());
  EXPECT_FALSE(c::validate_batch_attention_layout(*layout,true).ok());
}

TEST(BatchLayout, CacheBlockAndValueDepthOverflowRefuseBeforeVersionCreation){
  auto cap=capacity();cap.key_storage=c::CacheStorage::q8_0().value();cap.qk_head_dim=33;
  EXPECT_TRUE(c::BatchSchema::create(cap).error().IsInvalidArgument());
  cap=capacity();cap.value_head_dim=UINT32_MAX;
  EXPECT_TRUE(c::BatchSchema::create(cap).error().IsOutOfRange());
  cap=capacity();cap.value_head_dim=0;EXPECT_FALSE(c::BatchSchema::create(cap));
}
} // namespace
