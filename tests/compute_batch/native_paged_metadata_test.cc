#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <thread>
#include "hyle/compute/detail/lease_access.h"

#include "allocation_fault_fixture.h"
#include "hyle/compute/batch_paged_native.h"

namespace {
namespace c = hyle::compute;
c::BatchSchema schema(c::DataType type = c::DataType::kFloat16) {
  return c::BatchSchema::create({.storage = c::BatchStorage::kPaged, .query_type=type,.output_type=type,.key_storage=c::CacheStorage::plain(type).value(),.value_storage=c::CacheStorage::plain(type).value(),
      .max_requests = 3, .max_query_rows = 8, .max_kv_rows = 96, .max_pages = 4,
      .max_page_refs = 8, .query_heads = 8, .kv_heads = 2, .qk_head_dim=64,.value_head_dim=64, .page_size = 16}).value();
}
c::BatchLayoutVersion layout(const c::BatchSchema &schema) {
  const std::array<std::int32_t, 3> q{0, 1, 4}, k{0, 2, 4};
  const std::array<std::int32_t, 4> pages{3, 1, 1, 0};
  const std::array<std::int32_t, 2> tails{7, 15};
  return schema.version({q, k, pages, tails}).value();
}
TEST(NativePagedMetadata, FullNoncausalQGreaterThanKvAndIndependentEncodedRoots){
  c::BatchCapacity cap{.max_query_rows=2,.max_kv_rows=1,.query_heads=2,.kv_heads=1,
      .qk_head_dim=64,.value_head_dim=128};
  cap.key_storage=cap.value_storage=c::CacheStorage::q8_0().value();
  auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  const std::array<std::int32_t,2> q{0,2},k{0,1};const std::array<std::int32_t,1> ids{0},tail{1};
  auto selected=schema->version({q,k,ids,tail});ASSERT_TRUE(selected);
  auto full=c::NativePagedBatchProgram::prepare(*schema,{.max_host_metadata_bytes=4,.causal=false});ASSERT_TRUE(full);
  EXPECT_TRUE(full->version(*selected));
  EXPECT_NE(full->input_description(*selected,1).value(),full->input_description(*selected,2).value());
  auto causal=c::NativePagedBatchProgram::prepare(*schema,{.max_host_metadata_bytes=4});ASSERT_TRUE(causal);
  EXPECT_TRUE(causal->version(*selected).error().IsInvalidArgument());
  EXPECT_FALSE(c::NativePagedBatchProgram::prepare(*schema,{.max_host_metadata_bytes=4,.causal=false,.window_left=0}));
}
TEST(NativePagedMetadata, InputDescriptionsUseActiveQueriesWithoutAllocatingOrShrinkingCache){
  for(bool mla:{false,true}){
    auto make_schema=[&]{return mla?c::BatchSchema::create_mla({.max_requests=3,
        .max_query_rows=8,.max_kv_rows=96,.max_pages=4,.max_page_refs=8,
        .query_heads=64}).value():schema();};
    auto selected=make_schema(),foreign=make_schema();auto owned=layout(selected);
    const std::array<std::int32_t,2> empty_ptr{0,0};
    const std::array<std::int32_t,1> empty_tail{0};
    auto empty=selected.version({empty_ptr,empty_ptr,{},empty_tail}).value();
    auto other=layout(foreign);hyle_f5_fault_fixture::Allocator allocator;
    {
      auto recipe=c::NativePagedBatchProgram::prepare(selected,
          {.max_host_metadata_bytes=32,.allocator=allocator.ref()}).value();
      const auto attempts=allocator.attempts;allocator.fail=attempts;
      EXPECT_EQ(recipe.input_count(),mla?3u:4u);
      auto query=recipe.input_description(owned,0);ASSERT_TRUE(query);
      EXPECT_EQ(query->dimensions()[0],4);
      EXPECT_EQ(query->dimensions()[1],selected.capacity().query_heads);
      EXPECT_EQ(query->dimensions()[2],selected.capacity().qk_head_dim);
      auto zero=recipe.input_description(empty,0);ASSERT_TRUE(zero);
      EXPECT_EQ(zero->dimensions()[0],0);EXPECT_EQ(zero->required_bytes().value(),0u);
      auto cache=recipe.input_description(owned,1);ASSERT_TRUE(cache);
      EXPECT_EQ(*cache,c::batch_cache_description(selected,
          mla?c::BatchCacheRole::kMla:c::BatchCacheRole::kKey).value());
      for(std::size_t port=1;port<recipe.input_count();++port){
        EXPECT_EQ(recipe.input_description(empty,port).value(),
                  recipe.input_description(owned,port).value());
      }
      EXPECT_TRUE(recipe.input_description(other,0).error().IsInvalidArgument());
      EXPECT_TRUE(recipe.input_description({},0).error().IsInvalidArgument());
      EXPECT_TRUE(recipe.input_description(owned,recipe.input_count()).error().IsInvalidArgument());
      EXPECT_EQ(allocator.attempts,attempts);
    }
    EXPECT_EQ(allocator.live,0u);
    EXPECT_EQ(c::NativePagedBatchProgram{}.input_count(),0u);
    EXPECT_TRUE(c::NativePagedBatchProgram{}.input_description(owned,0).error().IsFailedPrecondition());
  }
}
TEST(NativePagedMetadata, CompactOwnedPageOrderAndTailGenerationAreIndependentOfQueryHeads) {
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    auto selected = schema(type);
    auto recipe = c::NativePagedBatchProgram::prepare(selected,
        {.max_host_metadata_bytes = 32, .scale = .125F, .causal = false}).value();
    EXPECT_EQ(recipe.input_count(), 4u);
    EXPECT_EQ(recipe.costs().index_bytes, 32u);
    EXPECT_TRUE(recipe.costs().native_private_bytes_unknown);
    EXPECT_EQ(recipe.scale(), .125F); EXPECT_FALSE(recipe.causal());
    std::array<std::int32_t, 3> q{0, 1, 4}, k{0, 2, 4};
    std::array<std::int32_t, 4> pages{3, 1, 1, 0};
    std::array<std::int32_t, 2> tails{7, 15};
    auto owned = selected.version({q, k, pages, tails}).value();
    const auto metadata=recipe.input_description(owned,3);ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->data_type(), c::DataType::kInt32);
    EXPECT_EQ(metadata->required_bytes().value(), 32u);
    auto version = recipe.version(owned).value();
    pages.fill(2); tails.fill(16); q[1] = 2;
    EXPECT_EQ(version.layout(), owned);
    EXPECT_TRUE(version.program().schema().owns(version.layout()));
    const auto ids = version.indices().value();
    std::array<std::int32_t, 8> actual;
    ASSERT_TRUE(ids.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, (std::array<std::int32_t, 8>{3,1,1,0,0,0,0,0}));
    EXPECT_EQ(version.layout().host().last_page_lengths[0], 7);
    auto next = recipe.version(layout(selected)).value();
    EXPECT_NE(next.layout().generation(), version.layout().generation());
    std::array<std::int32_t, 8> repeated;
    ASSERT_TRUE(next.indices()->view()->read(std::as_writable_bytes(std::span(repeated))).ok());
    EXPECT_EQ(repeated, actual);
  }
  auto capacity = schema().capacity();
  capacity.max_query_rows = 8192; capacity.query_heads = 16; capacity.kv_heads = 4;
  capacity.qk_head_dim = 128;
  auto wider = c::BatchSchema::create(capacity).value();
  auto compact = c::NativePagedBatchProgram::prepare(wider, {.max_host_metadata_bytes = 32}).value();
  EXPECT_EQ(compact.costs().index_bytes, 32u); // Query/head capacity does not expand metadata.
  const std::array<std::int32_t, 2> q{0, 2}, k{0, 6};
  const std::array<std::int32_t, 6> repeated_pages{3,1,3,1,3,1};
  const std::array<std::int32_t, 1> tail{15};
  auto repeated_layout = wider.version({q,k,repeated_pages,tail}).value();
  EXPECT_GT(repeated_layout.facts().kv_rows, capacity.max_pages * capacity.page_size);
  auto repeated_version = compact.version(repeated_layout).value();
  std::array<std::int32_t, 8> actual;
  ASSERT_TRUE(repeated_version.indices()->view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<std::int32_t, 8>{3,1,3,1,3,1,0,0}));
}
TEST(NativePagedMetadata, ExportedSnapshotRetainsOwnersAndEmptyVersionHasNoPageReads) {
  c::ReadLease exported;
  {
    auto selected = schema();
    auto recipe = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 32}).value();
    auto version = recipe.version(layout(selected)).value();
    exported = version.indices().value();
    auto moved = std::move(version);
    EXPECT_FALSE(version.valid()); EXPECT_TRUE(moved.valid());
    auto copied = moved;
    EXPECT_EQ(copied.layout(), moved.layout());
  }
  std::array<std::int32_t, 8> actual;
  ASSERT_TRUE(exported.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<std::int32_t, 8>{3,1,1,0,0,0,0,0}));
  auto selected = schema();
  auto recipe = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 32}).value();
  const std::array<std::int32_t, 1> empty{0};
  auto empty_layout = selected.version({empty, empty, {}, {}}).value();
  auto empty_version = recipe.version(empty_layout).value();
  ASSERT_TRUE(empty_version.indices()->view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<std::int32_t, 8>{}));
  EXPECT_EQ(empty_version.layout().facts().query_rows, 0u);
  EXPECT_FALSE(c::NativePagedBatchVersion{}.indices());
}
TEST(NativePagedMetadata, ForeignSchemaBudgetAndNonfiniteScaleRejectBeforePublishing) {
  auto selected = schema(), foreign = schema();
  auto recipe = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 32}).value();
  auto rejected = recipe.version(layout(foreign));
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument());
  auto small = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 31});
  ASSERT_FALSE(small); EXPECT_TRUE(small.error().IsResourceExhausted());
  auto invalid = c::NativePagedBatchProgram::prepare(selected,
      {.max_host_metadata_bytes = 32, .scale = std::numeric_limits<float>::infinity()});
  ASSERT_FALSE(invalid); EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_FALSE(c::NativePagedBatchProgram{}.version(layout(selected)));
}
TEST(NativePagedMetadata, EveryColdAllocationFailureReclaimsMetadataAndExportCanOutliveRecipe) {
  auto selected = schema(); auto owned = layout(selected);
  hyle_f5_fault_fixture::Allocator measured;
  {
    auto recipe = c::NativePagedBatchProgram::prepare(selected,
        {.max_host_metadata_bytes = 32, .allocator = measured.ref()}).value();
    auto version = recipe.version(owned).value();
    auto exported = version.indices().value();
    version = {}; recipe = {};
    EXPECT_GT(measured.live, 0u);
    exported.reset();
  }
  EXPECT_EQ(measured.live, 0u);
  const auto count = measured.attempts;
  ASSERT_GT(count, 0u);
  for (std::size_t fail = 0; fail < count; ++fail) {
    hyle_f5_fault_fixture::Allocator fault; fault.fail = fail;
    {
      auto recipe = c::NativePagedBatchProgram::prepare(selected,
          {.max_host_metadata_bytes = 32, .allocator = fault.ref()});
      if (!recipe) EXPECT_TRUE(recipe.error().IsResourceExhausted()) << fail;
      else {
        auto version = recipe->version(owned);
        if(!version)EXPECT_TRUE(version.error().IsResourceExhausted()) << fail;
        else{auto exported=version->indices();ASSERT_FALSE(exported)<<fail;EXPECT_TRUE(exported.error().IsResourceExhausted())<<fail;}
      }
    }
    EXPECT_EQ(fault.live, 0u) << fail;
  }
}
TEST(NativePagedMetadata, ActiveVersionDefersFullIndexAllocationUntilExplicitExport){
  auto cap=schema().capacity();cap.max_page_refs=8192;auto selected=c::BatchSchema::create(cap).value();
  auto owned=layout(selected);hyle_f5_fault_fixture::Allocator allocator;
  auto recipe=c::NativePagedBatchProgram::prepare(selected,{.max_host_metadata_bytes=32768,.allocator=allocator.ref()}).value();
  const auto before=allocator.attempts;auto version=recipe.version(owned).value();EXPECT_EQ(allocator.attempts,before+1);
  EXPECT_EQ(version.layout(),owned);const auto active_attempts=allocator.attempts;allocator.fail=active_attempts;
  auto failed=version.indices();ASSERT_FALSE(failed);EXPECT_TRUE(failed.error().IsResourceExhausted());EXPECT_EQ(version.layout(),owned);
  allocator.fail=(std::numeric_limits<std::size_t>::max)();auto exported=version.indices();ASSERT_TRUE(exported);EXPECT_EQ(exported->size(),32768u);
  const auto successful=allocator.attempts;auto repeated=version.indices();ASSERT_TRUE(repeated);EXPECT_EQ(allocator.attempts,successful);
  EXPECT_EQ(c::detail::LeaseAccess::identity(*exported),c::detail::LeaseAccess::identity(*repeated));
}
TEST(NativePagedMetadata, ConcurrentExplicitExportsShareOneReadonlyTable){
  auto selected=schema();auto recipe=c::NativePagedBatchProgram::prepare(selected,{.max_host_metadata_bytes=32}).value();
  auto version=recipe.version(layout(selected)).value();std::array<c::ReadLease,8> copies;std::array<std::thread,8> threads;
  for(unsigned i=0;i<threads.size();++i)threads[i]=std::thread([&,i]{auto exported=version.indices();EXPECT_TRUE(exported);if(exported)copies[i]=std::move(*exported);});
  for(auto&thread:threads)thread.join();
  for(const auto&copy:copies){ASSERT_TRUE(copy.valid());EXPECT_EQ(c::detail::LeaseAccess::identity(copy),c::detail::LeaseAccess::identity(copies[0]));}
}
} // namespace
