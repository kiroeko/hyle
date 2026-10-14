#include <hyle/compute/batch_materialize.h>
#include <hyle/mem/quota_allocator.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

namespace {
namespace c=hyle::compute;
c::BatchCapacity capacity(){return {.max_requests=2,.max_query_rows=4,.max_kv_rows=29,.max_pages=4,.max_page_refs=3,.query_heads=4,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=16};}
c::MaterializedBatchOptions options(){return {.max_kv_per_request=19,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=8u<<20,.scale=.125f};}
struct Host{std::array<std::int32_t,3> q{0,1,3},kv{0,2,3},pages{2,0,3};std::array<std::int32_t,2> tails{3,10};c::HostBatchLayout value(){return {q,kv,pages,tails};}};
template<class T>std::vector<T> read(const c::ReadLease&lease){std::vector<T> values(lease.size()/sizeof(T));EXPECT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(values))).ok());return values;}
TEST(BatchMaterialize, PhysicalPagesGqaAndRightAlignedCausalMaskExcludePadding){
  auto schema=c::BatchSchema::create(capacity()).value();Host h;auto layout=schema.version(h.value()).value();auto program=c::MaterializedBatchProgram::prepare(schema,options()).value();auto version=program.version(layout).value();
  const auto ids=read<std::int32_t>(version.indices().value());const auto mask=read<float>(version.mask().value());
  // Request zero reads physical page 2 then page 0, never arena order.
  EXPECT_EQ(ids[0],64);EXPECT_EQ(ids[15],94);EXPECT_EQ(ids[16],0);EXPECT_EQ(ids[18],4);
  EXPECT_EQ(ids[19],64);EXPECT_EQ(ids[38],65);EXPECT_EQ(ids[57],65);
  // Request one has two Q rows: first sees 9 KV rows, second sees all 10.
  EXPECT_EQ(ids[4*19],96);EXPECT_EQ(ids[4*19+9],114);EXPECT_EQ(ids[4*19+10],96);
  EXPECT_EQ(mask[19+8],0);EXPECT_TRUE(std::isinf(mask[19+9])&&mask[19+9]<0);EXPECT_EQ(mask[38+9],0);EXPECT_TRUE(std::isinf(mask[38+10]));
  EXPECT_EQ(mask[3*19],0);EXPECT_TRUE(std::isinf(mask[3*19+1]));EXPECT_EQ(version.layout(),layout);
  EXPECT_EQ(program.descriptions()[program.output_value()],program.descriptions()[0]);EXPECT_GT(program.costs().logical_intermediate_bytes,program.costs().gathered_kv_bytes);EXPECT_TRUE(program.costs().native_private_bytes_unknown);
}
TEST(BatchMaterialize, NoncausalUsesEveryValidKvButNeverTailPadding){
  auto schema=c::BatchSchema::create(capacity()).value();Host h;auto o=options();o.causal=false;auto program=c::MaterializedBatchProgram::prepare(schema,o).value();auto version=program.version(schema.version(h.value()).value()).value();auto mask=read<float>(version.mask().value());
  EXPECT_EQ(mask[19+9],0);EXPECT_TRUE(std::isinf(mask[19+10]));EXPECT_EQ(mask[18],0);
}
TEST(BatchMaterialize, CausalWindowUsesInclusivePastCountAtBottomRightPosition){
  auto schema=c::BatchSchema::create(capacity()).value();Host h;
  for(std::int32_t window:{0,15,16}){
    auto o=options();o.window_left=window;
    auto program=c::MaterializedBatchProgram::prepare(schema,o);ASSERT_TRUE(program);
    auto version=program->version(schema.version(h.value()).value());ASSERT_TRUE(version);
    const auto mask=read<float>(version->mask().value());
    const std::array<unsigned,3> positions{18,8,9};
    for(unsigned row=0;row<positions.size();++row){
      const auto first=positions[row]>unsigned(window)?positions[row]-unsigned(window):0;
      for(unsigned j=0;j<19;++j){
        if(j>=first && j<=positions[row])EXPECT_EQ(mask[row*19+j],0);
        else EXPECT_TRUE(std::isinf(mask[row*19+j]) && mask[row*19+j]<0);
      }
    }
  }
}
TEST(BatchMaterialize, FullNoncausalCrossAttentionUsesValueDepthAndOutputType){
  auto cap=capacity();cap.max_query_rows=2;cap.max_kv_rows=1;cap.value_head_dim=128;cap.output_type=c::DataType::kBFloat16;
  auto schema=c::BatchSchema::create(cap).value();
  const std::array<std::int32_t,2> q{0,2},k{0,1};const std::array<std::int32_t,1> ids{0},tail{1};
  auto layout=schema.version({q,k,ids,tail});ASSERT_TRUE(layout);
  auto o=options();o.max_kv_per_request=1;o.causal=false;
  auto program=c::MaterializedBatchProgram::prepare(schema,o);ASSERT_TRUE(program);
  auto version=program->version(*layout);ASSERT_TRUE(version);
  const auto mask=read<float>(version->mask().value());ASSERT_EQ(mask.size(),2u);EXPECT_EQ(mask[0],0);EXPECT_EQ(mask[1],0);
  const auto&output=program->descriptions()[program->output_value()];
  EXPECT_EQ(output.data_type(),c::DataType::kBFloat16);EXPECT_EQ(output.dimensions()[2],128);
  o.causal=true;auto causal=c::MaterializedBatchProgram::prepare(schema,o);ASSERT_TRUE(causal);
  EXPECT_TRUE(causal->version(*layout).error().IsInvalidArgument());
}
TEST(BatchMaterialize, EncodedCacheRequiresNativeRouteInsteadOfHistoryDecode){
  for(auto storage:{c::CacheStorage::q8_0().value(),c::CacheStorage::float8(.25F).value()}){
    auto cap=capacity();cap.key_storage=cap.value_storage=storage;
    auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
    EXPECT_TRUE(c::MaterializedBatchProgram::prepare(*schema,options()).error().IsUnimplemented());
  }
}
TEST(BatchMaterialize, InactiveQueryCapacityReadsOnlyPagesOfTheFirstActiveQuery){
  auto schema=c::BatchSchema::create(capacity()).value();Host h;
  h.q={0,0,3};h.pages={2,2,3};
  auto program=c::MaterializedBatchProgram::prepare(schema,options()).value();
  auto version=program.version(schema.version(h.value()).value()).value();
  const auto ids=read<std::int32_t>(version.indices().value());
  const auto cap=capacity();const auto bound=options().max_kv_per_request;
  // Request zero has no Q; its page 2 does not choose inactive-row padding.
  // The actual first query reads page 3. No generated gather may read page 0.
  for(std::size_t head=0;head<cap.query_heads;++head){
    const auto begin=(std::size_t(3)*cap.query_heads+head)*bound;
    for(std::size_t j=0;j<bound;++j)EXPECT_EQ(ids[begin+j],96+std::int32_t(head/2));
  }
  for(auto id:ids){const auto page=id/std::int32_t(cap.kv_heads*cap.page_size);EXPECT_TRUE(page==2 || page==3);}
  auto ragged=cap;ragged.storage=c::BatchStorage::kRagged;ragged.max_pages=ragged.max_page_refs=0;ragged.page_size=1;
  auto ragged_schema=c::BatchSchema::create(ragged).value();h.kv={0,19,29};
  auto ragged_program=c::MaterializedBatchProgram::prepare(ragged_schema,options()).value();
  auto ragged_version=ragged_program.version(ragged_schema.version({h.q,h.kv,{},{}}).value()).value();
  const auto ragged_ids=read<std::int32_t>(ragged_version.indices().value());
  for(std::size_t head=0;head<cap.query_heads;++head)
    for(std::size_t j=0;j<bound;++j)EXPECT_EQ(ragged_ids[(3*cap.query_heads+head)*bound+j],38+std::int32_t(head/2));
}
TEST(BatchMaterialize, RaggedOffsetsAndNewLayoutDoNotChangeOldOwnedIndices){
  auto cap=capacity();cap.storage=c::BatchStorage::kRagged;cap.max_pages=cap.max_page_refs=0;cap.page_size=1;auto schema=c::BatchSchema::create(cap).value();Host h;h.kv={0,19,29};auto program=c::MaterializedBatchProgram::prepare(schema,options()).value();
  auto layout=schema.version({h.q,h.kv,{},{}}).value();auto old=program.version(layout).value();auto old_indices=old.indices().value();h.q[1]=2;auto next=program.version(schema.version({h.q,h.kv,{},{}}).value()).value();
  EXPECT_EQ(read<std::int32_t>(old_indices)[4*19],38);EXPECT_EQ(read<std::int32_t>(next.indices().value())[4*19],0);EXPECT_NE(next.layout(),old.layout());
  old={};program={};schema={};EXPECT_EQ(read<std::int32_t>(old_indices)[4*19],38);EXPECT_FALSE(old_indices.try_write());
}
TEST(BatchMaterialize, ForeignEmptyAndOverlongLayoutsAreRejectedBeforeGeneration){
  auto schema=c::BatchSchema::create(capacity()).value();auto foreign=c::BatchSchema::create(capacity()).value();Host h;auto program=c::MaterializedBatchProgram::prepare(schema,options()).value();EXPECT_TRUE(program.version(foreign.version(h.value()).value()).error().IsInvalidArgument());
  std::array<std::int32_t,1> zero{0};EXPECT_TRUE(program.version(schema.version({zero,zero,{},{}}).value()).error().IsUnimplemented());
  auto o=options();o.max_kv_per_request=18;auto small=c::MaterializedBatchProgram::prepare(schema,o).value();EXPECT_TRUE(small.version(schema.version(h.value()).value()).error().IsOutOfRange());
}
TEST(BatchMaterialize, ExplicitHostAndLogicalByteCapsAreCheckedAtColdPreparation){
  auto schema=c::BatchSchema::create(capacity()).value();auto o=options();auto full=c::MaterializedBatchProgram::prepare(schema,o).value();auto costs=full.costs();o.max_host_metadata_bytes=costs.index_bytes+costs.mask_bytes-1;EXPECT_TRUE(c::MaterializedBatchProgram::prepare(schema,o).error().IsResourceExhausted());
  o=options();o.max_logical_intermediate_bytes=costs.logical_intermediate_bytes-1;EXPECT_TRUE(c::MaterializedBatchProgram::prepare(schema,o).error().IsResourceExhausted());o.max_logical_intermediate_bytes=costs.logical_intermediate_bytes;EXPECT_TRUE(c::MaterializedBatchProgram::prepare(schema,o));o=options();o.max_kv_per_request=0;EXPECT_FALSE(c::MaterializedBatchProgram::prepare(schema,o));
}
TEST(BatchMaterialize, FlattenedI32AndProductOverflowRejectWithoutPayloadAllocation){
  auto cap=capacity();cap.max_pages=INT32_MAX;auto schema=c::BatchSchema::create(cap).value();EXPECT_TRUE(c::MaterializedBatchProgram::prepare(schema,options()).error().IsOutOfRange());
  cap=capacity();cap.max_query_rows=INT32_MAX;cap.query_heads=INT32_MAX-1;cap.kv_heads=1;cap.qk_head_dim=cap.value_head_dim=1;cap.max_kv_rows=INT32_MAX;schema=c::BatchSchema::create(cap).value();auto o=options();o.max_kv_per_request=INT32_MAX;o.max_host_metadata_bytes=SIZE_MAX;o.max_logical_intermediate_bytes=SIZE_MAX;EXPECT_TRUE(c::MaterializedBatchProgram::prepare(schema,o).error().IsOutOfRange());
}
TEST(BatchMaterialize, AllocatorOomPreservesOldVersionAndExportRetiresAfterControls){
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());c::ReadLease exported;
  {auto schema=c::BatchSchema::create(capacity()).value();Host h;auto o=options();o.allocator=quota.get();auto program=c::MaterializedBatchProgram::prepare(schema,o).value();auto layout=schema.version(h.value()).value();auto old=program.version(layout).value();exported=old.indices().value();
    auto reserved=quota.try_reserve_bytes(quota.remaining());ASSERT_TRUE(reserved);EXPECT_TRUE(program.version(layout).error().IsResourceExhausted());EXPECT_EQ(read<std::int32_t>(exported)[0],64);
  }EXPECT_GT(quota.used(),0u);auto uses=c::LeaseUsePool::create().value();auto use=uses.begin().value();EXPECT_TRUE(use.read(exported).ok());exported.reset();EXPECT_GT(quota.used(),0u);use={};EXPECT_EQ(quota.used(),0u);
}
} // namespace
