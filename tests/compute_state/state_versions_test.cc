#include <gtest/gtest.h>
#include "hyle/compute/state_versions.h"
#include "hyle/mem/quota_allocator.h"
#include <array>
#include <thread>

namespace {
using namespace hyle::compute;
namespace m = hyle::mem;
StateEntrySpec resource_spec() {
  return {.description=TensorDesc::dense(DataType::kFloat32,std::array<std::int64_t,1>{4}).value()};
}
ReadLease ready(std::byte value) {
  auto buffer=m::Buffer::create(m::system_resource(),16,64).value();
  buffer.view().fill(value).IgnoreError();return ReadLease::adopt(std::move(buffer)).value();
}
std::array<std::byte,16> bytes(const ReadLease &lease) {
  std::array<std::byte,16> output;lease.view()->read(output).IgnoreError();return output;
}
TEST(StateVersions, SchemaIdentityAndCompleteImportAreCheckedBeforeConsumption) {
  const std::array specs{resource_spec(),resource_spec()};
  auto schema=StateSchema::create(specs).value(),foreign=StateSchema::create(specs).value();
  auto store=StateStore::create(schema).value();auto source=ready(std::byte{17});
  const auto a=schema.entry(0).value(),b=schema.entry(1).value();
  EXPECT_FALSE(store.import(std::array{StateMember{a,source}}));
  EXPECT_FALSE(store.import(std::array{StateMember{a,source},StateMember{a,source}}));
  EXPECT_FALSE(store.import(std::array{StateMember{a,source},StateMember{foreign.entry(1).value(),source}}));
  EXPECT_TRUE(source.valid());EXPECT_EQ(store.stats().versions,0u);
  auto version=store.import(std::array{StateMember{a,source},StateMember{b,source}}).value();
  EXPECT_EQ(store.stats().retained_blocks,1u);EXPECT_EQ(store.stats().backing_bytes,16u);
  EXPECT_EQ(bytes(version.member(a).value())[0],std::byte{17});
  EXPECT_FALSE(version.member(foreign.entry(0).value()));
}
TEST(StateVersions, SharedBranchReplacementAndExpectedBaseKeepOldVersion) {
  auto schema=StateSchema::create(std::array{resource_spec()}).value();auto id=schema.entry(0).value();
  auto store=StateStore::create(schema).value();
  auto base=store.import(std::array{StateMember{id,ready(std::byte{11})}}).value();
  auto head=StateHead::create(store,base).value();
  auto first=store.begin(base).value(),second=store.begin(base).value();
  ASSERT_TRUE(first.replace(id,ready(std::byte{21})).ok());
  ASSERT_TRUE(second.replace(id,ready(std::byte{31})).ok());
  auto one=first.finalize().value(),two=second.finalize().value();
  EXPECT_TRUE(head.publish(base.identity(),one).ok());
  EXPECT_TRUE(head.publish(base.identity(),two).IsUnavailable());
  EXPECT_EQ(bytes(head.current()->member(id).value())[0],std::byte{21});
  EXPECT_EQ(bytes(base.member(id).value())[0],std::byte{11});
  EXPECT_EQ(bytes(two.member(id).value())[0],std::byte{31});
}
TEST(StateVersions, MetadataSnapshotIsOwnedAndSeparateFromResource) {
  const std::array specs{resource_spec(),StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16}};
  auto schema=StateSchema::create(specs).value();auto store=StateStore::create(schema).value();
  auto id=schema.entry(0).value(),meta=schema.entry(1).value();
  std::array<std::byte,16> sidecar{};sidecar.fill(std::byte{7});
  auto base=store.import(std::array{StateMember{id,ready(std::byte{1})},StateMember{meta,{},sidecar}}).value();
  sidecar.fill(std::byte{99});EXPECT_EQ(bytes(base.member(meta).value())[0],std::byte{7});
  auto draft=store.begin(base).value();EXPECT_FALSE(draft.replace(meta,ready(std::byte{0})).ok());
  ASSERT_TRUE(draft.replace_metadata(meta,sidecar).ok());auto candidate=draft.finalize().value();
  EXPECT_EQ(bytes(candidate.member(meta).value())[0],std::byte{99});
  EXPECT_EQ(bytes(base.member(meta).value())[0],std::byte{7});
}
TEST(StateVersions, EscapedReaderKeepsBackingCapacityAfterDirectoryRelease) {
  auto schema=StateSchema::create(std::array{resource_spec()}).value();auto id=schema.entry(0).value();
  auto store=StateStore::create(schema,{.max_versions=1,.max_drafts=1,.max_backing_bytes=16}).value();
  auto base=store.import(std::array{StateMember{id,ready(std::byte{1})}}).value();
  auto reader=base.member(id).value();base.reset();ASSERT_TRUE(store.collect().ok());
  EXPECT_EQ(store.stats().versions,0u);EXPECT_EQ(store.stats().backing_bytes,16u);
  EXPECT_FALSE(store.import(std::array{StateMember{id,ready(std::byte{2})}}));
  reader.reset();ASSERT_TRUE(store.collect().ok());EXPECT_EQ(store.stats().backing_bytes,0u);
  EXPECT_TRUE(store.import(std::array{StateMember{id,ready(std::byte{3})}}));
}
TEST(StateVersions, BackingCapacityDeduplicatesDifferentBlocksOfOnePool) {
  auto schema=StateSchema::create(std::array{resource_spec(),resource_spec()}).value();
  auto store=StateStore::create(schema,{.max_backing_bytes=256}).value();
  auto pool=LeasePool::create(m::system_resource(),256,{.max_blocks=2,.backing_alignment=64}).value();
  auto x=pool.allocate(16,64).value(),y=pool.allocate(16,64).value();
  auto first=x.publish().value(),second=y.publish().value();
  auto version=store.import(std::array{StateMember{schema.entry(0).value(),first},StateMember{schema.entry(1).value(),second}}).value();
  EXPECT_EQ(store.stats().backing_roots,1u);EXPECT_EQ(store.stats().backing_bytes,256u);
  EXPECT_EQ(store.stats().retained_blocks,2u);(void)version;
}
TEST(StateVersions, MixedRootsMetadataAndCollectedPrefixKeepExactAccounting) {
  const std::array specs{resource_spec(), resource_spec(), resource_spec(),
      StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=4}};
  auto schema=StateSchema::create(specs).value();
  auto store=StateStore::create(schema).value();
  auto a=LeasePool::create(m::system_resource(),128,{.max_blocks=2,.backing_alignment=64}).value();
  auto b=LeasePool::create(m::system_resource(),64,{.max_blocks=1,.backing_alignment=64}).value();
  auto first=a.allocate(16,64).value().publish().value();
  auto middle=b.allocate(16,64).value().publish().value();
  auto last=a.allocate(16,64).value().publish().value();
  const std::array<std::byte,4> metadata{};
  auto base=store.import(std::array{
      StateMember{schema.entry(0).value(),first}, StateMember{schema.entry(1).value(),middle},
      StateMember{schema.entry(2).value(),last}, StateMember{schema.entry(3).value(),{},metadata}}).value();
  first.reset();
  const auto before=store.stats();
  EXPECT_EQ(before.retained_blocks,4u);EXPECT_EQ(before.backing_roots,3u);
  EXPECT_EQ(before.backing_bytes,196u);EXPECT_EQ(before.host_snapshot_bytes,4u);
  auto draft=store.begin(base).value();
  ASSERT_TRUE(draft.replace(schema.entry(0).value(),last).ok());
  auto updated=draft.finalize().value();base.reset();
  ASSERT_TRUE(store.collect().ok());
  const auto after=store.stats();
  EXPECT_EQ(after.retained_blocks,3u);EXPECT_EQ(after.backing_roots,3u);
  EXPECT_EQ(after.backing_bytes,196u);EXPECT_EQ(after.host_snapshot_bytes,4u);
  EXPECT_TRUE(updated.valid());
}
TEST(StateVersions, CapacityCancelCloseAndCrossStoreAreExplicit) {
  auto schema=StateSchema::create(std::array{resource_spec()}).value();auto id=schema.entry(0).value();
  auto store=StateStore::create(schema,{.max_versions=1,.max_drafts=1}).value(),other=StateStore::create(schema).value();
  auto base=store.import(std::array{StateMember{id,ready(std::byte{1})}}).value();
  EXPECT_FALSE(other.begin(base));auto draft=store.begin(base).value();EXPECT_FALSE(store.begin(base));
  EXPECT_FALSE(draft.finalize());ASSERT_TRUE(draft.cancel().ok());EXPECT_TRUE(draft.finalize().error().IsCancelled());
  ASSERT_TRUE(store.close().ok());EXPECT_FALSE(store.begin(base));
  EXPECT_EQ(bytes(base.member(id).value())[0],std::byte{1});EXPECT_TRUE(store.drain().ok());
}
TEST(StateVersions, MetadataQuotaFailureLeavesSourceAndReturnsAllocation) {
  auto spec=resource_spec();m::QuotaAllocator quota(m::system_allocator(),1);
  EXPECT_FALSE(StateSchema::create(std::array{spec},quota.get()));EXPECT_EQ(quota.used(),0u);
  auto schema=StateSchema::create(std::array{spec}).value();auto source=ready(std::byte{4});
  EXPECT_FALSE(StateStore::create(schema,{.allocator=quota.get()}));EXPECT_TRUE(source.valid());EXPECT_EQ(quota.used(),0u);
}
TEST(StateVersions, HeadConcurrentExpectedBaseHasExactlyOneWinner) {
  auto schema=StateSchema::create(std::array{resource_spec()}).value();auto id=schema.entry(0).value();
  auto store=StateStore::create(schema).value();auto base=store.import(std::array{StateMember{id,ready(std::byte{1})}}).value();
  auto head=StateHead::create(store,base).value();auto a=store.begin(base).value(),b=store.begin(base).value();
  ASSERT_TRUE(a.replace(id,ready(std::byte{2})).ok());ASSERT_TRUE(b.replace(id,ready(std::byte{3})).ok());
  auto one=a.finalize().value(),two=b.finalize().value();const auto expected=base.identity();
  std::atomic<unsigned> wins{0};
  std::thread first([&]{if(head.publish(expected,one).ok())++wins;});
  std::thread second([&]{if(head.publish(expected,two).ok())++wins;});first.join();second.join();
  EXPECT_EQ(wins.load(),1u);EXPECT_EQ(bytes(base.member(id).value())[0],std::byte{1});
}
TEST(StateVersions, EmptySchemaHasAnActualEmptyVersion) {
  auto schema=StateSchema::create({}).value();auto store=StateStore::create(schema).value();
  auto empty=store.import({}).value();auto draft=store.begin(empty).value();EXPECT_TRUE(draft.finalize());
}
TEST(StateVersions, PendingAdmittedUseKeepsBackingLedgerUntilReleased) {
  auto schema=StateSchema::create(std::array{resource_spec()}).value();auto id=schema.entry(0).value();
  auto store=StateStore::create(schema).value();auto base=store.import(std::array{StateMember{id,ready(std::byte{5})}}).value();
  auto pool=LeaseUsePool::create().value();auto use=pool.begin().value();
  ASSERT_TRUE(use.read(base.member(id).value()).ok());base.reset();store.collect().IgnoreError();
  EXPECT_EQ(store.stats().backing_bytes,16u);use={};store.collect().IgnoreError();
  EXPECT_EQ(store.stats().backing_bytes,0u);
}
TEST(StateVersions, HostSnapshotCapacityRejectsBeforeAnyDirectoryPublication) {
  auto schema=StateSchema::create(std::array{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=32}}).value();
  auto store=StateStore::create(schema,{.max_host_snapshot_bytes=16}).value();
  std::array<std::byte,32> source{};
  EXPECT_FALSE(store.import(std::array{StateMember{schema.entry(0).value(),{},source}}));
  EXPECT_EQ(store.stats().versions,0u);EXPECT_EQ(store.stats().retained_blocks,0u);
  EXPECT_EQ(store.stats().host_snapshot_bytes,0u);
}
TEST(StateVersions, FailedSnapshotReplacementLeavesTheSharedBaseIntact) {
  auto schema=StateSchema::create(std::array{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16}}).value();
  auto id=schema.entry(0).value();auto store=StateStore::create(schema,{.max_host_snapshot_bytes=16}).value();
  std::array<std::byte,16> source{};source.fill(std::byte{7});
  auto base=store.import(std::array{StateMember{id,{},source}}).value();auto draft=store.begin(base).value();
  source.fill(std::byte{9});EXPECT_TRUE(draft.replace_metadata(id,source).IsUnavailable());
  auto unchanged=draft.finalize().value();EXPECT_EQ(bytes(unchanged.member(id).value())[0],std::byte{7});
  EXPECT_EQ(store.stats().host_snapshot_bytes,16u);
}
class DistinctHostResource final : public m::MemoryResource {
public:
  std::string_view name()const noexcept override{return "state-test-distinct-host";}
  m::MemoryProperties properties()const noexcept override{return m::system_resource().properties();}
  hyle::Result<m::ResourceAllocation> allocate(std::size_t size,std::size_t alignment)noexcept override{return m::system_resource().resource().allocate(size,alignment);}
  void deallocate(m::ResourceAllocation allocation)noexcept override{m::system_resource().resource().deallocate(allocation);}
};
TEST(StateVersions, ResourceConstraintRejectsForeignBackingBeforeImport) {
  DistinctHostResource other;auto spec=resource_spec();spec.resource=m::MemoryResourceRef(other);
  auto schema=StateSchema::create(std::array{spec}).value();auto store=StateStore::create(schema).value();auto source=ready(std::byte{8});
  EXPECT_TRUE(store.import(std::array{StateMember{schema.entry(0).value(),source}}).error().IsInvalidArgument());
  EXPECT_TRUE(source.valid());EXPECT_EQ(store.stats().retained_blocks,0u);
}
} // namespace
