#include <gtest/gtest.h>
#include "hyle/compute/state_versions.h"
#include "hyle/compute/state_clone.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/mem/quota_allocator.h"
#include <array>
#include <thread>
#include <vector>

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
TEST(StateVersions, IndependentRootsPreserveMembersAndEscapedReadersAcrossCollection) {
  constexpr std::size_t count = 32;
  auto schema = StateSchema::create(std::vector<StateEntrySpec>(count, resource_spec())).value();
  auto store = StateStore::create(schema, {.max_retained_blocks = count,
                                         .max_backing_bytes = count * 16}).value();
  std::vector<ReadLease> sources;
  std::vector<StateMember> members;
  for (std::size_t i = 0; i < count; ++i)
    sources.push_back(ready(std::byte(i + 1)));
  for (std::size_t i = 0; i < count; ++i)
    members.push_back({schema.entry(i).value(), sources[(i * 13) % count]});
  auto version = store.import(members).value();
  sources.clear(); members.clear();
  for (int pass = 0; pass < 2; ++pass) {
    const auto stats = store.stats();
    EXPECT_EQ(stats.backing_roots, count);
    EXPECT_EQ(stats.retained_blocks, count);
    EXPECT_EQ(stats.backing_bytes, count * 16);
    for (std::size_t i = 0; i < count; ++i)
      EXPECT_EQ(bytes(version.member(schema.entry(i).value()).value())[0],
                std::byte((i * 13) % count + 1));
  }
  auto escaped = version.member(schema.entry(7).value()).value();
  version.reset();
  ASSERT_TRUE(store.collect().ok());
  EXPECT_EQ(store.stats().retained_blocks, 1u);
  EXPECT_EQ(store.stats().backing_roots, 1u);
  EXPECT_EQ(store.stats().backing_bytes, 16u);
  EXPECT_EQ(bytes(escaped)[0], std::byte((7 * 13) % count + 1));
  escaped.reset();
  ASSERT_TRUE(store.collect().ok());
  EXPECT_EQ(store.stats().backing_roots, 0u);
  EXPECT_EQ(store.stats().backing_bytes, 0u);
  auto shared = ready(std::byte{77});
  for (std::size_t i = 0; i < count; ++i)
    members.push_back({schema.entry(i).value(), shared});
  auto reused = store.import(members).value();
  EXPECT_EQ(store.stats().retained_blocks, 1u);
  EXPECT_EQ(store.stats().backing_roots, 1u);
  for (std::size_t i = 0; i < count; ++i)
    EXPECT_EQ(bytes(reused.member(schema.entry(i).value()).value())[0], std::byte{77});
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
struct SnapshotAllocator final {
  std::size_t calls=0;
  bool deny=false;
  m::AllocatorRef get() noexcept {
    return {
      [](void*p,std::size_t n,std::size_t a)noexcept->void*{auto&s=*static_cast<SnapshotAllocator*>(p);
        if(n)++s.calls;return s.deny?nullptr:m::system_allocator().allocate(n,a);},
      [](void*,void*p,std::size_t n,std::size_t a)noexcept{m::system_allocator().deallocate(p,n,a);},
      [](void*p,void*v,std::size_t old,std::size_t n,std::size_t a)noexcept->void*{auto&s=*static_cast<SnapshotAllocator*>(p);
        if(n)++s.calls;return s.deny&&n?nullptr:m::system_allocator().reallocate(v,old,n,a);},this};
  }
};
TEST(StateVersions, HostSnapshotsReuseOnlyRetiredExactBlocksWithoutWarmAllocations) {
  SnapshotAllocator allocator;
  auto schema=StateSchema::create(std::array{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16}}).value();
  auto id=schema.entry(0).value();
  auto store=StateStore::create(schema,{.max_versions=2,.max_drafts=1,.max_retained_blocks=2,
      .max_backing_bytes=32,.max_host_snapshot_bytes=32,.allocator=allocator.get()}).value();
  std::array<std::byte,16> source{};
  auto current=store.import(std::array{StateMember{id,{},source}}).value();
  std::size_t warmed=0;
  for(unsigned step=1;step<=32;++step){
    auto draft=store.begin(current).value();source.fill(std::byte(step));
    ASSERT_TRUE(draft.replace_metadata(id,source).ok());
    auto next=draft.finalize();ASSERT_TRUE(next)<<next.error().message();current=std::move(*next);
    EXPECT_EQ(bytes(current.member(id).value())[0],std::byte(step));
    if(step==2){warmed=allocator.calls;allocator.deny=true;}
  }
  EXPECT_EQ(allocator.calls,warmed);
  EXPECT_EQ(store.stats().retained_blocks,2u);EXPECT_EQ(store.stats().host_snapshot_bytes,32u);
  EXPECT_EQ(store.stats().backing_bytes,32u);
  current.reset();ASSERT_TRUE(store.collect().ok());EXPECT_EQ(store.stats().backing_bytes,0u);
}
TEST(StateVersions, EscapedHostSnapshotsSurviveRecordsStoreAndNewNativeReadAdmission) {
  auto schema=StateSchema::create(std::array{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16}}).value();
  auto id=schema.entry(0).value();auto store=StateStore::create(schema).value();
  std::array<std::byte,16> source{};source.fill(std::byte{7});
  auto current=store.import(std::array{StateMember{id,{},source}}).value();
  auto escaped=current.member(id).value();
  for(unsigned step=8;step<=12;++step){auto draft=store.begin(current).value();source.fill(std::byte(step));
    ASSERT_TRUE(draft.replace_metadata(id,source).ok());current=draft.finalize().value();
    EXPECT_EQ(bytes(escaped)[0],std::byte{7});}
  current.reset();ASSERT_TRUE(store.collect().ok());EXPECT_EQ(store.stats().host_snapshot_bytes,16u);
  ASSERT_TRUE(store.close().ok());store={};
  EXPECT_EQ(bytes(escaped)[0],std::byte{7});EXPECT_TRUE(escaped.try_write().error().IsUnavailable());
  auto uses=LeaseUsePool::create().value();auto use=uses.begin().value();ASSERT_TRUE(use.read(escaped).ok());
}
TEST(StateVersions, HostSnapshotReuseWaitsForActualNativeUseAndPreservesFailure) {
  auto schema=StateSchema::create(std::array{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16}}).value();
  auto id=schema.entry(0).value();auto store=StateStore::create(schema,{.max_retained_blocks=2,
      .max_backing_bytes=32,.max_host_snapshot_bytes=32}).value();
  std::array<std::byte,16> source{};source.fill(std::byte{1});
  auto current=store.import(std::array{StateMember{id,{},source}}).value();
  auto uses=LeaseUsePool::create().value();auto use=uses.begin().value();
  ASSERT_TRUE(use.read(current.member(id).value()).ok());
  {auto draft=store.begin(current).value();source.fill(std::byte{2});ASSERT_TRUE(draft.replace_metadata(id,source).ok());current=draft.finalize().value();}
  auto draft=store.begin(current).value();source.fill(std::byte{3});
  EXPECT_TRUE(draft.replace_metadata(id,source).IsUnavailable());
  EXPECT_EQ(bytes(current.member(id).value())[0],std::byte{2});
  use={};ASSERT_TRUE(draft.replace_metadata(id,source).ok());current=draft.finalize().value();
  EXPECT_EQ(bytes(current.member(id).value())[0],std::byte{3});
}
TEST(StateVersions, IdleHostSnapshotCapacityIsChargedAndReclaimedUnderPressure) {
  const std::array specs{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16},
      StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=32}};
  auto schema=StateSchema::create(specs).value();auto small=schema.entry(0).value(),large=schema.entry(1).value();
  auto store=StateStore::create(schema,{.max_retained_blocks=3,.max_backing_bytes=80,.max_host_snapshot_bytes=80}).value();
  std::array<std::byte,16> a{};std::array<std::byte,32>b{};
  auto current=store.import(std::array{StateMember{small,{},a},StateMember{large,{},b}}).value();
  {auto draft=store.begin(current).value();a.fill(std::byte{1});ASSERT_TRUE(draft.replace_metadata(small,a).ok());current=draft.finalize().value();}
  EXPECT_EQ(store.stats().host_snapshot_bytes,64u);
  {auto draft=store.begin(current).value();b.fill(std::byte{2});ASSERT_TRUE(draft.replace_metadata(large,b).ok());current=draft.finalize().value();}
  EXPECT_EQ(store.stats().host_snapshot_bytes,80u);EXPECT_EQ(store.stats().backing_bytes,80u);
  current.reset();ASSERT_TRUE(store.close().ok());EXPECT_EQ(store.stats().host_snapshot_bytes,0u);
}
TEST(StateVersions, HostSnapshotAllocationFailureKeepsTheBaseAndDirectoryIntact) {
  SnapshotAllocator allocator;
  auto schema=StateSchema::create(std::array{StateEntrySpec{.kind=StateEntryKind::kHostMetadata,.metadata_bytes=16}}).value();
  auto id=schema.entry(0).value();auto store=StateStore::create(schema,{.allocator=allocator.get()}).value();
  std::array<std::byte,16> source{};source.fill(std::byte{7});
  auto current=store.import(std::array{StateMember{id,{},source}}).value();auto draft=store.begin(current).value();
  allocator.deny=true;source.fill(std::byte{9});EXPECT_TRUE(draft.replace_metadata(id,source).IsResourceExhausted());
  EXPECT_EQ(store.stats().retained_blocks,1u);EXPECT_EQ(store.stats().host_snapshot_bytes,16u);
  auto unchanged=draft.finalize();ASSERT_TRUE(unchanged)<<unchanged.error().message();
  EXPECT_EQ(bytes(unchanged->member(id).value())[0],std::byte{7});
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
TEST(StateVersions, SlicedCloneWriterCannotPublishAnIncompleteMember) {
  auto context=Context::create(cpu::device()).value();
  auto queue=context.create_queue().value();
  const auto spec=resource_spec();
  auto schema=StateSchema::create(std::array{spec}).value();
  const auto id=schema.entry(0).value();
  auto copy=CopyBatchProgram::prepare(context,std::array{
      CopyResourceSpec{m::system_resource(),16},
      CopyResourceSpec{m::system_resource(),16}}).value();
  const std::array slices{std::array<std::size_t,2>{4,8},
                          std::array<std::size_t,2>{8,0}};
  for(const auto &slice:slices) {
    SCOPED_TRACE(::testing::Message()<<"offset="<<slice[0]<<", size="<<slice[1]);
    auto store=StateStore::create(schema).value();
    auto base=store.import(std::array{StateMember{id,ready(std::byte{37})}}).value();
    auto route=StateCloneRoute::prepare(store,
        {.resource=m::system_resource(),.capacity_bytes=1024},copy,
        [&queue](const CopyBatchProgram &program) noexcept {
          return CopyBatchInvocation::create(program,queue);
        });
    ASSERT_TRUE(route)<<route.error().message();
    auto draft_status=store.begin(base);
    ASSERT_TRUE(draft_status)<<draft_status.error().message();
    auto draft=std::move(*draft_status);
    auto ticket=draft.clone_for_write(id,*route);
    ASSERT_TRUE(ticket)<<ticket.error().message();
    auto writer_status=ticket->wait();
    ASSERT_TRUE(writer_status)<<writer_status.error().message();
    auto writer=std::move(*writer_status);
    const auto generation=writer.lease().generation();
    const auto original_view=writer.lease().view();
    ASSERT_TRUE(original_view)<<original_view.error().message();
    auto published=writer.lease().publish();
    ASSERT_TRUE(published)<<published.error().message();
    auto full=std::move(*published);
    auto piece_status=full.subview(slice[0],slice[1]);
    ASSERT_TRUE(piece_status)<<piece_status.error().message();
    auto piece=std::move(*piece_status);
    full.reset();
    auto shortened=piece.try_write();
    ASSERT_TRUE(shortened)<<shortened.error().message();
    writer.lease()=std::move(*shortened);
    EXPECT_EQ(writer.lease().generation(),generation);
    EXPECT_EQ(writer.lease().size(),slice[1]);
    const auto shortened_view=writer.lease().view();
    ASSERT_TRUE(shortened_view)<<shortened_view.error().message();
    EXPECT_EQ(shortened_view->native_handle(),original_view->native_handle());
    EXPECT_EQ(shortened_view->offset(),original_view->offset()+slice[0]);
    const auto before=store.stats();
    const auto rejected=writer.publish();
    EXPECT_TRUE(rejected.IsInvalidArgument())<<rejected.message();
    EXPECT_EQ(rejected.message(),"state resource byte extent disagrees with schema");
    EXPECT_TRUE(writer.lease().valid());
    EXPECT_EQ(writer.lease().generation(),generation);
    EXPECT_EQ(writer.lease().size(),slice[1]);
    EXPECT_EQ(store.stats().retained_blocks,before.retained_blocks);
    auto candidate=draft.finalize();
    EXPECT_FALSE(candidate);
    if(!candidate) EXPECT_TRUE(candidate.error().IsUnavailable())<<candidate.error().message();
    else {
      auto invalid_member=candidate->member(id);
      ASSERT_TRUE(invalid_member)<<invalid_member.error().message();
      EXPECT_EQ(invalid_member->size(),16U);
    }
    auto original=base.member(id);
    ASSERT_TRUE(original)<<original.error().message();
    EXPECT_EQ(original->size(),16U);
    EXPECT_EQ(bytes(*original)[0],std::byte{37});
  }
}
} // namespace
