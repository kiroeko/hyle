#include <gtest/gtest.h>
#include "hyle/compute/state_clone.h"
#include "hyle/compute/backends/cpu.h"
#include <array>

namespace {
using namespace hyle::compute;
namespace m=hyle::mem;
struct Fixture {
  Context context=Context::create(cpu::device()).value();
  Queue queue=context.create_queue().value();
  StateSchema schema=StateSchema::create(std::array{StateEntrySpec{
      .description=TensorDesc::dense(DataType::kFloat32,std::array<std::int64_t,1>{4}).value()}}).value();
  StateEntryId id=schema.entry(0).value();
  StateStore store=StateStore::create(schema).value();
  CopyBatchProgram copy=CopyBatchProgram::prepare(context,std::array{
      CopyResourceSpec{m::system_resource(),16},CopyResourceSpec{m::system_resource(),16}}).value();
  StateCloneRoute route=StateCloneRoute::prepare(store,{.resource=m::system_resource(),.capacity_bytes=1024},copy,
      [this](const CopyBatchProgram &p) noexcept {return CopyBatchInvocation::create(p,queue);}).value();
  StateVersion base() {
    auto buffer=m::Buffer::create(m::system_resource(),16,64).value();buffer.view().fill(std::byte{11}).IgnoreError();
    return store.import(std::array{StateMember{id,ReadLease::adopt(std::move(buffer)).value()}}).value();
  }
};
std::byte first(const ReadLease &lease) {std::array<std::byte,16> bytes;lease.view()->read(bytes).IgnoreError();return bytes[0];}
TEST(StateClone, OnlyDirtyMemberIsCopiedAndWriterCannotModifyOldVersion) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  auto ticket=draft.clone_for_write(f.id,f.route).value();EXPECT_FALSE(draft.finalize());
  auto writer=ticket.wait().value();ASSERT_TRUE(writer.lease().view()->fill(std::byte{23}).ok());
  ASSERT_TRUE(writer.publish().ok());auto candidate=draft.finalize().value();
  EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
  EXPECT_EQ(first(candidate.member(f.id).value()),std::byte{23});
  EXPECT_EQ(f.store.stats().active_clones,0u);
  EXPECT_EQ(f.store.stats().clone_capacity_bytes,1024u);
  EXPECT_EQ(f.store.stats().backing_bytes,1040u);
  const auto report=f.route.report().value();
  EXPECT_EQ(report.logical_bytes,16u);
  EXPECT_EQ(report.physical_bytes,16u);
  EXPECT_EQ(report.native_calls,1u);
}
TEST(StateClone, AbandonedTicketCancelsCandidateAndExplicitColdRearmRestoresSlot) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  {auto ticket=draft.clone_for_write(f.id,f.route).value();}
  EXPECT_TRUE(draft.finalize().error().IsCancelled());f.store.collect().IgnoreError();
  ASSERT_TRUE(f.route.rearm().ok());auto next=f.store.begin(base).value();
  auto ticket=next.clone_for_write(f.id,f.route).value();auto writer=ticket.poll().value();
  ASSERT_TRUE(writer.publish().ok());EXPECT_TRUE(next.finalize());
  EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
}
TEST(StateClone, AbandonedWriterNeverPublishesAndPoolOutlivesStoreHandle) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  {auto ticket=draft.clone_for_write(f.id,f.route).value();auto writer=ticket.wait().value();}
  EXPECT_TRUE(draft.finalize().error().IsCancelled());
  f.store={};EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
}
TEST(StateClone, NativeConsumerMustReturnTheExactWriterRootAndGeneration) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();auto ticket=draft.clone_for_write(f.id,f.route).value();
  auto writer=ticket.wait().value();auto mutable_lease=std::move(writer.lease());
  EXPECT_FALSE(writer.complete(base.member(f.id).value()).ok());
  ASSERT_TRUE(mutable_lease.view()->fill(std::byte{47}).ok());auto ready=mutable_lease.publish().value();
  ASSERT_TRUE(writer.complete(ready).ok());auto candidate=draft.finalize().value();
  EXPECT_EQ(first(candidate.member(f.id).value()),std::byte{47});
}
TEST(StateClone, RegistrationCapacityIsCheckedBeforeOriginalFactory) {
  Fixture f;unsigned calls=0;
  auto factory=[&](const CopyBatchProgram &p) noexcept {++calls;return CopyBatchInvocation::create(p,f.queue);};
  EXPECT_FALSE(StateCloneRoute::prepare(f.store,{.resource=m::system_resource(),.capacity_bytes=1024,.slots=4},f.copy,factory));
  EXPECT_EQ(calls,0u);
  EXPECT_FALSE(StateCloneRoute::prepare(f.store,{.resource=m::system_resource(),.capacity_bytes=SIZE_MAX},f.copy,factory));
  EXPECT_EQ(calls,0u);
}
TEST(StateClone, ForeignMutableReplacementCannotBePublishedAsOriginalClone) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  auto ticket=draft.clone_for_write(f.id,f.route).value();auto writer=ticket.wait().value();
  auto other=LeasePool::create(m::system_resource(),512).value();
  writer.lease()=other.allocate(16).value();EXPECT_TRUE(writer.publish().IsInvalidArgument());
  EXPECT_FALSE(draft.finalize());EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
}
TEST(StateClone, AcceptedProducerFailurePreventsReadyVersionAndPreservesOriginal) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  auto ticket=draft.clone_for_write(f.id,f.route).value();auto writer=ticket.wait().value();
  auto pool=LeaseUsePool::create().value();auto use=pool.begin().value();
  ASSERT_TRUE(use.write(writer.lease()).ok());
  auto completion=cpu::submit_host(f.queue,[]() noexcept {return hyle::Status::DataLoss("accepted writer failure");}).value();
  ASSERT_TRUE(use.attach(completion).ok());EXPECT_TRUE(completion.wait().IsDataLoss());
  EXPECT_TRUE(writer.publish().IsDataLoss());EXPECT_TRUE(writer.lease().valid());
  EXPECT_TRUE(draft.finalize().error().IsDataLoss());
  EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
}
} // namespace
