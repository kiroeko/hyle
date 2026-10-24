#include <gtest/gtest.h>
#include "hyle/compute/state_clone.h"
#include "hyle/compute/backends/cpu.h"
#include <array>
#include <atomic>
#include <thread>

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

// Transparent observation boundary over actual CPU F2 copies. Only wait is
// gated; all copying, bindings, publication and retirement remain in F2. An
// overlapping access is counted and withheld rather than racing its native
// handle, so the old implementation fails deterministically without UB.
struct CopyObservation final {
  std::atomic<bool> entered{false}, released{false}, active{false};
  std::atomic<bool> fail_next_wait{false}, unavailable_next_wait{false};
  std::atomic<unsigned> forbidden_accesses{0}, native_waits{0}, native_cancellations{0};
  bool retain_output = false;
  ReadLease retained_output;
  bool reject_access() noexcept {
    if (!active.load(std::memory_order_acquire)) return false;
    forbidden_accesses.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  void enter() noexcept {
    native_waits.fetch_add(1, std::memory_order_relaxed);
    active.store(true, std::memory_order_release);
    entered.store(true, std::memory_order_release); entered.notify_all();
    released.wait(false, std::memory_order_acquire);
  }
  void release() noexcept {
    released.store(true, std::memory_order_release); released.notify_all();
  }
};
struct ObservedCopySubmission final {
  CopyBatchSubmission original;
  CopyObservation *observation = nullptr;
  bool valid() const noexcept {
    if (observation && observation->reject_access()) return true;
    return original.valid();
  }
  hyle::Status wait() noexcept {
    observation->enter();
    const auto terminal = original.wait();
    observation->active.store(false, std::memory_order_release);
    if (terminal.ok() && observation->fail_next_wait.exchange(false))
      return hyle::Status::DataLoss("test original copy observation failed");
    if (terminal.ok() && observation->unavailable_next_wait.exchange(false))
      return hyle::Status::Unavailable("test original copy observation is not yet available");
    return terminal;
  }
  hyle::Status request_cancel() noexcept {
    observation->native_cancellations.fetch_add(1, std::memory_order_relaxed);
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native submission");
    return original.request_cancel();
  }
  hyle::Result<ReadLease> output(CopyPhaseId phase, CopyResourceId resource,
                                std::size_t offset, std::size_t bytes) noexcept {
    if (observation->reject_access()) return std::unexpected(hyle::Status::Unavailable("test explicit observer owns native submission"));
    auto ready = original.output(phase, resource, offset, bytes);
    if (ready && observation->retain_output) observation->retained_output = *ready;
    return ready;
  }
};
struct ObservedCopyInvocation final {
  CopyBatchInvocation original;
  CopyObservation *observation = nullptr;
  hyle::Status bind_source(CopyResourceId id, const ReadLease &source) noexcept {
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native invocation");
    return original.bind_source(id, source);
  }
  hyle::Status bind_mutable(CopyResourceId id, WriteLease &&destination) noexcept {
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native invocation");
    return original.bind_mutable(id, std::move(destination));
  }
  hyle::Status prepare_replay(std::span<const CopyPhase> phases,
                               std::span<const CopyRange> ranges) noexcept {
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native invocation");
    return original.prepare_replay(phases, ranges);
  }
  hyle::Result<ObservedCopySubmission> submit() noexcept {
    if (observation->reject_access()) return std::unexpected(hyle::Status::Unavailable("test explicit observer owns native invocation"));
    HYLE_ASSIGN_OR_RETURN(auto submitted, original.submit());
    return ObservedCopySubmission{std::move(submitted), observation};
  }
  hyle::Status reset() noexcept {
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native invocation");
    return original.reset();
  }
  hyle::Status close() noexcept {
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native invocation");
    return original.close();
  }
  hyle::Status drain() noexcept {
    if (observation->reject_access()) return hyle::Status::Unavailable("test explicit observer owns native invocation");
    return original.drain();
  }
  bool in_flight() const noexcept {
    if (observation->reject_access()) return false;
    return original.in_flight();
  }
  CopyBatchReport report() const noexcept {
    if (observation->reject_access()) return {};
    return original.report();
  }
};

TEST(StateClone, ExplicitObserverExcludesNativeCancellationCollectionAndClose) {
  for (const bool close_store : {false, true}) {
    SCOPED_TRACE(close_store);
    CopyObservation observation;
    Fixture f;
    auto *endpoint = &f.queue;
    auto *boundary = &observation;
    auto route = StateCloneRoute::prepare(f.store,
        {.resource=m::system_resource(), .capacity_bytes=1024}, f.copy,
        [endpoint, boundary](const CopyBatchProgram &program) noexcept
            -> hyle::Result<ObservedCopyInvocation> {
          HYLE_ASSIGN_OR_RETURN(auto original, CopyBatchInvocation::create(program, *endpoint));
          return ObservedCopyInvocation{std::move(original), boundary};
        }).value();
    auto base = f.base();
    auto draft = f.store.begin(base).value();
    auto other = f.store.begin(base).value();
    auto ticket = draft.clone_for_write(f.id, route).value();
    hyle::Status observed;
    std::thread waiter([&] {
      auto writer = ticket.wait();
      if (writer) observed = hyle::Status::Internal("cancelled/closed candidate became writable");
      else observed = writer.error();
    });
    // This notification is inside the route's original wait callback, after
    // CloneTicket has pinned the slot. A started thread alone proves nothing.
    observation.entered.wait(false, std::memory_order_acquire);
    if (close_store) {
      EXPECT_TRUE(f.store.close().IsUnavailable());
      EXPECT_TRUE(route.rearm().IsFailedPrecondition());
      EXPECT_FALSE(f.store.begin(base));
    } else {
      EXPECT_TRUE(draft.cancel().ok());
      EXPECT_TRUE(route.rearm().IsUnavailable());
      auto blocked = other.clone_for_write(f.id, route);
      EXPECT_FALSE(blocked);
      if (!blocked) EXPECT_TRUE(blocked.error().IsUnavailable());
    }
    EXPECT_TRUE(f.store.collect().IsUnavailable());
    const auto report = route.report();
    EXPECT_FALSE(report);
    if (!report) EXPECT_TRUE(report.error().IsUnavailable());
    EXPECT_EQ(observation.forbidden_accesses.load(std::memory_order_relaxed), 0u);
    observation.release();
    waiter.join();
    EXPECT_TRUE(close_store ? observed.IsFailedPrecondition() : observed.IsCancelled());
    EXPECT_FALSE(draft.finalize());
    EXPECT_EQ(first(base.member(f.id).value()), std::byte{11});
    ticket = {}; draft = {};
    EXPECT_TRUE(f.store.collect().ok());
    EXPECT_EQ(f.store.stats().active_clones, 0u);
    if (close_store) {
      EXPECT_TRUE(f.store.drain().ok());
    } else {
      ASSERT_TRUE(route.rearm().ok());
      auto next = f.store.begin(base).value();
      auto copied = next.clone_for_write(f.id, route).value();
      auto writer = copied.wait().value();
      ASSERT_TRUE(writer.lease().view()->fill(std::byte{47}).ok());
      ASSERT_TRUE(writer.publish().ok());
      auto candidate = next.finalize().value();
      EXPECT_EQ(first(candidate.member(f.id).value()), std::byte{47});
      EXPECT_EQ(first(base.member(f.id).value()), std::byte{11});
      EXPECT_EQ(route.report()->physical_bytes, 16u);
    }
    EXPECT_EQ(observation.forbidden_accesses.load(std::memory_order_relaxed), 0u);
  }
}

TEST(StateClone, OldFailedControlsCannotObserveOrCancelAReusedCopySlot) {
  enum class OldAction { kWait, kPoll, kCancelTicket, kDropTicket, kCancelDraft, kDropDraft };
  for (const auto action : {OldAction::kWait, OldAction::kPoll, OldAction::kCancelTicket,
                            OldAction::kDropTicket, OldAction::kCancelDraft, OldAction::kDropDraft}) {
    SCOPED_TRACE(static_cast<unsigned>(action));
    CopyObservation observation;
    observation.release(); // Sequential original CPU copies; no blocked observer.
    Fixture f;
    auto *endpoint = &f.queue;
    auto *boundary = &observation;
    auto route = StateCloneRoute::prepare(f.store,
        {.resource=m::system_resource(), .capacity_bytes=1024}, f.copy,
        [endpoint, boundary](const CopyBatchProgram &program) noexcept
            -> hyle::Result<ObservedCopyInvocation> {
          HYLE_ASSIGN_OR_RETURN(auto original, CopyBatchInvocation::create(program, *endpoint));
          return ObservedCopyInvocation{std::move(original), boundary};
        }).value();
    auto base = f.base();
    auto old_draft = f.store.begin(base).value();
    auto old_ticket = old_draft.clone_for_write(f.id, route).value();
    ASSERT_TRUE(old_draft.cancel().ok());
    EXPECT_TRUE(old_ticket.wait().error().IsCancelled());
    ASSERT_TRUE(f.store.collect().ok());
    ASSERT_EQ(f.store.stats().active_clones, 0u);
    ASSERT_TRUE(route.rearm().ok());

    auto current = f.store.begin(base).value();
    auto current_ticket = current.clone_for_write(f.id, route).value();
    ASSERT_EQ(f.store.stats().active_clones, 1u);
    const auto native_waits = observation.native_waits.load(std::memory_order_relaxed);
    const auto native_cancellations = observation.native_cancellations.load(std::memory_order_relaxed);
    switch (action) {
      case OldAction::kWait: EXPECT_TRUE(old_ticket.wait().error().IsCancelled()); break;
      case OldAction::kPoll: EXPECT_TRUE(old_ticket.poll().error().IsCancelled()); break;
      case OldAction::kCancelTicket: EXPECT_TRUE(old_ticket.request_cancel().ok()); break;
      case OldAction::kDropTicket: old_ticket = {}; break;
      case OldAction::kCancelDraft: EXPECT_TRUE(old_draft.cancel().ok()); break;
      case OldAction::kDropDraft: old_draft = {}; break;
    }
    EXPECT_EQ(observation.native_waits.load(std::memory_order_relaxed), native_waits);
    EXPECT_EQ(observation.native_cancellations.load(std::memory_order_relaxed), native_cancellations);
    // Also release whichever old controls remain while the new slot is live.
    old_ticket = {}; old_draft = {};
    EXPECT_EQ(observation.native_cancellations.load(std::memory_order_relaxed), native_cancellations);
    EXPECT_TRUE(f.store.collect().ok());
    EXPECT_EQ(f.store.stats().active_clones, 1u);
    auto writer = current_ticket.wait();
    EXPECT_TRUE(writer);
    if (!writer) continue;
    ASSERT_TRUE(writer->lease().view()->fill(std::byte{47}).ok());
    ASSERT_TRUE(writer->publish().ok());
    auto candidate = current.finalize().value();
    EXPECT_EQ(first(candidate.member(f.id).value()), std::byte{47});
    EXPECT_EQ(first(base.member(f.id).value()), std::byte{11});

    // An old control must not silently disable a successfully reused slot.
    auto next = f.store.begin(candidate).value();
    auto copied = next.clone_for_write(f.id, route);
    EXPECT_TRUE(copied);
    if (!copied) continue;
    auto next_writer = copied->wait().value();
    ASSERT_TRUE(next_writer.publish().ok());
    EXPECT_EQ(first(next.finalize()->member(f.id).value()), std::byte{47});
    EXPECT_EQ(observation.forbidden_accesses.load(std::memory_order_relaxed), 0u);
  }
}

TEST(StateClone, FailedNativeObservationRevokesOwnershipBeforeColdRearm) {
  for (const bool explicit_wait : {false, true}) {
    SCOPED_TRACE(explicit_wait);
    CopyObservation observation;
    observation.release();
    Fixture f;
    auto *endpoint = &f.queue;
    auto *boundary = &observation;
    auto route = StateCloneRoute::prepare(f.store,
        {.resource=m::system_resource(), .capacity_bytes=1024}, f.copy,
        [endpoint, boundary](const CopyBatchProgram &program) noexcept
            -> hyle::Result<ObservedCopyInvocation> {
          HYLE_ASSIGN_OR_RETURN(auto original, CopyBatchInvocation::create(program, *endpoint));
          return ObservedCopyInvocation{std::move(original), boundary};
        }).value();
    auto base = f.base();
    auto old_draft = f.store.begin(base).value();
    auto old_ticket = old_draft.clone_for_write(f.id, route).value();
    observation.fail_next_wait.store(true);
    auto failure = explicit_wait ? old_ticket.wait() : old_ticket.poll();
    ASSERT_FALSE(failure);
    EXPECT_TRUE(failure.error().IsDataLoss());
    EXPECT_TRUE(old_draft.finalize().error().IsDataLoss());
    ASSERT_TRUE(f.store.collect().ok());
    ASSERT_EQ(f.store.stats().active_clones, 0u);
    ASSERT_TRUE(route.rearm().ok());

    auto next = f.store.begin(base).value();
    auto ticket = next.clone_for_write(f.id, route).value();
    const auto waits = observation.native_waits.load(std::memory_order_relaxed);
    const auto cancellations = observation.native_cancellations.load(std::memory_order_relaxed);
    EXPECT_TRUE(old_ticket.wait().error().IsDataLoss());
    EXPECT_TRUE(old_ticket.poll().error().IsDataLoss());
    EXPECT_TRUE(old_ticket.request_cancel().ok());
    EXPECT_TRUE(old_draft.cancel().ok());
    old_ticket = {}; old_draft = {};
    EXPECT_EQ(observation.native_waits.load(std::memory_order_relaxed), waits);
    EXPECT_EQ(observation.native_cancellations.load(std::memory_order_relaxed), cancellations);
    EXPECT_TRUE(f.store.collect().ok());
    EXPECT_EQ(f.store.stats().active_clones, 1u);
    auto writer = ticket.wait().value();
    ASSERT_TRUE(writer.publish().ok());
    EXPECT_EQ(first(next.finalize()->member(f.id).value()), std::byte{11});
    EXPECT_EQ(first(base.member(f.id).value()), std::byte{11});
  }
}

TEST(StateClone, UnavailableExclusiveOutputRetainsTheSameCopyForRetry) {
  CopyObservation observation;
  observation.release();
  observation.retain_output = true;
  Fixture f;
  auto *endpoint = &f.queue;
  auto *boundary = &observation;
  auto route = StateCloneRoute::prepare(f.store,
      {.resource=m::system_resource(), .capacity_bytes=1024}, f.copy,
      [endpoint, boundary](const CopyBatchProgram &program) noexcept
          -> hyle::Result<ObservedCopyInvocation> {
        HYLE_ASSIGN_OR_RETURN(auto original, CopyBatchInvocation::create(program, *endpoint));
        return ObservedCopyInvocation{std::move(original), boundary};
      }).value();
  auto base = f.base();
  auto draft = f.store.begin(base).value();
  auto ticket = draft.clone_for_write(f.id, route).value();
  auto blocked = ticket.poll();
  ASSERT_FALSE(blocked);
  EXPECT_TRUE(blocked.error().IsUnavailable());
  ASSERT_TRUE(observation.retained_output.valid());
  EXPECT_EQ(first(observation.retained_output), std::byte{11});
  EXPECT_TRUE(draft.finalize().error().IsUnavailable());
  EXPECT_TRUE(f.store.collect().ok());
  EXPECT_EQ(f.store.stats().active_clones, 1u);
  const auto cancellations = observation.native_cancellations.load(std::memory_order_relaxed);
  observation.retained_output.reset();
  observation.retain_output = false;
  auto writer = ticket.poll().value();
  ASSERT_TRUE(writer.lease().view()->fill(std::byte{61}).ok());
  ASSERT_TRUE(writer.publish().ok());
  EXPECT_EQ(first(draft.finalize()->member(f.id).value()), std::byte{61});
  EXPECT_EQ(first(base.member(f.id).value()), std::byte{11});
  EXPECT_EQ(observation.native_cancellations.load(std::memory_order_relaxed), cancellations);
  EXPECT_EQ(route.report()->physical_bytes, 16u);
}

TEST(StateClone, UnavailableNativeObservationKeepsItsAcceptedSlotForRetry) {
  for (const bool explicit_wait : {false, true}) {
    SCOPED_TRACE(explicit_wait);
    CopyObservation observation;
    observation.release();
    Fixture f;
    auto *endpoint = &f.queue;
    auto *boundary = &observation;
    auto route = StateCloneRoute::prepare(f.store,
        {.resource=m::system_resource(), .capacity_bytes=1024}, f.copy,
        [endpoint, boundary](const CopyBatchProgram &program) noexcept
            -> hyle::Result<ObservedCopyInvocation> {
          HYLE_ASSIGN_OR_RETURN(auto original, CopyBatchInvocation::create(program, *endpoint));
          return ObservedCopyInvocation{std::move(original), boundary};
        }).value();
    auto base = f.base();
    auto draft = f.store.begin(base).value();
    auto ticket = draft.clone_for_write(f.id, route).value();
    observation.unavailable_next_wait.store(true);
    auto blocked = explicit_wait ? ticket.wait() : ticket.poll();
    ASSERT_FALSE(blocked);
    EXPECT_TRUE(blocked.error().IsUnavailable());
    EXPECT_TRUE(draft.finalize().error().IsUnavailable());
    EXPECT_TRUE(f.store.collect().ok());
    EXPECT_EQ(f.store.stats().active_clones, 1u);
    EXPECT_TRUE(route.rearm().ok()); // Must leave this accepted episode intact.
    EXPECT_EQ(f.store.stats().active_clones, 1u);
    auto other = f.store.begin(base).value();
    auto replacement = other.clone_for_write(f.id, route);
    EXPECT_FALSE(replacement);
    if (!replacement) EXPECT_TRUE(replacement.error().IsUnavailable());
    auto writer = ticket.poll();
    EXPECT_TRUE(writer);
    if (!writer) continue;
    ASSERT_TRUE(writer->publish().ok());
    EXPECT_EQ(first(draft.finalize()->member(f.id).value()), std::byte{11});
    EXPECT_EQ(route.report()->native_calls, 1u);
    EXPECT_EQ(observation.native_cancellations.load(std::memory_order_relaxed), 0u);
  }
}

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
TEST(StateClone, UnfinishedWriterUseRejectsPublishWithoutConsumingPermission) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  auto ticket=draft.clone_for_write(f.id,f.route).value();auto writer=ticket.wait().value();
  auto pool=LeaseUsePool::create().value();auto use=pool.begin().value();ASSERT_TRUE(use.write(writer.lease()).ok());
  EXPECT_TRUE(writer.publish().IsUnavailable());EXPECT_TRUE(writer.lease().valid());EXPECT_FALSE(draft.finalize());
  use={};ASSERT_TRUE(writer.publish().ok());EXPECT_TRUE(draft.finalize());
  EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
}
TEST(StateClone, CancelledCandidateDoesNotAdmitUnstartedSuffixClone) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  auto ticket=draft.clone_for_write(f.id,f.route).value();ticket.request_cancel().IgnoreError();
  EXPECT_TRUE(draft.clone_for_write(f.id,f.route).error().IsCancelled());
  EXPECT_TRUE(draft.finalize().error().IsCancelled());EXPECT_EQ(f.route.report()->native_calls,1u);
  EXPECT_EQ(first(base.member(f.id).value()),std::byte{11});
}
TEST(StateClone, NewReadonlyResultRetainsClonePoolAfterCallerStoreAndRouteDrop) {
  Fixture f;auto base=f.base();auto draft=f.store.begin(base).value();
  auto ticket=draft.clone_for_write(f.id,f.route).value();auto writer=ticket.wait().value();
  ASSERT_TRUE(writer.lease().view()->fill(std::byte{61}).ok());ASSERT_TRUE(writer.publish().ok());
  auto version=draft.finalize().value();auto head=StateHead::create(f.store,version).value();
  f.route={};f.store={};base.reset();draft={};writer={};
  EXPECT_EQ(first(head.current()->member(f.id).value()),std::byte{61});
  auto readers=LeaseUsePool::create().value();auto use=readers.begin().value();
  EXPECT_TRUE(use.read(version.member(f.id).value()).ok()); // pool domain is still open through version/head
}
TEST(StateClone, OpaqueReadonlyMemberSharesButCannotBeRawCloned) {
  Fixture f;
  const auto format=StorageFormat::opaque({0x53544154454f5041ULL,0x5155450000000001ULL}).value();
  const auto description=TensorDesc::opaque(DataType::kFloat32,std::array<std::int64_t,1>{4},format,16).value();
  auto schema=StateSchema::create(std::array{StateEntrySpec{.description=description}}).value();auto id=schema.entry(0).value();
  auto store=StateStore::create(schema).value();auto buffer=m::Buffer::create(m::system_resource(),16,64).value();buffer.view().fill(std::byte{71}).IgnoreError();
  auto source=ReadLease::adopt(std::move(buffer)).value();auto version=store.import(std::array{StateMember{id,source}}).value();
  auto route=StateCloneRoute::prepare(store,{.resource=m::system_resource(),.capacity_bytes=1024},f.copy,
      [&f](const CopyBatchProgram &p) noexcept {return CopyBatchInvocation::create(p,f.queue);}).value();
  auto draft=store.begin(version).value();EXPECT_TRUE(draft.clone_for_write(id,route).error().IsUnimplemented());
  EXPECT_EQ(first(draft.finalize()->member(id).value()),std::byte{71});
  EXPECT_EQ(route.report()->native_calls,0u);
}
} // namespace
