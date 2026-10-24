#include <gtest/gtest.h>
#include "allocation_fault_fixture.h"
#include <array>
#include <cstdio>
#include "../../src/compute/provider_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/batch_graph_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/mem/stats.h"

namespace {
namespace c=hyle::compute;namespace d=c::detail;namespace m=hyle::mem;
class CountedHost final:public m::MemoryResource {
public:
  explicit CountedHost(m::AllocatorRef allocator):allocator_(allocator){}
  std::string_view name()const noexcept override{return "counted batch backing";}
  m::MemoryProperties properties()const noexcept override{return m::system_resource().properties();}
  hyle::Result<m::ResourceAllocation> allocate(std::size_t bytes,std::size_t alignment)noexcept override{
    auto*p=allocator_.allocate(bytes,alignment);if(!p)return std::unexpected(hyle::Status::ResourceExhausted("counted batch backing"));
    return m::ResourceAllocation{p,bytes,alignment};
  }
  void deallocate(m::ResourceAllocation a)noexcept override{allocator_.deallocate(a.native_handle,a.size,a.alignment);}
  hyle::Result<m::ResourceHostMapping> map_host(void*p,std::size_t offset,std::size_t,m::HostAccess)noexcept override{
    return m::ResourceHostMapping{static_cast<std::byte*>(p)+offset,nullptr};
  }
private:m::AllocatorRef allocator_;
};
struct Mock final {
  c::BatchProgramOptions options;unsigned plans=0,runs=0,queries=0,updates=0;
  unsigned fresh_queries=0,update_queries=0;
  bool permit_reuse=false,last_allow_update=false;
  std::size_t float_bytes=1536,int_bytes=1024;
  std::size_t planned_float_bytes=0,planned_int_bytes=0;
  m::ConstBufferView planned_float_view;
  d::BatchProviderVTable batch{
    .qualify=[](void*,c::BatchCapacity,const c::BatchProgramOptions&)noexcept{return hyle::Status::Ok();},
    .create=[](void*p,c::BatchCapacity,const c::BatchProgramOptions&o,m::AllocatorRef,void**out)noexcept{static_cast<Mock*>(p)->options=o;*out=p;return hyle::Status::Ok();},
    .destroy=[](void*,m::AllocatorRef)noexcept{},
    .update=[](void*p,const c::BatchLayoutVersion&,c::BatchPlanReport*report)noexcept->hyle::Result<bool>{
      auto&s=*static_cast<Mock*>(p);++s.updates;if(!s.permit_reuse || !s.plans)return false;
      report->float_workspace_bytes=s.planned_float_bytes;report->int_workspace_bytes=s.planned_int_bytes;return true;},
    .plan=[](void*p,d::QueueSequence&,const c::BatchLayoutVersion&,d::BatchWorkspaceViews views,c::BatchPlanReport*)noexcept{
      auto&s=*static_cast<Mock*>(p);++s.plans;const auto status=s.inspect(views,true);
      if(status.ok()){s.planned_float_bytes=s.float_bytes;s.planned_int_bytes=s.int_bytes;s.planned_float_view=views.float_workspace;}
      return d::EnqueueResult::accepted_command(status);},
    .run=[](void*p,d::QueueSequence&,std::span<const c::ConstTensorView>,d::BatchWorkspaceViews views)noexcept{
      auto&s=*static_cast<Mock*>(p);++s.runs;const auto status=s.inspect(views,false);
      return d::EnqueueResult::accepted_command(status.ok()?views.output.fill(std::byte{7}):status);}
  };
  hyle::Status inspect(d::BatchWorkspaceViews v,bool planning)const noexcept{
    if(batch.uses_caller_workspace){
      const auto floating=planning?float_bytes:planned_float_bytes;
      const auto integer=planning?int_bytes:planned_int_bytes;
      if(!v.float_workspace.valid() || v.float_workspace.size()<floating ||
         !v.int_workspace.valid() || v.int_workspace.size()<integer ||
         !v.host_workspace.valid() || v.host_workspace.size()<integer)
        return hyle::Status::Internal("caller workspace differs from queried layout");
      HYLE_RETURN_IF_ERROR(v.float_workspace.fill(std::byte{1}));
      HYLE_RETURN_IF_ERROR(v.int_workspace.fill(std::byte{2}));
      if(planning)HYLE_RETURN_IF_ERROR(v.host_workspace.fill(std::byte{3}));
    }else if(v.float_workspace.valid() || v.int_workspace.valid() || v.host_workspace.valid())
      return hyle::Status::Internal("native owner received unused scratch");
    return hyle::Status::Ok();
  }
  c::Provider provider(){
    batch.workspace=[](void*p,c::Queue&,const c::BatchLayoutVersion&,bool allow_update,c::BatchPlanReport*report)noexcept{
      auto&s=*static_cast<Mock*>(p);++s.queries;s.last_allow_update=allow_update;
      if(allow_update)++s.update_queries;else ++s.fresh_queries;
      const bool reuse=allow_update && s.permit_reuse && s.plans;
      report->float_workspace_bytes=reuse?s.planned_float_bytes:s.float_bytes;
      report->int_workspace_bytes=reuse?s.planned_int_bytes:s.int_bytes;return hyle::Status::Ok();};
    static const d::ProviderVTable table{
      .identity={{0x48594c4554455354ULL,0x4241544348575301ULL},"batch workspace mock","1"},
      .create_context=[](const c::Provider&p,c::Device,void*,m::AllocatorRef,void**out)noexcept{*out=d::ProviderAccess::payload<Mock*>(p);return hyle::Status::Ok();},
      .destroy_context=[](void*,m::AllocatorRef)noexcept{},
      .propose=[](void*,const d::ProviderOperationRequest&,std::span<d::ResourceBindingRequirement>,std::span<d::ResourceBindingRequirement>,d::ProviderOperationProposal*)noexcept{return hyle::Status::Unimplemented("batch-only mock");},
      .prepare=[](void*,const d::ProviderPrepareRequest&,m::AllocatorRef,d::ProviderPrepared*)noexcept{return c::ProviderPrepareOutcome::unsupported(hyle::Status::Unimplemented("batch-only mock"));},
      .drain_context=[](void*)noexcept{return hyle::Status::Ok();},
      .query_extension=[](void*p,c::ProviderExtensionId id)noexcept->const void*{return id==d::kBatchPlanningExtensionId?&static_cast<Mock*>(p)->batch:nullptr;}
    };return d::ProviderAccess::make(&table,this);
  }
};
TEST(BatchWorkspace, ColdLayoutSizesCallerScratchAndNativeOptOutKeepsActualViews){
  std::array<std::uint64_t,2> backing{};
  for(unsigned native=0;native<2;++native){
    auto stats=m::StatsAllocator::create(m::system_allocator()).value();CountedHost resource(stats.get());
    c::ReadLease retained;Mock mock;if(native)mock.batch.uses_caller_workspace=false;
    const auto storage=c::CacheStorage::plain(c::DataType::kFloat16).value();
    auto schema=c::BatchSchema::create({.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
        .key_storage=storage,.value_storage=storage,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,
        .query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=1}).value();
    {
      auto context=c::Context::create(c::cpu::device(),mock.provider()).value();auto queue=context.create_queue().value();
      auto program=c::BatchProgram::prepare(context,schema,{.resource=m::MemoryResourceRef(resource),.host_workspace=m::MemoryResourceRef(resource),
          .max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}).value();
      const auto before=stats.snapshot();auto slot=c::BatchInvocation::create(program,queue).value();
      backing[native]=stats.snapshot().live_bytes-before.live_bytes;
    }
    EXPECT_TRUE(stats.balanced());
    // CPU upload deliberately accepts only proven system-resource writers.
    // Count actual custom-resource backings above; exercise uploads below.
    {
      auto context=c::Context::create(c::cpu::device(),mock.provider()).value();auto queue=context.create_queue().value();
      const c::ProviderExtensionId old{0x48594c4542415443ULL,0x504c414e4e494e47ULL,3};
      const auto rejected=d::ProviderExtensionAccess::query(context,0,old);ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsUnimplemented());
      const std::array<std::int32_t,2> offsets{0,1};auto layout=schema.version({offsets,offsets,{},{}}).value();
      auto program=c::BatchProgram::prepare(context,schema,{.resource=m::system_resource(),.host_workspace=m::system_resource(),
          .max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10,.allocator=stats.get()}).value();
      auto slot=c::BatchInvocation::create(program,queue).value();
      auto prepared=slot.prepare_batch(layout);ASSERT_TRUE(prepared)<<prepared.error().message();
      EXPECT_EQ(mock.queries,native?0u:1u);
      EXPECT_EQ(mock.fresh_queries,native?0u:1u);EXPECT_EQ(mock.update_queries,0u);
      auto pool=c::LeasePool::create(m::system_resource(),1024).value();std::array<c::ReadLease,3> inputs;
      for(auto&input:inputs){auto writer=pool.allocate(128).value();ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());input=writer.publish().value();}
      const auto before_run=stats.snapshot().total_allocs;
      auto done=prepared->run(inputs);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(stats.snapshot().total_allocs,before_run);
      retained=slot.output().value();std::array<std::byte,128> bytes;ASSERT_TRUE(retained.view()->read(bytes).ok());for(auto value:bytes)EXPECT_EQ(value,std::byte{7});
      EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.runs,1u);*done={};*prepared={};
      const auto queries=mock.queries;mock.float_bytes=2u<<20;
      auto busy=slot.update_batch(layout);ASSERT_FALSE(busy);EXPECT_TRUE(busy.error().IsUnavailable());EXPECT_EQ(mock.queries,queries);
      retained.reset();
      if(!native){auto capped=slot.update_batch(layout);ASSERT_FALSE(capped);EXPECT_TRUE(capped.error().IsResourceExhausted());EXPECT_EQ(mock.plans,1u);}
      mock.float_bytes=3072;
      auto resized=slot.update_batch(layout);ASSERT_TRUE(resized)<<resized.error().message();
      auto again=resized->run(inputs);ASSERT_TRUE(again)<<again.error().message();ASSERT_TRUE(again->wait().ok());
      *again={};*resized={};mock.float_bytes=1536;mock.int_bytes=512;
      const auto before_shrink=stats.snapshot().total_allocs;
      auto smaller=slot.update_batch(layout);ASSERT_TRUE(smaller)<<smaller.error().message();
      EXPECT_EQ(stats.snapshot().total_allocs,before_shrink);
      auto last=smaller->run(inputs);ASSERT_TRUE(last)<<last.error().message();ASSERT_TRUE(last->wait().ok());
      retained=slot.output().value();*last={};*smaller={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
    }
    std::array<std::byte,1> still_owned{};ASSERT_TRUE(retained.view()->subview(0,1)->read(still_owned).ok());EXPECT_EQ(still_owned[0],std::byte{7});retained.reset();EXPECT_TRUE(stats.balanced());
  }
  EXPECT_EQ(backing[0],backing[1]);
  std::printf("batch_create_backing_caller=%llu native=%llu; caller scratch allocated only for cold layout\n",static_cast<unsigned long long>(backing[0]),static_cast<unsigned long long>(backing[1]));
}
TEST(BatchWorkspace, RetainedPreparationUsesExactOpenContextAndSelectedCapability){
  Mock mock;
  const auto storage=c::CacheStorage::plain(c::DataType::kFloat16).value();
  auto schema=c::BatchSchema::create({.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
      .key_storage=storage,.value_storage=storage,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,
      .query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=1}).value();
  auto context=c::Context::create(c::cpu::device(),mock.provider()).value();
  auto other=c::Context::create(c::cpu::device(),mock.provider()).value();
  auto retained=d::ContextRef::acquire(context).value();
  auto exact=d::ProviderExtensionAccess::query(context,0,d::kBatchPlanningExtensionId).value();
  auto foreign=d::ProviderExtensionAccess::query(other,0,d::kBatchPlanningExtensionId).value();
  c::BatchProgramOptions options{.resource=m::system_resource()};
  auto rejected=d::BatchGraphAccess::prepare(retained,foreign,schema,options);ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsInvalidArgument());
  {auto prepared=d::BatchGraphAccess::prepare(retained,exact,schema,options);ASSERT_TRUE(prepared)<<prepared.error().message();}
  EXPECT_TRUE(context.is_open());
  ASSERT_TRUE(context.close().ok());
  auto closed=d::BatchGraphAccess::prepare(retained,exact,schema,options);ASSERT_FALSE(closed);EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(BatchWorkspace, ForcedFreshSizingPreservesOldEpisodeWhenNewPlanExceedsCap){
  Mock mock;mock.permit_reuse=true;
  auto stats=m::StatsAllocator::create(m::system_allocator()).value();
  const auto storage=c::CacheStorage::plain(c::DataType::kFloat16).value();
  auto schema=c::BatchSchema::create({.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
      .key_storage=storage,.value_storage=storage,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,
      .query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=1}).value();
  auto context=c::Context::create(c::cpu::device(),mock.provider()).value();auto queue=context.create_queue().value();
  auto program=c::BatchProgram::prepare(context,schema,{.resource=m::system_resource(),.host_workspace=m::system_resource(),
      .max_float_workspace_bytes=4096,.max_int_workspace_bytes=2048,.allocator=stats.get()}).value();
  auto slot=c::BatchInvocation::create(program,queue).value();
  const std::array<std::int32_t,2> offsets{0,1};const auto layout=schema.version({offsets,offsets,{},{}}).value();
  auto pool=c::LeasePool::create(m::system_resource(),1024).value();std::array<c::ReadLease,3> inputs;
  for(auto&input:inputs){auto writer=pool.allocate(128).value();ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());input=writer.publish().value();}
  auto initial=slot.prepare_batch(layout);ASSERT_TRUE(initial)<<initial.error().message();
  EXPECT_FALSE(mock.last_allow_update);EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.updates,0u);
  auto first=initial->run(inputs);ASSERT_TRUE(first)<<first.error().message();ASSERT_TRUE(first->wait().ok());
  *first={};*initial={};const auto previous=slot.report();
  mock.float_bytes=8192; // A fresh scheduler may need more than this retained plan.
  const auto before=stats.snapshot().total_allocs;
  auto rejected=slot.prepare_batch(layout);ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_FALSE(mock.last_allow_update);EXPECT_EQ(mock.fresh_queries,2u);EXPECT_EQ(mock.update_queries,0u);
  EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.updates,0u);EXPECT_EQ(stats.snapshot().total_allocs,before);
  const auto unchanged=slot.report();EXPECT_EQ(unchanged.layout_generation,previous.layout_generation);
  EXPECT_EQ(unchanged.float_workspace_bytes,previous.float_workspace_bytes);
  EXPECT_EQ(unchanged.int_workspace_bytes,previous.int_workspace_bytes);
  EXPECT_EQ(unchanged.reused_native_plan,previous.reused_native_plan);
  {auto old=slot.output();ASSERT_TRUE(old)<<old.error().message();std::array<std::byte,128> bytes;
    ASSERT_TRUE(old->view()->read(bytes).ok());for(auto value:bytes)EXPECT_EQ(value,std::byte{7});}
  auto reused=slot.update_batch(layout);ASSERT_TRUE(reused)<<reused.error().message();
  EXPECT_TRUE(mock.last_allow_update);EXPECT_EQ(mock.update_queries,1u);EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.updates,1u);
  EXPECT_TRUE(slot.report().reused_native_plan);EXPECT_EQ(slot.report().float_workspace_bytes,previous.float_workspace_bytes);
  auto second=reused->run(inputs);ASSERT_TRUE(second)<<second.error().message();ASSERT_TRUE(second->wait().ok());
  *second={};*reused={};EXPECT_EQ(mock.runs,2u);ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
}


TEST(BatchWorkspace, ScratchGrowOomForcesFreshSizingAndPlanOnRetry){
  hyle_f5_fault_fixture::Allocator fault;
  {
    Mock mock;mock.permit_reuse=true;
    const auto storage=c::CacheStorage::plain(c::DataType::kFloat16).value();
    auto schema=c::BatchSchema::create({.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
        .key_storage=storage,.value_storage=storage,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,
        .query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=1}).value();
    auto context=c::Context::create(c::cpu::device(),mock.provider()).value();auto queue=context.create_queue().value();
    auto program=c::BatchProgram::prepare(context,schema,{.resource=m::system_resource(),.host_workspace=m::system_resource(),
        .max_float_workspace_bytes=4096,.max_int_workspace_bytes=2048,.allocator=fault.ref()}).value();
    auto slot=c::BatchInvocation::create(program,queue).value();
    const std::array<std::int32_t,2> offsets{0,1};const auto layout=schema.version({offsets,offsets,{},{}}).value();
    auto pool=c::LeasePool::create(m::system_resource(),1024).value();std::array<c::ReadLease,3> inputs;
    for(auto&input:inputs){auto writer=pool.allocate(128).value();ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());input=writer.publish().value();}
    auto initial=slot.prepare_batch(layout);ASSERT_TRUE(initial)<<initial.error().message();
    auto first=initial->run(inputs);ASSERT_TRUE(first)<<first.error().message();ASSERT_TRUE(first->wait().ok());
    *first={};*initial={};EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.updates,0u);
    mock.float_bytes=3072;
    // Same metadata means the next injected metadata allocation is precisely
    // the growing scratch LeasePool, after the old floating lease was reset.
    const auto target=fault.attempts;fault.fail=target;
    auto failed=slot.prepare_batch(layout);fault.fail=SIZE_MAX;
    ASSERT_FALSE(failed);EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_EQ(fault.attempts,target+1);EXPECT_FALSE(mock.last_allow_update);
    EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.updates,0u);
    const auto fresh=mock.fresh_queries,updates=mock.update_queries;
    auto retried=slot.update_batch(layout);ASSERT_TRUE(retried)<<retried.error().message();
    EXPECT_FALSE(mock.last_allow_update);EXPECT_EQ(mock.fresh_queries,fresh+1);EXPECT_EQ(mock.update_queries,updates);
    EXPECT_EQ(mock.updates,0u);EXPECT_EQ(mock.plans,2u);EXPECT_FALSE(slot.report().reused_native_plan);
    EXPECT_EQ(slot.report().float_workspace_bytes,3072u);
    auto second=retried->run(inputs);ASSERT_TRUE(second)<<second.error().message();ASSERT_TRUE(second->wait().ok());
    {auto output=slot.output().value();std::array<std::byte,128> bytes;ASSERT_TRUE(output.view()->read(bytes).ok());
      for(auto value:bytes)EXPECT_EQ(value,std::byte{7});}
    *second={};*retried={};EXPECT_EQ(mock.runs,2u);ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
  }
  EXPECT_EQ(fault.live,0u);
}

TEST(BatchWorkspace, BorrowedCommandGatePrecedesScratchAndPlanChanges){
  auto stats=m::StatsAllocator::create(m::system_allocator()).value();
  {
    Mock mock;mock.permit_reuse=true;
    const auto storage=c::CacheStorage::plain(c::DataType::kFloat16).value();
    auto schema=c::BatchSchema::create({.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
        .key_storage=storage,.value_storage=storage,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,
        .query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=1}).value();
    auto context=c::Context::create(c::cpu::device(),mock.provider()).value();auto queue=context.create_queue().value();
    auto program=c::BatchProgram::prepare(context,schema,{.resource=m::system_resource(),.host_workspace=m::system_resource(),
        .max_float_workspace_bytes=4096,.max_int_workspace_bytes=2048,.allocator=stats.get()}).value();
    auto slot=c::BatchInvocation::create(program,queue).value();
    const std::array<std::int32_t,2> offsets{0,1};const auto layout=schema.version({offsets,offsets,{},{}}).value();
    auto initial=slot.prepare_batch(layout);ASSERT_TRUE(initial)<<initial.error().message();*initial={};
    struct Gate final {
      Mock*mock;m::StatsAllocator*stats;unsigned calls=0,expected_plans=1;
      std::uint64_t expected_frees=0;std::size_t expected_bytes=1536;bool reject=true;
      static hyle::Status before(void*opaque)noexcept{
        auto&s=*static_cast<Gate*>(opaque);++s.calls;
        // Check owner accounting before reading the old borrowed view. A late
        // callback fails without touching storage that may already be freed.
        if(s.stats->snapshot().total_frees!=s.expected_frees || s.mock->plans!=s.expected_plans ||
            s.mock->planned_float_bytes!=s.expected_bytes)
          return hyle::Status::Internal("command gate ran after owner/plan mutation");
        std::array<std::byte,1> byte{};
        HYLE_ASSIGN_OR_RETURN(auto first,s.mock->planned_float_view.subview(0,1));
        HYLE_RETURN_IF_ERROR(first.read(byte));
        if(byte[0]!=std::byte{1})return hyle::Status::Internal("old scratch contents changed before command gate");
        return s.reject?hyle::Status::Unavailable("test command owner veto"):hyle::Status::Ok();
      }
    } gate{&mock,&stats};
    const d::BeforeCommandChange callback{&gate,&Gate::before};
    auto stable=d::BatchGraphAccess::prepare(slot,layout,true,callback);ASSERT_TRUE(stable)<<stable.error().message();
    EXPECT_EQ(gate.calls,0u);EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.updates,1u);*stable={};

    mock.float_bytes=3072;gate.expected_frees=stats.snapshot().total_frees;
    const auto old=mock.planned_float_view;const auto before=stats.snapshot();
    auto blocked=d::BatchGraphAccess::prepare(slot,layout,false,callback);ASSERT_FALSE(blocked);
    EXPECT_TRUE(blocked.error().IsUnavailable());EXPECT_EQ(gate.calls,1u);EXPECT_EQ(mock.plans,1u);
    EXPECT_EQ(mock.planned_float_view.native_handle(),old.native_handle());
    EXPECT_EQ(stats.snapshot().total_frees,before.total_frees);EXPECT_EQ(stats.snapshot().total_allocs,before.total_allocs);
    gate.reject=false;
    auto grown=d::BatchGraphAccess::prepare(slot,layout,true,callback);ASSERT_TRUE(grown)<<grown.error().message();
    EXPECT_FALSE(mock.last_allow_update);EXPECT_EQ(gate.calls,2u);EXPECT_EQ(mock.plans,2u);
    EXPECT_EQ(mock.updates,1u);EXPECT_EQ(slot.report().float_workspace_bytes,3072u);*grown={};

    // Force a new native plan with the same scratch: the gate must still run
    // before plan(), and its rejection must preserve that original owner.
    gate.expected_plans=2;gate.expected_bytes=3072;gate.expected_frees=stats.snapshot().total_frees;gate.reject=true;
    const auto replaced=mock.planned_float_view;const auto before_plan=stats.snapshot();
    auto denied=d::BatchGraphAccess::prepare(slot,layout,false,callback);ASSERT_FALSE(denied);
    EXPECT_TRUE(denied.error().IsUnavailable());EXPECT_EQ(gate.calls,3u);EXPECT_EQ(mock.plans,2u);
    EXPECT_EQ(mock.planned_float_view.native_handle(),replaced.native_handle());
    EXPECT_EQ(stats.snapshot().total_frees,before_plan.total_frees);EXPECT_EQ(stats.snapshot().total_allocs,before_plan.total_allocs);
    gate.reject=false;
    auto retried=d::BatchGraphAccess::prepare(slot,layout,true,callback);ASSERT_TRUE(retried)<<retried.error().message();
    EXPECT_FALSE(mock.last_allow_update);EXPECT_EQ(gate.calls,4u);EXPECT_EQ(mock.plans,3u);EXPECT_EQ(mock.updates,1u);
    auto pool=c::LeasePool::create(m::system_resource(),1024).value();std::array<c::ReadLease,3> inputs;
    for(auto&input:inputs){auto writer=pool.allocate(128).value();ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());input=writer.publish().value();}
    auto done=retried->run(inputs);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    {auto output=slot.output().value();std::array<std::byte,128> bytes;ASSERT_TRUE(output.view()->read(bytes).ok());
      for(auto byte:bytes)EXPECT_EQ(byte,std::byte{7});}
    *done={};*retried={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

}
