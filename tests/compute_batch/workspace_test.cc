#include <gtest/gtest.h>
#include <array>
#include <cstdio>
#include "../../src/compute/provider_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/batch_provider.h"
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
  c::BatchProgramOptions options;unsigned plans=0,runs=0;
  d::BatchProviderVTable batch{
    .qualify=[](void*,c::BatchCapacity,const c::BatchProgramOptions&)noexcept{return hyle::Status::Ok();},
    .create=[](void*p,c::BatchCapacity,const c::BatchProgramOptions&o,m::AllocatorRef,void**out)noexcept{static_cast<Mock*>(p)->options=o;*out=p;return hyle::Status::Ok();},
    .destroy=[](void*,m::AllocatorRef)noexcept{},
    .update=[](void*,const c::BatchLayoutVersion&,c::BatchPlanReport*)noexcept->hyle::Result<bool>{return false;},
    .plan=[](void*p,c::Queue&,const c::BatchLayoutVersion&,d::BatchWorkspaceViews views,c::BatchPlanReport*)noexcept{
      auto&s=*static_cast<Mock*>(p);++s.plans;return d::EnqueueResult::accepted_command(s.inspect(views,true));},
    .run=[](void*p,c::Queue&,std::span<const c::ConstTensorView>,d::BatchWorkspaceViews views)noexcept{
      auto&s=*static_cast<Mock*>(p);++s.runs;const auto status=s.inspect(views,false);
      return d::EnqueueResult::accepted_command(status.ok()?views.output.fill(std::byte{7}):status);}
  };
  hyle::Status inspect(d::BatchWorkspaceViews v,bool planning)const noexcept{
    if(batch.uses_caller_workspace){
      if(!v.float_workspace.valid() || v.float_workspace.size()!=options.max_float_workspace_bytes ||
         !v.int_workspace.valid() || v.int_workspace.size()!=options.max_int_workspace_bytes ||
         !v.host_workspace.valid() || v.host_workspace.size()!=options.max_int_workspace_bytes)
        return hyle::Status::Internal("caller workspace limits/views changed");
      HYLE_RETURN_IF_ERROR(v.float_workspace.fill(std::byte{1}));
      HYLE_RETURN_IF_ERROR(v.int_workspace.fill(std::byte{2}));
      if(planning)HYLE_RETURN_IF_ERROR(v.host_workspace.fill(std::byte{3}));
    }else if(v.float_workspace.valid() || v.int_workspace.valid() || v.host_workspace.valid())
      return hyle::Status::Internal("native owner received unused scratch");
    return hyle::Status::Ok();
  }
  c::Provider provider(){
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
TEST(BatchWorkspace, DefaultCallerScratchAndNativeOptOutKeepActualBackingsAndViews){
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
      const c::ProviderExtensionId old{0x48594c4542415443ULL,0x504c414e4e494e47ULL,2};
      const auto rejected=d::ProviderExtensionAccess::query(context,0,old);ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsUnimplemented());
      const std::array<std::int32_t,2> offsets{0,1};auto layout=schema.version({offsets,offsets,{},{}}).value();
      auto program=c::BatchProgram::prepare(context,schema,{.resource=m::system_resource(),.host_workspace=m::system_resource(),
          .max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}).value();
      auto slot=c::BatchInvocation::create(program,queue).value();
      auto prepared=slot.prepare_batch(layout);ASSERT_TRUE(prepared)<<prepared.error().message();
      auto pool=c::LeasePool::create(m::system_resource(),1024).value();std::array<c::ReadLease,3> inputs;
      for(auto&input:inputs){auto writer=pool.allocate(128).value();ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());input=writer.publish().value();}
      auto done=prepared->run(inputs);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
      retained=slot.output().value();std::array<std::byte,128> bytes;ASSERT_TRUE(retained.view()->read(bytes).ok());for(auto value:bytes)EXPECT_EQ(value,std::byte{7});
      EXPECT_EQ(mock.plans,1u);EXPECT_EQ(mock.runs,1u);*done={};*prepared={};ASSERT_TRUE(slot.close().ok());ASSERT_TRUE(context.drain().ok());
    }
    std::array<std::byte,1> still_owned{};ASSERT_TRUE(retained.view()->subview(0,1)->read(still_owned).ok());EXPECT_EQ(still_owned[0],std::byte{7});retained.reset();EXPECT_TRUE(stats.balanced());
  }
  EXPECT_EQ(backing[0]-backing[1],(1u<<20)+2u*(64u<<10));
  std::printf("batch_backing_default=%llu native=%llu removed=%llu\n",static_cast<unsigned long long>(backing[0]),static_cast<unsigned long long>(backing[1]),static_cast<unsigned long long>(backing[0]-backing[1]));
}
}
