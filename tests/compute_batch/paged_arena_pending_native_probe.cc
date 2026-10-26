#include "dirty_page_native_risk_fixture.h"
#include <hyle/compute/detail/paged_arena_access.h>
#if defined(HYLE_G01_PAGE_RISK_APPLE)
#include "native_apple_failure_fixture.h"
#include <hyle/compute/detail/apple_plan.h>
#include "hyle/compute/detail/apple_copy_stream_access.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "../../src/compute_route_apple/apple_internal.h"
#else
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#endif

namespace {
namespace r=hyle_g03_dirty_risk;namespace c=r::c;namespace n=r::n;
using hyle::Status;using ArenaAccess=c::detail::PagedArenaAccess;using CopyAccess=n::detail::CopyStreamAccess;
Status episode(bool gap){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  // Resources are borrowed by late native cleanup as well as actual writes.
  // Destroy controls first, then join the original worker before resources.
  struct Close final {n::Context&context;~Close(){context.close().IgnoreError();}}close{context};
  HYLE_ASSIGN_OR_RETURN(auto state,r::prepare(context,device,staging));
  // Return the unsubmitted old page, then write a genuinely uninitialized page.
  HYLE_ASSIGN_OR_RETURN(state.writer.lease(),state.copy.unbind_mutable({1}));
  HYLE_ASSIGN_OR_RETURN(state.data.key_reads[0],state.writer.publish(state.data.schema.capacity().page_size));
  HYLE_ASSIGN_OR_RETURN(auto writer,state.data.keys.acquire(1));
  HYLE_RETURN_IF_ERROR(state.copy.bind_mutable({1},std::move(writer.lease())));
  const std::array<c::CopyPhase,3> phases{{{}, {.dependency_mask=1}, {.dependency_mask=2}}};
  const std::array<c::CopyRange,2> ranges{{{{0},{1},{1},0,0,gap?128u:256u},{{0},{1},{2},0,256,256}}};
  HYLE_RETURN_IF_ERROR(state.copy.prepare_replay(phases,ranges));
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  // Freeze only this original worker while checking observer cost. Its native
  // recording counters otherwise advance independently on the worker thread.
  namespace mx=mlx::core;
  auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Release final {decltype(gate) gate;~Release(){gate->release();}}release{gate};
  const std::array blocker_specs{n::InputSpec{{4},mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto&inputs,mx::Stream stream){
    auto prefix=mx::square(inputs[0],stream);
    return std::vector{mx::array(prefix.shape(),prefix.dtype(),
      std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{prefix})};
  },blocker_specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());
  const std::array input{mx::array({1.F,2.F,3.F,4.F})};HYLE_RETURN_IF_ERROR(blocking.bind(input));
  {std::lock_guard lock(gate->mutex);gate->armed=true;}
  HYLE_ASSIGN_OR_RETURN(auto prefix,blocking.submit());
  HYLE_RETURN_IF_ERROR(r::require(gate->await(),"pending page probe has no real queued native prefix"));
#endif
  HYLE_ASSIGN_OR_RETURN(auto submitted,state.copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,n::CopyBatchUpdate::take(std::move(state.copy),std::move(submitted)));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> ids{1},tail{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,state.data.schema.version({ptr,ptr,ids,tail}));
  const std::array<c::PagedPageUpdate,1> pages{{{&writer,{1}}}};
  HYLE_ASSIGN_OR_RETURN(auto frame,state.data.keys.pending_read(layout,{},pages));
  const auto original_report=update.report();
  const auto verdict=ArenaAccess::validate<n::CopyBatchUpdate,CopyAccess>(frame,update);
  HYLE_RETURN_IF_ERROR(r::require(update.valid() &&
      update.report().host_waits==original_report.host_waits,
      "native pending validation waited/consumed original copy"));
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  gate->release();
#endif
  if(gap){
    HYLE_RETURN_IF_ERROR(r::require(verdict.IsOutOfRange() &&
        CopyAccess::covers(update,{1},0,128).ok() && CopyAccess::covers(update,{1},256,256).ok(),
        "native accepted write coverage credited the unwritten row hole"));
    HYLE_RETURN_IF_ERROR(update.wait());
    HYLE_ASSIGN_OR_RETURN(writer.lease(),update.take_mutable({1}));
    HYLE_ASSIGN_OR_RETURN(auto ready,writer.publish(0));
    HYLE_RETURN_IF_ERROR(r::require(!state.data.keys.read(layout,std::span(&ready,1)),
        "native coverage rejection published uninitialized typed rows"));
    std::fprintf(stderr,"native pending %s: true fresh-page byte hole rejected, no source observation/consumption PASS\n",hyle_g01_page_risk::route);
    return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(verdict);
  HYLE_ASSIGN_OR_RETURN(auto dependency,CopyAccess::dependency(update));
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_RETURN_IF_ERROR(n::detail::qualify_recording_dependency(n::detail::ContextAccess::runtime(context),dependency));
#else
  HYLE_RETURN_IF_ERROR(n::detail::qualify_copy_dependency(n::detail::Access::runtime(context),dependency));
#endif
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());HYLE_ASSIGN_OR_RETURN(auto reading,uses.begin());
  HYLE_RETURN_IF_ERROR((ArenaAccess::admit_read<n::CopyBatchUpdate,CopyAccess>(reading,frame,update)));
  HYLE_RETURN_IF_ERROR(update.wait());
  auto rejected=update.take_mutable({1});
  HYLE_RETURN_IF_ERROR(r::require(!rejected && rejected.error().IsUnavailable() && update.valid(),
      "native original writer escaped a retained ordered-read use"));
  reading={};HYLE_ASSIGN_OR_RETURN(writer.lease(),update.take_mutable({1}));
  HYLE_ASSIGN_OR_RETURN(auto ready,writer.publish(2));
  HYLE_ASSIGN_OR_RETURN(auto qualified,state.data.keys.read(layout,std::span(&ready,1)));
  HYLE_RETURN_IF_ERROR(r::require(qualified.page_count()==1 && !state.data.keys.acquire(ready),
      "native exact ready frame lost original page protection"));
  // Oracle-only download of the two initialized rows, never the remaining page.
  const std::array specs{c::CopyResourceSpec{device.ref(),state.data.keys.page_bytes()},c::CopyResourceSpec{staging.ref(),512}};
  HYLE_ASSIGN_OR_RETURN(auto program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto down,n::CopyBatchInvocation::create(program));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),1024));HYLE_ASSIGN_OR_RETURN(auto target,pool.allocate(512));
  HYLE_RETURN_IF_ERROR(down.bind_source({0},ready));HYLE_RETURN_IF_ERROR(down.bind_mutable({1},std::move(target)));
  const std::array<c::CopyPhase,1> one{{{}}};const std::array<c::CopyRange,1> bytes{{{{0},{1},{0},0,0,512}}};
  HYLE_RETURN_IF_ERROR(down.prepare_replay(one,bytes));HYLE_ASSIGN_OR_RETURN(auto copied,down.submit());HYLE_RETURN_IF_ERROR(copied.wait());
  HYLE_ASSIGN_OR_RETURN(auto output,copied.output({0},{1},0,512));std::array<std::byte,512> actual{};HYLE_RETURN_IF_ERROR(output.view()->read(actual));
  for(auto byte:actual)HYLE_RETURN_IF_ERROR(r::require(byte==std::byte{31},"native pending page two-row byte oracle differs"));
  std::fprintf(stderr,"native pending %s: two accepted phases cover exactly two initialized rows, qualified original prefix, retained use refuses writer/nonconsume, exact frame and byte oracle PASS\n",hyle_g01_page_risk::route);
  return Status::Ok();
}
} // namespace
int main(){auto status=episode(false);if(status.ok())status=episode(true);if(!status.ok())std::fprintf(stderr,"pending frame: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
