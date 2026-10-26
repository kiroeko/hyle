#include "native_paged_risk_fixture.h"
#if defined(HYLE_G01_PAGE_RISK_APPLE)
#include "native_apple_failure_fixture.h"
#include "hyle/compute/detail/apple_copy_stream_access.h"
#else
#include "native_vulkan_failure_fixture.h"
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#endif

namespace {
namespace r=hyle_g01_page_risk;namespace c=r::c;namespace n=r::n;
using hyle::Status;
enum class Mode { success,hole,cancel_before,cancel_after,abandon,allocation,capacity,unknown,native_rejection,closed };
Status episode(Mode mode){
  hyle_f5_fault_fixture::Allocator allocator;
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  struct Close final {n::Context&context;~Close(){context.close().IgnoreError();}}close{context};
  HYLE_ASSIGN_OR_RETURN(auto data,r::inputs(context,device,staging));
  HYLE_ASSIGN_OR_RETURN(auto native,r::create(context,device,staging,data,allocator.ref()));
  const auto baseline=allocator.live;
  if(mode==Mode::allocation){
    bool completed=false;
    for(std::size_t fail=0;fail<16;++fail){
      allocator.fail=allocator.attempts+fail;
      auto made=n::PagedBatchReceiver::create(std::move(native),2);
      allocator.fail=SIZE_MAX;
      if(made){HYLE_RETURN_IF_ERROR(r::require(!native.valid(),"successful receiver creation did not transfer original owner"));completed=true;break;}
      HYLE_RETURN_IF_ERROR(r::require(made.error().IsResourceExhausted() && native.valid() && allocator.live==baseline,
          "receiver cold allocation failure leaked or consumed original controller"));
    }
    HYLE_RETURN_IF_ERROR(r::require(completed,"receiver cold allocation audit did not reach the finite success boundary"));
    HYLE_ASSIGN_OR_RETURN(native,r::create(context,device,staging,data,allocator.ref()));
  }
  HYLE_ASSIGN_OR_RETURN(auto receiver,n::PagedBatchReceiver::create(std::move(native),mode==Mode::capacity?1:2));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> ids{1},tail{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,data.schema.version({ptr,ptr,ids,tail}));
  HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
  HYLE_ASSIGN_OR_RETURN(auto kw,data.keys.acquire(1));HYLE_ASSIGN_OR_RETURN(auto vw,data.values.acquire(1));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto host,pool.allocate(512));HYLE_RETURN_IF_ERROR(host.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto ready,host.publish());
  const std::array specs{c::CopyResourceSpec{staging.ref(),512},c::CopyResourceSpec{device.ref(),data.keys.page_bytes()},
      c::CopyResourceSpec{device.ref(),data.values.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(program));
  HYLE_RETURN_IF_ERROR(copy.bind_source({0},ready));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(kw.lease())));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({2},std::move(vw.lease())));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,2> ranges{{{{0},{1},{1},0,0,mode==Mode::hole?128u:512u},{{0},{2},{1},0,0,512}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  namespace mx=mlx::core;
  auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Release final {decltype(gate) gate;n::Context&context;~Release(){gate->release();context.close().IgnoreError();}}release{gate,context};
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
  HYLE_RETURN_IF_ERROR(r::require(gate->await(),"receiver risk has no actual native queued prefix"));
  const auto release_native=[&]{gate->release();};
#else
  hyle_f5_vk_failure_fixture::Boundary boundary(context);
  static std::atomic<bool> masked{true};masked=true;
  boundary.device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{
    auto&hook=*hyle_f5_vk_failure_fixture::Boundary::active;
    if(hook.fail_poll.exchange(false))return VK_ERROR_OUT_OF_HOST_MEMORY;
    const auto real=hook.original.GetFenceStatus(device,fence);
    return masked.load()?VK_NOT_READY:real;
  };
  const auto release_native=[&]{masked=false;};
#endif
  HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,n::CopyBatchUpdate::take(std::move(copy),std::move(submitted)));
  const std::array<c::PagedPageUpdate,1> key_update{{{&kw,{1}}}},value_update{{{&vw,{2}}}};
  HYLE_ASSIGN_OR_RETURN(auto k,data.keys.pending_read(layout,{},key_update));
  HYLE_ASSIGN_OR_RETURN(auto v,data.values.pending_read(layout,{},value_update));
  std::array<c::Completion*,2> original_sources{};
  HYLE_ASSIGN_OR_RETURN(auto count,n::detail::CopyStreamAccess::retirement_sources(update,original_sources));
  HYLE_RETURN_IF_ERROR(r::require(count==2,"receiver risk lost original accepted phase controls"));
  if(mode==Mode::cancel_before)HYLE_RETURN_IF_ERROR(update.request_cancel());
  if(mode==Mode::allocation)allocator.fail=allocator.attempts;
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
  if(mode==Mode::native_rejection)boundary.fail_at=boundary.submits+1;
#endif
  if(mode==Mode::closed){release_native();HYLE_RETURN_IF_ERROR(update.wait());HYLE_RETURN_IF_ERROR(context.close());}
  auto result=receiver.submit(data.query,k,v,std::move(update));
  allocator.fail=SIZE_MAX;
  if(mode==Mode::hole || mode==Mode::cancel_before || mode==Mode::capacity || mode==Mode::closed || mode==Mode::native_rejection){
    HYLE_RETURN_IF_ERROR(r::require(!result && update.valid(),"receiver admission failure consumed caller Update"));
    HYLE_RETURN_IF_ERROR(r::require(mode==Mode::hole?result.error().IsOutOfRange():
        mode==Mode::cancel_before?result.error().IsCancelled():
        mode==Mode::closed?result.error().IsFailedPrecondition():result.error().IsResourceExhausted(),
        "receiver admission returned a different failure"));
    release_native();update.wait().IgnoreError();
    std::fprintf(stderr,"%s receiver rejection mode=%u preserves original Update PASS\n",r::route,unsigned(mode));return Status::Ok();
  }
  if(mode==Mode::allocation){
    HYLE_RETURN_IF_ERROR(r::require(!result && result.error().IsResourceExhausted() && update.valid(),
        "pending binding OOM accepted suffix or consumed source"));
    result=receiver.submit(data.query,k,v,std::move(update));
  }
  HYLE_RETURN_IF_ERROR(r::require(result.has_value() && !update.valid() && !receiver.take_update() &&
      data.query_pool.stats().active_uses>0,"accepted receiver lost source/native page guard"));
  if(mode==Mode::cancel_after)HYLE_RETURN_IF_ERROR(original_sources.back()->request_cancel());
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
  if(mode==Mode::unknown){
    boundary.fail_poll=true;auto diagnosis=result->poll();
    // The existing dependent observer records the original error, but remains
    // pending until actual retirement. Its final wait must return that error.
    HYLE_RETURN_IF_ERROR(r::require(diagnosis && *diagnosis==c::CompletionPoll::kPending &&
        !boundary.fail_poll.load() && data.query_pool.stats().active_uses>0 && !receiver.take_update(),
        "unknown receiver observation fabricated retirement or released source"));
  }
#endif
  if(mode==Mode::abandon){
    *result={};receiver={};
    HYLE_RETURN_IF_ERROR(r::require(pool.stats().active_uses>0 && data.query_pool.stats().active_uses>0,
        "abandoned receiver dropped source/guards before actual suffix retirement"));
    release_native();HYLE_RETURN_IF_ERROR(context.close());
    HYLE_RETURN_IF_ERROR(r::require(pool.stats().active_uses==0 && data.query_pool.stats().active_uses==0,
        "abandoned original work did not retire all source/suffix guards"));
  }else{
    release_native();auto outcome=result->wait();
    HYLE_RETURN_IF_ERROR(r::require(mode==Mode::cancel_after?!outcome && outcome.error().IsCancelled():
        mode==Mode::unknown?!outcome && outcome.error().IsResourceExhausted():outcome.has_value(),
        "receiver original result did not merge the expected source/suffix terminal"));
    const auto observed=result->completion().wait();
    HYLE_RETURN_IF_ERROR(r::require(mode==Mode::cancel_after?observed.IsCancelled():
        mode==Mode::unknown?observed.IsResourceExhausted():observed.ok(),
        "public completion differs from original result observation"));
    HYLE_RETURN_IF_ERROR(r::require(data.query_pool.stats().active_uses==0 && !receiver.take_update(),
        "retained public outcome allowed original source reuse"));
    if(outcome)*outcome={};*result={};
    HYLE_RETURN_IF_ERROR(r::require(receiver.prepare_batch(layout).IsUnavailable(),"unrecovered original source permitted receiver reuse"));
    HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
    if(mode==Mode::success || mode==Mode::allocation){
      HYLE_ASSIGN_OR_RETURN(kw.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(vw.lease(),update.take_mutable({2}));
      HYLE_ASSIGN_OR_RETURN(auto key,kw.publish(2));HYLE_ASSIGN_OR_RETURN(auto value,vw.publish(2));
      HYLE_RETURN_IF_ERROR(r::require(data.keys.read(layout,std::span(&key,1)).has_value() &&
          data.values.read(layout,std::span(&value,1)).has_value(),"receiver could not recover exact initialized ready frames"));
    }
  }
  std::fprintf(stderr,"%s actual original pending receiver mode=%u final result/completion, source controls and guards PASS\n",r::route,unsigned(mode));
  return Status::Ok();
}
}
int main(){
  for(auto mode:{Mode::success,Mode::hole,Mode::cancel_before,Mode::cancel_after,Mode::abandon,Mode::allocation,Mode::capacity,Mode::closed
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
      ,Mode::unknown,Mode::native_rejection
#endif
  }){auto status=episode(mode);if(!status.ok()){
    std::fprintf(stderr,"receiver risk mode=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());return 1;
  }}return 0;
}
