#include "dirty_page_native_risk_fixture.h"
#include "native_vulkan_failure_fixture.h"
#include "../../src/compute_route_vulkan/vulkan_file_stream_access.h"
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include <hyle/compute/detail/vulkan_batch_access.h>
#include <hyle/compute/detail/paged_arena_access.h>
#include <array>
#include <cstdio>

namespace {
namespace r = hyle_g03_dirty_risk;
namespace c = r::c;
namespace n = r::n;
using hyle::Status;
Status episode(bool cancelled) {
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto state,r::prepare(context,device,staging));
  // A genuine frame for only unchanged page2 supplies addressable capacity.
  // It does not claim read authority over the independently pending page0.
  const auto host=state.data.layout.host();
  const std::array<std::int32_t,3> unchanged_ids{2,2,2};
  HYLE_ASSIGN_OR_RETURN(auto unchanged_layout,state.data.schema.version(
      {host.query_indptr,host.kv_indptr,unchanged_ids,host.last_page_lengths}));
  HYLE_ASSIGN_OR_RETURN(auto unchanged,state.data.keys.read(unchanged_layout,
      std::span(state.data.key_reads).subspan(1,1)));
  HYLE_ASSIGN_OR_RETURN(auto value_frame,state.data.values.read(state.data.layout,state.data.value_reads));
  HYLE_ASSIGN_OR_RETURN(auto program,n::detail::PagedBatchAccess::prepare_program(
      context,state.data.recipe,state.data.layout,{}));
  HYLE_ASSIGN_OR_RETURN(auto invocation,program.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto q,device.binding(state.data.query,state.data.recipe.descriptions()[0]));
  HYLE_ASSIGN_OR_RETURN(auto k,n::detail::PagedBatchAccess::binding(device,unchanged));
  HYLE_ASSIGN_OR_RETURN(auto v,n::detail::PagedBatchAccess::binding(device,value_frame));
  HYLE_ASSIGN_OR_RETURN(auto version,state.data.recipe.version(state.data.layout));
  HYLE_ASSIGN_OR_RETURN(auto ids,version.indices());
  HYLE_ASSIGN_OR_RETURN(auto indices_pool,c::LeasePool::create(device.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto ids_writer,indices_pool.allocate(ids.view()->size()));
  HYLE_ASSIGN_OR_RETURN(auto metadata_host,c::LeasePool::create(staging.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto metadata_writer,metadata_host.allocate(ids.view()->size()));
  HYLE_ASSIGN_OR_RETURN(auto metadata_mapping,ids.view()->map_host());
  HYLE_RETURN_IF_ERROR(metadata_writer.view()->write(metadata_mapping.bytes()));
  HYLE_ASSIGN_OR_RETURN(auto staged_ids,metadata_writer.publish());
  const std::array metadata_specs{c::CopyResourceSpec{staging.ref(),ids.view()->size()},
                                 c::CopyResourceSpec{device.ref(),ids.view()->size()}};
  HYLE_ASSIGN_OR_RETURN(auto metadata_program,n::CopyBatchProgram::prepare(context,metadata_specs));
  HYLE_ASSIGN_OR_RETURN(auto metadata,n::CopyBatchInvocation::create(metadata_program));
  HYLE_RETURN_IF_ERROR(metadata.bind_source({0},staged_ids));
  HYLE_RETURN_IF_ERROR(metadata.bind_mutable({1},std::move(ids_writer)));
  const std::array<c::CopyPhase,1> metadata_phases{{{}}};
  const std::array<c::CopyRange,1> metadata_ranges{{{{0},{1},{0},0,0,ids.view()->size()}}};
  HYLE_RETURN_IF_ERROR(metadata.prepare_replay(metadata_phases,metadata_ranges));
  HYLE_ASSIGN_OR_RETURN(auto uploaded,metadata.submit());
  HYLE_RETURN_IF_ERROR(uploaded.wait()); // Cold metadata prepare, before pending producer.
  HYLE_ASSIGN_OR_RETURN(auto metadata_read,uploaded.output({0},{1},0,ids.view()->size()));
  HYLE_ASSIGN_OR_RETURN(auto table,device.binding(metadata_read,state.data.recipe.descriptions()[3]));
  const std::array native_inputs{q,k,v,table};
  HYLE_RETURN_IF_ERROR(invocation.bind(native_inputs));
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());
  struct Close final {n::Context &context;~Close(){context.close().IgnoreError();}} close{context};
  hyle_f5_vk_failure_fixture::Boundary boundary(context);
  static std::atomic<bool> unknown{true};unknown=true;
  boundary.device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{
    auto &hook=*hyle_f5_vk_failure_fixture::Boundary::active;
    const auto real=hook.original.GetFenceStatus(device,fence);
    return unknown.load() ? VK_NOT_READY : real;
  };
  HYLE_ASSIGN_OR_RETURN(auto submitted,state.copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto owner,n::CopyBatchUpdate::take(std::move(state.copy),std::move(submitted)));
  HYLE_ASSIGN_OR_RETURN(auto pending,n::detail::CopyStreamAccess::input(owner,{1}));
  HYLE_ASSIGN_OR_RETURN(auto dependency,n::detail::CopyStreamAccess::dependency(owner));
  HYLE_RETURN_IF_ERROR(n::detail::qualify_copy_dependency(n::detail::Access::runtime(context),dependency));
  HYLE_ASSIGN_OR_RETURN(auto base,c::detail::PagedArenaAccess::tensor(unchanged));
  HYLE_RETURN_IF_ERROR(r::require(pending.view.resource()==base.storage().resource() &&
      pending.view.native_handle()==base.storage().native_handle() &&
      pending.view.offset()==base.storage().offset() &&
      pending.view.size()==state.data.keys.page_bytes() && pending.writer->generation()==state.generation,
      "pending original page does not belong to the original attention arena"));
  HYLE_ASSIGN_OR_RETURN(auto reading,uses.begin());
  HYLE_RETURN_IF_ERROR(reading.read(state.data.query));
  HYLE_RETURN_IF_ERROR(reading.read(metadata_read));
  for(const auto &page:c::detail::PagedArenaAccess::pages(unchanged)) HYLE_RETURN_IF_ERROR(reading.read(page));
  for(const auto &page:c::detail::PagedArenaAccess::pages(value_frame)) HYLE_RETURN_IF_ERROR(reading.read(page));
  HYLE_RETURN_IF_ERROR(n::detail::CopyStreamAccess::admit_read(reading,owner,{1}));
  if(cancelled) HYLE_RETURN_IF_ERROR(owner.request_cancel());
#if defined(HYLE_PENDING_ADMISSION_QUALIFICATION)
  if(cancelled) {
    const auto rejected=n::detail::qualify_copy_dependency(n::detail::Access::runtime(context),dependency);
    HYLE_RETURN_IF_ERROR(r::require(rejected.IsCancelled() && owner.valid() && owner.in_flight() &&
        boundary.accepted>0,
        "Vulkan cancelled dependency admission consumed source or accepted suffix"));
    unknown=false;
    HYLE_RETURN_IF_ERROR(r::require(owner.wait().IsCancelled() && !owner.take_mutable({1}),
        "Vulkan cancelled source fabricated mutable success"));
    std::fprintf(stderr,"Vulkan original Run pending cancel dependency rejected; "
        "original controls kept, no attention suffix, actual final fence retirement preserved PASS\n");
    return Status::Ok();
  }
#endif
  HYLE_ASSIGN_OR_RETURN(auto attention,n::detail::PagedBatchAccess::submit(invocation,std::move(reading)));
  HYLE_RETURN_IF_ERROR(r::require(!attention.result() && uses.active_batches()==1,
      "original attention published result before its actual native retirement"));
  unknown=false;
  auto outcome=attention.wait(); // No producer/copy host wait after submission.
#if defined(HYLE_PENDING_ADMISSION_QUALIFICATION)
  // A retired suffix proves native ordering, but does not settle the source's
  // original logical observation. First observe its actual successful Run;
  // only a subsequent cancellation request must preserve that success.
  HYLE_RETURN_IF_ERROR(owner.observe());
  HYLE_RETURN_IF_ERROR(owner.request_cancel());
  HYLE_RETURN_IF_ERROR(n::detail::qualify_copy_dependency(n::detail::Access::runtime(context),dependency));
#endif
  const auto source=owner.observe();
  HYLE_RETURN_IF_ERROR(r::require(outcome.has_value() && (*outcome)->tensors().size()==1 &&
      (cancelled ? source.IsCancelled() : source.ok()) && uses.active_batches()==0,
      "receiver counterexample did not observe the expected original source/suffix statuses"));
  const auto result=*outcome;
  HYLE_RETURN_IF_ERROR(r::require(result->tensors()[0].description()==state.data.recipe.descriptions()[0],"receiver used a different numerical shape"));
#if defined(HYLE_PENDING_ADMISSION_QUALIFICATION)
  std::fprintf(stderr,"Vulkan original Run success remains successful after late cancel request; "
      "same Runtime/queue + exact guards, final attention shape2x8x64 PASS\n");
#else
  std::fprintf(stderr,"actual Vulkan source cancelled=%d, original source status=%d, attention SUCCESS shape2x8x64; "
      "same original Runtime/queue+exact original F1 guards; only retirement observations masked, only final attention wait; "
      "cancelled case witnesses missing source preflight/status merge, not a safe public receiver\n",
      cancelled,int(source.code()));
#endif
  return Status::Ok();
}
Status run(){ HYLE_RETURN_IF_ERROR(episode(false));return episode(true); }
} // namespace
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"receiver probe:%.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
